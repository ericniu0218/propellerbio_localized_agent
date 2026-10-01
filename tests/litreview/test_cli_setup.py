"""`bedrock-litreview doctor` and `bedrock-litreview forget`.

Both exist because of something the deploy-readiness audit found and could not
fix in code: a scientist sets this up on a machine nobody has set it up on
before, and the only check that ever ran was the first-run banner — which runs
only while there is no config file at all, i.e. exactly once, before anything is
configured. Every later mistake was found by starting a review and waiting for
it to refuse.

`doctor` reports every finding rather than the first, which is what makes it
worth running twice in a row; `forget` is the delete that `/forget` for a chat
has always had and a review did not.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from bedrock.inference import InferenceError, ModelInfo
from bedrock.litreview import cli
from bedrock.litreview.db import Database
from tests.litreview.fixtures import QUESTION, corpus


@pytest.fixture()
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("BEDROCK_LITREVIEW_DIR", str(tmp_path / "data"))
    return tmp_path / "data"


@pytest.fixture()
def corpus_file(tmp_path):
    from bedrock.litreview.sources.corpus import write_papers

    path = tmp_path / "papers.jsonl"
    write_papers(corpus(), path)
    return path


class Server:
    """A model server offering `models`, or refusing to answer at all."""

    def __init__(self, models=("qwen3:8b",), silent: bool = False) -> None:
        self.models, self.silent = list(models), silent

    async def list_models(self):
        if self.silent:
            raise InferenceError("connection refused", unavailable=True)
        return [ModelInfo(id=m) for m in self.models]

    async def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# doctor
#
# `bedrock-litreview doctor` is `litreview/readiness.py`'s `review_readiness`,
# printed. These tests are about what this branch adds to it: advice that does
# not stop a review (a short timeout, a nearly full disk), shown beside the one
# thing that does. What the readiness check itself decides - windows, models,
# structured output, authentication - is main's, and is tested with it.
# ---------------------------------------------------------------------------


def _ready(monkeypatch, *, timeout: str | None = "600"):
    monkeypatch.setenv("BEDROCK_CONTEXT_WINDOW", "12288")
    if timeout is not None:
        monkeypatch.setenv("BEDROCK_TIMEOUT", timeout)


def test_doctor_is_ready_on_a_machine_that_is_set_up(data_dir, monkeypatch, capsys):
    monkeypatch.setattr("bedrock.inference.create_backend", lambda s: Server())
    _ready(monkeypatch)

    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out

    assert "operational" in out
    assert "FAIL" not in out
    assert "free disk" in out, "the disk advice is part of the check"


def test_doctor_shows_advice_that_does_not_stop_a_review(data_dir, monkeypatch, capsys):
    """bedrock's default 120-second timeout is the common case. It is worth
    saying and must not turn a working machine into a failing one: a review
    that hits it stops between calls and resumes."""
    monkeypatch.setattr("bedrock.inference.create_backend", lambda s: Server())
    _ready(monkeypatch, timeout=None)

    assert cli.main(["doctor"]) == 0, "advice must not make this exit non-zero"
    out = capsys.readouterr().out

    assert "request timeout: worth knowing" in out
    assert "request_timeout = 600" in out
    assert "operational" in out


def test_doctor_names_what_the_server_actually_offers(data_dir, monkeypatch, capsys):
    """The `office` model names are guesses until IT says what the vLLM server
    serves, and nobody can confirm them before that server exists. The next
    best thing is for the failure to say what this server calls its models and
    which file to write them into."""
    monkeypatch.setattr(
        "bedrock.inference.create_backend", lambda s: Server(models=["Qwen/Qwen3-30B-A3B-Instruct-2507"])
    )
    _ready(monkeypatch)

    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out

    assert "Qwen/Qwen3-30B-A3B-Instruct-2507" in out, "it must say what the server does offer"
    assert "--served-model-name" in out
    assert "models.yaml" in out


def test_doctor_shows_the_advice_even_when_the_server_is_down(data_dir, monkeypatch, capsys):
    """The advice is configuration and the disk, so it needs no server - and a
    server that is down is when someone is most likely to be reading this."""
    monkeypatch.setattr("bedrock.inference.create_backend", lambda s: Server(silent=True))
    _ready(monkeypatch, timeout=None)

    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out

    assert "could not list models" in out
    assert "free disk" in out and "request timeout" in out


def test_doctor_reports_a_broken_config_as_a_finding_not_a_traceback(data_dir, monkeypatch, capsys):
    monkeypatch.setenv("BEDROCK_LITREVIEW_MODELS", "not-a-real-set")

    assert cli.main(["doctor"]) == 1
    assert "unknown model set" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# forget
# ---------------------------------------------------------------------------


def _one_run(corpus_file) -> str:
    assert cli.main(["ask", QUESTION, "--corpus", str(corpus_file), "--no-model", "--models", "laptop"]) == 0
    from bedrock.litreview.config import load_settings

    db = Database(load_settings(check_endpoint=False, require_models=False).db_path)
    try:
        return db.recent_runs(1)[0].id
    finally:
        db.conn.close()


def test_forget_asks_first_and_leaves_the_run_alone_when_told_no(
    data_dir, corpus_file, monkeypatch, capsys
):
    """Confirmed by default, because there is no trash to recover from and a
    report can be hours of a machine's work."""
    run_id = _one_run(corpus_file)
    capsys.readouterr()
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")

    assert cli.main(["forget", run_id]) == 1
    assert "left alone" in capsys.readouterr().err

    db = Database(data_dir / "litreview.sqlite3")
    assert db.get_run(run_id) is not None


def test_forget_shows_the_question_before_asking(data_dir, corpus_file, monkeypatch, capsys):
    """Run ids are hashes. The way this goes wrong is deleting a neighbour of the
    one you meant, so the question is shown before anything happens."""
    run_id = _one_run(corpus_file)
    capsys.readouterr()
    monkeypatch.setattr("builtins.input", lambda _prompt="": "n")

    cli.main(["forget", run_id])
    err = capsys.readouterr().err
    assert QUESTION in err
    assert "cannot be undone" in err


def test_forget_with_yes_deletes_the_run_and_its_files(data_dir, corpus_file, monkeypatch, capsys):
    run_id = _one_run(corpus_file)
    capsys.readouterr()

    assert cli.main(["forget", run_id, "--yes"]) == 0
    out = capsys.readouterr().out

    assert f"Deleted run {run_id}" in out
    assert "stage artifact(s)" in out
    assert list((data_dir / "reports").glob("*/*/report.html")) == []

    db = Database(data_dir / "litreview.sqlite3")
    assert db.get_run(run_id) is None


def test_forget_refuses_an_unknown_run_id_and_says_how_to_list_them(data_dir, capsys):
    assert cli.main(["forget", "0123456789abcdef", "--yes"]) == 1
    assert "no run" in capsys.readouterr().err


def test_forget_does_not_delete_on_an_unanswered_prompt(data_dir, corpus_file, monkeypatch, capsys):
    """Piped input with nothing in it is not a yes."""
    run_id = _one_run(corpus_file)
    capsys.readouterr()

    def no_answer(_prompt=""):
        raise EOFError

    monkeypatch.setattr("builtins.input", no_answer)
    assert cli.main(["forget", run_id]) == 1
    assert "nothing was deleted" in capsys.readouterr().err
    assert Database(data_dir / "litreview.sqlite3").get_run(run_id) is not None


def test_forget_leaves_the_question_so_it_can_be_asked_again(
    data_dir, corpus_file, monkeypatch, capsys
):
    run_id = _one_run(corpus_file)
    capsys.readouterr()
    assert cli.main(["forget", run_id, "--yes"]) == 0
    capsys.readouterr()

    assert cli.main(["ask", QUESTION, "--no-model", "--models", "laptop"]) == 0
    assert Path(capsys.readouterr().out.strip().splitlines()[-1]).exists()

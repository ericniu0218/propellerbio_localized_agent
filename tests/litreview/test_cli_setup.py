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
# ---------------------------------------------------------------------------


def test_doctor_says_what_it_read_so_the_answer_can_be_checked(data_dir, monkeypatch, capsys):
    """Which files were read is half the answer. "The model set is wrong" is
    unactionable until you know which of two models.yaml files it came from."""
    monkeypatch.setattr("bedrock.inference.create_backend", lambda s: Server())
    monkeypatch.setenv("BEDROCK_CONTEXT_WINDOW", "12288")
    monkeypatch.setenv("BEDROCK_TIMEOUT", "600")

    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out

    assert "models.yaml:" in out and "defaults/models.yaml" in out
    assert "model set:" in out and "laptop" in out
    assert str(data_dir) in out
    assert "Ready." in out, "a fully set up machine should say so plainly"


def test_doctor_separates_worth_knowing_from_would_stop_you(data_dir, monkeypatch, capsys):
    """bedrock's default 120-second timeout is the common case, and it is a
    warning rather than a refusal: a review that hits it resumes."""
    monkeypatch.setattr("bedrock.inference.create_backend", lambda s: Server())
    monkeypatch.setenv("BEDROCK_CONTEXT_WINDOW", "12288")

    assert cli.main(["doctor"]) == 0, "a warning must not make this exit non-zero"
    out = capsys.readouterr().out
    assert "[warn] request timeout" in out
    assert "A review can run." in out


def test_doctor_reports_every_problem_not_just_the_first(data_dir, monkeypatch, capsys):
    """The whole reason it exists. A review refuses at the first blocker, so
    setting a machine up that way is one failed review per mistake."""
    monkeypatch.setattr("bedrock.inference.create_backend", lambda s: Server(models=["something-else"]))
    monkeypatch.setenv("BEDROCK_CONTEXT_WINDOW", "8192")  # too small

    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out

    assert "[FAIL] context window" in out
    assert "[FAIL] models" in out, "the model check must still run after the window failed"
    assert "would stop a review" in out


def test_doctor_names_what_the_server_actually_offers(data_dir, monkeypatch, capsys):
    """OPEN_ISSUES item 16: the `office` model names are guesses, and nobody can
    confirm them before the office server exists. The next best thing is for the
    failure to say what this server calls its models, and which file to write
    them into."""
    monkeypatch.setattr(
        "bedrock.inference.create_backend",
        lambda s: Server(models=["Qwen/Qwen3-30B-A3B-Instruct-2507"]),
    )
    monkeypatch.setenv("BEDROCK_CONTEXT_WINDOW", "12288")

    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out

    assert "Qwen/Qwen3-30B-A3B-Instruct-2507" in out
    assert "--served-model-name" in out
    assert "models.yaml" in out


def test_doctor_reports_a_silent_server_rather_than_crashing(data_dir, monkeypatch, capsys):
    monkeypatch.setattr("bedrock.inference.create_backend", lambda s: Server(silent=True))
    monkeypatch.setenv("BEDROCK_CONTEXT_WINDOW", "12288")

    assert cli.main(["doctor"]) == 1
    assert "not answering" in capsys.readouterr().out


def test_doctor_without_a_server_checks_the_configuration_alone(data_dir, monkeypatch, capsys):
    """The state of a machine before anyone has started Ollama, which is when
    somebody is most likely to run this."""
    def refuse(_settings):
        raise AssertionError("--no-server must not open a backend")

    monkeypatch.setattr("bedrock.inference.create_backend", refuse)
    monkeypatch.setenv("BEDROCK_CONTEXT_WINDOW", "12288")

    assert cli.main(["doctor", "--no-server"]) == 0
    out = capsys.readouterr().out
    assert "[ok  ] context window" in out
    assert "model server:" in out, "it still says which address it would use"


def test_doctor_reports_a_broken_config_table_as_a_finding(data_dir, monkeypatch, capsys):
    """A typo in [litreview] stops every review. It should be the answer to
    "check my setup", not a traceback."""
    monkeypatch.setenv("BEDROCK_LITREVIEW_MODELS", "not-a-real-set")

    assert cli.main(["doctor", "--no-server"]) == 1
    assert "unknown model set" in capsys.readouterr().out


def test_doctor_checks_the_model_set_it_was_asked_about(data_dir, monkeypatch, capsys):
    """`--models office` is how somebody checks the office server's names from a
    laptop, before going anywhere near it."""
    monkeypatch.setattr("bedrock.inference.create_backend", lambda s: Server())
    monkeypatch.setenv("BEDROCK_CONTEXT_WINDOW", "12288")

    assert cli.main(["doctor", "--models", "office"]) == 1
    out = capsys.readouterr().out
    assert "office" in out
    assert "qwen3-30b-a3b-instruct" in out, "the office set's names, not the laptop set's"


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

"""Issue #26: the benchmark runs `service.preflight` before any model call.

Before this, `bench run` built its own settings and scheduler and called the
pipeline with no check at all, so a server started with Ollama's default
window truncated every prompt silently and every number came out wrong
without saying so. These tests drive `python -m bedrock.litreview.bench`
itself, with the backend factory replaced, so they cover the command a person
types rather than a helper it happens to call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import AsyncIterator, Callable

import pytest
import yaml

from bedrock.inference import ModelInfo, RequestOptions, Usage
from bedrock.litreview.bench.__main__ import main, run_reaches_model
from bedrock.litreview.bench.history import read_history
from bedrock.litreview.bench.models import Scorecard
from bedrock.litreview.config import ConfigError, load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import Paper, PaperSignals
from bedrock.litreview.runner import answer_question
from bedrock.litreview.service import preflight
from bedrock.litreview.sources.corpus import write_papers
from tests.litreview.fixtures import QUESTION, corpus
from tests.litreview.test_bench import bench_responder

QID = "psoriasis-test"


class WindowBackend:
    """A model server that offers every model the review needs, reports one
    context window for all of them, and answers like the benchmark's fake."""

    def __init__(self, models: list[str], window: int) -> None:
        self.models, self.window = models, window
        self.calls: list[str] = []

    async def check_connection(self) -> bool:
        return True

    async def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(id=m, context_length=self.window) for m in self.models]

    async def stream_chat(self, model: str, messages: list[dict[str, str]], options: RequestOptions | None = None,
                          on_usage: Callable[[Usage], None] | None = None) -> AsyncIterator[str]:
        self.calls.append(model)
        yield bench_responder(messages[-1]["content"], options.json_schema if options else None)

    async def close(self) -> None:
        pass


@pytest.fixture()
def bench_files(tmp_path, monkeypatch):
    """A question file, its frozen corpus, and a private data directory, laid
    out the way `bench run` reads them."""
    monkeypatch.setenv("BEDROCK_LITREVIEW_DIR", str(tmp_path / "data"))
    # A configured window large enough that only the server's report decides.
    monkeypatch.setenv("BEDROCK_LITREVIEW_CONTEXT_WINDOW", "16384")
    question = tmp_path / "q.yaml"
    question.write_text(yaml.safe_dump({"id": QID, "question": QUESTION, "corpus": {"queries": ["x"]},
                                        "references": {"reviews_max": 1}}))
    corpus_dir = tmp_path / "corpora" / QID
    write_papers(corpus(), corpus_dir / "papers.jsonl")
    write_papers([Paper(
        id="R1", title="IL-23 in psoriasis: a review",
        abstract="IL-23 inhibitors such as guselkumab are highly effective in plaque psoriasis. " * 3,
        signals=PaperSignals(publication_types=["Review"]),
    )], corpus_dir / "references.jsonl")
    (corpus_dir / "reference_targets.json").write_text(json.dumps([
        {"symbol": "IL23A", "max_phase": 4, "drugs": ["GUSELKUMAB"]},
    ]))
    return question, tmp_path / "corpora", tmp_path / "results"


def _models() -> list[str]:
    settings = load_settings()
    return sorted({settings.model_for(job) for job in settings.jobs})


def _use_backend(monkeypatch, backend) -> None:
    # `service.review_backend` builds its backend through bedrock's factory,
    # the one place a backend class is chosen - and the seam tests replace.
    monkeypatch.setattr("bedrock.inference.create_backend", lambda *_a, **_k: backend)


def _scorecards(results: Path) -> list[Path]:
    return sorted(results.glob(f"{QID}/*/scorecard.json"))


def test_a_4096_token_window_stops_the_run_before_any_model_call(bench_files, monkeypatch, capsys):
    question, corpora, results = bench_files
    backend = WindowBackend(_models(), window=4096)
    _use_backend(monkeypatch, backend)

    code = main(["run", str(question), "--corpora", str(corpora), "--results", str(results)])

    assert code != 0
    assert backend.calls == [], "a model was called before preflight refused the run"
    assert not (results / "history.jsonl").exists(), "a refused run wrote a history row"
    with pytest.raises(ConfigError) as refused:
        asyncio.run(preflight(load_settings(), backend))
    err = capsys.readouterr().err
    assert str(refused.value) in err, "preflight's message is printed unchanged"
    assert "4,096-token context window" in err and "12,288" in err


def test_the_scorecard_records_the_window_preflight_established(bench_files, monkeypatch):
    question, corpora, results = bench_files
    backend = WindowBackend(_models(), window=16384)
    _use_backend(monkeypatch, backend)

    code = main(["run", str(question), "--corpora", str(corpora), "--results", str(results),
                 "--skip-baselines", "--skip-distractors", "--skip-perturb"])

    assert code == 0 and backend.calls
    (path,) = _scorecards(results)
    card = Scorecard.model_validate_json(path.read_text())
    assert card.context_windows == {"qwen3:8b": 16384}
    assert card.report_context_windows == {"qwen3:8b": 16384} and card.report_context_source == "this run"
    md = path.with_name("scorecard.md").read_text()
    assert "Context window: qwen3:8b 16,384 tokens (checked by preflight before the first model call)" in md
    (row,) = read_history(results / "history.jsonl")
    assert row.context_windows == row.report_context_windows == {"qwen3:8b": 16384}


def test_a_stored_report_judged_from_a_file_needs_no_server(bench_files, monkeypatch, capsys):
    question, corpora, results = bench_files

    def no_backend(*_a, **_k):
        raise AssertionError("the offline path opened a model server")

    monkeypatch.setattr("bedrock.inference.create_backend", no_backend)
    settings = load_settings()
    db = Database(settings.data_dir / "bench.sqlite3")
    stored = answer_question(QUESTION, settings=settings, client=FakeLLMClient(responder=bench_responder),
                             papers=corpus(), db=db, out_dir=settings.data_dir / "stored")

    code = main(["run", str(question), "--corpora", str(corpora), "--results", str(results),
                 "--from-run", stored.run.id, "--judge-file", str(results.parent / "answers.jsonl"),
                 "--skip-baselines", "--skip-perturb"])

    assert code == 0
    (path,) = _scorecards(results)
    assert capsys.readouterr().out.strip() == str(path.with_name("scorecard.md"))
    card = Scorecard.model_validate_json(path.read_text())
    assert card.context_windows is None and card.report_context_windows is None
    md = path.with_name("scorecard.md").read_text()
    assert "Context window the report was written at: not recorded" in md
    assert "Context window of this run's model calls: no model was called" in md


def test_a_re_scored_report_carries_the_window_it_was_written_at(bench_files, monkeypatch):
    """A live run records its window; re-scoring the same report offline
    reads it back from that run's history row."""
    question, corpora, results = bench_files
    _use_backend(monkeypatch, WindowBackend(_models(), window=16384))
    assert main(["run", str(question), "--corpora", str(corpora), "--results", str(results),
                 "--skip-baselines", "--skip-distractors", "--skip-perturb"]) == 0
    (first,) = read_history(results / "history.jsonl")
    db = Database(load_settings().data_dir / "bench.sqlite3")
    run_id = next(r.id for r in db.recent_runs() if db.report_id_for(r.id) == first.report_id)

    monkeypatch.setattr("bedrock.inference.create_backend",
                        lambda *_a, **_k: pytest.fail("the offline re-score opened a model server"))
    assert main(["run", str(question), "--corpora", str(corpora), "--results", str(results),
                 "--from-run", run_id, "--judge-file", str(results.parent / "answers.jsonl"),
                 "--skip-baselines", "--skip-perturb"]) == 0
    rescored = read_history(results / "history.jsonl")[-1]
    assert rescored.report_id == first.report_id
    assert rescored.context_windows is None
    assert rescored.report_context_windows == {"qwen3:8b": 16384}
    card = Scorecard.model_validate_json(_scorecards(results)[-1].read_text())
    assert card.report_context_source == first.created
    md = _scorecards(results)[-1].with_name("scorecard.md").read_text()
    assert f"qwen3:8b 16,384 tokens (recorded by the run of {first.created})" in md


@pytest.mark.parametrize("flags, reaches", [
    ([], True),                                                                  # a fresh pipeline run
    (["--from-run", "r"], True),                                                 # baselines and a model judge
    (["--from-run", "r", "--skip-baselines"], True),                             # a model judge
    (["--from-run", "r", "--judge-file", "a.jsonl"], True),                      # baselines
    (["--judge-file", "a.jsonl", "--skip-baselines"], True),                     # a fresh pipeline run
    (["--from-run", "r", "--judge-file", "a.jsonl", "--skip-baselines"], False),  # files only
    (["--from-run", "r", "--judge-file", "a.jsonl", "--skip-baselines", "--skip-perturb"], False),
])
def test_which_runs_reach_a_model(flags, reaches):
    """Perturbations are judged by whichever judge the run has, so with a
    judge file they read the file, not a model."""
    args = argparse.Namespace(from_run=None, skip_baselines=False, judge_file=None, skip_perturb=False)
    it = iter(flags)
    for flag in it:
        name = flag.lstrip("-").replace("-", "_")
        setattr(args, name, True if name.startswith("skip") else next(it))
    assert run_reaches_model(args) is reaches


def test_history_rows_written_before_preflight_read_as_not_recorded(tmp_path, capsys):
    old = {"question_id": "q", "created": "20260926T000000Z", "report_id": "abc",
           "overall": {"pipeline": 66.6}, "gates_passed": 9, "gates_total": 9}
    (tmp_path / "history.jsonl").write_text(json.dumps(old) + "\n")
    assert main(["history", "--results", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "| 20260926T000000Z | q | abc | 66.6 | 9 of 9 | not recorded |" in out


def test_every_row_of_the_real_history_still_loads():
    from bedrock.litreview.bench.run import RESULTS

    rows = read_history(RESULTS / "history.jsonl")
    assert rows, "evals/litreview/results/history.jsonl is checked in"
    assert all(r.report_id for r in rows)

"""A running review says it is still working, and saying so changes nothing.

The complaint this answers: `rank` grades one paper per model call and said
nothing until all of them were done - about four minutes of silence at 84
papers - and a whole laptop run is over two hours. Silence and a hung process
look identical, so someone watching cannot tell whether to wait or to kill it.

Two things are worth testing, and the second matters more than the first:

1. The counts are real - they reach the caller, they count the right items,
   and they end at the total rather than stopping short.
2. **The report does not depend on them.** A progress indicator that changed a
   byte of the output would break the determinism this project promises by
   name, so `test_the_report_is_identical_whether_or_not_anyone_is_watching`
   runs the whole pipeline twice and compares the bytes.
"""

from __future__ import annotations

import hashlib

import pytest

from bedrock.litreview.config import load_settings
from bedrock.litreview.db import Database
from bedrock.litreview.progress import format_elapsed, format_tick, safe_tick, throttled
from bedrock.litreview.runner import answer_question
from tests.litreview.fixtures import QUESTION, corpus, fake_client


@pytest.fixture()
def settings(tmp_path):
    s = load_settings(model_set="laptop", env={}, check_endpoint=False)
    s.cache_dir = tmp_path / "cache"
    s.out_dir = tmp_path / "out"
    s.db_path = tmp_path / "db.sqlite3"
    return s


def run(settings, tmp_path, **kwargs):
    return answer_question(
        QUESTION, settings=settings, client=fake_client(), papers=corpus(),
        db=Database(tmp_path / "db.sqlite3"), **kwargs,
    )


# --- the counts ------------------------------------------------------------------


def test_the_silent_stages_all_report_progress(settings, tmp_path):
    """rank, extract and write each make one model call per item and each used
    to say nothing between milestones."""
    seen: list[tuple] = []
    run(settings, tmp_path, tick=lambda *a: seen.append(a))

    stages = {stage for stage, *_ in seen}
    assert {"rank", "extract", "write"} <= stages, f"a silent stage is still silent: {stages}"


def test_each_stage_counts_up_to_its_total_and_never_past_it(settings, tmp_path):
    seen: list[tuple] = []
    run(settings, tmp_path, tick=lambda *a: seen.append(a))

    for stage in ("rank", "extract", "write"):
        counts = [(done, total) for s, done, total, _ in seen if s == stage]
        assert counts, f"no ticks for {stage}"
        totals = {total for _, total in counts}
        assert len(totals) == 1, f"{stage} changed its total mid-stage: {totals}"
        assert max(done for done, _ in counts) == totals.pop(), f"{stage} never reached its total"
        assert all(done >= 0 for done, _ in counts)


def test_progress_lines_and_ticks_are_separate_channels(settings, tmp_path):
    """The milestone lines stay few; the ticks are the many. Sending the ticks
    through `progress` would be 40 toasts in the app."""
    lines: list[str] = []
    ticks: list[tuple] = []
    run(settings, tmp_path, progress=lines.append, tick=lambda *a: ticks.append(a))

    assert len(ticks) > len(lines), "ticks should be the fine-grained channel"
    assert not any(line.startswith("review: ") for line in lines)


# --- and none of it changes the run ------------------------------------------------


def test_the_report_is_identical_whether_or_not_anyone_is_watching(settings, tmp_path):
    """The invariant that matters. `CLAUDE.md`: the same question over the same
    papers produces a byte-identical report."""
    quiet = run(settings, tmp_path)
    quiet_bytes = quiet.paths["json"].read_bytes()

    settings.db_path = tmp_path / "second.sqlite3"
    watched = answer_question(
        QUESTION, settings=settings, client=fake_client(), papers=corpus(),
        db=Database(tmp_path / "second.sqlite3"), tick=lambda *a: None,
    )

    assert watched.report_id == quiet.report_id
    assert hashlib.sha256(watched.paths["json"].read_bytes()).hexdigest() == \
        hashlib.sha256(quiet_bytes).hexdigest()


def test_a_tick_that_raises_cannot_take_the_run_down(settings, tmp_path):
    """Hours of model calls must not be lost to a broken progress indicator."""
    def broken(*_args):
        raise RuntimeError("the status bar is on fire")

    result = run(settings, tmp_path, tick=broken)

    assert result.report_id, "the run should have finished despite the tick raising"


def test_safe_tick_without_a_callback_is_a_no_op():
    safe_tick(None)("rank", 1, 2, "")  # must not raise


# --- formatting --------------------------------------------------------------------


def test_a_tick_line_shows_counts_and_elapsed_but_never_an_estimate():
    line = format_tick("rank", 12, 40, elapsed=130)
    assert line == "review: rank 12/40 · 2m 10s"
    for guess in ("remaining", "left", "eta", "~"):
        assert guess not in line.lower()


def test_a_tick_line_carries_its_sub_position():
    assert format_tick("extract", 3, 16, "paper 3 chunk 2/6") == "review: extract 3/16 · paper 3 chunk 2/6"


@pytest.mark.parametrize("seconds,shown", [(0, "0s"), (45, "45s"), (130, "2m 10s"), (3671, "1h 01m")])
def test_elapsed_reads_the_way_a_person_would_say_it(seconds, shown):
    assert format_elapsed(seconds) == shown


# --- the shell's throttle -----------------------------------------------------------


def test_the_shell_prints_about_one_line_per_tenth_not_one_per_item():
    """40 permanent lines in a terminal is the noise this avoids."""
    clock = [0.0]
    lines: list[str] = []
    tick = throttled(lines.append, now=lambda: clock[0])

    for i in range(1, 41):
        clock[0] += 1.0
        tick("rank", i, 40)

    assert 8 <= len(lines) <= 12, f"expected about a tenth of 40, got {len(lines)}: {lines}"
    assert "rank 1/40" in lines[0], "the first item should say the stage has started"
    assert "rank 40/40" in lines[-1], "the last item should say it finished"


def test_a_slow_stage_does_not_print_more_lines_than_a_fast_one():
    """The bug the first draft of `throttled` had: with a small `min_seconds`,
    the OR meant slower calls tripped the time branch on *every* item, so a
    slow stage was noisier than a fast one - backwards."""
    def lines_for(seconds_per_item: float) -> int:
        clock = [0.0]
        out: list[str] = []
        tick = throttled(out.append, now=lambda: clock[0])
        for i in range(1, 41):
            clock[0] += seconds_per_item
            tick("rank", i, 40)
        return len(out)

    fast, slow = lines_for(1.0), lines_for(30.0)
    assert slow <= fast * 2, f"a slow stage printed {slow} lines against a fast stage's {fast}"
    assert slow < 40, "a slow stage must not degrade to one line per item"


def test_the_throttle_still_speaks_when_a_stage_is_slow_but_short():
    """Three items over ten minutes must not be silence then three lines."""
    clock = [0.0]
    lines: list[str] = []
    tick = throttled(lines.append, now=lambda: clock[0])

    for i in range(1, 4):
        clock[0] += 200.0
        tick("extract", i, 3)

    assert len(lines) == 3


def test_stages_are_throttled_independently():
    clock = [0.0]
    lines: list[str] = []
    tick = throttled(lines.append, now=lambda: clock[0])

    tick("rank", 1, 40)
    tick("extract", 1, 16)

    assert len(lines) == 2, "a new stage's first line must not be swallowed by another stage's throttle"

"""`/confidence` — the numbers, the intervals, and the PAE overview.

Confidence is the part of this feature where a pretty display can do real
harm: a scientist who reads "very high" as "verified", or who reads a PAE
heatmap along the wrong axis, has been misled by the interface rather than by
the model. So these tests check the wording and the direction as carefully as
they check the arithmetic.
"""

from __future__ import annotations

from textual.widgets import Input

from bedrock.app import BedrockApp
from bedrock.config import Settings
from bedrock.ui.confidence_view import format_confidence_report, format_pae_overview
from bedrock.ui.widgets import SystemNotice

from tests.conftest import (
    FakeBackend,
    FakeRunCatalog,
    make_confidence_report,
    make_pae,
    make_services,
)


def make_app(reports) -> BedrockApp:
    services = make_services(runs=FakeRunCatalog(reports=reports))
    return BedrockApp(
        settings=Settings(model="test-model"),
        backend=FakeBackend(),
        services=services,
    )


def notices(app) -> str:
    return "\n".join(str(widget.source) for widget in app.query(SystemNotice))


async def submit(app, pilot, text: str) -> None:
    app.query_one("#prompt", Input).value = text
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


async def test_the_page_shows_bands_intervals_model_and_the_caveats(tmp_path):
    report = make_confidence_report(
        "run_1",
        model_id="model_3",
        plddt=(95.0, 92.0, 80.0, 65.0, 45.0, 40.0, 99.0),
        low_regions=((4, 6),),
        very_low_regions=((5, 6),),
    )
    app = make_app({"run_1": report})

    async with app.run_test() as pilot:
        await submit(app, pilot, "/confidence run_1")

        text = notices(app)
        # Four bands, with counts and shares.
        assert "Very low" in text and "Low" in text and "Confident" in text and "Very high" in text
        assert "| 3 | 42.9% |" in text  # 95, 92, 99 are >= 90
        assert "| 2 | 28.6% |" in text  # 45 and 40 are < 50
        # Intervals, labelled as what they actually are.
        assert "input sequence positions" in text.lower()
        assert "pLDDT below 70: 4–6" in text
        assert "pLDDT below 50: 5–6" in text
        # The model that was actually scored, not the rank.
        assert "model_3" in text
        # The caveat, verbatim.
        assert (
            "pLDDT estimates local model confidence; it is not experimental "
            "validation or a stability measurement." in text
        )
        assert "disorder" in text


async def test_ranked_zero_pointing_at_model_three_displays_model_three(tmp_path):
    """The rank is not the model. Whatever pairing the catalogue reports is
    what the page names, with no reinterpretation here."""
    app = make_app({"run_1": make_confidence_report("run_1", model_id="model_3")})

    async with app.run_test() as pilot:
        await submit(app, pilot, "/confidence run_1")

        assert "model_3" in notices(app)
        assert "ranked_0" not in notices(app)


async def test_a_missing_metric_is_unavailable_not_zero(tmp_path):
    report = make_confidence_report(
        "run_1",
        plddt=None,
        low_regions=(),
        very_low_regions=(),
        notices=("No supported confidence JSON was found next to the selected model.",),
    )
    app = make_app({"run_1": report})

    async with app.run_test() as pilot:
        await submit(app, pilot, "/confidence run_1")

        text = notices(app)
        assert "unavailable" in text
        assert "0.0%" not in text  # an absent metric is never displayed as a score
        assert "No supported confidence JSON" in text


async def test_one_broken_metric_does_not_suppress_the_other(tmp_path):
    """PAE and pLDDT come from different files; a bad PAE must not hide good
    per-residue scores, and the notice must say which one is missing."""
    report = make_confidence_report(
        "run_1",
        pae=None,
        notices=("Predicted aligned error JSON failed its hash check.",),
    )
    app = make_app({"run_1": report})

    async with app.run_test() as pilot:
        await submit(app, pilot, "/confidence run_1")

        text = notices(app)
        assert "Very high" in text  # pLDDT still rendered
        assert "Predicted aligned error is unavailable" in text
        assert "failed its hash check" in text


async def test_an_unknown_run_is_a_plain_error(tmp_path):
    app = make_app({})

    async with app.run_test() as pilot:
        await submit(app, pilot, "/confidence nope")

        assert "Could not read confidence" in notices(app)
        assert app.is_running


# ---- The PAE overview ------------------------------------------------------


def _grid_lines(rendered: str) -> list[str]:
    inside = rendered.split("```text")[1].split("```")[0].splitlines()
    return [line for line in inside if "|" in line]


def test_the_heatmap_keeps_its_direction_on_an_asymmetric_matrix():
    """``[i][j]`` is aligned on *i*, scoring *j*. A matrix with one high
    corner must show that error in that corner and nowhere else - transposing
    it would tell a scientist the wrong half of the structure is misplaced."""
    pae = make_pae(
        values=((1.0, 25.0), (1.0, 1.0)),
        ranges=((1, 4), (5, 8)),
        maximum_angstrom=31.75,
    )

    rows = _grid_lines(format_pae_overview(pae))

    assert len(rows) == 2
    assert rows[0].startswith("1-4")
    assert rows[0].endswith(". #")  # aligned on 1-4, residues 5-8 badly placed
    assert rows[1].startswith("5-8")
    assert rows[1].endswith(". .")  # aligned on 5-8, nothing is high


def test_the_heatmap_states_its_aggregation_scale_ranges_and_source():
    pae = make_pae(
        values=((1.0, 25.0), (1.0, 1.0)),
        ranges=((1, 4), (5, 8)),
        source_path="/private/runs/r1/pae_model_3.json",
    )

    rendered = format_pae_overview(pae)

    assert "maximum per block" in rendered
    assert "rows = aligned residue" in rendered and "columns = scored residue" in rendered
    assert "ångström" in rendered
    assert "0–5 Å" in rendered and "20 Å and above" in rendered  # numeric legend
    assert "1:1-4" in rendered and "2:5-8" in rendered           # residue ranges
    assert "/private/runs/r1/pae_model_3.json" in rendered       # the full matrix
    assert "does not establish correct domain placement" in rendered


def test_sparse_high_error_survives_aggregation():
    """One bad cell in a block of otherwise excellent values must still show.
    Averaging would erase it; the maximum is what makes the overview safe."""
    values = [[1.0] * 4 for _ in range(4)]
    values[0][3] = 29.0
    pae = make_pae(values=values, ranges=((1, 2), (3, 4), (5, 6), (7, 8)))

    rows = _grid_lines(format_pae_overview(pae))

    assert rows[0].endswith(". . . #")


def test_the_overview_is_bounded_even_if_a_larger_grid_arrives():
    values = [[1.0] * 40 for _ in range(40)]
    pae = make_pae(values=values, ranges=[(i * 10 + 1, i * 10 + 10) for i in range(40)])

    rows = _grid_lines(format_pae_overview(pae))

    assert len(rows) == 32
    assert len(rows[0].split("|")[1].split()) == 32


def test_the_report_renders_without_a_pae_section_when_there_is_none():
    rendered = format_confidence_report(make_confidence_report(pae=None))

    assert "Predicted aligned error is unavailable" in rendered
    assert "```text" not in rendered


async def test_the_heatmap_survives_onto_the_actual_screen(tmp_path):
    """Not the Markdown source — the rendered screen.

    A fenced block is the only way to keep a grid aligned inside a Markdown
    widget. If that ever stops holding, the rows reflow into prose and the
    heatmap silently becomes nonsense, which no assertion on the source
    string would catch.
    """
    import re

    report = make_confidence_report(
        "run_1",
        plddt=(95.0,) * 8 + (40.0,) * 8,
        pae=make_pae(values=((2.0, 28.0), (1.0, 2.0)), ranges=((1, 8), (9, 16))),
    )
    app = make_app({"run_1": report})

    async with app.run_test(size=(100, 60)) as pilot:
        await submit(app, pilot, "/confidence run_1")
        screen = re.sub(r"<[^>]+>", "", app.export_screenshot()).replace("&#160;", " ")

    # Substrings, not whole lines: the screen has a sidebar and borders down
    # the left, and this test is about the grid keeping its shape, not about
    # what else is drawn beside it.
    assert "1-8 | . #" in screen        # aligned on 1-8, residues 9-16 are far off
    assert "9-16 | . ." in screen       # and not the other way round

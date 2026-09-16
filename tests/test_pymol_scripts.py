"""The fixed PyMOL scripts, and how they are launched.

PyMOL itself is not installed in this test environment and should not need to
be: the styling is ordinary Python driving a ``cmd`` object, so a recording
double proves exactly which commands a real PyMOL would receive.

The test that matters most is the boring-looking one at the end of the first
section: the headless preview and the interactive window must colour by the
same thresholds. If those ever drift, a scientist reads a saved PNG one way
and the live structure another, and nothing in either picture says so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from bedrock.biotech import _pymol_style
from bedrock.biotech.errors import ToolIntegrationError
from bedrock.biotech.pymol import PyMOLAdapter


class FakeCmd:
    """Records what PyMOL would have been told to do."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []

    def __getattr__(self, name: str):
        def record(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return record

    def named(self, name: str) -> list[tuple[tuple, dict]]:
        return [(args, kwargs) for called, args, kwargs in self.calls if called == name]

    def colour_rules(self) -> list[tuple]:
        """Just the confidence colouring: (colour name, selection)."""
        return [args for args, _ in self.named("color") if len(args) == 2 and "b " in args[1]]


ENV = {"BEDROCK_PYMOL_INPUT": "/private/runs/r1/ranked_0.pdb"}
PLDDT_ENV = {**ENV, "BEDROCK_PYMOL_CONFIDENCE": "plddt"}
RENDER_ENV = {
    **PLDDT_ENV,
    "BEDROCK_PYMOL_PREVIEW": "/private/runs/r1/pymol/preview.png",
    "BEDROCK_PYMOL_SESSION": "/private/runs/r1/pymol/structure.pse",
}


def test_confidence_colouring_uses_the_four_published_bands():
    cmd = FakeCmd()

    _pymol_style.apply_style(cmd, "/private/runs/r1/ranked_0.pdb", "plddt")

    assert cmd.colour_rules() == [
        ("plddt_very_low", "prediction and b < 50"),
        ("plddt_low", "prediction and b >= 50 and b < 70"),
        ("plddt_confident", "prediction and b >= 70 and b < 90"),
        ("plddt_very_high", "prediction and b >= 90"),
    ]
    assert cmd.named("load")[0][0] == ("/private/runs/r1/ranked_0.pdb", "prediction")
    assert ("cartoon", "polymer.protein") in [args for args, _ in cmd.named("show")]


def test_an_arbitrary_structure_stays_neutral():
    """An experimental file's B-factors are displacement, not confidence.
    Painting them in AlphaFold's colours would invent a claim it never made."""
    cmd = FakeCmd()

    _pymol_style.apply_style(cmd, "/home/scientist/crystal.pdb", "")

    assert cmd.colour_rules() == []
    assert cmd.named("set_color") == []
    assert ("gray70", "all") in [args for args, _ in cmd.named("color")]


def test_headless_and_interactive_colour_by_identical_thresholds():
    headless, interactive = FakeCmd(), FakeCmd()

    _pymol_style.render(headless, RENDER_ENV)
    _pymol_style.open_interactive(interactive, PLDDT_ENV)

    assert headless.colour_rules() == interactive.colour_rules()
    assert headless.named("set_color") == interactive.named("set_color")


def test_only_the_headless_path_writes_files():
    headless, interactive = FakeCmd(), FakeCmd()

    _pymol_style.render(headless, RENDER_ENV)
    _pymol_style.open_interactive(interactive, PLDDT_ENV)

    assert headless.named("png")[0][0] == ("/private/runs/r1/pymol/preview.png",)
    assert headless.named("save")[0][0] == ("/private/runs/r1/pymol/structure.pse",)
    assert interactive.named("png") == []
    assert interactive.named("save") == []  # nothing is written to a scientist's disk


def test_the_terminal_bands_and_the_pymol_bands_have_the_same_edges():
    """One set of numbers, two renderings of it."""
    from bedrock.ui.confidence_view import PLDDT_BANDS as TERMINAL_BANDS

    edges = [selection for _, selection in _pymol_style.PLDDT_BANDS]
    assert edges == ["b < 50", "b >= 50 and b < 70", "b >= 70 and b < 90", "b >= 90"]
    assert [lower for _, lower, _ in TERMINAL_BANDS] == [0.0, 50.0, 70.0, 90.0]
    assert [upper for _, _, upper in TERMINAL_BANDS] == [50.0, 70.0, 90.0, 100.0]


# ---- How the scripts are launched ------------------------------------------


class RecordingLauncher:
    def __init__(self) -> None:
        self.argv: tuple[str, ...] = ()
        self.env: dict[str, str] = {}

    async def __call__(self, argv, env) -> None:
        self.argv = tuple(argv)
        self.env = dict(env)


def structure(tmp_path: Path) -> Path:
    path = tmp_path / "ranked_0.pdb"
    path.write_text("ATOM\n")
    return path


async def test_interactive_opening_passes_an_argv_list_and_the_fixed_script(tmp_path):
    launcher = RecordingLauncher()
    adapter = PyMOLAdapter(sys.executable, launcher=launcher)

    opened = await adapter.open_interactive(structure(tmp_path), confidence_metric="plddt")

    assert opened == structure(tmp_path).resolve()
    assert launcher.argv[0] == sys.executable
    assert launcher.argv[1] == "-r"
    assert launcher.argv[2].endswith("_pymol_interactive.py")
    # The structure arrives as data, never as part of a command PyMOL parses.
    assert launcher.env["BEDROCK_PYMOL_INPUT"] == str(structure(tmp_path).resolve())
    assert launcher.env["BEDROCK_PYMOL_CONFIDENCE"] == "plddt"


async def test_the_child_environment_does_not_carry_credentials(tmp_path, monkeypatch):
    monkeypatch.setenv("BEDROCK_API_KEY", "do-not-pass-this")
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "nor-this")
    launcher = RecordingLauncher()
    adapter = PyMOLAdapter(sys.executable, launcher=launcher)

    await adapter.open_interactive(structure(tmp_path))

    assert "BEDROCK_API_KEY" not in launcher.env
    assert "SOME_SERVICE_TOKEN" not in launcher.env
    assert launcher.env["BEDROCK_PYMOL_CONFIDENCE"] == ""  # neutral by default


async def test_a_saved_session_is_still_not_an_accepted_input(tmp_path):
    """Opening a supplied ``.pse`` would mean running whatever it contains.
    Interactive styling exists precisely so that is not needed."""
    session = tmp_path / "structure.pse"
    session.write_text("not really a session")
    adapter = PyMOLAdapter(sys.executable, launcher=RecordingLauncher())

    with pytest.raises(ToolIntegrationError, match=r"\.pdb"):
        await adapter.open_interactive(session)


async def test_an_unsupported_confidence_metric_is_refused(tmp_path):
    adapter = PyMOLAdapter(sys.executable, launcher=RecordingLauncher())

    with pytest.raises(ToolIntegrationError, match="Unsupported confidence metric"):
        await adapter.open_interactive(structure(tmp_path), confidence_metric="bfactor")


async def test_a_missing_structure_is_refused_before_pymol_starts(tmp_path):
    launcher = RecordingLauncher()
    adapter = PyMOLAdapter(sys.executable, launcher=launcher)

    with pytest.raises(ToolIntegrationError, match="does not exist"):
        await adapter.open_interactive(tmp_path / "absent.pdb")

    assert launcher.argv == ()

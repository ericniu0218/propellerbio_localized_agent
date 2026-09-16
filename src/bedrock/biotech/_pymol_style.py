"""The one definition of how Bedrock styles a predicted structure in PyMOL.

Both PyMOL entry points import this module: the headless renderer that writes
the PNG and the ``.pse`` session, and the interactive opener that puts a
structure on the scientist's desktop. They share this file so that the
colours and - much more importantly - the *thresholds* cannot drift apart.
A preview whose bands meant something slightly different from the window the
scientist is looking at would be a quiet, plausible, wrong result.

The bands are AlphaFold's published pLDDT bands, and they are the same four
the terminal confidence page uses (``bedrock/ui/confidence_view.py``).

Two safety properties are structural rather than checked:

* **Everything here is fixed code.** No path, name or selection is built from
  a string this program was given. The structure to load arrives as *data*,
  through an environment variable, so there is nothing for a crafted file
  name to inject into a PyMOL command.
* **Unknown provenance stays grey.** Colouring by the B-factor column only
  happens when the caller explicitly declares ``plddt``. An experimental
  coordinate file's B-factors are crystallographic displacement, not
  confidence, and painting them in AlphaFold's colours would invent a claim
  the file never made.

There is no ``pymol`` import at module level, so the styling logic can be
tested with a fake ``cmd`` object in the ordinary test suite.
"""

from __future__ import annotations

from typing import Any, Mapping

OBJECT_NAME = "prediction"

SUPPORTED_METRICS = ("plddt",)

# name, RGB - AlphaFold's very low / low / confident / very high colours.
PLDDT_COLORS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("plddt_very_low", (1.0, 0.49, 0.27)),
    ("plddt_low", (1.0, 0.86, 0.07)),
    ("plddt_confident", (0.40, 0.80, 0.93)),
    ("plddt_very_high", (0.00, 0.33, 0.66)),
)

# name, PyMOL selection over the B-factor column. Half-open bands, in the same
# order and with the same edges as ui/confidence_view.PLDDT_BANDS.
PLDDT_BANDS: tuple[tuple[str, str], ...] = (
    ("plddt_very_low", "b < 50"),
    ("plddt_low", "b >= 50 and b < 70"),
    ("plddt_confident", "b >= 70 and b < 90"),
    ("plddt_very_high", "b >= 90"),
)


def apply_style(cmd: Any, source: str, confidence: str = "") -> None:
    """Load ``source`` and give it Bedrock's standard appearance."""
    cmd.load(source, OBJECT_NAME)
    cmd.hide("everything", "all")
    cmd.show("cartoon", "polymer.protein")
    cmd.color("gray70", "all")

    if confidence == "plddt":
        for name, rgb in PLDDT_COLORS:
            cmd.set_color(name, list(rgb))
        for name, selection in PLDDT_BANDS:
            cmd.color(name, f"{OBJECT_NAME} and {selection}")

    cmd.orient(OBJECT_NAME)
    cmd.zoom(OBJECT_NAME, buffer=4)


def render(cmd: Any, environ: Mapping[str, str]) -> None:
    """Headless: style the structure, then write the preview and session."""
    apply_style(
        cmd,
        environ["BEDROCK_PYMOL_INPUT"],
        environ.get("BEDROCK_PYMOL_CONFIDENCE", ""),
    )
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.png(environ["BEDROCK_PYMOL_PREVIEW"], width=1200, height=900, dpi=150, ray=1)
    cmd.save(environ["BEDROCK_PYMOL_SESSION"])


def open_interactive(cmd: Any, environ: Mapping[str, str]) -> None:
    """Desktop: the same styling, and then PyMOL is the scientist's to drive.

    Nothing is written and no saved session is loaded - this opens coordinates
    and colours them, which is exactly what ``/pymol`` promises.
    """
    apply_style(
        cmd,
        environ["BEDROCK_PYMOL_INPUT"],
        environ.get("BEDROCK_PYMOL_CONFIDENCE", ""),
    )

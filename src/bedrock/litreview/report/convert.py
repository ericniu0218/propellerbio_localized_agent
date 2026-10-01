"""The target-proposal deck as a PDF, for anyone without PowerPoint.

Converted from the real .pptx by LibreOffice, headless, on this machine - not
redrawn by a second renderer - so the PDF is the same slides, charts and
figures as the PowerPoint, and nothing can drift between the two. Nothing
leaves the machine: LibreOffice reads one temporary file and writes another.

LibreOffice is not a Python package, so it is optional: `soffice()` finds it
(BEDROCK_SOFFICE, then `soffice`/`libreoffice` on PATH, then the macOS
application), and the Research tab offers "Slides (PDF)" only when it is
there. Install it on the server with the system's package manager
(`apt install libreoffice-impress`, `dnf install libreoffice-impress`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

_MAC_APP = Path("/Applications/LibreOffice.app/Contents/MacOS/soffice")


class ConversionUnavailable(RuntimeError):
    """No LibreOffice on this machine."""


class ConversionFailed(RuntimeError):
    """LibreOffice ran and did not produce a PDF."""


def soffice() -> str | None:
    """The LibreOffice binary to convert with, or None when there is none."""
    configured = os.environ.get("BEDROCK_SOFFICE", "").strip()
    if configured:
        return configured if Path(configured).exists() else None
    for name in ("soffice", "libreoffice"):
        found = shutil.which(name)
        if found:
            return found
    return str(_MAC_APP) if _MAC_APP.exists() else None


def pptx_to_pdf(deck: bytes, *, timeout: float = 180.0) -> bytes:
    """The deck as a PDF. Raises ConversionUnavailable or ConversionFailed."""
    binary = soffice()
    if binary is None:
        raise ConversionUnavailable(
            "Slides as PDF need LibreOffice on the server (apt install libreoffice-impress). "
            "The PowerPoint file has the same slides.")
    with tempfile.TemporaryDirectory(prefix="bedrock-slides-") as tmp:
        work = Path(tmp)
        source = work / "deck.pptx"
        source.write_bytes(deck)
        # A profile of its own: a LibreOffice already open for someone else, or
        # a second conversion at the same time, would otherwise share one and
        # fail with nothing written.
        profile = (work / "profile").as_uri()
        command = [binary, f"-env:UserInstallation={profile}", "--headless", "--norestore", "--nolockcheck",
                   "--convert-to", "pdf", "--outdir", str(work), str(source)]
        try:
            done = subprocess.run(command, capture_output=True, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise ConversionFailed(f"LibreOffice did not finish converting the slides in {timeout:.0f}s") from exc
        except OSError as exc:
            raise ConversionUnavailable(f"LibreOffice could not be started: {exc}") from exc
        result = work / "deck.pdf"
        if done.returncode != 0 or not result.exists():
            detail = (done.stderr or done.stdout or b"").decode("utf-8", "replace").strip()[-400:]
            raise ConversionFailed(f"LibreOffice could not convert the slides{': ' + detail if detail else ''}")
        return result.read_bytes()

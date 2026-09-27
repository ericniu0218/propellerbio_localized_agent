"""Choices a scientist makes in the app, kept on this machine.

Today there is one: whether a review checks its papers for retractions and
other publication notices automatically, or only when someone runs the check
by hand. It is saved in `preferences.json` in the review data directory
(0600, like the database), set from the app with `/reviews settings`, and
read by every front end. A site can set its default in the `[litreview]`
table of `config.toml` (`integrity_checks = "manual"`), and the environment
variable `BEDROCK_LITREVIEW_INTEGRITY_CHECKS` overrides both - that is how
the tests keep every review offline.

The wording below is shown to scientists as it is. Keep it plain.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Mapping

INTEGRITY_MODES = ("auto", "manual")
DEFAULT_INTEGRITY_MODE = "auto"
FILE_NAME = "preferences.json"

INTEGRITY_EXPLAINED = {
    "auto": (
        "**Automatic (recommended).** Every time a review starts, the app asks PubMed and Crossref whether any of "
        "its papers has been retracted, has an expression of concern, or has been corrected - so the report "
        "reflects what is known today.\n"
        "- *Good:* you never have to remember to check; a retracted paper is kept out and a questioned one is "
        "flagged before you read the report.\n"
        "- *Trade-off:* the app sends the papers' ID numbers (never your question or the report) to those two "
        "services, so they could tell which papers this machine is reading. A review takes up to about a "
        "minute longer, and on a machine with no internet the check is skipped and the report says so. Papers "
        "checked in the last 7 days are not asked about again."
    ),
    "manual": (
        "**Manual.** Reviews never contact anyone about retractions. The status comes only from what is already "
        "on this machine - from the search itself, or from a check someone runs by hand with "
        "`bedrock-litreview integrity refresh`.\n"
        "- *Good:* nothing about which papers you read leaves the machine unless you choose; reviews are a "
        "little faster.\n"
        "- *Trade-off:* a paper retracted or questioned since it was collected may appear in a report without a "
        "warning, until someone runs the check. Reports say how many of their papers have never been checked."
    ),
}


class PreferenceError(ValueError):
    """A preference value that is not one of the allowed ones."""


def _path(data_dir: Path) -> Path:
    return Path(data_dir) / FILE_NAME


def load(data_dir: Path) -> dict:
    """The saved preferences, or {} when there are none or the file is unreadable."""
    try:
        data = json.loads(_path(data_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def integrity_mode(data_dir: Path, table: Mapping | None = None, env: Mapping[str, str] | None = None) -> str:
    """auto | manual: environment, then the app's saved choice, then config.toml, then auto."""
    env = os.environ if env is None else env
    for source, value in (
        ("BEDROCK_LITREVIEW_INTEGRITY_CHECKS", env.get("BEDROCK_LITREVIEW_INTEGRITY_CHECKS")),
        ("the saved preference", load(data_dir).get("integrity_checks")),
        ("[litreview] integrity_checks", (table or {}).get("integrity_checks")),
    ):
        if value is None or value == "":
            continue
        value = str(value).strip().lower()
        if value not in INTEGRITY_MODES:
            raise PreferenceError(f"{source} must be 'auto' or 'manual', not {value!r}")
        return value
    return DEFAULT_INTEGRITY_MODE


def save_integrity_mode(data_dir: Path, mode: str) -> Path:
    """Save the choice made in the app. Written atomically, readable only by this account."""
    mode = mode.strip().lower()
    if mode not in INTEGRITY_MODES:
        raise PreferenceError(f"retraction checks can be 'auto' or 'manual', not {mode!r}")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    data = load(data_dir)
    data["integrity_checks"] = mode
    fd, tmp = tempfile.mkstemp(dir=str(data_dir), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, sort_keys=True)
        os.chmod(tmp, 0o600)
        os.replace(tmp, _path(data_dir))
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    return _path(data_dir)

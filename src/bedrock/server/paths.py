"""Turning an identity into a folder name without letting it choose the folder.

The meeting recorder files each person's audio under
``<recordings>/<user id>/``. With single sign-on the user id is the Keycloak
username, which the company's directory decides, not this server; a name such
as ``..`` or one containing ``/`` would put files somewhere else. So an id is
used as a folder name only if it is plainly safe, and is otherwise replaced by a
stable, unambiguous stand-in derived from it.
"""

from __future__ import annotations

import hashlib
import re

_SAFE = re.compile(r"[A-Za-z0-9._@-]{1,64}")


def safe_component(identity: str) -> str:
    if _SAFE.fullmatch(identity) and identity not in {".", ".."}:
        return identity
    return "u-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]

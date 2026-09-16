"""Fixed PyMOL script: headless preview and session.

All file values arrive as data through the environment; the styling itself
lives in ``_pymol_style`` so that the interactive path cannot drift from it.
PyMOL runs this file directly, in its own interpreter, where the ``bedrock``
package is usually not importable - hence the sibling-directory import.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _pymol_style  # noqa: E402  (must follow the path fix above)
from pymol import cmd  # noqa: E402

_pymol_style.render(cmd, os.environ)

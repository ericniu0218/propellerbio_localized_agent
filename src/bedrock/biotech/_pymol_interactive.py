"""Fixed PyMOL script: open a structure on the desktop, styled.

The counterpart of ``_pymol_render``. PyMOL keeps running after this script
finishes, leaving the scientist with a normal PyMOL window containing the
structure, coloured by the same confidence bands the saved preview used.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _pymol_style  # noqa: E402  (must follow the path fix above)
from pymol import cmd  # noqa: E402

_pymol_style.open_interactive(cmd, os.environ)

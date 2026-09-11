"""Fixed PyMOL script; all file values arrive as data through the environment."""

from __future__ import annotations

import os

from pymol import cmd


source = os.environ["BEDROCK_PYMOL_INPUT"]
preview = os.environ["BEDROCK_PYMOL_PREVIEW"]
session = os.environ["BEDROCK_PYMOL_SESSION"]
confidence = os.environ.get("BEDROCK_PYMOL_CONFIDENCE", "")

cmd.load(source, "prediction")
cmd.hide("everything", "all")
cmd.show("cartoon", "polymer.protein")
cmd.color("gray70", "all")

if confidence == "plddt":
    cmd.set_color("plddt_very_low", [1.0, 0.49, 0.27])
    cmd.set_color("plddt_low", [1.0, 0.86, 0.07])
    cmd.set_color("plddt_confident", [0.40, 0.80, 0.93])
    cmd.set_color("plddt_very_high", [0.00, 0.33, 0.66])
    cmd.color("plddt_very_low", "prediction and b < 50")
    cmd.color("plddt_low", "prediction and b >= 50 and b < 70")
    cmd.color("plddt_confident", "prediction and b >= 70 and b < 90")
    cmd.color("plddt_very_high", "prediction and b >= 90")

cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.orient("prediction")
cmd.zoom("prediction", buffer=4)
cmd.png(preview, width=1200, height=900, dpi=150, ray=1)
cmd.save(session)

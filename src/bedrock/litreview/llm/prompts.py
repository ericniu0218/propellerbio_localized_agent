"""Load and render the prompts in `prompts/*.md`.

Each file has YAML front matter with a `version`, bumped on every change. The
version is recorded on every EvidenceUnit and every report, so we can always
tell which prompt produced what — and because the rendered prompt is part of
the call-cache key, editing a prompt invalidates exactly its own cache entries.

HTML comments in a prompt file are notes for the people editing it. They are
stripped before the prompt is sent, so authors can explain a rule without the
explanation costing tokens or confusing the model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

PROMPT_DIR = Path(__file__).parent / "prompts"

_FRONT = re.compile(r"\A---\n(.*?)\n---\n", re.S)
_COMMENT = re.compile(r"<!--.*?-->\n?", re.S)
_SLOT = re.compile(r"\{\{(\w+)\}\}")


@dataclass(frozen=True)
class Prompt:
    name: str
    version: str
    job: str
    body: str

    @property
    def tag(self) -> str:
        return f"{self.name}@v{self.version}"

    def render(self, **values: object) -> str:
        slots = set(_SLOT.findall(self.body))
        missing = slots - set(values)
        if missing:
            raise KeyError(f"prompt {self.name!r} missing values: {sorted(missing)}")
        # One pass, so a value that happens to contain "{{x}}" — a paper can —
        # is inserted literally and never expanded.
        text = _SLOT.sub(lambda m: str(values[m.group(1)]), self.body)
        return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> Prompt:
    raw = (PROMPT_DIR / f"{name}.md").read_text(encoding="utf-8")
    meta: dict = {}
    match = _FRONT.match(raw)
    if match:
        meta = yaml.safe_load(match.group(1)) or {}
        raw = raw[match.end() :]
    body = _COMMENT.sub("", raw)
    if "version" not in meta:
        raise ValueError(f"prompt {name!r} has no version in its front matter")
    return Prompt(name=name, version=str(meta["version"]), job=str(meta.get("job", name)), body=body)

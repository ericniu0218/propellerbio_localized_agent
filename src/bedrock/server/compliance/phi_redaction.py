"""PHI/PII detection and redaction, run by the Runner before anything is
logged or handed to another agent (docs/server/ARCHITECTURE.md section 8).

This module ships a regex-based baseline (dates of birth, SSNs, MRNs,
emails, phone numbers) that catches the common structured cases without
needing a model call, plus a hook (`ner_hook`) to plug in a real clinical
NER model later — the spec calls for "detect and redact patient data"
without naming a specific tool, so this is intentionally swappable rather
than a hardcoded final answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_PATTERNS: dict[str, re.Pattern] = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "MRN": re.compile(r"\bMRN[:\s#]*\d{5,10}\b", re.IGNORECASE),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "PHONE": re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "DOB": re.compile(
        r"\b(0?[1-9]|1[0-2])[/-](0?[1-9]|[12]\d|3[01])[/-](\d{4}|\d{2})\b"
    ),
}


@dataclass
class RedactionResult:
    text: str
    redacted: bool = False
    findings: list[str] = field(default_factory=list)


NERHook = None  # type: ignore  # set to a callable(text) -> list[(span, label)] to extend


def redact_phi(text: str, ner_hook: "NERHook" = None) -> RedactionResult:
    findings: list[str] = []
    out = text
    for label, pattern in _PATTERNS.items():
        if pattern.search(out):
            findings.append(label)
            out = pattern.sub(f"[REDACTED:{label}]", out)

    if ner_hook is not None:
        for span, label in ner_hook(out):
            out = out.replace(span, f"[REDACTED:{label}]")
            findings.append(label)

    return RedactionResult(text=out, redacted=bool(findings), findings=findings)

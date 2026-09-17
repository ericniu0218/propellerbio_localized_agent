"""Bounded local extraction shared by attachments and meeting ingestion."""
import csv
import io
import json
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

TEXT_TYPES = {".txt", ".md", ".csv", ".tsv", ".json", ".fasta", ".fa", ".faa", ".fna", ".pdb", ".vtt", ".srt", ".seq"}
SENSITIVE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b|\b\d{3}-\d{2}-\d{4}\b|\b(?:patient|MRN)\s*[:#]\s*[^\n,;]+", re.I)


def extract(name, raw):
    suffix = Path(name).suffix.lower()
    if suffix in TEXT_TYPES:
        return raw.decode("utf-8-sig", errors="replace"), "parsed"
    if suffix in {".docx", ".xlsx", ".pptx"}:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            entries = z.infolist()
            if len(entries) > 5000 or sum(e.file_size for e in entries) > 40_000_000:
                raise ValueError("This Office file expands beyond the 40 MB extraction limit.")
            prefixes = {".docx": ("word/document.xml",), ".xlsx": ("xl/sharedStrings.xml", "xl/worksheets/"), ".pptx": ("ppt/slides/slide",)}
            texts = []
            for entry in entries:
                if entry.filename.endswith(".xml") and entry.filename.startswith(prefixes[suffix]):
                    root = ElementTree.fromstring(z.read(entry))
                    texts.append(" ".join(e.text for e in root.iter() if e.text and e.text.strip()))
            return "\n".join(texts), "text extracted; layout and spreadsheet cell references not preserved"
    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            return "", "PDF extraction unavailable; upload a text export"
        reader = PdfReader(io.BytesIO(raw))
        if len(reader.pages) > 200:
            raise ValueError("Please split PDFs longer than 200 pages.")
        return "\n".join(f"Page {i+1}\n{p.extract_text() or ''}" for i, p in enumerate(reader.pages)), "text extracted; scanned pages need OCR"
    return "", "Stored for download; this file type has no text parser configured"


def profile(name, text):
    suffix = Path(name).suffix.lower()
    if suffix in {".csv", ".tsv"}:
        rows = list(csv.DictReader(io.StringIO(text), delimiter="\t" if suffix == ".tsv" else ","))
        fields = list(rows[0]) if rows else []
        stats = []
        for field in fields[:25]:
            values = []
            for row in rows:
                try:
                    value = float(row.get(field, ""))
                    if value == value and abs(value) != float("inf"):
                        values.append(value)
                except (ValueError, TypeError):
                    pass
            if values:
                stats.append(f"{field}: min {min(values):g}; max {max(values):g}; mean {sum(values)/len(values):.4g}")
        return f"{len(rows)} rows · {len(fields)} columns\n" + "\n".join(stats)
    if suffix in {".fa", ".fasta", ".fna", ".faa", ".seq"}:
        records = re.findall(r">([^\n]+)\n([^>]+)", text)
        return "\n".join(f"{header}: {len(re.sub(r'\s', '', seq))} sequence characters" for header, seq in records[:30]) or "No FASTA records detected."
    if suffix == ".json":
        try:
            data = json.loads(text)
            return f"JSON {type(data).__name__} · {len(data) if isinstance(data, (list, dict)) else 1} top-level entries"
        except ValueError:
            return "Invalid JSON; review the source."
    return f"{len(text.split()):,} words · {len(text.splitlines()):,} lines" if text else "Preview unavailable"


def transcript_summary(text):
    # One deterministic extraction service for manual uploads and future connectors.
    lines = [line.strip() for line in text.splitlines() if line.strip() and not line.strip().isdigit() and "-->" not in line and line != "WEBVTT"]
    choose = lambda words: [line for line in lines if any(w in line.lower() for w in words)][:12]
    return {"summary": "Extracted transcript passages; review speaker assignments, decisions and deadlines before use.",
        "decisions": choose(["decision", "agreed", "decided"]), "actions": choose(["action", "will ", "to do"]),
        "deadlines": choose(["deadline", "friday", "monday", " by "]), "questions": choose(["?", "need to check", "unclear"]),
        "people": list(dict.fromkeys(re.findall(r"(?:^|\n)(?:\[[^]]+\]\s*)?([A-Za-z][A-Za-z ]{1,35}):", text)))[:20],
        "findings": choose(["result", "found", "binding", "assay", "control"]), "method": "rule-based extraction; human review required"}

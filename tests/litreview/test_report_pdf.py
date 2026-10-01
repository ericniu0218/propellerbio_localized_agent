"""The report as a PDF (report/pdf.py) and the slides as a PDF
(report/convert.py). Both are renderers: they must carry everything the report
says, draw every character it uses, and change nothing in it."""

from __future__ import annotations

import io
import stat

import pytest

pytest.importorskip("reportlab")
from pypdf import PdfReader  # noqa: E402

from bedrock.litreview.report import convert, pdf  # noqa: E402
from bedrock.litreview.stages.s8_render import to_markdown  # noqa: E402
from tests.litreview.test_presentation_preservation import _dense  # noqa: E402
from tests.litreview.test_synthesis import _report  # noqa: E402


def _text(data: bytes) -> str:
    return " ".join(" ".join((page.extract_text() or "").split()) for page in PdfReader(io.BytesIO(data)).pages)


def test_the_pdf_carries_the_answer_the_references_and_their_links(tmp_path):
    report = _report(tmp_path)
    data = pdf.to_pdf(report)
    assert data[:5] == b"%PDF-"
    text = _text(data)
    for section in report.sections:
        if section.id == "answer":
            for s in section.sentences:
                if s.kept:
                    assert " ".join(s.text.split()[:8]) in text, s.text
    for ref in report.references:
        assert ref.citation.split()[0] in text
    links = {a["/A"]["/URI"] for page in PdfReader(io.BytesIO(data)).pages for a in
             (x.get_object() for x in page.get("/Annots", []) or []) if "/A" in a and "/URI" in a["/A"]}
    assert {r.url for r in report.references if r.url} <= links


def test_a_dense_report_with_a_failed_integrity_check_renders_and_says_so():
    data = pdf.to_pdf(_dense())
    text = _text(data)
    assert "failed its integrity check" in text
    assert "Liver fat content fell by 12.5%" in text  # the last of many sentences: nothing cut off


def test_the_bundled_font_draws_what_reports_contain():
    """The air-gapped server has no fonts to fall back on: ≥ ± · − η² β and
    accented author names must all come from the bundled one."""
    assert pdf.missing_glyphs("≥ ≤ ± · − ‐ – — η² β α Δ μ κ “ ” … → ⚠ Fernández Dağcı Šimić") == set()


def test_a_character_the_font_lacks_is_shown_not_dropped():
    """A bioRxiv abstract printed "GLP⍰1": the PDF must not draw an empty box."""
    assert "⍰" in pdf.missing_glyphs("GLP⍰1")
    assert "GLP[?]1" in pdf.inline("GLP⍰1")


def test_links_survive_inline_markup():
    out = pdf.inline("see <https://europepmc.org/article/MED/1> and **bold** *it*")
    assert '<link href="https://europepmc.org/article/MED/1"' in out and "<b>bold</b>" in out and "<i>it</i>" in out
    assert "[?]" not in out


def test_the_pdf_is_the_same_bytes_twice_and_changes_nothing(tmp_path):
    report = _report(tmp_path)
    before = report.model_dump_json()
    assert pdf.to_pdf(report) == pdf.to_pdf(report)
    assert report.model_dump_json() == before
    assert "/Users/" not in _text(pdf.to_pdf(report)) and str(tmp_path) not in _text(pdf.to_pdf(report))


# --- slides as PDF: LibreOffice, when there is one ---------------------------------------------------------


def _fake_soffice(tmp_path, body: str):
    script = tmp_path / "soffice"
    script.write_text("#!/bin/sh\n" + body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def test_no_libreoffice_means_a_clear_refusal(monkeypatch, tmp_path):
    monkeypatch.setenv("BEDROCK_SOFFICE", str(tmp_path / "nowhere"))
    assert convert.soffice() is None
    with pytest.raises(convert.ConversionUnavailable, match="LibreOffice"):
        convert.pptx_to_pdf(b"PK")


def test_the_deck_is_converted_by_libreoffice_in_its_own_profile(monkeypatch, tmp_path):
    # Writes deck.pdf where --outdir says, as LibreOffice does; records its arguments.
    script = _fake_soffice(tmp_path, 'for a; do last=$a; done\nout=$(dirname "$last")\n'
                                     f'echo "$@" > {tmp_path}/args\nprintf "%%PDF-fake" > "$out/deck.pdf"\n')
    monkeypatch.setenv("BEDROCK_SOFFICE", str(script))
    assert convert.pptx_to_pdf(b"PK fake deck") == b"%PDF-fake"
    args = (tmp_path / "args").read_text()
    assert "--headless" in args and "--convert-to pdf" in args and "-env:UserInstallation=file://" in args


def test_a_conversion_that_writes_nothing_is_a_failure_not_an_empty_file(monkeypatch, tmp_path):
    monkeypatch.setenv("BEDROCK_SOFFICE", str(_fake_soffice(tmp_path, "echo broken >&2\nexit 1\n")))
    with pytest.raises(convert.ConversionFailed, match="broken"):
        convert.pptx_to_pdf(b"PK")

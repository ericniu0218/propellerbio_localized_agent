"""Fetching the image files of the figures a deck may show, from Europe PMC.

`report/figures.py` decides, from a paper's own JATS, which figures a
target-proposal deck may show. It names each by the `<graphic xlink:href>` the
paper uses ("ppat.1002485.g002"), and downloads nothing. This module turns
those names into image files on disk.

Europe PMC has no "give me this one figure" request. What it has is a bundle:

    GET https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/supplementaryFiles?includeInlineImage=true

which returns a zip of the paper's inline figure images *and* its supplementary
files (spreadsheets, Word files, raw data), and that can be tens of megabytes.
So the rules here are about not trusting the size or the contents of that zip:

* The download is streamed and abandoned the moment it passes `MAX_BUNDLE_BYTES`
  (50 MB). It goes to an unnamed temporary file, so an abandoned or failed
  download leaves nothing behind.
* Only the members that match a chosen figure are read out of the zip. A
  figure's href may carry no file extension (`nihms-1-f1`); then `.jpg`, `.png`
  and `.gif` are tried in that order, because the bundle often holds a small
  `.gif` thumbnail of the same figure beside the full-size `.jpg`. Each image
  is capped at `MAX_IMAGE_BYTES` (10 MB) and must start with the signature of
  a real image, since the deck's renderer would otherwise fail on it.
* Images are kept in `<litreview data dir>/figures/<PMCID>/`, and a figure
  already there is reused without a request: the same paper cited by many
  reviews is downloaded once.

A paper that cannot be fetched (an HTTP error, a bad zip, too big, too slow)
costs its own figures and nothing else: it is logged as a warning and the next
paper goes on. A deck without a picture is a normal deck; a review must never
fail because a picture could not be had.

Choosing a figure needs the paper's JATS, and a stored run does not have it:
ingestion splits the article into sections and then discards the raw text
(`ingest.prepare`). So `ArticleTexts` re-reads it from Europe PMC
(`.../{PMCID}/fullTextXML`) for the papers the report cites, one paper at a
time and only when `choose_figures` reaches it, and keeps each in
`<data dir>/figures/<PMCID>/fulltext.xml` so a later run asks for nothing. That
text is used only to find which figure a paper cites beside a quote the review
already verified. The article may have been corrected upstream since the review
read it; then the quote is simply not found in the new text and no figure is
shown. It never changes what the report says.

These are the only requests a figure ever causes, and they go through the same
allowlisted client as the search (`sources/http.py`), so they can reach
`www.ebi.ac.uk` and nothing else, redirects included. Each carries a PubMed
Central id and no part of the scientist's question.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import threading
import time
import zipfile
from collections.abc import Iterator, Mapping
from dataclasses import replace
from pathlib import Path

from bedrock.litreview.models import Paper
from bedrock.litreview.report.deck_model import FigureAsset
from bedrock.litreview.sources.europepmc import EuropePMC
from bedrock.litreview.sources.http import SourceError, SourceHTTP

log = logging.getLogger(__name__)

BUNDLE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles?includeInlineImage=true"
MAX_BUNDLE_BYTES = 50 * 1024 * 1024
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_SECONDS = 300.0
"""A bundle that drips in slowly is abandoned after this long, whatever its size."""

MAX_TEXTS = 20
"""Most articles re-read for one deck. Each is a whole paper (up to a few
megabytes) fetched just to look for a figure, so a review that cites hundreds
of papers must not turn into hundreds of downloads."""

IMAGE_EXTENSIONS = (".jpg", ".png", ".gif")
"""What is tried, best first, when a figure's href has no file extension."""
_KNOWN_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff"})
_PMCID = re.compile(r"PMC\d+")
_CHUNK = 64 * 1024


class _TooBig(SourceError):
    pass


# --- which file in the bundle is a figure -------------------------------------


def candidate_names(href: str) -> list[str]:
    """The file names (lower case, best first) that can hold the figure `href`.

    An href that already ends in an image extension names exactly one file.
    Otherwise (`nihms-1-f1`, or `pone.0001-g002`, whose dot is part of the
    name and not an extension) the file is that name plus `.jpg`, `.png` or
    `.gif`, in that order of preference."""
    base = os.path.basename((href or "").replace("\\", "/")).strip().lower()
    if not base:
        return []
    if os.path.splitext(base)[1] in _KNOWN_EXTENSIONS:
        return [base]
    return [base + ext for ext in IMAGE_EXTENSIONS]


def _safe_name(name: str) -> str:
    """A file name that cannot leave its directory or start with a dot."""
    name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(name))
    return name if name and not name.startswith(".") else "_" + name


def _image_signature(head: bytes) -> bool:
    return head.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n", b"GIF87a", b"GIF89a", b"II*\x00", b"MM\x00*"))


def _cached(directory: Path, href: str) -> Path | None:
    """The file already on disk for `href`, preferring the same extension
    order as a fresh download would."""
    if not directory.is_dir():
        return None
    on_disk = {p.name.lower(): p for p in directory.iterdir() if p.is_file() and not p.name.endswith(".part")}
    for name in candidate_names(href):
        if name in on_disk:
            return on_disk[name]
    return None


# --- one paper ----------------------------------------------------------------


def _download_bundle(http: SourceHTTP, pmcid: str, spool) -> None:
    """Stream the bundle into `spool`, giving up past the size or time limit."""
    deadline = time.monotonic() + MAX_SECONDS
    with http.stream("GET", BUNDLE_URL.format(pmcid=pmcid)) as resp:
        if resp.status_code != 200:
            raise SourceError(f"HTTP {resp.status_code}")
        declared = resp.headers.get("Content-Length", "")
        if declared.isdigit() and int(declared) > MAX_BUNDLE_BYTES:
            raise _TooBig(f"the file bundle is {int(declared)} bytes, over the {MAX_BUNDLE_BYTES} limit")
        total = 0
        # No chunk size: pieces are handed over as they arrive, so the limit is
        # noticed within one network read rather than after a buffer fills.
        for chunk in resp.iter_bytes():
            total += len(chunk)
            if total > MAX_BUNDLE_BYTES:
                raise _TooBig(f"the file bundle is over the {MAX_BUNDLE_BYTES} byte limit")
            if time.monotonic() > deadline:
                raise SourceError(f"the file bundle took over {MAX_SECONDS:.0f} seconds")
            spool.write(chunk)


def _extract(zf: zipfile.ZipFile, info: zipfile.ZipInfo, directory: Path) -> Path | None:
    """Copy one image out of the zip, or None when it is too big or not an
    image. Written under a temporary name and renamed, so a half-written file
    is never mistaken for a cached image."""
    if info.file_size > MAX_IMAGE_BYTES:
        log.warning("figures: %s is %d bytes, over the %d limit; not used", info.filename, info.file_size, MAX_IMAGE_BYTES)
        return None
    target = directory / _safe_name(info.filename)
    part = target.with_name(target.name + ".part")
    try:
        written = 0
        with zf.open(info) as src, open(part, "wb") as out:
            head = src.read(16)
            if not _image_signature(head):
                log.warning("figures: %s is not an image file; not used", info.filename)
                return None
            chunk = head
            while chunk:
                written += len(chunk)
                if written > MAX_IMAGE_BYTES:  # the zip's own size claim was wrong
                    log.warning("figures: %s is over the %d byte limit; not used", info.filename, MAX_IMAGE_BYTES)
                    return None
                out.write(chunk)
                chunk = src.read(_CHUNK)
        part.chmod(0o600)
        os.replace(part, target)
        return target
    finally:
        part.unlink(missing_ok=True)


def _fetch_paper(http: SourceHTTP, pmcid: str, wanted: dict[int, str], directory: Path) -> dict[int, Path]:
    """Images for the figures `wanted` (index -> href) of one paper: what is
    cached, else one request for the bundle. Raises on anything that stops the
    paper as a whole; a figure that is merely missing from the bundle is just
    absent from the result."""
    found: dict[int, Path] = {}
    missing: dict[int, str] = {}
    for index, href in wanted.items():
        path = _cached(directory, href)
        if path is not None:
            found[index] = path
        else:
            missing[index] = href
    if not missing:
        return found

    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    # An unnamed temporary file: deleted when closed, so an abandoned download
    # leaves nothing on disk. Kept on disk rather than in memory because the
    # bundle can be 50 MB and zipfile needs to seek in it.
    with tempfile.TemporaryFile() as spool:
        _download_bundle(http, pmcid, spool)
        spool.seek(0)
        with zipfile.ZipFile(spool) as zf:
            members = {}
            for info in zf.infolist():
                if not info.is_dir():
                    members.setdefault(os.path.basename(info.filename.replace("\\", "/")).lower(), info)
            for index, href in missing.items():
                for name in candidate_names(href):
                    info = members.get(name)
                    if info is None:
                        continue
                    try:
                        path = _extract(zf, info, directory)
                    except Exception as exc:  # noqa: BLE001 - this figure only
                        log.warning("figures: could not read %s from the %s bundle: %s", info.filename, pmcid, exc)
                        path = None
                    # The best match decides: when the full-size image is
                    # unusable, its thumbnail is not offered in its place.
                    if path is not None:
                        found[index] = path
                    break
                else:
                    log.warning("figures: %s has no image for %r in its file bundle", pmcid, href)
    return found


# --- all papers ---------------------------------------------------------------


def download_figures(
    figures: list[FigureAsset],
    cache_dir: Path,
    *,
    http: SourceHTTP | None = None,
    cancel: threading.Event | None = None,
) -> list[FigureAsset]:
    """The same figures, with `path` set on those whose image is now on disk.

    One request per paper, however many of its figures were chosen, and none
    for a paper whose images are all cached already. `cancel` is looked at
    between papers: a stop request ends the downloads there and returns what
    was fetched so far. Never raises for a paper's sake: any failure is logged
    as a warning and that paper's figures keep `path=None`.

    `http` is for tests; otherwise a client is opened (and closed) here.
    """
    if not figures:
        return []
    own = http is None
    if own:
        http = SourceHTTP(min_interval=0.1, timeout=60.0)
    by_paper: dict[str, dict[int, str]] = {}
    for index, figure in enumerate(figures):
        by_paper.setdefault(figure.pmcid, {})[index] = figure.href
    paths: dict[int, Path] = {}
    try:
        for pmcid, wanted in by_paper.items():
            if cancel is not None and cancel.is_set():
                break
            if not _PMCID.fullmatch(pmcid or ""):
                log.warning("figures: %r is not a PubMed Central id; its figures are skipped", pmcid)
                continue
            try:
                paths.update(_fetch_paper(http, pmcid, wanted, cache_dir / pmcid))
            except Exception as exc:  # noqa: BLE001 - one paper never stops the others
                log.warning("figures: could not fetch the figures of %s: %s", pmcid, exc)
    finally:
        if own:
            http.close()
    return [replace(f, path=str(paths[i])) if i in paths else f for i, f in enumerate(figures)]


# --- the articles' text ---------------------------------------------------------


class ArticleTexts(Mapping[str, Paper]):
    """The run's papers, each with its JATS full text filled in on first use.

    Hand this to `report.figures.choose_figures` in place of a plain dict. A
    paper is looked up only when `choose_figures` reaches it (in the order the
    report first cites it), and gets a copy with `full_text` and
    `full_text_format="jats"` set; the stored paper is never changed. Only a
    paper with a PMC id whose text came from Europe PMC's XML is re-read. The
    text comes from `<cache_dir>/<PMCID>/fulltext.xml` when it is there (no
    request), else from one request, and is then cached. After `MAX_TEXTS`
    requests, or once `cancel` is set, papers come back as they are, without
    text, and so contribute no figure. A paper that cannot be read is
    logged as a warning and does the same.
    """

    def __init__(
        self,
        papers: Mapping[str, Paper],
        cache_dir: Path,
        *,
        http: SourceHTTP,
        cancel: threading.Event | None = None,
        max_texts: int | None = None,
    ) -> None:
        self._papers = papers
        self._cache_dir = cache_dir
        self._http = http
        self._cancel = cancel
        self._max = MAX_TEXTS if max_texts is None else max_texts
        self._ready: dict[str, Paper] = {}
        self.requests = 0
        """Articles asked for from Europe PMC so far (cache hits do not count)."""

    def __getitem__(self, paper_id: str) -> Paper:
        if paper_id not in self._ready:
            paper = self._papers[paper_id]
            self._ready[paper_id] = self._with_text(paper)
        return self._ready[paper_id]

    def __contains__(self, paper_id: object) -> bool:
        return paper_id in self._papers  # asking is not a reason to read an article

    def __iter__(self) -> Iterator[str]:
        return iter(self._papers)

    def __len__(self) -> int:
        return len(self._papers)

    def _with_text(self, paper: Paper) -> Paper:
        if paper.full_text and paper.full_text_format == "jats":
            return paper  # a caller that still has the text needs no request
        if not paper.pmcid or paper.text_source != "europepmc_xml":
            return paper
        if not _PMCID.fullmatch(paper.pmcid):
            log.warning("figures: %r is not a PubMed Central id; its article is not re-read", paper.pmcid)
            return paper
        try:
            text = self._text(paper.pmcid)
        except Exception as exc:  # noqa: BLE001 - this paper only
            log.warning("figures: could not read the article text of %s: %s", paper.pmcid, exc)
            return paper
        if not text:
            return paper
        return paper.model_copy(update={"full_text": text, "full_text_format": "jats"})

    def _text(self, pmcid: str) -> str | None:
        path = self._cache_dir / pmcid / "fulltext.xml"
        if path.is_file():
            return path.read_text(encoding="utf-8")
        if self.requests >= self._max:
            log.warning("figures: %d articles already re-read; %s is not", self._max, pmcid)
            return None
        if self._cancel is not None and self._cancel.is_set():
            return None
        self.requests += 1
        text = EuropePMC(http=self._http).full_text(pmcid)
        if text:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            part = path.with_name(path.name + ".part")
            try:
                part.write_text(text, encoding="utf-8")
                part.chmod(0o600)
                os.replace(part, path)
            finally:
                part.unlink(missing_ok=True)
        return text

"""Downloading the images of the figures a deck may show (sources/figures.py).

Europe PMC answers a figure request with a zip of the paper's inline images
*and* its supplementary files, which can be huge. These tests pin down what
this module does with that zip: read only the images the deck chose, prefer the
full-size picture to its thumbnail, give up on an oversized download without
leaving a file behind, keep going when one paper fails, and never ask twice
for what is already on disk. Nothing here touches the network: every request is
answered by an `httpx.MockTransport` with a zip built in memory.
"""

from __future__ import annotations

import io
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

import httpx

from bedrock.litreview import service
from bedrock.litreview.models import Paper
from bedrock.litreview.report.deck_model import FigureAsset
from bedrock.litreview.report.figures import MAX_PAPERS, choose_figures
from bedrock.litreview.service import ReviewRequest
from bedrock.litreview.sources import figures as fetch
from bedrock.litreview.sources.http import SourceHTTP
from tests.litreview.test_deck_figures import PARA_F1, PARA_NONE, QUOTE_F1, jats, lic, make_report

JPG = b"\xff\xd8\xff\xe0" + b"the full-size picture"
PNG = b"\x89PNG\r\n\x1a\n" + b"a png picture"
GIF = b"GIF89a" + b"a thumbnail"


def bundle(members: dict[str, bytes]) -> bytes:
    """A zip like Europe PMC's: images and supplementary files side by side."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return out.getvalue()


def figure(href: str, *, pmcid: str = "PMC100", fig_id: str = "F2") -> FigureAsset:
    return FigureAsset(
        paper_id=pmcid, pmcid=pmcid, fig_id=fig_id, label="Figure 2", caption="Food intake",
        href=href, licence="CC BY 4.0", evidence_ids=["E1"],
    )


class Server:
    """Plays Europe PMC: `answers` maps a PMCID to a zip (bytes), a status
    code (int) or a ready `httpx.Response`. Every request is recorded."""

    def __init__(self, answers: dict[str, object]) -> None:
        self.answers = answers
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        pmcid = request.url.path.split("/")[-2]
        answer = self.answers[pmcid]
        if isinstance(answer, httpx.Response):
            return answer
        if isinstance(answer, int):
            return httpx.Response(answer)
        return httpx.Response(200, content=answer)

    def http(self) -> SourceHTTP:
        return SourceHTTP(transport=httpx.MockTransport(self), sleep=lambda s: None)


def names(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir()) if directory.is_dir() else []


# --- which file is a figure ---------------------------------------------------


def test_an_href_with_an_image_extension_names_exactly_that_file():
    assert fetch.candidate_names("ppat.1002485.g002.JPG") == ["ppat.1002485.g002.jpg"]
    assert fetch.candidate_names("g1.gif") == ["g1.gif"]


def test_an_href_without_an_extension_tries_jpg_then_png_then_gif():
    # "pone.0001-g002" has a dot, but it is part of the name, not an extension.
    assert fetch.candidate_names("pone.0001-g002") == ["pone.0001-g002.jpg", "pone.0001-g002.png", "pone.0001-g002.gif"]
    assert fetch.candidate_names("nihms-1-f1")[0] == "nihms-1-f1.jpg"
    assert fetch.candidate_names("") == []


# --- one paper ----------------------------------------------------------------


def test_only_the_chosen_images_are_extracted(tmp_path):
    server = Server({"PMC100": bundle({
        "ppat.1002485.g002.jpg": JPG,
        "ppat.1002485.g002.gif": GIF,         # thumbnail of the same figure
        "ppat.1002485.g003.jpg": JPG + b"3",  # a figure nobody chose
        "ppat.1002485.s003.tif": b"II*\x00data",
        "supplement.docx": b"PK-not-for-the-deck",
    })})

    [got] = fetch.download_figures([figure("ppat.1002485.g002.jpg")], tmp_path, http=server.http())

    assert Path(got.path) == tmp_path / "PMC100" / "ppat.1002485.g002.jpg"
    assert Path(got.path).read_bytes() == JPG
    assert names(tmp_path / "PMC100") == ["ppat.1002485.g002.jpg"]  # nothing else was kept


def test_a_href_without_an_extension_prefers_the_jpg_to_the_gif_thumbnail(tmp_path):
    server = Server({"PMC100": bundle({"f1.gif": GIF, "f1.png": PNG, "f1.jpg": JPG})})

    [got] = fetch.download_figures([figure("f1")], tmp_path, http=server.http())

    assert Path(got.path).name == "f1.jpg"
    assert names(tmp_path / "PMC100") == ["f1.jpg"]


def test_the_png_is_preferred_to_the_gif_and_a_lone_gif_is_still_used(tmp_path):
    server = Server({
        "PMC100": bundle({"f1.gif": GIF, "f1.png": PNG}),
        "PMC200": bundle({"g1.gif": GIF}),
    })

    first, second = fetch.download_figures(
        [figure("f1"), figure("g1", pmcid="PMC200")], tmp_path, http=server.http()
    )

    assert Path(first.path).name == "f1.png"
    assert Path(second.path).name == "g1.gif"


def test_the_request_goes_to_europe_pmc_for_the_papers_bundle_and_nothing_else(tmp_path):
    server = Server({"PMC100": bundle({"f1.jpg": JPG})})

    fetch.download_figures([figure("f1")], tmp_path, http=server.http())

    [request] = server.requests
    assert request.url.host == "www.ebi.ac.uk"
    assert request.url.scheme == "https"
    assert request.url.path == "/europepmc/webservices/rest/PMC100/supplementaryFiles"
    assert dict(request.url.params) == {"includeInlineImage": "true"}
    assert request.method == "GET"


def test_two_figures_of_one_paper_cost_one_request(tmp_path):
    server = Server({"PMC100": bundle({"f1.jpg": JPG, "f2.jpg": JPG + b"2"})})

    got = fetch.download_figures(
        [figure("f1", fig_id="F1"), figure("f2", fig_id="F2")], tmp_path, http=server.http()
    )

    assert len(server.requests) == 1
    assert [Path(f.path).name for f in got] == ["f1.jpg", "f2.jpg"]


def test_a_figure_missing_from_the_bundle_stays_without_a_path_and_the_others_are_kept(tmp_path):
    server = Server({"PMC100": bundle({"f1.jpg": JPG})})

    first, second = fetch.download_figures(
        [figure("f1", fig_id="F1"), figure("nope", fig_id="F9")], tmp_path, http=server.http()
    )

    assert first.path is not None and second.path is None


def test_the_figures_that_come_back_keep_everything_but_the_path(tmp_path):
    server = Server({"PMC100": bundle({"f1.jpg": JPG})})
    chosen = figure("f1")

    [got] = fetch.download_figures([chosen], tmp_path, http=server.http())

    assert chosen.path is None  # the input is not modified
    assert {**got.to_dict(), "path": None} == chosen.to_dict()


# --- limits -------------------------------------------------------------------


def test_a_download_past_the_size_limit_is_abandoned_and_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "MAX_BUNDLE_BYTES", 1000)
    sent = []

    def endless():
        for _ in range(500):
            sent.append(1)
            yield b"x" * 100

    server = Server({"PMC100": httpx.Response(200, content=endless())})

    [got] = fetch.download_figures([figure("f1")], tmp_path, http=server.http())

    assert got.path is None
    assert len(sent) < 500  # stopped reading as soon as the limit was passed
    assert names(tmp_path / "PMC100") == []  # no image, no half-written file


def test_a_bundle_that_says_it_is_too_big_is_not_read_at_all(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "MAX_BUNDLE_BYTES", 1000)
    read = []

    def body():
        read.append(1)
        yield b"x"

    server = Server({"PMC100": httpx.Response(200, headers={"Content-Length": "5000"}, content=body())})

    [got] = fetch.download_figures([figure("f1")], tmp_path, http=server.http())

    assert got.path is None
    assert read == []


def test_an_image_over_the_image_limit_is_not_extracted_and_its_thumbnail_is_not_offered(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "MAX_IMAGE_BYTES", 50)
    server = Server({"PMC100": bundle({"f1.jpg": JPG + b"x" * 100, "f1.gif": GIF, "f2.jpg": JPG})})

    big, small = fetch.download_figures(
        [figure("f1", fig_id="F1"), figure("f2", fig_id="F2")], tmp_path, http=server.http()
    )

    assert big.path is None
    assert Path(small.path).name == "f2.jpg"
    assert names(tmp_path / "PMC100") == ["f2.jpg"]


def test_a_file_that_is_not_an_image_is_refused(tmp_path):
    server = Server({"PMC100": bundle({"f1.jpg": b"<html>not a picture</html>"})})

    [got] = fetch.download_figures([figure("f1")], tmp_path, http=server.http())

    assert got.path is None
    assert names(tmp_path / "PMC100") == []


# --- failures stay with their paper -------------------------------------------


def test_one_failing_paper_does_not_stop_the_next(tmp_path):
    server = Server({
        "PMC100": 500,
        "PMC200": b"this is not a zip file",
        "PMC300": 404,
        "PMC400": bundle({"f1.jpg": JPG}),
    })

    got = fetch.download_figures(
        [figure("f1", pmcid=p) for p in ("PMC100", "PMC200", "PMC300", "PMC400")], tmp_path, http=server.http()
    )

    assert [f.path is None for f in got] == [True, True, True, False]
    assert len(server.requests) == 4


def test_a_dropped_connection_is_a_failed_paper_not_an_exception(tmp_path):
    def refuse(request):
        raise httpx.ConnectError("no route")

    http = SourceHTTP(transport=httpx.MockTransport(refuse), sleep=lambda s: None)

    [got] = fetch.download_figures([figure("f1")], tmp_path, http=http)

    assert got.path is None


def test_a_redirect_off_europe_pmc_is_refused_not_followed(tmp_path):
    server = Server({"PMC100": httpx.Response(302, headers={"Location": "https://elsewhere.example/f.zip"})})

    [got] = fetch.download_figures([figure("f1")], tmp_path, http=server.http())

    assert got.path is None
    assert [r.url.host for r in server.requests] == ["www.ebi.ac.uk"]  # the second hop was never sent


def test_an_id_that_is_not_a_pubmed_central_id_is_skipped_without_a_request(tmp_path):
    server = Server({})

    [got] = fetch.download_figures([figure("f1", pmcid="../../etc")], tmp_path, http=server.http())

    assert got.path is None
    assert server.requests == []


def test_a_stop_request_ends_the_downloads_between_papers(tmp_path):
    cancel = threading.Event()
    server = Server({"PMC100": bundle({"f1.jpg": JPG}), "PMC200": bundle({"f1.jpg": JPG})})
    original = server.__call__

    def stop_after_first(request):
        cancel.set()
        return original(request)

    http = SourceHTTP(transport=httpx.MockTransport(stop_after_first), sleep=lambda s: None)

    first, second = fetch.download_figures(
        [figure("f1", pmcid="PMC100"), figure("f1", pmcid="PMC200")], tmp_path, http=http, cancel=cancel
    )

    assert first.path is not None and second.path is None


# --- the cache ----------------------------------------------------------------


def test_a_cached_figure_makes_no_request(tmp_path):
    server = Server({"PMC100": bundle({"f1.jpg": JPG})})
    [first] = fetch.download_figures([figure("f1")], tmp_path, http=server.http())
    assert len(server.requests) == 1

    [again] = fetch.download_figures([figure("f1")], tmp_path, http=server.http())

    assert len(server.requests) == 1
    assert again.path == first.path


def test_only_what_is_missing_is_extracted_when_part_of_a_paper_is_cached(tmp_path):
    server = Server({"PMC100": bundle({"f1.jpg": JPG, "f2.jpg": JPG + b"2"})})
    fetch.download_figures([figure("f1", fig_id="F1")], tmp_path, http=server.http())
    (tmp_path / "PMC100" / "f1.jpg").write_bytes(JPG + b"edited")  # proves f1 is reused, not rewritten

    got = fetch.download_figures(
        [figure("f1", fig_id="F1"), figure("f2", fig_id="F2")], tmp_path, http=server.http()
    )

    assert len(server.requests) == 2
    assert Path(got[0].path).read_bytes() == JPG + b"edited"
    assert Path(got[1].path).read_bytes() == JPG + b"2"


def test_nothing_chosen_means_no_client_and_no_request(tmp_path, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("no client may be opened when there is nothing to fetch")

    monkeypatch.setattr(fetch, "SourceHTTP", forbidden)

    assert fetch.download_figures([], tmp_path) == []


# --- the article text, re-read to find the figures ----------------------------
#
# Ingestion throws a paper's raw JATS away once it has split it into sections,
# so the papers of a stored run have `full_text=None` and only
# `text_source="europepmc_xml"` to say where their text came from. These tests
# use papers exactly as ingestion leaves them.

ARTICLE = jats(PARA_F1 + PARA_NONE, license_xml=lic())
NO_FIGURE_ARTICLE = jats(PARA_NONE, license_xml=lic())


def stored(pid: str = "P1", pmcid: str | None = "PMC100", source: str = "europepmc_xml") -> Paper:
    """A paper as the database keeps it: text split into sections, raw text gone."""
    return Paper(id=pid, title=f"Paper {pid}", pmcid=pmcid, text_source=source, full_text=None)


class EuropePMC:
    """Plays both Europe PMC requests a deck makes: the article and the bundle."""

    def __init__(self, articles: dict[str, object], bundles: dict[str, bytes]) -> None:
        self.articles, self.bundles = articles, bundles
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        *_, pmcid, kind = request.url.path.split("/")
        table = self.articles if kind == "fullTextXML" else self.bundles
        answer = table.get(pmcid, 404)
        return httpx.Response(answer) if isinstance(answer, int) else httpx.Response(200, content=answer)

    def kinds(self) -> list[str]:
        return [r.url.path.split("/")[-1] for r in self.requests]

    def http(self) -> SourceHTTP:
        return SourceHTTP(transport=httpx.MockTransport(self), sleep=lambda s: None)


def run_manifest(tmp_path, monkeypatch, server, papers, report, *, search=True):
    """`service._write_figure_manifest` for a finished run, with Europe PMC played by `server`."""
    monkeypatch.setattr(service, "SourceHTTP", lambda **kw: server.http())
    settings = SimpleNamespace(data_dir=tmp_path / "data", db_path=tmp_path / "data" / "db.sqlite3")
    result = SimpleNamespace(report=report, papers=papers, run=SimpleNamespace(id="run1"))
    service._write_figure_manifest(
        ReviewRequest(question="q", deck=True, search=search), settings, result, lambda m: None, threading.Event()
    )
    return settings


def test_papers_as_ingestion_leaves_them_get_a_figure_with_a_path_end_to_end(tmp_path, monkeypatch):
    server = EuropePMC({"PMC100": ARTICLE}, {"PMC100": bundle({"nihms-1-f1.jpg": JPG, "nihms-1-f1.gif": GIF})})
    assert stored().full_text is None  # the premise: nothing to read the JATS from

    settings = run_manifest(tmp_path, monkeypatch, server, [stored()], make_report([("u1", "P1", QUOTE_F1, None)]))

    [fig] = service.deck_figures(settings, "run1")
    assert (fig.paper_id, fig.fig_id, fig.href, fig.licence) == ("P1", "F1", "nihms-1-f1", "CC BY 4.0")
    assert Path(fig.path) == tmp_path / "data" / "figures" / "PMC100" / "nihms-1-f1.jpg"
    assert Path(fig.path).read_bytes() == JPG
    assert server.kinds() == ["fullTextXML", "supplementaryFiles"]
    assert {r.url.host for r in server.requests} == {"www.ebi.ac.uk"}
    assert (tmp_path / "data" / "figures" / "PMC100" / "fulltext.xml").read_text() == ARTICLE


def test_a_cached_article_and_image_make_a_second_run_ask_for_nothing(tmp_path, monkeypatch):
    server = EuropePMC({"PMC100": ARTICLE}, {"PMC100": bundle({"nihms-1-f1.jpg": JPG})})
    report = make_report([("u1", "P1", QUOTE_F1, None)])
    run_manifest(tmp_path, monkeypatch, server, [stored()], report)
    assert len(server.requests) == 2

    settings = run_manifest(tmp_path, monkeypatch, server, [stored()], report)

    assert len(server.requests) == 2
    assert [f.path is not None for f in service.deck_figures(settings, "run1")] == [True]


def test_without_a_search_nothing_is_requested_and_a_stored_paper_has_no_figure(tmp_path, monkeypatch):
    def forbidden(**kw):
        raise AssertionError("no client may be opened without a search")

    monkeypatch.setattr(service, "SourceHTTP", forbidden)
    settings = SimpleNamespace(data_dir=tmp_path / "data", db_path=tmp_path / "data" / "db.sqlite3")
    result = SimpleNamespace(
        report=make_report([("u1", "P1", QUOTE_F1, None)]), papers=[stored()], run=SimpleNamespace(id="run1")
    )

    service._write_figure_manifest(
        ReviewRequest(question="q", deck=True), settings, result, lambda m: None, threading.Event()
    )

    assert service.deck_figures(settings, "run1") == []
    assert not (tmp_path / "data" / "figures").exists()


def test_an_article_changed_upstream_gives_no_figure_because_the_quote_is_not_found(tmp_path, monkeypatch):
    server = EuropePMC({"PMC100": NO_FIGURE_ARTICLE}, {"PMC100": bundle({"nihms-1-f1.jpg": JPG})})

    settings = run_manifest(tmp_path, monkeypatch, server, [stored()], make_report([("u1", "P1", QUOTE_F1, None)]))

    assert service.deck_figures(settings, "run1") == []
    assert server.kinds() == ["fullTextXML"]  # no figure chosen, so no image bundle either


def test_only_papers_with_a_pmcid_and_europe_pmc_text_are_re_read(tmp_path):
    server = EuropePMC({"PMC100": ARTICLE, "PMC200": ARTICLE}, {})
    papers = {
        "P1": stored("P1", pmcid=None),
        "P2": stored("P2", pmcid="PMC200", source="grobid_pdf"),
        "P3": stored("P3", pmcid="PMC300"),  # Europe PMC has no article for it
    }
    texts = fetch.ArticleTexts(papers, tmp_path, http=server.http())

    assert [texts[p].full_text for p in ("P1", "P2", "P3")] == [None, None, None]
    assert server.kinds() == ["fullTextXML"]  # only P3 was asked about
    assert papers["P3"].full_text is None  # the stored paper is never changed


def test_a_paper_that_cannot_be_read_costs_only_itself(tmp_path):
    server = EuropePMC({"PMC200": ARTICLE}, {})  # PMC100 answers 404
    texts = fetch.ArticleTexts({"P1": stored("P1", "PMC100"), "P2": stored("P2", "PMC200")}, tmp_path, http=server.http())

    assert texts["P1"].full_text is None
    assert texts["P2"].full_text == ARTICLE and texts["P2"].full_text_format == "jats"


def test_looking_a_paper_up_by_membership_reads_nothing(tmp_path):
    server = EuropePMC({"PMC100": ARTICLE}, {})
    texts = fetch.ArticleTexts({"P1": stored()}, tmp_path, http=server.http())

    assert "P1" in texts and "P9" not in texts and len(texts) == 1 and list(texts) == ["P1"]
    assert server.requests == []


def test_no_more_than_twenty_articles_are_re_read_for_one_deck(tmp_path, monkeypatch):
    assert fetch.MAX_TEXTS == 20
    papers = [stored(f"P{n}", f"PMC{n}") for n in range(1, 26)]
    # None of the first twenty cites a figure beside the quote, so every one
    # is read and the search goes on; the figure is only in the 21st onward.
    articles = {f"PMC{n}": NO_FIGURE_ARTICLE if n <= 20 else ARTICLE for n in range(1, 26)}
    server = EuropePMC(articles, {f"PMC{n}": bundle({"nihms-1-f1.jpg": JPG}) for n in range(1, 26)})
    report = make_report([(f"u{n}", f"P{n}", QUOTE_F1, None) for n in range(1, 26)])

    settings = run_manifest(tmp_path, monkeypatch, server, papers, report)

    assert server.kinds().count("fullTextXML") == 20
    assert service.deck_figures(settings, "run1") == []


def test_choose_figures_stops_asking_for_papers_once_its_caps_are_full():
    asked: list[str] = []

    class Recording(dict):
        def get(self, key, default=None):
            asked.append(key)
            return super().get(key, default)

    papers = Recording({f"P{n}": Paper(id=f"P{n}", title="t", pmcid=f"PMC{n}", full_text_format="jats",
                                       full_text=ARTICLE) for n in range(1, 13)})
    report = make_report([(f"u{n}", f"P{n}", QUOTE_F1, None) for n in range(1, 13)])

    got = choose_figures(report, papers)

    assert len(got) == MAX_PAPERS == 8
    assert asked == [f"P{n}" for n in range(1, 9)]  # P9 to P12 could not add a figure, so were never asked for

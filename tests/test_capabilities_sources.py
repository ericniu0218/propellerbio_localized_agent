"""sources.fetch and sources.search: the connectors, reused rather than rebuilt.

These tests deliberately use a real ``ShareConnector`` over a temporary folder
instead of a stand-in. The point of this adapter is that it adds nothing to the
connector layer, and a fake connector would let it drift without anyone
noticing.
"""

from pathlib import Path

import pytest

from bedrock.agents.capabilities.sources import (
    CAPABILITIES,
    SourcesFetch,
    SourcesSearch,
)
from bedrock.agents.contract import Risk
from bedrock.connectors.hub import SourceHub
from bedrock.connectors.share import ShareConnector


@pytest.fixture
def hub(tmp_path: Path) -> SourceHub:
    share = tmp_path / "lims-exports"
    (share / "plates").mkdir(parents=True)
    (share / "plates" / "P42-results.csv").write_text(
        "well,od600\nA1,0.42\n", encoding="utf-8"
    )
    (share / "plates" / "P42-scan.bin").write_bytes(b"\x00\x01\x02binary")
    return SourceHub(
        {"lims": ShareConnector("lims", root=str(share))},
        user="alice",
        staging_dir=tmp_path / "staging",
    )


# ---- What the roster finds ---------------------------------------------------


def test_both_source_capabilities_are_read_only():
    assert {c.name: c.risk for c in CAPABILITIES} == {
        "sources.fetch": Risk.READ_ONLY,
        "sources.search": Risk.READ_ONLY,
    }


# ---- sources.fetch -----------------------------------------------------------


async def test_a_fetch_stages_the_record_and_keeps_its_provenance(hub: SourceHub):
    outcome = await SourcesFetch(hub).run({"ref": "lims://plates/P42-results.csv"})

    assert outcome.failed is None
    staged = Path(outcome.artifacts[0])
    assert staged.read_text(encoding="utf-8") == "well,od600\nA1,0.42\n"

    sidecar = Path(outcome.artifacts[1])
    assert sidecar.name.endswith(".provenance.json")
    assert "lims://plates/P42-results.csv" in outcome.completed[0]
    assert "sha256" in outcome.completed[0]
    assert "fetched by alice" in outcome.completed[1]


async def test_a_fetched_text_record_is_shown_but_a_binary_one_is_not(hub: SourceHub):
    text = await SourcesFetch(hub).run({"ref": "lims://plates/P42-results.csv"})
    assert "A1,0.42" in text.completed[-1]

    binary = await SourcesFetch(hub).run({"ref": "lims://plates/P42-scan.bin"})
    assert binary.failed is None
    assert not any("binary" in line for line in binary.completed[3:])


async def test_a_web_address_is_not_a_source_reference(hub: SourceHub):
    outcome = await SourcesFetch(hub).run({"ref": "https://example.com/seq.fasta"})
    assert "is not a source reference" in (outcome.failed or "")


async def test_a_plain_file_path_is_not_a_source_reference(hub: SourceHub):
    outcome = await SourcesFetch(hub).run({"ref": "/tmp/candidate.fasta"})
    assert "is not a source reference" in (outcome.failed or "")


async def test_fetch_without_a_reference_says_what_one_looks_like(hub: SourceHub):
    outcome = await SourcesFetch(hub).run({})
    assert "registry://SEQ-00123" in (outcome.failed or "")


async def test_an_unknown_source_is_reported_in_the_connector_layer_s_own_words(
    hub: SourceHub,
):
    outcome = await SourcesFetch(hub).run({"ref": "registry://SEQ-1"})
    assert "No source called `registry`" in (outcome.failed or "")
    assert "Configured: `lims`" in (outcome.failed or "")


async def test_a_reference_that_climbs_out_of_the_share_is_refused(hub: SourceHub):
    outcome = await SourcesFetch(hub).run({"ref": "lims://../../etc/passwd"})
    assert "points outside" in (outcome.failed or "")
    assert outcome.artifacts == ()


async def test_a_missing_record_is_reported_without_an_exception(hub: SourceHub):
    outcome = await SourcesFetch(hub).run({"ref": "lims://plates/P99.csv"})
    assert "No record" in (outcome.failed or "")


# ---- sources.search ----------------------------------------------------------


async def test_search_lists_references_that_can_be_fetched_next(hub: SourceHub):
    outcome = await SourcesSearch(hub).run({"source": "lims", "query": "P42 results"})
    assert outcome.failed is None
    assert any("lims://plates/P42-results.csv" in line for line in outcome.completed)


async def test_search_needs_both_a_source_and_a_query(hub: SourceHub):
    assert "needs a `source`" in (await SourcesSearch(hub).run({"query": "x"})).failed
    assert "needs a `query`" in (await SourcesSearch(hub).run({"source": "lims"})).failed


async def test_search_says_when_a_source_has_nothing(hub: SourceHub):
    outcome = await SourcesSearch(hub).run({"source": "lims", "query": "unobtainium"})
    assert outcome.failed is None
    assert "found nothing" in outcome.completed[0]


# ---- Configuration -----------------------------------------------------------


def test_the_hub_is_not_built_until_a_capability_is_actually_used():
    # Importing this module - which the roster does, for every adapter - must
    # not read the config file or contact a company system.
    def explode() -> SourceHub:  # pragma: no cover - called only if the test fails
        raise AssertionError("the hub was built during import or description")

    capability = SourcesFetch(hub_factory=explode)
    assert "registry://SEQ-1" in capability.describe({"ref": "registry://SEQ-1"})

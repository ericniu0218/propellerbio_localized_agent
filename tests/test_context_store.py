"""The store's two promises: it gives the bytes back, and it finds things fast.

Everything here runs against a temporary directory, so no test touches a real
store on disk.
"""

import pytest

from bedrock.context import ContextStore, ingest
from bedrock.context.adapters import Source
from bedrock.context.chunker import chunk_segments
from bedrock.context.types import Segment

CSV = "\n".join(
    ["sample,ph,note"]
    + [f"S{i:04d},{6.0 + i % 20 * 0.1:.1f},routine" for i in range(400)]
)


@pytest.fixture
def store(tmp_path) -> ContextStore:
    return ContextStore(tmp_path / "context")


def test_round_trip_is_lossless(store: ContextStore):
    """Compression must not quietly lose rows - the retrieval layer is only
    worth having if what comes back is what went in."""
    artifact = ingest(Source.from_text(CSV, name="a.csv", suffix=".csv"), store=store)

    recovered = set()
    for index in range(artifact.chunk_count):
        for line in artifact.chunk(index).splitlines():
            if line.startswith("S"):
                recovered.add(line)

    assert recovered == {line for line in CSV.splitlines() if line.startswith("S")}


def test_stats_report_both_kinds_of_saving(store: ContextStore):
    artifact = ingest(Source.from_text(CSV, name="a.csv", suffix=".csv"), store=store)
    stats = artifact.stats

    assert stats.stored_bytes < stats.source_bytes
    assert stats.storage_ratio > 1
    # The digest has to be dramatically cheaper to read than the source, or the
    # context half of the design is not doing anything.
    assert stats.digest_tokens < stats.source_tokens / 10
    assert stats.chunk_count > 1


def test_same_content_is_stored_once(store: ContextStore):
    first = ingest(Source.from_text(CSV, name="a.csv", suffix=".csv"), store=store)
    second = ingest(Source.from_text(CSV, name="renamed.csv", suffix=".csv"), store=store)

    assert first.id == second.id
    assert len(store.list()) == 1


def test_search_finds_the_chunk_holding_the_answer(store: ContextStore):
    text = CSV.replace("S0123,6.3,routine", "S0123,9.9,CONTAMINATION in plate 42")
    artifact = ingest(Source.from_text(text, name="a.csv", suffix=".csv"), store=store)

    hits = artifact.search("contamination", limit=3)
    assert hits
    assert any("CONTAMINATION in plate 42" in hit.text for hit in hits)
    assert hits[0].score > 0


def test_search_returns_nothing_rather_than_noise(store: ContextStore):
    ingest(Source.from_text(CSV, name="a.csv", suffix=".csv"), store=store)
    assert store.search("kangaroo", limit=5) == []


def test_search_spans_every_artifact_and_can_be_narrowed(store: ContextStore):
    ingest(Source.from_text("the alpha protocol covers storage", name="one.md"), store=store)
    beta = ingest(Source.from_text("the beta protocol covers shipping", name="two.md"), store=store)

    assert {hit.artifact_name for hit in store.search("protocol", limit=5)} == {"one.md", "two.md"}
    narrowed = store.search("protocol", limit=5, artifact_id=beta.id)
    assert {hit.artifact_name for hit in narrowed} == {"two.md"}


def test_resolve_accepts_an_id_prefix_or_a_name(store: ContextStore):
    artifact = ingest(Source.from_text(CSV, name="a.csv", suffix=".csv"), store=store)

    assert store.resolve(artifact.id[:6]).id == artifact.id
    assert store.resolve("a.csv").id == artifact.id
    assert store.resolve("nope") is None


def test_delete_removes_it_from_disk(store: ContextStore):
    artifact = ingest(Source.from_text(CSV, name="a.csv", suffix=".csv"), store=store)

    assert store.delete(artifact.id) is True
    assert store.list() == []
    assert store.delete(artifact.id) is False


def test_reading_a_missing_chunk_is_an_error_not_a_wrong_answer(store: ContextStore):
    artifact = ingest(Source.from_text(CSV, name="a.csv", suffix=".csv"), store=store)
    with pytest.raises(IndexError):
        artifact.chunk(artifact.chunk_count + 10)


def test_a_fresh_store_is_empty_not_broken(tmp_path):
    store = ContextStore(tmp_path / "never-created")
    assert store.list() == []
    assert store.search("anything") == []
    assert store.totals().source_bytes == 0


def test_chunker_keeps_segments_whole_and_splits_only_oversized_ones():
    segments = [Segment(text="short one"), Segment(text="short two")]
    assert len(chunk_segments(segments, max_tokens=200)) == 1

    big = Segment(text="\n\n".join(f"paragraph {i} " + "word " * 40 for i in range(30)))
    chunks = chunk_segments([big], max_tokens=100)
    assert len(chunks) > 1
    assert all(chunk.tokens <= 160 for chunk in chunks)


# ---- Plural folding, and the stores written before it existed ---------------


def test_singular_and_plural_find_each_other(store: ContextStore):
    ingest(
        Source.from_text(
            "Samples showing turbidity are discarded.\n\nEach plate holds ninety six wells.",
            name="protocol.md",
            suffix=".md",
        ),
        store=store,
    )

    assert store.search("sample", limit=3), "plural text should answer a singular query"
    assert store.search("plates", limit=3), "singular text should answer a plural query"


def test_words_that_only_look_plural_are_left_alone():
    from bedrock.context.search import stem

    assert stem("samples") == stem("sample") == "sample"
    assert stem("studies") == stem("study") == "study"
    # Folding these would merge unrelated words and mangle real ones.
    for word in ("glass", "apparatus", "analysis", "ph", "plus"):
        assert stem(word) == word


def test_artifacts_stored_before_stemming_are_still_searchable(tmp_path):
    """Ids are content hashes, so an existing artifact is never rewritten. Its
    old, unstemmed word counts have to keep working."""
    import json

    store = ContextStore(tmp_path / "context")
    artifact = ingest(
        Source.from_text("Samples showing turbidity are discarded.", name="protocol.md", suffix=".md"),
        store=store,
    )

    # Rewrite the manifest the way the previous version would have written it:
    # schema_version 1, with unstemmed terms.
    manifest_path = store.root / artifact.id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] = 1
    for entry in manifest["chunks"]:
        entry["terms"] = {"samples": 1, "showing": 1, "turbidity": 1, "discarded": 1}
        entry["term_count"] = 4
    manifest_path.write_text(json.dumps(manifest))

    fresh = ContextStore(tmp_path / "context")  # a new store: no cached manifest
    assert fresh.search("sample", limit=3), "an old artifact must not go quiet"
    assert fresh.search("turbidity", limit=3)

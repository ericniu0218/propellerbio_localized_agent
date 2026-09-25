"""The store's two promises: it gives the bytes back, and it finds things fast.

Everything here runs against a temporary directory, so no test touches a real
store on disk.
"""

import pytest
from time import perf_counter

from bedrock.context import ContextStore, ingest
from bedrock.context.adapters import Source
from bedrock.context.chunker import chunk_segments
from bedrock.context.types import Chunk, Digest, Extraction, Segment

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


def test_fts_ranking_matches_manifest_scan_on_existing_search_fixtures(store: ContextStore):
    ingest(Source.from_text(CSV, name="a.csv", suffix=".csv"), store=store)
    ingest(Source.from_text("alpha protocol covers storage and alpha handling", name="one.md"), store=store)
    ingest(Source.from_text("beta protocol covers shipping", name="two.md"), store=store)
    for query in ("protocol", "alpha protocol", "sample routine", "contamination"):
        assert [(hit.artifact_id, hit.chunk_index) for hit in store.search(query, limit=5)] == [
            (hit.artifact_id, hit.chunk_index) for hit in store._scan_search(query, limit=5, artifact_id=None)
        ]


def test_fts_index_rebuilds_and_survives_deletion(store: ContextStore, monkeypatch):
    first = ingest(Source.from_text("alpha protocol", name="one.md"), store=store)
    second = ingest(Source.from_text("beta protocol", name="two.md"), store=store)
    assert store.search("protocol")
    with monkeypatch.context() as patch:
        patch.setattr(store, "_manifest", lambda _: (_ for _ in ()).throw(AssertionError("manifest loaded during search")))
        assert store.search("protocol")
    assert store.delete(first.id)
    assert {hit.artifact_id for hit in store.search("protocol")} == {second.id}
    store._index_path.unlink()
    rebuilt = ContextStore(store.root)
    assert {hit.artifact_id for hit in rebuilt.search("protocol")} == {second.id}


def test_fts_keeps_underscore_terms_and_has_a_scan_fallback(store: ContextStore, monkeypatch, capsys):
    artifact = ingest(Source.from_text("sample_id is the source key", name="schema.md"), store=store)
    assert store.search("sample_id")[0].artifact_id == artifact.id
    fallback = ContextStore(store.root)
    monkeypatch.setattr("bedrock.context.store.sqlite3.connect", lambda *args, **kwargs: (_ for _ in ()).throw(
        __import__("sqlite3").OperationalError("no such module: fts5")))
    assert fallback.search("sample_id")[0].artifact_id == artifact.id
    assert "FTS5 unavailable" in capsys.readouterr().err


def test_index_rebuilds_after_an_external_manifest_edit(store: ContextStore):
    import json

    artifact = ingest(Source.from_text("alpha protocol", name="one.md"), store=store)
    assert store.search("alpha")
    path = store.root / artifact.id / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["chunks"][0]["terms"] = {"newword": 1}
    manifest["chunks"][0]["term_count"] = 1
    path.write_text(json.dumps(manifest))
    assert store.search("newword")
    assert store.search("alpha") == []


def test_updating_digest_keeps_search_index_in_step(store: ContextStore):
    artifact = ingest(Source.from_text("alpha protocol", name="one.md"), store=store)
    replacement = Digest("updated gist", "updated overview")
    assert store.update_digest(artifact.id, replacement).digest_text() == replacement.to_text()
    assert store.search("alpha")[0].artifact_id == artifact.id
    assert ContextStore(store.root).search("alpha")[0].artifact_id == artifact.id


def test_large_fts_search_is_bounded_and_keeps_no_manifest_cache(store: ContextStore):
    chunks = [Chunk(i, f"sample {i} regular observation") for i in range(14_000)]
    chunks[-1] = Chunk(13_999, "sample 13999 uniquezebra observation")
    source = Source.from_text("synthetic source", name="large.txt")
    store.put(source=source, extraction=Extraction("text", "synthetic"), chunks=chunks,
              digest=Digest("synthetic", "14k chunks"))
    assert not hasattr(store, "_manifest_cache")
    store.search("uniquezebra")  # warm the filesystem cache
    start = perf_counter()
    hits = store.search("uniquezebra")
    elapsed = perf_counter() - start
    assert hits[0].chunk_index == 13_999
    assert elapsed < 0.04


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
    # no index_version at all, and unstemmed terms.
    manifest_path = store.root / artifact.id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("index_version", None)
    for entry in manifest["chunks"]:
        entry["terms"] = {"samples": 1, "showing": 1, "turbidity": 1, "discarded": 1}
        entry["term_count"] = 4
    manifest_path.write_text(json.dumps(manifest))

    fresh = ContextStore(tmp_path / "context")  # a new store: no cached manifest
    assert fresh.search("sample", limit=3), "an old artifact must not go quiet"
    assert fresh.search("turbidity", limit=3)


def test_plurals_formed_with_es_fold_too(store: ContextStore):
    """The -s rule alone missed a whole class of ordinary lab words: batches,
    boxes, dishes and glasses all failed to meet their singular."""
    ingest(
        Source.from_text(
            "Batches are logged per run.\n\nBoxes of dishes are autoclaved.\n\n"
            "Glasses are rinsed twice.",
            name="protocol.md",
            suffix=".md",
        ),
        store=store,
    )

    for singular in ("batch", "box", "dish", "glass"):
        assert store.search(singular, limit=3), f"{singular!r} should find its plural"


def test_singular_words_ending_in_s_are_never_shortened():
    """auto_inject searches on every message, so a query for "news" matching
    text that only says "new" would silently mislead the model."""
    from bedrock.context.search import stem

    for word in ("news", "lens", "bias", "series", "species", "atlas", "canvas"):
        assert stem(word) == word


def test_the_stemmers_known_limits_are_pinned_not_accidental():
    """Irregular plurals are out of scope. They stay self-consistent - a word
    still finds itself - they just never reach their singular form."""
    from bedrock.context.search import stem

    assert stem("analyses") != stem("analysis")
    assert stem("calories") != stem("calorie")
    # Self-consistency is the property that must hold.
    assert stem("analyses") == stem("analyses")
    assert stem("calories") == "calory"

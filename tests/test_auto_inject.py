"""find_relevant_context is what closes the "you have to remember /digest
exists" gap - these tests are about that connection actually firing, not
about search ranking quality (bedrock.context.search already covers that).
"""

from bedrock.context import ContextStore, ingest
from bedrock.context.auto_inject import find_relevant_context, with_injected_context


def _store_with(tmp_path, name: str, text: str) -> ContextStore:
    store = ContextStore(tmp_path / "context")
    path = tmp_path / name
    path.write_text(text)
    ingest(path, store=store)
    return store


def test_no_match_returns_none(tmp_path):
    store = _store_with(tmp_path, "notes.txt", "Meeting notes about scheduling next week.")
    assert find_relevant_context("what is the capital of France", store) is None


def test_relevant_stored_file_is_found_and_summarised(tmp_path):
    store = _store_with(
        tmp_path,
        "assay.csv",
        "\n".join(["sample,ph,passed"] + [f"s{i},{6.0 + i * 0.1:.1f},true" for i in range(10)]),
    )
    injected = find_relevant_context("what was the ph in the assay results", store)
    assert injected is not None
    assert "assay.csv" in injected.summary
    assert "ph" in injected.prompt_block.lower()


def test_a_one_word_message_is_never_searched(tmp_path):
    store = _store_with(tmp_path, "assay.csv", "sample,ph\ns1,7.0\n")
    assert find_relevant_context("ph", store) is None


def test_empty_store_returns_none(tmp_path):
    store = ContextStore(tmp_path / "context")
    assert find_relevant_context("what was the ph in the assay", store) is None


def test_with_injected_context_leaves_message_untouched_when_nothing_matched():
    assert with_injected_context("hello there", None) == "hello there"


def test_with_injected_context_prepends_the_block_and_keeps_the_original_message(tmp_path):
    store = _store_with(tmp_path, "assay.csv", "sample,ph\ns1,7.0\ns2,7.2\ns3,7.4\n")
    injected = find_relevant_context("remind me what the ph values were", store)
    assert injected is not None
    combined = with_injected_context("remind me what the ph values were", injected)
    assert combined.endswith("remind me what the ph values were")
    assert "Relevant stored context" in combined


def test_a_search_failure_is_swallowed_not_raised(tmp_path, monkeypatch):
    store = _store_with(tmp_path, "assay.csv", "sample,ph\ns1,7.0\n")

    def boom(*args, **kwargs):
        raise RuntimeError("disk went away")

    monkeypatch.setattr(store, "search", boom)
    assert find_relevant_context("what was the ph reading", store) is None


def test_only_the_auto_verb_is_read_as_the_switch():
    """`/digest automatic.csv` is a file to compress, not the switch."""
    from bedrock.ui.digest_command import auto_context_argument

    assert auto_context_argument("auto") == ""
    assert auto_context_argument("AUTO  off ") == "off"
    assert auto_context_argument("  auto on") == "on"
    assert auto_context_argument("automatic.csv") is None
    assert auto_context_argument("search auto") is None
    assert auto_context_argument("") is None

"""Sources under an answer name the file or memory note, not its id."""

from __future__ import annotations

from bedrock.server.answers import format_answer
from bedrock.server.orchestrator.context import render_context
from bedrock.server.schemas.api import ContextBlock

NOTE = ContextBlock(id="f79a5ee4508fb2e2e83e753069dcaf74", title="User-approved context: Preferred assay units",
                    kind="memory", text="Report concentrations in uM.")
FILE = ContextBlock(id="art1", title="Literature review - tirzepatide.md", kind="artifact", text="Liver fat fell 47%.")


def test_cited_material_is_numbered_and_named():
    text = format_answer(
        "Liver fat fell by 47% [source:art1]. Concentrations are in uM [source:f79a5ee4508fb2e2e83e753069dcaf74].",
        ["source:art1", "source:f79a5ee4508fb2e2e83e753069dcaf74"],
        [NOTE, FILE],
    )
    assert "[source:" not in text
    assert text.startswith("Liver fat fell by 47% [1]. Concentrations are in uM [2].")
    assert text.endswith("**Sources**\n1. File: Literature review - tirzepatide.md\n2. Memory: Preferred assay units")


def test_a_citation_repeated_is_listed_once():
    text = format_answer("A [source:art1]. B [source:art1].", ["source:art1", "source:art1"], [FILE])
    assert text == "A [1]. B [1].\n\n**Sources**\n1. File: Literature review - tirzepatide.md"


def test_an_invented_id_is_removed_from_the_text():
    assert format_answer("It works [source:deadbeef].", [], [FILE]) == "It works."


def test_a_tools_own_reference_is_kept_as_it_is():
    text = format_answer("See the paper.", ["PMID:123"], [])
    assert text.endswith("**Sources**\n1. PMID:123")


def test_no_citations_means_no_sources_section():
    assert format_answer("Hello.", [], [NOTE]) == "Hello."


def test_the_model_is_told_to_leave_unrelated_material_alone():
    prompt = render_context([NOTE], 4000)
    assert "ignore the rest" in prompt and "[source:ID]" in prompt

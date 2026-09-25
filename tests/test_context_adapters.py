"""Each adapter should find the structure in its own kind of data.

These tests are about the *content* of the overview, not its wording: the
overview is what a model reads instead of the file, so what matters is that
the facts a person would ask for are actually in there.
"""

from pathlib import Path

import pytest

from bedrock.context.adapters import (
    CodeAdapter,
    LogAdapter,
    Source,
    StructuredAdapter,
    TabularAdapter,
    TextAdapter,
    select_adapter,
)
from bedrock.context.adapters import code as code_adapter
from bedrock.context.adapters.base import MAX_DECODE_BYTES
from bedrock.context.adapters.logs import templatise

CSV = "\n".join(
    ["name,ph,plate,passed"]
    + [f"sample-{i},{6.0 + i * 0.1:.1f},P{i % 3},{'true' if i % 2 else 'false'}" for i in range(30)]
)

LOG = "\n".join(
    [f"2026-09-01T00:0{i % 10}:00Z INFO worker-{i % 4} request completed in {i}ms for /api/v1/runs/{i}" for i in range(40)]
    + [f"2026-09-01T01:00:0{i % 10}Z ERROR worker-1 connection refused to 10.0.0.{i}:8000" for i in range(10)]
)

CODE = '''\
"""A module that does a thing."""


def compress(data, level=6):
    """Squash it."""
    return data


class Store:
    def put(self, item):
        return item
'''


def test_csv_overview_carries_schema_and_statistics():
    extraction = TabularAdapter().extract(Source.from_text(CSV, name="a.csv", suffix=".csv"))
    overview = extraction.overview

    assert "30 rows x 4 columns" in overview
    # Column types are inferred, not guessed from the header name.
    assert "ph (number)" in overview
    assert "passed (boolean)" in overview
    assert "plate (category)" in overview
    # Real statistics, computed rather than summarised.
    assert "min 6, max 8.9" in overview
    assert extraction.metadata["rows"] == 30
    # Every row is still kept for exact lookup, not just the sample.
    rows = "\n".join(s.text for s in extraction.segments if s.kind == "rows")
    assert "sample-29" in rows


def test_log_overview_collapses_repetition_into_counted_patterns():
    extraction = LogAdapter().extract(Source.from_text(LOG, name="s.log", suffix=".log"))
    overview = extraction.overview

    # 50 lines, two shapes. That collapse is the whole point of this adapter.
    assert "50 lines, 2 distinct patterns" in overview
    assert "ERROR" in overview and "INFO" in overview
    assert "<TS>" in overview and "<IP>" in overview
    assert extraction.metadata["levels"]["ERROR"] == 10


def test_templatise_replaces_values_but_keeps_the_message():
    a = templatise("2026-09-01T00:00:00Z INFO took 146ms for /api/v1/runs/42")
    b = templatise("2026-09-02T11:30:00Z INFO took 9ms for /api/v1/runs/7")
    assert a == b
    assert "took <N>ms" in a


def test_json_overview_is_the_shape_not_the_values():
    source = Source.from_text(
        '[{"id":1,"meta":{"owner":"a"},"tags":["x"]},{"id":2,"meta":{"owner":"b"},"tags":["y"]}]',
        name="b.json",
        suffix=".json",
    )
    extraction = StructuredAdapter().extract(source)
    assert "JSON array: 2 items" in extraction.overview
    assert "meta.owner: string" in extraction.overview
    # Arrays collapse to one path instead of one per element - this is what
    # keeps the summary small on a file with thousands of records.
    assert "tags[]: string" in extraction.overview


def test_code_overview_is_the_signatures():
    extraction = CodeAdapter().extract(Source.from_text(CODE, name="m.py", suffix=".py"))
    assert "def compress(data, level) - Squash it." in extraction.overview
    assert "class Store" in extraction.overview
    assert "def put(self, item)" in extraction.overview
    # Definitions are whole segments, so retrieval returns complete functions.
    assert any(s.label == "def compress(data, level)" for s in extraction.segments)


def test_unparseable_python_still_ingests_as_code():
    extraction = CodeAdapter().extract(
        Source.from_text("def broken(:\n    ???", name="x.py", suffix=".py")
    )
    assert extraction.source_kind == "code"
    assert extraction.segments


@pytest.mark.parametrize("failure", [ValueError("null byte"), RecursionError("too deep")])
def test_python_parser_failures_fall_back_to_generic_code(monkeypatch, failure):
    def fail_parse(_text):
        raise failure

    monkeypatch.setattr(code_adapter.ast, "parse", fail_parse)
    extraction = CodeAdapter().extract(
        Source.from_text("def still_visible():\n    pass", name="x.py", suffix=".py")
    )

    assert extraction.source_kind == "code"
    assert "def still_visible()" in extraction.overview


def test_source_rejects_an_oversized_path_before_reading_it(tmp_path, monkeypatch):
    path = tmp_path / "too-large.txt"
    with path.open("wb") as handle:
        handle.truncate(MAX_DECODE_BYTES + 1)

    def unexpected_read(_path: Path):
        raise AssertionError("oversized source was read into memory")

    monkeypatch.setattr(Path, "read_bytes", unexpected_read)
    with pytest.raises(ValueError, match="over the .* MB limit"):
        Source.from_path(path)


def test_csv_cells_larger_than_the_parser_default_are_supported():
    large_cell = "x" * 200_000
    extraction = TabularAdapter().extract(
        Source.from_text(f"id,payload\n1,{large_cell}\n", name="large.csv", suffix=".csv")
    )

    assert extraction.metadata["rows"] == 1
    assert large_cell in extraction.segments[-1].text


def test_text_adapter_reports_the_outline():
    extraction = TextAdapter().extract(
        Source.from_text("# Title\n\nsome words here\n\n## Part\n\nmore words", suffix=".md")
    )
    assert "Outline:" in extraction.overview
    assert "- Title" in extraction.overview


def test_registry_routes_each_type_to_the_right_adapter():
    cases = [
        (Source.from_text(CSV, name="a.csv", suffix=".csv"), "tabular"),
        (Source.from_text(LOG, name="s.log", suffix=".log"), "logs"),
        (Source.from_text('{"a":1}', name="b.json", suffix=".json"), "structured"),
        (Source.from_text(CODE, name="m.py", suffix=".py"), "code"),
        (Source.from_text("just some prose about things", name="n.txt"), "text"),
    ]
    for source, expected in cases:
        assert select_adapter(source).name == expected, source.name


def test_prose_containing_a_date_is_not_mistaken_for_a_log():
    source = Source.from_text(
        "We met on 2026-09-01 to discuss the plan.\n\nIt went well.\n\nNotes follow.",
        name="notes.md",
    )
    assert select_adapter(source).name == "text"

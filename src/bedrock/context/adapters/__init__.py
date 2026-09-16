"""Data-type adapters: one file per kind of data, one shared protocol."""

from bedrock.context.adapters.base import Source, SourceAdapter
from bedrock.context.adapters.code import CodeAdapter
from bedrock.context.adapters.fasta import FastaAdapter, classify_alphabet, gc_content
from bedrock.context.adapters.logs import LogAdapter
from bedrock.context.adapters.registry import DEFAULT_ADAPTERS, extract, select_adapter
from bedrock.context.adapters.structured import StructuredAdapter
from bedrock.context.adapters.tabular import TabularAdapter
from bedrock.context.adapters.text import TextAdapter

__all__ = [
    "Source",
    "SourceAdapter",
    "CodeAdapter",
    "FastaAdapter",
    "LogAdapter",
    "StructuredAdapter",
    "TabularAdapter",
    "TextAdapter",
    "DEFAULT_ADAPTERS",
    "classify_alphabet",
    "extract",
    "gc_content",
    "select_adapter",
]

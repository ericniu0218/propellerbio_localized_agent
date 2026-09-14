"""Picks the adapter for a source.

Order is by how strong the evidence is, not by how useful the adapter would
be. ``StructuredAdapter`` goes first because "does this parse as JSON?" is a
yes-or-no fact; ``TextAdapter`` goes last because it accepts everything and
would otherwise swallow sources a better adapter could have handled.

Adding a data type means writing one adapter and adding it to this tuple.
"""

from __future__ import annotations

from bedrock.context.adapters.base import Source, SourceAdapter
from bedrock.context.adapters.code import CodeAdapter
from bedrock.context.adapters.logs import LogAdapter
from bedrock.context.adapters.structured import StructuredAdapter
from bedrock.context.adapters.tabular import TabularAdapter
from bedrock.context.adapters.text import TextAdapter
from bedrock.context.types import Extraction

DEFAULT_ADAPTERS: tuple[SourceAdapter, ...] = (
    StructuredAdapter(),
    CodeAdapter(),
    LogAdapter(),
    TabularAdapter(),
    TextAdapter(),
)


def select_adapter(
    source: Source, adapters: tuple[SourceAdapter, ...] | None = None
) -> SourceAdapter:
    """First adapter that claims the source; ``TextAdapter`` always will."""
    for adapter in adapters or DEFAULT_ADAPTERS:
        try:
            if adapter.can_handle(source):
                return adapter
        except Exception:
            # A broken sniff must never stop the pipeline - fall through to the
            # next adapter, and to plain text if nothing else claims it.
            continue
    return TextAdapter()


def extract(source: Source, adapters: tuple[SourceAdapter, ...] | None = None) -> Extraction:
    return select_adapter(source, adapters).extract(source)

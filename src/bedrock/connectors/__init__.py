"""Pull data straight from the company's systems of record.

Instead of exporting a file and typing its path, an employee names the record::

    /fold registry://SEQ-00123
    /digest lims://plates/P42-results.csv
    /sources search registry kinase

Each ``name`` before ``://`` is a ``[connectors.<name>]`` table in the config
file. Two general connector types cover most internal systems without
product-specific code: ``rest`` for anything with an HTTP/JSON API, and
``share`` for systems that export into a controlled folder. A product that
needs more (paging, a signing scheme, a query language) gets its own class
implementing ``SourceConnector`` and one entry in ``hub.CONNECTOR_TYPES``.

Access is decided by the source system, never re-implemented here, and every
fetched record carries provenance (system, record, version, who, when,
SHA-256) into staging sidecars and AlphaFold run manifests.
"""

from bedrock.connectors.base import (
    AccessDenied,
    ConnectorConfigError,
    ConnectorError,
    FetchedRecord,
    RecordNotFound,
    RecordSummary,
    SourceConnector,
    SourceRef,
    parse_source_ref,
)
from bedrock.connectors.hub import CONNECTOR_TYPES, SourceHub, build_connectors
from bedrock.connectors.rest import RestConnector
from bedrock.connectors.share import ShareConnector

__all__ = [
    "AccessDenied",
    "CONNECTOR_TYPES",
    "ConnectorConfigError",
    "ConnectorError",
    "FetchedRecord",
    "RecordNotFound",
    "RecordSummary",
    "RestConnector",
    "ShareConnector",
    "SourceConnector",
    "SourceHub",
    "SourceRef",
    "build_connectors",
    "parse_source_ref",
]

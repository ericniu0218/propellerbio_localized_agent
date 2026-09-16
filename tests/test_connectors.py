"""Company data-source connectors: fetching, refusing, and remembering provenance.

The REST connector runs against ``httpx.MockTransport`` and the share connector
against a temporary folder, so no real company system is involved.
"""

import asyncio
import hashlib
import json
import os
import stat
from pathlib import Path

import httpx
import pytest

from bedrock.config.settings import load_settings
from bedrock.connectors import (
    AccessDenied,
    ConnectorError,
    RecordNotFound,
    RestConnector,
    ShareConnector,
    SourceHub,
    SourceRef,
    build_connectors,
    parse_source_ref,
)
from bedrock.connectors.base import safe_filename

# ---- References -------------------------------------------------------------


def test_source_refs_are_recognised_but_web_addresses_and_paths_are_not():
    assert parse_source_ref("registry://SEQ-00123") == SourceRef("registry", "SEQ-00123")
    assert parse_source_ref("  LIMS://plates/P42.csv ") == SourceRef("lims", "plates/P42.csv")
    assert parse_source_ref("https://example.com/seq.fasta") is None
    assert parse_source_ref("/approved/project/candidate.fasta") is None
    assert parse_source_ref("candidate.fasta") is None
    assert parse_source_ref("registry://") is None


def test_filenames_from_a_source_cannot_name_a_path():
    assert safe_filename("../../.bashrc") == "bashrc"
    assert safe_filename("my seq (v2).fasta") == "my_seq_v2_.fasta"
    assert safe_filename("...") == "record"


# ---- REST ---------------------------------------------------------------------


def registry(handler, **options) -> RestConnector:
    settings = dict(
        base_url="https://registry.internal/api/v1",
        record_path="/sequences/{id}",
        content_field="data.fasta",
        filename_field="data.name",
        version_field="data.revision",
        suffix=".fasta",
        token_env="REGISTRY_TOKEN",
        env={"REGISTRY_TOKEN": "t0ken"},
    )
    settings.update(options)
    return RestConnector("registry", transport=httpx.MockTransport(handler), **settings)


def sequence_payload(name: str = "SEQ-00123") -> dict:
    return {"data": {"fasta": f">{name} kinase\nACDEFGHIK\n", "name": name, "revision": "r7"}}


async def test_rest_fetch_sends_identity_and_returns_content_with_provenance():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=sequence_payload())

    record = await registry(handler).fetch("SEQ-00123", user="alice")

    request = seen[0]
    assert request.url.path == "/api/v1/sequences/SEQ-00123"
    assert request.headers["X-Bedrock-User"] == "alice"
    assert request.headers["Authorization"] == "Bearer t0ken"
    assert record.text().startswith(">SEQ-00123 kinase")
    assert record.filename == "SEQ-00123.fasta"

    provenance = record.provenance()
    assert provenance["ref"] == "registry://SEQ-00123"
    assert provenance["version"] == "r7"
    assert provenance["fetched_by"] == "alice"
    assert provenance["sha256"] == hashlib.sha256(record.content).hexdigest()
    # The token authenticates the request; it must never be written down.
    assert "t0ken" not in json.dumps(provenance)


async def test_rest_record_ids_cannot_walk_the_api_url_tree():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(404)

    with pytest.raises(RecordNotFound):
        await registry(handler).fetch("../../admin/users", user="alice")
    assert b"..%2F..%2Fadmin%2Fusers" in seen[0].url.raw_path

    with pytest.raises(ConnectorError):
        await registry(handler).fetch("..", user="alice")
    assert len(seen) == 1  # rejected before any request was made


@pytest.mark.parametrize(
    "status, error",
    [(401, AccessDenied), (403, AccessDenied), (404, RecordNotFound), (500, ConnectorError)],
)
async def test_rest_maps_the_sources_own_answers_to_plain_errors(status, error):
    connector = registry(lambda request: httpx.Response(status))

    with pytest.raises(ConnectorError) as caught:
        await connector.fetch("SEQ-1", user="alice")

    assert type(caught.value) is error
    if error is AccessDenied:
        assert "alice" in str(caught.value)


async def test_rest_unreachable_source_is_explained():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(ConnectorError, match="Could not reach `registry`"):
        await registry(handler).fetch("SEQ-1", user="alice")


async def test_rest_missing_token_stops_before_any_request():
    seen: list[httpx.Request] = []
    connector = registry(lambda request: seen.append(request) or httpx.Response(200), env={})

    assert connector.readiness_issues()
    with pytest.raises(ConnectorError, match="REGISTRY_TOKEN"):
        await connector.fetch("SEQ-1", user="alice")
    assert seen == []


async def test_rest_filenames_from_the_source_are_sanitised():
    payload = {"data": {"fasta": ">x\nAC\n", "name": "../../.bashrc"}}
    record = await registry(lambda request: httpx.Response(200, json=payload)).fetch("SEQ-1", user="alice")
    assert record.filename == "bashrc.fasta"


async def test_rest_search_maps_results_and_skips_entries_without_ids():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["q"] == "kinase"
        return httpx.Response(
            200,
            json={"results": [{"id": "SEQ-1", "name": "Kinase A"}, {"id": "SEQ-2"}, {"name": "no id"}]},
        )

    connector = registry(handler, search_path="/sequences", results_field="results")
    results = await connector.search("kinase", user="alice")

    assert [(str(r.ref), r.title) for r in results] == [
        ("registry://SEQ-1", "Kinase A"),
        ("registry://SEQ-2", "SEQ-2"),
    ]


# ---- Share --------------------------------------------------------------------


@pytest.fixture
def share(tmp_path: Path) -> ShareConnector:
    root = tmp_path / "lims-exports"
    (root / "plates").mkdir(parents=True)
    (root / "plates" / "P42.csv").write_text("well,od\nA1,0.5\n")
    (root / "candidate.fasta").write_text(">cand\nACDE\n")
    (tmp_path / "secret.fasta").write_text(">secret\nACDE\n")
    return ShareConnector("lims", root=str(root))


async def test_share_fetches_records_inside_the_share(share: ShareConnector):
    record = await share.fetch("plates/P42.csv", user="alice")

    assert record.text().startswith("well,od")
    assert record.suffix == ".csv"
    assert record.provenance()["connector_type"] == "share"
    assert record.version and record.version.startswith("modified ")


@pytest.mark.parametrize("record_id", ["../secret.fasta", "plates/../../secret.fasta"])
async def test_share_refuses_relative_paths_outside_the_share(share: ShareConnector, record_id: str):
    with pytest.raises(AccessDenied):
        await share.fetch(record_id, user="alice")


async def test_share_refuses_absolute_paths_and_escaping_symlinks(share: ShareConnector, tmp_path: Path):
    with pytest.raises(AccessDenied):
        await share.fetch(str(tmp_path / "secret.fasta"), user="alice")

    (share.root / "shortcut.fasta").symlink_to(tmp_path / "secret.fasta")
    with pytest.raises(AccessDenied):
        await share.fetch("shortcut.fasta", user="alice")


async def test_share_missing_records_and_the_file_type_allow_list(share: ShareConnector):
    with pytest.raises(RecordNotFound):
        await share.fetch("plates/P99.csv", user="alice")

    restricted = ShareConnector("lims", root=str(share.root), extensions=["fasta"])
    with pytest.raises(ConnectorError, match="file type"):
        await restricted.fetch("plates/P42.csv", user="alice")
    assert (await restricted.fetch("candidate.fasta", user="alice")).filename == "candidate.fasta"


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores file permissions")
async def test_share_permission_errors_become_access_denied(share: ShareConnector):
    target = share.root / "plates" / "P42.csv"
    target.chmod(0)
    try:
        with pytest.raises(AccessDenied):
            await share.fetch("plates/P42.csv", user="alice")
    finally:
        target.chmod(stat.S_IRUSR | stat.S_IWUSR)


async def test_share_search_matches_every_term(share: ShareConnector):
    results = await share.search("plates p42", user="alice")

    assert [str(result.ref) for result in results] == ["lims://plates/P42.csv"]
    assert await share.search("kangaroo", user="alice") == []


# ---- Configuration and the hub -----------------------------------------------


def test_one_bad_connector_table_is_reported_without_losing_the_rest(tmp_path: Path):
    connectors, errors = build_connectors(
        {
            "lims": {"type": "share", "root": str(tmp_path)},
            "registry": {"type": "rest", "bse_url": "https://typo.internal"},
            "https": {"type": "share", "root": str(tmp_path)},
            "eln": {"type": "sharepoint"},
            "sneaky": {"type": "rest", "base_url": "https://x.internal", "transport": "anything"},
        }
    )

    assert list(connectors) == ["lims"]
    joined = "\n".join(errors)
    assert "[connectors.registry]: unknown option `bse_url`." in joined
    assert "[connectors.https]" in joined
    assert "[connectors.eln]" in joined and "sharepoint" in joined
    assert "[connectors.sneaky]: unknown option `transport`." in joined


def test_settings_read_connector_tables_from_the_config_file(tmp_path: Path):
    config = tmp_path / "config.toml"
    config.write_text(
        "[connectors.registry]\n"
        'type = "rest"\n'
        'base_url = "https://registry.internal/api"\n'
        'token_env = "REGISTRY_TOKEN"\n'
    )

    settings = load_settings(config_path=config, env={})

    assert settings.connectors["registry"]["base_url"] == "https://registry.internal/api"
    assert "connectors" not in settings.extra


async def test_hub_stages_private_files_with_a_provenance_sidecar(share: ShareConnector, tmp_path: Path):
    hub = SourceHub({"lims": share}, user="alice", staging_dir=tmp_path / "staging")

    record, path = await hub.fetch_to_file("lims://candidate.fasta")

    assert path.read_bytes() == record.content
    assert path.parent == hub.staging_dir
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(hub.staging_dir.stat().st_mode) == 0o700
    sidecar = json.loads(path.with_name(path.name + ".provenance.json").read_text())
    assert sidecar["ref"] == "lims://candidate.fasta"
    assert sidecar["fetched_by"] == "alice"
    assert sidecar["sha256"] == record.sha256


async def test_hub_explains_unknown_and_unconfigured_sources(share: ShareConnector, tmp_path: Path):
    with pytest.raises(ConnectorError, match="Configured: `lims`"):
        await SourceHub({"lims": share}, user="alice", staging_dir=tmp_path).fetch("regsitry://SEQ-1")

    with pytest.raises(ConnectorError, match="No company sources are configured"):
        await SourceHub({}, user="alice", staging_dir=tmp_path).fetch("registry://SEQ-1")


# ---- Regressions: a bad config or a full disk must never crash the app -------


def test_wrongly_typed_settings_are_reported_not_raised(tmp_path: Path):
    """The module promises one broken table never stops the app. Every one of
    these used to escape as an unhandled exception and kill `/sources`."""
    connectors, errors = build_connectors(
        {
            "listtype": {"type": ["rest"]},  # unhashable, broke the lookup itself
            "badtimeout": {"type": "rest", "base_url": "https://x.internal", "timeout": "30s"},
            "badurl": {"type": "rest", "base_url": 123},
            "badcount": {"type": "share", "root": str(tmp_path), "max_search_files": "lots"},
            "good": {"type": "share", "root": str(tmp_path)},
        }
    )

    assert list(connectors) == ["good"]
    joined = "\n".join(errors)
    for section in ("listtype", "badtimeout", "badurl", "badcount"):
        assert f"[connectors.{section}]" in joined


async def test_share_accepts_a_single_extension_written_as_a_string(share: ShareConnector):
    """`extensions = "fasta"` used to become ('.f','.a','.s','.t','.a') - the
    connector reported itself fine and then refused every single file."""
    for written_as in (".fasta", "fasta"):
        connector = ShareConnector("lims", root=str(share.root), extensions=written_as)
        record = await connector.fetch("candidate.fasta", user="alice")
        assert record.filename == "candidate.fasta"


async def test_staging_failures_are_reported_as_connector_errors(share: ShareConnector, tmp_path: Path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("")
    hub = SourceHub({"lims": share}, user="alice", staging_dir=blocked / "sources")

    with pytest.raises(ConnectorError, match="Could not save"):
        await hub.fetch_to_file("lims://candidate.fasta")


async def test_an_oversized_response_stops_being_read(monkeypatch):
    monkeypatch.setattr("bedrock.connectors.rest.MAX_RECORD_BYTES", 100)
    connector = registry(lambda request: httpx.Response(200, content=b"x" * 500))

    with pytest.raises(ConnectorError, match="larger than"):
        await connector.fetch("SEQ-1", user="alice")


async def test_concurrent_stages_of_one_record_do_not_corrupt_each_other(
    share: ShareConnector, tmp_path: Path
):
    """/sources fetch and /fold are separate workers, so two stages of the same
    record can overlap. A shared temp filename used to let them clobber one
    another, leaving bytes the sidecar no longer vouched for."""
    hub = SourceHub({"lims": share}, user="alice", staging_dir=tmp_path / "staging")

    results = await asyncio.gather(
        *[hub.fetch_to_file("lims://candidate.fasta") for _ in range(6)]
    )

    for record, path in results:
        assert path.read_bytes() == record.content
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record.sha256
    assert not list((tmp_path / "staging").glob("*.tmp"))

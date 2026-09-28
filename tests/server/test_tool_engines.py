"""bedrock's engines behind the agent-pack tools (D23-D25, F17).

rag_search on bedrock's context store with per-file access groups; eln_lims
through bedrock's connectors; the tools no chat could reach, reached by
deterministic detectors; slow subprocess tools off the event loop.
"""

from __future__ import annotations

import asyncio
import subprocess
from types import SimpleNamespace

import pytest

from bedrock.server.compliance.rbac import RBACFilter, StaticPermissionSource
from bedrock.server.config import Settings
from bedrock.server.events import get_event_bus
from bedrock.server.orchestrator.loop import handle_message
from bedrock.server.schemas.enums import EventType, TaskDomain
from bedrock.server.tools import lab_connectors, sequence_tools
from bedrock.server.tools.intent import detect_tool
from bedrock.server.tools.knowledge_index import (
    ContextStoreIndex, EmptyIndex, build_index, default_index, load_access,
)
from bedrock.server.tools.lab_connectors import ELNLIMSArgs, ELNLIMSTool
from bedrock.server.tools.rag_search import RAGSearchArgs, RAGSearchTool
from bedrock.server.tools.sequence_tools import SequenceStructureTool, SequenceToolArgs
from bedrock.context import ContextStore

SEQ = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"


# --- detectors ---------------------------------------------------------------------------


def test_a_sequence_with_an_operation_picks_the_sequence_tool():
    assert detect_tool(f"Please translate {SEQ}", TaskDomain.BIOINFORMATICS, set()) == (
        "sequence_structure", {"operation": "translate", "sequence": SEQ})
    fasta = f"Reverse complement this:\n>candidate_1\n{SEQ[:20]}\n{SEQ[20:]}"
    assert detect_tool(fasta, TaskDomain.GENERAL, set())[1] == {"operation": "reverse_complement", "sequence": SEQ}


def test_no_operation_or_no_sequence_means_no_tool():
    assert detect_tool(f"What is {SEQ}?", TaskDomain.BIOINFORMATICS, set()) is None
    assert detect_tool("Please translate this into French", TaskDomain.GENERAL, set()) is None


def test_a_record_reference_needs_a_configured_system():
    assert detect_tool("Summarise lims://P42 for me", TaskDomain.LAB_DATA, set()) is None
    assert detect_tool("Summarise the sample lims://P42", TaskDomain.LAB_DATA, {"lims"}) == (
        "eln_lims", {"operation": "get_sample", "source": "lims://P42"})


def test_a_drafting_request_gets_its_template():
    tool, args = detect_tool("Draft an SOP for plate washing", TaskDomain.GENERAL, set())
    assert tool == "doc_template" and args["template"] == "sop"
    assert detect_tool("What does the SOP say?", TaskDomain.GENERAL, set()) is None


def test_smiles_in_a_chemistry_request():
    pytest.importorskip("rdkit")
    assert detect_tool("Properties of CC(=O)Oc1ccccc1C(=O)O please", TaskDomain.CHEMISTRY, set()) == (
        "chemistry", {"operation": "properties", "smiles": "CC(=O)Oc1ccccc1C(=O)O"})
    tool, args = detect_tool("Similarity of CC(C)O and c1ccccc1O", TaskDomain.GENERAL, set())
    assert tool == "chemistry" and args["operation"] == "similarity"
    assert detect_tool("CC(C)O and c1ccccc1O", TaskDomain.GENERAL, set()) is None  # not a chemistry request


async def test_a_chat_request_now_reaches_the_sequence_tool(session_id):
    await handle_message(session_id, "alice", f"Translate this gene sequence {SEQ}")
    called = [e.payload for e in get_event_bus().history(session_id) if e.type == EventType.TOOL_CALLED]
    assert called and called[0]["tool_name"] == "sequence_structure"
    assert called[0]["arguments"]["operation"] == "translate"


# --- rag_search on bedrock's context store, with access groups ------------------------------------


async def test_the_built_index_ranks_and_keeps_restricted_files_restricted(tmp_path):
    corpus = tmp_path / "docs"
    (corpus / "clinical").mkdir(parents=True)
    (corpus / "SOP-7.txt").write_text("Mycoplasma testing is done monthly on every cell line.")
    (corpus / "clinical" / "trial.txt").write_text("Mycoplasma results for trial subjects are held by QA.")
    (corpus / "access.toml").write_text('"clinical/*" = ["clinical"]\n')
    assert build_index(corpus, tmp_path / "index") == 2

    permissions = StaticPermissionSource()
    index = ContextStoreIndex(ContextStore(tmp_path / "index"), load_access(corpus))
    tool = RAGSearchTool(index=index, rbac=RBACFilter(permissions))

    alice = await tool.run(RAGSearchArgs(query="mycoplasma testing", user_id="alice"))
    assert [c.split("#")[0] for c in alice.citations] == ["SOP-7"]
    permissions.grant("bob", "clinical")
    bob = await tool.run(RAGSearchArgs(query="mycoplasma testing", user_id="bob"))
    assert {c.split("#")[0] for c in bob.citations} == {"SOP-7", "trial"}


def test_a_real_server_never_searches_the_fictional_samples(tmp_path):
    sample = tmp_path / "samples"
    sample.mkdir()
    (sample / "fiction.txt").write_text("An invented SOP.")
    real = Settings(mode="real", data_dir=tmp_path, database_url="sqlite://")
    assert isinstance(default_index(real, sample), EmptyIndex)
    mock = Settings(mode="mock", data_dir=tmp_path, database_url="sqlite://")
    assert default_index(mock, sample).search("invented SOP", 3)


# --- eln_lims through bedrock's connectors ----------------------------------------------------------


async def test_eln_lims_fetches_a_record_from_a_configured_system(tmp_path, monkeypatch):
    exports = tmp_path / "lims"
    exports.mkdir()
    (exports / "P42.csv").write_text("well,od\nA1,0.51\n")
    settings = SimpleNamespace(connectors={"lims": {"type": "share", "root": str(exports)}}, data_dir=tmp_path)
    monkeypatch.setattr("bedrock.server.config.get_settings", lambda: settings)

    result = await ELNLIMSTool().run(ELNLIMSArgs(operation="get_experiment", source="lims://P42.csv", user_id="alice"))
    assert "A1,0.51" in result.content
    assert result.citations and result.citations[0].startswith("lims://P42.csv@")

    missing = await ELNLIMSTool().run(ELNLIMSArgs(operation="get_sample", source="lims://nope.csv"))
    assert missing.content == "" and "Could not fetch" in missing.summary


async def test_without_a_configured_system_eln_lims_says_so(monkeypatch, tmp_path):
    monkeypatch.setattr("bedrock.server.config.get_settings", lambda: SimpleNamespace(connectors={}, data_dir=tmp_path))
    result = await ELNLIMSTool().run(ELNLIMSArgs(operation="get_experiment", record_id="EXP-1"))
    assert "No ELN/LIMS system is connected" in result.summary


# --- slow tools do not stop the server ---------------------------------------------------------------


async def test_a_slow_blast_does_not_block_the_event_loop(monkeypatch):
    monkeypatch.setattr(sequence_tools.shutil, "which", lambda name: "/usr/bin/blastn")

    def slow_run(*_args, **_kwargs):
        import time
        time.sleep(0.3)
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="hit\t99.0\n", stderr="")

    monkeypatch.setattr(sequence_tools.subprocess, "run", slow_run)
    ticks = 0

    async def ticker():
        nonlocal ticks
        for _ in range(20):
            await asyncio.sleep(0.01)
            ticks += 1

    result, _ = await asyncio.gather(
        SequenceStructureTool().run(SequenceToolArgs(operation="blastn", sequence=SEQ)), ticker())
    assert result.content.startswith("hit") and ticks == 20

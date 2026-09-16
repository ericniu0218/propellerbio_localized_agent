import pytest

from bedrock.ui.commands import Command, parse_command, parse_scientific_request


def test_plain_text_is_not_a_command():
    assert parse_command("hello /model") is None
    assert parse_command("") is None


def test_command_without_args():
    assert parse_command("/help") == Command("help", "")
    assert parse_command("  /CLEAR  ") == Command("clear", "")


def test_command_with_args():
    assert parse_command("/model llama3.2:3b") == Command("model", "llama3.2:3b")


def test_unknown_command_is_still_parsed():
    assert parse_command("/bogus 1 2") == Command("bogus", "1 2")


@pytest.mark.parametrize(
    "phrase, expected",
    [
        ("Open the latest structure interactively in PyMOL", Command("pymol", "")),
        ("Please use PyMOL to view /tmp/model.cif", Command("pymol", "/tmp/model.cif")),
        (
            'Could you display "/approved/project/my structure.pdb" with PyMOL?',
            Command("pymol", "/approved/project/my structure.pdb"),
        ),
        (
            "Run AlphaFold2 on /approved/project/candidate.fasta",
            Command("fold", "/approved/project/candidate.fasta"),
        ),
        ("I want to predict the structure in target.faa", Command("fold", "target.faa")),
        ("PyMOL: open model.mmcif", Command("pymol", "model.mmcif")),
        ("Run AlphaFold 2 on registry://SEQ-00123", Command("fold", "registry://SEQ-00123")),
        ("Fold registry://SEQ-00123 please", Command("fold", "registry://SEQ-00123")),
    ],
)
def test_clear_scientific_actions_are_recognized(phrase: str, expected: Command):
    assert parse_scientific_request(phrase) == expected


@pytest.mark.parametrize(
    "message",
    [
        "How do I open a structure in PyMOL?",
        "Can PyMOL open mmCIF files?",
        "Can you explain how to use PyMOL?",
        "Tell me about AlphaFold 2 structure prediction",
        "PyMOL is commonly used to view structures",
        "Compare AlphaFold 2 with Rosetta",
        "Open PDB entry 1ABC in PyMOL",
        "What is registry://SEQ-00123?",
        "Tell me about registry://SEQ-00123",
    ],
)
def test_questions_and_discussion_stay_in_chat(message: str):
    assert parse_scientific_request(message) is None

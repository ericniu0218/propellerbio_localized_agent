from bedrock.ui.commands import Command, parse_command


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

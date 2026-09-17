from bedrock.ui.rendering import markdown_text


def test_markdown_text_strips_controls_and_escapes_markup():
    assert markdown_text("[report](bad)\x1b\n*") == r"\[report\]\(bad\)\*"

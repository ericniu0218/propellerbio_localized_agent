"""The small-window prompt contract, without a model server or Textual app."""

from bedrock.attachments import Attachment, DATA_NOT_INSTRUCTIONS
from bedrock.models.conversation import Conversation
from bedrock.prompting import PromptBudget, PromptBuilder


def test_fifty_turns_with_files_never_exceed_laptop_window():
    conversation = Conversation(system_prompt="Help with the employee's question.")
    builder = PromptBuilder(PromptBudget(8192, 1024))
    for turn in range(50):
        files = []
        if turn in (1, 7, 18, 31, 43):
            files = [Attachment(f"file-{turn}.txt", f"/file-{turn}.txt", 12_000,
                                "whole", "data " * 2400, f"Digest of file {turn}.")]
        conversation.add_user(f"Question {turn}: " + "sample " * 85, files)
        result = builder.build(conversation)
        assert result.estimated_tokens + 1024 <= 8192
        assert result.estimated_tokens == sum((len(item["content"]) + 3) // 4 + 4 for item in result.messages)
        conversation.add_assistant("Answer: " + "measurement " * 60)


def test_sent_prompt_is_a_prefix_until_a_trim():
    conversation = Conversation(system_prompt="Be helpful.")
    builder = PromptBuilder(PromptBudget(2048, 256))
    previous = None
    trims = 0
    for turn in range(35):
        conversation.add_user(f"Question {turn} " + "sample " * 35)
        result = builder.build(conversation)
        if previous is not None and not result.trimmed:
            assert result.messages[:len(previous)] == previous
        if result.trimmed:
            trims += 1
        previous = result.messages
        conversation.add_assistant("Answer " + "measurement " * 20)
    assert trims >= 2


def test_turn_extras_do_not_change_the_fixed_prompt_start():
    conversation = Conversation(system_prompt="Be helpful.")
    builder = PromptBuilder(PromptBudget(8192, 1024))
    conversation.add_user("First question")
    first = builder.build(conversation)
    conversation.add_assistant("First answer")
    conversation.add_user("Second question")
    second = builder.build(conversation, {"note": "one-turn note", "recall": "old excerpt", "passages": "stored passage"})
    assert second.messages[:len(first.messages)] == first.messages
    assert second.messages[0] == first.messages[0]
    assert "one-turn note" in second.messages[-1]["content"]
    assert "[Excerpts from my earlier chats]" in second.messages[-1]["content"]
    assert "stored passage" in second.messages[-1]["content"]
    conversation.add_assistant("Second answer")
    conversation.add_user("Third question")
    third = builder.build(conversation)
    assert third.messages[:len(second.messages)] == second.messages
    assert "stored passage" not in third.messages[-1]["content"]


def test_attachment_is_whole_once_then_fenced_digest():
    conversation = Conversation(system_prompt="Be helpful.")
    builder = PromptBuilder(PromptBudget(8192, 1024))
    file = Attachment("notes.txt", "/notes.txt", 100, "whole", "the full source text", "short digest")
    conversation.add_user("Read @notes.txt", [file])
    first = builder.build(conversation)
    assert "the full source text" in first.messages[-1]["content"]
    conversation.add_assistant("Okay")
    conversation.add_user("Follow up")
    second = builder.build(conversation)
    assert "the full source text" not in second.messages[1]["content"]
    assert "short digest" in second.messages[1]["content"]
    assert DATA_NOT_INSTRUCTIONS in second.messages[1]["content"]
    assert second.trimmed


def test_notice_counts_earlier_messages_left_out():
    conversation = Conversation(system_prompt="Be brief.")
    builder = PromptBuilder(PromptBudget(1024, 128))
    for turn in range(20):
        conversation.add_user(f"Question {turn} " + "sample " * 30)
        result = builder.build(conversation)
        conversation.add_assistant("answer " * 15)
    assert result.left_out > 0
    assert f"Left out {result.left_out} earlier messages" in result.notices[0]

"""Plain-language labels never leak jargon and stay stable for known states."""

from bedrock.agentpack.friendly import friendly_agent_line, friendly_error


def test_director_planning():
    assert friendly_agent_line("director", "thinking", "Plan task") == (
        "Planner: breaking your request into steps"
    )


def test_director_reviewing():
    assert friendly_agent_line("director", "thinking", "Final synthesis and review") == (
        "Planner: double-checking the results"
    )


def test_specialist_researching_vs_drafting():
    researching = friendly_agent_line("specialist", "running_tool", "Literature scan")
    drafting = friendly_agent_line("specialist", "thinking", "Draft the SOP")
    assert researching.startswith("Researcher:")
    assert drafting.startswith("Assistant:")


def test_runner_is_always_assistant():
    assert friendly_agent_line("runner", "thinking", "Classify request").startswith("Assistant:")


def test_no_role_name_or_status_enum_leaks_into_the_label():
    for role in ("director", "specialist", "runner"):
        for status in ("idle", "thinking", "running_tool", "waiting_for_review", "done", "error"):
            line = friendly_agent_line(role, status, "Some task")
            assert role not in line.lower()
            assert status not in line.lower()


def test_friendly_error_never_echoes_python_exception_text():
    raw = "RuntimeError: BudgetExceededError: specialist exceeded budget on task_123"
    plain = friendly_error(raw)
    assert "RuntimeError" not in plain
    assert "BudgetExceededError" not in plain
    assert "task_123" not in plain


def test_friendly_error_has_a_fallback_for_unknown_errors():
    plain = friendly_error("some totally new error type nobody planned for")
    assert plain  # never empty
    assert "Traceback" not in plain

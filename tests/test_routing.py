from app.orchestrator.router import effective_complexity, is_high_stakes
from app.schemas.enums import TaskComplexity, TaskDomain


def test_regulatory_domain_is_always_high_stakes():
    assert is_high_stakes(TaskDomain.REGULATORY, "just a routine question") is True


def test_clinical_domain_is_always_high_stakes():
    assert is_high_stakes(TaskDomain.CLINICAL, "anything") is True


def test_keyword_triggers_high_stakes_regardless_of_domain():
    assert is_high_stakes(TaskDomain.GENERAL, "please write the final report for this study") is True


def test_general_domain_without_keywords_is_not_high_stakes():
    assert is_high_stakes(TaskDomain.GENERAL, "what's the boiling point of water?") is False


def test_high_stakes_forces_hard_complexity_even_if_classified_simple():
    assert effective_complexity(TaskComplexity.SIMPLE, high_stakes=True) == TaskComplexity.HARD


def test_non_high_stakes_keeps_classified_complexity():
    assert effective_complexity(TaskComplexity.NORMAL, high_stakes=False) == TaskComplexity.NORMAL

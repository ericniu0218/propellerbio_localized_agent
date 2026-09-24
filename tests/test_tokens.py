"""The local token estimate and how real counts correct it."""

from bedrock.inference import Usage, estimate_tokens, tokens


def test_estimate_is_about_four_characters_a_token_and_rounds_up():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2
    assert estimate_tokens("x" * 4000) == 1000


def test_observed_counts_move_the_ratio_towards_what_the_server_measured():
    for _ in range(20):
        tokens.observe(3000, Usage(prompt_tokens=1000))  # 3 characters a token
    assert abs(tokens.characters_per_token() - 3.0) < 0.05
    assert 1000 <= estimate_tokens("x" * 3000) <= 1020


def test_missing_or_tiny_observations_teach_nothing():
    tokens.observe(5000, Usage())
    tokens.observe(5000, Usage(prompt_tokens=0))
    tokens.observe(50, Usage(prompt_tokens=40))  # mostly per-message wrapping
    assert tokens.characters_per_token() == tokens.DEFAULT_CHARACTERS_PER_TOKEN


def test_one_strange_prompt_cannot_swing_the_ratio_out_of_bounds():
    for _ in range(50):
        tokens.observe(10_000, Usage(prompt_tokens=100))
    assert tokens.characters_per_token() <= tokens.MAX_CHARACTERS_PER_TOKEN
    for _ in range(50):
        tokens.observe(1_000, Usage(prompt_tokens=1_000))
    assert tokens.characters_per_token() >= tokens.MIN_CHARACTERS_PER_TOKEN

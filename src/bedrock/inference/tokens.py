"""How many tokens a piece of text will take, estimated here without a server.

A token is the unit a model reads and writes: about four characters of
English, fewer for code or other scripts. The context window, and so the
memory the server sets aside, is measured in tokens, so anything that has to
fit a prompt into the window has to count them.

Counting exactly would need the model's own tokenizer, which lives on the
server. Instead we divide the length by a characters-per-token ratio. It
starts at 4. Each time the server reports how many tokens a chat prompt really
took (``Usage.prompt_tokens``), ``observe`` moves the ratio part of the way
towards what was measured, so the estimate settles on the model in use.

The ratio is kept in memory only, for this run of the app. It is never
written anywhere or sent anywhere.
"""

from __future__ import annotations

import math

from bedrock.inference.base import Usage

DEFAULT_CHARACTERS_PER_TOKEN = 4.0
"""The usual rule of thumb for English text with a modern tokenizer."""

MIN_CHARACTERS_PER_TOKEN = 2.0
MAX_CHARACTERS_PER_TOKEN = 6.0
"""Bounds on the learned ratio, so one odd prompt (say, all digits or all
emoji) cannot swing later estimates wildly."""

LEARNING_RATE = 0.3
"""How far each observation moves the ratio. Small enough that one unusual
prompt does not undo what earlier ones taught, large enough to settle within a
few turns."""

MIN_OBSERVED_CHARACTERS = 200
"""Very short prompts are mostly the server's own per-message wrapping tokens,
which would teach a misleadingly low ratio. They are ignored."""

_characters_per_token = DEFAULT_CHARACTERS_PER_TOKEN


def estimate_tokens(text: str) -> int:
    """Roughly how many tokens ``text`` will take. Rounds up, never negative.

    Rounding up is the safe direction: a prompt thought slightly too big is
    trimmed a little early, while one thought too small is cut by the server.
    """
    if not text:
        return 0
    return math.ceil(len(text) / _characters_per_token)


def observe(characters: int, usage: Usage) -> None:
    """Learn from a prompt of ``characters`` that the server says took ``usage.prompt_tokens``.

    Does nothing when the server sent no count. The server's count includes
    the few tokens it wraps around each message, so the learned ratio leans
    slightly low, and estimates lean slightly high - the safe direction again.
    """
    global _characters_per_token
    tokens = usage.prompt_tokens
    if not tokens or tokens <= 0 or characters < MIN_OBSERVED_CHARACTERS:
        return
    measured = min(max(characters / tokens, MIN_CHARACTERS_PER_TOKEN), MAX_CHARACTERS_PER_TOKEN)
    _characters_per_token += LEARNING_RATE * (measured - _characters_per_token)


def characters_per_token() -> float:
    """The ratio currently in use, for tests and for anyone curious."""
    return _characters_per_token


def reset() -> None:
    """Forget what was learned. Tests call this so one test cannot affect another."""
    global _characters_per_token
    _characters_per_token = DEFAULT_CHARACTERS_PER_TOKEN

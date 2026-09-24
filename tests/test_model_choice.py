"""Picking a model that fits the machine when none is configured."""

from bedrock.config import PROFILES
from bedrock.inference import ModelInfo, choose_default_model, format_size

GB = 1_000_000_000
LAPTOP = PROFILES["laptop"].max_auto_model_bytes


def test_the_largest_model_that_fits_is_picked():
    models = [ModelInfo("tiny", size_bytes=1 * GB), ModelInfo("big", size_bytes=5 * GB), ModelInfo("mid", size_bytes=2 * GB)]
    choice = choose_default_model(models, LAPTOP, "laptop")
    assert choice.model.id == "mid"
    assert "`mid`" in choice.reason and "laptop" in choice.reason


def test_without_any_sizes_the_first_listed_model_is_kept():
    models = [ModelInfo("first"), ModelInfo("second")]
    choice = choose_default_model(models, LAPTOP, "laptop")
    assert choice.model.id == "first"
    assert "does not report" in choice.reason


def test_when_nothing_fits_the_smallest_is_picked_with_a_warning():
    models = [ModelInfo("huge", size_bytes=40 * GB), ModelInfo("large", size_bytes=5 * GB)]
    choice = choose_default_model(models, LAPTOP, "laptop")
    assert choice.model.id == "large"
    assert "larger than" in choice.reason


def test_no_limit_picks_the_largest_model():
    models = [ModelInfo("a", size_bytes=1 * GB), ModelInfo("b", size_bytes=40 * GB)]
    assert choose_default_model(models, None).model.id == "b"


def test_an_empty_server_has_no_choice():
    assert choose_default_model([], LAPTOP) is None


def test_sizes_are_shown_in_decimal_units():
    assert format_size(2_019_393_189) == "2.0 GB"
    assert format_size(274_302_450) == "274.3 MB"
    assert format_size(None) == "size unknown"

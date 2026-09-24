"""Which model to start with when none is configured, and how to show sizes.

Loading a model that doesn't fit is the quickest way to make an 8 GB laptop
unusable: the operating system starts swapping and everything slows to a
crawl. So when the employee hasn't named a model, the app picks the largest
one the server lists whose weights fit ``MachineProfile.max_auto_model_bytes``.
"Largest that fits" because, among models that fit, a bigger one usually
answers better.

When the server doesn't say how big its models are, we cannot tell what fits,
so the app falls back to the first model the server lists (what it always did)
and says so, rather than pretending to know.
"""

from __future__ import annotations

from dataclasses import dataclass

from bedrock.inference.base import ModelInfo


@dataclass(frozen=True)
class ModelChoice:
    """The model picked, and a sentence the app can show to explain why."""

    model: ModelInfo
    reason: str


def fits(model: ModelInfo, max_bytes: int | None) -> bool | None:
    """True if the model fits the limit, False if not, None if its size is unknown."""
    if max_bytes is None:
        return True
    if model.size_bytes is None:
        return None
    return model.size_bytes <= max_bytes


def choose_default_model(models: list[ModelInfo], max_bytes: int | None, profile_name: str = "") -> ModelChoice | None:
    """Pick a model to start with, or None when the server lists none."""
    if not models:
        return None
    label = f"the {profile_name} profile" if profile_name else "this machine"
    sized = [model for model in models if model.size_bytes is not None]
    if max_bytes is None:
        if sized:
            largest = max(sized, key=lambda model: model.size_bytes or 0)
            return ModelChoice(largest, f"Picked `{largest.id}`, the largest model the server offers ({format_size(largest.size_bytes)}).")
        return ModelChoice(models[0], f"Picked `{models[0].id}`, the first model the server lists.")
    fitting = [model for model in sized if model.size_bytes <= max_bytes]
    if fitting:
        best = max(fitting, key=lambda model: model.size_bytes or 0)
        return ModelChoice(
            best,
            f"Picked `{best.id}` ({format_size(best.size_bytes)}), the largest model that fits "
            f"{label} (up to {format_size(max_bytes)}). Use `/model` to choose another.",
        )
    if sized and len(sized) == len(models):
        # Every size is known and none fits. Take the smallest rather than
        # nothing, and say plainly that it may be too big.
        smallest = min(sized, key=lambda model: model.size_bytes or 0)
        return ModelChoice(
            smallest,
            f"Picked `{smallest.id}` ({format_size(smallest.size_bytes)}), the smallest model the server offers, "
            f"but it is larger than {label} allows ({format_size(max_bytes)}) and may run slowly. "
            "Ask IT for a 3-4B model at 4-bit.",
        )
    unknown = next(model for model in models if model.size_bytes is None)
    return ModelChoice(
        unknown,
        f"Picked `{unknown.id}`, the first model whose size the server does not report, so it may not fit "
        f"{label}. Use `/model` to choose another.",
    )


def format_size(size_bytes: int | None) -> str:
    """``2019393189`` becomes ``2.0 GB``. Decimal units, as Ollama and disk tools show them."""
    if size_bytes is None:
        return "size unknown"
    for unit, scale in (("GB", 1_000_000_000), ("MB", 1_000_000), ("KB", 1_000)):
        if size_bytes >= scale:
            return f"{size_bytes / scale:.1f} {unit}"
    return f"{size_bytes} B"

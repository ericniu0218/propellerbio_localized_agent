"""Machine profiles: one setting that sizes everything to the computer running the model.

Why this exists. What decides whether Bedrock runs well on a laptop is not the
app, which uses about 44 MB, but the model server's memory. It holds two things:

* the model's **weights**, fixed for a given model (``llama3.2:3b`` at 4-bit is
  about 2.0 GB); and
* the **KV cache**, the server's working memory for the prompt it is reading.
  Ollama reserves it for the whole context window as soon as the model loads,
  so the window size *is* the memory setting.

For ``llama3.2:3b`` at 4-bit:

=========================  ======
Context window (tokens)    KV cache
=========================  ======
4k                         0.5 GB
8k                         0.9 GB
16k                        1.9 GB
32k                        3.8 GB
=========================  ======

On an 8 GB laptop, macOS and the employee's other apps take roughly half, so
weights plus cache have to stay near 3 GB: a window of 4k-8k tokens. A
profile writes that decision down once, and every feature reads its limits
from here instead of inventing its own constant.

Choose with ``profile = "laptop"`` in ``~/.config/bedrock/config.toml`` or
``BEDROCK_PROFILE=laptop``. ``context_window = N`` (or
``BEDROCK_CONTEXT_WINDOW``) overrides just the window. The window must match
what the server was started with (``OLLAMA_CONTEXT_LENGTH`` for Ollama,
``--max-model-len`` for vLLM); the client cannot change the server's setting.

With nothing configured the profile is ``laptop`` (decision D4 in
``docs/roadmaps/LAPTOP_MEMORY.md``), because the default endpoint is Ollama on
the employee's own machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GB = 1_000_000_000

DEFAULT_PROFILE = "laptop"


@dataclass(frozen=True)
class MachineProfile:
    """The limits one kind of machine can afford. See the module docstring for the numbers."""

    name: str
    """``laptop``, ``workstation`` or ``server``."""

    context_window: int
    """Tokens the server holds for one request, prompt and reply together.
    Must match ``OLLAMA_CONTEXT_LENGTH`` or vLLM's ``--max-model-len``."""

    reply_limit: int
    """The longest chat reply, in tokens (``max_tokens``). The prompt builder
    keeps this much of the window free so the answer has room."""

    attachment_share: float
    """The fraction of the window that files attached this turn may use.
    Anything beyond it is sent as a digest instead."""

    auto_inject_passages: int
    """How many ``/digest`` passages may be added to a question automatically.
    0 turns automatic adding off."""

    background: Literal["idle", "concurrent"]
    """When background calls such as titles may run. ``idle``: only when nothing
    else is running or waiting. ``concurrent``: alongside other work, at a
    lower priority."""

    max_parallel: int
    """How many model calls may run at once. Each parallel slot costs the
    server another full KV cache, so a laptop runs one."""

    max_auto_model_bytes: int | None
    """The largest model the app picks by itself when none is configured.
    ``None`` means no limit. An employee can still choose a bigger one."""

    idle_seconds: float = 2.0
    """With ``background="idle"``: how long after a chat reply finishes before
    background work may start. The employee often types the next message
    within a few seconds; a title started in that gap would only be cancelled
    again, after making the server set aside its saved work on the chat."""


PROFILES: dict[str, MachineProfile] = {
    # 8 GB laptop. An 8k window costs ~0.9 GB of cache next to ~2 GB of
    # weights. One call at a time, background work only when idle, and never
    # auto-pick anything above 3.5 GB (a 3-4B model at 4-bit is 2-2.5 GB; an
    # 8B one is ~4.9 GB and will not fit alongside the cache and macOS).
    # 1024 reply tokens is a long answer yet leaves 7k for the prompt;
    # attachments may take about a third of the window in the turn they are
    # added; two auto-added passages keep the rest for the conversation.
    "laptop": MachineProfile(
        name="laptop",
        context_window=8192,
        reply_limit=1024,
        attachment_share=0.35,
        auto_inject_passages=2,
        background="idle",
        max_parallel=1,
        max_auto_model_bytes=int(3.5 * GB),
    ),
    # A 32-64 GB desktop or a single-GPU box: a 32k window (~3.8 GB of cache
    # for a 3B model, more for bigger ones) and models up to ~12 GB, which
    # covers 14B at 4-bit. Still one call at a time, because one person's
    # machine serves one person.
    "workstation": MachineProfile(
        name="workstation",
        context_window=32768,
        reply_limit=2048,
        attachment_share=0.40,
        auto_inject_passages=3,
        background="idle",
        max_parallel=1,
        max_auto_model_bytes=12 * GB,
    ),
    # A shared vLLM/SGLang server sized by IT. It batches requests, so several
    # may run together and background work need not wait for idle. No size
    # limit: IT decides what is deployed.
    "server": MachineProfile(
        name="server",
        context_window=32768,
        reply_limit=4096,
        attachment_share=0.50,
        auto_inject_passages=3,
        background="concurrent",
        max_parallel=4,
        max_auto_model_bytes=None,
        idle_seconds=0.0,
    ),
}

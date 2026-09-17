"""Small shared pieces every capability adapter needs.

A *capability* is the only code in this product that touches the outside
world - a file, a web server, a calendar, a mail server, a subprocess. Each one
lives in its own file under ``capabilities/`` so that adding an adapter means
adding a file rather than editing a shared registry.

This module holds the three things those files must not each invent:

1. **Where configuration comes from.** ``capability_config()`` reads
   ``[agents.capabilities.<name>]`` out of the settings object without any
   capability needing to know how settings are laid out. Endpoints are always
   configured, never discovered: no adapter may invent a host.
2. **How a result gets back.** ``contract.Outcome`` is frozen and has no field
   for "what the tool found", so the convention here is that ``completed``
   carries the readable result lines and ``artifacts`` carries only paths this
   run actually created or staged. ``done()`` and ``failed()`` build both.
3. **The promise never to replace a file.** ``create_without_replacing()`` is
   the single implementation of it, because two copies of that rule are two
   rules, and the one that drifts is the one that destroys a week of work.

**Grants are not checked here.** ``agents/protocol.py`` gives a capability
``run(arguments)`` and nothing else, and ``Supervisor.execute`` refuses to call
it at all until ``GrantStore.consume`` has accepted a grant that was issued for
this exact plan, names this step, and has not been spent. That check is
stronger than anything an adapter could do with the ``Grant`` value alone - a
hand-built ``Grant`` passes a "is my name in ``steps_allowed``?" test and fails
``consume`` - so it lives in one place, above these files, and these files do
not repeat a weaker version of it.

What an adapter *does* own is the lock that has nothing to do with approval:
``mail.send`` and ``shell.run`` ship switched off, mail has a recipient
allowlist, the web has a domain allowlist and a proxy, and files have roots.
Those hold whether or not anybody approved anything.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from bedrock.agents.contract import Outcome

MAX_RESULT_CHARS = 200_000
"""Ceiling on any one readable result line handed back to a caller.

A capability that reads a 400 MB log must not turn it into a 400 MB string that
then gets rendered in a terminal and posted into a model's context.
"""


class CapabilityConfigError(RuntimeError):
    """A ``[agents.capabilities.<name>]`` table is missing or unusable.

    The message is shown to the employee as it is, so it says which setting is
    wrong and where that setting lives, not which exception was raised.
    """


# ---- Results -----------------------------------------------------------------
#
# ``plan_id`` is left empty. A capability does not know which plan it belongs
# to and has no business guessing: the supervisor holds the grant, and builds
# the plan's own Outcome from the ones its steps returned. ``echo`` does the
# same thing for the same reason.


def done(
    completed: Sequence[str],
    artifacts: Sequence[str | Path] = (),
) -> Outcome:
    """A run that finished. ``completed`` is what to show a person."""
    return Outcome(
        plan_id="",
        completed=tuple(bounded(str(line)) for line in completed),
        artifacts=tuple(str(path) for path in artifacts),
    )


def failed(
    reason: str,
    *,
    completed: Sequence[str] = (),
    artifacts: Sequence[str | Path] = (),
) -> Outcome:
    """A run that did not finish, with the plain-English reason why.

    ``completed`` and ``artifacts`` are still carried: a step that wrote two
    files and then failed on the third has to say that it wrote two files.
    """
    return Outcome(
        plan_id="",
        completed=tuple(bounded(str(line)) for line in completed),
        artifacts=tuple(str(path) for path in artifacts),
        failed=bounded(reason, 2_000),
    )


def bounded(text: str, limit: int = MAX_RESULT_CHARS) -> str:
    """``text`` cut to ``limit`` characters, saying so when it was cut."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n[... {len(text) - limit} more characters not shown]"


# ---- Configuration -----------------------------------------------------------


def capability_config(name: str, settings: Any = None) -> Mapping[str, Any]:
    """The ``[agents.capabilities.<name>]`` table, or an empty one.

    Settings are loaded lazily and only when a capability is actually used, so
    importing an adapter never reads the employee's config file or contacts
    anything - which is what lets the roster import every adapter just to list
    what exists.

    A dotted name is looked up twice and merged, family first::

        [agents.capabilities.files]           # roots, shared by all three verbs
        [agents.capabilities."files.write"]   # anything only writing needs

    That exists because one adapter file offers several verbs over one set of
    rules. ``files.search``, ``files.read`` and ``files.write`` all check the
    same roots, and writing those roots into three tables is how two of them
    end up saying something different from the third.
    """
    if settings is None:
        from bedrock.config.settings import load_settings

        settings = load_settings()
    agents = getattr(settings, "agents", None)
    if not isinstance(agents, Mapping):
        # A settings object from before ``[agents]`` was a recognised key keeps
        # it among the unrecognised ones instead.
        extra = getattr(settings, "extra", None)
        agents = extra.get("agents") if isinstance(extra, Mapping) else None
    if not isinstance(agents, Mapping):
        return {}
    capabilities = agents.get("capabilities")
    if not isinstance(capabilities, Mapping):
        return {}

    merged: dict[str, Any] = {}
    family = name.split(".", 1)[0]
    for key in dict.fromkeys((family, name)):
        table = capabilities.get(key)
        if isinstance(table, Mapping):
            merged.update(table)
    return merged


def as_text(table: Mapping[str, Any], key: str, default: str = "") -> str:
    value = table.get(key, default)
    if value is None:
        return default
    if not isinstance(value, (str, int, float)):
        raise CapabilityConfigError(f"`{key}` must be text, not {type(value).__name__}.")
    return str(value).strip()


def as_list(table: Mapping[str, Any], key: str) -> tuple[str, ...]:
    """A list-valued setting, accepting a bare string as the one-item list.

    Writing ``allowed_domains = "fda.gov"`` in a config file is the natural
    thing to do, and iterating that string would produce one entry per letter -
    a setting that reads as configured and matches nothing.
    """
    value = table.get(key)
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise CapabilityConfigError(f"`{key}` must be a list of strings.")
    return tuple(str(item).strip() for item in value if str(item).strip())


def as_int(table: Mapping[str, Any], key: str, default: int) -> int:
    value = table.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise CapabilityConfigError(f"`{key}` must be a whole number, not {value!r}.") from exc


def as_float(table: Mapping[str, Any], key: str, default: float) -> float:
    value = table.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise CapabilityConfigError(f"`{key}` must be a number, not {value!r}.") from exc


def as_bool(table: Mapping[str, Any], key: str, default: bool) -> bool:
    value = table.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise CapabilityConfigError(f"`{key}` must be true or false, not {value!r}.")


def as_path(table: Mapping[str, Any], key: str, default: str = "") -> Path | None:
    text = as_text(table, key, default)
    return Path(text).expanduser() if text else None


def as_paths(table: Mapping[str, Any], key: str) -> tuple[Path, ...]:
    return tuple(Path(item).expanduser() for item in as_list(table, key))


# ---- Writing without replacing -----------------------------------------------


def create_without_replacing(target: Path, data: bytes) -> Path:
    """Create ``target``, or ``name (2).ext``, ``name (3).ext``, and so on.

    The one implementation of a promise two adapters make: an agent creates
    files, it never replaces them. Exclusive creation (``"xb"``) rather than
    "check whether it exists, then write": two agents can run at once, and
    between that check and that write is exactly where the overwrite would
    happen.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    stem, suffix = target.stem, target.suffix
    for attempt in range(1, 1000):
        candidate = target if attempt == 1 else target.with_name(f"{stem} ({attempt}){suffix}")
        try:
            with candidate.open("xb") as handle:
                handle.write(data)
            return candidate
        except FileExistsError:
            continue
    raise OSError(f"{target.name} and 999 numbered variations of it already exist.")


def secret_from_env(
    table: Mapping[str, Any],
    key: str,
    *,
    env: Mapping[str, str] | None = None,
) -> str:
    """A credential named by the config file and held in the environment.

    Config files get copied into tickets and checked into repositories, so a
    table names the variable (``password_env = "BEDROCK_CALDAV_PASSWORD"``)
    and the value itself never appears in it. This is the same rule the
    company data-source connectors follow.
    """
    variable = as_text(table, key)
    if not variable:
        return ""
    source = os.environ if env is None else env
    return source.get(variable, "")

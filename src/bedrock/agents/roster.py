"""Load agent permissions and discover only the capabilities they name.

There is deliberately no registry in this module.  A roster entry naming
``echo`` causes ``bedrock.agents.capabilities.echo`` to be imported and its
module-level ``CAPABILITY`` or ``CAPABILITIES`` value to be inspected.  A
capability absent from the roster is never imported and therefore cannot be
offered to the planner.
"""

from __future__ import annotations

import importlib
import re
import tomllib
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from bedrock.agents.contract import Risk
from bedrock.agents.protocol import Capability

_MODULE_NAME = re.compile(r"^[a-z][a-z0-9_]*$")


class RosterError(ValueError):
    """The roster or a capability module is missing or invalid."""


@dataclass(frozen=True)
class Agent:
    """One rostered agent and the exact resources it may consume."""

    name: str
    capabilities: Mapping[str, Capability]
    max_steps: int
    max_seconds: float
    max_bytes: int


class Roster(Mapping[str, Agent]):
    """Read-only collection of configured agents."""

    def __init__(self, agents: Mapping[str, Agent]) -> None:
        self._agents = MappingProxyType(dict(agents))

    def __getitem__(self, name: str) -> Agent:
        return self._agents[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._agents)

    def __len__(self) -> int:
        return len(self._agents)


def load_roster(
    path: Path | str,
    *,
    max_steps: int = 8,
    max_seconds: float = 30.0,
    max_bytes: int = 1_048_576,
) -> Roster:
    """Read ``agents.toml`` and import only capabilities named by each agent.

    The file shape is intentionally small::

        [agents.reader]
        capabilities = ["echo"]
        max_steps = 4       # optional per-agent overrides
    """
    roster_path = Path(path).expanduser()
    try:
        with roster_path.open("rb") as handle:
            document = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RosterError(f"Could not read agent roster: {roster_path}") from exc

    table = document.get("agents")
    if not isinstance(table, dict):
        raise RosterError("Agent roster needs an [agents.<name>] table.")

    defaults = (
        _positive_int(max_steps, "max_steps"),
        _positive_number(max_seconds, "max_seconds"),
        _positive_int(max_bytes, "max_bytes"),
    )
    agents: dict[str, Agent] = {}
    for name, entry in table.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(entry, dict):
            raise RosterError("Every agent roster entry must be a named table.")
        capability_names = entry.get("capabilities")
        if not isinstance(capability_names, list) or not capability_names:
            raise RosterError(f"Agent {name!r} needs a non-empty capabilities list.")
        if any(not isinstance(item, str) or not item for item in capability_names):
            raise RosterError(f"Agent {name!r} has an invalid capability name.")
        if len(set(capability_names)) != len(capability_names):
            raise RosterError(f"Agent {name!r} lists a capability more than once.")

        capabilities = {
            capability_name: discover_capability(capability_name)
            for capability_name in capability_names
        }
        agents[name] = Agent(
            name=name,
            capabilities=MappingProxyType(capabilities),
            max_steps=_positive_int(entry.get("max_steps", defaults[0]), "max_steps"),
            max_seconds=_positive_number(
                entry.get("max_seconds", defaults[1]), "max_seconds"
            ),
            max_bytes=_positive_int(entry.get("max_bytes", defaults[2]), "max_bytes"),
        )
    return Roster(agents)


def discover_capability(name: str) -> Capability:
    """Import the adapter module for ``name`` and return its exported capability."""
    if not isinstance(name, str) or not name:
        raise RosterError("Capability name must be a non-empty string.")
    module_name = name.split(".", 1)[0]
    if not _MODULE_NAME.fullmatch(module_name):
        raise RosterError(f"Invalid capability name: {name!r}")
    qualified_name = f"bedrock.agents.capabilities.{module_name}"
    try:
        module = importlib.import_module(qualified_name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise RosterError(f"Capability module is unavailable: {module_name}") from exc
    export = getattr(module, "CAPABILITY", None)
    if export is None:
        export = getattr(module, "CAPABILITIES", None)
    if export is None:
        raise RosterError(
            f"Capability module {module_name!r} has no CAPABILITY or CAPABILITIES export."
        )

    capability = _select_export(export, name)
    _validate_capability(capability, name)
    return capability


def _select_export(export: Any, name: str) -> Any:
    """Allow a family module to export one adapter or several named adapters."""
    if isinstance(export, Mapping):
        try:
            return export[name]
        except KeyError as exc:
            raise RosterError(f"Capability module does not export {name!r}.") from exc
    if isinstance(export, Sequence) and not isinstance(export, (str, bytes, bytearray)):
        matches = [candidate for candidate in export if getattr(candidate, "name", None) == name]
        if len(matches) != 1:
            raise RosterError(f"Capability module does not export exactly one {name!r}.")
        return matches[0]
    return export


def _validate_capability(capability: Any, requested_name: str) -> None:
    if getattr(capability, "name", None) != requested_name:
        raise RosterError(
            f"Capability export name does not match roster name {requested_name!r}."
        )
    if not isinstance(getattr(capability, "risk", None), Risk):
        raise RosterError(f"Capability {requested_name!r} has no valid risk.")
    if not callable(getattr(capability, "describe", None)):
        raise RosterError(f"Capability {requested_name!r} cannot describe a plan step.")
    if not callable(getattr(capability, "run", None)):
        raise RosterError(f"Capability {requested_name!r} cannot run.")


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise RosterError(f"{name} must be a positive integer.")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise RosterError(f"{name} must be a positive number.")
    return float(value)

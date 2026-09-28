"""Startup check that fails loudly if the process can reach the internet.

This catches *misconfiguration* (a stray default gateway, a container that
wasn't actually put on the isolated network) — it is not a defense against a
determined bypass from inside the box, and is documented as such in
docs/server/ARCHITECTURE.md section 8. Disable via AIRGAP_CHECK=false only for
local development off the air-gapped network.
"""

from __future__ import annotations

import socket
from dataclasses import dataclass, field

# A handful of well-known external hosts. Reachability is expected to FAIL
# on a correctly air-gapped network — if any of these succeed, we refuse to
# start.
_PROBE_TARGETS: list[tuple[str, int]] = [
    ("8.8.8.8", 53),
    ("1.1.1.1", 53),
    ("api.anthropic.com", 443),
    ("api.openai.com", 443),
]


@dataclass
class AirgapViolation(Exception):
    reachable_hosts: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        hosts = ", ".join(self.reachable_hosts)
        return (
            f"Air-gap check failed: outbound connectivity detected to: {hosts}. "
            "Refusing to start — this deployment must not reach the internet."
        )


def run_airgap_check(timeout_s: float = 1.5) -> None:
    reachable: list[str] = []
    for host, port in _PROBE_TARGETS:
        try:
            with socket.create_connection((host, port), timeout=timeout_s):
                reachable.append(f"{host}:{port}")
        except OSError:
            continue  # expected: unreachable means the air gap is intact
    if reachable:
        raise AirgapViolation(reachable_hosts=reachable)

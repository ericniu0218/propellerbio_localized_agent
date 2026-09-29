"""EPO OPS's `X-Throttling-Control` header: read on every response, not just
on failure. This is the header the fair-use monitoring chapter of the OPS
reference guide (§2.3.3) describes, quoted directly from its own example:

    X-Throttling-Control: idle (retrieval=green:200, search=yellow:20,
    inpadoc=red:30, images=green:200, other=green:1000)

`system-state` is idle/busy/overloaded; each service category then carries a
colour (green/yellow/red/black) and a numeric limit. OPS does not wait for a
429 to tell you it is unhappy — a search can be sitting at `yellow` or `red`
while every individual call still returns 200, and the docs' own worked
example shows the header only starts recovering a few requests after a client
stops hammering it. A client that ignores the colour and free-runs until the
first 403 (`X-Rejection-Reason: RegisteredQuotaPerWeek`) has already spent
most of the week's fair-use budget by the time it notices.

So `EpoThrottle.wait()` is called before every request, not after a failure:
it slows down proactively as a category's colour degrades, using the delays
below rather than anything this repo invented — the doc's example table shows
the system moving idle -> busy -> overloaded across a handful of requests and
recovering within roughly 20 seconds of quiet, which is the scale these
delays are picked to respect.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from typing import Callable

_HEADER = re.compile(r"^\s*(?P<state>\w+)\s*\((?P<services>.*)\)\s*$")
_SERVICE = re.compile(r"(?P<name>[\w-]+)\s*=\s*(?P<colour>green|yellow|red|black)\s*:\s*(?P<limit>\d+)", re.I)

# Seconds to pause before the *next* request in a category at this colour.
# green/idle: no extra delay, the retry/backoff in SourceHTTP already covers
# ordinary transport hiccups. yellow: a short pause is enough to stop adding
# to a category that is already elevated. red: near the fair-use ceiling for
# that category — a real pause, not a token-bucket nudge. black: that
# category will reject the next request outright; there is nothing to gain
# by retrying inside the same minute, so this is a deliberately long wait
# rather than a busy loop against a server that has already said no.
_DELAY_S = {"green": 0.0, "yellow": 1.5, "red": 8.0, "black": 60.0}


@dataclass(frozen=True)
class ThrottleStatus:
    system_state: str
    services: dict[str, tuple[str, int]]
    """service name -> (colour, limit)."""

    def colour(self, service: str) -> str | None:
        entry = self.services.get(service)
        return entry[0].lower() if entry else None


def parse(header: str) -> ThrottleStatus | None:
    """A malformed or absent header is not an error — it means "assume
    green" and let the transport-level retry logic in SourceHTTP handle
    whatever actually goes wrong."""
    if not header:
        return None
    m = _HEADER.match(header)
    if not m:
        return None
    services = {
        s.group("name").lower(): (s.group("colour").lower(), int(s.group("limit")))
        for s in _SERVICE.finditer(m.group("services"))
    }
    return ThrottleStatus(system_state=m.group("state").lower(), services=services)


class EpoThrottle:
    """Tracks the last throttle status per service category and sleeps
    before a caller's next request in that category, proportional to how
    close it is to being rejected outright.

    One instance is shared by every request `EpoOps` makes, because the
    quota it describes is per *consumer key*, not per call.
    """

    def __init__(self, sleep: Callable[[float], None] = time.sleep) -> None:
        self._sleep = sleep
        self._lock = threading.Lock()
        self._last: dict[str, str] = {}
        self.last_status: ThrottleStatus | None = None

    def observe(self, header: str | None) -> None:
        status = parse(header or "")
        if status is None:
            return
        with self._lock:
            self.last_status = status
            self._last.update({name: colour for name, (colour, _limit) in status.services.items()})

    def wait_before(self, service: str) -> None:
        with self._lock:
            colour = self._last.get(service, "green")
        delay = _DELAY_S.get(colour, 0.0)
        if delay:
            self._sleep(delay)

"""Putting a hold in the calendar, over CalDAV.

This is what turns "let's meet next Tuesday" into something a person can see
in the week view::

    [agents.capabilities.calendar]
    url = "https://caldav.internal/dav/calendars/alice/work/"
    username = "alice"
    password_env = "BEDROCK_CALDAV_PASSWORD"   # the value stays out of the file
    timezone = "Europe/London"
    default_minutes = 30
    timeout = 30.0

``calendar.create`` is ``CHANGES_THINGS``, so the supervisor will not call it
until somebody has approved the plan it belongs to.

**It creates a hold, not an invitation.** Attendees are deliberately not
supported. Adding them makes the CalDAV server send iTIP mail to everybody
named, which turns a calendar write into eight emails - that is
``REACHES_PEOPLE`` wearing a calendar's clothes, and it belongs behind the same
approval as ``mail.send`` rather than sneaking in as a side effect of a
scheduling step. So this capability blocks time on the employee's own calendar,
and the employee invites people themselves.

No CalDAV library is used: the product depends on ``httpx`` and the standard
library, and creating one event is one authenticated ``PUT`` of one iCalendar
object to a URL (RFC 4791 §5.3.2). ``If-None-Match: *`` asks the server to
refuse if something is already at that address, so a retry cannot quietly
replace an event that already exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as utc_timezone
from typing import Any, Mapping
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from bedrock.agents.capability_support import (
    CapabilityConfigError,
    as_float,
    as_int,
    as_text,
    capability_config,
    done,
    failed,
    secret_from_env,
)
from bedrock.agents.contract import Outcome, Risk
from bedrock.agents.protocol import Capability

MAX_SUMMARY_CHARS = 300
MAX_TEXT_CHARS = 5_000


# ---- Configuration -----------------------------------------------------------


@dataclass(frozen=True)
class CalendarConfig:
    url: str = ""
    username: str = ""
    password: str = ""
    timezone_name: str = "UTC"
    default_minutes: int = 30
    timeout: float = 30.0

    @classmethod
    def from_settings(
        cls, settings: Any = None, *, env: Mapping[str, str] | None = None
    ) -> "CalendarConfig":
        table = capability_config("calendar", settings)
        return cls(
            url=as_text(table, "url").rstrip("/"),
            username=as_text(table, "username"),
            password=secret_from_env(table, "password_env", env=env),
            timezone_name=as_text(table, "timezone", "UTC") or "UTC",
            default_minutes=as_int(table, "default_minutes", 30),
            timeout=as_float(table, "timeout", 30.0),
        )

    def checked(self) -> "CalendarConfig":
        if not self.url:
            raise CapabilityConfigError(
                "No calendar is configured. Add `url` to [agents.capabilities.calendar] "
                "in ~/.config/bedrock/config.toml - it is the CalDAV address of the "
                "calendar to write into."
            )
        return self

    def zone(self) -> ZoneInfo:
        try:
            return ZoneInfo(self.timezone_name)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise CapabilityConfigError(
                f"`timezone` in [agents.capabilities.calendar] is not a known time zone "
                f"({self.timezone_name!r}). Use a name like `Europe/London`."
            ) from exc


class CalendarRefused(Exception):
    """An event an agent asked for cannot be written as asked."""


class CalendarCreate:
    """Block time on the configured calendar. One event, no invitations."""

    name = "calendar.create"
    risk = Risk.CHANGES_THINGS

    def __init__(
        self,
        config: CalendarConfig | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        # ``transport`` is a code-only test seam, as in the company data-source
        # connectors: no config file can set it.
        self._config = config
        self._transport = transport

    @property
    def config(self) -> CalendarConfig:
        if self._config is None:
            self._config = CalendarConfig.from_settings()
        return self._config

    def describe(self, arguments: dict[str, str]) -> str:
        summary = str(arguments.get("summary", "")).strip() or "(no title given)"
        start = str(arguments.get("start", "")).strip() or "(no start time given)"
        minutes = str(arguments.get("duration_minutes", "")).strip()
        end = str(arguments.get("end", "")).strip()
        length = f" until {end}" if end else (f" for {minutes} minutes" if minutes else "")
        return f"Hold {summary!r} in the calendar from {start}{length}. Nobody is invited."

    async def run(self, arguments: dict[str, str]) -> Outcome:
        try:
            config = self.config.checked()
            event, uid, starts, ends = _build_event(arguments, config)
        except (CapabilityConfigError, CalendarRefused) as exc:
            return failed(str(exc))

        url = f"{config.url}/{uid}.ics"
        options: dict[str, Any] = {
            "timeout": httpx.Timeout(config.timeout, connect=10.0),
            "follow_redirects": False,
        }
        if self._transport is not None:
            options["transport"] = self._transport
        if config.username:
            options["auth"] = (config.username, config.password)

        try:
            async with httpx.AsyncClient(**options) as client:
                response = await client.put(
                    url,
                    content=event.encode("utf-8"),
                    headers={
                        "Content-Type": "text/calendar; charset=utf-8",
                        # "Only if nothing is there": a retried request cannot
                        # replace an event somebody else put at this address.
                        "If-None-Match": "*",
                    },
                )
        except httpx.TimeoutException:
            return failed("The calendar server took too long to answer.")
        except httpx.HTTPError as exc:
            return failed(f"Could not reach the calendar server: {exc}")

        if response.status_code in (401, 403):
            return failed(
                f"The calendar server refused `{config.username or 'this client'}`. "
                "Check `username` and the password named by `password_env`.",
            )
        if response.status_code == 412:
            return failed(
                "The calendar server already has an event at that address."
            )
        if response.status_code >= 400:
            return failed(
                f"The calendar server answered HTTP {response.status_code}."
            )

        return done(
            (
                f"Held {_display(arguments)} from {starts.isoformat()} to "
                f"{ends.isoformat()} (UTC) on {config.url}.",
                f"Event {uid}. Nobody was invited; send the invitation yourself.",
            ),
        )


def _display(arguments: Mapping[str, str]) -> str:
    return repr(str(arguments.get("summary", "")).strip() or "(untitled)")


def _build_event(
    arguments: Mapping[str, str], config: CalendarConfig
) -> tuple[str, str, datetime, datetime]:
    """The iCalendar object to PUT, its UID, and the times it covers in UTC."""
    summary = str(arguments.get("summary", "")).strip()
    if not summary:
        raise CalendarRefused("calendar.create needs a `summary`, the title of the hold.")

    starts = _moment(str(arguments.get("start", "")), config, "start")
    end_text = str(arguments.get("end", "")).strip()
    if end_text:
        ends = _moment(end_text, config, "end")
    else:
        minutes = arguments.get("duration_minutes", config.default_minutes)
        try:
            length = int(minutes)
        except (TypeError, ValueError) as exc:
            raise CalendarRefused(
                f"`duration_minutes` must be a whole number of minutes, not {minutes!r}."
            ) from exc
        if length <= 0:
            raise CalendarRefused("`duration_minutes` must be more than zero.")
        ends = starts + timedelta(minutes=length)
    if ends <= starts:
        raise CalendarRefused("The end of the hold is not after its start.")

    uid = f"bedrock-{uuid4().hex}"
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Bedrock Private AI//Agent Calendar//EN",
        "CALSCALE:GREGORIAN",
        "BEGIN:VEVENT",
        f"UID:{uid}@bedrock.local",
        f"DTSTAMP:{_stamp(datetime.now(utc_timezone.utc))}",
        f"DTSTART:{_stamp(starts)}",
        f"DTEND:{_stamp(ends)}",
        f"SUMMARY:{_escape(summary[:MAX_SUMMARY_CHARS])}",
        "TRANSP:OPAQUE",
    ]
    description = str(arguments.get("description", "")).strip()
    if description:
        lines.append(f"DESCRIPTION:{_escape(description[:MAX_TEXT_CHARS])}")
    location = str(arguments.get("location", "")).strip()
    if location:
        lines.append(f"LOCATION:{_escape(location[:MAX_TEXT_CHARS])}")
    lines += ["END:VEVENT", "END:VCALENDAR"]
    return "\r\n".join(_fold(line) for line in lines) + "\r\n", uid, starts, ends


def _moment(text: str, config: CalendarConfig, label: str) -> datetime:
    """An ISO 8601 time, as a moment in UTC.

    A time with no zone in it is read in the calendar's configured time zone,
    because "14:00" in a meeting means the local two o'clock. Everything is
    then converted to UTC before it is written, which avoids shipping a
    VTIMEZONE block whose rules would go stale.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise CalendarRefused(
            f"calendar.create needs a `{label}` time, written as 2026-03-04T14:00."
        )
    try:
        moment = datetime.fromisoformat(cleaned)
    except ValueError as exc:
        raise CalendarRefused(
            f"`{label}` must be a date and time like 2026-03-04T14:00, not {cleaned!r}. "
            "Working out what \"next Tuesday\" means is the planner's job, not this one's."
        ) from exc
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=config.zone())
    return moment.astimezone(utc_timezone.utc)


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y%m%dT%H%M%SZ")


def _escape(value: str) -> str:
    r"""iCalendar text escaping (RFC 5545 §3.3.11).

    Order matters: backslashes first, or every escape added afterwards gets
    escaped again and a comma arrives as ``\\,`` instead of ``\,``.
    """
    return (
        value.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
    )


def _fold(line: str) -> str:
    """Wrap a content line at 75 octets, as RFC 5545 §3.1 requires.

    The limit counts bytes, not characters, and a continuation line starts with
    one space. Splitting in the middle of a multi-byte character would produce
    an object some servers reject and others store broken, so the cut is moved
    back off any continuation byte.
    """
    raw = line.encode("utf-8")
    if len(raw) <= 75:
        return line
    parts: list[str] = []
    limit = 75
    while len(raw) > limit:
        cut = limit
        while cut > 0 and (raw[cut] & 0xC0) == 0x80:
            cut -= 1
        if cut == 0:  # pragma: no cover - one character wider than the line
            cut = limit
        parts.append(raw[:cut].decode("utf-8", errors="replace"))
        raw = raw[cut:]
        limit = 74  # a continuation line spends one octet on its leading space
    parts.append(raw.decode("utf-8", errors="replace"))
    return "\r\n ".join(parts)


CAPABILITY: tuple[Capability, ...] = (CalendarCreate(),)
"""What the roster finds in this file.

A tuple, which ``roster.discover_capability`` unpacks by name. One file
per adapter, several verbs per file where they share one set of rules.
"""

CAPABILITIES = CAPABILITY
"""Plural alias, for reading a module's contents rather than the roster's."""

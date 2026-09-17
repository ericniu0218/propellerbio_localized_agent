"""calendar.create: a hold appears, nobody gets invited, nothing is replaced.

The CalDAV server is an ``httpx.MockTransport`` that records what it was sent,
so these tests check the actual iCalendar bytes rather than trusting that the
right thing was built.
"""

from pathlib import Path

import httpx
import pytest

from bedrock.agents.capabilities.calendar import (
    CAPABILITIES,
    CalendarConfig,
    CalendarCreate,
    _fold,
)
from bedrock.agents.contract import Ask, AskKind, Evidence, Plan, Risk, Step
from bedrock.agents.grants import GrantStore
from bedrock.agents.supervisor import Supervisor, SupervisorError


class FakeCalDAV:
    """A CalDAV server that remembers the one request it was given."""

    def __init__(self, status: int = 201) -> None:
        self.status = status
        self.requests: list[httpx.Request] = []

    @property
    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            self.requests.append(request)
            return httpx.Response(self.status)

        return httpx.MockTransport(handler)

    @property
    def event(self) -> str:
        return self.requests[-1].content.decode("utf-8")


@pytest.fixture
def config() -> CalendarConfig:
    return CalendarConfig(
        url="https://caldav.internal/dav/calendars/alice/work",
        username="alice",
        password="from-the-environment",
        timezone_name="Europe/London",
    )


class _NoJournal:
    """A journal the supervisor never gets as far as writing to."""

    def append(self, operation_id: str, **_: object) -> int:  # pragma: no cover
        raise AssertionError("a capability ran without an approved plan")


def unapproved_hold() -> Plan:
    """A plan nobody has approved, for the supervisor to refuse."""
    return Plan(
        ask=Ask(
            kind=AskKind.SCHEDULE,
            text="block an hour for DSMB prep",
            evidence=Evidence(quote="let's meet next Tuesday"),
        ),
        steps=(
            Step(
                tool="calendar.create",
                summary="Hold 'DSMB prep'",
                risk=Risk.CHANGES_THINGS,
                arguments=hold(),
            ),
        ),
    )


def hold(**arguments: str) -> dict[str, str]:
    return {"summary": "DSMB prep", "start": "2026-03-04T14:00", **arguments}


# ---- What the roster finds ---------------------------------------------------


def test_creating_a_calendar_entry_changes_things():
    assert {c.name: c.risk for c in CAPABILITIES} == {
        "calendar.create": Risk.CHANGES_THINGS
    }


def test_the_description_says_that_nobody_is_invited():
    described = CalendarCreate(CalendarConfig()).describe(hold(duration_minutes="60"))
    assert "'DSMB prep'" in described
    assert "for 60 minutes" in described
    assert "Nobody is invited" in described


# ---- Nothing happens without an approved plan -----------------------------------------


async def test_an_unapproved_hold_never_reaches_the_calendar_server(
    config: CalendarConfig,
):
    # ``protocol.Capability`` gives run() nothing but its arguments, so the gate
    # is the supervisor's: it refuses before the capability is called at all,
    # and the CalDAV server is never contacted.
    server = FakeCalDAV()
    capability = CalendarCreate(config, transport=server.transport)
    supervisor = Supervisor({"calendar.create": capability}, GrantStore(), _NoJournal())

    with pytest.raises(SupervisorError):
        await supervisor.execute(unapproved_hold(), None)
    assert server.requests == []


# ---- Writing the hold --------------------------------------------------------


async def test_a_hold_is_put_once_to_its_own_address(config: CalendarConfig):
    server = FakeCalDAV()
    outcome = await CalendarCreate(config, transport=server.transport).run(
        hold(duration_minutes="45", location="Room 2")
    )

    assert outcome.failed is None
    request = server.requests[0]
    assert request.method == "PUT"
    assert str(request.url).startswith(config.url + "/bedrock-")
    assert str(request.url).endswith(".ics")
    # "Only if nothing is there": a retry cannot replace somebody's event.
    assert request.headers["if-none-match"] == "*"
    assert request.headers["content-type"].startswith("text/calendar")

    event = server.event
    assert "BEGIN:VEVENT" in event
    assert "SUMMARY:DSMB prep" in event
    assert "LOCATION:Room 2" in event
    assert "\r\n" in event


async def test_a_time_with_no_zone_is_read_in_the_configured_zone(config: CalendarConfig):
    server = FakeCalDAV()
    await CalendarCreate(config, transport=server.transport).run(
        hold(start="2026-03-04T14:00", duration_minutes="30")
    )
    # 4 March is GMT in London, so 14:00 local is 14:00 UTC.
    assert "DTSTART:20260304T140000Z" in server.event
    assert "DTEND:20260304T143000Z" in server.event


async def test_a_summer_time_date_is_converted_not_copied(config: CalendarConfig):
    server = FakeCalDAV()
    await CalendarCreate(config, transport=server.transport).run(
        hold(start="2026-07-04T14:00")
    )
    # July is BST, one hour ahead of UTC.
    assert "DTSTART:20260704T130000Z" in server.event


async def test_an_explicit_offset_in_the_time_is_honoured(config: CalendarConfig):
    server = FakeCalDAV()
    await CalendarCreate(config, transport=server.transport).run(
        hold(start="2026-03-04T14:00+02:00")
    )
    assert "DTSTART:20260304T120000Z" in server.event


async def test_attendees_are_not_supported_so_no_invitation_is_sent(config: CalendarConfig):
    server = FakeCalDAV()
    outcome = await CalendarCreate(config, transport=server.transport).run(
        hold(attendees="bob@company.example,carol@company.example")
    )
    assert "ATTENDEE" not in server.event
    assert "Nobody was invited" in outcome.completed[1]


async def test_text_is_escaped_so_a_comma_cannot_become_a_second_value(
    config: CalendarConfig,
):
    server = FakeCalDAV()
    await CalendarCreate(config, transport=server.transport).run(
        hold(summary="Review; amendment 3, part 2")
    )
    assert "SUMMARY:Review\\; amendment 3\\, part 2" in server.event


def test_a_long_line_is_folded_at_seventy_five_octets():
    folded = _fold("DESCRIPTION:" + "a" * 200)
    first, *rest = folded.split("\r\n")
    assert len(first.encode("utf-8")) == 75
    assert all(line.startswith(" ") for line in rest)


def test_folding_never_splits_a_multi_byte_character():
    folded = _fold("SUMMARY:" + "é" * 100)
    for line in folded.split("\r\n"):
        line.lstrip(" ").encode("utf-8").decode("utf-8")  # would raise if split badly
        assert len(line.encode("utf-8")) <= 75


# ---- Refusing sensibly -------------------------------------------------------


async def test_working_out_next_tuesday_is_not_this_capability_s_job(
    config: CalendarConfig,
):
    server = FakeCalDAV()
    outcome = await CalendarCreate(config, transport=server.transport).run(
        hold(start="next Tuesday")
    )
    assert "planner's job" in (outcome.failed or "")
    assert server.requests == []


async def test_an_end_before_the_start_is_refused(config: CalendarConfig):
    outcome = await CalendarCreate(config).run(
        hold(end="2026-03-04T13:00")
    )
    assert "not after its start" in (outcome.failed or "")


async def test_a_hold_needs_a_title(config: CalendarConfig):
    outcome = await CalendarCreate(config).run(
        {"start": "2026-03-04T14:00"}
    )
    assert "needs a `summary`" in (outcome.failed or "")


async def test_an_unconfigured_calendar_says_where_to_configure_it():
    outcome = await CalendarCreate(CalendarConfig()).run(
        hold()
    )
    assert "[agents.capabilities.calendar]" in (outcome.failed or "")


async def test_a_refusal_from_the_server_names_the_account(config: CalendarConfig):
    server = FakeCalDAV(status=403)
    outcome = await CalendarCreate(config, transport=server.transport).run(
        hold()
    )
    assert "refused `alice`" in (outcome.failed or "")


async def test_an_address_already_in_use_is_reported_rather_than_overwritten(
    config: CalendarConfig,
):
    server = FakeCalDAV(status=412)
    outcome = await CalendarCreate(config, transport=server.transport).run(
        hold()
    )
    assert "already has an event" in (outcome.failed or "")


def test_the_password_comes_from_the_environment_not_the_config_file(tmp_path: Path):
    settings = type(
        "S",
        (),
        {
            "extra": {
                "agents": {
                    "capabilities": {
                        "calendar": {
                            "url": "https://caldav.internal/dav",
                            "username": "alice",
                            "password_env": "TEST_CALDAV_PASSWORD",
                        }
                    }
                }
            }
        },
    )()
    config = CalendarConfig.from_settings(settings, env={"TEST_CALDAV_PASSWORD": "s3cret"})
    assert config.password == "s3cret"
    assert config.url == "https://caldav.internal/dav"

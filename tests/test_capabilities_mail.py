"""mail.draft and mail.send: four locks on the riskiest verb in the product.

No test here may reach a mail server, so ``MailSend`` takes a sender function
and every test passes one that records instead of sending. A test that failed
to record is a test that sent nothing - which is the safe direction to fail in.
"""

from email.message import EmailMessage
from pathlib import Path

import pytest

from bedrock.agents.capabilities.mail import (
    CAPABILITIES,
    MailConfig,
    MailDraft,
    MailSend,
)
from bedrock.agents.contract import Ask, AskKind, Evidence, Plan, Risk, Step
from bedrock.agents.grants import GrantStore
from bedrock.agents.supervisor import Supervisor, SupervisorError


class Outbox:
    """A sender that keeps the message instead of putting it on the wire."""

    def __init__(self) -> None:
        self.sent: list[EmailMessage] = []

    def __call__(self, message: EmailMessage, config: MailConfig) -> None:
        self.sent.append(message)


@pytest.fixture
def drafting(tmp_path: Path) -> MailConfig:
    return MailConfig(from_address="alice@company.example", draft_dir=tmp_path / "drafts")


@pytest.fixture
def sending(tmp_path: Path) -> MailConfig:
    return MailConfig(
        from_address="alice@company.example",
        draft_dir=tmp_path / "drafts",
        enabled=True,
        allowed_recipient_domains=("company.example",),
        smtp_host="smtp.internal",
    )


class _NoJournal:
    """A journal the supervisor never gets as far as writing to."""

    def append(self, operation_id: str, **_: object) -> int:  # pragma: no cover
        raise AssertionError("a capability ran without an approved plan")


def unapproved_send() -> Plan:
    """A plan nobody has approved, for the supervisor to refuse."""
    return Plan(
        ask=Ask(
            kind=AskKind.CREATE,
            text="send Bob the enrolment numbers",
            evidence=Evidence(quote="can you mail Bob the numbers"),
        ),
        steps=(
            Step(
                tool="mail.send",
                summary="Send 'Enrolment numbers' to bob@company.example",
                risk=Risk.REACHES_PEOPLE,
                arguments=message(),
            ),
        ),
    )


def message(**arguments: str) -> dict[str, str]:
    return {
        "to": "bob@company.example",
        "subject": "Enrolment numbers",
        "body": "412 by March.",
        **arguments,
    }


# ---- What the roster finds ---------------------------------------------------


def test_drafting_and_sending_are_two_capabilities_at_two_risk_levels():
    assert {c.name: c.risk for c in CAPABILITIES} == {
        "mail.draft": Risk.CHANGES_THINGS,
        "mail.send": Risk.REACHES_PEOPLE,
    }


def test_the_description_of_sending_says_it_cannot_be_taken_back():
    described = MailSend(MailConfig()).describe(message())
    assert "SEND" in described
    assert "cannot be taken back" in described


# ---- mail.draft --------------------------------------------------------------


async def test_a_draft_is_an_ordinary_eml_file_the_employee_can_open(
    drafting: MailConfig,
):
    outcome = await MailDraft(drafting).run(message())

    assert outcome.failed is None
    written = Path(outcome.artifacts[0])
    assert written.suffix == ".eml"
    text = written.read_text(encoding="utf-8")
    assert "To: bob@company.example" in text
    assert "From: alice@company.example" in text
    assert "412 by March." in text
    assert "Nothing was sent" in outcome.completed[1]


async def test_two_drafts_with_the_same_subject_do_not_replace_each_other(
    drafting: MailConfig,
):
    first = await MailDraft(drafting).run(message())
    second = await MailDraft(drafting).run(message(body="Corrected."))
    assert first.artifacts[0] != second.artifacts[0]
    assert "412 by March." in Path(first.artifacts[0]).read_text(encoding="utf-8")


async def test_a_newline_in_a_header_cannot_add_a_second_header(drafting: MailConfig):
    outcome = await MailDraft(drafting).run(
        message(subject="Numbers\nBcc: everyone@company.example")
    )
    assert "line break or control character" in (outcome.failed or "")
    assert outcome.artifacts == ()


async def test_an_address_that_is_not_an_address_is_refused(drafting: MailConfig):
    outcome = await MailDraft(drafting).run(
        message(to="not-an-address")
    )
    assert "is not an email address" in (outcome.failed or "")


async def test_a_message_needs_a_recipient_and_a_subject(drafting: MailConfig):
    no_recipient = await MailDraft(drafting).run(message(to=""))
    assert "at least one address" in (no_recipient.failed or "")
    no_subject = await MailDraft(drafting).run(message(subject=""))
    assert "needs a `subject`" in (no_subject.failed or "")


# ---- mail.send: the four locks -----------------------------------------------


async def test_lock_one_an_unapproved_send_never_reaches_the_capability(
    sending: MailConfig,
):
    # ``protocol.Capability`` gives run() nothing but its arguments, so this
    # lock is the supervisor's: ``GrantStore.consume`` refuses a missing grant,
    # and it refuses a hand-built one too, which no check inside this adapter
    # could do. Nothing is sent, and the journal is never even reached.
    outbox = Outbox()
    supervisor = Supervisor(
        {"mail.send": MailSend(sending, sender=outbox)}, GrantStore(), _NoJournal()
    )

    with pytest.raises(SupervisorError):
        await supervisor.execute(unapproved_send(), None)
    assert outbox.sent == []


async def test_lock_two_sending_ships_switched_off(tmp_path: Path):
    off = MailConfig(
        from_address="alice@company.example",
        draft_dir=tmp_path / "drafts",
        allowed_recipient_domains=("company.example",),
        smtp_host="smtp.internal",
    )
    assert off.enabled is False
    outbox = Outbox()
    outcome = await MailSend(off, sender=outbox).run(message())
    assert "switched off" in (outcome.failed or "")
    assert outbox.sent == []


async def test_lock_three_a_mail_server_has_to_be_configured(tmp_path: Path):
    no_server = MailConfig(
        from_address="alice@company.example",
        draft_dir=tmp_path / "drafts",
        enabled=True,
        allowed_recipient_domains=("company.example",),
    )
    outbox = Outbox()
    outcome = await MailSend(no_server, sender=outbox).run(
        message()
    )
    assert "No `smtp_host`" in (outcome.failed or "")
    assert outbox.sent == []


async def test_lock_four_an_empty_recipient_allowlist_means_nobody(tmp_path: Path):
    unrestricted = MailConfig(
        from_address="alice@company.example",
        draft_dir=tmp_path / "drafts",
        enabled=True,
        smtp_host="smtp.internal",
    )
    outbox = Outbox()
    outcome = await MailSend(unrestricted, sender=outbox).run(
        message()
    )
    assert "empty deliberately means nobody" in (outcome.failed or "")
    assert outbox.sent == []


async def test_a_recipient_outside_the_allowed_domains_is_refused(sending: MailConfig):
    outbox = Outbox()
    outcome = await MailSend(sending, sender=outbox).run(
        message(to="journalist@newspaper.example")
    )
    assert "outside the domains this agent may mail" in (outcome.failed or "")
    assert outbox.sent == []


async def test_a_copied_recipient_is_checked_as_well_as_the_main_one(sending: MailConfig):
    outbox = Outbox()
    outcome = await MailSend(sending, sender=outbox).run(
        message(cc="journalist@newspaper.example")
    )
    assert "outside the domains" in (outcome.failed or "")
    assert outbox.sent == []


async def test_with_all_four_locks_open_the_message_goes(sending: MailConfig):
    outbox = Outbox()
    outcome = await MailSend(sending, sender=outbox).run(
        message(cc="carol@company.example")
    )

    assert outcome.failed is None
    assert len(outbox.sent) == 1
    sent = outbox.sent[0]
    assert sent["To"] == "bob@company.example"
    assert sent["Cc"] == "carol@company.example"
    assert sent["From"] == "alice@company.example"
    assert sent.get_content().strip() == "412 by March."
    assert "cannot be taken back" in outcome.completed[1]


async def test_a_subdomain_of_an_allowed_domain_is_allowed(sending: MailConfig):
    outbox = Outbox()
    outcome = await MailSend(sending, sender=outbox).run(
        message(to="bob@mail.company.example")
    )
    assert outcome.failed is None
    assert len(outbox.sent) == 1


async def test_a_failure_from_the_mail_server_says_nothing_was_sent(sending: MailConfig):
    import smtplib

    def refuse(message: EmailMessage, config: MailConfig) -> None:
        raise smtplib.SMTPRecipientsRefused({"bob@company.example": (550, b"no")})

    outcome = await MailSend(sending, sender=refuse).run(
        message()
    )
    assert "Nothing was sent" in (outcome.failed or "")


async def test_too_many_recipients_is_not_a_recipient_list(sending: MailConfig):
    outbox = Outbox()
    crowd = ", ".join(f"person{n}@company.example" for n in range(30))
    outcome = await MailSend(sending, sender=outbox).run(
        message(to=crowd)
    )
    assert "not a recipient list" in (outcome.failed or "")
    assert outbox.sent == []

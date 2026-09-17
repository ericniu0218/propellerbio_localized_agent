"""Writing mail, and - separately, reluctantly - sending it.

Two capabilities, because drafting and sending are not two settings of one
verb::

    [agents.capabilities.mail]
    from_address = "alice@company.example"
    draft_dir = "~/work/agent-drafts"
    enabled = false                              # mail.send stays off until this is true
    allowed_recipient_domains = ["company.example"]
    smtp_host = "smtp.internal"
    smtp_port = 587
    smtp_username = "alice"
    smtp_password_env = "BEDROCK_SMTP_PASSWORD"  # the value stays out of the file
    smtp_starttls = true
    timeout = 30.0

``mail.draft`` is ``CHANGES_THINGS``: it writes an ordinary ``.eml`` file that
the employee can open, read, fix and send themselves. Nothing leaves the
machine.

``mail.send`` is ``REACHES_PEOPLE`` and is the single riskiest verb in this
product. An email cannot be unsent, cannot be corrected, and arrives with the
employee's name on it. It therefore has four locks, and all four must be open:

1. a ``Grant`` that names ``mail.send``, checked by the supervisor before this
   capability is called at all,
2. ``enabled = true`` in the config file - **it ships false**,
3. an ``smtp_host``,
4. every recipient inside ``allowed_recipient_domains``, which is **empty by
   default and empty means nobody**. A company that wants an agent sending mail
   has to write down where it may send. That is a sentence somebody has to
   type, on purpose, in a file, which is the point.

Address syntax is checked for both capabilities, and a newline in a header is
refused outright: that is how one "To:" line becomes a Bcc nobody approved.
"""

from __future__ import annotations

import asyncio
import re
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.message import EmailMessage
from email.utils import format_datetime, getaddresses, parseaddr
from pathlib import Path
from typing import Any, Callable, Mapping

from bedrock.agents.capability_support import (
    CapabilityConfigError,
    as_bool,
    as_float,
    as_int,
    as_list,
    as_path,
    as_text,
    capability_config,
    create_without_replacing,
    done,
    failed,
    secret_from_env,
)
from bedrock.agents.contract import Outcome, Risk
from bedrock.agents.protocol import Capability

DEFAULT_DRAFT_DIR = Path.home() / ".local" / "share" / "bedrock" / "agents" / "drafts"
MAX_RECIPIENTS = 25
MAX_SUBJECT_CHARS = 300

_ADDRESS = re.compile(r"^[^@\s,;<>]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")


# ---- Configuration -----------------------------------------------------------


@dataclass(frozen=True)
class MailConfig:
    from_address: str = ""
    draft_dir: Path = DEFAULT_DRAFT_DIR
    enabled: bool = False
    allowed_recipient_domains: tuple[str, ...] = ()
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_starttls: bool = True
    timeout: float = 30.0

    @classmethod
    def from_settings(
        cls, settings: Any = None, *, env: Mapping[str, str] | None = None
    ) -> "MailConfig":
        table = capability_config("mail", settings)
        return cls(
            from_address=as_text(table, "from_address"),
            draft_dir=as_path(table, "draft_dir") or DEFAULT_DRAFT_DIR,
            enabled=as_bool(table, "enabled", False),
            allowed_recipient_domains=tuple(
                domain.lower().lstrip("@").lstrip(".")
                for domain in as_list(table, "allowed_recipient_domains")
            ),
            smtp_host=as_text(table, "smtp_host"),
            smtp_port=as_int(table, "smtp_port", 587),
            smtp_username=as_text(table, "smtp_username"),
            smtp_password=secret_from_env(table, "smtp_password_env", env=env),
            smtp_starttls=as_bool(table, "smtp_starttls", True),
            timeout=as_float(table, "timeout", 30.0),
        )

    def sender(self) -> str:
        if not self.from_address:
            raise CapabilityConfigError(
                "No `from_address` is configured. Add it to [agents.capabilities.mail] "
                "in ~/.config/bedrock/config.toml."
            )
        return self.from_address


class MailRefused(Exception):
    """A message an agent asked for cannot be written or sent as asked."""


class _MailCapability:
    def __init__(self, config: MailConfig | None = None) -> None:
        self._config = config

    @property
    def config(self) -> MailConfig:
        if self._config is None:
            self._config = MailConfig.from_settings()
        return self._config


# ---- Building the message ----------------------------------------------------


def _clean_header(value: str, label: str) -> str:
    """One header value, with no way to add a second header to it.

    A newline in a subject or an address is how an agent that was allowed to
    mail one person mails a mailing list: everything after the newline becomes
    its own header. Control characters are refused rather than stripped,
    because silently changing what somebody approved is its own problem.
    """
    text = (value or "").strip()
    if _CONTROL.search(text):
        raise MailRefused(f"The {label} contains a line break or control character.")
    return text


def _addresses(raw: str, label: str) -> tuple[str, ...]:
    text = _clean_header(raw, label)
    if not text:
        return ()
    found: list[str] = []
    for _, address in getaddresses([text]):
        address = address.strip()
        if not address:
            continue
        if not _ADDRESS.match(address):
            raise MailRefused(f"`{address}` in the {label} is not an email address.")
        found.append(address)
    if len(found) > MAX_RECIPIENTS:
        raise MailRefused(
            f"{len(found)} addresses in the {label} is more than the limit of "
            f"{MAX_RECIPIENTS}. A mailing list is not a recipient list."
        )
    return tuple(found)


def build_message(arguments: Mapping[str, str], config: MailConfig) -> EmailMessage:
    """The message itself, with every header checked before it exists."""
    sender = parseaddr(config.sender())[1]
    if not _ADDRESS.match(sender):
        raise MailRefused(
            f"`from_address` is not an email address ({config.from_address!r})."
        )

    to = _addresses(str(arguments.get("to", "")), "To address")
    if not to:
        raise MailRefused("A message needs at least one address in `to`.")
    copies = _addresses(str(arguments.get("cc", "")), "Cc address")

    subject = _clean_header(str(arguments.get("subject", "")), "subject")
    if not subject:
        raise MailRefused("A message needs a `subject`.")

    message = EmailMessage()
    message["From"] = sender
    message["To"] = ", ".join(to)
    if copies:
        message["Cc"] = ", ".join(copies)
    message["Subject"] = subject[:MAX_SUBJECT_CHARS]
    message["Date"] = format_datetime(datetime.now().astimezone())
    message["X-Bedrock-Agent"] = "bedrock-private-ai"
    message.set_content(str(arguments.get("body", "")))
    return message


def recipients_of(message: EmailMessage) -> tuple[str, ...]:
    everyone = getaddresses(message.get_all("To", []) + message.get_all("Cc", []))
    return tuple(address for _, address in everyone if address)


def check_recipients_allowed(
    addresses: tuple[str, ...], allowed: tuple[str, ...]
) -> None:
    """Every recipient must be in a domain somebody wrote down."""
    if not allowed:
        raise MailRefused(
            "No recipient domains are allowed, so nothing is sent. Add "
            "`allowed_recipient_domains` to [agents.capabilities.mail] - it is empty by "
            "default, and empty deliberately means nobody."
        )
    for address in addresses:
        domain = address.rsplit("@", 1)[-1].lower()
        if not any(domain == entry or domain.endswith("." + entry) for entry in allowed):
            raise MailRefused(
                f"`{address}` is outside the domains this agent may mail "
                f"({', '.join(allowed)})."
            )


# ---- mail.draft --------------------------------------------------------------


class MailDraft(_MailCapability):
    """Write a message to a file the employee can open, read and send themselves."""

    name = "mail.draft"
    risk = Risk.CHANGES_THINGS

    def describe(self, arguments: dict[str, str]) -> str:
        to = str(arguments.get("to", "")).strip() or "(no recipient given)"
        subject = str(arguments.get("subject", "")).strip() or "(no subject given)"
        return f"Save a draft to {to}, subject {subject!r}. Nothing is sent."

    async def run(self, arguments: dict[str, str]) -> Outcome:
        try:
            config = self.config
            message = build_message(arguments, config)
        except (CapabilityConfigError, MailRefused) as exc:
            return failed(str(exc))

        target = config.draft_dir / _draft_name(message["Subject"])
        try:
            written = await asyncio.to_thread(
                create_without_replacing, target, bytes(message)
            )
        except PermissionError:
            return failed(f"No permission to write into {config.draft_dir}.")
        except OSError as exc:
            return failed(f"Could not save the draft: {exc.strerror or exc}")

        return done(
            (
                f"Drafted {message['Subject']!r} to {message['To']}.",
                f"Saved to {written}. Nothing was sent.",
            ),
            artifacts=(written,),
        )


def _draft_name(subject: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", subject).strip("-").lower()[:60] or "message"
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slug}.eml"


# ---- mail.send ---------------------------------------------------------------


def _send_over_smtp(message: EmailMessage, config: MailConfig) -> None:
    """The real sender. Blocking, so it is called in a thread."""
    with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=config.timeout) as server:
        if config.smtp_starttls:
            server.starttls()
        if config.smtp_username:
            server.login(config.smtp_username, config.smtp_password)
        server.send_message(message)


class MailSend(_MailCapability):
    """Send a message. Off by default; see this module's docstring for why."""

    name = "mail.send"
    risk = Risk.REACHES_PEOPLE

    def __init__(
        self,
        config: MailConfig | None = None,
        *,
        sender: Callable[[EmailMessage, MailConfig], None] | None = None,
    ) -> None:
        # ``sender`` is a code-only test seam. No config file can set it, and
        # no test in this repository is allowed to reach a real mail server.
        super().__init__(config)
        self._sender = sender or _send_over_smtp

    def describe(self, arguments: dict[str, str]) -> str:
        to = str(arguments.get("to", "")).strip() or "(no recipient given)"
        subject = str(arguments.get("subject", "")).strip() or "(no subject given)"
        return (
            f"SEND an email to {to}, subject {subject!r}. "
            "This leaves the company and cannot be taken back."
        )

    async def run(self, arguments: dict[str, str]) -> Outcome:
        try:
            config = self.config
            if not config.enabled:
                return failed(
                    "Sending mail is switched off. Nothing was sent. Set `enabled = true` "
                    "in [agents.capabilities.mail] if this company wants agents sending "
                    "email - it ships off on purpose.",
                )
            if not config.smtp_host:
                return failed(
                    "No `smtp_host` is configured, so nothing was sent.",
                )
            message = build_message(arguments, config)
            check_recipients_allowed(
                recipients_of(message), config.allowed_recipient_domains
            )
        except (CapabilityConfigError, MailRefused) as exc:
            return failed(str(exc))

        try:
            await asyncio.to_thread(self._sender, message, config)
        except smtplib.SMTPAuthenticationError:
            return failed(
                f"The mail server refused `{config.smtp_username}`. Nothing was sent.",
            )
        except (smtplib.SMTPException, OSError) as exc:
            return failed(f"Nothing was sent: {exc}")

        return done(
            (
                f"Sent {message['Subject']!r} to {message['To']}"
                + (f", copying {message['Cc']}" if message["Cc"] else "")
                + ".",
                "This has left the company and cannot be taken back.",
            ),
        )


CAPABILITY: tuple[Capability, ...] = (MailDraft(), MailSend())
"""What the roster finds in this file.

A tuple, which ``roster.discover_capability`` unpacks by name. One file
per adapter, several verbs per file where they share one set of rules.
"""

CAPABILITIES = CAPABILITY
"""Plural alias, for reading a module's contents rather than the roster's."""

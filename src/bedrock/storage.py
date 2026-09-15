"""Private atomic conversation files; only sidebar metadata is cached."""
from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bedrock.config import Settings
from bedrock.models.conversation import SCHEMA_VERSION, UNTITLED, Conversation, parse_time

DIR_MODE = 0o700
FILE_MODE = 0o600


class StorageError(Exception):
    """A recoverable failure to read or save history."""


class StorageConflict(StorageError):
    """Another writer changed this conversation after it was loaded."""


class ConversationDeleted(StorageError):
    """A stale window must not resurrect a forgotten conversation."""


@dataclass(frozen=True)
class ConversationSummary:
    id: str
    title: str
    model: str | None
    updated_at: datetime
    message_count: int
    generated_title: str | None = None
    excerpt: str = ""
    title_checked_at: int = 0


def _revision(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def _content_revision(data: dict) -> str:
    metadata = {"title", "generated_title", "title_checked_at", "updated_at", "excerpt", "family_id"}
    return _revision({key: value for key, value in data.items() if key not in metadata})


def _family_id(data: dict) -> str:
    if data.get("family_id"):
        return str(data["family_id"])
    # Older copies have no parent ID, but retain the original creation time
    # and first message including its timestamp. Never group by title alone.
    messages = data.get("messages")
    if data.get("created_at") and isinstance(messages, list) and messages and isinstance(messages[0], dict) and messages[0].get("timestamp"):
        return "family-" + _revision({"created_at": data["created_at"], "first_message": messages[0]})
    return str(data["id"])


class ConversationStore:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._summaries: dict[str, ConversationSummary] = {}
        self._fingerprints: dict[str, tuple[int, int, int]] = {}
        self._mutex = threading.RLock()
        self._cleaned = False
        self._listed_signature: tuple[int, int, int] | None = None

    @contextmanager
    def _directory_lock(self):
        """Fallback for filesystems with atomic mkdir but no advisory locks.

        Never break a possibly active lock based on its age. A writer killed
        during this fallback leaves a lock that must be removed by its operator.
        """
        path = self.directory / ".history-writer"
        deadline = time.monotonic() + 2
        while True:
            try:
                path.mkdir(mode=DIR_MODE)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise StorageError("History is locked by another writer (.history-writer). If no writer is running, remove that empty lock directory.")
                time.sleep(0.02)
        try:
            yield
        finally:
            path.rmdir()

    @contextmanager
    def _writer(self):
        """Serialize revision checks, writes and deletes across processes."""
        with self._mutex:
            self.directory.mkdir(mode=DIR_MODE, parents=True, exist_ok=True)
            try:
                os.chmod(self.directory, DIR_MODE)
            except OSError as exc:
                if exc.errno not in {errno.EPERM, errno.EACCES, errno.ENOTSUP, errno.EOPNOTSUPP}:
                    raise
            descriptor = os.open(self.directory / ".history.lock", os.O_CREAT | os.O_RDWR, FILE_MODE)
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX)
                except OSError as exc:
                    if exc.errno not in {errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.ENOLCK}:
                        raise
                    with self._directory_lock():
                        self._cleanup_temps()
                        yield
                else:
                    self._cleanup_temps()
                    yield
            finally:
                os.close(descriptor)

    def _cleanup_temps(self) -> None:
        # All writers hold the lock: matching temp files cannot still be active.
        if self._cleaned:
            return
        for path in self.directory.iterdir():
            if re.fullmatch(r"(?:[A-Za-z0-9_-]+\.json|\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)\.tmp", path.name):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass
        self._cleaned = True

    def save(self, conversation: Conversation, touch: bool = True) -> Path:
        if conversation.source_version > SCHEMA_VERSION:
            raise StorageError("This conversation was written by a newer app version and is read-only. Use /new to chat.")
        path = self.path_for(conversation.id)
        temporary = None
        try:
            with self._writer():
                self.ensure_available(conversation)
                if path.exists():
                    disk = self._read(path)
                    if _revision(disk) != conversation.storage_revision:
                        if _content_revision(disk) != conversation.storage_content_revision:
                            raise StorageConflict("This conversation changed in another window.")
                        # The remote transcript is unchanged. Carry forward
                        # its title/check result while saving our new turns.
                        if conversation.generated_title == conversation.storage_title:
                            title = disk.get("generated_title")
                            conversation.generated_title = title if isinstance(title, str) else None
                        checked = disk.get("title_checked_at")
                        if isinstance(checked, int):
                            conversation.title_checked_at = max(conversation.title_checked_at, checked)
                        if not touch:
                            conversation.updated_at = max(conversation.updated_at, parse_time(disk.get("updated_at")))
                conversation.family_id = conversation.family_id or conversation.id
                if touch:
                    conversation.touch()
                data = conversation.to_dict()
                descriptor, name = tempfile.mkstemp(prefix=f".{conversation.id}.", suffix=".tmp", dir=self.directory)
                temporary = Path(name)
                # mkstemp restricts permissions before any content is written.
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    json.dump(data, handle, ensure_ascii=True, indent=2)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                conversation.storage_revision = _revision(data)
                conversation.storage_content_revision = _content_revision(data)
                conversation.storage_title = conversation.generated_title
                self._summaries[conversation.id] = self._summary(data, conversation.id)
                self._fingerprints[conversation.id] = self._fingerprint(path)
        except (OSError, ValueError, TypeError) as exc:
            raise StorageError(f"Could not save the conversation: {exc}") from exc
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        return path

    def delete(self, conversation_id: str, *, family_id: str | None = None) -> bool:
        path = self.path_for(conversation_id)
        try:
            with self._writer():
                family = _family_id(self._read(path)) if path.exists() else (family_id or conversation_id)
                members = {conversation_id}
                # Forget is an explicit history operation: read candidates only
                # here, and retain their IDs rather than their message bodies.
                for candidate in self.directory.glob("*.json"):
                    try:
                        self.path_for(candidate.stem)
                        data = self._read(candidate)
                        if data.get("id") == candidate.stem and _family_id(data) == family:
                            members.add(candidate.stem)
                    except StorageError:
                        continue
                self.path_for(family)  # Validate before writing the family marker.
                for identifier in members | {family}:
                    descriptor = os.open(self._tombstone(identifier), os.O_CREAT | os.O_WRONLY, FILE_MODE)
                    os.close(descriptor)
                existed = False
                for identifier in members:
                    member = self.path_for(identifier)
                    existed = member.exists() or existed
                    member.unlink(missing_ok=True)
                    self._summaries.pop(identifier, None)
                    self._fingerprints.pop(identifier, None)
                return existed
        except OSError as exc:
            raise StorageError(f"Could not delete the conversation: {exc}") from exc

    def load(self, conversation_id: str, system_prompt: str | None = None) -> Conversation:
        try:
            if self._tombstone(conversation_id).exists():
                raise ConversationDeleted("This conversation has been deleted.")
            data = self._read(self.path_for(conversation_id))
            if data.get("id", conversation_id) != conversation_id:
                raise StorageError("The saved conversation ID does not match its filename.")
            conversation = Conversation.from_dict(data, system_prompt=system_prompt)
            conversation.id = conversation_id
            conversation.family_id = _family_id(data | {"id": conversation_id})
            self.ensure_available(conversation)
            conversation.storage_revision = _revision(data)
            conversation.storage_content_revision = _content_revision(data)
            conversation.storage_title = conversation.generated_title
            return conversation
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"Could not load conversation {conversation_id}: {exc}") from exc

    def latest(self, system_prompt: str | None = None) -> Conversation | None:
        for summary in self.list():
            try:
                return self.load(summary.id, system_prompt=system_prompt)
            except StorageError:
                continue
        return None

    def list(self, limit: int | None = None) -> list[ConversationSummary]:
        """Read changed summaries, never retaining message bodies.

        Fingerprints are per-file, so local writes cannot hide external edits.
        Legacy JSON files require one parse to extract their summary.
        """
        with self._mutex:
            seen = set()
            if self.directory.exists() and not self._cleaned:
                try:
                    with self._writer():
                        pass
                except (OSError, StorageError):
                    self._cleaned = True  # A busy/read-only store still permits browsing.
            signature = self._directory_signature()
            for path in self.directory.glob("*.json"):
                try:
                    self.path_for(path.stem)
                    if self._tombstone(path.stem).exists():
                        continue
                    seen.add(path.stem)
                    fingerprint = self._fingerprint(path)
                    if fingerprint == self._fingerprints.get(path.stem):
                        continue
                    data = self._read(path)
                    if data.get("id", path.stem) != path.stem or not isinstance(data.get("messages"), list):
                        raise ValueError("Invalid conversation")
                    data.setdefault("id", path.stem)
                    if self._tombstone(_family_id(data)).exists():
                        raise ValueError("Deleted conversation family")
                    self._summaries[path.stem] = self._summary(data, path.stem)
                    self._fingerprints[path.stem] = fingerprint
                except (StorageError, OSError, ValueError, TypeError):
                    self._summaries.pop(path.stem, None)
                    self._fingerprints.pop(path.stem, None)
            for stale in self._summaries.keys() - seen:
                self._summaries.pop(stale, None)
                self._fingerprints.pop(stale, None)
            summaries = sorted(self._summaries.values(), key=lambda item: item.updated_at, reverse=True)
            # Use the signature from BEFORE the scan. Local writes must never
            # acknowledge and hide a concurrent external directory change.
            self._listed_signature = signature
            return summaries[:limit] if limit is not None else summaries

    def ensure_available(self, conversation: Conversation) -> None:
        family = conversation.family_id or _family_id(conversation.to_dict())
        if (self._tombstone(conversation.id).exists() or self._tombstone(family).exists()
                or (conversation.storage_revision and not self.path_for(conversation.id).exists())):
            raise ConversationDeleted("This conversation was deleted in another window. Use /new to continue.")

    def directory_changed(self) -> bool:
        """One directory stat for idle polling, with no per-chat file checks."""
        return self._directory_signature() != self._listed_signature

    def _directory_signature(self) -> tuple[int, int, int] | None:
        try:
            stat = self.directory.stat()
            return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino
        except FileNotFoundError:
            return None

    @staticmethod
    def _fingerprint(path: Path) -> tuple[int, int, int]:
        stat = path.stat()
        return stat.st_mtime_ns, stat.st_size, stat.st_ino

    @staticmethod
    def _summary(data: dict, conversation_id: str) -> ConversationSummary:
        title = data.get("generated_title")
        title = title.strip() if isinstance(title, str) else None
        checked = data.get("title_checked_at")
        return ConversationSummary(
            id=conversation_id, title=title or UNTITLED,
            model=data.get("model") if isinstance(data.get("model"), str) else None,
            updated_at=parse_time(data.get("updated_at")),
            message_count=len(data["messages"]), generated_title=title or None,
            excerpt=str(data.get("excerpt") or data.get("title") or ""),
            title_checked_at=checked if isinstance(checked, int) and checked > 0 else 0,
        )

    def _tombstone(self, conversation_id: str) -> Path:
        return self.path_for(conversation_id).with_suffix(".deleted")

    def path_for(self, conversation_id: str) -> Path:
        if not isinstance(conversation_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", conversation_id):
            raise StorageError(f"Not a usable conversation id: {conversation_id!r}")
        return self.directory / f"{conversation_id}.json"

    def _read(self, path: Path) -> dict:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StorageError(f"Could not read {path.name}: {exc}") from exc
        if not isinstance(data, dict):
            raise StorageError(f"{path.name} does not contain a conversation.")
        return data


def create_store(settings: Settings) -> ConversationStore | None:
    return ConversationStore(settings.history_dir) if settings.save_history else None

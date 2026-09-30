"""Audio file in, meeting notes out — all of it on this machine.

The heavy lifting is bedrock's own meeting engine (bedrock.meetings), which the
terminal app's `bedrock-meeting` command uses for Zoom's *local* recordings.
This module reuses it rather than keeping a second copy, as
`src/bedrock/server/tools/meeting_transcripts.py` does with the VTT parser: one
implementation, so a fix reaches both.

What this module adds is the part that was missing — a recording that did not
come from Zoom. Browser audio arrives as WebM/Opus, gets normalised to the
16 kHz mono PCM whisper wants, and comes back as WEBVTT plus notes.

**Speakers are worked out, not known.** Zoom's local recordings can write a
separate file per participant, which is where bedrock-private-ai's perfect
speaker labels come from — every word in Bob's file was said by Bob, by
construction. A microphone in a room has no such thing to offer, so the voices
have to be told apart from the audio itself, in `diarize.py`. That is a good
deal better than nothing and a good deal worse than Zoom's per-participant
tracks, so the labels are provisional, renameable, and the meeting page says
how they were arrived at. Without the diarisation models installed a recording
transcribes exactly as before and is reported as unattributed.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path

from bedrock.server.meetings.config import RecordingConfig, readiness_issues
from bedrock.meetings.diarize import assign, consolidate, diarize, speaker_change_points
from bedrock.meetings.enrollment import cluster_embeddings, identify, rescore_cues
from bedrock.meetings.vocabulary import VocabularyReport, correct_texts

logger = logging.getLogger(__name__)

# The transcription engine is bedrock's own (bedrock.meetings), shipped in the
# same package, so it is always importable. Only its optional speaker and
# screen-reading extras are guarded, where they are used.
from bedrock.meetings.summarize import suggest_title, translate_cues, MeetingSummary, summarize_meeting
from bedrock.meetings.transcribe import (
    WhisperConfig,
    WhisperTranscriber,
    to_vtt,
)
from bedrock.meetings.transcript import Cue


class RecordingError(RuntimeError):
    """Raised when a recording cannot be turned into a transcript."""


@dataclass
class TranscriptionOutcome:
    """Everything one recording produced."""

    vtt: str
    cues: list["Cue"]
    duration_ms: int
    summary: "MeetingSummary | None" = None
    attributed: bool = False
    """True once speakers have been told apart. One mixed track carries no
    attribution of its own, so this is whether diarisation succeeded."""
    speaker_count: int = 0
    vocabulary: VocabularyReport = field(default_factory=VocabularyReport)
    """Which mis-heard words were repaired against the configured terms."""
    notes: list[str] = field(default_factory=list)
    """Plain-language caveats worth showing the person who pressed Record."""
    language: str | None = None
    """What whisper ran as: a language code, or ``auto`` for per-turn detection."""
    languages_heard: dict[str, float] = field(default_factory=dict)
    """Share of sampled speech per detected language, when detection ran."""
    suggested_title: str | None = None
    """A name taken from the content, for a meeting that was not given one."""
    translations: dict[int, str] = field(default_factory=dict)
    """English for cues spoken in a CJK language, by cue index."""

    @property
    def is_empty(self) -> bool:
        return not self.cues


from bedrock.meetings.transcribe import MANDARIN_PROMPT  # noqa: E402  (re-exported for tests)


_GENERIC_LABEL = re.compile(r"^Speaker \d+$")
NEIGHBOUR_GAP_MS = 2000
PHANTOM_MAX_MS = 3000


def tidy_speakers(cues: list["Cue"], gap_ms: int = NEIGHBOUR_GAP_MS) -> tuple[list["Cue"], int]:
    """Contiguous labels, and a label for the lines that had none.

    Clusters that were dropped as noise leave holes in the numbering, so a
    two-person meeting read "Speaker 1, Speaker 2, Speaker 5". Labels are
    renumbered in order of first appearance; a real name from enrolment or
    the meeting window is not a generic label and is left alone.

    A cue the diariser did not cover — too short to segment, or spoken over
    somebody — is given to whoever spoke just before it, or failing that just
    after, when that neighbour is within ``gap_ms``. Further away than that it
    stays unlabelled: a line ten seconds from anything is not obviously
    anyone's.

    Returns the cues and the number of distinct speakers left.
    """
    from dataclasses import replace

    # A generic speaker with one short line is not a person; it is a cough, a
    # laugh or a word of overlap that clustered on its own. On a real
    # two-person recording the diariser produced "Speaker 5" for two such
    # lines. That line goes to whoever spoke just before it. A named voice
    # is never folded, and neither is anyone with two lines or a long one.
    counts: dict[str, int] = {}
    for cue in cues:
        if cue.speaker and _GENERIC_LABEL.match(cue.speaker):
            counts[cue.speaker] = counts.get(cue.speaker, 0) + 1
    phantom = {label for label, n in counts.items() if n == 1}
    if phantom and len(counts) > len(phantom):
        folded: list["Cue"] = []
        for cue in cues:
            if (cue.speaker in phantom and cue.end_ms - cue.start_ms <= PHANTOM_MAX_MS
                    and folded and folded[-1].speaker):
                cue = replace(cue, speaker=folded[-1].speaker)
            folded.append(cue)
        cues = folded

    order: dict[str, str] = {}
    for cue in cues:
        if cue.speaker and _GENERIC_LABEL.match(cue.speaker) and cue.speaker not in order:
            order[cue.speaker] = f"Speaker {len(order) + 1}"
    tidied = [
        replace(cue, speaker=order[cue.speaker]) if cue.speaker in order else cue
        for cue in cues
    ]

    filled: list["Cue"] = []
    for index, cue in enumerate(tidied):
        if cue.speaker:
            filled.append(cue)
            continue
        before = filled[index - 1] if index else None
        after = next((c for c in tidied[index + 1:] if c.speaker), None)
        if before is not None and before.speaker and cue.start_ms - before.end_ms <= gap_ms:
            filled.append(replace(cue, speaker=before.speaker))
        elif after is not None and after.start_ms - cue.end_ms <= gap_ms:
            filled.append(replace(cue, speaker=after.speaker))
        else:
            filled.append(cue)
    return filled, len({c.speaker for c in filled if c.speaker})


def _speakers(n: int) -> str:
    return "1 speaker was" if n == 1 else f"{n} speakers were"


def english_vtt(outcome: "TranscriptionOutcome") -> str | None:
    """The transcript with translated lines in English, same timing and
    speakers, or None when nothing was translated."""
    if not outcome.translations:
        return None
    cues = [replace(c, text=outcome.translations.get(i, c.text)) for i, c in enumerate(outcome.cues)]
    return to_vtt(cues)[:100_000]


def describe_language(decision) -> str:
    """One line for the meeting page on what was heard and how it was run."""
    names = {"en": "English", "zh": "Mandarin", "es": "Spanish", "ja": "Japanese", "ko": "Korean",
             "de": "German", "fr": "French", "pt": "Portuguese", "hi": "Hindi"}
    heard = ", ".join(f"{names.get(code, code)} {round(share * 100)}%" for code, share in decision.heard.items())
    if decision.language == "auto":
        return (f"More than one language was heard ({heard}), so each speaker turn was "
                "transcribed in its own language.")
    return f"Transcribed as {names.get(decision.language, decision.language)} (detected: {heard})."


# whisper reads at most n_text_ctx/2 = 224 tokens of prompt and, when handed
# more, keeps the LAST 224 — so an over-long hint loses its front, which is
# the Mandarin script hint and the names the pack put first on purpose. The
# budget is therefore conservative, and the estimate deliberately pessimistic
# for the tokens whisper's BPE actually produces on jargon: "IL-17A" is four
# tokens, "dupilumab" three or four, a CJK character one or two. Everything
# past the cut still repairs near-misses; it just does not steer the decoder.
HINT_TOKEN_BUDGET = 170
_CJK = re.compile("[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]")


def whisper_tokens(text: str) -> int:
    """A pessimistic count of whisper BPE tokens for a hint."""
    cjk = len(_CJK.findall(text))
    latin = _CJK.sub("", text)
    pieces = re.findall(r"[A-Za-z]+|\d+|[^\sA-Za-z\d]", latin)
    return 2 * cjk + sum(-(-len(piece) // 3) for piece in pieces)


def hint_within_budget(vocabulary: str | None, budget: int = HINT_TOKEN_BUDGET) -> str | None:
    if not vocabulary:
        return None
    kept, used = [], 0
    for term in (t.strip() for t in re.split(r"[,;\n]", vocabulary)):
        if not term:
            continue
        cost = whisper_tokens(term) + 1  # the separator
        if used + cost > budget:
            break
        kept.append(term)
        used += cost
    return ", ".join(kept) or None


def _initial_prompt(config: RecordingConfig) -> str | None:
    """The vocabulary, cut to whisper's prompt budget. The transcriber adds
    the Mandarin script hint itself once it knows the recording is Mandarin,
    whether chosen or detected."""
    return hint_within_budget(config.vocabulary)


def _whisper_config(config: RecordingConfig) -> "WhisperConfig":
    return WhisperConfig(
        model=config.speech_model,
        vad_model=config.vad_model,
        threads=config.threads,
        language=config.language,
        initial_prompt=_initial_prompt(config),
        normalize_audio=config.normalize_audio,
        vad_threshold=config.vad_threshold,
        vad_speech_pad_ms=config.vad_speech_pad_ms,
        models_by_language={"zh": config.speech_model_zh} if config.speech_model_zh else {},
    )


async def transcribe(
    audio: Path,
    config: RecordingConfig,
    work_dir: Path,
    on_progress=None,
    expected_speakers: int | None = None,
    minimum_speakers: int | None = None,
    voice_prints: list | None = None,
    screen=None,
) -> TranscriptionOutcome:
    """Turn one recorded audio file into timestamped cues and WEBVTT.

    `on_progress` is an optional callable taking a short status string. It is
    how the browser gets told what is happening during the minutes a long
    meeting spends in whisper, instead of watching a spinner and assuming the
    thing has hung.
    """
    issues = readiness_issues(config)
    if issues:
        raise RecordingError(
            "This server is not set up to transcribe recordings yet:\n- "
            + "\n- ".join(issues)
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    transcriber = WhisperTranscriber(_whisper_config(config))

    if on_progress:
        on_progress("Converting audio")
    try:
        wav = await transcriber.to_wav(audio, work_dir)
    except Exception as exc:
        raise RecordingError(
            "The recording could not be decoded. It may have been cut off "
            f"before the browser finished writing it. ({exc})"
        ) from exc

    duration_ms = await transcriber.duration_ms(wav)

    # Read once, here. Diarisation, the cluster vectors and the cue re-scoring
    # each used to open the file for themselves — three full passes, and about
    # 1.8 GB of float32 apiece for an eight-hour recording. They now share this.
    # Every heavy step below runs in a worker thread. They are CPU-bound and
    # synchronous — ONNX inference and a whole-file decode — and on the event
    # loop they block the server outright: during a forty-minute meeting even
    # signing in timed out, and an eight-hour one would be unreachable for
    # minutes at a time while the person who recorded it watches a dead page.
    samples = None
    try:
        import soundfile

        def _read():
            decoded, _rate = soundfile.read(str(wav), dtype="float32", always_2d=True)
            return decoded[:, 0]

        samples = await asyncio.to_thread(_read)
    except Exception as exc:
        logger.warning("Could not preload %s; each step will read it itself: %s", wav.name, exc)

    # Speakers are worked out *before* transcription, not after, because the
    # cue is the unit a speaker label attaches to. Whisper cuts on silence, so
    # two people talking without a gap land in one cue, and labelling that cue
    # afterwards can only pick a winner — on a real recording it produced one
    # nineteen-second cue holding both sides of "What's your name? My name's
    # Luke". Handing the voice-change offsets to the slicer instead means each
    # cue holds one person to begin with.
    if on_progress:
        on_progress("Identifying speakers")
    speakers = diarize(
        wav,
        config.segmentation_model,
        config.embedding_model,
        expected_speakers=expected_speakers,
        threads=min(config.threads, 4),
    )
    # Nobody said how many people were here, which is the normal case — you do
    # not always know, and a club meeting does not have a guest list. Raw
    # clustering splinters badly on real audio (17 speakers on a genuine
    # three-person call), so clusters that are the same voice are merged back
    # together before anything else uses them.
    if expected_speakers is None and speakers.ok:
        try:
            vectors = await asyncio.to_thread(
                cluster_embeddings, wav, speakers.turns, config.embedding_model,
                min(config.threads, 4), samples,
            )
            speakers = consolidate(speakers, vectors, minimum_speakers=minimum_speakers)
        except Exception as exc:
            logger.warning("Speaker consolidation failed, using raw clusters: %s", exc)

    boundaries = speaker_change_points(speakers)

    if on_progress:
        on_progress("Transcribing speech")
    try:
        # Run it as a task so the stage can be updated while it works. A
        # forty-minute meeting spends ten minutes or more in here, and a label
        # that has not changed in ten minutes is indistinguishable from a hung
        # one — which cost real time diagnosing a job that was working fine.
        work = asyncio.create_task(
            transcriber.transcribe_track(wav, work_dir, split_at_ms=boundaries)
        )
        if on_progress:
            await _report_progress(work, work_dir, on_progress)
        cues = await work
        decision = transcriber.last_language
    except Exception as exc:
        # An empty recording is a real, ordinary outcome — somebody pressed
        # Record on a muted microphone. It is reported as an empty transcript
        # with an explanation, not as a server error, because the person needs
        # to know their meeting was not captured.
        message = str(exc)
        if "no transcript" in message.lower():
            return TranscriptionOutcome(
                vtt="WEBVTT\n",
                cues=[],
                duration_ms=duration_ms,
                notes=[
                    "No speech was found in this recording. Check that the "
                    "right microphone was selected and that it was not muted."
                ],
            )
        raise RecordingError(f"Transcription failed: {exc}") from exc

    # Whisper's vocabulary hint is a bias, not a guarantee: the same audio gave
    # "KRAS" on one run and "CRAS" on the next. Repairing the near-misses here,
    # before anything reads the cues, means the notes and the transcript agree
    # and retrieval can match the word somebody actually said.
    repaired, vocabulary_report = correct_texts([cue.text for cue in cues], config.vocabulary)
    cues = [
        cue if cue.text == text else replace(cue, text=text)
        for cue, text in zip(cues, repaired)
    ]

    # Names come from three places, in order of how much they are worth.
    #
    # First the meeting window itself, if this was an online call: every client
    # draws participant names and rings whoever is speaking, so that is not an
    # inference at all — it is what the software displayed at the time. It
    # needs no enrolment and no guess about how many people are there.
    names: dict[int, str] = {}
    if screen is not None and getattr(screen, "ok", False) and speakers.ok:
        try:
            names = screen.name_clusters(speakers.turns)
        except Exception as exc:
            logger.warning("Could not name speakers from the meeting window: %s", exc)

    # Then enrolled voices, for whoever the screen did not account for — an
    # in-person meeting, or somebody whose tile was never highlighted.
    if voice_prints and speakers.ok:
        try:
            from_voice = identify(
                await asyncio.to_thread(
                    cluster_embeddings, wav, speakers.turns, config.embedding_model,
                    min(config.threads, 4), samples,
                ),
                voice_prints,
            )
            taken = set(names.values())
            for speaker, name in from_voice.items():
                if speaker not in names and name not in taken:
                    names[speaker] = name
                    taken.add(name)
        except Exception as exc:  # never lose a meeting over a name
            logger.warning("Voice-print matching failed: %s", exc)

    cues, speaker_note, speaker_count = _apply_speakers(
        cues, speakers, names, from_screen=bool(screen is not None and getattr(screen, 'ok', False))
    )

    # Then, where the meeting window saw who was speaking, let it overrule the
    # label a cue inherited from its cluster. Clustering is the hard part and
    # the screen is the certain one, so a line the window can account for
    # should not be at the mercy of how well the voices separated.
    if screen is not None and getattr(screen, "ok", False):
        try:
            direct = screen.name_cues(cues)
            for index, name in direct.items():
                cues[index] = replace(cues[index], speaker=name)
            if direct:
                named = len({n for n in direct.values()})
                speaker_note = (
                    f"{len(direct)} of {len(cues)} lines were named directly from the "
                    f"meeting window ({named} people). Anything left is from the audio."
                )
                speaker_count = max(speaker_count, named)
        except Exception as exc:
            logger.warning("Could not name lines from the meeting window: %s", exc)

    # A cue near a handover inherits whichever cluster won the overlap, which
    # is where most misattribution comes from. Cues long enough to embed on
    # their own get a second opinion from their own audio.
    if names and voice_prints:
        try:
            changes = await asyncio.to_thread(
                rescore_cues, wav, cues, voice_prints, config.embedding_model,
                min(config.threads, 4), 0.08, samples,
            )
            for index, name in changes.items():
                cues[index] = replace(cues[index], speaker=name)
            if changes:
                speaker_note += f" {len(changes)} line(s) were reassigned on a closer listen."
        except Exception as exc:
            logger.warning("Cue re-scoring failed: %s", exc)

    # Folding a one-line phantom can lower the count after the note above
    # was written; the note must say the number the page shows.
    before = speaker_count
    cues, speaker_count = tidy_speakers(cues)
    if speaker_count != before and speaker_note.startswith(_speakers(before)):
        speaker_note = _speakers(speaker_count) + speaker_note[len(_speakers(before)):]

    outcome = TranscriptionOutcome(
        vtt=to_vtt(cues),
        cues=cues,
        duration_ms=duration_ms,
        attributed=speaker_count > 0,
        speaker_count=speaker_count,
        vocabulary=vocabulary_report,
        language=decision.language if decision else None,
        languages_heard=dict(decision.heard) if decision else {},
        notes=[speaker_note],
    )
    if decision and decision.sampled:
        outcome.notes.append(describe_language(decision))
    if vocabulary_report.total:
        outcome.notes.append(
            f"Corrected {vocabulary_report.total} mis-heard word(s) against the "
            f"configured vocabulary: {vocabulary_report.describe().split(': ', 1)[1]}."
        )
    if not cues:
        outcome.notes.append("No speech was found in this recording.")
    return outcome


async def _report_progress(work, work_dir: Path, on_progress) -> None:
    """Count finished slices while whisper runs, so the stage keeps moving.

    Whisper writes one `.vtt` beside each slice as it finishes it, and the
    slices are cut before any of them are transcribed — so the two counts are
    a real fraction of the work, not an invented one. Nothing here can fail
    the transcription: it only reads the directory, and stops the moment the
    real work does.
    """
    last = ""
    while not work.done():
        try:
            await asyncio.wait_for(asyncio.shield(work), timeout=5)
        except asyncio.TimeoutError:
            pass
        except Exception:
            return  # the real await below will surface it
        try:
            slices = list(work_dir.glob("*.slices/*.wav"))
            done = list(work_dir.glob("*.slices/*.vtt"))
        except OSError:
            continue
        if slices:
            stage = f"Transcribing speech — {len(done)} of {len(slices)} segments"
            if stage != last:
                on_progress(stage)
                last = stage


def _apply_speakers(cues, result, names=None, from_screen=False):
    """Label the cues with who said them, or explain why nobody could be.

    Returns the cues, one plain-language note for the person who recorded, and
    how many speakers were found. Never raises: a meeting whose voices could
    not be told apart is still worth keeping, and the transcript is the part
    that cannot be recreated.
    """
    if not result.ok:
        return (
            cues,
            "Speakers are not identified. "
            + (result.note or "This was recorded as a single mixed audio track.")
            + " The transcript records what was said, not who said it.",
            0,
        )

    labelled, counts = assign(cues, result, names)
    found = len(counts)
    if result.estimated:
        note = (
            f"{_speakers(found)} identified from the audio. Where people talk "
            "over each other this can split one person across two labels — if "
            "two of these are the same person, rename them to the same name and "
            "they merge."
        )
    elif names:
        recognised = sum(1 for speaker in counts if speaker in names)
        source = "read from the meeting window" if from_screen else "matched to a saved voice"
        note = (
            f"{_speakers(found)} identified, {recognised} of them {source}. "
            "Any still labelled 'Speaker N' can be renamed here."
        )
    else:
        note = (
            f"{_speakers(found)} identified. Labels are provisional — rename "
            "them on this page, or save a voice sample so they are named "
            "automatically next time."
        )
    return labelled, note, found


async def summarise(
    outcome: TranscriptionOutcome,
    config: RecordingConfig,
    *,
    title: str,
    meeting_date: date | None = None,
    notes: str | None = None,
    on_progress=None,
    name_it: bool = False,
) -> TranscriptionOutcome:
    """Write the notes, and keep going if the model is unavailable.

    `name_it` is for a meeting nobody typed a name for: once the notes exist a
    short title is taken from them, and the dated placeholder is kept only if
    that fails.

    `summarize_meeting` already degrades to computed facts — duration, word
    count, speaker count — when inference is down, and sets `degraded_reason`
    to say why. That behaviour is the whole reason a meeting is never lost to a
    model outage, so this function must not turn a degraded summary into an
    error. The transcript is the thing that cannot be recreated; the notes can
    always be rewritten later.
    """
    if outcome.is_empty:
        return outcome

    if on_progress:
        on_progress("Writing notes")

    # One queue per model server for the whole process (bedrock.server.models.hub):
    # notes wait behind people's chats on the same server instead of competing
    # with them, and are sized to the notes model's own window rather than a
    # laptop's. `notes` in models.yaml reads the same MEETING_SUMMARY_* variables.
    from bedrock.server.config import load_models_config
    from bedrock.server.models.hub import get_model_hub

    notes_model = load_models_config().for_role("notes")
    try:
        outcome.summary = await summarize_meeting(
            outcome.cues,
            scheduler=get_model_hub().scheduler_for("notes"),
            model=notes_model.model_name,
            title=title,
            meeting_date=meeting_date,
            notes=notes,
        )
        if outcome.summary is not None and outcome.summary.degraded_reason is None:
            try:
                if on_progress:
                    on_progress("Translating")
                outcome.translations = await translate_cues(
                    get_model_hub().scheduler_for("notes"), notes_model.model_name, outcome.cues
                )
            except Exception as exc:  # the original transcript is the record; English is a courtesy
                logger.info("No translation for %r: %s", title, exc)
        if name_it and outcome.summary is not None and outcome.summary.degraded_reason is None:
            try:
                outcome.suggested_title = await suggest_title(
                    get_model_hub().scheduler_for("notes"), notes_model.model_name, outcome.summary
                )
            except Exception as exc:  # the dated name is fine; never fail the job for this
                logger.info("No title suggested for %r: %s", title, exc)
    except Exception as exc:
        # Deliberately broad: a transcript that exists must be saved even if
        # the summariser fails in a way its own degradation path did not catch.
        logger.warning("Meeting summarisation failed for %r: %s", title, exc)
        outcome.notes.append(
            "The transcript was saved, but notes could not be written because "
            f"the model at {notes_model.endpoint} did not answer. "
            "You can ask about this meeting in chat regardless."
        )
        return outcome

    if outcome.summary and outcome.summary.degraded_reason:
        outcome.notes.append(
            f"Notes are limited to computed facts: {outcome.summary.degraded_reason}."
        )
    return outcome


# A heading in the rewritten notes, in either style the model uses: "# Topic"
# or a line that is entirely bold. Both name the section rather than say
# anything, so as a card subtitle they repeat the title and tell the reader
# nothing.
_MARKDOWN_HEADING = re.compile(r"^\s*(?:#{1,6}\s|\*\*[^*]+\*\*\s*$|__[^_]+__\s*$)")
_MARKDOWN_EMPHASIS = re.compile(r"(\*\*|__|\*|_|`)")


def _first_paragraph(text: str, limit: int = 400) -> str:
    """The first real sentence of the rewritten notes, for the meeting card.

    The card shows plain text, so the markdown the model writes has to come
    off — a card reading "**Enrollment Numbers**" looks like a rendering bug.
    """
    for block in (text or "").split("\n\n"):
        lines = [line for line in block.splitlines() if not _MARKDOWN_HEADING.match(line)]
        cleaned = _MARKDOWN_EMPHASIS.sub("", " ".join(line.strip() for line in lines)).strip()
        cleaned = cleaned.lstrip("-•* ").strip()
        if cleaned:
            return cleaned[:limit]
    return ""


def to_record_fields(outcome: TranscriptionOutcome) -> dict:
    """Shape the summary the way the workspace Meetings view already renders it.

    The view expects lists of strings under `decisions`, `actions`, `deadlines`,
    `questions` and `findings` — the shape `src/bedrock/server/workspace/files.py` produces for
    an uploaded transcript. Rendering each point through `Point.render()` keeps
    the owner and the verified timestamp attached to the sentence they belong to.
    """
    # A transcript that was edited must say so. The rest of this product is
    # careful to separate what was said from what a model inferred, and a
    # silent repair would undo that.
    repair = outcome.vocabulary.describe()
    repair_suffix = f"; {repair}" if repair else ""

    summary = outcome.summary
    if summary is None:
        return {
            "summary": "",
            "decisions": [],
            "actions": [],
            "deadlines": [],
            "questions": [],
            "findings": [],
            "people": [],
            "method": "local transcription; no notes written" + repair_suffix,
        }

    method = "whisper.cpp transcription on this machine; notes written by "
    method += summary.model_used or "no model"
    # Every timestamp is checked against the line it quotes and dropped if it
    # cannot be resolved, so a point that still carries one has been verified.
    # This used to report a count from a `citations` field that no longer
    # exists upstream — verification moved inline — and the fallback branch
    # then told every reader to "update the pin" on a pin that was current.
    method += "; timestamps verified against the quoted line, unverifiable ones dropped"
    method += repair_suffix

    # `summarize_meeting` fills in *either* the overview or the rewritten
    # notes, never both: when somebody typed notes during the call, those
    # become the output and no separate overview is generated. The Meetings
    # card shows `summary`, so without this fallback every meeting where the
    # person actually took notes displays a blank card.
    overview = summary.overview or _first_paragraph(summary.enhanced_notes)

    return {
        "summary": overview,
        "decisions": [point.render() for point in summary.decisions],
        "actions": [point.render() for point in summary.actions],
        "findings": [point.render() for point in summary.details],
        "deadlines": [],
        "questions": [],
        "language": outcome.language,
        "languagesHeard": outcome.languages_heard,
        "transcriptEnglish": english_vtt(outcome),
        "translatedLines": len(outcome.translations),
        "people": list(summary.facts.speakers),
        "enhancedNotes": summary.enhanced_notes,
        "durationMinutes": summary.facts.duration_minutes,
        "wordCount": summary.facts.word_count,
        "method": method,
    }

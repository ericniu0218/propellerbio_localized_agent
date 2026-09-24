# Roadmap: better meeting summaries, and acting on what was asked

Two workstreams that share a transcript and nothing else. Read
`AGENT_ACCESS.md` for the half that executes; this document covers the half
that *understands*.

## Where the current logic actually stands

```
audio → whisper (per-participant tracks, VAD, per-turn slicing)
      → Cue(start_ms, end_ms, speaker, text)          transcript.py
      → chunk_cues(1100 words)                        summarize.py
      → one _EXTRACT_PROMPT call per chunk → JSON     (map)
      → _points(): quote → citations.resolve_timestamp
      → _dedupe(): 0.5 Jaccard on 4+ char words       (reduce)
      → one _NOTES_PROMPT or _OVERVIEW_PROMPT call over cues[:400]
      → MeetingSummary.render() → notes.md
```

What is solid: speaker attribution is structural, not inferred, because Zoom
writes one track per person. Timestamps are now evidence-backed. Facts
(duration, speakers, word count) never touch a model, so the thing degrades to
something true rather than to nothing.

Six real weaknesses, in the order they cost accuracy:

1. **Chunks cannot see each other.** A decision announced in chunk 2 and owned
   in chunk 5 is two half-points. Nothing carries state between calls.
2. **`cues[:400]` silently truncates.** Past roughly 25 minutes, the overview
   and the enhanced notes are summarising the first half of the meeting and
   nothing says so.
3. **`_dedupe` is a bag-of-words Jaccard.** It merges "move the interim
   analysis to March" with "move the interim analysis to May" (4 of 6 words
   shared) and keeps two phrasings of the same decision that share none.
4. **Decision recall is measured at 1 of ~4.** Group decisions phrased in
   passing are missed; explicit assignments are found. That is a prompt and
   model-size problem, and it is the largest single quality gap.
5. **Nothing knows what kind of meeting this is.** A standup, a DSMB review and
   a 1:1 get identical prompts.
6. **No confidence anywhere.** Every extracted point renders the same whether
   the model was certain or guessing.

## Phase 1 — stop losing things (no new dependencies)

- **Overlapping chunks.** Give `chunk_cues` a `overlap_cues` parameter, default
  2, so a sentence that straddles a boundary is read whole by one of the two
  calls. Dedup already exists to absorb the repeats.
- **A carry-forward header.** Prefix each chunk's prompt with the points
  already extracted, capped at ~150 tokens. This is the cheapest fix for
  problem 1 and needs no architectural change: it is still one call per chunk.
- **Say when you truncated.** If `len(cues) > 400`, run the notes/overview pass
  over the *whole* meeting in chunks and stitch, or - minimum viable - set a
  `MeetingSummary.coverage_note` that `render()` prints. Silent truncation is
  the kind of bug that only shows up in a long meeting nobody re-reads.

## Phase 2 — dedupe that understands negation and numbers

Replace the Jaccard with a small deterministic comparison that is still
model-free: normalize through `citations.normalize`, then treat two points as
the same only when their content words match **and** their numbers, dates and
negations match. "March" vs "May" and "are" vs "are not" must split a pair, not
merge it. Keep it in one function with a table-driven test; this is a place
where a fuzzy library would quietly make things worse.

## Phase 3 — recall, the expensive problem

Three things to try, cheapest first, each measured on the same fixture:

1. **A two-pass extraction.** Pass one asks only "list every sentence where the
   group settled something", pass two turns those sentences into points. Small
   models do much better when a call has one job.
2. **A bigger model.** The 1-of-4 number is from `qwen2.5:7b`. A 14B is the
   normal fix for this class of miss; note it is ~9 GB at Q4.
3. **Meeting-type prompts.** A `MeetingKind` (standup / review / 1:1 /
   decision) selected by a cheap rule on speaker count and duration, each with
   its own extraction rules.

Gate all three on a **fixture harness**: 3-5 real transcripts with
hand-labelled decisions and actions, and a scoring script that prints
precision and recall per run. Without it, every prompt change is a vibe. This
harness is the single highest-value item in this document.

## Phase 4 — ask detection (`meetings/asks.py`)

Detect the requests a meeting makes of its participants and hand them across
the seam as `Ask` values from `bedrock/agents/contract.py`.

Deterministic first, model second - the same split as everything else here:

- **A cue-level trigger pass.** Regex/keyword families per `AskKind`:
  `let's meet|put something in|find time` → SCHEDULE; `remind|ping me|don't let
  me forget` → REMIND; `pull up|have ready|bring the` → PREPARE; `draft|write
  up|put together` → CREATE; `find out|look into|check whether` → LOOK_UP.
  Cheap, auditable, and it narrows the model's job to a handful of cues.
- **A model pass over only the triggered cues**, returning the paraphrase, the
  owner, the subject, and the time expression *as spoken*.
- **Evidence through the existing resolver.** Every ask carries a verbatim
  quote validated by `citations.resolve_timestamp`. An ask whose quote cannot
  be located is still shown to the person and is never executed unattended.
- **Never resolve dates here.** "next Tuesday" stays a string in `when_text`.
  Turning it into a date needs the meeting date and a calendar, which live on
  the other side of the seam.

Render detected asks as a `## Requests` section, each with its timestamp, so
the feature is useful before any executor exists.

## Phase 5 — confidence that survives to the page

Carry a coarse confidence on each point (`located evidence`, `explicit trigger`,
`repeated across chunks`) and render the weak ones in a separate "Possibly"
block rather than dropping them. Under-extraction is the measured problem;
hiding uncertain points makes it worse, and mixing them in silently makes the
confident ones untrustworthy.

---

## Splitting this between two models

The seam is **evidence quality vs. request detection**. They touch different
functions in different files and meet only at `contract.py`, which is frozen
and belongs to neither.

### Model B1 — "the summary is right"

Owns Phases 1, 2, 3, 5. Files:

- `src/bedrock/meetings/summarize.py` — `chunk_cues`, `_dedupe`,
  `summarize_meeting`, `MeetingSummary`, `_EXTRACT_PROMPT`, `_OVERVIEW_PROMPT`,
  `_NOTES_PROMPT`
- `src/bedrock/meetings/transcript.py`
- `tests/test_meetings.py`
- `tests/fixtures/transcripts/` and the scoring harness (new)
- `docs/MEETINGS.md`

**Must not touch:** `citations.py` (frozen for the duration - B2 depends on its
exact behaviour), `asks.py`, anything under `agents/`.

### Model B2 — "the asks are found"

Owns Phase 4. Files:

- `src/bedrock/meetings/asks.py` (new, whole file)
- `tests/test_asks.py` (new, whole file)
- `docs/roadmaps/ASK_DETECTION.md` (new)

**Must not touch:** `summarize.py`, `transcript.py`, `citations.py`,
`tests/test_meetings.py`.

**The one shared edit, and how to avoid it:** B2 needs its asks rendered by
`MeetingSummary`. Do not let B2 edit `render()`. Instead B1 adds, in its first
commit, a single optional field and a render block:

```python
asks: list[object] = field(default_factory=list)   # bedrock.agents.Ask
```

and `summarize_meeting` gains a `detect_asks: Callable | None = None`
parameter it calls if given. B1 writes both in Phase 1 against a stub; B2 fills
the callable. After that first commit the two never share a line.

# Meeting notes

Records a Zoom meeting, transcribes it here, and turns it into notes someone
will read. Runs entirely against
a model you host — Ollama on a laptop, vLLM on the company server. No audio,
no transcript, and no notes leave the machine.

## Why local recording, not cloud

Zoom only produces transcripts for **cloud** recordings, and cloud recording is
a paid feature. Local recording is on every plan including the free one, writes
to the host's own disk, and uploads nothing — so we produce the transcript
ourselves. For an air-gapped customer that is the better answer anyway: the
audio never reaches a third party at all.

Turn on, in the Zoom desktop client's Recording settings:

- **Record a separate audio file for each participant who speaks**
- Optionally *Record video during screen sharing* off, to keep files small

That first setting is the important one. It makes Zoom write one full-length
track per person with silence where they are not talking, which gives perfect
speaker attribution for free — every word in Bob's file was said by Bob, by
construction. No diarisation model, no voice clustering, and better labels than
either would produce.

## Using it

```
bedrock-meeting --list     # recordings on this machine
bedrock-meeting            # the most recent one
bedrock-meeting --notes my-notes.md --terms "pembrolizumab, site four, Priya Nair"
```

Writes `transcript.vtt` and `notes.md` next to the recording. With no model
server reachable it still writes the transcript and says so, rather than
failing outright.

## Transcribing a recording (library)

```
brew install whisper.cpp ffmpeg
mkdir -p ~/.cache/whisper-models && cd ~/.cache/whisper-models
curl -LO https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin
curl -LO https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin
```

```python
from bedrock.meetings.transcribe import WhisperConfig, WhisperTranscriber, transcribe_recording

transcriber = WhisperTranscriber(WhisperConfig(
    model=Path("~/.cache/whisper-models/ggml-small.en.bin").expanduser(),
    vad_model=Path("~/.cache/whisper-models/ggml-silero-v5.1.2.bin").expanduser(),
    initial_prompt="pembrolizumab, Phase II enrollment, interim analysis, site four",
))
result = await transcribe_recording(Path("~/Documents/Zoom/2026-09-15 09.00.12 …"), transcriber)
```

A 100-second three-person meeting transcribes in about 6 seconds on an M-series
laptop, and the whole path — audio to finished notes — took 27 seconds.

### Two things that are not optional

**VAD.** Without it, whisper's segment boundaries span silence: a cue begins
where the previous one ended rather than where speech starts. On a
per-participant track, which is mostly silence by design, that puts a line
minutes from when it was said. Measured here: one speaker's opening line was
reported at `00:00:00` without VAD and `00:00:08.830` with it, against a true
8.9s. A missing VAD model is a readiness failure, not a downgrade.

**Slicing per turn.** Even with VAD, running whisper across a whole track
produces segments that bundle utterances twenty-five seconds apart under one
timestamp. So speech regions are grouped into turns, each is cut out with
ffmpeg, and all slices go to one whisper invocation — model loaded once per
track — with each cue offset back onto the meeting timeline. Against synthetic
ground truth: **18 of 18 turns matched, median timestamp error 0.15s, worst
1.0s.**

**Vocabulary.** Whisper renders unfamiliar proper nouns phonetically — it wrote
"PREA" for "Priya" until the name was added to `initial_prompt`. Drug and assay
names will be worse. A per-customer term list is the cheapest accuracy fix
available.

## Try it

```
brew install ollama
ollama serve
ollama pull qwen2.5:7b
```

```python
from bedrock.inference.openai_compatible import OpenAICompatibleBackend
from bedrock.meetings import parse_vtt, summarize_meeting

cues = parse_vtt(open("meeting.vtt").read())
backend = OpenAICompatibleBackend("http://localhost:11434/v1", timeout=300.0)
summary = await summarize_meeting(
    cues, backend=backend, model="qwen2.5:7b", title="Phase II weekly sync",
    notes=open("my-notes.md").read(),   # optional
)
print(summary.render())
```

A five-minute transcript takes about 18 seconds on an M-series laptop.

## The two modes

**Without notes** you get an overview, decisions, and action items.

**With notes** — the rough lines someone typed during the call — those notes
become the skeleton. Their headings and bullets stay, in their order, and detail
from the transcript is filled in underneath with timestamps. This is the mode
worth having. A generic summary ignores what the person in the room thought was
worth writing down; this finishes what they started.

## Why it is built the way it is

**Rules first, model second.** Who spoke, how long the meeting ran, how many
words — computed, never wrong, never dependent on a GPU. The model is asked only
for what rules cannot do. If inference is down, `summarize_meeting` still
returns, `degraded_reason` explains why, and the facts are intact. The same
split `context/digest.py` makes.

**Map-reduce, not one giant prompt.** An hour of speech is ~11,000 tokens and
Ollama's default context is 4,096. The transcript is cut on cue boundaries into
chunks that fit anywhere, each read separately, findings merged and deduplicated.
Pass `words_per_chunk` to use a bigger window on a server that has one.

**Every timestamp is backed by words, not by a list of timestamps.** Models
invent citations, and — less obviously — they attach *real* timestamps to the
wrong sentence. So the model is asked for two separate things: the short line
you read, and a verbatim quote copied out of one transcript cue. Python then
finds that quote and prints the timestamp of the cue it is in.

The rules, in `meetings/citations.py`, and all of them plain string work — no
model is asked to confirm a model, and the check still runs with every server
switched off:

- The quote must appear **contiguously and whole-word** inside a single cue.
  Case, surrounding whitespace and curly-vs-straight quotes are folded; nothing
  else is. Word order, numbers, negation and punctuation all still count, so
  "140 of 200" does not match "148 of 200" and "coordinator never left" does not
  match "coordinator left".
- Evidence shorter than **three words** is rejected. Quoting "Yes." locates
  nothing.
- If the model's suggested timestamp names a cue that really contains the
  quote, it is kept. This is also how a phrase repeated later in the meeting
  gets pinned to the occurrence that was meant.
- Otherwise, if every cue containing the quote shows the same timestamp, the
  citation is **repaired** to it.
- If the quote appears at two different times and nothing chooses between them,
  or does not appear at all, the citation is **dropped** — the decision, action
  or note keeps its words and loses its bracket. Picking the earliest or the
  nearest would be a guess printed as a fact.
- A quote is only ever looked for in the cues that model call was actually
  shown, never the whole meeting.

The prose halves go through the same door. Enhanced notes come back as a small
JSON envelope — the Markdown plus one record per citation — so each timestamp in
them is checked before anyone sees it; the overview is asked not to cite at all.
Any bracketed timestamp a model writes directly into its output is removed
before the verified ones are written in, **including one it copied out of your
own rough notes**. Typing "[00:14:00]" during a call is a reminder, not a
verified citation, and once the model echoes it the two are indistinguishable.
Your own notes file is never modified.

**What a timestamp here does and does not mean.** It means the quote was found
at that point in the transcript. It does **not** mean the paraphrase next to it
logically follows from the quote — a model can summarise badly while quoting
accurately — and it is **not** word-level audio alignment. Citations are
cue-level and floored to whole seconds: they point at the start of the utterance
the quote is in, and the transcript's own timing is only as good as the
transcription (see the VAD and slicing numbers above).

## Quality, measured not assumed

With `qwen2.5:7b` on a real 38-turn transcript. These numbers predate the
citation rework described above and its prompt changes, and have not been
re-measured since:

| | Result |
|---|---|
| Fabricated citations | 0 |
| Action items found | 4–5 of 5 |
| Owner attributed correctly | ~80%, varies run to run |
| Decisions found | **1 of ~4 — under-extracted** |

Decision extraction is the weak point. The model reliably finds explicit
assignments and misses group decisions phrased in passing ("then we're not
opening a new site"). A measured 14B run was more conservative rather than
more complete, returning no decisions or actions where the 7B default found
both, so model changes need to be evaluated against the same real fixture.

## Licensing

`qwen2.5:7b` is **Apache 2.0**, which is why it is the default here rather than
a Llama build: it can ship to a paying client without a license review. Licenses
differ between sizes within the same family — check each model card individually
and record the name, revision, hash, and license alongside the deployment.

## Known gaps

- **No PHI redaction.** Meeting transcripts in a biotech company will discuss
  patients, and nothing on this path redacts anything.
- **Citations are cue-level, not word-level.** A verified timestamp is the
  start of the cue containing the quote, floored to a whole second — not the
  instant the quoted word was spoken. And it says nothing about whether the
  paraphrase beside it is a fair reading of that quote; it locates the quote,
  no more.
- **A quote spanning two cues loses its citation.** Evidence is matched inside
  one cue, so a sentence a speaker finished in their next utterance cannot be
  located. Cross-cue alignment is deliberately not built.
- **Whisper is not run against meetings with crosstalk.** Per-participant
  tracks make overlapping speech easy in principle, but it has only been tested
  on synthesised audio where nobody interrupts.
- **The VTT parser is shared with bedrock-agent-pack.** Its server-side
  `meeting_transcripts` tool imports `bedrock.meetings.transcript`, pinned to a
  commit in its `requirements.txt`, rather than keeping a copy. It relies on
  `Cue`, `parse_vtt`, `speakers`, `Cue.timestamp` and `Cue.render()`; changing
  those needs a matching backend change when the pin is next updated.
- **Prompts are only hand-verified.** The tests fake the backend, so they check
  the plumbing, not the wording. Changing a prompt needs a manual run.

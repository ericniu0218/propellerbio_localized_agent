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

**Never cite a timestamp that was not said.** Models invent citations. Every
timestamp is checked against the real cue list; one that does not exist is
re-derived by locating the quoted text, and dropped if that fails. A note with no
timestamp is honest; a fabricated one is not. Across test runs this holds at
zero fabricated citations.

## Quality, measured not assumed

With `qwen2.5:7b` on a real 38-turn transcript:

| | Result |
|---|---|
| Fabricated citations | 0 |
| Action items found | 4–5 of 5 |
| Owner attributed correctly | ~80%, varies run to run |
| Decisions found | **1 of ~4 — under-extracted** |

Decision extraction is the weak point. The model reliably finds explicit
assignments and misses group decisions phrased in passing ("then we're not
opening a new site"). Before tuning prompts further, try a 14B — this is the
kind of gap model size usually closes. Note a 14B at Q4 is ~9 GB and will be
tight on a 16 GB machine alongside the KV cache.

## Licensing

`qwen2.5:7b` is **Apache 2.0**, which is why it is the default here rather than
a Llama build: it can ship to a paying client without a license review. Licenses
differ between sizes within the same family — check each model card individually
and record the name, revision, hash, and license alongside the deployment.

## Known gaps

- **No PHI redaction.** Meeting transcripts in a biotech company will discuss
  patients, and nothing on this path redacts anything.
- **A verified timestamp is not a correct one.** The citation guard checks that
  a timestamp exists in the transcript, not that it is the moment the quoted
  thing was said. Observed once in testing: an action item cited a cue six
  seconds after the one it came from. Real, and much smaller than a fabricated
  citation, but not nothing.
- **Whisper is not run against meetings with crosstalk.** Per-participant
  tracks make overlapping speech easy in principle, but it has only been tested
  on synthesised audio where nobody interrupts.
- **The VTT parser is duplicated** in bedrock-agent-pack for its server-side
  `meeting_transcripts` tool. The repos share no package; keep them in step.
- **Prompts are only hand-verified.** The tests fake the backend, so they check
  the plumbing, not the wording. Changing a prompt needs a manual run.

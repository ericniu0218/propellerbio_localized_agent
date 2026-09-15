# Meeting notes

Turns a meeting transcript into notes someone will read. Runs entirely against
a model you host — Ollama on a laptop, vLLM on the company server. No audio,
no transcript, and no notes leave the machine.

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
- **The VTT parser is duplicated** in bedrock-agent-pack for its server-side
  `meeting_transcripts` tool. The repos share no package; keep them in step.
- **Prompts are only hand-verified.** The tests fake the backend, so they check
  the plumbing, not the wording. Changing a prompt needs a manual run.

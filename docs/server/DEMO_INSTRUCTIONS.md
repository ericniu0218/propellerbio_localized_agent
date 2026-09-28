# How to run the demo

## One-time setup

```
git clone https://github.com/Bedrock-Technical-Consulting/bedrock-agent-pack.git
cd bedrock-agent-pack
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

(The workspace UI is merged into `main` now — no branch checkout needed.)

## Option 1: Safe demo — use this one

```
./scripts/demo-safe.sh
```

Browser opens automatically to `http://localhost:8000/`. Click **"Continue as demo user"**.

That's it. Fictional data, no setup, nothing that can fail.

## Option 2: Real Zoom demo — only if you tested it once beforehand

One-time setup (do this before demo day, not during):

```
cd ..
git clone https://github.com/Bedrock-Technical-Consulting/zoom-transcript-vault.git
cd zoom-transcript-vault
npm install
printf 'ZOOM_WEBHOOK_SECRET_TOKEN=demo-zoom-secret-123\nDATACENTER_BASE_URL=http://localhost:9100\nDATACENTER_SIGNING_SECRET=demo-datacenter-secret-456\nRETRIEVAL_API_KEY=demo-retrieval-key-789\n' > .dev.vars
npx drizzle-kit generate
npx wrangler d1 migrations apply zoom-transcript-vault --local
```

Then, from `bedrock-agent-pack`:

```
./scripts/demo-zoom.sh
```

Sign in with any name (type-a-name, no password). Ask:

```
What happened in today's meeting about AlphaFold?
```

You should get a real answer quoting the transcript, with a citation.

**Known gap as of tonight's merges:** this script uses `AGENT_PACK_MODE=mock`
to avoid needing a real model. A separate change on `main` now requires
`AGENT_PACK_MODE=real` (a real local model, e.g. via Option 3 below) to get
genuine AI-read answers — otherwise you'll get the canned demo text instead
of the real transcript quote. If this script stops giving a real answer,
that's why; ask before "fixing" it, since two of us made different calls
here and it needs one decision, not two more patches.

## Option 3: Test real answers with your own local model

If you already have Ollama installed and a model pulled:

```
./scripts/demo-safe.sh
```

It auto-detects Ollama. If it finds a model at `http://localhost:11434/v1`
(default `llama3.1:8b` — override with `OLLAMA_MODEL=your-model` if you
pulled something else), it gives real, genuinely-generated answers. If it
can't find Ollama, it says so and falls back to the same safe canned demo
— nothing breaks either way.

**To verify the memory-context fix:** click "Continue as demo user" →
**Memory** in the sidebar → drop a text file in → open (or start) a chat
→ ask a question about something in that file. It should cite the file
and actually answer from its content, not just report word/line counts.
That "actually answer from content" part only happens with a real model
connected — that's what needs testing here.

Note: `llama3.2:3b` runs but is too small to reliably follow this app's
structured output — use `llama3.1:8b` or larger.

## To stop either one

Press `Ctrl+C` in the terminal it's running in.

## If something breaks

Use Option 1. It always works and is what most of the demo should be anyway.

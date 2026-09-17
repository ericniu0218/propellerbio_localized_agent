# How to run the demo

## One-time setup

```
git clone https://github.com/Bedrock-Technical-Consulting/bedrock-agent-pack.git
cd bedrock-agent-pack
git checkout feature/scientist-workspace
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
```

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

## To stop either one

Press `Ctrl+C` in the terminal it's running in.

## If something breaks

Use Option 1. It always works and is what most of the demo should be anyway.

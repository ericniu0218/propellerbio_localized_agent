# Installing Bedrock Private AI

This is written for anyone installing this for the first time — no
programming experience assumed. Follow the numbered steps for your
computer. Wherever you see a gray box, copy the whole thing and paste it
into the window described just above it.

## Before you start: getting access to the company's code

This project lives in a **private** GitHub repository, so before
installing it your computer needs to be able to download private company
code at all — a one-time setup, separate from installing the app itself,
and something you only do once no matter how many company projects you
install after this. If you've already cloned or installed something else
from this company's GitHub before, you've already done this — skip to
"Install the app" below.

Otherwise, do **one** of these:

- **Easiest, no command line:** install
  [GitHub Desktop](https://desktop.github.com), open it, and sign in with
  your company GitHub account when it asks. That's it.
- **If you're already comfortable with a terminal:** install the
  [GitHub CLI](https://cli.github.com), then run `gh auth login` and
  follow the prompts (choose HTTPS when asked).

Either way, ask whoever manages the company's GitHub org to confirm your
account has access to `Bedrock-Technical-Consulting/bedrock-private-ai`
if you're not sure.

## Install the app

### macOS or Linux

1. Open **Terminal**. (On macOS: press `Cmd + Space`, type `Terminal`, press Enter.)
2. Paste this, press Enter, and wait for it to finish:

   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. If that's the first thing like this you've ever installed on this
   computer, close the Terminal window and open a new one now. (This
   lets your computer notice the new command. If you're not sure, do it
   anyway — it doesn't hurt.)
4. Paste this, press Enter, and wait for it to finish:

   ```
   uv tool install "git+https://github.com/Bedrock-Technical-Consulting/bedrock-private-ai.git"
   ```

5. Type `bedrock` and press Enter. The app should open, taking over the
   whole terminal window.

### Windows

1. Open **PowerShell**. (Press the Windows key, type `PowerShell`, press Enter.)
2. Paste this, press Enter, and wait for it to finish:

   ```
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. If that's the first thing like this you've ever installed on this
   computer, close the PowerShell window and open a new one now.
4. Paste this, press Enter, and wait for it to finish:

   ```
   uv tool install "git+https://github.com/Bedrock-Technical-Consulting/bedrock-private-ai.git"
   ```

5. Type `bedrock` and press Enter. The app should open, taking over the
   whole window.

### If you already have the repo cloned

`scripts/install.sh` (or `scripts/install.ps1` on Windows) in this repo
does steps 2 and 4 above for you in one go — run it from inside your
clone.

## What just happened?

Step 2/3 installed a small helper called `uv` — this is what manages the
Python version and an isolated environment for you, so you never install
Python yourself or deal with a "virtual environment." Step 4 used `uv` to
install this app the same way, downloading it straight from the company's
private GitHub repo using the access you set up above.

## Connecting to your company's AI system

The first time you run `bedrock`, it checks whether it can reach a real
AI system and tells you plainly if it can't — that's expected right after
installing, since nothing is configured yet. Ask your team for:

- The address of the AI system (looks like `https://something.company.com`)
- Whether you need an access key

Then create a file at:

- macOS/Linux: `~/.config/bedrock/config.toml`
- Windows: `%USERPROFILE%\.config\bedrock\config.toml`

containing (for the company's full agent system - IT runs its server,
`bedrock-server`; you never install that part yourself):

```toml
backend = "agentpack"
agent_pack_api_base = "https://the-address-your-team-gave-you"
api_key = "the-key-your-team-gave-you"   # needed for literature reviews (/review)
```

In this mode `/review` runs literature reviews on your machine and sends each
of their model calls to the AI system's models; the `api_key` is the key IT
set on the server (`BEDROCK_SERVER_INFERENCE_TOKEN`). Check it is all working
with `bedrock-litreview doctor` - it says whether the server answers and,
separately, whether a literature review can run.

or (for a single private model server):

```toml
api_base = "https://the-address-your-team-gave-you"
api_key = "the-key-your-team-gave-you"
```

Restart `bedrock` after saving the file. The startup check goes away once
this is in place.

## Running on a laptop

If the model runs on your own laptop (with Ollama) rather than on a company
server, it needs most of the laptop's memory. Two things keep it usable on
8 GB:

1. Use a small model: a 3-4B model at 4-bit, for example
   `ollama pull llama3.2:3b`. Bedrock won't pick anything bigger than 3.5 GB by
   itself.
2. Start Ollama with these settings (your team can put them in place for you):

   ```
   OLLAMA_CONTEXT_LENGTH=8192
   OLLAMA_NUM_PARALLEL=1
   OLLAMA_MAX_LOADED_MODELS=1
   OLLAMA_FLASH_ATTENTION=1
   OLLAMA_KV_CACHE_TYPE=q8_0
   ```

   If it's still slow, use `4096` instead of `8192`, and add
   `context_window = 4096` to your `config.toml` so Bedrock knows.

On a bigger machine, add `profile = "workstation"` to `config.toml`; on a shared
company server, IT uses `profile = "server"`.

## Literature reviews (`/review`)

`/review` answers a research question with a report in which every sentence
cites a checked quote from a paper. It needs more from the machine than chat:

- **A model server with a big enough "context window".** A review sends long
  passages of papers, so the window must be at least 12,288 tokens. On the
  office server this is IT's job. On a laptop (16 GB or more), start Ollama
  with `OLLAMA_CONTEXT_LENGTH=12288` instead of the `8192` above, pull the
  review's model with `ollama pull qwen3:8b`, and add to `config.toml`:

  ```toml
  context_window = 12288
  request_timeout = 600
  ```

  `request_timeout` matters on a laptop: one extraction step can take several
  minutes, and the default of 120 seconds stops the review halfway.
- **A company server name it trusts.** A review sends paper text to the model
  server, so it refuses any address that is not on this machine or the
  company network. If your server's address is a name such as
  `ai.company.com`, IT adds it to `config.toml`:

  ```toml
  [litreview]
  allowed_hosts = ["ai.company.com"]
  ```
- **Papers.** Either a papers file your team gives you (`/review your question
  --corpus papers.jsonl`; you can drag the file onto the terminal after
  `--corpus`), or `--search` to fetch papers from Europe PMC and PubMed now.
  Searching sends only the words of your question to those two public
  literature databases, never to an AI service.

**Check all of this before you start.** `/reviews check` in the app, or
`bedrock-litreview doctor` in a terminal, goes through every item above and
prints what is right, what is worth knowing, and what would stop a review —
including whether the server has the model, how much disk is free, and which
settings file it read. Run it whenever something changes; it starts nothing and
costs a second.

The review then runs beside your chat — your own messages always go first — and
tells you where the finished `report.html` is; open it in your browser.
`/reviews` lists past reviews, and `/reviews forget <run-id>` deletes one
(permanently — the question stays, so you can ask it again). A review takes tens
of minutes on a laptop.

If a review stops with a message, it says what to change. The common ones:

| What you see | What to do |
|---|---|
| "needs a context window of at least 12,288 tokens" | Set `context_window = 12288` and restart Ollama with `OLLAMA_CONTEXT_LENGTH=12288`, as above. |
| "does not have 'qwen3:8b'" (or another name) | `ollama pull` that name, or ask IT to serve it on the office server. |
| "refusing model endpoint ... not a local address" | Ask IT to add the server's name to `allowed_hosts`, as above. |
| "the model server ... is not answering", or the review stops partway | Start the server, or raise `request_timeout`; then run the same `/review` with `--resume` and the run id it printed. |
| "no papers were found for this question" | The search found nothing: reword the question, or use a papers file. |
| "The literature review is not installed in this environment" | The app was updated without its new parts. Run the `uv tool install ...` command from step 4 again, then restart. |
| Anything else, or nothing obvious | Run `/reviews check` — it lists everything that is wrong at once. |

## Updating to a newer version later

Run the same `uv tool install ...` command from step 4 again — it
reinstalls over the previous version. (Add `--force` if it says the
version is unchanged but you know the code has changed on GitHub.)

## Troubleshooting

| What you see | What it means / what to do |
|---|---|
| `command not found: bedrock` right after installing | Close your terminal and open a fresh one — the most common first-time issue, see step 3. |
| `uv tool install` fails with something about `git`, `Authentication failed`, `Permission denied`, or `repository not found` | You don't have working GitHub access to this private repo yet from this computer — see "Before you start" above. |
| `bedrock` opens but says it can't reach the AI system | Expected before you've configured it — see "Connecting to your company's AI system" above. Once configured, this means the address is wrong or the AI system is down; check with your team. |
| `/fold` or `/pymol` says a tool "isn't set up yet" | Those two are optional and only needed for protein-structure work — run `bedrock-biotech status` for exactly what's missing, and ask your team if you need them set up. |
| Something else entirely | Run `bedrock-biotech status` and share its output, along with what you were trying to do, with your team. |

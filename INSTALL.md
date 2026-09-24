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

containing (for the company's full agent system):

```toml
backend = "agentpack"
agent_pack_api_base = "https://the-address-your-team-gave-you"
```

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

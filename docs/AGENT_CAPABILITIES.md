# Agent capabilities

The capabilities are the only code in Bedrock Private AI that touches the
outside world. Everything else reads a transcript, writes a plan, or asks a
person. This document is for whoever configures them and whoever changes them.

Read [`docs/roadmaps/AGENT_ACCESS.md`](roadmaps/AGENT_ACCESS.md) first: it
explains why agents work by *evidence, plans and grants* rather than by tools,
and where these files sit in that picture. Read
[`src/bedrock/agents/contract.py`](../src/bedrock/agents/contract.py) second:
it is frozen, and it is the only thing both halves of the agent work import.

## What exists

| Capability | Risk | What it does | Ships |
| --- | --- | --- | --- |
| `files.search` | read-only | Finds files whose name or contents mention every word in a query | on |
| `files.read` | read-only | Reads one file as text | on |
| `files.write` | changes things | Creates a new file under one output folder | on |
| `sources.fetch` | read-only | Pulls one record from a company system of record | on |
| `sources.search` | read-only | Finds records in one company system | on |
| `web.fetch` | read-only | Fetches one allowlisted address through the egress proxy | needs a proxy |
| `web.search` | read-only | Searches the company's own SearxNG instance | needs a `search_url` |
| `calendar.create` | changes things | Blocks time on a CalDAV calendar | needs a `url` |
| `mail.draft` | changes things | Writes an `.eml` file the employee can open and send | on |
| `mail.send` | **reaches people** | Sends an email over SMTP | **off** |
| `shell.run` | changes things | Runs one command in a sandbox, with a clock on it | **off** |

"Ships on" means the capability works as soon as it is configured. `mail.send`
and `shell.run` additionally require `enabled = true`, and they ship `false`.

None of them is offered to any agent until it is named in that agent's roster
entry in `~/.config/bedrock/agents.toml`. A capability absent from the roster is
never imported, so it cannot appear in a plan and be approved by a tired person
at six in the evening.

## Two rules that hold for all of them

**Nothing above read-only runs without a grant, and the check is not in these
files.** [`protocol.Capability`](../src/bedrock/agents/protocol.py) hands a
capability its arguments and nothing else, because the gate belongs one level
up: `Supervisor.execute` calls `GrantStore.consume`, which requires a grant
that was issued for this exact plan (by fingerprint), names this step, and has
not already been spent. No capability is called until that passes.

That is deliberately stronger than anything an adapter could do with a `Grant`
value in hand. A hand-written `Grant` passes an "is my name in `steps_allowed`?"
test and fails `consume`. So there is one implementation of the rule, above
these files, and they do not carry a weaker copy of it. Each capability's test
file proves the property where it lives: drive the real supervisor with no
grant, and assert that no file appeared, no request was made and no process
was forked.

What an adapter *does* own is every lock that has nothing to do with approval —
`mail.send` and `shell.run` shipping switched off, the recipient allowlist, the
domain allowlist and proxy, the file roots. Those hold whether or not anybody
approved anything.

`describe()` is always available, never needs configuration, and never does
anything. It exists so the plan a person reads says what would happen.

**Endpoints are configured, never discovered.** No adapter invents a host, and
none of them reads an endpoint from the model's output. Every address comes
from `~/.config/bedrock/config.toml`, which follows the same rule the company
data-source connectors already follow: the config file names the environment
variable that holds a credential, and the credential itself never appears in
the file.

## Configuring them

All of this lives in `~/.config/bedrock/config.toml`. Every table is optional;
a capability with no configuration refuses with a message that says which
setting is missing and where to put it.

A dotted capability name is looked up twice and merged, family table first,
so verbs that share a rule share its setting:

```toml
[agents.capabilities.files]
roots = ["~/work/projects", "/mnt/share/protocols"]  # what may be read
output_root = "~/work/agent-output"                  # what may be written
extensions = [".md", ".txt", ".csv"]                 # optional; empty means every file
max_bytes = 1000000                                  # per file, for reads and content search
max_results = 50
max_scan_files = 20000

[agents.capabilities.web]
proxy = "http://egress.internal:3128"    # required; empty turns web.fetch off
allowed_domains = ["fda.gov", "clinicaltrials.gov"]
search_url = "https://searx.internal"    # a SearxNG instance the company runs
search_via_proxy = false                 # it is usually on the inside
store_dir = "~/.local/share/bedrock/agents/web"
max_bytes = 2000000
max_excerpt_chars = 20000
timeout = 30.0

[agents.capabilities.calendar]
url = "https://caldav.internal/dav/calendars/alice/work/"
username = "alice"
password_env = "BEDROCK_CALDAV_PASSWORD"
timezone = "Europe/London"
default_minutes = 30

[agents.capabilities.mail]
from_address = "alice@company.example"
draft_dir = "~/work/agent-drafts"
enabled = false                               # mail.send; ships off
allowed_recipient_domains = ["company.example"]
smtp_host = "smtp.internal"
smtp_port = 587
smtp_username = "alice"
smtp_password_env = "BEDROCK_SMTP_PASSWORD"
smtp_starttls = true

[agents.capabilities.shell]
enabled = false                               # ships off
workdir = "~/work/agent-scratch"
sandbox_command = ["bwrap", "--unshare-net", "--die-with-parent",
                   "--ro-bind", "/usr", "/usr", "--ro-bind", "/bin", "/bin",
                   "--bind", "{workdir}", "{workdir}"]
allow_network = false
timeout_seconds = 30
max_output_bytes = 200000
env_passthrough = ["PATH", "HOME", "LANG"]
```

`sources.fetch` and `sources.search` have **no table of their own**, on
purpose. They use the `[connectors.<name>]` tables that `/digest`, `/fold` and
`/sources` already use. A second place to configure the registry is a second
place to configure it wrongly, and an agent that reached a different endpoint
from the one `/fold` uses would be writing provenance records that quietly mean
something else.

## The decisions worth knowing about

### files: two roots, and nothing is ever replaced

Reading and writing have separate settings. `roots` is what may be read;
`output_root` is the one folder that may be written. Being allowed to read the
protocol folder never silently means being allowed to rewrite it.

Every path is resolved with symlinks followed and *then* checked to be inside a
root. That ordering is the whole protection: a symlink sitting inside a root
and pointing at `/etc` is refused rather than followed. A path outside the
roots is a hard error, not a warning.

`files.write` creates `name (2).md` rather than replacing `name.md`, using
exclusive creation rather than "check, then write" — two agents can run at
once, and between that check and that write is exactly where the overwrite
would happen.

### web: a missing proxy means no fetching, not direct fetching

If `proxy` is empty, `web.fetch` refuses. It does not quietly go straight out.
A capability that silently downgrades its own protection is worse than one that
does not exist.

Redirects are not followed. A redirect comes back as an address to ask for, so
the second address gets the same domain check as the first — otherwise an
allowlisted host chooses where the agent goes next. A bare domain in
`allowed_domains` covers its subdomains and nothing else: `notfda.gov` does not
end in `.fda.gov`.

Every fetch is written to `store_dir` and hashed before any model sees it, so
what the model was shown can be compared with what the server actually sent.

`web.search` needs a SearxNG instance the company runs. Not a commercial search
API: with search, the query text is the leak, and "what is our sponsor's
withdrawal rate" is a sentence that must not leave the building.

### calendar: a hold, not an invitation

`calendar.create` blocks time on the employee's own calendar. **Attendees are
deliberately not supported.** Adding them makes the CalDAV server send iTIP mail
to everybody named, which turns one calendar write into eight emails — that is
`REACHES_PEOPLE` wearing a calendar's clothes, and it belongs behind the same
approval as `mail.send` rather than arriving as a side effect of a scheduling
step. The employee sends the invitation.

Times with no zone in them are read in the configured `timezone` and converted
to UTC before being written, so no `VTIMEZONE` block is shipped whose rules
could go stale. Working out what "next Tuesday" means is the planner's job:
this capability takes an ISO 8601 time and refuses anything else.

No CalDAV library is used. Creating one event is one authenticated `PUT` of one
iCalendar object (RFC 4791 §5.3.2), with `If-None-Match: *` so a retry cannot
replace an event that already exists.

### mail: four locks on the riskiest verb

An email cannot be unsent, cannot be corrected, and arrives with the employee's
name on it. `mail.send` has four locks and all four must be open:

1. a `Grant` that names `mail.send` — held by the supervisor, checked before
   this capability is called at all,
2. `enabled = true` — **it ships `false`**,
3. an `smtp_host`,
4. every recipient inside `allowed_recipient_domains`, which is **empty by
   default, and empty means nobody**.

A company that wants an agent sending mail has to write down where it may send.
That is a sentence somebody has to type, on purpose, in a file.

`mail.draft` is the one to reach for. It writes an ordinary `.eml` the employee
opens, reads, fixes and sends themselves, and nothing leaves the machine.

Headers are checked before the message exists. A newline in a subject or an
address is refused outright — that is how one `To:` line becomes a `Bcc:`
nobody approved.

### shell: the box, not the arguments

Every other adapter is safe because its arguments can be checked. A command
line has no such property: `python -c '...'` is every capability in this package
at once, plus the ones nobody wrote. So `shell.run` is not made safe by
inspecting what it is asked to do.

- **No shell.** The command is split with `shlex` and run with `exec`. There is
  no `/bin/sh`, so `;`, `&&` and backticks are ordinary characters inside an
  argument. Pipes and redirection do not work here; that is the feature.
- **No network, and no pretending.** Nothing in the standard library can take a
  network away from a child process. So `allow_network = false` is only honoured
  when `sandbox_command` is configured; with neither, `shell.run` **refuses to
  run**. Claiming a sandbox that does not exist is worse than having none.
- **A wall clock.** The child gets its own process group and is killed as a
  group, so a command cannot leave sleeping children behind.
- **A clean environment.** Only `env_passthrough` variables are handed over.
  Tokens for the registry, the calendar and the mail server are not in scope for
  a subprocess.

`{workdir}` inside `sandbox_command` is substituted, so a bubblewrap line can
bind the working directory without the path being written down twice.

## For whoever changes this code

### Adding a capability

Add a file. Do not edit a list.

1. Create `src/bedrock/agents/capabilities/<name>.py`.
2. Implement [`protocol.Capability`](../src/bedrock/agents/protocol.py): a
   `name`, a `risk`, a `describe(arguments) -> str` and an
   `async run(arguments) -> Outcome`. Do not add a grant parameter; see above.
3. Export `CAPABILITY` at the bottom of the module — one capability, or a tuple
   of them. `roster.discover_capability` takes either and picks by name.
4. Add `tests/test_capabilities_<name>.py`. Every capability above read-only
   needs a test that drives the real `Supervisor` with no grant and proves
   *nothing happened* — no file, no request, no process — not merely that an
   error was returned.

### How a result gets back

`contract.Outcome` is frozen and has no field for "what the tool found", so the
convention here is:

- `completed` carries the readable result lines. For `files.read` that is the
  file's text; for `web.fetch` it is the hash line, the path and an excerpt.
- `artifacts` carries only paths this run actually **created or staged** — not
  paths it merely looked at. `files.search` returns matches in `completed` and
  no artifacts; `sources.fetch` returns the staged file and its provenance
  sidecar.
- `failed` carries the plain-English reason, and `completed` is still filled in
  alongside it. A step that wrote two files and failed on the third has to say
  that it wrote two files.

- `plan_id` is left empty. A capability does not know which plan it belongs to,
  and the supervisor builds the plan's own `Outcome` from the ones its steps
  returned. `echo` does the same.

Build all of it with `done()` and `failed()` rather than by hand, so the length
caps apply.

### Configuration lookup

`capability_config("<name>")` finds `[agents.capabilities.<name>]` wherever the
settings object keeps it. It deliberately tolerates two layouts: today an
`[agents]` table is an unrecognised key and lands in `Settings.extra`, and once
`config/settings.py` grows a real `agents` field it will be found there instead.
No adapter needs to change either way, and no adapter edits `settings.py`.

Settings are read lazily, the first time a capability is used. Importing an
adapter must never read the config file or contact anything — the roster imports
every adapter just to list what exists.

### Two things that are still in motion

- **Configuration tables.** `config/settings.py` ships one empty table per
  planned capability, keyed by dotted name (`[agents.capabilities."files.read"]`).
  These adapters also read a family table (`[agents.capabilities.files]`) and
  merge the dotted one over it, because `files.search`, `files.read` and
  `files.write` check the same roots and writing those roots into three tables
  is how two of them end up disagreeing with the third.
- **`CAPABILITY` is a tuple where a file offers several verbs.**
  `roster.discover_capability` accepts one capability, a tuple, or a mapping,
  and selects by name — so a file can hold every verb that shares one set of
  rules without splitting that rule across files. `CAPABILITIES` is exported as
  a plural alias for reading a module's contents directly.

### What these capabilities do not do

They do not write to the journal. `bedrock.provenance` is an append-only record
with a fixed set of actions and its own contract
([`docs/compliance_mvp/CONTRACT.md`](compliance_mvp/CONTRACT.md)); recording
what an agent did belongs to the supervisor, which knows the plan, the grant and
the person, and which has all of the outcomes rather than one. A capability
returns an `Outcome` and that is the whole of its reporting duty.

They also do not decide *whether* a step should happen — only whether it is
permitted to. Choosing the steps is the planner's job, approving them is a
person's, and turning "next Tuesday" into a date is the planner's too.

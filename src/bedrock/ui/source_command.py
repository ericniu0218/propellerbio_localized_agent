"""The `/sources` command, shared by both apps.

Same pattern as ``bedrock.ui.digest_command``: written against a ``notify``
callback rather than against Textual, so both apps word it identically and it
can be tested on its own.

Sub-commands::

    /sources                          list configured sources and their status
    /sources search <source> <words>  find records without knowing the id
    /sources fetch <source>://<id>    pull a record into the private staging folder

Using a record in a workflow does not need this command - ``/digest`` and
``/fold`` accept ``<source>://<id>`` directly. ``fetch`` covers everything
else: it leaves a file plus a provenance sidecar that any tool can pick up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Callable

from bedrock.connectors import ConnectorError, FetchedRecord, RecordSummary, SourceHub

Notify = Callable[[str, str], Awaitable[None]]

USAGE = """\
**`/sources` — pull data straight from company systems**

| Command | What it does |
|---|---|
| `/sources` | List configured sources and whether each is set up correctly |
| `/sources search <source> <words>` | Find records in one source |
| `/sources fetch <source>://<id>` | Pull a record into your private staging folder |
| `/digest <source>://<id>` | Pull a record and compress it for the model |
| `/fold <source>://<id>` | Pull a sequence and run AlphaFold 2 on it |
| `/pymol <source>://<id>` | Pull a structure and open it in PyMOL |

Each source decides for itself what you are allowed to see. Everything pulled \
is saved with a record of which system, record and version it came from.
"""

NOT_CONFIGURED = """\
**No company data sources are configured yet.**

Add one table per system to `~/.config/bedrock/config.toml`, for example:

```toml
[connectors.registry]
type = "rest"
base_url = "https://registry.internal.example.com/api/v1"
record_path = "/sequences/{id}"
content_field = "data.fasta"
suffix = ".fasta"
token_env = "REGISTRY_TOKEN"

[connectors.lims]
type = "share"
root = "/mnt/lims-exports"
```

Then `/digest lims://plates/P42.csv` or `/fold registry://SEQ-00123` pulls the \
record directly. See the README's "Company data sources" section.
"""


async def run_sources_command(args: str, *, hub: SourceHub, notify: Notify) -> None:
    """Handle one ``/sources ...`` invocation."""
    args = args.strip()
    verb, _, rest = args.partition(" ")
    verb, rest = verb.lower(), rest.strip()

    if not args:
        await notify(format_sources(hub), "info")
        return

    if verb in {"help", "?"}:
        await notify(USAGE, "info")
        return

    if verb == "search":
        name, _, query = rest.partition(" ")
        query = query.strip()
        if not name or not query:
            await notify(
                "Usage: `/sources search <source> <words>`, e.g. `/sources search registry kinase`.",
                "error",
            )
            return
        try:
            results = await hub.search(name.lower(), query)
        except ConnectorError as exc:
            await notify(str(exc), "error")
            return
        await notify(format_results(name.lower(), query, results), "info")
        return

    if verb == "fetch":
        reference = rest.strip("'\"")
        if not reference:
            await notify("Usage: `/sources fetch <source>://<id>`.", "error")
            return
        try:
            record, path = await hub.fetch_to_file(reference)
        except ConnectorError as exc:
            await notify(str(exc), "error")
            return
        await notify(format_fetched(record, path), "info")
        return

    await notify(f"Unknown `/sources {verb}`. Type `/sources help` for what it takes.", "error")


def format_sources(hub: SourceHub) -> str:
    if not hub.connectors and not hub.config_errors:
        return NOT_CONFIGURED

    lines = [f"**Company data sources** (requests are made as `{hub.user}`)", ""]
    if hub.connectors:
        # "ok" means the configuration checks out locally - a folder that is
        # mounted, a token that is set. It does not contact the source, so the
        # listing is instant and never waits on a slow system.
        lines += ["| source | type | points at | setup |", "|---|---|---|---|"]
        for connector in hub.connectors:
            issues = connector.readiness_issues()
            status = "ok" if not issues else "problem: " + "; ".join(issues)
            lines.append(f"| `{connector.name}` | {connector.kind} | {connector.describe()} | {status} |")
        lines.append("")
    if hub.config_errors:
        lines.append("**Config problems** (these sources are unavailable until fixed):")
        lines.extend(f"- {error}" for error in hub.config_errors)
        lines.append("")
    lines.append(
        "Use `/digest <source>://<id>` to pull and compress a record, or `/sources help` for more."
    )
    return "\n".join(lines)


def format_results(name: str, query: str, results: list[RecordSummary]) -> str:
    if not results:
        return f"**No records in `{name}` match “{query}”.**"
    plural = "s" if len(results) != 1 else ""
    lines = [f"**{len(results)} record{plural} in `{name}` matching “{query}”**", ""]
    lines.extend(f"- `{result.ref}` — {result.title}" for result in results)
    lines.append("")
    lines.append("Use one with `/digest <ref>`, `/fold <ref>` or `/sources fetch <ref>`.")
    return "\n".join(lines)


def format_fetched(record: FetchedRecord, path: Path) -> str:
    return "\n".join(
        [
            f"**Fetched `{record.ref}`** as `{record.fetched_by}`",
            "",
            f"- Saved to: `{path}`",
            f"- Provenance: `{path.name}.provenance.json` alongside it",
            f"- Version: {record.version or 'not reported by the source'}",
            f"- SHA-256: `{record.sha256}`",
            f"- Size: {len(record.content):,} bytes",
        ]
    )

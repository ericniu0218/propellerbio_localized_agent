"""The entailment check — an optional, local, model-based second look at a
report sentence, after the citation guard. OFF BY DEFAULT, AND UNMEASURED.

Why it exists. The citation guard (`report/guard.py`) is code, and its checks
are word-level: a number, a name or a who-was-studied word the cited text does
not contain drops a sentence, and so do a few direction and negation flips it
knows the words for. What no string check can see is a sentence that uses the
quote's words and says something else with them - "IL-23 blockade prevented
psoriasis" citing a quote that it reduced ear thickness in mice by a third.
`docs/litreview/OPEN_ISSUES.md` #10 is that gap. This module asks the local
model the one question the guard cannot answer: does the paper's text support
this sentence fully, partially, not at all, or say the opposite?

How it is wired, and why each choice:

- It runs in Stage 7 (`stages/s7_write.write_section_text`), once per sentence
  the guard KEPT, never on one the guard dropped: the guard's code checks come
  first and are never overruled by a model, only added to.
- It sees the paper's own words a reader sees beside the sentence -
  `guard.source_text` of each cited finding (quote, context sentence,
  qualifying next sentence) - and the finding's "about" name, which the writer
  saw too. Never `effect_detail` or any other model-written summary.
- One structured call per sentence: JSON schema, temperature 0, through the
  run's own client - `SchedulerClient` at `Priority.USER_JOB` behind the call
  cache - on the model `models.yaml` pins for the `entail` job, with its own
  versioned prompt (`llm/prompts/entail.md`). A cached rerun therefore makes
  the same decisions, and the run records `entail` in its `model_versions`
  and `prompt_versions` so a stored report says the check ran.
- Code decides what an answer does:
    supported       kept
    partial         kept, and counted (`sentences_entailment_partial` in the
                    report's stats). Not marked on the sentence: an unmeasured
                    small model's "partial" is not something to show a reader
                    as a finding about the sentence, and a new field on
                    `Sentence` would change every stored report's bytes
    not_supported   dropped, reason "the entailment check found the cited
    contradicted    text does not support it: <the model's reason>"
  A call that fails (a malformed reply, a cut-off answer) leaves the sentence
  as the guard decided and is counted as `sentences_entailment_failed`: the
  check only ever removes, so being unable to ask it must not remove anything.
  A server that has gone away still raises `ReviewInterrupted` and stops the
  run, as everywhere else.
- A dropped sentence is stored with `kept=False` like any other dropped
  sentence, so `report/invariants.check_sentences` - a pure-code re-check that
  re-runs the guard over KEPT sentences only - never needs the model and
  never disagrees with the run: it simply does not see those sentences.

Why off by default: a false drop loses real evidence, and nobody has yet
measured how often this check is wrong. `evals/litreview/entailment_agreement.py`
compares its verdicts with the strong judge's on stored reports; it stays off
until that and a scientist's review say it can be trusted. Turn it on with
`entailment_check = true` in the `[litreview]` table of
`~/.config/bedrock/config.toml`, or `BEDROCK_LITREVIEW_ENTAILMENT_CHECK=1`.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ValidationError

from bedrock.litreview.llm import LLMError, ask_json, load_prompt, strict_schema
from bedrock.litreview.models import EvidenceUnit
from bedrock.litreview.report.guard import source_text

log = logging.getLogger(__name__)

PROMPT = "entail"
#: The models.yaml job whose model answers the check.
JOB = "entail"
#: A one-line reason and a label; far more than enough.
MAX_TOKENS = 300
#: The answers that remove a sentence. Everything else keeps it.
DROPS = ("not_supported", "contradicted")
DROP_REASON = "the entailment check found the cited text does not support it: "
#: A reason is the model's own words, shown in the report's list of removed
#: sentences; one line, and not a paragraph.
MAX_REASON_CHARS = 200

Label = Literal["supported", "partial", "not_supported", "contradicted"]


class EntailmentAnswer(BaseModel):
    """What the model returns. `reason` comes first so it compares before it labels."""

    reason: str
    verdict: Label


@dataclass
class EntailmentVerdict:
    label: Label | None
    """None when the call failed; the sentence then stays as the guard left it."""
    reason: str = ""

    @property
    def drops(self) -> bool:
        return self.label in DROPS

    @property
    def drop_reason(self) -> str:
        return DROP_REASON + (self.reason or self.label or "no reason given")


def evidence_text(cited: dict[str, EvidenceUnit]) -> str:
    """The paper's text the model is shown: per cited finding, its short id,
    what it is about, and `guard.source_text` - the same words the guard
    checks and the reader sees."""
    return "\n".join(
        f'[{sid}] about: {u.target_raw or u.target}\n    "{source_text(u)}"' for sid, u in cited.items()
    )


def _one_line(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= MAX_REASON_CHARS else text[: MAX_REASON_CHARS - 1].rstrip() + "…"


class EntailmentCheck:
    """One narrow question per sentence, through the run's own client."""

    def __init__(self, client, model: str) -> None:
        self.client = client
        self.model = model

    def check(self, sentence: str, cited: dict[str, EvidenceUnit]) -> EntailmentVerdict:
        """`cited`: short id -> finding, for the ids the sentence cites."""
        prompt = load_prompt(PROMPT).render(sentence=sentence, evidence=evidence_text(cited))
        schema = strict_schema(EntailmentAnswer.model_json_schema())
        try:
            answer = EntailmentAnswer.model_validate(
                ask_json(self.client, prompt, model=self.model, schema=schema, max_tokens=MAX_TOKENS)
            )
        except (LLMError, ValidationError) as exc:
            log.warning("entailment check failed on %r: %s", sentence[:60], exc)
            return EntailmentVerdict(None)
        return EntailmentVerdict(answer.verdict, _one_line(answer.reason))

    def review(self, sentence: str, cited: dict[str, EvidenceUnit], stats: Counter) -> str | None:
        """Check one sentence the guard kept and count the outcome in `stats`.
        Returns the drop reason when the sentence must go, else None."""
        verdict = self.check(sentence, cited)
        if verdict.label is None:
            stats["entailment_failed"] += 1
            return None
        stats["entailment_checked"] += 1
        if verdict.drops:
            stats["entailment_dropped"] += 1
            return verdict.drop_reason
        if verdict.label == "partial":
            stats["entailment_partial"] += 1
        return None


def report_stats(stats: Counter) -> dict[str, int]:
    """The check's counts as they go into `Report.stats` - only when it ran,
    so a report written with it off is byte-identical to one written before
    it existed."""
    return {
        "sentences_entailment_checked": stats.get("entailment_checked", 0),
        "sentences_entailment_dropped": stats.get("entailment_dropped", 0),
        "sentences_entailment_partial": stats.get("entailment_partial", 0),
        "sentences_entailment_failed": stats.get("entailment_failed", 0),
    }

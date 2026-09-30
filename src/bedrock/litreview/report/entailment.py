"""The entailment check — an optional, local, model-based second look at a
report sentence, after the citation guard. OFF BY DEFAULT.

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
- It sees the paper's own words a reader sees beside the sentence, and the
  finding's "about" name, which the writer saw too; never `effect_detail` or
  any other model-written summary. The quote is shown APART from the
  sentences beside it (the context sentence, the qualifying next sentence,
  the naming sentence), each labelled with its role, and the model is told
  that a sentence's main claim must be stated in a quote: the others may add
  a name, who was studied, a sample size or a limit, never the claim. This is guard check 10 (`guard.rests_on_quote`)
  asked as a question of meaning. Prompt version 1 showed all four sentences
  as one passage, so a sentence copied from a finding's qualifier while
  citing a quote about something else - OPEN_ISSUES #3, the failure class the
  second hand audit found - was "supported": every word of it really is the
  paper's. Check 10 catches that shape by counting words, against a tuned
  threshold; this is the second look behind it, for a sentence that rewords
  the borrowed sentence enough to get past the count.
- One structured call per sentence: JSON schema, temperature 0, through the
  run's own client - `SchedulerClient` at `Priority.USER_JOB` behind the call
  cache - on the model `models.yaml` pins for the `entail` job, with its own
  versioned prompt (`llm/prompts/entail.md`). A cached rerun therefore makes
  the same decisions, and the run records `entail` in its `model_versions`
  and `prompt_versions` so a stored report says the check ran.
- Code decides what an answer does:
    supported       kept
    partial         kept, and counted (`sentences_entailment_partial` in the
                    report's stats). Not marked on the sentence: a small
                    model's "partial" is not something to show a reader as a
                    finding about the sentence (on the hand audit, 35 of 105
                    sentences moved between "supported" and "partial" when
                    only the prompt changed), and a new field on `Sentence`
                    would change every stored report's bytes
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

Why off by default: a false drop loses real evidence. Two tools measure it.
`evals/litreview/entailment_audit.py` compares its verdicts with the hand
audits (105 graded sentences, including every OPEN_ISSUES #3 case), and
`evals/litreview/entailment_agreement.py` with the strong judge on stored
benchmark reports; `docs/litreview/BENCHMARK_RESULTS.md` has the numbers. On
the laptop model (qwen3:8b, 30 September 2026) it dropped none of the hand
audit's 93 faithful sentences and 2 of its 12 unfaithful ones, and none of
the five OPEN_ISSUES #3 cases: safe on that sample, but it adds little. It
stays off until a scientist has marked the sentences it drops (the audit
tool's `--sheet`) and it has been measured on the office model. Turn it on with `entailment_check = true` in the
`[litreview]` table of `~/.config/bedrock/config.toml`, or
`BEDROCK_LITREVIEW_ENTAILMENT_CHECK=1`.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ValidationError

from bedrock.litreview.llm import LLMError, ask_json, load_prompt, strict_schema
from bedrock.litreview.models import EvidenceUnit

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


#: The sentences shown beside a quote (`guard._shown`), in the order a reader
#: meets them, each with the name the prompt gives its role. They may give a
#: sentence a name, who was studied, a sample size or a limit; its claim must
#: come from the quote.
BESIDE = (
    ("context", "context_quote"),
    ("next sentence", "qualifier_quote"),
    ("naming sentence", "naming_quote"),
)


def evidence_text(cited: dict[str, EvidenceUnit]) -> str:
    """The paper's text the model is shown, per cited finding: its short id,
    what it is about, the quote, and then the other sentences a reader sees
    beside it, each labelled with its role. Apart rather than joined, because
    the prompt asks whether the sentence's claim is in a quote, and a model
    shown one run-on passage cannot tell (OPEN_ISSUES #3)."""
    blocks = []
    for sid, u in cited.items():
        lines = [f"[{sid}] about: {u.target_raw or u.target}", f'    Quote: "{u.quote}"']
        beside = [(role, getattr(u, name)) for role, name in BESIDE if getattr(u, name)]
        if beside:
            lines.append("    Shown beside the quote, for names, who was studied, a sample size or a limit only:")
            lines += [f'    - {role}: "{text}"' for role, text in beside]
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


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

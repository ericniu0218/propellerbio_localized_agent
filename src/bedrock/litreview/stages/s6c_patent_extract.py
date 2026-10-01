"""Stage 6c - read each patent filing once and copy what it says it covers.

Same shape as papers - model copies, code verifies and decides - and the same
rules, applied to what a patent actually is. A patent reports no results. It says
what the filer *claims or describes*, and whether that works is not the patent's
to say. So everything here is a statement about the filing, never about biology,
and the line `sources/patents_base.py` draws stays where it was: patents are never
evidence, they feed the "Competitor landscape" section and nothing else. The model
may now copy what a filing says; it may still not judge it.

What the model is asked (`llm/prompts/patent_extract.md`): one verbatim quote, the
target, the kind of thing claimed (`modality`), a few words copied for what it
covers, and the indication. What *code* does, because code gives the same answer
every time:

- finds the quote in the filing's own text and stores the filing's words at their
  offsets (`extract/verify.locate_quote`) - never the model's copy. No match, the
  finding is destroyed.
- finds the target in the text. Not named there, the finding is destroyed.
- decides whether the target is the filing's *subject* (named in its title, its
  first claim or the first sentence of its abstract, without a list of other
  targets beside it) or a *mention* (listed among others). The paper
  pipeline calls this `target_is_primary` and lets the model say; here code says.
  Mentions are kept for the record and never summarised.
- blanks `covers` and `indication` when their words are not in the text.
- reads the kind of claim off claim 1's own first words (method of treatment,
  composition, use, process), when there is a claim 1.
- checks the modality against the text, so "antibody" for a filing that never says
  antibody becomes `other`.
- skips a filing whose text is not English, and says so. A model reading Chinese
  and a verifier matching it are two more ways to be wrong; the English abstract
  EPO supplies for most CN/KR/JP filings is usually enough to extract from.

Text read: title, abstract, and claim 1 where EPO has it (WO, EP, CA in practice).
Never the description or the other claims. Each unit says which it was.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import Callable, Literal

from pydantic import BaseModel, Field, ValidationError

from bedrock.litreview.extract.verify import locate_quote, normalize, split_sentences
from bedrock.litreview.knowledge import Knowledge, TermMatcher
from bedrock.litreview.llm import LLMError, ask_json, load_prompt, strict_schema
from bedrock.litreview.models import Modality, PatentRecord, QuestionSpec, stable_id

log = logging.getLogger(__name__)

PROMPT = "patent_extract"
ARTIFACT = "patent_findings.v1"
DEFAULT_EXTRACT_MAX = 40

QuoteSource = Literal["claim_1", "abstract", "title"]
ClaimKind = Literal["method_of_treatment", "method", "composition", "use", "not_stated"]
Basis = Literal["claim_1_and_abstract", "abstract_only"]
Subject = Literal["subject", "mention"]


class ExtractedPatentFinding(BaseModel):
    """One finding exactly as the model returns it; compiled into the JSON schema
    handed to guided decoding."""

    quote: str = Field(max_length=700, description="Verbatim: one abstract sentence or a stretch of claim 1")
    source: QuoteSource
    target: str = Field(max_length=120, description="As the text writes it")
    modality: Modality
    covers: str = Field(max_length=200, description="Words copied from the text: what the filing covers")
    indication: str | None = Field(default=None, max_length=160, description="As written, or null")


class PatentExtraction(BaseModel):
    findings: list[ExtractedPatentFinding] = Field(default_factory=list, max_length=3)


class PatentUnit(BaseModel):
    """A verified statement about one filing: what it says it covers."""

    id: str
    patent_id: str
    family_id: str | None = None
    target: str
    target_raw: str
    modality: Modality
    covers: str | None = None
    indication: str | None = None
    claim_kind: ClaimKind = "not_stated"
    subject: Subject
    quote: str
    """The filing's own words at `quote_start:quote_end` of `quote_source`'s text,
    never the model's copy."""
    quote_source: QuoteSource
    quote_start: int
    quote_end: int
    quote_status: Literal["exact", "fuzzy"]
    basis: Basis
    corrected: list[str] = Field(default_factory=list)
    extracted_by_model: str
    prompt_version: str


class PatentReading(BaseModel):
    """Everything Stage 6c decided, saved as the `patent_findings.v1` artifact."""

    units: list[PatentUnit] = Field(default_factory=list)
    skipped: dict[str, str] = Field(default_factory=dict)
    """patent id -> why it was not read ("not English", "over the limit", ...)."""
    stats: dict[str, int] = Field(default_factory=dict)
    short_ids: dict[str, str] = Field(default_factory=dict)
    """unit id -> "P1": what the writing model cites. Subjects only."""

    def subjects(self) -> list[PatentUnit]:
        return [u for u in self.units if u.subject == "subject"]

    def by_short_id(self) -> dict[str, PatentUnit]:
        by_id = {u.id: u for u in self.units}
        return {short: by_id[uid] for uid, short in self.short_ids.items()}


# --- the text a filing is read from ----------------------------------------------


_ENGLISH_WORDS = re.compile(r"\b(the|of|and|to|for|in|is|a|an|comprising|wherein|method|use|with|said|by)\b", re.I)
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


def looks_english(text: str | None, *, min_words: int = 8) -> bool:
    """Whether the text is English, by counting English function words. Crude on
    purpose: it only has to separate English from Chinese, Korean, Japanese,
    French and German, and it errs towards skipping."""
    if not text:
        return False
    letters = [c for c in text if not c.isspace()]
    if letters and len(_CJK.findall(text)) / len(letters) > 0.1:
        return False
    words = re.findall(r"[^\W\d_]+", text)
    if len(words) < min_words:
        return False
    return len(_ENGLISH_WORDS.findall(text)) >= max(2, len(words) // 15)


def texts(record: PatentRecord) -> dict[str, str]:
    """The text a filing is read from. Claim 1 counts only when it is English: a WO
    filing often has an English abstract above a Chinese claim 1, and a quote in
    Chinese in an English report, found by a verifier and a model that were both
    built for English, is a result nobody can check."""
    claim = record.claim_1 if looks_english(record.claim_1, min_words=4) else ""
    return {"title": record.title or "", "abstract": record.abstract or "", "claim_1": claim}


def is_english(record: PatentRecord) -> bool:
    return looks_english(f"{record.title} {record.abstract or ''}")


_KIND_CUES = (
    ("method_of_treatment", re.compile(r"^(?:an?\s+|the\s+)?(?:\w+\s+){0,3}?(?:method|process)\s+(?:of|for)\s+"
                                       r"(?:\w+\s+){0,3}?(?:treat|prevent|prophyla|ameliorat|manag|reduc|inhibit|alleviat)", re.I)),
    ("use", re.compile(r"^(?:the\s+)?use\s+of\b|^(?:\w+\s+){0,6}?for\s+use\s+in\b", re.I)),
    ("method", re.compile(r"^(?:an?\s+|the\s+)?(?:\w+\s+){0,3}?(?:method|process)\b", re.I)),
)


def claim_kind(claim_1: str | None) -> ClaimKind:
    """What kind of claim claim 1 is, from its own first words. Nothing before it
    to read (no claim 1) is "not_stated", never a guess from the abstract."""
    if not claim_1:
        return "not_stated"
    head = claim_1.strip()
    for kind, pattern in _KIND_CUES:
        if pattern.match(head):
            return kind  # type: ignore[return-value]
    return "composition"


_MODALITY_CUES: dict[str, re.Pattern[str]] = {
    "antibody": re.compile(r"antibod|immunoglobulin|\bigg\d?\b|nanobod|\bscfv\b|\bfab\b|bispecific|monoclonal|vhh", re.I),
    "peptide": re.compile(r"peptid|macrocycl|cyclic", re.I),
    "siRNA": re.compile(r"sirna|rnai|antisense|oligonucleotide|\brna\b|\bshrna\b", re.I),
    "small_molecule": re.compile(r"compound|small molecule|inhibitor|formula|salt thereof", re.I),
    "vaccine": re.compile(r"vaccin|immuniz|antigen", re.I),
    "genetic": re.compile(r"\bgene\b|crispr|vector|transgen|knock", re.I),
}


def modality_supported(modality: str, text: str) -> bool:
    cue = _MODALITY_CUES.get(modality)
    return True if cue is None else bool(cue.search(text))


def modality_from_text(text: str) -> str:
    """The one kind of thing the text's own words point to, else "other". Used to
    replace a modality the text does not support: with exactly one kind of cue
    in the text there is nothing to guess, and with several (an antibody-drug
    conjugate, a peptide-antibody fusion) the model's pick would be a guess too."""
    found = [name for name, cue in _MODALITY_CUES.items() if cue.search(text)]
    return found[0] if len(found) == 1 else "other"


_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]{3,}")


def found_in(value: str | None, text: str) -> bool:
    """Every word of `value` of four letters or more is in `text`. How `covers`
    and `indication` earn their place: words the model copied are in the text,
    words it made up are not."""
    if not value:
        return False
    haystack = normalize(text).lower()
    return all(t.lower() in haystack for t in _TOKEN.findall(normalize(value)))


# --- the assignee, so "JANSSEN PHARMACEUTICA NV [BE]" and "Janssen Pharmaceutica N.V." are one filer ----

_COUNTRY_TAG = re.compile(r"\s*\[[A-Z]{2}\]\s*$")
_SUFFIX = re.compile(r"\b(?:inc|incorporated|ltd|limited|llc|plc|corp|corporation|co|company|gmbh|ag|nv|bv|sa|sas|"
                     r"kk|kabushiki|kaisha|ab|as|spa|srl|lp|pty|oy|sarl|aps|s a|n v|b v)\b\.?", re.I)


def assignee_key(raw: str | None) -> str:
    """A grouping key for a filer: case, punctuation, country tags and corporate
    suffixes removed. Only ever used to *group*; the name shown is the filer's own."""
    if not raw:
        return ""
    name = _COUNTRY_TAG.sub("", raw).replace(".", "")  # "N.V." and "NV" are the same suffix
    name = _SUFFIX.sub(" ", name)
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def assignee_name(raw: str | None) -> str:
    """The name as shown: the register's own, minus the country tag and tidied case."""
    if not raw:
        return "Unknown / unassigned"
    name = _COUNTRY_TAG.sub("", raw).strip()
    return name.title() if name.isupper() else name


# --- the stage -----------------------------------------------------------------------


def _locate(quote: str, source: str, record: PatentRecord):
    """The quote in the text it claims to come from, else in the others (a model
    that says "abstract" for a sentence of claim 1 has still copied it)."""
    order = [source] + [s for s in ("claim_1", "abstract", "title") if s != source]
    all_texts = texts(record)
    for name in order:
        text = all_texts.get(name) or ""
        if not text:
            continue
        found = locate_quote(quote, text, widen=(name == "abstract"))
        if found.status != "failed":
            return name, found
    return None, None


def read_patent(record: PatentRecord, spec: QuestionSpec, kb: Knowledge, client, model: str,
                targets: list[str], stats: Counter) -> list[PatentUnit]:
    """One model call for one filing, then every check above."""
    claim = texts(record)["claim_1"]
    claim_block = (f"First claim: {claim}" if claim else
                   "First claim: (not available for this filing - read from the title and abstract only)")
    prompt = load_prompt(PROMPT).render(
        question=spec.text, targets=", ".join(targets) or "the question's own terms",
        title=record.title, abstract=record.abstract or "(none)", claim_block=claim_block)
    schema = strict_schema(PatentExtraction.model_json_schema())
    try:
        reply = PatentExtraction.model_validate(ask_json(client, prompt, model=model, schema=schema, max_tokens=1500))
    except (LLMError, ValidationError) as exc:
        log.warning("reading patent %s failed: %s", record.id, exc)
        stats["model_failed"] += 1
        return []

    full_text = " ".join(texts(record).values())  # claim 1 only when English
    basis: Basis = "claim_1_and_abstract" if claim else "abstract_only"
    best: dict[str, PatentUnit] = {}
    for finding in reply.findings:
        source, found = _locate(finding.quote, finding.source, record)
        if found is None:
            stats["quote_failed"] += 1
            continue
        canonical = kb.normalize(finding.target, "target")
        entity = kb.get(canonical)
        matcher = entity.matcher() if entity else TermMatcher([finding.target])
        if not matcher.exact_count(full_text):
            stats["target_not_in_text"] += 1
            continue
        corrected: list[str] = []
        if entity is None and len(record.matched_targets) == 1:
            # A name the knowledge table does not know, in a filing that was found
            # by searching for exactly one target: "IL-23pl9" is how a scanned
            # filing spells "IL-23p19", and counting it as a target of its own
            # would split one landscape in two. The name was verified to be in the
            # text, and the search that found the filing was for this target only.
            corrected.append(f"target: {canonical} -> {record.matched_targets[0]} "
                             "(the only target this filing was searched for)")
            canonical = record.matched_targets[0]
            entity = kb.get(canonical)
            matcher = entity.matcher() if entity else matcher
            raw_matcher = TermMatcher([finding.target])
        else:
            raw_matcher = None
        modality = finding.modality
        if not modality_supported(modality, full_text):
            replacement = modality_from_text(full_text)
            corrected.append(f"modality: {modality} -> {replacement} (the text never says {modality})")
            modality = replacement
        covers = finding.covers.strip() if found_in(finding.covers, full_text) else None
        if finding.covers.strip() and covers is None:
            corrected.append("covers: blanked (its words are not in the text)")
        indication = finding.indication.strip() if finding.indication and found_in(finding.indication, full_text) else None
        if finding.indication and indication is None:
            corrected.append("indication: blanked (its words are not in the text)")
        is_subject = names_target_as_subject(matcher, canonical, record, kb) or (
            raw_matcher is not None and names_target_as_subject(raw_matcher, canonical, record, kb))
        unit = PatentUnit(
            id=stable_id("patent", record.id, canonical), patent_id=record.id, family_id=record.family_id,
            target=canonical, target_raw=finding.target, modality=modality, covers=covers, indication=indication,
            claim_kind=claim_kind(claim or None), subject="subject" if is_subject else "mention",
            quote=found.text, quote_source=source, quote_start=found.start, quote_end=found.end,
            quote_status=found.status, basis=basis, corrected=corrected,
            extracted_by_model=model, prompt_version=str(load_prompt(PROMPT).version))
        # One statement per filing and target. The model sometimes returns several
        # about the same one (an English abstract sentence and a claim), and each
        # would carry the same citation number: keep the one from claim 1, which says
        # what is actually claimed, else the first.
        held = best.get(canonical)
        if held is None or (held.quote_source != "claim_1" and unit.quote_source == "claim_1"):
            best[canonical] = unit
    for unit in best.values():
        stats[unit.subject] += 1
    return list(best.values())


def names_target_as_subject(matcher: TermMatcher, canonical: str, record: PatentRecord, kb: Knowledge) -> bool:
    """Whether the filing is directed at this target rather than listing it.

    The paper pipeline lets the model say whether a target is the paper's
    subject (`target_is_primary`). Code says here, from where the filing names it:
    in its title, in claim 1, or in the first sentence of its abstract (where "The
    invention relates to ..." says what it is). And in a place that does not also
    name more than one *other* target: "cytokines selected from IL-6, IL-17 and
    IL-23" names IL-23 without being about it, while a bispecific antibody
    against two targets is about both.
    """
    first_sentence = (split_sentences(record.abstract or "") or [""])[0]
    for place in (record.title, texts(record)["claim_1"], first_sentence):
        if place and matcher.exact_count(place):
            others = {e.canonical for e, _ in kb.scan(place, ("target",))} - {canonical}
            if len(others) <= 1:
                return True
    return False


def subject_targets(record: PatentRecord, kb: Knowledge, names: list[str]) -> list[str]:
    """The targets (of `names`) the filing names up front, by `names_target_as_subject`."""
    out = []
    for name in names:
        entity = kb.get(name)
        if names_target_as_subject(entity.matcher() if entity else TermMatcher([name]), name, record, kb):
            out.append(name)
    return out


def order_key(unit: PatentUnit, records: dict[str, PatentRecord], target_order: list[str]):
    r = records[unit.patent_id]
    rank = target_order.index(unit.target) if unit.target in target_order else len(target_order)
    return (rank, assignee_key(r.assignee), str(r.priority_date or "9999"), r.id)


def number_units(units: list[PatentUnit], records: dict[str, PatentRecord], target_order: list[str]) -> dict[str, str]:
    """P1, P2, ... for the filings that are about a target, in a fixed order: the
    targets in Stage 6's order, then filer, then priority date. One unit per filing
    and target, so a filing that names two targets is cited as two."""
    subjects = sorted((u for u in units if u.subject == "subject"), key=lambda u: order_key(u, records, target_order))
    return {u.id: f"P{i}" for i, u in enumerate(subjects, 1)}


def load_reading(db, run_id: str) -> PatentReading | None:
    body = db.get_artifact(run_id, ARTIFACT)
    return PatentReading.model_validate_json(body) if body else None


def extract_patents(
    patents: list[PatentRecord], *, db, run_id: str, spec: QuestionSpec, kb: Knowledge, client, model: str,
    target_order: list[str], limit: int, say: Callable[[str], None], tick=None,
) -> PatentReading:
    """Read the filings (at most `limit`, in the order EPO ranked them), keep what
    verifies, and save it. A resumed run reuses what it saved: the answers would be
    the same (the model is cached at temperature 0) but this does not spend the
    time to find that out."""
    saved = load_reading(db, run_id)
    if saved is not None:
        say(f"patents: reusing the {len(saved.subjects())} filing statements this run already read")
        return saved

    records = {p.id: p for p in patents}
    stats: Counter = Counter()
    skipped: dict[str, str] = {}
    units: list[PatentUnit] = []
    readable = []
    for record in patents:
        if not is_english(record):
            skipped[record.id] = "not English: listed, not read"
        else:
            readable.append(record)
    # Reading costs a model call each, so when there are more than `limit` the ones
    # to leave out are the ones that look least like they are about a target: named
    # up front first, then the rest, each in the order EPO ranked them. Under the
    # limit this changes nothing.
    likely = [r for r in readable if subject_targets(r, kb, r.matched_targets or target_order)]
    unlikely = [r for r in readable if r not in likely]
    todo = (likely + unlikely)[:limit]
    for record in (likely + unlikely)[limit:]:
        skipped[record.id] = f"over the limit of {limit} filings read"
    for n, record in enumerate(todo, 1):
        if tick is not None:
            tick("patents", n - 1, len(todo), record.id)
        units.extend(read_patent(record, spec, kb, client, model,
                                 record.matched_targets or target_order, stats))
    if tick is not None:
        tick("patents", len(todo), len(todo))
    stats["read"], stats["skipped"] = len(todo), len(skipped)
    reading = PatentReading(units=units, skipped=skipped, stats=dict(stats),
                            short_ids=number_units(units, records, target_order))
    db.put_artifact(run_id, ARTIFACT, reading.model_dump_json())
    say(f"patents: read {len(todo)} filings - {stats['subject']} are about a target, {stats['mention']} only "
        f"mention one, {len(skipped)} skipped, {stats['quote_failed']} quotes failed verification")
    return reading

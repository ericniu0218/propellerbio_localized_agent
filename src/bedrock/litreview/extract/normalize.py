"""Turn what the paper said into what the database stores.

The model's job is to copy. This file's job is to translate. Keeping those
separate is deliberate: copying is something a small model does reliably, and
translating is something code does *deterministically*, which matters because
the client's loudest complaint is getting different answers to the same
question. A lookup table returns the same answer every time; a model does not.

So the model returns "C57BL/6 mice" and this file returns "mouse". The model
returns "interleukin-23", "IL-23" or "p19" and this file returns "IL23A" —
using the same alias tables Stage 1 built for searching, so that a paper found
by searching `IL23A` and a finding extracted as `IL-23` end up in the same row
of the same report.

When normalisation fails, we keep the raw value and carry on. A finding with an
unrecognised species is still a finding; dropping it would lose real evidence
because of a missing dictionary entry.
"""

from __future__ import annotations

import re

# --- species ----------------------------------------------------------------

_SPECIES_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(human|patient|volunteer|participant|subject|men|women)", re.I), "human"),
    (re.compile(r"\b(mouse|mice|murine|c57bl|balb/?c|nod scid|knockout mice)", re.I), "mouse"),
    (re.compile(r"\b(rat|rats|sprague|wistar|lewis rat)", re.I), "rat"),
    (re.compile(r"\b(cynomolgus|cyno|macaque|rhesus|non-?human primate|nhp)", re.I), "cyno"),
    (re.compile(r"\b(zebrafish|danio)", re.I), "zebrafish"),
    (re.compile(r"\b(pig|porcine|swine|minipig)", re.I), "pig"),
    (re.compile(r"\b(dog|canine|beagle)", re.I), "dog"),
    (re.compile(r"\b(rabbit|oryctolagus)", re.I), "rabbit"),
    # Cell lines last: "HEK293 cells from a human donor" is a cell line, not a
    # human subject, but the human pattern would have fired first. Ordering
    # here is load-bearing — see test_normalize.py.
    (re.compile(r"\b(cell line|cells|in vitro|hek|hela|jurkat|thp-?1|caco-?2|primary culture)", re.I), "cell_line"),
]


# Markers that mean "this happened in a dish", checked before the organism
# patterns below. A human-derived cell line is in-vitro evidence, not clinical,
# and getting that backwards turns the weakest evidence class into the
# strongest one.
#
# Deliberately narrower than a bare /cells?/: "CD4+ T cell counts rose in
# patients" is a clinical observation and must not be caught here. We require a
# cell-work context — a named line, or cells being isolated/derived/cultured.
_IN_DISH = re.compile(
    r"\b(cell\s+lines?|cells?\s+(were|was|are|is|from|derived|isolated|treated|cultured|incubated)"
    r"|in\s+vitro|cultures?|cultured"
    r"|hek\s*-?\s*293|hela|jurkat|thp-?1|caco-?2|raw\s*264|cho\s+cells?|a549|u937)",
    re.I,
)


def normalize_species(raw: str) -> str:
    """'C57BL/6 mice' -> 'mouse'. Unknown input returns 'other'."""
    if not raw:
        return "other"
    if _IN_DISH.search(raw):
        return "cell_line"
    for pattern, canonical in _SPECIES_PATTERNS:
        if pattern.search(raw):
            return canonical
    return "other"


# --- targets ----------------------------------------------------------------

_PUNCT_STRIP = re.compile(r"[\s\-_/.]")


def _fold(name: str) -> str:
    """'IL-23' and 'il 23' and 'IL_23' all fold to 'il23' for lookup."""
    return _PUNCT_STRIP.sub("", name).lower()


def normalize_target(raw: str, alias_table: dict[str, list[str]]) -> str:
    """Map a molecule name onto the canonical symbol for this project.

    `alias_table` is the approved map from ProjectSpec — canonical symbol to
    its accepted synonyms. It is built in Stage 1 and approved by a human, so
    this function never has to guess, and two runs of the same project always
    resolve a name the same way.

    Returns the raw string unchanged when nothing matches. That is not an
    error: papers mention molecules outside the project's scope all the time,
    and Stage 6 filters on the canonical names it knows about.
    """
    if not raw:
        return raw
    folded = _fold(raw)
    for canonical, aliases in alias_table.items():
        if folded == _fold(canonical):
            return canonical
        if any(folded == _fold(alias) for alias in aliases):
            return canonical
    return raw


# --- diseases ---------------------------------------------------------------


def normalize_disease(raw: str, disease_table: dict[str, list[str]]) -> str:
    """Same contract as `normalize_target`, against the project's disease map.

    Kept as its own function rather than sharing one, because disease names
    will eventually need MeSH tree lookups (so 'plaque psoriasis' rolls up to
    'psoriasis') and targets will not. Splitting them now avoids a later
    untangling.
    """
    if not raw:
        return raw
    folded = _fold(raw)
    for canonical, aliases in disease_table.items():
        if folded == _fold(canonical):
            return canonical
        if any(folded == _fold(alias) for alias in aliases):
            return canonical
    return raw


# --- sample size ------------------------------------------------------------

_N_PATTERN = re.compile(r"\bn\s*[=:]\s*(\d[\d,]*)", re.I)


def extract_n(text: str) -> int | None:
    """Pull 'n = 507' out of free text.

    The model is also asked for `n` directly; this is a backstop for when it
    returns null but the number is sitting in the sentence. Code is better at
    this than a model, and cheaper.
    """
    if not text:
        return None
    match = _N_PATTERN.search(text)
    if not match:
        return None
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return None


# --- consistency checks -----------------------------------------------------


def infer_evidence_type(species: str, stated: str) -> str:
    """Cross-check the model's evidence_type against the normalised species.

    The most common extraction error by a wide margin is a mouse study labelled
    human_clinical — it is the failure mode the gold set exists to catch. The
    species string is usually more reliable than the label, because it is
    copied from the text rather than inferred, so where they disagree we trust
    the species and let the eval harness report the correction rate.

    A rising correction rate means the prompt needs work; it should not stay
    high, because silently patching a model error is a worse outcome than
    fixing the prompt that caused it.
    """
    if species == "cell_line":
        return "in_vitro"
    if species == "human" and stated == "in_vivo":
        return "human_clinical"
    if species in {"mouse", "rat", "cyno", "zebrafish", "pig", "dog", "rabbit"}:
        # An animal is never human evidence, clinical or genetic.
        if stated in ("human_clinical", "human_genetic"):
            return "in_vivo"
    return stated


# --- direction and outcome ----------------------------------------------------
#
# The model copies two phrases out of its quote: what was done to the target
# ("deficiency", "knockout", "overexpression") and what happened to the disease
# ("aggravated", "did not significantly reduce"). These tables classify them.
# The model also picks `direction` and `outcome` itself; code overrides that
# choice only when the copied words are in the quote and unambiguously say
# otherwise. MedGemma 4B read "PGGT1B deficiency aggravated the psoriasis-like
# lesions" as direction `unclear`, outcome `improved` — the opposite of the
# finding, and exactly what a report must never say.

_BLOCK_WORDS = re.compile(
    r"(deficien|knock[\s-]?out|knock[\s-]?down|\bKO\b|-/-|delet|ablat|\bloss\b|\black|silenc|inhibit|"
    r"block|neutrali[sz]|antagonis|deplet|abrogat|disrupt|null\b|\banti-)", re.I)
_ACTIVATE_WORDS = re.compile(
    r"(over-?express|agonis|gain[\s-]of[\s-]function|supplement|administration of recombinant|"
    r"\brecombinant\b|\bactivat)", re.I)

_NO_EFFECT_WORDS = re.compile(
    r"((?:did|does|do|was|were|could) not|didn't|failed to)\s+(?:significantly\s+|further\s+)?"
    r"(reduc|decreas|alter|affect|chang|improv|attenuat|prevent|protect|differ|amelior|alleviat|influenc|"
    r"impact|modif|rescu|aggravat|exacerbat|worsen|increas|enhanc)"
    r"|no significant|not significantly|no (?:detectable |measurable )?effect|no difference|no change|"
    r"unchanged|unaltered|comparable|similar to", re.I)
_WORSE_WORDS = re.compile(
    r"(aggravat|exacerbat|worse|more severe|enhanc|increas|promot|accelerat|intensif|augment|exaggerat|"
    r"thicker|earlier onset|higher incidence|susceptib)", re.I)
_BETTER_WORDS = re.compile(
    r"(reduc|decreas|attenuat|ameliorat|alleviat|suppress|prevent|inhibit|mitigat|reliev|resolv|"
    r"lower|less\b|milder|diminish|abolish|abrogat|improv)", re.I)
# Words whose presence flips what "increased" or "reduced" means: increased
# survival is good, reduced survival is bad. Any of these, and code does not
# overrule the model.
_POLARITY_WORDS = re.compile(
    r"(surviv|titer|titre|antibod|immun|respons|protect|clearance|efficac|remission|resist|toleran|"
    r"weight|barrier|repair|healing)", re.I)


def classify_direction(change: str) -> str | None:
    """'block' or 'activate' when the copied words say so unambiguously."""
    block, activate = bool(_BLOCK_WORDS.search(change)), bool(_ACTIVATE_WORDS.search(change))
    if block and not activate:
        return "block"
    if activate and not block:
        return "activate"
    return None


def classify_outcome(effect: str) -> str | None:
    """'no_effect', 'worsened' or 'improved' when the copied words say so
    unambiguously. Negation is checked first: "did not reduce" contains
    "reduce" — but "did not develop lesions" is not a null result, so only a
    negated *effect* verb counts. Words from both lists, or a word that flips
    polarity ("reduced survival"), decide nothing."""
    if _NO_EFFECT_WORDS.search(effect):
        return "no_effect"
    if _POLARITY_WORDS.search(effect):
        return None
    worse, better = bool(_WORSE_WORDS.search(effect)), bool(_BETTER_WORDS.search(effect))
    if worse and not better:
        return "worsened"
    if better and not worse:
        return "improved"
    return None


# --- setting ------------------------------------------------------------------
#
# Was the intervention given to prevent the disease, or to treat it? MedGemma
# 4B labelled 74 of 74 RSV findings `unclear` or `treatment`, including
# "Nirsevimab ... has shown high efficacy in RSV prophylaxis in newborns", so
# a prevention question got a report with no prevention evidence in it. The
# words are in the sentence the model already copied; code reads them.

_PROPHYLAXIS = re.compile(
    r"\b(prophyla\w*|immunoprophyla\w*|vaccinat\w*|vaccines?|immuni[sz]\w*|pre-?exposure|"
    r"passive immunity|maternal (?:immuni[sz]|vaccin)\w*)\b", re.I)
_PREVENT_WORDS = re.compile(
    r"\b(prevent\w*|pre-?emptive|"
    # "before imiquimod challenge", "prior to RSV season exposure"
    r"(?:before|prior to)(?:\s+\w+){0,3}\s+(?:onset|exposure|challenge|infection|symptoms|season))\b", re.I)


def classify_setting(text: str, stated: str) -> str | None:
    """`prevention` when the sentence says so, else None.

    An explicit prophylactic modality — prophylaxis, a vaccine, immunisation —
    overrules any label, because that is what those words mean. The weaker
    "prevent" family only fills in a setting the model left `unclear`, since a
    paper about established disease may mention prevention in passing.
    """
    if _PROPHYLAXIS.search(text):
        return "prevention"
    if stated == "unclear" and _PREVENT_WORDS.search(text):
        return "prevention"
    return None

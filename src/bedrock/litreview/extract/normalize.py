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


# --- evidence type from the quote's own words ------------------------------------
#
# `infer_evidence_type` trusts the species, and the species can come from the
# wrong sentence. On the qwen3:8b asthma run a quote about bronchial epithelial
# cells in a dish ("HBECs were exposed to CRE ... EN4 effectively suppressed
# CRE-induced c-Myc expression") was labelled in_vivo, species "mice": the mice
# were in the context sentence, not in the experiment the quote reports. And on
# the obesity run a Western blot of "mast cells from individuals with obesity"
# was labelled human_genetic, though nothing about it is genetic.
#
# So the quote's own words about the system decide, ahead of the species:
# cells in a dish -> in_vitro, animals -> in_vivo, people or their tissue ->
# human. When the quote names none of these, or names two ("in mice and in
# patients"), it decides nothing and the species-based inference above stands.
# `stages/s5_extract.py` `settle_evidence_type` applies it and writes every
# change on the unit.
#
# `human_clinical` is the bucket for any human study that is not genetic,
# observational tissue work included. The four evidence types are in_vitro,
# in_vivo, human_clinical and human_genetic (`models.EvidenceType`); a Western
# blot of patients' cells is human evidence and it is not genetic, so it is
# human_clinical - the same bucket a cross-sectional cohort already lands in.
# `human_genetic` needs genetic-study words (a variant, a polymorphism, GWAS,
# Mendelian randomisation, carriers of an allele ...) in the quote or its
# context sentence, *and* people.
#
# Measured on the 593 stored qwen3:8b findings (obesity, asthma, migraine):
# 43 change type - 26 human_genetic -> human_clinical (tissue, expression and
# single-cell data from patients; background sentences), 9 in_vivo -> in_vitro
# (HBECs, BMMCs, hiPSC cardiomyocytes, co-cultures, each with mice only in the
# context sentence), 3 in_vitro -> human_clinical (pharmacokinetics in healthy
# individuals, "in pre-school children"), 2 in_vitro -> in_vivo, 1 each
# in_vitro -> human_genetic, in_vivo -> human_clinical and human_clinical ->
# in_vitro ("our ... cell culture studies"). Read one by one; the rules below
# that look fussy (singular "mouse", the in_vitro guard, no pre-verb
# "treated ... cells", no promotion of a trial to human_genetic) each remove
# a wrong change found that way.

# Work in a dish. Deliberately not a bare "cells": "Th17 cells rose in
# patients" and "mast cells from individuals with obesity" are observations of
# people's cells, not cell culture. A named line, an organoid, culture words,
# or cells described as being exposed, treated, stimulated or transfected.
# Not "G36-treated MAIT cells" on its own: those were transferred into mice,
# and the quote that says so need not name the mice.
_CELL_WORK = re.compile(
    r"\b(?:in\s+vitro|cell\s+lines?|cell\s+cultures?|cultured|in\s+culture|co-?cultures?"
    r"|culture\s+(?:system|medium|media|model|studies|assays?)|organoids?|spheroids?"
    r"|n?hbecs?|hbe|beas-?2b|16hbe\w*|calu-?3|a549|hek\s*-?\s*293\w*|hela|jurkat|thp-?1|u937|caco-?2|hacat"
    r"|raw\s*264(?:\.7)?|3t3-?l1|hepg2|huh-?7|c2c12|mcf-?7|sh-?sy5y|pc-?12|cho\s+cells?"
    r"|h?ipsc\w*|bmdms?|bmdcs?|bmmcs?"
    r"|(?:cells?|lines?)\s+(?:were|was|are|is)\s+(?:\w+\s+){0,2}?(?:exposed|treated|stimulated|transfected"
    r"|pretreated|incubated|infected|challenged|cultured|seeded|differentiated))\b",
    re.I,
)
# Animals, as the animals themselves. The singular "mouse", "rat" and
# "murine" are nearly always adjectives - "mouse Chrm1", "mouse WT protein",
# "human and murine dendritic cells" - and name an animal only before a model
# or an organ ("mouse model", "mouse lungs"). "In vivo" and a named disease
# model count too: "in both the HDM asthma model and the in vitro cell models"
# is both, and must decide nothing.
_ANIMALS = re.compile(
    r"\b(?:mice|rats|c57bl\w*|balb/?c|db/db|ob/ob|zebrafish|primates|macaques|monkeys|cynomolgus|rhesus"
    r"|pigs|piglets|minipigs|dogs|rabbits|hamsters|guinea\s+pigs|sheep|animals|rodents|pups|dams|in\s+vivo"
    r"|(?:rodent|animal|primate|hdm|ova|asthma|obesity)\s+models?"
    r"|(?:mouse|murine|rat)\s+(?:[\w-]+\s+)?(?:models?|lungs?|livers?|brains?|hearts?|kidneys?|tissues?|adipose"
    r"|skin|airways?|hypothalam\w*|intestines?|colons?|sera|serum|plasma|blood))\b",
    re.I,
)
# People, and tissue or samples taken from them. The adjective "human" alone
# does not count ("human bronchial epithelial cells" are cells); "human" with
# a tissue or sample noun does. "Subjects" and "cohort" are left out: mouse
# studies use both (`knowledge/meaning_words.yaml`).
_PEOPLE = re.compile(
    r"\b(?:patients?|participants?|volunteers?|individuals|people|persons?|children|adolescents|adults"
    r"|infants|neonates|newborns|women|men|donors?|carriers|twins|probands?|cases\s+and\s+controls"
    r"|healthy\s+controls?|clinical\s+trials?|phase\s+(?:[1-4]|i{1,3}v?)[ab]?\s+(?:clinical\s+)?(?:trial|study)"
    r"|(?:human|patient)\s+(?:[\w-]+\s+){0,2}?(?:tissues?|samples?|biops(?:y|ies)|specimens?|sera|serum|plasma"
    r"|blood|sputum|skin|lungs?|airways?|adipose|fat|brains?|livers?|muscle|cohorts?|subjects?|populations?"
    r"|studies|data))\b",
    re.I,
)
# A word joined to the next by a hyphen names no people where it stands:
# "adult-onset deletion" was done in mice, "donor-derived cells" are cells.
_HYPHENATED = re.compile(r"\b[^\W\d_]+-(?=[^\W\d_])")
# Any cells at all, cultured or not ("MAIT cells", "PBMCs", "macrophages").
# "Single-cell analysis" is a method, not cells.
_ANY_CELLS = re.compile(
    r"(?<!single-)\b(?:cells?|pbmcs?|macrophages?|monocytes?|neutrophils?|eosinophils?|lymphocytes?"
    r"|fibroblasts?|keratinocytes?|adipocytes?|myocytes?|cardiomyocytes?|neurons?|hepatocytes?|epithelium)\b",
    re.I,
)
_GENETIC = re.compile(
    r"\b(?:variants?|polymorphisms?|snps?|gwas|genome-?wide\s+association|mendelian\s+randomi[sz]ation"
    r"|heritab\w*|loss-of-function|gain-of-function|lof|alleles?|allelic|genotypes?|genotyp\w*|haplotypes?"
    r"|rs\d{3,}|carriers?|homozyg\w*|heterozyg\w*|exome\w*|eqtls?|polygenic|germline|susceptibility\s+loc(?:us|i)"
    r"|risk\s+(?:loc(?:us|i)|genes?|alleles?)|mutations?\s+carriers?|genetic(?:ally)?\s+(?:associat\w*|variation"
    r"|predispos\w*|risk|determined|proxied|predicted))\b",
    re.I,
)


def genetic_words(text: str) -> str | None:
    """The first genetic-study word in `text` ("variant", "GWAS", "carriers"),
    or None."""
    m = _GENETIC.search(text or "")
    return m.group(0) if m else None


def system_in_quote(quote: str, stated: str, context: str = "") -> tuple[str, str] | None:
    """(evidence type, the word that decided it) from the quote's own words
    about what was studied, or None when the quote does not say, or says two
    things.

    - Cells in a dish win over people: "HBECs from patients were exposed to
      CRE" is an experiment on cells, whoever donated them. Cells in a dish
      and an animal together ("in both the HDM asthma model and the in vitro
      cell models") decide nothing.
    - A model's `in_vitro` label is not overruled by animals or people when
      the quote names cells of any kind: a dish of people's or animals'
      cells is named for its donors ("ligand-activated MAIT cells from
      healthy individuals", "PBMCs from asthmatic children").
    - Animals alone -> in_vivo; people alone -> human; both decide nothing.
    - People with genetic-study words in the quote or context sentence are
      human_genetic, unless the model said human_clinical: a Setmelanotide
      trial "in MC4R mutation carriers" is a clinical trial whose population
      is defined by a gene, not a genetic study. People without genetic words
      are human_clinical, whatever the model said.

    The context sentence is read only for genetic words, never for the
    system: reading the system from it is the mistake this function exists
    to undo.
    """
    cells = _CELL_WORK.search(quote)
    animal = _ANIMALS.search(quote)
    people = _PEOPLE.search(_HYPHENATED.sub(" ", quote))
    if cells:
        return None if animal else ("in_vitro", cells.group(0))
    if stated == "in_vitro" and _ANY_CELLS.search(quote):
        return None
    if animal and not people:
        return "in_vivo", animal.group(0)
    if people and not animal:
        genetic = genetic_words(quote) or genetic_words(context)
        if not genetic:
            return "human_clinical", people.group(0)
        return (stated, genetic) if stated.startswith("human") else ("human_genetic", genetic)
    return None


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


_DOSING_WORDS = re.compile(r"(\btreat|administ|inject|infus|\bdos(?:e|ed|es|ing)\b)", re.I)


def names_intervention(change: str) -> bool:
    """Do the words the model copied for "what was done to the target" name
    an intervention at all, even one whose direction is not clear? A block or
    activate word, a genetic variant ("LOF variants", "variant homozygotes"),
    or dosing ("treatment with").

    The model fills the field with whatever is to hand when nothing was done:
    "observed", "showed increased", "continued progressively", "children
    under 10 years of age" (all from the qwen3:8b obesity run). Those name
    nothing done to anything, and neither does a diet or an allergen that
    makes the disease model - which is why feeding and exposure are not here.
    """
    return bool(_BLOCK_WORDS.search(change) or _ACTIVATE_WORDS.search(change) or genetic_words(change)
                or _DOSING_WORDS.search(change))


def target_changed_in(quote: str, target: str) -> bool:
    """Does the quote itself put a block or activate word right beside the
    target's name - "PGGT1B deficiency", "Sirt6-deficient cells", "anti-IL-17A",
    "inhibition of TSLP"? Within two words either side, hyphen-joined words
    included. A backstop for when the words the model copied for the change
    are not in the quote: the quote may still say the target was changed, and
    a knockout that worsened disease must not be filed as an association."""
    target = (target or "").strip()
    if not target:
        return False
    near = re.compile(rf"(?:\S+\s+){{0,2}}\S*{re.escape(target)}\S*(?:\s+\S+){{0,2}}", re.I)
    for m in near.finditer(quote):
        if _BLOCK_WORDS.search(m.group(0)) or _ACTIVATE_WORDS.search(m.group(0)):
            return True
    return False


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

"""Issue #3: a sentence built out of the qualifier or the naming sentence.

Showing the reader the paper's qualifying next sentence (issue #1) and the
sentence that names the target (issue #6) also made both of them citable
text, and the writing model started mining them: it writes a sentence whose
content is one of those sentences while citing a unit whose *quote* says
something else. Nothing in the guard could see it, because every word really
is the paper's own text.

Guard check 10 is the answer: the claim must rest on the quote. The other
displayed sentences may add a name, a limit, a subject or a sample size -
they may never be the whole of the sentence.

Everything below is verbatim from the three benchmark reports and the papers
they read. The keeps matter as much as the drops: the qualifier and the
naming sentence must still be able to do the job they were added for, which
`test_qualifier.py` and `test_naming_sentence.py` also cover
(`test_tubastatin_a_acting_on_hdac6_is_kept` is the naming half).
"""

from bedrock.litreview.knowledge import load_knowledge
from bedrock.litreview.models import EvidenceUnit
from bedrock.litreview.report.guard import CitationGuard, rests_on_quote
from bedrock.litreview.stages.s0_question import parse_question

KB = load_knowledge()
OBESITY = parse_question("If I want to treat obesity, which targets should I pursue?")
ASTHMA = parse_question("If I want to prevent asthma exacerbations, what should I target?")
MIGRAINE = parse_question("If I want to prevent migraine attacks, what areas should I target?")

DROPPED = "restates the context, qualifier or naming sentence, not the quote it cites"


def _unit(quote: str, *, context: str | None = None, qualifier: str | None = None, naming: str | None = None,
          **kw) -> EvidenceUnit:
    """One evidence unit, with whatever the reader is shown beside its quote.
    The offsets are not what this check reads, so they are only made valid."""
    base = dict(
        id="u1", paper_id="P1", question_id="q", target="MC4R", target_raw="MC4R", direction="unclear",
        modality="other", intervention=None, disease="obesity", disease_raw="obesity", evidence_type="in_vivo",
        species="mouse", species_raw="mice", outcome="association", effect_detail="", n=None, combo_with=None,
        target_is_primary=True, quote=quote, quote_status="exact", section="results",
        extracted_by_model="fake", prompt_version="1", provenance={"target": "quote"},
        context_quote=context, context_start=0 if context else None,
        context_end=len(context) if context else None,
        qualifier_quote=qualifier, qualifier_start=0 if qualifier else None,
        qualifier_end=len(qualifier) if qualifier else None,
        naming_quote=naming, naming_start=0 if naming else None, naming_end=len(naming) if naming else None,
    )
    base.update(kw)
    return EvidenceUnit(**base)


def _check(spec, sentence: str, *units: EvidenceUnit):
    table = {f"E{i}": u for i, u in enumerate(units, 1)}
    guard = CitationGuard(spec, KB, table, paper_labels={"P1": "Author 2024 Journal"})
    return guard.check(sentence, list(table))


# --- the four failures the after-audit found, verbatim --------------------------

# obesity S6 (PMID35102358). Every number in the sentence - 80% and 60% - is
# the naming sentence's, and "in animal studies ... in HEK293 cells"
# contradicts itself. The quote is about mice gaining weight.
GPR75_QUOTE = ("Female control mice gained weight by 19.0 ± 1.6g, while sh517 females gained 11.8 ± 1.8g and "
               "sh1109 females gained 16.5 ± 1.5g, leading to 18% and 5% lower final body weight than controls.")
GPR75_NAMING = ("Two shRNA sequences targeting Gpr75, sh517 and sh1109, were effective in reducing Gpr75 mRNA and "
                "protein levels in Gpr75-overexpressing HEK293 cells by approximately 80% and 60%, respectively ().")
GPR75_SENTENCE = ("In animal studies, shRNA sequences targeting Gpr75, sh517 and sh1109, reduced Gpr75 mRNA and "
                  "protein levels in HEK293 cells by approximately 80% and 60%, respectively.")

# obesity S18 (PMID41582137): the sentence is the qualifier, near verbatim;
# the quote is a different result about glucose homeostasis.
TMEM135_QUOTE = ("Assessment of glucose homeostasis in these animals revealed that the TMEM135 knockout results in "
                 "glucose intolerance, hyperinsulinemia and insulin resistance (Fig. ).")
TMEM135_CONTEXT = ("Transgenic mice that overexpress TMEM135 ( Tmem135 TG ) under the control of CAG promoter have "
                   "been previously described .")
TMEM135_QUALIFIER = ("In contrast to female mice, high fat feeding in male mice did not elicit a significant body "
                     "weight difference between the genotypes, presumably because the already rapid diet-induced "
                     "weight gain in male C57BL/6J mice could not be further increased by the TMEM135 inactivation "
                     "(Supplementary Fig. ).")
TMEM135_SENTENCE = ("High fat feeding in male mice did not elicit a significant body weight difference between "
                    "TMEM135 knockout and control genotypes, presumably due to the already rapid diet-induced "
                    "weight gain in male C57BL/6J mice.")

# obesity S29 (PMID41588213): the same shape, and the quote is a textbook
# line about what AMPK does.
AMPK_QUOTE = ("In response to energy deprivation, AMPK promotes FAO to replenish ATP stores by attenuating "
              "malonyl-CoA production through inhibitory phosphorylation of ACC ().")
AMPK_CONTEXT = "Mice were excluded from experiments if they showed any sign of morbidity."
AMPK_QUALIFIER = ("However, fasting plasma levels of FFAs and β-hydroxybutyrate, a ketone body synthesized from "
                  "FAO-derived acetyl-CoA and a surrogate biomarker for hepatic FAO, were higher and lower, "
                  "respectively, in ob/ob HKO mice than in ob/ob F/F mice (), suggestive of impaired hepatic FAO "
                  "that might give rise to suppression of gluconeogenesis in ob/ob HKO mice.")
AMPK_SENTENCE = ("However, fasting plasma levels of FFAs and β-hydroxybutyrate were higher and lower, "
                 "respectively, in ob/ob HKO mice than in ob/ob F/F mice, suggesting impaired hepatic fatty acid "
                 "oxidation.")

# migraine S5: the sentence is the qualifier word for word - and it is an open
# question, not a finding at all.
CCL2_QUOTE = "Some migraine patients had elevated CCL2 levels in CSF or cranial periosteum."
CCL2_QUALIFIER = "However, whether the CCL2-CCR2 signalling pathway contributes to chronic migraine is not clear."
CCL2_SENTENCE = CCL2_QUALIFIER


def test_the_naming_sentences_numbers_are_not_the_quotes_result():
    verdict = _check(OBESITY, GPR75_SENTENCE, _unit(GPR75_QUOTE, naming=GPR75_NAMING, target="GPR75",
                                                    target_raw="Gpr75"))
    assert not verdict.kept
    assert verdict.reason == DROPPED


def test_the_qualifier_restated_as_the_finding_is_dropped():
    unit = _unit(TMEM135_QUOTE, context=TMEM135_CONTEXT, qualifier=TMEM135_QUALIFIER,
                 target="TMEM135", target_raw="TMEM135")
    verdict = _check(OBESITY, TMEM135_SENTENCE, unit)
    assert not verdict.kept
    assert verdict.reason == DROPPED


def test_the_qualifier_restated_beside_an_unrelated_quote_is_dropped():
    unit = _unit(AMPK_QUOTE, context=AMPK_CONTEXT, qualifier=AMPK_QUALIFIER, target="AMPK", target_raw="AMPK")
    verdict = _check(OBESITY, AMPK_SENTENCE, unit)
    assert not verdict.kept
    assert verdict.reason == DROPPED


def test_an_open_question_lifted_from_the_qualifier_is_dropped():
    unit = _unit(CCL2_QUOTE, qualifier=CCL2_QUALIFIER, target="CCL2", target_raw="CCL2",
                 disease="migraine", disease_raw="migraine", evidence_type="human_clinical",
                 species="human", species_raw="patients")
    verdict = _check(MIGRAINE, CCL2_SENTENCE, unit)
    assert not verdict.kept
    assert verdict.reason == DROPPED


# --- what must still be kept ----------------------------------------------------

# obesity, PMID29031731: the setmelanotide result with the paper's own limit.
# This is the whole point of the qualifier fix (`test_qualifier.py`).
S6_QUOTE = ("In the MC4R mutation carriers, placebo subtracted group mean differences in weight loss (dose "
            "0.01 mg/kg/24 h) of ∼0.6 kg/week were observed ().")
S6_NEXT = ("Although this result did not reach statistical significance (p = 0.09) when compared to placebo, "
           "weight loss from baseline was significantly reduced (p < 0.0001).")

# asthma, PMID41699846: the claim (3.0 mg/kg reaching the target Cmax) is the
# quote's; only the population comes from the context and the conclusion.
CMAX_QUOTE = ("The target Cmax was reached in 67%, 80%, and 90% of participants at doses of 2.0, 2.5, and "
              "3.0 mg/kg, respectively.")
CMAX_CONTEXT = ("Background and Purpose: Children with acute asthma exacerbations are frequently hospitalized due "
                "to insufficient response to inhaled β-agonist and systemic corticosteroid potentially due to "
                "leukotriene mediated airway inflammation.")
CMAX_NAMING = ("Conclusion and Implications: Among children with moderate or severe acute asthma exacerbations not "
               "responsive to initial inhaled albuterol, oral montelukast at a dose of 3.0 mg/kg reliably achieves "
               "a potentially therapeutic Cmax.")
CMAX_SENTENCE = ("Montelukast at a dose of 3.0 mg/kg achieved a potentially therapeutic Cmax in children with "
                 "moderate or severe acute asthma exacerbations not responsive to initial inhaled albuterol.")


def test_the_quotes_result_with_the_qualifiers_limit_is_kept():
    """A sentence may carry the limit the paper states - that is what the
    qualifier is shown for. It states the quote's ∼0.6 kg/week result too."""
    unit = _unit(S6_QUOTE, qualifier=S6_NEXT, evidence_type="human_clinical", species="human",
                 species_raw="patients")
    sentence = ("In the MC4R mutation carriers, placebo subtracted group mean differences in weight loss of "
                "∼0.6 kg/week did not reach statistical significance (p = 0.09).")
    # The rule itself: this quote reports no result in check 6's reading, so
    # the full guard drops it for that instead (see `test_qualifier.py`).
    assert rests_on_quote(sentence, sentence, unit)


def test_a_sentence_that_is_only_the_qualifiers_limit_is_kept():
    """migraine, PMID40542538: the qualifier `extract/verify.py` recognises
    as limiting this quote's result is about the quote, so reporting the
    limit is reporting the quote's finding (see `test_qualifier.py`)."""
    quote = ("The response rates for the first secondary efficacy endpoint (i.e., participants with ≥50% reduction "
             "from the OP in moderate to severe MMDs in the last 4 weeks of the DBT phase) numerically favored "
             "rimegepant versus placebo and were 41.7% (95% CI = 35.5%–47.9%) in the rimegepant group and 34.4% "
             "(95% CI = 28.5%–40.4%) in the placebo group ( p = 0.099; Table ).")
    nxt = ("Because the findings for the first secondary efficacy endpoint were not statistically significant, "
           "formal hypothesis testing did not proceed.")
    context = ("A total of 496 participants were randomized to rimegepant ( n = 247) or placebo ( n = 249), and "
               "all were treated with study drug (Figure ).")
    unit = _unit(quote, context=context, qualifier=nxt, target="MMDs", target_raw="MMDs",
                 intervention="rimegepant", disease="migraine", disease_raw="migraine",
                 evidence_type="human_clinical", species="human", species_raw="patients")
    sentence = ("Response rates for the first secondary efficacy endpoint were not statistically significant for "
                "rimegepant versus placebo, so formal hypothesis testing did not proceed.")
    assert rests_on_quote(sentence, sentence, unit)


def test_a_number_the_quote_states_keeps_the_sentence():
    """Even when the naming sentence repeats that number and supplies the
    population: the quote is where the result is, and the other sentences
    only say who it was in."""
    unit = _unit(CMAX_QUOTE, context=CMAX_CONTEXT, naming=CMAX_NAMING, target="leukotriene receptor",
                 target_raw="leukotriene receptor", intervention="montelukast", disease="asthma",
                 disease_raw="asthma", evidence_type="human_clinical", species="human", species_raw="children")
    assert rests_on_quote(CMAX_SENTENCE, CMAX_SENTENCE, unit)
    assert _check(ASTHMA, CMAX_SENTENCE, unit).kept


def test_a_unit_with_nothing_shown_beside_the_quote_is_left_alone():
    """obesity, PMID41576935: the sentence shares little wording with its
    quote, but there is no other sentence it could have come from, so this
    check has nothing to say about it."""
    quote = ("Human Genotype-Tissue Expression (GTEx) analysis of VAT and SAT also revealed a positive correlation "
             "between the mast cell marker gene CPA3 and fibrosis-related genes like COL6A1 , COL1A1 , TGFB1 , "
             "TIMP2 , and COL3A1 (Supplementary Fig. ).")
    unit = _unit(quote, target="CPA3", target_raw="CPA3")
    sentence = ("This correlation was also observed in subcutaneous adipose tissue (SAT), suggesting a potential "
                "role for CPA3 in adipose tissue fibrosis.")
    assert rests_on_quote(sentence, sentence, unit)


def test_another_cited_quote_the_sentence_does_rest_on_keeps_it():
    """The sentence may be citing the restated unit for a name and taking its
    finding from a second quote, so every cited unit has to fail."""
    lifted = _unit(CCL2_QUOTE, qualifier=CCL2_QUALIFIER, target="CCL2", target_raw="CCL2",
                   disease="migraine", disease_raw="migraine")
    other = _unit(CCL2_QUALIFIER, id="u2", target="CCL2", target_raw="CCL2",
                  disease="migraine", disease_raw="migraine")
    assert _check(MIGRAINE, CCL2_SENTENCE, lifted, other).kept

"""The benchmark, end to end, with a scripted fake model.

This checks the benchmark's own plumbing — claims, judge calls, nuggets,
reference targets, baselines, distractors, perturbations, determinism,
scorecard — not any model's quality. Quality is measured by running it for
real (evals/litreview/README.md).
"""

import json
import re

import pytest

from bedrock.litreview.bench.claims import split_sentences, text_claims
from bedrock.litreview.bench.distractors import make_distractors, pick_near_miss
from bedrock.litreview.bench.models import BenchQuestion, CorpusSpec, ReferenceSpec
from bedrock.litreview.bench.perturb import perturb_flip, perturb_number, perturbations
from bedrock.litreview.bench.run import evaluate
from bedrock.litreview.config import load_settings
from bedrock.litreview.llm import FakeLLMClient
from bedrock.litreview.models import Paper, PaperSignals
from bedrock.litreview.sources.corpus import write_papers
from bedrock.litreview.stages.s0_question import parse_question
from tests.litreview.fixtures import QUESTION, corpus, responder


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 3}


def bench_responder(prompt: str, schema: dict | None) -> str:
    title = (schema or {}).get("title")
    if schema is None:
        if "Papers:" in prompt:
            return ("Guselkumab achieved PASI-90 in 73% of patients [P1]. "
                    "Dupilumab helps atopic dermatitis patients [P2]. "
                    "Etanercept cures psoriasis in 95% of cases.")
        return "Secukinumab and ixekizumab block IL-17A. Most patients with psoriasis respond to TNF blockers."
    if title == "SupportAnswers":
        # The best-overlapping evidence sentence, as a small judge would find it.
        claim = prompt.split('Claim:\n"', 1)[1].split('"\n', 1)[0]
        evidence = prompt.split("Evidence (quoted from research papers):", 1)[1].split("## Answer", 1)[0]
        sentences = [x.strip(' "[]0123456789') for x in re.split(r'(?<=[.])\s+|\n', evidence) if len(x.strip()) > 25]
        best = max(sentences, key=lambda x: len(_tokens(claim) & _tokens(x)), default="")
        overlap = len(_tokens(claim) & _tokens(best)) / max(1, len(_tokens(claim)))
        return json.dumps({"reasoning": "fake", "evidence_sentence": best if overlap >= 0.5 else "", "missing": []})
    if title == "NuggetList":
        text = prompt.split("Reference (", 1)[1].split("):\n", 1)[1]
        first = re.split(r"(?<=[.])\s+", text.strip())[0]
        return json.dumps({"nuggets": [{"fact": "Blocking IL-23 improves psoriasis in patients.",
                                        "quote": first, "importance": "vital"}]})
    if title == "CoverageAnswers":
        answer = prompt.split("Answer:", 1)[1].split("## Answer each field", 1)[0]
        hit = next((x.strip() for x in re.split(r"(?<=[.])\s+|\n", answer) if "IL-23" in x), "")
        return json.dumps({"reasoning": "fake", "answer_sentence": hit})
    return responder(prompt, schema)


@pytest.fixture()
def bench_setup(tmp_path):
    settings = load_settings(model_set="laptop", env={}, check_endpoint=False)
    settings.cache_dir = tmp_path / "cache"
    settings.out_dir = tmp_path / "out"
    settings.data_dir = tmp_path / "data"
    corpus_dir = tmp_path / "corpus"
    write_papers(corpus(), corpus_dir / "papers.jsonl")
    write_papers([Paper(
        id="R1", title="IL-23 in psoriasis: a review",
        abstract="IL-23 inhibitors such as guselkumab are highly effective in plaque psoriasis. " * 3,
        signals=PaperSignals(publication_types=["Review"]),
    )], corpus_dir / "references.jsonl")
    (corpus_dir / "reference_targets.json").write_text(json.dumps([
        {"symbol": "IL23A", "max_phase": 4, "drugs": ["GUSELKUMAB"]},
        {"symbol": "TNF", "max_phase": 4, "drugs": ["ETANERCEPT"]},
    ]))
    bq = BenchQuestion(id="psoriasis-test", question=QUESTION, corpus=CorpusSpec(queries=["x"]),
                       references=ReferenceSpec(reviews_max=1))
    return settings, corpus_dir, bq


def test_the_benchmark_scores_all_three_systems(bench_setup, tmp_path):
    settings, corpus_dir, bq = bench_setup
    client = FakeLLMClient(responder=bench_responder)
    card, out_dir = evaluate(bq, settings=settings, client=client, corpus_dir=corpus_dir, results_root=tmp_path / "results")
    systems = {s.system: s for s in card.systems}
    assert set(systems) == {"pipeline", "closed_book", "naive_rag"}
    assert systems["pipeline"].claims > 0
    assert card.pipeline["quote_validity"] == 1.0
    assert card.pipeline["deterministic"] == "yes"
    # The fake grades relevance by keyword ("IL-23" in the title), which is
    # exactly the grader distractors exist to expose: they leak, and the gate fails.
    assert card.pipeline["distractor_leak_rate"] > 0
    assert not card.gates["no distractor leaked"]
    assert systems["naive_rag"].near_miss_claims >= 1, "the naive baseline cited an atopic dermatitis paper"
    assert systems["closed_book"].cited_support is None, "a closed-book answer has nothing to cite"
    assert (out_dir / "scorecard.md").exists() and (out_dir / "pipeline").exists()
    assert (tmp_path / "results" / "history.jsonl").read_text().strip()


def test_sentences_split_without_breaking_numbers_or_abbreviations():
    text = "Guselkumab (e.g. at 100 mg) gave PASI-90 in 73.5% of patients vs. 3% on placebo. Next sentence here."
    assert split_sentences(text) == [
        "Guselkumab (e.g. at 100 mg) gave PASI-90 in 73.5% of patients vs. 3% on placebo.",
        "Next sentence here.",
    ]


def test_baseline_citations_map_to_papers():
    claims = text_claims("IL-23 blockade helps psoriasis patients a lot [P1, P2]. Short.", "naive_rag", {"P1": "a", "P2": "b"})
    assert len(claims) == 1 and claims[0].citations == ["a", "b"]


def test_perturbations_change_what_they_claim_to():
    assert perturb_number("PASI-90 in 73% of patients") == "PASI-90 in 153% of patients"
    assert perturb_flip("IL-23 blockade reduced ear thickness") == "IL-23 blockade increased ear thickness"
    spec = parse_question(QUESTION)
    kinds = dict(perturbations("Blocking IL-23 reduced psoriasis severity by 60%.", spec))
    assert kinds["entity"] == "Blocking IL-23 reduced atopic dermatitis severity by 60%."


def test_distractors_swap_the_entity_and_keep_a_passing_mention():
    from bedrock.litreview.facets import facets_for
    from bedrock.litreview.stages.s2_gate import gate_papers
    from bedrock.litreview.stages.s3_rank import rank_papers
    from tests.litreview.fixtures import fake_client

    spec = parse_question(QUESTION)
    papers = {p.id: p for p in corpus()}
    gates = gate_papers(list(papers.values()), spec)
    ranked = rank_papers(list(papers.values()), gates, spec, facets_for(spec), fake_client(), model="m")
    assert pick_near_miss(spec) == "atopic dermatitis"
    made = make_distractors(papers, ranked, spec, k=2)
    assert made and all(p.synthetic for p in made)
    assert "psoriasis" not in made[0].title.lower()
    assert made[0].abstract.endswith("These findings may not extend to psoriasis.")


def test_a_verbatim_claim_is_settled_in_code_and_a_flipped_one_is_not():
    from bedrock.litreview.bench.judge import Judge
    from bedrock.litreview.llm import FakeLLMClient

    quote = "Anti-IL-23 antibody given before imiquimod challenge reduced ear thickness by 60% in C57BL/6 mice."
    judge = Judge(FakeLLMClient(replies=[json.dumps({"reasoning": "x", "evidence_sentence": quote, "missing": []})]), "m")
    assert judge.support(quote, [f"Methods aside. {quote} Further text."]).verdict == "supported"
    assert judge.calls == 0 and judge.verbatim == 1
    flipped = quote.replace("reduced", "increased")
    assert judge.support(flipped, [quote]).verdict == "contradicted", "a changed claim still goes to the judge"
    assert judge.calls == 1


def test_the_judge_is_checked_by_code():
    """The judge finds a sentence; code decides. A detail it calls missing
    that is in the evidence is struck off; a number or a name in none of the
    evidence is missing whatever the judge says; a sentence it copied that is
    not in the evidence counts for nothing."""
    from bedrock.litreview.bench.judge import verdict_from
    from bedrock.litreview.bench.models import SupportAnswers

    sentence = "Primidone inhibited RIPK1 and prevented imiquimod-induced inflammation in mice."
    evidence = [sentence]
    wrong = SupportAnswers(reasoning="r", evidence_sentence=sentence, missing=["RIPK1", "mice"])
    verdict, struck, _ = verdict_from(wrong, "Primidone inhibits RIPK1 and prevents inflammation in mice.", evidence)
    assert verdict.verdict == "supported" and struck == 2
    found = SupportAnswers(reasoning="r", evidence_sentence=sentence, missing=[])
    assert verdict_from(found, "Primidone prevented inflammation in 90% of mice.", evidence)[0].verdict \
        == "partially_supported"
    assert verdict_from(found, "Bapineuzumab inhibits VEGF in psoriasis.", evidence)[0].verdict == "unsupported"
    invented = SupportAnswers(reasoning="r", evidence_sentence="Primidone cures psoriasis in everyone.", missing=[])
    assert verdict_from(invented, "Primidone cures psoriasis.", evidence)[0].verdict == "unsupported"
    echoed = SupportAnswers(reasoning="r", evidence_sentence="Primidone inhibited RIPK1 in mice.", missing=[])
    verdict, _, by_code = verdict_from(echoed, "Primidone inhibited RIPK1 and prevented inflammation in mice.", evidence)
    assert verdict.verdict == "supported" and by_code, "an echoed claim falls back to code's closest sentence"


def test_opposite_effects_are_caught_and_paraphrases_are_not():
    from bedrock.litreview.bench.judge import opposite

    s = "Anti-IL-23 treatment did not significantly reduce ear thickening compared with vehicle."
    assert opposite("Anti-IL-23 treatment did significantly reduce ear thickening.", s)
    assert opposite("Blocking IL-23 increased ear thickening.", "Blocking IL-23 reduced ear thickening.")
    assert not opposite("Blocking IL-23 decreased ear thickening.", "Blocking IL-23 reduced ear thickening.")
    assert not opposite("X reduced A.", "X reduced A but did not affect B.")


def test_two_measurements_are_not_a_contradiction():
    """From the MedGemma run: 'enhanced proliferation' beside 'reduced
    fluorescence ... increased cell division' was called contradicted."""
    from bedrock.litreview.bench.judge import opposite

    sentence = ("DGAT2-treated CD8+ T cells exhibited a markedly higher proportion of cells with reduced "
                "fluorescence intensity, indicative of increased cell division.")
    assert not opposite("DGAT2 significantly enhanced CD8+ T cell proliferation in blood-derived T cells.", sentence)
    evidence = "EPA did not significantly affect CXCR6 expression. EPA increased IFN-γ expression in CD8+ T cells."
    assert not opposite("EPA did not affect CXCR6 expression.", "EPA increased IFN-γ expression in CD8+ T cells.",
                        evidence), "the null result is in the evidence; the judge copied the wrong sentence"


def test_a_claim_mostly_absent_from_its_evidence_is_unsupported():
    from bedrock.litreview.bench.judge import verdict_from
    from bedrock.litreview.bench.models import SupportAnswers

    sentence = "SLC15A3 enhances the recruitment of TASL to augment IRF5 signaling in macrophages."
    found = SupportAnswers(reasoning="r", evidence_sentence=sentence, missing=[])
    verdict = verdict_from(found, "The study showed that activating SLC15A3 improved patient outcomes.", [sentence])[0]
    assert verdict.verdict == "unsupported"


def test_dropping_a_negation_is_a_flip():
    """From the RSV run: 'no significant difference in AE types' became 'a
    significant difference in AE types' and the judge passed it, because
    'difference' is in no effect-word list."""
    from bedrock.litreview.bench.judge import opposite

    evidence = "There was no significant difference in AE types and frequencies between groups."
    assert opposite("a significant difference in AE types and frequencies.", evidence)
    assert opposite("39.7% did receive due to nonavailability.",
                    "Among those that did not receive the maternal vaccine, the most common reasons were "
                    "nonavailability (39.7%) or no provider immunization offer (22.2%).")
    assert opposite("Anti-IL-23 reduced ear thickening.",
                    "Anti-IL-23 did not reduce ear thickening compared with vehicle.")
    assert not opposite("There was no significant difference in AE types.", evidence), "both say no difference"
    assert not opposite("PASI-75 was achieved by 78% of patients.",
                        "Patients did not receive corticosteroids; PASI-75 was achieved by 78%."), \
        "the negation governs something the claim is not about"
    assert not opposite("X reduced lesion area.", "X did not affect body weight, but reduced lesion area.")


def test_the_judge_can_be_answered_from_a_file(tmp_path):
    """The judge is pluggable so a stronger judge can measure a weak model,
    and an unanswered question is recorded rather than guessed."""
    import json as _json

    from bedrock.litreview.bench.filejudge import FileClient, task_key
    from bedrock.litreview.bench.judge import Judge
    from bedrock.litreview.llm.client import LLMError

    quote = "Blocking IL-23 reduced ear thickness by 60% in mice."
    claim = "IL-23 blockade reduced ear thickness in mice."
    answers = tmp_path / "answers.jsonl"
    client = FileClient(answers)

    # Nothing on file yet: the question is recorded, not answered.
    assert judge_asks(client, claim, [quote]) is None
    pending = client.flush()
    rows = [_json.loads(x) for x in pending.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["kind"] == "SupportAnswers"
    key = rows[0]["key"]

    # Answer it, and the same question is now judged — and still checked.
    answers.write_text(_json.dumps({
        "key": key,
        "answer": {"reasoning": "stated", "evidence_sentence": quote, "missing": []},
    }) + "\n")
    verdict = Judge(FileClient(answers), "m").support(claim, [quote])
    assert verdict.verdict == "supported"

    # A judge that invents its evidence sentence earns no trust: code looks
    # for the claim in the evidence itself, and an unsupported claim stays
    # unsupported however confidently it was answered.
    wild = "Guselkumab cleared plaques in 78% of patients."
    invented = tmp_path / "invented.jsonl"
    invented.write_text(_json.dumps({
        "key": task_key_for(wild, [quote]),
        "answer": {"reasoning": "x", "evidence_sentence": "Guselkumab cleared plaques in patients.", "missing": []},
    }) + "\n")
    assert Judge(FileClient(invented), "m").support(wild, [quote]).verdict == "unsupported"
    assert task_key("a") != task_key("b")
    with pytest.raises(LLMError):
        FileClient(tmp_path / "missing.jsonl").complete("unseen", model="m")


def judge_asks(client, claim, evidence):
    from bedrock.litreview.bench.judge import Judge

    return Judge(client, "m").support(claim, evidence)


def task_key_for(claim, evidence):
    """The key the judge will look up for this claim — the prompt it would send."""
    from bedrock.litreview.bench.filejudge import task_key
    from bedrock.litreview.bench.judge import _numbered
    from bedrock.litreview.llm import load_prompt

    return task_key(load_prompt("judge_support").render(claim=claim, evidence=_numbered(evidence)))


def test_a_judge_that_finds_nothing_is_not_overruled():
    """Code looks for the sentence a judge mis-copied, but an empty answer
    means the judge looked and found nothing. Overruling that let a claim
    with the entity swapped for a near-miss pass as supported."""
    from bedrock.litreview.bench.judge import verdict_from
    from bedrock.litreview.bench.models import SupportAnswers

    evidence = ["one unvaccinated neonate with RSV infection required invasive ventilation."]
    swapped = "one unvaccinated neonate with human metapneumovirus infection required invasive ventilation."
    refused = SupportAnswers(reasoning="the evidence says RSV", evidence_sentence="", missing=[])
    assert verdict_from(refused, swapped, evidence)[0].verdict == "unsupported"
    mangled = SupportAnswers(reasoning="r", evidence_sentence="one neonate needed ventilation", missing=[])
    verdict, _, by_code = verdict_from(mangled, evidence[0], evidence)
    assert verdict.verdict == "supported" and by_code, "a mis-copied sentence is still repaired"


def test_an_empty_pending_list_is_removed(tmp_path):
    """A stale list of questions that have since been answered is worse than
    no list: somebody answers them twice."""
    import json as _json

    from bedrock.litreview.bench.filejudge import FileClient

    from bedrock.litreview.llm.client import LLMError

    answers = tmp_path / "a.jsonl"
    client = FileClient(answers)
    with pytest.raises(LLMError):
        client.complete("unanswered", model="m")
    assert client.flush().exists()
    answered = FileClient(answers)
    assert answered.flush() is None and not client.pending_path.exists()
    answers.write_text(_json.dumps({"key": "k", "answer": {}}) + "\n")
    assert len(FileClient(answers).answers) == 1

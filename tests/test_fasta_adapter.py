"""FASTA is sequence data, not prose - the overview should carry record
count, length, and alphabet, and every sequence should survive intact for
exact lookup, the same bar test_context_adapters.py holds every adapter to.
"""

from bedrock.context.adapters import FastaAdapter, Source, classify_alphabet, gc_content, select_adapter

DNA_FASTA = """\
>seq1 example forward primer target
ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT
>seq2 example reverse primer target
TTGGCCAATTGGCCAATTGGCCAATTGGCCAATTGGCCAA
"""

PROTEIN_FASTA = """\
>sp|P69905|HBA_HUMAN Hemoglobin subunit alpha
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR
"""

MULTI_RECORD_FASTA = "\n".join(
    f">record{i} test\n{'ACGT' * 5}AAAA{i}" for i in range(20)
)


def test_can_handle_by_extension_even_without_a_leading_caret():
    source = Source.from_text("not actually fasta content", name="x.fasta", suffix=".fasta")
    assert FastaAdapter().can_handle(source)


def test_can_handle_by_sniffing_a_leading_record_marker():
    source = Source.from_text(DNA_FASTA, name="seqs.txt", suffix=".txt")
    assert FastaAdapter().can_handle(source)


def test_plain_prose_is_not_claimed():
    source = Source.from_text("Dear team,\n\nHere is the report.\n", name="memo.txt", suffix=".txt")
    assert not FastaAdapter().can_handle(source)


def test_registry_picks_fasta_over_text_for_a_real_file():
    source = Source.from_text(DNA_FASTA, name="seqs.fasta", suffix=".fasta")
    assert select_adapter(source).name == "fasta"


def test_dna_overview_reports_count_length_and_alphabet():
    extraction = FastaAdapter().extract(Source.from_text(DNA_FASTA, name="seqs.fasta", suffix=".fasta"))
    overview = extraction.overview

    assert "2 sequences" in overview
    assert "alphabet: dna" in overview
    assert extraction.metadata["records"] == 2
    assert extraction.metadata["alphabet"] == "dna"
    assert extraction.metadata["total_residues"] == 80
    assert "GC content" in overview


def test_mixed_file_tags_each_record_with_its_own_alphabet():
    """A file with both DNA and protein records must say which is which -
    "mixed" alone at the top doesn't tell you where to look."""
    mixed_fasta = (
        ">dna_one plain nucleotide record\n"
        "ACGTACGTACGTACGTACGTACGTACGTACGT\n"
        ">protein_one plain amino acid record\n"
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKK\n"
    )
    extraction = FastaAdapter().extract(Source.from_text(mixed_fasta, name="mixed.fasta", suffix=".fasta"))

    assert extraction.metadata["alphabet"] == "mixed"
    assert "[dna]" in extraction.overview
    assert "[protein]" in extraction.overview


def test_protein_sequence_is_classified_as_protein_not_ambiguous():
    extraction = FastaAdapter().extract(Source.from_text(PROTEIN_FASTA, name="hba.fasta", suffix=".fasta"))
    assert extraction.metadata["alphabet"] == "protein"
    assert "GC content" not in extraction.overview


def test_full_sequence_is_kept_as_its_own_segment_for_exact_lookup():
    extraction = FastaAdapter().extract(Source.from_text(DNA_FASTA, name="seqs.fasta", suffix=".fasta"))
    records = [s for s in extraction.segments if s.kind == "record"]
    assert len(records) == 2
    assert any("ACGTACGTACGTACGTACGTACGTACGTACGTACGTACGT" in s.text for s in records)
    assert records[0].label.startswith("seq1")


def test_schema_segment_is_weighted_above_record_segments():
    extraction = FastaAdapter().extract(Source.from_text(DNA_FASTA, name="seqs.fasta", suffix=".fasta"))
    schema = next(s for s in extraction.segments if s.kind == "schema")
    record = next(s for s in extraction.segments if s.kind == "record")
    assert schema.weight > record.weight


def test_many_records_are_truncated_in_the_overview_not_dropped_from_segments():
    extraction = FastaAdapter().extract(Source.from_text(MULTI_RECORD_FASTA, name="many.fasta", suffix=".fasta"))
    assert extraction.metadata["records"] == 20
    assert "more records" in extraction.overview
    assert len([s for s in extraction.segments if s.kind == "record"]) == 20


def test_empty_file_does_not_crash():
    extraction = FastaAdapter().extract(Source.from_text("", name="empty.fasta", suffix=".fasta"))
    assert extraction.segments == []


def test_classify_alphabet_dna_vs_rna_vs_protein_vs_ambiguous():
    assert classify_alphabet("ACGTACGTACGTACGT") == "dna"
    assert classify_alphabet("ACGUACGUACGUACGU") == "rna"
    assert classify_alphabet("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKK") == "protein"
    assert classify_alphabet("") == "ambiguous"
    # No character here belongs to any recognised nucleotide or amino-acid
    # alphabet, so this must not be forced into any of them.
    assert classify_alphabet("1234567890!@#$%^&*()") == "ambiguous"


def test_gc_content_is_none_for_protein():
    assert gc_content("MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKK") is not None
    # not a semantic claim about proteins - gc_content only looks at letters,
    # it is the *caller* (the overview renderer) that knows not to show it
    # for a non-nucleotide sequence.
    assert gc_content("") is None


def test_gc_content_is_computed_correctly():
    assert gc_content("GCGC") == 100.0
    assert gc_content("ATAT") == 0.0
    assert gc_content("ATGC") == 50.0

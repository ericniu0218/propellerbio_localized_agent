# Biotech integration research

Verified against primary sources on 2026-09-11. This note supports implementation planning; no applications, weights, or databases were installed. Recommendations below are engineering judgments, distinguished from upstream requirements.

## Recommended first integrations

Build **self-hosted AlphaFold 2** for protein structure prediction and **open-source PyMOL** for viewing, selecting, and exporting molecular structures. The first useful workflow is:

`Company sequence artifact → AlphaFold 2 job → structure + confidence artifacts → PyMOL session`

These integrations exercise an expensive asynchronous computation and an interactive scientific application with a natural file handoff. Keep the company source connector separate: accepting a FASTA export proves the computation, while authenticated retrieval from the company's actual ELN, LIMS, registry, or document system proves cross-application data access. Neither PyMOL nor AlphaFold substitutes for that source system.

Call the operation **structure prediction**. The result is a predicted configuration, not a time-resolved folding trajectory, an experimental structure, or a validated biological conclusion. AlphaFold's relaxation step improves geometry; it should not be presented as a molecular-dynamics experiment. This distinction follows from the inference and relaxation operations in the [AlphaFold runner](https://github.com/google-deepmind/alphafold/blob/c77e5d2a8961d1a353632c462914ff0a32a950f6/run_alphafold.py).

## Alternatives and commercial deployment constraints

| Candidate | Actual role | Decision for the first release |
| --- | --- | --- |
| AlphaFold 2 | Sequence-to-structure inference | Selected. Code is Apache 2.0 and released parameters are CC BY 4.0. CC BY permits commercial adaptation and sharing subject to its conditions. Retain attribution and an inventory of database/dependency terms. [Upstream licensing](https://github.com/google-deepmind/alphafold#model-parameters), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Open-source PyMOL | Visualization and molecular-data import/export | Selected. Its license permits use, modification, and distribution for any purpose, with copyright/permission notices and restrictions on endorsement/trademarks. This is distinct from licensed commercial PyMOL distributions; choose and record the actual distribution. [Open-source license](https://github.com/schrodinger/pymol-open-source/blob/master/LICENSE), [product offerings](https://pymol.org/) |
| AlphaFold 3 | Broader biomolecular structure prediction | Exclude the publicly supplied weights from this company pilot. Current repository code is Apache 2.0, but separate weight terms restrict parameters **and outputs** to noncommercial organizations/use and prohibit commercial research. Code availability does not clear model usage. [Current code license](https://github.com/google-deepmind/alphafold3/blob/main/LICENSE), [parameter/output terms](https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md) |
| ESM-2 | Protein language model: embeddings and sequence-level analysis | Useful subsequent integration for sequence search/analysis. Its embeddings are not a structure file to feed directly to AlphaFold; pass a sequence artifact through the folding interface. [Official repository](https://github.com/facebookresearch/esm) |
| ESMFold | Structure predictor using an ESM-2 backbone | Strong later alternative if AF2 database operations dominate costs. It does not require an MSA or external sequence databases. The first-party ESM-2 and ESMFold model cards mark these releases MIT. [ESMFold model card](https://huggingface.co/facebook/esmfold_v1), [ESM-2 model card](https://huggingface.co/facebook/esm2_t33_650M_UR50D) |
| Rosetta / PyRosetta | Structural modeling, scoring, refinement, and design | Add after choosing one scientifically defined protocol. Commercial use requires a paid agreement; PyRosetta has a distinct license. A research license should not be assumed to permit repackaging or exposing the software to third parties. [Rosetta license](https://github.com/RosettaCommons/rosetta/blob/main/LICENSE.md), [licensing FAQ](https://rosettacommons.org/licensing-faq/), [downloads/licensing](https://rosettacommons.org/software/download/) |

The original Meta ESM repository was archived in August 2024. Its original ESMFold instructions require Python ≤3.9 and OpenFold/CUDA build dependencies; it also points to a Transformers implementation with simpler dependencies. If chosen later, qualify a maintained, pinned runtime in its own worker image rather than installing the legacy environment into Bedrock's Python 3.11+ client. [Official ESM repository](https://github.com/facebookresearch/esm)

### The likely OpenDDE match

The likely intended project is Aurekaresearch's **OpenDDE**, released as a preview in July 2026. It provides an all-atom co-folding model, conversion, preprocessing, and inference commands; it overlaps the prediction layer rather than demonstrably replacing company authorization, job durability, or artifact management. Its README explicitly says APIs/checkpoints may change and it is not intended for production pipelines. Code and its first-party model card identify Apache 2.0 licensing. Confirm the URL with the user before treating this identification as certain. [Repository](https://github.com/aurekaresearch/OpenDDE), [model card](https://huggingface.co/aurekaresearch/OpenDDE)

One immediate privacy concern is concrete: its protein MSA preprocessing uses the **public ColabFold MMseqs2 API** unless precomputed A3M paths are supplied. A private deployment must supply approved local preprocessing/MSAs or explicitly disable that feature after scientific evaluation; simply self-hosting the predictor does not prevent this outbound sequence transfer. [Official inference instructions](https://github.com/aurekaresearch/OpenDDE/blob/main/docs/inference_instructions.md#preprocess-optional-features)

## AlphaFold 2: specific adapter approach

### Deployment and capacity

Upstream supports Linux and a modern NVIDIA GPU. Current documentation gives approximately **556 GB to download**, **2.62 TB unpacked**, and up to **3 TB for databases**, with SSD recommended. The reduced preset documents **600 GB disk, 8 CPU cores, and 8 GB RAM**; these figures are not a guarantee of end-to-end memory requirements for every target. Its tested full configuration used an A100, 12 vCPUs, and 85 GB RAM. Model parameters occupy about 5.3 GB. [Current installation and runtime documentation](https://github.com/google-deepmind/alphafold#installation-and-running-your-first-prediction)

Planning recommendation: start with one dedicated GPU worker, a declared maximum sequence length, and one folding job at a time. Reserve additional capacity for unpacking, scratch, outputs, backups, and future database snapshots; do not interpret 3 TB as the complete machine disk budget. Benchmark representative company proteins before purchasing more capacity or promising latency. Start the plumbing with reduced databases if necessary, but identify that preset in every result and compare it with full databases before adopting it scientifically.

Keep the chat-model server and folding worker separately deployed. Their installation stacks, resource use, restart behavior, and latency differ. The employee terminal only submits and observes jobs; it should remain responsive if a scientific process takes hours or fails.

### Invocation contract

Use the official container launcher as the reference, then expose a typed job contract around its fixed arguments: `fasta_paths`, `data_dir`, `output_dir`, `model_preset`, `db_preset`, `max_template_date`, `random_seed`, and relaxation settings. The Docker launcher maps host data into its container and supports GPU selection. Its CLI is a batch interface, not a hosted inference API. [Official launcher](https://github.com/google-deepmind/alphafold/blob/main/docker/run_docker.py)

Proposed initial capability: `alphafold2.predict_monomer(sequence_artifact_id, preset_id, seed)`. An operator-owned preset supplies database paths, image digest, template cutoff, GPU allocation, and allowed flags. Use `monomer_ptm` when pairwise confidence is required; keep this scientific setting explicit and qualified. Reject unsupported input instead of silently switching to multimer or truncating sequences. Multimers, nucleic acids, ligands, and modified residues require separate capabilities and validation.

### Outputs and version differences

At immutable upstream revision `c77e5d2a8961d1a353632c462914ff0a32a950f6`, the runner writes PDB **and mmCIF** structures, per-model confidence JSON, optional PAE JSON, ranking/timing JSON, and pickle intermediates. The v2.3.2 tagged runner predates those mmCIF/confidence-JSON helpers. Therefore, a plan that promises these files must pin the newer revision or implement documented conversion/extraction for the older release. [Current output source](https://github.com/google-deepmind/alphafold/blob/c77e5d2a8961d1a353632c462914ff0a32a950f6/run_alphafold.py#L428), [mmCIF writing](https://github.com/google-deepmind/alphafold/blob/c77e5d2a8961d1a353632c462914ff0a32a950f6/run_alphafold.py#L465), [ranked output writing](https://github.com/google-deepmind/alphafold/blob/c77e5d2a8961d1a353632c462914ff0a32a950f6/run_alphafold.py#L518), [v2.3.2 source](https://github.com/google-deepmind/alphafold/blob/v2.3.2/run_alphafold.py)

Recommended published result bundle: immutable original FASTA; chosen PDB/mmCIF; other model candidates; normalized per-residue confidence; PAE when available; model ranking; relaxation status; run manifest; and logs with restricted visibility. Keep raw worker files for provenance according to retention policy. Do not let the user interface or general API deserialize uploaded pickle files; extraction of trusted worker-produced intermediates belongs inside the isolated worker.

## PyMOL: specific adapter approach

PyMOL offers command-line batch mode (`-c`), a Python command API, and managed `pymol2.PyMOL` instances. Use a short-lived process with a fixed script or a managed Python instance for rendering/export. [Command-line implementation](https://github.com/schrodinger/pymol-open-source/blob/master/modules/pymol/invocation.py), [instance API](https://github.com/schrodinger/pymol-open-source/blob/master/modules/pymol2/__init__.py)

Its import API loads molecular files, and its export API supports writing structures, sessions, images, and FASTA sequence strings. Sequence export operates on selected atoms, so an exported sequence from an incomplete coordinate model should not be assumed to contain unresolved residues. [Import implementation](https://github.com/schrodinger/pymol-open-source/blob/master/modules/pymol/importing.py), [export and FASTA implementation](https://github.com/schrodinger/pymol-open-source/blob/master/modules/pymol/exporting.py)

Proposed capabilities:

1. `pymol.render_structure(structure_artifact_id, style_preset_id)`: produce a PNG preview and versioned session using a reviewed style preset. Add confidence coloring only when the artifact explicitly declares that its stored values mean pLDDT.
2. `pymol.prepare_session(structure_artifact_id)`: materialize an authorized bundle on the employee machine and open it in configured native PyMOL. This preserves Bedrock's terminal interface and allows interactive 3D viewing.
3. Later, add an optional PyMOL plugin to export a chosen object/chain with provenance. Treat this as a separate desktop integration milestone, not an automatic consequence of server rendering.

Keep the original structure unchanged. Exports, selections, and rendered sessions are derived artifacts. Preserve atom/residue identifiers, chains, insertion codes, alternate conformations, and source mappings. Test headless rendering in the actual container; graphics/runtime dependencies can differ from desktop installations. Use one process per job initially to simplify cleanup and isolate crashes.

## Difficulties and precautions that affect the design

- **Scientific confidence travels with the structure.** pLDDT measures local confidence and PAE concerns relative placement. High local confidence does not establish that two domains have the right orientation. Show distributions and low-confidence regions, not just an average or green success indicator. [EMBL-EBI confidence guidance](https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/)
- **Stored values have semantics.** AF predictions use the coordinate file's B-factor field for confidence; experimental structures can use it differently. Low pLDDT can reflect disorder, and confidence is not an experimental stability measurement. Record `confidence_metric=plddt` explicitly. [EMBL-EBI pLDDT guidance](https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/evaluating-alphafolds-predicted-structures-using-confidence-scores/plddt-understanding-local-confidence/), [input/output recap](https://www.ebi.ac.uk/training/online/courses/alphafold/inputs-and-outputs/alphafold-inputs-and-outputs-recap/)
- **Reproducibility requires more than a seed.** AF2 warns GPU inference can remain nondeterministic; precomputed-MSA reuse does not verify that sequence, database, or settings match. Use cache keys covering sequence digest, database snapshot, preprocessing settings, and tool version; retain all of those in provenance. [Runner flags](https://github.com/google-deepmind/alphafold/blob/v2.3.2/run_alphafold.py#L131)
- **Private execution must be tested.** Stage approved images, weights, and databases before private runs. Deny external worker egress and verify a real job succeeds. Prevent PyMOL remote fetching and unapproved runtime downloads by exposing only local artifact paths through fixed operations.
- **Data transfer is an authorization operation.** Recheck access when a downstream job starts and when results are downloaded. Propagate project restrictions to derived files and caches. Imported filenames and descriptive text must never become shell syntax or executable PyMOL scripts.
- **Failures need precise outcomes.** Separate prediction failure, GPU out-of-memory, relaxation failure, rendering failure, cancellation, and artifact-publication failure. A failed preview should not destroy a successful prediction. A disconnected terminal should not cancel the computation implicitly.

## Integration acceptance evidence

Before calling the two integrations complete, demonstrate a real public test sequence passing through ingestion, prediction, confidence extraction, and native PyMOL opening. Verify structure/sequence correspondence, identifiers, confidence indexing, and immutable lineage. Include a low-confidence example so the UI's interpretation is tested, not merely the happy path.

Also demonstrate that a worker restart is recoverable, unauthorized project access is denied, a repeated submission does not duplicate an expensive job, an oversized sequence receives a clear response, and a rendering failure leaves the structure available. Run the full workflow with outbound internet blocked after provisioning. Record measured wall time, GPU memory, CPU memory, storage growth, and software/data digests. These are proposed acceptance criteria; no such integration tests were executed during this research.

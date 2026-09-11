# AlphaFold 2 and open-source PyMOL integration

This project now contains a local, working sequence-to-structure-to-visualization path:

```text
one-protein FASTA
  -> validated and copied into a private run directory
  -> official AlphaFold 2 Docker launcher
  -> ranked PDB or mmCIF + confidence files
  -> open-source PyMOL headless preview + session
  -> manifest with hashes, settings, lineage, and tool logs
```

This first slice runs tools on the machine where Bedrock is running. The separate agent-pack backend referenced by this repository was not available when the integration was built. Move the same adapters behind that backend before multi-user production deployment so jobs survive terminal disconnects and execute on centrally scheduled GPUs.

## Provisioning

1. Inventory the GPU host: Linux distribution, NVIDIA driver, CUDA compatibility, GPU count and VRAM, CPU/RAM, and free SSD capacity. The purchase price of the GPUs does not determine which sequence sizes fit. Benchmark representative sequences before setting a production limit.
2. Install Docker Engine and NVIDIA Container Toolkit on the private GPU host.
3. Clone and pin [AlphaFold 2](https://github.com/google-deepmind/alphafold) to a reviewed commit. Build its Docker image with an immutable tag or digest, and install `docker/requirements.txt` in a small launcher virtual environment.
4. Download the AlphaFold model parameters and databases using the pinned repository's scripts. Current upstream guidance calls for up to roughly 3 TB for full databases. Keep additional room for unpacking, scratch space, run outputs, and database updates.
5. Install [open-source PyMOL](https://github.com/schrodinger/pymol-open-source) in a pinned environment. Confirm that `pymol -cq` can render with the GPU host's headless graphics libraries.
6. After provisioning, block outbound network access from both tool workers and run a known public sequence end to end. AlphaFold 2 must use only the locally staged databases.

No weights or databases are downloaded by Bedrock. Provisioning remains an operator-controlled step because the database download is large, versions must be reviewed, and the worker must be tested with network egress disabled.

## Configuration

Put the following in `~/.config/bedrock/config.toml` on the worker host. Values shown are examples:

```toml
backend = "agentpack"
biotech_work_dir = "/srv/bedrock-biotech"

alphafold_launcher = "/opt/alphafold/docker/run_docker.py"
alphafold_python = "/opt/alphafold-launcher/bin/python"
alphafold_data_dir = "/data/alphafold"
alphafold_max_template_date = "2026-08-31"
alphafold_docker_image = "alphafold@sha256:REPLACE_WITH_REAL_DIGEST"
alphafold_database_version = "snapshot-2026-08-31"
alphafold_model_preset = "monomer_ptm"
alphafold_db_preset = "full_dbs"
alphafold_gpu_devices = "0"
alphafold_models_to_relax = "best"
biotech_max_sequence_length = 2500

pymol_executable = "/opt/pymol/bin/pymol"
```

The same settings have environment-variable equivalents:

| Setting | Environment variable |
|---|---|
| `biotech_work_dir` | `BEDROCK_BIOTECH_WORK_DIR` |
| `alphafold_launcher` | `ALPHAFOLD_LAUNCHER` |
| `alphafold_python` | `ALPHAFOLD_PYTHON` |
| `alphafold_data_dir` | `ALPHAFOLD_DATA_DIR` |
| `alphafold_max_template_date` | `ALPHAFOLD_MAX_TEMPLATE_DATE` |
| `alphafold_docker_image` | `ALPHAFOLD_DOCKER_IMAGE` |
| `alphafold_database_version` | `ALPHAFOLD_DATABASE_VERSION` |
| `alphafold_model_preset` | `ALPHAFOLD_MODEL_PRESET` |
| `alphafold_db_preset` | `ALPHAFOLD_DB_PRESET` |
| `alphafold_gpu_devices` | `ALPHAFOLD_GPU_DEVICES` |
| `alphafold_models_to_relax` | `ALPHAFOLD_MODELS_TO_RELAX` |
| `biotech_max_sequence_length` | `BEDROCK_MAX_SEQUENCE_LENGTH` |
| `pymol_executable` | `PYMOL_EXECUTABLE` |

`alphafold_max_template_date`, the Docker image, and the database snapshot are deliberately explicit. A random seed alone does not make AlphaFold GPU inference reproducible, so each run manifest also hashes the launcher, input FASTA, normalized sequence, and published outputs.

## Use

Reinstall the editable package once to register the new command:

```bash
source .venv/bin/activate
pip install -e ".[dev]"
bedrock-biotech status
bedrock-biotech fold /approved/project/candidate.fasta
bedrock-biotech pymol
```

In `BEDROCK_BACKEND=agentpack` mode, the same operations are available inside the terminal application:

```text
/tools
/fold /approved/project/candidate.fasta
/pymol
/pymol /path/to/an/existing/structure.cif
```

The same actions can be written naturally. For example:

```text
Run AlphaFold 2 on /approved/project/candidate.fasta
Could you display "/approved/project/my structure.pdb" with PyMOL?
Please open the latest structure interactively in PyMOL
```

Clear action requests run the integration directly. Explanatory questions such
as “How do I open a structure in PyMOL?” continue to the conversational agent.
Quote a path when it contains spaces.

The first capability accepts exactly one FASTA record and rejects sequences above the configured limit. It uses `monomer_ptm` by default. Multimers, ligands, nucleic acids, modified residues, and structure refinement need separate, scientifically validated capabilities.

Run directories default to `~/.local/share/bedrock/biotech/runs/<run-id>`. A run contains the original FASTA snapshot, a normalized AlphaFold input, raw tool output, logs, a PNG preview, a `.pse` session, and `manifest.json`. Directories are created with owner-only permissions. Treat the original FASTA as authoritative; a sequence exported from a coordinate structure may omit unresolved residues.

## Production precautions and remaining work

- The present runner is local and process-based. Central deployment needs a durable queue, database-backed job state, GPU admission control, cancellation, retries, per-project access checks, quotas, and recovery after worker or client restarts.
- The source company system is still unknown. Start with an approved file export. Once the ELN, LIMS, registry, or object store is identified, add a read-only source adapter that returns the same FASTA artifact and propagates project permissions to every derived file.
- Filenames never enter a shell or PyMOL command string. The adapters use argument arrays and a fixed rendering script. Preserve this rule when adding new operations.
- Keep workers offline during execution. Runtime downloads, PyMOL fetches, and public MSA services can expose proprietary sequences.
- PyMOL colors AlphaFold structures by pLDDT only because the run declares that semantic explicitly. Do not apply that palette to arbitrary experimental B-factors.
- A successful prediction remains usable if preview rendering fails. Users receive the structure and the manifest records a render warning.
- Prediction is not a folding trajectory, molecular-dynamics simulation, or experimental validation. Low pLDDT and PAE must travel with the coordinates and remain visible during review.

The source and licensing research behind these choices is in [BIOTECH_INTEGRATION_RESEARCH.md](BIOTECH_INTEGRATION_RESEARCH.md).

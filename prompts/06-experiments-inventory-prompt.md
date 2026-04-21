Analyse this repository's experiment landscape and produce a single documentation page that distinguishes:

1. every distinct model family used across the repo
2. every distinct experiment family / workstream
3. the purpose, inputs, outputs, and status of each experiment family
4. how the experiments relate to each other

Repository root: `/home/daniel/git/vidiq-hpc`

## Deliverables

- Create `/home/daniel/git/vidiq-hpc/experiments.md`
- Update `/home/daniel/git/vidiq-hpc/README.md` to add a visible link to `experiments.md` in an appropriate top-level section

## What To Analyse

- `README.md`
- `experiments/`
- `hpc/`
- `reports/`
- `prompts/`
- `meetings/`
- any run/config/metadata files needed to identify model names, dataset names, and experiment intent

## Primary Objective

Write a clear repo-level map of the experiments so a new contributor can tell:

- which models are being used
- which are embedding models vs classifier/fine-tuning models
- which experiments are binary vs multiclass
- which are text embedding geometry analyses vs Qwen fine-tuning runs vs supporting analysis/reporting work
- which datasets are involved
- which outputs are checked in

## Required Structure For `experiments.md`

- Title and 2-4 sentence overview
- `Experiment Families` section
- `Models Used` section
- `Datasets Used` section
- `How The Workstreams Connect` section
- `Key Output Locations` section
- `Gaps / Ambiguities` section

## Requirements

- Distinguish model families precisely. Do not collapse different embedding backbones and fine-tuned classifier models into one bucket.
- Identify concrete model names from the repo where possible, for example from configs, metadata, scripts, SLURM files, and existing READMEs.
- Separate base/pretrained embedding models from fine-tuned training backbones.
- Separate analysis pipelines from training pipelines.
- Call out whether a directory contains code, artifacts, reports, or run outputs.
- Infer the nature of each experiment from code and tracked artifacts, but do not invent claims unsupported by the repo.
- When something is ambiguous or partially documented, say so explicitly in `Gaps / Ambiguities`.
- Prefer concise, high-signal prose over exhaustive file listings.

## Specific Things To Resolve

- What embedding models are used in `experiments/text/` and related artifacts
- What model variants appear in SST-2 binary experiments
- What models appear in multiclass emotion experiments
- What the `experiments/embeddings_field/` workstream is doing and how it differs from `experiments/text/`
- How `experiments/text_model/` and `hpc/*.slurm` relate to the Qwen fine-tuning work
- Which tracked outputs suggest completed runs versus plans or scaffolding

## Editing Guidance

- Make the new document repo-level, not a dump of directory contents
- Add only the minimal README change needed to link to `experiments.md`
- Preserve existing README tone and structure

## Acceptance Criteria

- A reader can identify all distinct model families used in the repo
- A reader can distinguish the main experiment families without opening subdirectories
- The README links to `experiments.md`
- Any uncertain classifications are explicitly labelled as uncertain

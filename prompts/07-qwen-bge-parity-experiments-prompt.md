Design and implement a Qwen-based experiment set that mirrors the existing `BAAI/bge-base-en-v1.5` experiment outputs as closely as is technically defensible.

Repository root: `/home/daniel/git/vidiq-hpc`

## Goal

We already have a substantial set of experiments and artifacts built around `BAAI/bge-base-en-v1.5`. We now want the repository to support an analogous experiment family for `Qwen/Qwen3-1.7B` so that we can inspect:

- embeddings
- geometry metrics
- projections and plots
- run metadata
- any other directly comparable derived artifacts from the original BGE experiments

The objective is not to pretend Qwen is the same kind of model as BGE. The objective is to create a careful, explicit Qwen counterpart so we can compare the resulting representations and downstream analysis artifacts against the existing BGE-based experiment outputs.

## Primary Deliverable

Implement a new Qwen experiment path in the most appropriate existing experiment location, with clear naming that signals this is the Qwen counterpart to the current BGE experiments.

You should choose the location and naming using the current repo conventions, but prefer extending the dataset-specific experiment roots rather than inventing a disconnected structure unless there is a strong reason not to.

## What To Analyse First

Read and use:

- `README.md`
- `experiments.md`
- `experiments/text/binary/sst2/`
- `experiments/text/multiclass/dair-ai-emotion/`
- `experiments/text_model/`
- `experiments/embeddings_field/`
- `hpc/`
- relevant run configs, artifacts, metrics, and plotting scripts

Focus especially on the BGE experiment surfaces that already exist, including:

- embedding-generation outputs
- raw / l2 / centered_l2 variants where applicable
- metric summaries
- projection summaries
- plots
- run metadata

## Task

Create the repo changes needed so that Qwen can be run through an experiment flow that produces the closest practical analogue to the BGE experiment family.

This should include:

1. identifying which existing BGE experiments are genuinely portable to Qwen
2. identifying which ones need adaptation because Qwen is a causal LM rather than a sentence-embedding model
3. implementing the Qwen experiment flow
4. storing outputs in a clear, discoverable location
5. documenting exactly how the Qwen embeddings are formed

## Required Outcomes

Produce a Qwen counterpart for the existing BGE experiment family with:

- exported embeddings
- metadata describing embedding formation
- metrics comparable to the existing BGE metrics where comparison is valid
- plots analogous to the current BGE plots where technically meaningful
- run folders and artifact files that fit existing naming conventions

If a BGE artifact has no valid Qwen analogue, do not fake it. Instead:

- skip it
- document why it is not comparable
- produce the closest honest alternative if one exists

## Scope Guidance

Treat the multiclass `dair-ai/emotion` experiment family as the primary target, because the repo already contains a tracked Qwen fine-tuning pipeline and a dataset-level bridge there.

You may extend the binary SST-2 flow only if there is a clean and justified way to do so with the current repo architecture. Do not force binary Qwen work into scope if that would require speculative or low-quality design.

## Specific Requirements

- Reuse existing experiment conventions where possible.
- Prefer extending existing `src/`, `configs/`, `runs/`, `artifacts/`, and plotting patterns rather than building one-off scripts in random locations.
- Keep Qwen outputs clearly distinguished from BGE outputs in names and directories.
- Make the embedding-extraction method explicit in code and documentation.
- If Qwen embeddings are taken from the final hidden state, specify the exact token position and any pooling rule used.
- Preserve provenance in run metadata, including model name, dataset source, split information, and artifact paths.
- Generate artifacts that are actually useful for comparison to the BGE stage.

## Questions You Must Resolve

- Which BGE experiment outputs already have a natural Qwen analogue?
- Should Qwen use the existing fine-tuned run outputs, a new embedding-generation pass, or both?
- Should the Qwen comparison be based on:
  - the fine-tuned Qwen model
  - the base Qwen model
  - or both

Use repo evidence to decide. If you include only one, justify why.

- Where should the new Qwen comparison experiments live so the structure remains coherent?
- Which existing plotting and metric scripts can be reused directly, and which require Qwen-specific adaptation?

## Implementation Expectations

Implement the necessary code, configs, and documentation changes, not just a plan.

At minimum, add:

- the new or updated experiment code
- any new configs
- any README or experiment documentation updates needed so the new Qwen experiment family is discoverable

If useful, also add:

- a concise experiment README in the chosen root
- notes explaining comparability limits between BGE and Qwen representations

## Naming

Use names that make the relationship obvious. A new contributor should be able to tell, from directory and run names alone, that these are Qwen counterparts to the BGE experiment family.

Prefer names along the lines of:

- `qwen-embedding-generation`
- `qwen-variant-visuals`
- `qwen-bge-parity`

Do not use vague names like `new-experiment` or `alt-model`.

## Acceptance Criteria

- The repo contains a clearly named Qwen counterpart to the relevant BGE experiment family.
- The Qwen experiment flow produces embeddings plus comparable downstream artifacts.
- The output location is coherent with the existing repo structure.
- The embedding formation method is explicit and documented.
- Any non-comparable pieces are clearly called out rather than silently omitted.
- The result is usable for side-by-side inspection of BGE-derived versus Qwen-derived experiment outputs.

## Final Output

When finished:

- summarize what you implemented
- list the exact files you changed
- explain what is directly comparable between BGE and Qwen
- explain what is not directly comparable and why

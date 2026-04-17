# Work Diary

## 2026-03-31

- Created `prompts/00-read-work-diary.md` as the starter prompt that instructs the agent to read this diary first, report ready for work, and log all work here.
- Read `work-diary.md` to action the starter prompt and activated diary logging for all subsequent work.
- Read `prompts/00-read-work-diary.md` and re-read `work-diary.md` to follow the repository starter prompt before proceeding with any new task.
- Updated `prompts/00-read-work-diary.md` so diary entries are only required after significant work or when explicitly requested by the user.
- Added `prompts/01-survey-text-datasets-for-embedding-analysis.md`, a prompt directing an AI agent to shortlist simple, interpretable text datasets for embedding-space research, with emphasis on clean semantic contrasts and concrete recommendation criteria.
- Added `TODO.md` with report assignments split across Pritish and Daniel: `ChatGPT` and `Gemini` for Pritish, `Grok` and `Claude` for Daniel.
- Checked Git and SSH setup for GitHub access. Current repo remote uses HTTPS, `~/.ssh` does not contain a GitHub key yet, and a direct `ssh -T git@github.com` check could not complete in the sandbox because hostname resolution is blocked.
- Created `howto.md` for setup and operating notes, and added `howto.md` to `.gitignore` so it remains untracked.
- Generated a new GitHub SSH key at `~/.ssh/id_ed25519_github`, captured the public key for GitHub account setup, and updated the repo `origin` remote to use SSH in `.git/config`.
- Added GitHub's SSH host key to `~/.ssh/known_hosts` and tested authentication with `ssh -i ~/.ssh/id_ed25519_github -o IdentitiesOnly=yes -T git@github.com`. Connection reached GitHub, but authentication is still waiting on the public key being added to the GitHub account.
- Created `README.md` with a reminder for Pritish to run the project inside `gemini_env`.
- Prepared the repo for publishing: verified pending files, confirmed branch `main`, and confirmed `origin` points to `git@github.com:pritishrv/vidiq-hpc.git`.
- Committed the initial project files as `d6dab1d` with message `Add project prompts and setup notes`, then pushed `main` to `origin` over SSH successfully.
- Attempted to copy the GitHub SSH keypair to the HPC account at `adhp543@localhost` on port `2000`, but the SSH connection was rejected with `Permission denied (publickey,gssapi-keyex,gssapi-with-mic,password)` before any files were transferred.
- Retried the transfer with direct `scp -P 2000` to `adhp543@localhost:~/.ssh/`. The copy still failed with the same authentication error before any files could be written remotely.
## 2026-04-17

- Fixed the Qwen/Hugging Face resolution in the text-model training workflow: `experiments/text_model/train_multiclass.py` now defaults to the real repo id `Qwen/Qwen3-1.7B`, and the multiclass SLURM scripts no longer require a separate `MODEL_REF` environment variable.
- Added `hpc/check_model_access.slurm` as a fast smoke test that loads the default Qwen tokenizer/model and runs a tiny forward pass before committing to longer training jobs.
- Added six new long-run multiclass batch scripts with 72-hour limits: `train_multiclass_balanced_{250e,500e,1000e}.slurm` and `train_multiclass_unbalanced_{250e,500e,1000e}.slurm`.
- Updated `.gitignore` so `experiments/text_model/runs/<run_id>/analysis`, `run_metadata.json`, and `train_metrics.json` can be tracked, while `model/` and `tokenizer/` remain ignored.
- Pulled the first completed balanced fine-tuning run artifacts into the repo under `experiments/text_model/runs/tmqb0010_17763/`, including `analysis/centroids.json`, `analysis/eval_embeddings.npy`, `analysis/eval_logits.npy`, `analysis/eval_metrics.json`, `run_metadata.json`, and `train_metrics.json`.
- Confirmed the first completed balanced run exists and appears structurally successful; detailed metric interpretation is still pending.
- Added `prompts/04-image-sentiment-dataset-search.md` to drive external dataset-search agents.
- Confirmed six image-dataset survey reports are now present for analysis:
  - `reports/chatgpt-image-datasets.md`
  - `reports/claude-image-datasets.md`
  - `reports/deepseek-image-datasets.md`
  - `reports/gemini-image-datasets.md`
  - `reports/grok-image-datasets.md`
  - `reports/lechat-image-datasets.md`
- The image-dataset search prompt `prompts/04-image-sentiment-dataset-search.md` and the generated image-dataset reports are still local/untracked in this checkout and need to be added to the repo.
- Main handoff for tomorrow:
  - inspect `experiments/text_model/runs/tmqb0010_17763/train_metrics.json` and `analysis/eval_metrics.json`
  - assess whether the 10-epoch balanced run is learning useful class structure
  - read across the six image-dataset reports and synthesize a ranked shortlist
  - add `prompts/04-image-sentiment-dataset-search.md` plus the image-dataset reports to git
  - compare the shortlist against the 16 April minutes item on the image extension
  - decide the first image dataset and whether to start with full-image embeddings only or immediately plan segmented-object follow-up

- Created `~/.ssh/config` with a `github.com` host entry that forces Git to use `~/.ssh/id_ed25519_github` via `IdentitiesOnly yes`.
- Added `reports/gemini-text-dataset-report.md`, a survey report on text datasets for embedding-space analysis, and prepared it for publishing to GitHub.
- Added `reports/openai-text-dataset-report.md` and prepared the new report plus the diary update for publishing to GitHub.

## 2026-04-11

- Read the 11 April meeting transcript from `meetings/2026-04-11_transcript_llm-embedding-geometry_sentiment-experiments_neurips.docx`.
- Created `meetings/2026-04-11_minutes_llm-embedding-geometry_sentiment-experiments_neurips.md` with full meeting minutes, discussion notes, and action points for both Pritish and Daniel through to the 4 May NeurIPS deadline.
- Created `experiments/text/multiclass/dair-ai-emotion/reports/next-experiments-plan.md` defining five new experiment stages: (1) inter-centroid Euclidean distance matrix in 768D, (2) cross-model validation of the belt-density pattern, (3) fine-tuned classifier with logit-geometry correlation analysis, (4) improved stacked overlap bar charts, (5) multi-panel pairwise scatter plot figure.

## 2026-04-12

- Pulled latest changes from remote: Pritish had pushed `experiments/embeddings_field/text/binary/plan_density_analysis.md`, `run_density_overlap_analysis.py`, multiclass embeddings_field placeholders, and a new `experiments/text_model/` directory (`train_multiclass.py`, `requirements.txt`, `README.md`).
- Read three meeting transcripts from today's sessions (`meetings/2026-04-12_1_minutes.docx`, `_2_minutes.docx`, `_3_minutes.docx`).
- Created `meetings/2026-04-12_minutes_embedding-geometry_density-analysis_hpc-training.md` with full minutes and action points covering: density calculation correction (y-axis label fix, per-unit-volume density only drops), the void finding (no points from centroid to r≈7.5 for any emotion), the no-pure-emotion observation, the corrected overlap definition (distance-to-centroid comparison, no radius threshold), centroid distance matrix results (sadness nearest anger / furthest joy), radial scatter results (fear most dispersed at 12.77, love tightest at 11.32), and the HPC fine-tuned model training plan (Qwen3 1.7B, 768-dim embedding layer, 5-class head).
- Committed and pushed all four files (3 transcripts + minutes) as commit `f329143`.

## 2026-04-13

- Updated `experiments/text_model/train_multiclass.py` to cast the final hidden state to `float32` before the projection head and increased the classifier output size from 5 to 6 labels for the balanced 6-class setup.
- Checked the processed multiclass training labels and confirmed the current `texts.jsonl` split contains labels `0-5`, with 572 examples in class `5`.
- Added Hyperion batch scripts under `hpc/`, including `train_multiclass.slurm` and `train_multiclass_frozen_backbone.slurm`, with GPU A100 resource settings, run-duration footer, and completion email handling.
- Added raw source datasets `experiments/text/multiclass/dair-ai-emotion/data/raw/balanced_emotions_6classes.csv` and `experiments/text/multiclass/dair-ai-emotion/data/raw/20-emotion-dataset.csv`.
- Added planning prompts `prompts/02-archive-text-model-runs.md` and `prompts/03-slurm-run-id-from-job-name-and-id.md` to define the archive/symlink workflow and the SLURM-derived run-id scheme.
- Extended `experiments/text_model/train_multiclass.py` with an archive-root workflow: large `model/` and `tokenizer/` artifacts now save under `/users/aczd097/archive/vidiq-hpc/text_model/<run_id>/`, the local run directory keeps analysis and metrics, and symlinks reconnect the archived model/tokenizer paths back into the visible run folder.
- Added `run_metadata.json` generation with timestamp, git commit, args, dataset information, archive paths, symlink targets, and SLURM job metadata (`SLURM_JOB_NAME`, `SLURM_JOB_ID`).
- Updated `.gitignore` to ignore `experiments/text_model/runs/` so generated run outputs do not accumulate in Git.
- Changed the text-model run-id scheme so SLURM jobs use `<job_name>_<job_id>` as the run identifier, and updated the default/fallback naming logic accordingly.
- Updated `hpc/train_multiclass.slurm` and `hpc/train_multiclass_frozen_backbone.slurm` to use short 8-character SLURM job names and to pass `--run-name "${SLURM_JOB_NAME}_${SLURM_JOB_ID}"`.
- Extended `experiments/text_model/train_multiclass.py` to support direct CSV training via `--csv-path`, automatically infer the text column, map string emotion labels to numeric ids, and size the classifier output layer from the dataset label set so both the balanced 6-class CSV and the unbalanced 20-class CSV can be trained directly.
- Added six new SLURM scripts for balanced and unbalanced CSV training at 10, 50, and 100 epochs: `train_multiclass_balanced_{10e,50e,100e}.slurm` and `train_multiclass_unbalanced_{10e,50e,100e}.slurm`.
- Increased the wall-clock limit on those six new SLURM jobs from 4 hours to 24 hours.
- Diagnosed the suspected large-file Git problem: the current repo and `origin/main` are clean (`.git` around single-digit MB locally), so the slowdown appears isolated to the old Hyperion clone rather than current remote history.
- Verified this by making a fresh filtered temp clone from `origin`; the clean clone measured roughly `7.2M` for `.git` and `23M` total checkout size before deletion.
- Identified that Hyperion cannot clone over SSH (`github.com:22` timeout), so the recovery path there must use HTTPS instead of `git@github.com:...`.
- Confirmed the current SLURM failures are due to `--model-path models/qwen3-1.7B` not existing on Hyperion. The next required change is to update the batch scripts to stop using that fake local path and instead use either a real local checkpoint path or the real Hugging Face model id, with explicit HF/proxy environment exports inside the SLURM scripts rather than relying on `.bashrc`.

## 2026-04-07

- Reviewed the local literature set in `/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq/lit-survey/gemini` using the extracted summary report in `/Users/pritishrv/Documents/VIDEO_UNDERSTANDIG/vidiq/reports/title-abstract-conclusion.md`.
- Verified that the extraction report covers all 82 PDFs in the local literature folder and used it as the basis for a full pass over the set.
- Synthesized model-selection guidance for sentence embeddings, including model-family choice, pooling strategy, normalization strategy, and validation metrics for embedding geometry experiments.
- Added `reports/binary-emotion-dataset-experiment-report.md` as a reusable prompt-context document for later implementation and analysis work in `vidiq-hpc`.
- Reviewed current SST-2 dataset information from public online sources and updated `reports/binary-emotion-dataset-experiment-report.md` with SST-2-specific guidance on splits, preprocessing, sequence length, validation priorities, baselines, and dataset-specific risks.
- Reviewed the existing OpenAI and Gemini dataset survey reports to select a multiclass follow-up dataset.
- Chose `dair-ai/emotion` as the recommended next multiclass dataset and added `reports/multiclass-emotion-dataset-experiment-report.md` with dataset-specific guidance on model choice, embedding construction, multiclass validation metrics, confusion analysis, class-imbalance handling, and recommended baselines.
- Created an experiment-first directory structure under `experiments/text/` with separate roots for `binary/sst2` and `multiclass/dair-ai-emotion`.
- Moved the binary and multiclass experiment reports into their dataset-specific `reports/` folders and added placeholders / README files for prompts, source code, configs, data staging, artifacts, and run outputs.
- Added `experiments/text/binary/sst2/reports/model-selection-first-plan.md` to define a controlled model-comparison stage for SST-2 before running broader embedding-formation and validation ablations.
- Added `experiments/text/binary/sst2/requirements.txt` to track the current Python dependencies required by the SST-2 experiment code and to keep that dependency list updated as the experiment code evolves.
- Updated the SST-2 model-selection setup to use a configurable stratified `10%` sample from the train split with the full validation split retained, and aligned the config, data pipeline, and planning report with that smaller first-pass selection strategy.
- Completed the first SST-2 model-selection run on the stratified `10%` train subset plus the full validation split, selected `BAAI/bge-base-en-v1.5` as the main experiment model based on stronger geometry-oriented behavior, updated the main SST-2 report with the result, and added a dedicated BGE ablation-stage config plus runner for the next embedding-formation experiments.
- Added `tqdm` progress reporting to the embedding batch loop so future experiment runs expose mid-process progress during dataset encoding, and updated the experiment requirements file to list `tqdm` explicitly.
- Added stage-level progress logging to the SST-2 model-selection and BGE-ablation runners, including variant/model progress bars, explicit metric-stage start/finish logs, and incremental metric writes so partial results are visible before a long run fully completes.
- Added a dedicated visualization-only plotting script for the SST-2 BGE variants that projects the validation embeddings with PCA and UMAP for inspection while keeping all quantitative metrics on the original 768-dimensional vectors.
- Completed the SST-2 BGE ablation analysis, updated the main binary experiment report with the final quantitative and visual conclusions, and added a dedicated findings report that closes the current binary experiment with `BAAI/bge-base-en-v1.5` raw mean pooling as the default setup.
- Added `experiments/text/multiclass/dair-ai-emotion/reports/model-selection-first-plan.md` to mirror the binary workflow for the multiclass `dair-ai/emotion` experiment: fixed-recipe model selection first, then selected-model ablations, then baselines and visualization.
- Revised the `dair-ai/emotion` multiclass plan to skip model selection and carry forward `BAAI/bge-base-en-v1.5` from the completed binary experiment to save time and avoid redundant compute, then added the first multiclass embedding-generation config, requirements file, and source pipeline to start producing multiclass BGE embeddings directly.
- Added the multiclass evaluation and visualization stage for `dair-ai/emotion`, including original-space metrics for the three saved BGE variants and visualization-only PCA / nonlinear projections plus centroid-distance heatmaps.
- Documented the finalized multiclass findings in `reports/multiclass-emotion-dataset-findings.md` and updated `reports/multiclass-emotion-dataset-experiment-report.md` with the current metrics snapshot plus the decision to keep raw mean pooling as the default.

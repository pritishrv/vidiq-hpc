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
- Created `~/.ssh/config` with a `github.com` host entry that forces Git to use `~/.ssh/id_ed25519_github` via `IdentitiesOnly yes`.
- Added `reports/gemini-text-dataset-report.md`, a survey report on text datasets for embedding-space analysis, and prepared it for publishing to GitHub.
- Added `reports/openai-text-dataset-report.md` and prepared the new report plus the diary update for publishing to GitHub.

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

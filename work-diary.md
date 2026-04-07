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

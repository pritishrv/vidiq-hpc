# Work Diary

## 2026-03-31

- Created `prompts/00-read-work-diary.md` as the starter prompt that instructs the agent to read this diary first, report ready for work, and log all work here.
- Read `work-diary.md` to action the starter prompt and activated diary logging for all subsequent work.
- Checked Git and SSH setup for GitHub access. Current repo remote uses HTTPS, `~/.ssh` does not contain a GitHub key yet, and a direct `ssh -T git@github.com` check could not complete in the sandbox because hostname resolution is blocked.
- Created `howto.md` for setup and operating notes, and added `howto.md` to `.gitignore` so it remains untracked.
- Generated a new GitHub SSH key at `~/.ssh/id_ed25519_github`, captured the public key for GitHub account setup, and updated the repo `origin` remote to use SSH in `.git/config`.
- Added GitHub's SSH host key to `~/.ssh/known_hosts` and tested authentication with `ssh -i ~/.ssh/id_ed25519_github -o IdentitiesOnly=yes -T git@github.com`. Connection reached GitHub, but authentication is still waiting on the public key being added to the GitHub account.
- Created `README.md` with a reminder for Pritish to run the project inside `gemini_env`.
- Prepared the repo for publishing: verified pending files, confirmed branch `main`, and confirmed `origin` points to `git@github.com:pritishrv/vidiq-hpc.git`.

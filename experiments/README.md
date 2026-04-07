# Experiments

This directory contains experiment-scoped work.

Current convention:

- `text/` contains text embedding experiments
- datasets are grouped by task type first, then by dataset name
- each dataset experiment owns its own reports, prompts, source code, configs, data staging, artifacts, and runs

This structure is intended to keep:

- experiment context isolated
- outputs reproducible
- model and validation variants easy to compare

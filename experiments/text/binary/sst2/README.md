# SST-2 Binary Experiment

This experiment root owns all SST-2 binary sentiment work.

Subdirectories:

- `reports/` experiment reports and conclusions
- `prompts/` reusable prompt context
- `src/` experiment-specific code
- `configs/` run and pipeline configs
- `requirements.txt` Python dependencies for this experiment
- `data/` raw and processed dataset material
- `artifacts/` reusable embeddings, metrics, plots, and logs
- `runs/` concrete run folders for model / pooling / validation variants

Current entry points:

- `src/run_model_selection.py` for fixed-recipe model selection
- `src/run_bge_ablation.py` for selected-model embedding-formation ablations

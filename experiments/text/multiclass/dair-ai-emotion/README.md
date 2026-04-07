# dair-ai/emotion Multiclass Experiment

This experiment root owns all `dair-ai/emotion` multiclass work.

Subdirectories:

- `reports/` experiment reports and conclusions
- `prompts/` reusable prompt context
- `src/` experiment-specific code
- `configs/` run and pipeline configs
- `requirements.txt` Python dependencies for this experiment
- `data/` raw and processed dataset material
- `artifacts/` reusable embeddings, metrics, plots, and logs
- `runs/` concrete run folders for embedding / validation variants

Current entry points:

- `src/run_embedding_generation.py` for direct BGE embedding generation
- `src/run_bge_ablation.py` for multiclass embedding-variant evaluation
- `src/plot_bge_variants.py` for visualization-only projections

This experiment root owns all multiclass emotion work for `dair-ai/emotion`.

Subdirectories:

- `reports/` experiment reports and conclusions
- `prompts/` reusable prompt context
- `src/` experiment-specific code
- `configs/` run and pipeline configs
- `data/` raw and processed dataset material
- `artifacts/` reusable embeddings, metrics, plots, and logs
- `runs/` concrete run folders for model / pooling / validation variants

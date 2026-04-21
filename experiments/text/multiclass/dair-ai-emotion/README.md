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
- `src/plot_qwen_finetune_run.py` for dataset-level plotting of the integrated Qwen fine-tuned run
- `src/analyze_qwen_logit_geometry.py` for Phase 3 logit-geometry analysis of the integrated Qwen fine-tuned run
- `src/run_qwen_bge_parity.py` for Qwen embedding-variant parity runs derived from the tracked fine-tuned holdout exports
- `src/plot_qwen_bge_parity.py` for visualization-only projections of the Qwen parity variants

Qwen fine-tuned stage:

- dataset-level bridge run:
  - `runs/run-201-qwen3-1-7b-finetune-10e/`
- dataset-level plot output:
  - `artifacts/plots/qwen-finetune-10e/`
- dataset-level Phase 3 metrics:
  - `artifacts/metrics/qwen-finetune-10e/`
- dataset-level Phase 3 logit-geometry plots:
  - `artifacts/plots/qwen-finetune-10e/logit-geometry/`
- dataset-level Phase 3 findings note:
  - `reports/qwen-finetune-logit-geometry-findings.md`

Qwen BGE-parity stage:

- config:
  - `configs/qwen-bge-parity.json`
- HPC batch script:
  - `hpc/run_qwen_bge_parity.slurm`
- exported heldout embedding bundle:
  - `artifacts/embeddings/dair_ai_emotion_qwen_holdout_qwen3-1-7b_*`
- parity runs:
  - `runs/run-301-qwen3-1-7b-bge-parity-raw/`
  - `runs/run-302-qwen3-1-7b-bge-parity-l2/`
  - `runs/run-303-qwen3-1-7b-bge-parity-centered-l2/`
- parity metric bundle:
  - `artifacts/metrics/qwen-bge-parity/`
- parity plot bundle:
  - `artifacts/plots/qwen-bge-parity/`

This stage is the closest practical analogue to the earlier BGE variant experiments using the currently tracked Qwen artifacts. It reuses the exported held-out Qwen embeddings, derives raw / L2 / centered+L2 variants, and computes geometry-style summaries and plots for side-by-side inspection.

Important limitation:

- this parity stage is eval-only because the tracked Qwen bridge/source artifacts do not include a train/validation embedding bundle analogous to the earlier BGE embedding-generation stage
- therefore the parity runs do not include BGE-style logistic-regression or kNN probe results

Qwen parity limitations versus the original BGE experiment family:

- the original BGE stage owns full train and validation embedding exports, while the current Qwen parity stage is built from held-out evaluation embeddings only
- the original BGE stage evaluates embedding variants with downstream logistic-regression and kNN probes; the current Qwen parity stage cannot reproduce those probes honestly from the tracked Qwen artifacts
- the Qwen embeddings come from the fine-tuned classifier pipeline and use the final hidden state at the last token position, so they are not directly equivalent to BGE mean-pooled sentence embeddings
- the Qwen stage uses `happiness` where the earlier BGE materials use `joy`, so any cross-stage comparison must explain that label mapping explicitly
- the Qwen parity outputs are best treated as a geometry-side comparison bundle, not a drop-in replacement for the original BGE experiment surface

## Incorporating Additional Qwen Results

More Qwen runs are expected from the queued and running Hyperion jobs, including the balanced and unbalanced `50e`, `100e`, `250e`, `500e`, and `1000e` variants.

Those new results should be incorporated using the same pattern already used for the first tracked balanced run:

1. Keep the raw training output in `experiments/text_model/runs/<run_id>/`.
2. Add a lightweight dataset-level bridge run under `runs/` that points back to that source run.
3. Regenerate dataset-level plots with:
   - `src/plot_qwen_finetune_run.py`
4. Regenerate dataset-level logit-geometry analysis with:
   - `src/analyze_qwen_logit_geometry.py`
5. Update the dataset-level findings notes only after the artifacts and metrics are in place.

This means additional results do not require any repo restructuring. Each new run gets:

- one source run under `experiments/text_model/runs/`
- one bridge run under `runs/`
- one dataset-level plot bundle under `artifacts/plots/`
- one dataset-level metrics bundle under `artifacts/metrics/`
- updated report text only if the run materially changes the narrative

For a fuller description of the implemented Phase 3 flow, see:

- `reports/qwen-finetune-phase3-implementation-report-codex.md`

# FER2013 CLIP Signal Retention Report

Date: 2026-09-01

## Scope

This report summarizes the first image-side directional erasure run for FER2013 CLIP embeddings.

Run configuration:

- erasure steps completed: `3`
- evaluation mode: `stratified_holdout_from_train`
- seed: `42`
- holdout size: `20%`
- train examples: `15220`
- eval examples: `3806`

Source files:

- `retention_metrics.json`
- `rdm_checkpoints.json`
- `cliff_zoom_top25.png`
- `full_erasure.png`

## Accuracy Retention

Chance level:

- `0.1667`

Observed accuracies:

- step `0`: `0.6905`
- step `1`: `0.6902`
- step `2`: `0.6934`
- step `3`: `0.6936`

Removed singular weights:

- step `1`: `10.0603`
- step `2`: `10.2138`
- step `3`: `9.8741`

## Interpretation

- The baseline linear probe on the held-out split is moderate at about `69%`, well above chance for six classes.
- Removing the first three dominant classifier directions does not materially reduce accuracy.
- This suggests that, at least in the first-pass image run, the class signal is not concentrated in a tiny number of top directions.

## Caveat

- This is a short first-pass run, not a full 50- or 100-step erasure curve.
- The experiment used a stratified holdout from `train` because `test_raw.npy` is not currently present in tracked artifacts.

## Outputs

- `retention_metrics.json`
- `rdm_checkpoints.json`
- `cliff_zoom_top25.png`
- `full_erasure.png`


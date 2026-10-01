# FER2013 CLIP RSA Report

Date: 2026-09-01

## Scope

This report summarizes representational similarity analysis across centroid-distance checkpoints produced by the image erasure run.

Source files:

- `../phase3/rdm_checkpoints.json`
- `rsa_results.json`
- `rsa_correlation_matrix.png`

Metric used:

- `euclidean_norm_matrix`

Checkpoints compared:

- `0`
- `1`
- `2`
- `3`

## RSA Matrix Summary

Key cross-checkpoint Spearman correlations:

- checkpoint `0` vs `1`: `0.9893`
- checkpoint `0` vs `2`: `0.9821`
- checkpoint `0` vs `3`: `0.9821`
- checkpoint `1` vs `2`: `0.9964`
- checkpoint `1` vs `3`: `0.9964`
- checkpoint `2` vs `3`: `1.0000`

## Interpretation

- The image centroid geometry is extremely stable across the first three erasure steps.
- The class-to-class relational template barely changes even when the top linear directions are removed.
- Combined with the phase-3 retention result, this suggests the early image emotion geometry is distributed rather than fragile.

## Caveat

- These RSA results inherit the same short-horizon limitation as the phase-3 run.
- A longer erasure schedule is still needed before making a stronger low-rank or high-rank claim.

## Outputs

- `rsa_results.json`
- `rsa_correlation_matrix.png`


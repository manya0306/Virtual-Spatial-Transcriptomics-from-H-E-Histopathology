# Extended benchmark: more data and stronger models

Educational research prototype. It does **not** reproduce OmiCLIP or STPath, and nothing here has clinical validity.
The original pipeline and `outputs/experiments/` are untouched; everything new is written to
`outputs/extended_benchmark/` and `outputs/figures/extended/`.

## What the synthetic dataset is (from inspection)

| Item | Finding |
|---|---|
| Files | `images/` (1,000 PNG, 256x256 RGB), `metadata.csv`, `expression_56_genes.csv`, `README.txt` |
| Metadata | patch_id, x, y, tissue_type (epithelial/mixed/stromal/immune), sample_id (20 x 50 patches), split |
| Expression | 56 columns named `Gene_01..Gene_56`, dense, range 1.06-3.07, no zeros. Its README says values are **simulated** from morphology latent factors |
| Shortcut | tissue-type label alone gives grouped-CV R² 0.46; 8 colour statistics give 0.41 (56/56 genes > 0) |

**Cannot be used:** pooled with the real training set (Gene_01 is not CASP14; there is no gene correspondence, so any
merged label would be fabricated); as evidence that real-data performance improved; for any biological claim.

**Can be used:** (1) software and scalability test of the identical pipeline, (2) a sanity check that models recover a
known, planted image-to-expression relationship, (3) runtime scaling. Always reported separately and stamped SIMULATED.

## Experiment design (replaces "pool synthetic into training")

| Track | Data | Models | Question |
|---|---|---|---|
| A. Original | real, 19 samples | 6 original models | baseline (already in `RESULTS.md`, re-run under identical folds) |
| B. Data scaling | real, training restricted to 25/50/75/100% of training samples | pca_ridge, mlp_fusion, strong_mlp, transformer_fusion, extra_trees | does more real data help? |
| C. Stronger models | real, 19 samples | + strong_mlp, transformer_fusion, extra_trees | does capacity help at fixed data? |
| S. Simulated | synthetic, 20 samples | all 9 | pipeline check only |

Same folds for every model: `GroupKFold(5)` by sample, with an assertion that no sample appears in train and test.
Metrics: MAE, RMSE, mean R², Pearson, within-sample Pearson, genes with R² > 0 (count and %), per-gene table.

## Commands (project root)

```
pip install -r requirements.txt
# 0. original pipeline, if outputs/ is missing
python scripts/extract_resnet_features.py && python scripts/build_spatial_features.py && python scripts/build_gene_targets.py
# 1. synthetic audit (no weights needed)
python scripts/audit_synthetic_dataset.py --data-dir data/synthetic_spatial_transcriptomics_1000
# 2. real: models + data scaling, then figures
python scripts/run_extended_benchmark.py --dataset real --experiment both
python scripts/make_extended_figures.py --tag real
# 3. simulated pipeline check (needs internet once for ResNet18 weights)
python scripts/extract_synthetic_features.py --data-dir data/synthetic_spatial_transcriptomics_1000
python scripts/run_extended_benchmark.py --dataset synthetic --experiment both
python scripts/make_extended_figures.py --tag synthetic
```

Laptop timing (1 CPU, measured on the 1,000-spot set): the whole main benchmark took about 2.5 minutes, transformer being
the slowest model. On the 13.7k-spot real set expect roughly 10-15x longer; add `--models ...` to run a subset.

## Files
Added: `scripts/extended_models.py`, `run_extended_benchmark.py`, `make_extended_figures.py`,
`audit_synthetic_dataset.py`, `extract_synthetic_features.py`, this file. Changed: `README.md` (pointer only).

## Outputs saved per run
`metrics_aggregate.csv`, `metrics_per_gene.csv`, `predictions/*.csv`, `fold_assignments.csv`, `run_config.json`
(hyper-parameters, seed), `training_history.json`, `learning_curve.csv`, figures 01-09.

## Reporting rule
If the stronger models do not beat `mlp_fusion` on real data, report that. Plausible reasons: only 19 samples, frozen
ImageNet features carry limited histology signal, and the target is noisy Visium counts. Do not tune on the test folds.

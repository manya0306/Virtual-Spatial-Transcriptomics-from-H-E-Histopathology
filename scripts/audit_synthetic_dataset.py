"""Audit the synthetic dataset and state what it can / cannot be used for.

Writes outputs/synthetic_audit/audit.json and audit.md. Uses only files in the
dataset folder; no ImageNet weights needed.

Usage: python scripts/audit_synthetic_dataset.py --data-dir data/synthetic_spatial_transcriptomics_1000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
REAL_GENES_JSON = ROOT / "outputs" / "gene_targets" / "target_config.json"


def cv_r2(X, Y, groups):
    P = np.zeros_like(Y)
    for tr, te in GroupKFold(5).split(X, Y, groups):
        sc = StandardScaler().fit(X[tr])
        P[te] = Ridge(alpha=1.0).fit(sc.transform(X[tr]), Y[tr]).predict(sc.transform(X[te]))
    return 1 - ((Y - P) ** 2).sum(0) / ((Y - Y.mean(0)) ** 2).sum(0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    d = Path(ap.parse_args().data_dir)
    meta = pd.read_csv(d / "metadata.csv")
    expr = pd.read_csv(d / "expression_56_genes.csv")
    genes = [c for c in expr.columns if c != "patch_id"]
    Y = expr[genes].to_numpy()
    sizes = {Image.open(d / "images" / f"{p}.png").size for p in meta.patch_id[:100]}

    real_genes = None
    overlap = None
    if REAL_GENES_JSON.exists():
        real_genes = json.load(open(REAL_GENES_JSON))["genes"]
        overlap = sorted(set(real_genes) & set(genes))

    tt = meta.tissue_type.values
    onehot = pd.get_dummies(meta.tissue_type).to_numpy(float)
    feats = []
    for p in meta.patch_id:
        a = np.asarray(Image.open(d / "images" / f"{p}.png").convert("RGB"), dtype=np.float32) / 255
        feats.append([*a.mean((0, 1)), *a.std((0, 1)), (a.mean(2) < 0.45).mean()])
    g = meta.sample_id.values
    r_tissue = cv_r2(onehot, Y, g)
    r_color = cv_r2(np.array(feats), Y, g)

    res = {
        "n_patches": int(len(meta)), "image_sizes": [list(s) for s in sizes],
        "n_samples": int(meta.sample_id.nunique()), "gene_columns": genes[:3] + ["..."] + genes[-1:],
        "n_genes": len(genes), "gene_names_are_generic_placeholders": all(x.startswith("Gene_") for x in genes),
        "overlap_with_real_gene_names": overlap,
        "expression_min_max": [float(Y.min()), float(Y.max())], "fraction_zero": float((Y == 0).mean()),
        "tissue_type_counts": meta.tissue_type.value_counts().to_dict(),
        "grouped_cv_R2_tissue_type_onehot_only": float(r_tissue.mean()),
        "grouped_cv_R2_8_colour_stats_only": float(r_color.mean()),
        "genes_positive_R2_from_colour_stats": int((r_color > 0).sum()),
    }
    out = ROOT / "outputs" / "synthetic_audit"
    out.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out / "audit.json", "w"), indent=2)
    md = f"""# Synthetic dataset audit

* {res['n_patches']} PNG patches, {res['image_sizes']}, {res['n_samples']} simulated samples.
* Expression columns: {res['n_genes']} generic names ({', '.join(res['gene_columns'])}).
  Overlap with the real gene panel: {overlap if overlap is not None else 'real target_config.json not found'}.
* Values: min/max {res['expression_min_max'][0]:.2f}/{res['expression_min_max'][1]:.2f}, zero fraction {res['fraction_zero']:.3f}
  (real Visium counts are sparse; these are dense, narrow-range simulated values).
* Its own README states expression is simulated from morphology-associated latent factors.
* Shortcut check (sample-grouped ridge): tissue-type label alone R2 = {res['grouped_cv_R2_tissue_type_onehot_only']:.3f};
  8 trivial colour statistics R2 = {res['grouped_cv_R2_8_colour_stats_only']:.3f}
  ({res['genes_positive_R2_from_colour_stats']}/56 genes > 0). The images encode the label generator, so
  high scores here would measure the simulator, not tissue biology.

## Valid uses
* Software/scale test of the pipeline (loading, feature extraction, grouped CV, models, figures).
* Sanity check that models recover a *known* image->expression relationship.
* Runtime scaling.

## Invalid uses
* Pooling with the real 19-sample training set: Gene_01..Gene_56 are not CASP14, KRT1, ...; there is no
  gene correspondence, so pooled targets would be fabricated labels.
* Claiming that real-data performance improved because synthetic rows were added.
* Any biological or clinical claim.
"""
    (out / "audit.md").write_text(md)
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

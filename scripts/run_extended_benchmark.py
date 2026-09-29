"""Extended benchmark: original models -> stronger models, plus data scaling.

Does NOT modify or overwrite anything in outputs/experiments/ (the original
benchmark). Everything is written to outputs/extended_benchmark/<dataset>/.

Experiments
-----------
main            all models under identical sample-grouped CV
                (6 original models + strong_mlp + transformer_fusion + extra_trees)
learning_curve  same folds, training set restricted to a fraction of the
                training SAMPLES (whole samples only); test folds unchanged.
                This is the valid way to ask "does more data help?".

Datasets
--------
real       the 19-sample real benchmark (outputs/image_features, gene_targets)
synthetic  the simulated 1,000-patch set (Gene_01..Gene_56 are NOT the real
           genes). Reported separately, never pooled with real data.

Usage (project root):
    python scripts/run_extended_benchmark.py --dataset real
    python scripts/run_extended_benchmark.py --dataset real --experiment learning_curve
    python scripts/run_extended_benchmark.py --dataset synthetic
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_experiments as base  # noqa: E402  (original pipeline, reused unchanged)
import extended_models as ext  # noqa: E402

ROOT = base.PROJECT_ROOT
SYN_DIR = ROOT / "outputs" / "synthetic_features"
OUT_ROOT = ROOT / "outputs" / "extended_benchmark"

ORIGINAL_MODELS = ["mean_baseline", "ridge_image", "ridge_image_neighbor", "pca_ridge_image",
                   "pca_ridge_image_neighbor_coords", "mlp_fusion"]
NEW_MODELS = ["strong_mlp", "transformer_fusion", "extra_trees"]
ALL_MODELS = ORIGINAL_MODELS + NEW_MODELS
CURVE_MODELS = ["pca_ridge_image_neighbor_coords", "mlp_fusion", "strong_mlp",
                "transformer_fusion", "extra_trees"]
FRACTIONS = [0.25, 0.5, 0.75, 1.0]


# ------------------------------- data loading -----------------------------

def load_real():
    return base.load_dataset()


def load_synthetic():
    need = ["resnet18_features.npy", "neighbor_features.npy", "metadata.csv", "targets.csv"]
    for f in need:
        if not (SYN_DIR / f).exists():
            raise FileNotFoundError(f"Missing {SYN_DIR / f}. Run scripts/extract_synthetic_features.py first.")
    img = np.load(SYN_DIR / "resnet18_features.npy").astype(np.float32)
    nb = np.load(SYN_DIR / "neighbor_features.npy").astype(np.float32)
    meta = pd.read_csv(SYN_DIR / "metadata.csv")
    tg = pd.read_csv(SYN_DIR / "targets.csv")
    genes = [c for c in tg.columns if c != "patch_id"]
    assert (tg["patch_id"].values == meta["barcode"].values).all()
    coords = base.normalise_coordinates_within_sample(meta)
    return img, nb, coords, tg[genes].to_numpy(np.float32), meta, genes


# --------------------------------- models ---------------------------------

def make_runners(image, neighbor, coords, targets, groups, seed, n_seeds) -> Dict[str, Callable]:
    """name -> f(train_idx, test_idx) -> (predictions, info)."""
    inn = np.hstack([image, neighbor])
    inc = np.hstack([image, neighbor, coords])
    A, K = base.RIDGE_ALPHAS, base.N_PCA_COMPONENTS

    def r_mean(tr, te):
        return base.fit_predict_mean(targets[tr], len(te)), {}

    def r_ridge(x):
        def f(tr, te):
            p, a = base.fit_predict_ridge(x[tr], targets[tr], x[te], A, seed)
            return p, {"alpha": a}
        return f

    def r_pca(x):
        def f(tr, te):
            p, a = base.fit_predict_pca_ridge(x[tr], targets[tr], x[te], K, A, seed)
            return p, {"alpha": a}
        return f

    def r_mlp(tr, te):
        return base.fit_predict_mlp(image[tr], neighbor[tr], coords[tr], targets[tr], groups[tr],
                                    image[te], neighbor[te], coords[te], seed)

    def r_strong(tr, te):
        return ext.fit_predict_strong_mlp(image[tr], neighbor[tr], coords[tr], targets[tr], groups[tr],
                                          image[te], neighbor[te], coords[te], seed, n_seeds=n_seeds)

    def r_tf(tr, te):
        return ext.fit_predict_transformer(image[tr], neighbor[tr], coords[tr], targets[tr], groups[tr],
                                           image[te], neighbor[te], coords[te], seed)

    def r_trees(tr, te):
        return ext.fit_predict_extra_trees(image[tr], neighbor[tr], coords[tr], targets[tr],
                                           image[te], neighbor[te], coords[te], seed), {}

    return {"mean_baseline": r_mean, "ridge_image": r_ridge(image), "ridge_image_neighbor": r_ridge(inn),
            "pca_ridge_image": r_pca(image), "pca_ridge_image_neighbor_coords": r_pca(inc),
            "mlp_fusion": r_mlp, "strong_mlp": r_strong, "transformer_fusion": r_tf,
            "extra_trees": r_trees}


# -------------------------------- evaluation ------------------------------

def evaluate(name: str, pred: np.ndarray, targets, groups, genes, tag: Dict) -> (Dict, List[Dict]):
    rows = []
    for j, g in enumerate(genes):
        m = base.gene_metrics(targets[:, j], pred[:, j])
        m["within_sample_pearson"] = base.within_sample_pearson(targets[:, j], pred[:, j], groups)
        m.update({"model": name, "gene": g, **tag})
        rows.append(m)
    d = pd.DataFrame(rows)
    agg = {"model": name, **tag, "MAE": d.MAE.mean(), "RMSE": d.RMSE.mean(), "R2_mean": d.R2.mean(),
           "genes_with_positive_R2": int((d.R2 > 0).sum()), "n_genes": len(genes),
           "pct_genes_positive_R2": 100 * float((d.R2 > 0).mean()),
           "pearson_mean": d.pearson.mean(), "within_sample_pearson_mean": d.within_sample_pearson.mean()}
    return agg, rows


def make_splitter(cv, folds, groups):
    if cv == "loso":
        return LeaveOneGroupOut(), len(np.unique(groups))
    n = min(folds, len(np.unique(groups)))
    return GroupKFold(n_splits=n), n


def subsample_samples(train_idx, groups, frac, seed):
    """Keep a fraction of whole training SAMPLES (never partial samples)."""
    if frac >= 1.0:
        return train_idx
    u = np.unique(groups[train_idx])
    rng = np.random.default_rng(seed)
    k = max(2, int(round(frac * len(u))))
    keep = rng.choice(u, size=min(k, len(u)), replace=False)
    return train_idx[np.isin(groups[train_idx], keep)]


# ----------------------------------- main ---------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["real", "synthetic"], default="real")
    ap.add_argument("--experiment", choices=["main", "learning_curve", "both"], default="main")
    ap.add_argument("--models", nargs="*", default=None, help="subset of models for the main run")
    ap.add_argument("--cv", choices=["groupkfold", "loso"], default="groupkfold")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=base.RANDOM_SEED)
    ap.add_argument("--ensemble-seeds", type=int, default=3, help="seeds averaged by strong_mlp")
    ap.add_argument("--fractions", type=float, nargs="*", default=FRACTIONS)
    ap.add_argument("--tag", default=None, help="output sub-folder name (default: dataset name)")
    args = ap.parse_args()

    t0 = time.time()
    try:
        image, neighbor, coords, targets, meta, genes = (load_real() if args.dataset == "real" else load_synthetic())
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        return 1
    groups = meta["sample"].to_numpy().astype(str)
    out = OUT_ROOT / (args.tag or args.dataset)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    banner = "REAL DATA" if args.dataset == "real" else "SIMULATED DATA - pipeline check only, not biology"
    print(f"\n[{banner}] spots={len(targets)} genes={len(genes)} samples={len(np.unique(groups))}")

    splitter, n_folds = make_splitter(args.cv, args.folds, groups)
    folds = list(splitter.split(image, targets, groups))
    runners = make_runners(image, neighbor, coords, targets, groups, args.seed, args.ensemble_seeds)

    assign = np.full(len(targets), -1, int)
    for k, (_, te) in enumerate(folds, 1):
        assign[te] = k
    spots = meta[["sample", "barcode", "x_coordinate", "y_coordinate"]].copy()
    spots["fold"] = assign
    spots.to_csv(out / "fold_assignments.csv", index=False)
    # Hard leakage guard: no sample may appear in both train and test of a fold.
    for tr, te in folds:
        assert not set(groups[tr]) & set(groups[te]), "sample leaked across train/test"

    agg_rows, gene_rows, histories = [], [], []

    if args.experiment in ("main", "both"):
        names = args.models or ALL_MODELS
        preds = {n: np.full_like(targets, np.nan) for n in names}
        for k, (tr, te) in enumerate(folds, 1):
            print(f"\nFold {k}/{n_folds} train={len(tr)} test={len(te)} held-out: {sorted(set(groups[te]))}")
            for n in names:
                ts = time.time()
                p, info = runners[n](tr, te)
                preds[n][te] = p
                if "history" in info:
                    histories.append({"model": n, "fold": k, **info})
                print(f"   {n:34s} done ({time.time() - ts:5.1f}s)")
        truth = spots.copy()
        for j, g in enumerate(genes):
            truth[g] = targets[:, j]
        truth.to_csv(out / "predictions" / "ground_truth.csv", index=False)
        for n in names:
            a, r = evaluate(n, preds[n], targets, groups, genes, {"experiment": "main"})
            agg_rows.append(a)
            gene_rows += r
            f = spots.copy()
            for j, g in enumerate(genes):
                f[g] = preds[n][:, j]
            f.to_csv(out / "predictions" / f"{n}.csv", index=False)
        pd.DataFrame(agg_rows).sort_values("within_sample_pearson_mean", ascending=False).to_csv(
            out / "metrics_aggregate.csv", index=False)
        pd.DataFrame(gene_rows).to_csv(out / "metrics_per_gene.csv", index=False)
        with open(out / "training_history.json", "w") as h:
            json.dump(histories, h)

    if args.experiment in ("learning_curve", "both"):
        lc_rows = []
        names = [m for m in CURVE_MODELS if args.models is None or m in args.models]
        for frac in args.fractions:
            preds = {n: np.full_like(targets, np.nan) for n in names}
            n_train, n_train_samples = [], []
            for k, (tr, te) in enumerate(folds, 1):
                sub = subsample_samples(tr, groups, frac, args.seed + k)
                n_train.append(len(sub))
                n_train_samples.append(len(np.unique(groups[sub])))
                for n in names:
                    p, _ = runners[n](sub, te)
                    preds[n][te] = p
                print(f"  frac={frac:.2f} fold {k}/{n_folds} train_spots={len(sub)}")
            for n in names:
                a, _ = evaluate(n, preds[n], targets, groups, genes,
                                {"experiment": "learning_curve", "train_fraction": frac,
                                 "mean_train_spots": float(np.mean(n_train)),
                                 "mean_train_samples": float(np.mean(n_train_samples))})
                lc_rows.append(a)
        pd.DataFrame(lc_rows).to_csv(out / "learning_curve.csv", index=False)

    cfg = {"dataset": args.dataset, "banner": banner, "cv": args.cv, "n_folds": n_folds, "seed": args.seed,
           "n_spots": int(len(targets)), "n_samples": int(len(np.unique(groups))), "genes": genes,
           "fractions": args.fractions, "ensemble_seeds": args.ensemble_seeds,
           "hyperparameters": {k: v for k, v in vars(ext).items() if k.isupper() and isinstance(v, (int, float))},
           "baseline_hyperparameters": {"pca": base.N_PCA_COMPONENTS, "alphas": base.RIDGE_ALPHAS,
                                        "mlp_hidden": base.MLP_HIDDEN, "mlp_epochs": base.MLP_EPOCHS},
           "python": platform.python_version(), "platform": platform.platform(),
           "runtime_seconds": round(time.time() - t0, 1)}
    with open(out / "run_config.json", "w") as h:
        json.dump(cfg, h, indent=2)

    if agg_rows:
        print("\n" + pd.DataFrame(agg_rows).sort_values("within_sample_pearson_mean", ascending=False)[
            ["model", "MAE", "RMSE", "R2_mean", "genes_with_positive_R2", "pearson_mean",
             "within_sample_pearson_mean"]].round(4).to_string(index=False))
    print(f"\nSaved to {out.relative_to(ROOT)}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Figures for the extended benchmark (reads outputs/extended_benchmark/<tag>/).

Usage: python scripts/make_extended_figures.py --tag real
       python scripts/make_extended_figures.py --tag synthetic      (labelled SIMULATED)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["mean_baseline", "ridge_image", "ridge_image_neighbor", "pca_ridge_image",
         "pca_ridge_image_neighbor_coords", "mlp_fusion", "strong_mlp", "transformer_fusion", "extra_trees"]
NEW = {"strong_mlp", "transformer_fusion", "extra_trees"}
ABLATION = [("ridge_image", "Image only\n(ridge)"), ("ridge_image_neighbor", "Image +\nspatial (ridge)"),
            ("pca_ridge_image_neighbor_coords", "Image + spatial\n+ coords (PCA-ridge)"),
            ("mlp_fusion", "Multimodal MLP"), ("strong_mlp", "Stronger MLP"),
            ("transformer_fusion", "Transformer\nfusion")]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130})


def col(m):
    return "#d1495b" if m in NEW else ("#8d99ae" if m == "mean_baseline" else "#2e6f9e")


def bar(ax, agg, metric, title, lower_better=False):
    d = agg.set_index("model").reindex([m for m in ORDER if m in set(agg.model)])
    ax.barh(range(len(d)), d[metric], color=[col(m) for m in d.index])
    ax.set_yticks(range(len(d)), d.index)
    ax.invert_yaxis()
    ax.set_title(title + ("  (lower = better)" if lower_better else "  (higher = better)"))
    for i, v in enumerate(d[metric]):
        if np.isfinite(v):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="real")
    ap.add_argument("--n-maps-genes", type=int, default=3)
    a = ap.parse_args()
    src = ROOT / "outputs" / "extended_benchmark" / a.tag
    out = ROOT / "outputs" / "figures" / "extended" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.load(open(src / "run_config.json"))
    sim = cfg["dataset"] == "synthetic"
    stamp = "SIMULATED DATA - pipeline check only, not biology\n" if sim else ""
    agg = pd.read_csv(src / "metrics_aggregate.csv")
    pg = pd.read_csv(src / "metrics_per_gene.csv")
    saved = []

    def save(fig, name):
        if sim:
            fig.text(0.5, 0.995, "SIMULATED DATA - pipeline check only, not biology", ha="center", va="top",
                     color="crimson", fontsize=9, weight="bold")
        fig.savefig(out / name, bbox_inches="tight")
        plt.close(fig)
        saved.append(name)

    # 1 dataset size
    real_cfg = ROOT / "outputs" / "extended_benchmark" / "real" / "run_config.json"
    rs, rn = (json.load(open(real_cfg))["n_spots"], json.load(open(real_cfg))["n_samples"]) if real_cfg.exists() else (13742, 19)
    ss, sn = 1000, 20
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].bar(["Real\n(benchmark)", "Simulated\n(separate)"], [rs, ss], color=["#2e6f9e", "#e9c46a"])
    ax[0].set_title("Patches / spots")
    ax[1].bar(["Real\n(benchmark)", "Simulated\n(separate)"], [rn, sn], color=["#2e6f9e", "#e9c46a"])
    ax[1].set_title("Tissue samples")
    for x in ax:
        for p in x.patches:
            x.text(p.get_x() + p.get_width() / 2, p.get_height(), f"{int(p.get_height()):,}", ha="center", va="bottom")
    fig.suptitle("Dataset sizes. Simulated genes are Gene_01..56 and cannot be merged with real targets", y=1.03, fontsize=10)
    save(fig, "01_dataset_size.png")

    # 2 model comparison
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    bar(ax[0], agg, "R2_mean", "Mean R²")
    bar(ax[1], agg, "pearson_mean", "Mean Pearson")
    bar(ax[2], agg, "within_sample_pearson_mean", "Within-sample Pearson")
    for x in ax[1:]:
        x.set_yticklabels([])
    fig.suptitle(f"Model performance, sample-grouped CV ({cfg['n_folds']} folds). Red = newly added models", y=1.03)
    save(fig, "02_model_performance.png")

    # 3 R2 progression + learning curve
    lc_path = src / "learning_curve.csv"
    fig, ax = plt.subplots(1, 2 if lc_path.exists() else 1, figsize=(12 if lc_path.exists() else 6.5, 4.2), squeeze=False)
    d = agg.set_index("model").reindex([m for m in ORDER if m in set(agg.model)])
    ax[0, 0].plot(range(len(d)), d.R2_mean, "o-", color="#2e6f9e")
    for i, m in enumerate(d.index):
        ax[0, 0].scatter(i, d.R2_mean[m], color=col(m), zorder=3, s=50)
    ax[0, 0].axhline(0, color="k", lw=.6)
    ax[0, 0].set_xticks(range(len(d)), [m.replace("_", "\n") for m in d.index], fontsize=7)
    ax[0, 0].set_ylabel("Mean R² (out-of-fold)")
    ax[0, 0].set_title("R² progression: baselines → current MLP → stronger models")
    if lc_path.exists():
        lc = pd.read_csv(lc_path)
        for m, g in lc.groupby("model"):
            g = g.sort_values("train_fraction")
            ax[0, 1].plot(g.mean_train_spots, g.R2_mean, "o-", label=m, color=col(m) if m in NEW else None)
        ax[0, 1].set_xlabel("Training spots per fold (whole samples added)")
        ax[0, 1].set_ylabel("Mean R² (same test folds)")
        ax[0, 1].set_title("Effect of training-set size")
        ax[0, 1].legend(fontsize=7)
    save(fig, "03_r2_progression_and_data_scaling.png")

    # 4 MAE / RMSE
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    bar(ax[0], agg, "MAE", "MAE", True)
    bar(ax[1], agg, "RMSE", "RMSE", True)
    ax[1].set_yticklabels([])
    save(fig, "04_mae_rmse.png")

    # best model
    cand = agg[agg.model != "mean_baseline"].sort_values("within_sample_pearson_mean", ascending=False)
    best = cand.iloc[0].model
    pb = pg[pg.model == best].sort_values("within_sample_pearson", ascending=False)

    # 5 per-gene
    fig, ax = plt.subplots(figsize=(6, 9 if len(pb) > 40 else 6))
    ax.barh(range(len(pb)), pb.within_sample_pearson, color=np.where(pb.R2 > 0, "#2e6f9e", "#c9c9c9"))
    ax.set_yticks(range(len(pb)), pb.gene, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Within-sample Pearson  (grey = R² ≤ 0)")
    ax.set_title(f"Per-gene performance: {best}")
    save(fig, "05_per_gene.png")

    # 6 predicted vs measured
    truth = pd.read_csv(src / "predictions" / "ground_truth.csv")
    pred = pd.read_csv(src / "predictions" / f"{best}.csv")
    show = list(pb.gene.head(3)) + list(pb.gene.tail(3))
    fig, ax = plt.subplots(2, 3, figsize=(11, 6.5))
    for x, g in zip(ax.ravel(), show):
        x.scatter(truth[g], pred[g], s=4, alpha=.3, color="#2e6f9e")
        lo, hi = min(truth[g].min(), pred[g].min()), max(truth[g].max(), pred[g].max())
        x.plot([lo, hi], [lo, hi], "k--", lw=.7)
        r = pb.set_index("gene").loc[g]
        x.set_title(f"{g}  r={r.pearson:.2f}  R²={r.R2:.2f}", fontsize=9)
        x.set_xlabel("Measured")
        x.set_ylabel("Predicted")
    fig.suptitle(f"Predicted vs measured ({best}). Top row: strongest, bottom row: weakest within-sample genes", y=1.0)
    fig.tight_layout()
    save(fig, "06_predicted_vs_measured.png")

    # 7 spatial maps
    samples = truth["sample"].value_counts().index[:2]
    genes3 = list(pb.gene.head(a.n_maps_genes))
    fig, ax = plt.subplots(len(samples) * 2, len(genes3), figsize=(3.6 * len(genes3), 3.2 * len(samples) * 2), squeeze=False)
    for si, s in enumerate(samples):
        mt, mp = truth[truth["sample"] == s], pred[pred["sample"] == s]
        for gi, g in enumerate(genes3):
            vmin, vmax = mt[g].quantile(.02), mt[g].quantile(.98)
            for ri, (df, lab) in enumerate([(mt, "measured"), (mp, best)]):
                x = ax[si * 2 + ri, gi]
                sc = x.scatter(df.x_coordinate, -df.y_coordinate, c=df[g], s=14, cmap="magma", vmin=vmin, vmax=vmax)
                x.set_title(f"{s} · {g} · {lab}", fontsize=7)
                x.set_xticks([]); x.set_yticks([]); x.set_aspect("equal")
    fig.suptitle("Spatial maps (shared colour scale per gene/sample)", y=1.0)
    fig.tight_layout()
    save(fig, "07_spatial_maps.png")

    # 8 strong vs weak
    k = min(8, len(pb) // 2)
    sw = pd.concat([pb.head(k).assign(g="Strongest"), pb.tail(k).assign(g="Weakest")])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(sw)), sw.within_sample_pearson, color=["#2a9d8f"] * k + ["#e76f51"] * k)
    ax.set_xticks(range(len(sw)), sw.gene, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Within-sample Pearson")
    ax.set_title(f"Strong vs weak genes ({best})")
    save(fig, "08_strong_vs_weak_genes.png")

    # 9 ablation
    ab = [(m, l) for m, l in ABLATION if m in set(agg.model)]
    d = agg.set_index("model")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for x, met, t in [(ax[0], "R2_mean", "Mean R²"), (ax[1], "within_sample_pearson_mean", "Within-sample Pearson")]:
        x.bar(range(len(ab)), [d.loc[m, met] for m, _ in ab], color=[col(m) for m, _ in ab])
        x.set_xticks(range(len(ab)), [l for _, l in ab], fontsize=7)
        x.set_title(t)
    fig.suptitle("Ablation: input modalities and model class (same folds)", y=1.03)
    save(fig, "09_ablation.png")

    print("Saved:", *saved, sep="\n  ")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

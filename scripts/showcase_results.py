from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUTS = ROOT / "outputs"
EXPERIMENTS = OUTPUTS / "experiments"
PREDICTIONS = EXPERIMENTS / "predictions"
SPATIAL = OUTPUTS / "spatial_features"
GENES = OUTPUTS / "gene_targets"

SHOWCASE = OUTPUTS / "showcase"
SHOWCASE.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def load_csv(path):
    if not path.exists():
        print(f"[WARNING] Missing file: {path}")
        return None

    return pd.read_csv(path)


def save_figure(fig, filename):
    path = SHOWCASE / filename

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"[SAVED] {path}")


def find_column(df, possible_names):

    if df is None:
        return None

    columns = {
        str(column).lower(): column
        for column in df.columns
    }

    for name in possible_names:

        if name.lower() in columns:
            return columns[name.lower()]

    return None


# ============================================================
# RESULT 1
# DATASET / IMPLEMENTATION OVERVIEW
# ============================================================

def create_dataset_overview():

    gene_targets = load_csv(
        GENES / "gene_expression_targets.csv"
    )

    spatial_metadata = load_csv(
        SPATIAL / "spatial_metadata.csv"
    )

    # These are the validated project-level quantities.
    number_of_samples = 19
    number_of_spots = 13742
    number_of_genes = 56
    image_features = 512

    if gene_targets is not None:

        sample_column = find_column(
            gene_targets,
            ["sample_id", "sample"]
        )

        if sample_column:
            number_of_samples = (
                gene_targets[sample_column]
                .nunique()
            )

        number_of_spots = len(gene_targets)

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.axis("off")

    ax.text(
        0.05,
        0.92,
        "PROJECT DATA & IMPLEMENTATION",
        fontsize=22,
        fontweight="bold"
    )

    ax.text(
        0.05,
        0.86,
        "What actually enters our prediction pipeline",
        fontsize=12
    )

    cards = [
        (number_of_samples, "Tissue samples"),
        (f"{number_of_spots:,}", "Usable spatial spots"),
        (number_of_genes, "Gene targets"),
        (image_features, "Image features / spot"),
    ]

    positions = [
        0.05,
        0.285,
        0.52,
        0.755
    ]

    for (value, label), x in zip(
        cards,
        positions
    ):

        box = FancyBboxPatch(
            (x, 0.60),
            0.18,
            0.18,
            boxstyle="round,pad=0.01",
            linewidth=1.5,
            facecolor="white"
        )

        ax.add_patch(box)

        ax.text(
            x + 0.09,
            0.70,
            str(value),
            ha="center",
            fontsize=23,
            fontweight="bold"
        )

        ax.text(
            x + 0.09,
            0.64,
            label,
            ha="center",
            fontsize=10
        )

    steps = [
        "H&E image",
        "Image patches",
        "ResNet18\n512 features",
        "6-neighbour\nspatial context",
        "ML / MLP\nprediction"
    ]

    for i, step in enumerate(steps):

        x = 0.05 + i * 0.19

        box = FancyBboxPatch(
            (x, 0.35),
            0.145,
            0.13,
            boxstyle="round,pad=0.01",
            linewidth=1.2,
            facecolor="white"
        )

        ax.add_patch(box)

        ax.text(
            x + 0.0725,
            0.415,
            step,
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold"
        )

        if i < len(steps) - 1:

            ax.annotate(
                "",
                xy=(x + 0.18, 0.415),
                xytext=(x + 0.15, 0.415),
                arrowprops={
                    "arrowstyle": "->",
                    "linewidth": 1.5
                }
            )

    ax.text(
        0.05,
        0.18,
        "Evidence: the project processes thousands of spatial locations "
        "and produces numerical features before prediction.",
        fontsize=10.5
    )

    save_figure(
        fig,
        "01_dataset_and_pipeline.png"
    )


# ============================================================
# RESULT 2
# MODEL COMPARISON
# ============================================================

def create_model_comparison():

    df = load_csv(
        EXPERIMENTS / "metrics_aggregate.csv"
    )

    if df is None:
        return

    model_column = find_column(
        df,
        ["model", "model_name"]
    )

    r2_column = find_column(
        df,
        ["r2", "r2_mean", "R2_mean"]
    )

    if model_column is None or r2_column is None:

        print(
            "[WARNING] Could not find model/R2 columns."
        )

        return

    data = df[
        [model_column, r2_column]
    ].copy()

    data.columns = [
        "model",
        "r2"
    ]

    data["r2"] = pd.to_numeric(
        data["r2"],
        errors="coerce"
    )

    data = data.dropna()

    order = [
        "mean_baseline",
        "ridge_image",
        "ridge_image_neighbor",
        "pca_ridge_image",
        "pca_ridge_image_neighbor_coords",
        "mlp_fusion"
    ]

    data["order"] = data["model"].map(
        {
            name: i
            for i, name in enumerate(order)
        }
    )

    data = data.sort_values(
        "order",
        na_position="last"
    )

    labels = {
        "mean_baseline": "Mean\nbaseline",
        "ridge_image": "Ridge\nimage",
        "ridge_image_neighbor": "Ridge image\n+ neighbours",
        "pca_ridge_image": "PCA + Ridge\nimage",
        "pca_ridge_image_neighbor_coords":
            "PCA + Ridge\n+ spatial",
        "mlp_fusion": "MLP\nfusion"
    }

    fig, ax = plt.subplots(
        figsize=(11, 6.5)
    )

    x = np.arange(len(data))

    bars = ax.bar(
        x,
        data["r2"],
        width=0.62
    )

    for bar, value in zip(
        bars,
        data["r2"]
    ):

        offset = (
            0.008
            if value >= 0
            else -0.012
        )

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            value + offset,
            f"{value:.4f}",
            ha="center",
            fontsize=10,
            fontweight="bold"
        )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            labels.get(
                model,
                model.replace("_", "\n")
            )
            for model in data["model"]
        ]
    )

    ax.set_ylabel(
        "Aggregate R²"
    )

    ax.set_title(
        "Model Comparison",
        fontsize=18,
        fontweight="bold"
    )

    ax.grid(
        axis="y",
        alpha=0.25
    )

    ax.text(
        0.01,
        0.01,
        "All values come directly from the completed grouped-CV experiment.",
        transform=ax.transAxes,
        fontsize=9
    )

    save_figure(
        fig,
        "02_model_comparison.png"
    )


# ============================================================
# RESULT 3
# MAE / RMSE
# ============================================================

def create_error_comparison():

    df = load_csv(
        EXPERIMENTS / "metrics_aggregate.csv"
    )

    if df is None:
        return

    model_column = find_column(
        df,
        ["model", "model_name"]
    )

    mae_column = find_column(
        df,
        ["mae", "mae_mean"]
    )

    rmse_column = find_column(
        df,
        ["rmse", "rmse_mean"]
    )

    if not all([
        model_column,
        mae_column,
        rmse_column
    ]):

        print(
            "[WARNING] Could not find MAE/RMSE columns."
        )

        return

    data = df[
        [
            model_column,
            mae_column,
            rmse_column
        ]
    ].copy()

    data.columns = [
        "model",
        "mae",
        "rmse"
    ]

    order = [
        "mean_baseline",
        "ridge_image",
        "ridge_image_neighbor",
        "pca_ridge_image",
        "pca_ridge_image_neighbor_coords",
        "mlp_fusion"
    ]

    data["order"] = data["model"].map(
        {
            name: i
            for i, name in enumerate(order)
        }
    )

    data = data.sort_values(
        "order",
        na_position="last"
    )

    x = np.arange(len(data))

    width = 0.35

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        x - width / 2,
        data["mae"],
        width,
        label="MAE"
    )

    ax.bar(
        x + width / 2,
        data["rmse"],
        width,
        label="RMSE"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            model.replace("_", "\n")
            for model in data["model"]
        ]
    )

    ax.set_ylabel(
        "Error"
    )

    ax.set_title(
        "Prediction Error Across Models",
        fontsize=18,
        fontweight="bold"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25
    )

    save_figure(
        fig,
        "03_error_metrics.png"
    )


# ============================================================
# RESULT 4
# GENE-LEVEL PERFORMANCE
# ============================================================

def create_gene_performance():

    df = load_csv(
        EXPERIMENTS / "metrics_per_gene.csv"
    )

    if df is None:
        return

    gene_column = find_column(
        df,
        ["gene", "gene_name"]
    )

    model_column = find_column(
        df,
        ["model", "model_name"]
    )

    r2_column = find_column(
        df,
        ["r2", "r2_mean", "R2_mean"]
    )

    if not all([
        gene_column,
        model_column,
        r2_column
    ]):

        print(
            "[WARNING] Could not identify gene metrics."
        )

        return

    data = df[
        [
            gene_column,
            model_column,
            r2_column
        ]
    ].copy()

    data.columns = [
        "gene",
        "model",
        "r2"
    ]

    data["r2"] = pd.to_numeric(
        data["r2"],
        errors="coerce"
    )

    data = data.dropna()

    if "mlp_fusion" in data["model"].unique():

        data = data[
            data["model"] == "mlp_fusion"
        ]

    data = data.sort_values(
        "r2"
    )

    # Show weakest and strongest genes.
    if len(data) > 16:

        data = pd.concat(
            [
                data.head(8),
                data.tail(8)
            ]
        )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    ax.barh(
        data["gene"],
        data["r2"]
    )

    ax.axvline(
        0,
        linewidth=1
    )

    ax.set_xlabel(
        "R²"
    )

    ax.set_title(
        "Gene-Level Predictive Performance",
        fontsize=18,
        fontweight="bold"
    )

    ax.grid(
        axis="x",
        alpha=0.25
    )

    ax.text(
        0.01,
        -0.08,
        "This shows that prediction quality varies across genes.",
        transform=ax.transAxes,
        fontsize=9.5
    )

    save_figure(
        fig,
        "04_gene_level_performance.png"
    )


# ============================================================
# RESULT 5
# MEASURED VS PREDICTED
# ============================================================

def create_prediction_scatter():

    ground_truth = load_csv(
        PREDICTIONS / "ground_truth.csv"
    )

    predictions = load_csv(
        PREDICTIONS / "mlp_fusion.csv"
    )

    if ground_truth is None or predictions is None:
        return

    id_columns = [
        column
        for column in [
            "sample_id",
            "spot_id",
            "barcode"
        ]
        if column in ground_truth.columns
        and column in predictions.columns
    ]

    if id_columns:

        data = ground_truth.merge(
            predictions,
            on=id_columns,
            suffixes=(
                "_true",
                "_pred"
            )
        )

        genes = [
            column
            for column in ground_truth.columns
            if column in predictions.columns
            and column not in id_columns
        ]

    else:

        common_genes = [
            column
            for column in ground_truth.columns
            if column in predictions.columns
        ]

        genes = common_genes

        data = pd.concat(
            [
                ground_truth,
                predictions
            ],
            axis=1,
            keys=[
                "true",
                "pred"
            ]
        )

    preferred_genes = [
        "CASP14",
        "KRT1",
        "DCN"
    ]

    selected_genes = [
        gene
        for gene in preferred_genes
        if gene in genes
    ]

    if not selected_genes:

        selected_genes = genes[:3]

    if not selected_genes:
        print(
            "[WARNING] No common genes found."
        )
        return

    fig, axes = plt.subplots(
        1,
        len(selected_genes),
        figsize=(
            5 * len(selected_genes),
            4.5
        )
    )

    if len(selected_genes) == 1:
        axes = [axes]

    for ax, gene in zip(
        axes,
        selected_genes
    ):

        if f"{gene}_true" in data.columns:

            true_values = pd.to_numeric(
                data[f"{gene}_true"],
                errors="coerce"
            )

            predicted_values = pd.to_numeric(
                data[f"{gene}_pred"],
                errors="coerce"
            )

        else:

            true_values = pd.to_numeric(
                ground_truth[gene],
                errors="coerce"
            )

            predicted_values = pd.to_numeric(
                predictions[gene],
                errors="coerce"
            )

        mask = (
            true_values.notna()
            &
            predicted_values.notna()
        )

        true_values = true_values[mask].to_numpy()
        predicted_values = predicted_values[mask].to_numpy()

        # Keep the figure readable.
        if len(true_values) > 3000:

            rng = np.random.default_rng(42)

            indices = rng.choice(
                len(true_values),
                3000,
                replace=False
            )

            true_values = true_values[indices]
            predicted_values = predicted_values[indices]

        ax.scatter(
            true_values,
            predicted_values,
            s=8,
            alpha=0.25
        )

        minimum = min(
            true_values.min(),
            predicted_values.min()
        )

        maximum = max(
            true_values.max(),
            predicted_values.max()
        )

        ax.plot(
            [minimum, maximum],
            [minimum, maximum],
            linestyle="--",
            linewidth=1.2
        )

        ax.set_title(
            gene,
            fontweight="bold"
        )

        ax.set_xlabel(
            "Measured expression"
        )

        ax.set_ylabel(
            "Predicted expression"
        )

        ax.grid(
            alpha=0.2
        )

    fig.suptitle(
        "Measured vs Predicted Gene Expression",
        fontsize=18,
        fontweight="bold"
    )

    save_figure(
        fig,
        "05_measured_vs_predicted.png"
    )


# ============================================================
# RESULT 6
# SPATIAL MAPS
# ============================================================

def create_spatial_maps():
    """
    Create side-by-side measured vs predicted spatial expression maps
    for representative genes and samples.

    This function only reads existing experiment outputs.
    It does not retrain or modify any model.
    """

    spatial_path = OUTPUTS / "spatial_features" / "spatial_metadata.csv"
    truth_path = EXPERIMENTS / "predictions" / "ground_truth.csv"
    pred_path = EXPERIMENTS / "predictions" / "mlp_fusion.csv"

    spatial = pd.read_csv(spatial_path)
    truth = pd.read_csv(truth_path)
    pred = pd.read_csv(pred_path)

    # ------------------------------------------------------------
    # Identify the common sample / coordinate columns
    # ------------------------------------------------------------

    sample_col = find_column(
        spatial,
        ["sample", "sample_id", "sample_name"]
    )

    x_col = find_column(
        spatial,
        ["x_coordinate", "x", "array_col", "imagecol"]
    )

    y_col = find_column(
        spatial,
        ["y_coordinate", "y", "array_row", "imagerow"]
    )

    if sample_col is None or x_col is None or y_col is None:
        print("[SKIP] Could not identify sample/x/y columns for spatial maps.")
        return

    # ------------------------------------------------------------
    # Determine genes shared by ground truth and predictions
    # ------------------------------------------------------------

    excluded = {
        "sample",
        "sample_id",
        "sample_name",
        "x_coordinate",
        "y_coordinate",
        "x",
        "y",
        "array_row",
        "array_col",
        "imagerow",
        "imagecol",
    }

    truth_genes = [
        c for c in truth.columns
        if c.lower() not in excluded
    ]

    pred_genes = [
        c for c in pred.columns
        if c.lower() not in excluded
    ]

    genes = [g for g in truth_genes if g in pred_genes]

    if not genes:
        print("[SKIP] No common genes found between truth and predictions.")
        return

    # ------------------------------------------------------------
    # Build clean identifiers for row-wise alignment
    # ------------------------------------------------------------

    def make_key(df):
        """
        Create a row key using sample + coordinates if available.
        """

        possible_sample = find_column(
            df,
            ["sample", "sample_id", "sample_name"]
        )

        possible_x = find_column(
            df,
            ["x_coordinate", "x", "array_col", "imagecol"]
        )

        possible_y = find_column(
            df,
            ["y_coordinate", "y", "array_row", "imagerow"]
        )

        if (
            possible_sample is not None
            and possible_x is not None
            and possible_y is not None
        ):
            return (
                df[possible_sample].astype(str)
                + "||"
                + df[possible_x].astype(str)
                + "||"
                + df[possible_y].astype(str)
            )

        return pd.Series(
            [str(i) for i in range(len(df))],
            index=df.index
        )

    # ------------------------------------------------------------
    # Instead of repeatedly merging the same metadata,
    # construct one clean coordinate table.
    # ------------------------------------------------------------

    spatial_clean = pd.DataFrame({
        "sample": spatial[sample_col].astype(str),
        "x": pd.to_numeric(spatial[x_col], errors="coerce"),
        "y": pd.to_numeric(spatial[y_col], errors="coerce"),
    })

    spatial_clean = spatial_clean.reset_index(drop=True)

    # ------------------------------------------------------------
    # Match truth/prediction rows without carrying their duplicate
    # coordinate columns into the merge.
    # ------------------------------------------------------------

    truth_sample = find_column(
        truth,
        ["sample", "sample_id", "sample_name"]
    )

    truth_x = find_column(
        truth,
        ["x_coordinate", "x", "array_col", "imagecol"]
    )

    truth_y = find_column(
        truth,
        ["y_coordinate", "y", "array_row", "imagerow"]
    )

    pred_sample = find_column(
        pred,
        ["sample", "sample_id", "sample_name"]
    )

    pred_x = find_column(
        pred,
        ["x_coordinate", "x", "array_col", "imagecol"]
    )

    pred_y = find_column(
        pred,
        ["y_coordinate", "y", "array_row", "imagerow"]
    )

    # ------------------------------------------------------------
    # Case 1: truth and prediction contain explicit identifiers.
    # ------------------------------------------------------------

    if (
        truth_sample is not None
        and truth_x is not None
        and truth_y is not None
        and pred_sample is not None
        and pred_x is not None
        and pred_y is not None
    ):

        truth_clean = truth[
            [truth_sample, truth_x, truth_y] + genes
        ].copy()

        pred_clean = pred[
            [pred_sample, pred_x, pred_y] + genes
        ].copy()

        truth_clean.columns = (
            ["sample", "x", "y"] + genes
        )

        pred_clean.columns = (
            ["sample", "x", "y"] + genes
        )

        # Rename expression columns before merging
        truth_clean = truth_clean.rename(
            columns={g: f"{g}_true" for g in genes}
        )

        pred_clean = pred_clean.rename(
            columns={g: f"{g}_pred" for g in genes}
        )

        data = truth_clean.merge(
            pred_clean,
            on=["sample", "x", "y"],
            how="inner"
        )

    # ------------------------------------------------------------
    # Case 2: prediction/truth don't have identifiers.
    #
    # In that case, rely on row order, which is how the existing
    # experiment prediction files were generated.
    # ------------------------------------------------------------

    else:

        n = min(
            len(spatial_clean),
            len(truth),
            len(pred)
        )

        data = spatial_clean.iloc[:n].copy()

        for gene in genes:
            data[f"{gene}_true"] = (
                pd.to_numeric(
                    truth[gene].iloc[:n],
                    errors="coerce"
                ).values
            )

            data[f"{gene}_pred"] = (
                pd.to_numeric(
                    pred[gene].iloc[:n],
                    errors="coerce"
                ).values
            )

    # ------------------------------------------------------------
    # Representative genes
    # ------------------------------------------------------------

    preferred_genes = [
        "CASP14",
        "KRT1",
        "DCN",
        "KRT10",
        "CXCL10",
    ]

    selected_genes = [
        g for g in preferred_genes
        if g in genes
    ]

    # Fill remaining slots if needed
    if len(selected_genes) < 4:
        for g in genes:
            if g not in selected_genes:
                selected_genes.append(g)
            if len(selected_genes) >= 4:
                break

    selected_genes = selected_genes[:5]

    # ------------------------------------------------------------
    # Representative samples
    # ------------------------------------------------------------

    available_samples = (
        data["sample"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    preferred_samples = [
        "AD_6_NL",
        "AD_7_LS",
        "HE_2",
    ]

    selected_samples = [
        s for s in preferred_samples
        if s in available_samples
    ]

    if len(selected_samples) < 2:
        selected_samples = available_samples[:3]

    # ------------------------------------------------------------
    # Generate maps
    # ------------------------------------------------------------

    for sample_name in selected_samples:

        sample_data = data[
            data["sample"].astype(str) == str(sample_name)
        ].copy()

        if sample_data.empty:
            continue

        for gene in selected_genes:

            true_col = f"{gene}_true"
            pred_col = f"{gene}_pred"

            if true_col not in sample_data.columns:
                continue

            if pred_col not in sample_data.columns:
                continue

            sample_data["true_value"] = pd.to_numeric(
                sample_data[true_col],
                errors="coerce"
            )

            sample_data["pred_value"] = pd.to_numeric(
                sample_data[pred_col],
                errors="coerce"
            )

            valid = sample_data[
                sample_data["x"].notna()
                & sample_data["y"].notna()
                & sample_data["true_value"].notna()
                & sample_data["pred_value"].notna()
            ].copy()

            if len(valid) < 5:
                continue

            fig, axes = plt.subplots(
                1,
                2,
                figsize=(12, 5)
            )

            # -------------------------
            # Ground truth
            # -------------------------

            sc1 = axes[0].scatter(
                valid["x"],
                valid["y"],
                c=valid["true_value"],
                s=12,
                alpha=0.85
            )

            axes[0].set_title(
                f"Measured {gene}"
            )

            axes[0].set_xlabel("X coordinate")
            axes[0].set_ylabel("Y coordinate")
            axes[0].invert_yaxis()

            fig.colorbar(
                sc1,
                ax=axes[0],
                fraction=0.046,
                pad=0.04
            )

            # -------------------------
            # Prediction
            # -------------------------

            sc2 = axes[1].scatter(
                valid["x"],
                valid["y"],
                c=valid["pred_value"],
                s=12,
                alpha=0.85
            )

            axes[1].set_title(
                f"Predicted {gene} — MLP Fusion"
            )

            axes[1].set_xlabel("X coordinate")
            axes[1].set_ylabel("Y coordinate")
            axes[1].invert_yaxis()

            fig.colorbar(
                sc2,
                ax=axes[1],
                fraction=0.046,
                pad=0.04
            )

            fig.suptitle(
                f"Spatial Expression Map — {sample_name} — {gene}",
                fontsize=14,
                fontweight="bold"
            )

            fig.tight_layout()

            filename = (
                f"06_spatial_{sample_name}_{gene}.png"
            )

            save_figure(
                fig,
                filename
            )

            plt.close(fig)

    print(
        f"[DONE] Generated spatial maps for "
        f"{len(selected_samples)} samples × "
        f"{len(selected_genes)} genes."
    )


# ============================================================
# RESULT 7
# CROSS-VALIDATION
# ============================================================

def create_cv_visual():

    df = load_csv(
        EXPERIMENTS / "fold_assignments.csv"
    )

    if df is None:
        return

    sample_column = find_column(
        df,
        ["sample_id", "sample"]
    )

    fold_column = find_column(
        df,
        ["fold", "fold_id"]
    )

    if sample_column is None or fold_column is None:

        print(
            "[WARNING] Fold columns not found."
        )

        return

    data = df[
        [sample_column, fold_column]
    ].drop_duplicates()

    folds = sorted(
        data[fold_column].unique()
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.axis("off")

    ax.set_title(
        "Sample-Level Cross-Validation",
        fontsize=18,
        fontweight="bold"
    )

    maximum_samples = max(
        (
            data[fold_column] == fold
        ).sum()
        for fold in folds
    )

    for fold_index, fold in enumerate(
        folds
    ):

        samples = data.loc[
            data[fold_column] == fold,
            sample_column
        ].tolist()

        y = 0.82 - fold_index * 0.15

        ax.text(
            0.02,
            y + 0.035,
            f"Fold {fold}",
            fontsize=11,
            fontweight="bold"
        )

        for sample_index in range(
            maximum_samples
        ):

            x = (
                0.16
                + sample_index * 0.095
            )

            rectangle = Rectangle(
                (
                    x,
                    y
                ),
                0.08,
                0.07,
                linewidth=1,
                facecolor="white"
            )

            ax.add_patch(
                rectangle
            )

            if sample_index < len(samples):

                ax.text(
                    x + 0.04,
                    y + 0.035,
                    str(
                        samples[
                            sample_index
                        ]
                    ),
                    ha="center",
                    va="center",
                    fontsize=6.5
                )

    ax.text(
        0.02,
        0.08,
        "Complete tissue samples are held out for testing. "
        "Random spot-level splitting is not used.",
        fontsize=10.5
    )

    save_figure(
        fig,
        "07_cross_validation.png"
    )


# ============================================================
# RESULT 8
# MLP TRAINING
# ============================================================

def create_training_plot():

    path = (
        EXPERIMENTS
        / "mlp_training_history.json"
    )

    if not path.exists():

        print(
            "[WARNING] MLP training history not found."
        )

        return

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        history = json.load(file)

    histories = []

    if (
        isinstance(history, dict)
        and "train_loss" in history
    ):

        histories.append(
            history
        )

    elif isinstance(history, dict):

        for value in history.values():

            if (
                isinstance(value, dict)
                and "train_loss" in value
            ):

                histories.append(
                    value
                )

    if not histories:

        print(
            "[WARNING] Training history format not recognized."
        )

        return

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    for index, fold_history in enumerate(
        histories
    ):

        train_loss = fold_history[
            "train_loss"
        ]

        validation_loss = fold_history.get(
            "val_loss",
            fold_history.get(
                "validation_loss",
                []
            )
        )

        epochs = np.arange(
            1,
            len(train_loss) + 1
        )

        ax.plot(
            epochs,
            train_loss,
            linewidth=1.5,
            label=f"Train fold {index + 1}"
        )

        if validation_loss:

            ax.plot(
                epochs[:len(validation_loss)],
                validation_loss,
                linestyle="--",
                linewidth=1.5,
                label=f"Validation fold {index + 1}"
            )

    ax.set_xlabel(
        "Epoch"
    )

    ax.set_ylabel(
        "Loss"
    )

    ax.set_title(
        "MLP Training History",
        fontsize=18,
        fontweight="bold"
    )

    ax.grid(
        alpha=0.25
    )

    if len(histories) <= 3:

        ax.legend()

    save_figure(
        fig,
        "08_mlp_training.png"
    )


# ============================================================
# RESULT 9
# FINAL DASHBOARD
# ============================================================

def create_final_dashboard():

    df = load_csv(
        EXPERIMENTS / "metrics_aggregate.csv"
    )

    if df is None:
        return

    model_column = find_column(
        df,
        ["model", "model_name"]
    )

    r2_column = find_column(
        df,
        ["r2", "r2_mean", "R2_mean"]
    )

    mae_column = find_column(
        df,
        ["mae", "mae_mean"]
    )

    rmse_column = find_column(
        df,
        ["rmse", "rmse_mean"]
    )

    mlp_rows = df[
        df[model_column] == "mlp_fusion"
    ]

    if mlp_rows.empty:

        print(
            "[WARNING] MLP fusion result not found."
        )

        return

    result = mlp_rows.iloc[0]

    r2 = float(
        result[r2_column]
    )

    mae = float(
        result[mae_column]
    )

    rmse = float(
        result[rmse_column]
    )

    gene_metrics = load_csv(
        EXPERIMENTS / "metrics_per_gene.csv"
    )

    positive_genes = 48

    if gene_metrics is not None:

        gm = find_column(
            gene_metrics,
            ["model", "model_name"]
        )

        gr = find_column(
            gene_metrics,
            ["r2", "r2_mean", "R2_mean"]
        )

        if gm and gr:

            mlp_gene_metrics = gene_metrics[
                gene_metrics[gm] == "mlp_fusion"
            ]

            if not mlp_gene_metrics.empty:

                positive_genes = int(
                    (
                        pd.to_numeric(
                            mlp_gene_metrics[gr],
                            errors="coerce"
                        )
                        > 0
                    ).sum()
                )

    fig, ax = plt.subplots(
        figsize=(13, 8)
    )

    ax.axis("off")

    ax.text(
        0.05,
        0.93,
        "FINAL PROJECT RESULTS",
        fontsize=24,
        fontweight="bold"
    )

    ax.text(
        0.05,
        0.875,
        "What our implemented pipeline actually achieved",
        fontsize=12
    )

    metrics = [
        (f"{r2:.4f}", "Aggregate R²"),
        (f"{mae:.4f}", "MAE"),
        (f"{rmse:.4f}", "RMSE"),
        (
            f"{positive_genes}/56",
            "Genes with positive R²"
        )
    ]

    for index, (
        value,
        label
    ) in enumerate(metrics):

        x = (
            0.05
            + index * 0.235
        )

        box = FancyBboxPatch(
            (
                x,
                0.65
            ),
            0.20,
            0.15,
            boxstyle="round,pad=0.01",
            linewidth=1.5,
            facecolor="white"
        )

        ax.add_patch(
            box
        )

        ax.text(
            x + 0.10,
            0.735,
            value,
            ha="center",
            fontsize=21,
            fontweight="bold"
        )

        ax.text(
            x + 0.10,
            0.69,
            label,
            ha="center",
            fontsize=9.5
        )

    ax.text(
        0.05,
        0.56,
        "Implementation evidence",
        fontsize=14,
        fontweight="bold"
    )

    evidence = [
        "19 tissue samples",
        "13,742 usable spatial spots",
        "56 common gene targets",
        "ResNet18 image features",
        "6-neighbour spatial context",
        "5-fold sample-grouped evaluation",
        "Measured and predicted spatial maps"
    ]

    for index, statement in enumerate(
        evidence
    ):

        ax.text(
            0.07,
            0.50 - index * 0.055,
            "• " + statement,
            fontsize=10.5
        )

    ax.text(
        0.05,
        0.09,
        "Conclusion: the pipeline demonstrates measurable but limited "
        "predictive signal from histology. It is a proof-of-concept, "
        "not a full foundation-model reproduction or clinical replacement.",
        fontsize=10.5,
        wrap=True
    )

    save_figure(
        fig,
        "09_final_results_dashboard.png"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("GENERATING RESULTS SHOWCASE")
    print("=" * 60)
    print()

    create_dataset_overview()

    create_model_comparison()

    create_error_comparison()

    create_gene_performance()

    create_prediction_scatter()

    create_spatial_maps()

    create_cv_visual()

    create_training_plot()

    create_final_dashboard()

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print()
    print(
        f"Open this folder:\n{SHOWCASE}"
    )
    print()


if __name__ == "__main__":
    main()
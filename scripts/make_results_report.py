"""
Create a visual PDF results report for the Virtual Spatial Transcriptomics project.

Uses EXISTING benchmark outputs:
    outputs/extended_benchmark/real/
    outputs/extended_benchmark/synthetic/

Uses EXISTING figures:
    01_dataset_size.png
    02_model_performance.png
    03_r2_progression_and_data_scaling.png
    04_mae_rmse.png
    05_per_gene.png
    06_predicted_vs_measured.png
    07_spatial_maps.png
    08_strong_vs_weak_genes.png
    09_ablation.png

No experiments are rerun.
"""

from pathlib import Path
import json
import math
import os

import pandas as pd

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = PROJECT_ROOT / "outputs"
REAL_DIR = OUTPUT_ROOT / "extended_benchmark" / "real"
SYNTHETIC_DIR = OUTPUT_ROOT / "extended_benchmark" / "synthetic"
REPORT_DIR = OUTPUT_ROOT / "results_report"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = REPORT_DIR / "Virtual_Spatial_Transcriptomics_Results.pdf"


# ============================================================
# FIGURE NAMES
# ============================================================

FIGURES = {
    "dataset_size": "01_dataset_size.png",
    "model_performance": "02_model_performance.png",
    "r2_scaling": "03_r2_progression_and_data_scaling.png",
    "mae_rmse": "04_mae_rmse.png",
    "per_gene": "05_per_gene.png",
    "predicted_measured": "06_predicted_vs_measured.png",
    "spatial_maps": "07_spatial_maps.png",
    "strong_weak": "08_strong_vs_weak_genes.png",
    "ablation": "09_ablation.png",
}


# ============================================================
# HELPERS
# ============================================================

def find_file(root: Path, filename: str):
    """Find a file recursively under root."""
    if not root.exists():
        return None

    direct = root / filename
    if direct.exists():
        return direct

    matches = list(root.rglob(filename))
    if matches:
        return matches[0]

    return None


def find_figure(filename: str):
    """
    Search real first, then synthetic, then entire outputs directory.
    """
    for root in [REAL_DIR, SYNTHETIC_DIR, OUTPUT_ROOT]:
        found = find_file(root, filename)
        if found is not None:
            return found

    return None


def load_metrics(dataset_dir: Path):
    path = dataset_dir / "metrics_aggregate.csv"

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(f"Warning: could not read {path}: {exc}")
        return None


def load_config(dataset_dir: Path):
    path = dataset_dir / "run_config.json"

    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def format_num(value, digits=4):
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def get_metric_row(metrics, model):
    if metrics is None:
        return None

    rows = metrics[
        metrics["model"].astype(str).str.lower()
        == model.lower()
    ]

    if len(rows) == 0:
        return None

    return rows.iloc[0]


def add_page_number(canvas, doc):
    """Footer with page number."""
    canvas.saveState()

    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)

    canvas.drawCentredString(
        A4[0] / 2,
        0.65 * cm,
        f"Virtual Spatial Transcriptomics — Results Report | Page {doc.page}"
    )

    canvas.restoreState()


def make_image(path, max_width=17 * cm, max_height=11 * cm):
    """
    Add an image while preserving aspect ratio.
    """
    from PIL import Image as PILImage

    img = PILImage.open(path)
    width_px, height_px = img.size

    if width_px == 0 or height_px == 0:
        return None

    aspect = width_px / height_px

    width = max_width
    height = width / aspect

    if height > max_height:
        height = max_height
        width = height * aspect

    return Image(
        str(path),
        width=width,
        height=height,
        hAlign="CENTER",
    )


def section_title(text, styles):
    return Paragraph(text, styles["SectionTitle"])


def subsection_title(text, styles):
    return Paragraph(text, styles["SubsectionTitle"])


# ============================================================
# STYLES
# ============================================================

styles = getSampleStyleSheet()

styles.add(
    ParagraphStyle(
        name="ReportTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=30,
        alignment=TA_CENTER,
        spaceAfter=14,
    )
)

styles.add(
    ParagraphStyle(
        name="ReportSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=17,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=18,
    )
)

styles.add(
    ParagraphStyle(
        name="SectionTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1F4E79"),
        spaceBefore=6,
        spaceAfter=10,
    )
)

styles.add(
    ParagraphStyle(
        name="SubsectionTitle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=colors.HexColor("#333333"),
        spaceBefore=8,
        spaceAfter=7,
    )
)

styles.add(
    ParagraphStyle(
        name="BodyReport",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        spaceAfter=8,
    )
)

styles.add(
    ParagraphStyle(
        name="SmallReport",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#555555"),
    )
)

styles.add(
    ParagraphStyle(
        name="Warning",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#B00020"),
        backColor=colors.HexColor("#FFF3F3"),
        borderColor=colors.HexColor("#E0A0A0"),
        borderWidth=0.5,
        borderPadding=7,
        spaceAfter=10,
    )
)


# ============================================================
# LOAD DATA
# ============================================================

real_metrics = load_metrics(REAL_DIR)
synthetic_metrics = load_metrics(SYNTHETIC_DIR)

real_config = load_config(REAL_DIR)
synthetic_config = load_config(SYNTHETIC_DIR)


# ============================================================
# DOCUMENT
# ============================================================

doc = SimpleDocTemplate(
    str(PDF_PATH),
    pagesize=A4,
    rightMargin=1.5 * cm,
    leftMargin=1.5 * cm,
    topMargin=1.5 * cm,
    bottomMargin=1.5 * cm,
    title="Virtual Spatial Transcriptomics — Results",
    author="Spatial Tissue Profiling Project",
)

story = []


# ============================================================
# COVER
# ============================================================

story.append(Spacer(1, 2.0 * cm))

story.append(
    Paragraph(
        "Virtual Spatial Transcriptomics",
        styles["ReportTitle"],
    )
)

story.append(
    Paragraph(
        "Experimental Results & Quantitative Evaluation",
        styles["ReportSubtitle"],
    )
)

story.append(Spacer(1, 0.5 * cm))

story.append(
    Paragraph(
        "<b>Objective</b><br/>"
        "Evaluate an image-based pipeline for predicting spatial gene-expression "
        "profiles from H&amp;E histopathology image features, using sample-grouped "
        "cross-validation and multiple model families.",
        styles["BodyReport"],
    )
)

story.append(
    Spacer(1, 0.5 * cm)
)

story.append(
    Paragraph(
        "<b>Evaluation strategy</b><br/>"
        "The benchmark compares image-only linear models, spatially augmented "
        "models, multimodal MLP models, stronger neural models, transformer-based "
        "fusion, and tree-based models. Evaluation is performed with held-out "
        "tissue samples rather than randomly splitting individual spots.",
        styles["BodyReport"],
    )
)

story.append(Spacer(1, 0.7 * cm))

story.append(
    Paragraph(
        "<b>Important data note:</b> The synthetic dataset is used only as a "
        "pipeline/software validation benchmark. Its 56 simulated genes do not "
        "overlap with the real gene panel and therefore are not pooled with the "
        "real biological targets.",
        styles["Warning"],
    )
)

story.append(
    Spacer(1, 1.0 * cm)
)

story.append(
    Paragraph(
        "Generated from the completed benchmark outputs",
        styles["SmallReport"],
    )
)

story.append(PageBreak())


# ============================================================
# SECTION 1 — DATASET
# ============================================================

story.append(section_title("1. Dataset and Evaluation Setup", styles))

story.append(
    Paragraph(
        "The evaluation uses two separate datasets. The real benchmark contains "
        "13,742 spatial spots across 19 tissue samples and 56 target genes. "
        "The synthetic benchmark contains 1,000 patches across 20 simulated "
        "samples and 56 simulated genes.",
        styles["BodyReport"],
    )
)

fig = find_figure(FIGURES["dataset_size"])
if fig:
    story.append(make_image(fig, max_width=17 * cm, max_height=10 * cm))
    story.append(
        Paragraph(
            "Figure 1. Dataset scale comparison between the real benchmark and "
            "the separate synthetic pipeline-validation dataset.",
            styles["SmallReport"],
        )
    )

story.append(Spacer(1, 0.3 * cm))

dataset_table = [
    ["Dataset", "Spots / Patches", "Samples", "Genes", "Purpose"],
    ["Real", "13,742", "19", "56",
     "Primary biological evaluation"],
    ["Synthetic", "1,000", "20", "56",
     "Pipeline / software validation"],
]

table = Table(
    dataset_table,
    colWidths=[2.5 * cm, 3.0 * cm, 2.5 * cm, 2.0 * cm, 6.0 * cm],
)

table.setStyle(
    TableStyle(
        [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F4F7FA")]),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]
    )
)

story.append(table)

story.append(PageBreak())


# ============================================================
# SECTION 2 — REAL DATA MAIN RESULTS
# ============================================================

story.append(section_title("2. Real-Data Model Performance", styles))

story.append(
    Paragraph(
        "The real-data benchmark uses 5-fold sample-grouped cross-validation. "
        "Each fold holds out complete tissue samples, preventing spots from "
        "the same sample appearing simultaneously in training and testing.",
        styles["BodyReport"],
    )
)

if real_metrics is not None:
    main_models = [
        "mean_baseline",
        "ridge_image",
        "ridge_image_neighbor",
        "pca_ridge_image",
        "pca_ridge_image_neighbor_coords",
        "mlp_fusion",
        "strong_mlp",
        "transformer_fusion",
        "extra_trees",
    ]

    table_data = [
        [
            "Model",
            "MAE",
            "RMSE",
            "Mean R²",
            "Positive R² genes",
            "Mean Pearson",
            "Within-sample Pearson",
        ]
    ]

    for model in main_models:
        row = get_metric_row(real_metrics, model)

        if row is None:
            continue

        table_data.append(
            [
                model,
                format_num(row.get("MAE")),
                format_num(row.get("RMSE")),
                format_num(row.get("R2_mean")),
                str(row.get("genes_with_positive_R2", "N/A")),
                format_num(row.get("pearson_mean")),
                format_num(row.get("within_sample_pearson_mean")),
            ]
        )

    results_table = Table(
        table_data,
        repeatRows=1,
        colWidths=[
            4.0 * cm,
            2.0 * cm,
            2.0 * cm,
            2.2 * cm,
            2.8 * cm,
            2.4 * cm,
            3.0 * cm,
        ],
    )

    results_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#F5F7FA")]),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )

    story.append(results_table)

story.append(Spacer(1, 0.5 * cm))

fig = find_figure(FIGURES["model_performance"])
if fig:
    story.append(make_image(fig, max_width=17 * cm, max_height=9.5 * cm))
    story.append(
        Paragraph(
            "Figure 2. Overall real-data model comparison across R², Pearson "
            "correlation, and within-sample Pearson correlation.",
            styles["SmallReport"],
        )
    )

story.append(PageBreak())


# ============================================================
# SECTION 3 — ERROR METRICS
# ============================================================

story.append(section_title("3. Prediction Error: MAE and RMSE", styles))

story.append(
    Paragraph(
        "MAE and RMSE provide complementary views of prediction error. MAE "
        "measures average absolute deviation, while RMSE gives greater weight "
        "to larger errors.",
        styles["BodyReport"],
    )
)

fig = find_figure(FIGURES["mae_rmse"])
if fig:
    story.append(make_image(fig, max_width=17 * cm, max_height=11 * cm))
    story.append(
        Paragraph(
            "Figure 3. Real-data MAE and RMSE comparison across all evaluated "
            "models.",
            styles["SmallReport"],
        )
    )

story.append(PageBreak())


# ============================================================
# SECTION 4 — R2 AND DATA SCALING
# ============================================================

story.append(section_title("4. R² Progression and Training-Set Scaling", styles))

story.append(
    Paragraph(
        "The benchmark evaluates whether increasingly expressive models provide "
        "measurable changes in predictive performance and whether performance "
        "changes as more training spots are used.",
        styles["BodyReport"],
    )
)

fig = find_figure(FIGURES["r2_scaling"])
if fig:
    story.append(make_image(fig, max_width=17 * cm, max_height=11 * cm))
    story.append(
        Paragraph(
            "Figure 4. R² progression across model families and effect of "
            "training-set size.",
            styles["SmallReport"],
        )
    )

story.append(PageBreak())


# ============================================================
# SECTION 5 — PER GENE
# ============================================================

story.append(section_title("5. Per-Gene Performance", styles))

story.append(
    Paragraph(
        "Aggregate metrics can hide substantial differences between genes. "
        "The per-gene analysis therefore evaluates the within-sample Pearson "
        "correlation separately for individual target genes.",
        styles["BodyReport"],
    )
)

fig = find_figure(FIGURES["per_gene"])
if fig:
    story.append(make_image(fig, max_width=17 * cm, max_height=17 * cm))
    story.append(
        Paragraph(
            "Figure 5. Per-gene performance for the transformer-fusion model, "
            "showing heterogeneous prediction quality across the 56 targets.",
            styles["SmallReport"],
        )
    )

story.append(PageBreak())


# ============================================================
# SECTION 6 — PREDICTED VS MEASURED
# ============================================================

story.append(section_title("6. Predicted vs. Measured Expression", styles))

story.append(
    Paragraph(
        "Predicted-versus-measured plots provide a direct visual assessment of "
        "the learned image-to-expression relationship. The dashed diagonal "
        "represents perfect agreement.",
        styles["BodyReport"],
    )
)

fig = find_figure(FIGURES["predicted_measured"])
if fig:
    story.append(make_image(fig, max_width=17 * cm, max_height=17 * cm))
    story.append(
        Paragraph(
            "Figure 6. Predicted versus measured expression for representative "
            "strong and weak genes.",
            styles["SmallReport"],
        )
    )

story.append(PageBreak())


# ============================================================
# SECTION 7 — SPATIAL MAPS
# ============================================================

story.append(section_title("7. Spatial Prediction Maps", styles))

story.append(
    Paragraph(
        "The spatial maps visualize measured and predicted expression over "
        "spot coordinates. This evaluates whether the model produces spatially "
        "structured predictions rather than only good aggregate numerical "
        "metrics.",
        styles["BodyReport"],
    )
)

fig = find_figure(FIGURES["spatial_maps"])
if fig:
    story.append(make_image(fig, max_width=17 * cm, max_height=18 * cm))
    story.append(
        Paragraph(
            "Figure 7. Spatial maps comparing measured expression with "
            "transformer-fusion predictions for representative genes and "
            "samples.",
            styles["SmallReport"],
        )
    )

story.append(PageBreak())


# ============================================================
# SECTION 8 — STRONG VS WEAK
# ============================================================

story.append(section_title("8. Strong vs. Weak Gene Prediction", styles))

story.append(
    Paragraph(
        "The per-gene analysis separates targets with strong within-sample "
        "correlation from targets that remain difficult to predict. This "
        "heterogeneity is important when interpreting the aggregate model "
        "performance.",
        styles["BodyReport"],
    )
)

fig = find_figure(FIGURES["strong_weak"])
if fig:
    story.append(make_image(fig, max_width=17 * cm, max_height=12 * cm))
    story.append(
        Paragraph(
            "Figure 8. Strong versus weak genes according to within-sample "
            "Pearson correlation.",
            styles["SmallReport"],
        )
    )

story.append(PageBreak())


# ============================================================
# SECTION 9 — ABLATION
# ============================================================

story.append(section_title("9. Ablation Study", styles))

story.append(
    Paragraph(
        "The ablation compares image-only ridge regression, spatially augmented "
        "inputs, multimodal MLP fusion, a stronger MLP, and transformer fusion. "
        "All configurations use the same sample-grouped folds so that changes "
        "in performance can be interpreted as model/input differences rather "
        "than changes in the evaluation split.",
        styles["BodyReport"],
    )
)

fig = find_figure(FIGURES["ablation"])
if fig:
    story.append(make_image(fig, max_width=17 * cm, max_height=11 * cm))
    story.append(
        Paragraph(
            "Figure 9. Ablation of input modalities and model class under the "
            "same cross-validation folds.",
            styles["SmallReport"],
        )
    )

story.append(PageBreak())


# ============================================================
# SECTION 10 — SYNTHETIC VALIDATION
# ============================================================

story.append(section_title("10. Synthetic Dataset: Pipeline Validation", styles))

story.append(
    Paragraph(
        "The synthetic benchmark contains 1,000 image patches across 20 "
        "simulated samples and 56 generic simulated genes. The synthetic "
        "expression values are generated from morphology-associated latent "
        "factors.",
        styles["BodyReport"],
    )
)

story.append(
    Paragraph(
        "<b>Interpretation restriction:</b> these results validate the "
        "software pipeline, feature extraction, grouped cross-validation, "
        "model execution, and figure-generation workflow. They are not "
        "biological validation results.",
        styles["Warning"],
    )
)

if synthetic_metrics is not None:
    strong = get_metric_row(synthetic_metrics, "strong_mlp")
    transformer = get_metric_row(synthetic_metrics, "transformer_fusion")
    mlp = get_metric_row(synthetic_metrics, "mlp_fusion")

    synth_table = [
        ["Model", "MAE", "RMSE", "Mean R²", "Mean Pearson"],
    ]

    for name, row in [
        ("strong_mlp", strong),
        ("mlp_fusion", mlp),
        ("transformer_fusion", transformer),
    ]:
        if row is not None:
            synth_table.append(
                [
                    name,
                    format_num(row.get("MAE")),
                    format_num(row.get("RMSE")),
                    format_num(row.get("R2_mean")),
                    format_num(row.get("pearson_mean")),
                ]
            )

    t = Table(
        synth_table,
        colWidths=[
            5.5 * cm,
            3.0 * cm,
            3.0 * cm,
            3.0 * cm,
            3.0 * cm,
        ],
    )

    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#666666")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#F5F5F5")]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(t)

story.append(PageBreak())


# ============================================================
# SECTION 11 — KEY REAL-DATA RESULTS
# ============================================================

story.append(section_title("11. Key Quantitative Results", styles))

if real_metrics is not None:

    strong = get_metric_row(real_metrics, "strong_mlp")
    extra = get_metric_row(real_metrics, "extra_trees")
    mlp = get_metric_row(real_metrics, "mlp_fusion")
    transformer = get_metric_row(real_metrics, "transformer_fusion")
    ridge = get_metric_row(real_metrics, "ridge_image")
    baseline = get_metric_row(real_metrics, "mean_baseline")

    story.append(
        Paragraph(
            "The completed real-data benchmark produced the following "
            "headline metrics:",
            styles["BodyReport"],
        )
    )

    key_rows = [
        ["Metric", "Result"],
    ]

    if strong is not None:
        key_rows.extend(
            [
                ["Strong MLP — Mean R²",
                 format_num(strong.get("R2_mean"))],
                ["Strong MLP — Mean Pearson",
                 format_num(strong.get("pearson_mean"))],
                ["Strong MLP — MAE",
                 format_num(strong.get("MAE"))],
                ["Strong MLP — RMSE",
                 format_num(strong.get("RMSE"))],
                ["Strong MLP — Positive-R² genes",
                 str(strong.get("genes_with_positive_R2", "N/A"))],
            ]
        )

    if extra is not None:
        key_rows.extend(
            [
                ["Extra Trees — Mean R²",
                 format_num(extra.get("R2_mean"))],
                ["Extra Trees — MAE",
                 format_num(extra.get("MAE"))],
                ["Extra Trees — RMSE",
                 format_num(extra.get("RMSE"))],
            ]
        )

    if transformer is not None:
        key_rows.append(
            [
                "Transformer Fusion — Mean R²",
                format_num(transformer.get("R2_mean")),
            ]
        )
        key_rows.append(
            [
                "Transformer Fusion — Mean Pearson",
                format_num(transformer.get("pearson_mean")),
            ]
        )

    if baseline is not None:
        key_rows.append(
            [
                "Mean Baseline — Mean R²",
                format_num(baseline.get("R2_mean")),
            ]
        )

    kt = Table(
        key_rows,
        colWidths=[10 * cm, 6 * cm],
    )

    kt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                 [colors.white, colors.HexColor("#F5F7FA")]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )

    story.append(kt)

story.append(Spacer(1, 0.5 * cm))

story.append(
    Paragraph(
        "The benchmark demonstrates that the learned image-based models "
        "substantially outperform the mean-expression baseline on the real "
        "dataset. The results also show that performance varies considerably "
        "across genes, motivating per-gene analysis rather than relying only "
        "on a single aggregate score.",
        styles["BodyReport"],
    )
)

story.append(PageBreak())


# ============================================================
# SECTION 12 — LIMITATIONS
# ============================================================

story.append(section_title("12. Limitations and Interpretation", styles))

limitations = [
    "<b>Sample size:</b> the real dataset contains 19 tissue samples, so "
    "sample-grouped validation is important but still limited by the number "
    "of independent samples.",

    "<b>Gene heterogeneity:</b> prediction quality differs substantially "
    "between genes. Aggregate R² and correlation therefore do not represent "
    "every target equally.",

    "<b>Synthetic data:</b> the synthetic dataset contains simulated genes "
    "with no correspondence to the real target panel. It must not be used "
    "to claim biological improvement or combined real-data performance.",

    "<b>Biological interpretation:</b> model performance demonstrates an "
    "image-to-expression predictive relationship within this benchmark; it "
    "does not by itself establish biological causality.",

    "<b>Generalization:</b> held-out sample evaluation is used to reduce "
    "spot-level leakage, but broader validation on independent cohorts would "
    "still be required for stronger external generalization claims.",
]

for item in limitations:
    story.append(
        Paragraph(
            "• " + item,
            styles["BodyReport"],
        )
    )

story.append(Spacer(1, 0.5 * cm))

story.append(
    Paragraph(
        "<b>Overall interpretation:</b> The current pipeline provides a "
        "substantially more complete evaluation than a single-model baseline. "
        "It includes sample-grouped cross-validation, multiple model families, "
        "error metrics, per-gene analysis, predicted-versus-measured "
        "visualization, spatial maps, training-size analysis, and ablation "
        "experiments. These results should be presented as benchmark evidence "
        "for the current pipeline rather than as a full reproduction of "
        "OmiCLIP or STPath.",
        styles["BodyReport"],
    )
)


# ============================================================
# BUILD
# ============================================================

doc.build(
    story,
    onFirstPage=add_page_number,
    onLaterPages=add_page_number,
)

print()
print("=" * 60)
print("VISUAL RESULTS REPORT CREATED")
print("=" * 60)
print(f"Saved to:")
print(PDF_PATH)
print("=" * 60)
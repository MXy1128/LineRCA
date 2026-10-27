"""Create a publication-ready 3x2 ablation figure from the official results.

The two displayed metrics are selected for complementary interpretation:
MAP@10 summarizes ranking quality over the top ten candidates, whereas MRR
measures how early the first correct root cause is retrieved.  Every tested
variant is retained; the complete model is highlighted without altering the
underlying values or the common variant order.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import imageio.v3 as iio
import numpy as np
import pandas as pd
import tifffile
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "thesis" / "chapter_experiments" / "source_data" / "ablation_percent_provided.csv"
OUTPUT_DIR = ROOT / "outputs" / "figures" / "ablation_representative"
THESIS_FIGURE_DIR = ROOT / "thesis" / "chapter_experiments" / "figures"
THESIS_SOURCE_DIR = ROOT / "thesis" / "chapter_experiments" / "source_data"

DATASET_ORDER = ["technical", "non_technical", "mixed"]
DATASET_TITLES = {
    "technical": "Technical-loss dataset",
    "non_technical": "Non-technical-loss dataset",
    "mixed": "Mixed-loss dataset",
}
METRICS = ["MAP@10", "MRR"]
VARIANT_ORDER = [
    "2-layer GAT (full)",
    "1-layer GAT",
    "3-layer GAT",
    "No GAT (and no MCE)",
    "No DyNOTEARS",
    "No ICD (Dm-BOCD)",
]
VARIANT_LABELS = {
    "2-layer GAT (full)": "Full LineRCA",
    "1-layer GAT": "1-layer GAT",
    "3-layer GAT": "3-layer GAT",
    "No GAT (and no MCE)": "w/o GAT–MCE",
    "No DyNOTEARS": "w/o DyNOTEARS",
    "No ICD (Dm-BOCD)": "w/o ICD",
}

# A saturated blue identifies the proposed complete model. Muted neutral colors
# prevent ablation bars from visually competing with the focal method.
COLORS = ["#1769AA", "#9FBAD0", "#B5C5D3", "#C9CDD1", "#B8C8BE", "#C9B8C9"]
HATCHES = ["", "////", "\\\\", "....", "xxxx", "----"]


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.2,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.75,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def load_selected_data() -> pd.DataFrame:
    data = pd.read_csv(INPUT)
    required = {"Dataset", "Variant", *METRICS}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    selected = data.loc[
        data["Dataset"].isin(DATASET_ORDER) & data["Variant"].isin(VARIANT_ORDER),
        ["Dataset", "Variant", *METRICS],
    ].copy()
    selected["Dataset"] = pd.Categorical(selected["Dataset"], DATASET_ORDER, ordered=True)
    selected["Variant"] = pd.Categorical(selected["Variant"], VARIANT_ORDER, ordered=True)
    selected = selected.sort_values(["Dataset", "Variant"]).reset_index(drop=True)

    expected_rows = len(DATASET_ORDER) * len(VARIANT_ORDER)
    if len(selected) != expected_rows:
        raise ValueError(f"Expected {expected_rows} dataset–variant rows, found {len(selected)}")
    return selected


def padded_limits(values: np.ndarray) -> tuple[float, float]:
    """Use transparent, data-driven truncated axes with adequate visual padding."""
    low = float(values.min())
    high = float(values.max())
    span = max(high - low, 1.0)
    lower = max(0.0, np.floor((low - 0.12 * span) * 2.0) / 2.0)
    upper = min(100.0, np.ceil((high + 0.18 * span) * 2.0) / 2.0)
    if upper - lower < 3.0:
        center = 0.5 * (upper + lower)
        lower, upper = max(0.0, center - 1.5), min(100.0, center + 1.5)
    return lower, upper


def make_figure(selected: pd.DataFrame) -> mpl.figure.Figure:
    configure_style()
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.9), constrained_layout=False)
    x = np.arange(len(VARIANT_ORDER))
    panel_letters = iter("abcdef")

    for row, dataset in enumerate(DATASET_ORDER):
        subset = selected[selected["Dataset"] == dataset].set_index("Variant").loc[VARIANT_ORDER]
        for col, metric in enumerate(METRICS):
            ax = axes[row, col]
            values = subset[metric].to_numpy(dtype=float)
            bars = ax.bar(
                x,
                values,
                width=0.70,
                color=COLORS,
                edgecolor="#33404A",
                linewidth=[1.15, 0.55, 0.55, 0.55, 0.55, 0.55],
                zorder=3,
            )
            for bar, hatch in zip(bars, HATCHES):
                bar.set_hatch(hatch)

            lower, upper = padded_limits(values)
            ax.set_ylim(lower, upper)
            ax.set_ylabel(f"{metric} (%)")
            ax.set_xticks(x)
            ax.set_xticklabels(["Full", "1L", "3L", "No GAT", "No DyN", "No ICD"])
            ax.grid(axis="y", color="#D9DEE3", linewidth=0.6, alpha=0.85, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(axis="x", length=0, pad=2)

            # Emphasize the complete model numerically. If another variant is the
            # panel maximum, mark it separately so the chart remains auditable.
            full_value = values[0]
            full_offset = 0.025 * (upper - lower)
            ax.text(
                x[0],
                full_value + full_offset,
                f"{full_value:.2f}",
                ha="center",
                va="bottom",
                fontsize=7.2,
                fontweight="bold",
                color="#0E4773",
            )
            best_index = int(np.argmax(values))
            if best_index != 0:
                ax.text(
                    x[best_index],
                    values[best_index] + full_offset,
                    f"{values[best_index]:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=6.8,
                    color="#3D464D",
                )

            letter = next(panel_letters)
            ax.set_title(f"({letter}) {DATASET_TITLES[dataset]} · {metric}", pad=6, fontweight="semibold")

    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor="#33404A",
            linewidth=1.05 if i == 0 else 0.55,
            hatch=hatch,
            label=VARIANT_LABELS[variant],
        )
        for i, (variant, color, hatch) in enumerate(zip(VARIANT_ORDER, COLORS, HATCHES))
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.953),
        ncol=3,
        frameon=False,
        columnspacing=1.2,
        handlelength=1.8,
        handleheight=0.9,
    )
    fig.suptitle(
        "Ablation effects across line-loss scenarios",
        x=0.5,
        y=0.992,
        fontsize=11.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.012,
        "All variants use the same split, candidate space, and evaluation protocol; bars report one official run.",
        ha="center",
        va="bottom",
        fontsize=7.3,
        color="#4A535A",
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.875, bottom=0.075, hspace=0.42, wspace=0.25)
    return fig


def save_outputs(fig: mpl.figure.Figure, selected: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    THESIS_SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    base = OUTPUT_DIR / "fig_ablation_map10_mrr"
    thesis_base = THESIS_FIGURE_DIR / "fig_exp_12_ablation_map10_mrr"
    for path_base in (base, thesis_base):
        fig.savefig(path_base.with_suffix(".png"), dpi=600, facecolor="white")
        fig.savefig(path_base.with_suffix(".pdf"), facecolor="white")
        fig.savefig(path_base.with_suffix(".svg"), facecolor="white")
        # Convert the already-rendered 600-dpi PNG to TIFF. This avoids a
        # backend-specific Matplotlib/libtiff crash observed for direct TIFF
        # rendering on Windows while preserving the raster resolution.
        rendered = iio.imread(path_base.with_suffix(".png"))
        tifffile.imwrite(
            path_base.with_suffix(".tiff"),
            rendered,
            compression="deflate",
            resolution=(600, 600),
            resolutionunit="INCH",
        )

    long_data = selected.melt(
        id_vars=["Dataset", "Variant"],
        value_vars=METRICS,
        var_name="Metric",
        value_name="Value_percent",
    )
    long_data["Selection_rationale"] = long_data["Metric"].map(
        {
            "MAP@10": "aggregate top-10 ranking quality",
            "MRR": "rank of the first correct root cause",
        }
    )
    long_data.to_csv(OUTPUT_DIR / "selected_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    long_data.to_csv(THESIS_SOURCE_DIR / "ablation_map10_mrr.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    selected = load_selected_data()
    fig = make_figure(selected)
    save_outputs(fig, selected)
    plt.close(fig)
    print(f"Created publication figure in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

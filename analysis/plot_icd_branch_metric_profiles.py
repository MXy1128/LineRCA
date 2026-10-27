"""Plot manuscript-ready ranking profiles for the three LineRCA ICD branches.

The script reads the official consolidated result table rather than embedding
metric values in plotting code.  It exports a traceable long-format source-data
file together with PNG, editable SVG/PDF, and 600-dpi TIFF outputs.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "model_vs_all_baselines" / "all_results_percent.csv"
OUTPUT = ROOT / "outputs" / "figures" / "icd_branch_comparison"

MODEL_MAP = {
    "LineRCA (ours)-DM-BOCD": "DM-BOCD",
    "LineRCA (ours)-BOCPD": "BOCPD",
    "LineRCA (ours)-MAD": "MAD",
}
ORDER = ["DM-BOCD", "BOCPD", "MAD"]
COLORS = {"DM-BOCD": "#C8463C", "BOCPD": "#4C78A8", "MAD": "#5B9A68"}
MARKERS = {"DM-BOCD": "o", "BOCPD": "s", "MAD": "^"}
LINESTYLES = {"DM-BOCD": "-", "BOCPD": "--", "MAD": "-."}


def percent_to_float(value: object) -> float:
    return float(str(value).strip().rstrip("%"))


def load_source() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(INPUT)
    # The consolidated presentation table writes the dataset name once at the
    # final row of each three-branch block. Back-fill only for selecting the
    # block; metric values themselves are never imputed.
    raw["DatasetBlock"] = raw["Dataset"].bfill()
    selected = raw.loc[
        raw.Model.isin(MODEL_MAP) & raw.DatasetBlock.eq("mixed")
    ].copy()
    selected["Branch"] = selected.Model.map(MODEL_MAP)
    selected = selected.set_index("Branch").loc[ORDER].reset_index()
    if len(selected) != 3:
        raise ValueError("Expected exactly three LineRCA ICD branch rows")
    metric_columns = [
        "PR@1", "PR@3", "PR@5", "PR@7", "PR@10",
        "MAP@3", "MAP@5", "MAP@7", "MAP@10", "MRR",
        "NDCG@1", "NDCG@3", "NDCG@5", "NDCG@7", "NDCG@10",
    ]
    for column in metric_columns:
        selected[column] = selected[column].map(percent_to_float)

    records = []
    for branch in ORDER:
        row = selected.loc[selected.Branch.eq(branch)].iloc[0]
        for family, ks in {"PR": [1, 3, 5, 7, 10], "MAP": [3, 5, 7, 10], "NDCG": [1, 3, 5, 7, 10]}.items():
            for k in ks:
                records.append(
                    {"Branch": branch, "MetricFamily": family, "K": k,
                     "ScorePercent": row[f"{family}@{k}"]}
                )
        records.append(
            {"Branch": branch, "MetricFamily": "MRR", "K": np.nan,
             "ScorePercent": row["MRR"]}
        )
    return selected, pd.DataFrame(records)


def style_axis(ax: plt.Axes, ylabel: str, ylim: tuple[float, float]) -> None:
    ax.set_xlabel("K")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)


def main() -> None:
    selected, source = load_source()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source.to_csv(OUTPUT / "icd_branch_metric_profiles_source_data.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.15), constrained_layout=True)
    panel_specs = [
        (axes[0, 0], "PR", "PR@K (%)", (64, 82)),
        (axes[0, 1], "MAP", "MAP@K (%)", (56, 63)),
        (axes[1, 0], "NDCG", "NDCG@K (%)", (61, 68)),
    ]
    for ax, family, ylabel, ylim in panel_specs:
        panel = source.loc[source.MetricFamily.eq(family)]
        for branch in ORDER:
            values = panel.loc[panel.Branch.eq(branch)].sort_values("K")
            ax.plot(
                values.K,
                values.ScorePercent,
                label=branch,
                color=COLORS[branch],
                marker=MARKERS[branch],
                linestyle=LINESTYLES[branch],
                linewidth=1.45,
                markersize=4.2,
                markerfacecolor="white",
                markeredgewidth=1.0,
            )
        ax.set_xticks(sorted(panel.K.unique()))
        style_axis(ax, ylabel, ylim)

    ax = axes[1, 1]
    mrr = source.loc[source.MetricFamily.eq("MRR")].set_index("Branch").loc[ORDER]
    x = np.arange(len(ORDER))
    bars = ax.bar(
        x,
        mrr.ScorePercent,
        width=0.58,
        color=[COLORS[b] for b in ORDER],
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xticks(x, ORDER)
    ax.set_ylabel("MRR (%)")
    ax.set_ylim(70, 74)
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.55, alpha=0.85)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, mrr.ScorePercent):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.10,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=6.8,
            fontweight="bold",
            color="#26323A",
        )

    for index, ax in enumerate(axes.ravel()):
        ax.text(
            -0.10,
            1.085,
            chr(ord("a") + index),
            transform=ax.transAxes,
            fontsize=8.5,
            fontweight="bold",
            va="top",
            clip_on=False,
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=3)
    fig.suptitle(
        "Effect of the ICD branch on root-cause entity ranking",
        fontsize=9.5,
        fontweight="bold",
        y=1.055,
    )
    base = OUTPUT / "fig_icd_branch_metric_profiles"
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    dm = selected.loc[selected.Branch.eq("DM-BOCD")].iloc[0]
    bocpd = selected.loc[selected.Branch.eq("BOCPD")].iloc[0]
    mad = selected.loc[selected.Branch.eq("MAD")].iloc[0]
    caption = (
        "Figure X | Effect of the ICD branch on LineRCA entity-level ranking on "
        "the mixed-loss dataset. (a) PR@K, (b) MAP@K, (c) NDCG@K, and (d) MRR. "
        "The GAT, DyNOTEARS, MCE, CEAF, chronological split, candidate space, and "
        "evaluation protocol are held fixed; only the individual-change-detection "
        "branch is replaced. Markers show the result of the official run for each "
        "branch; uncertainty intervals are not available for this comparison."
    )
    (OUTPUT / "figure_caption_en.txt").write_text(caption, encoding="utf-8")
    paragraph = f"""### Effect of the individual-change-detection branch

To isolate the contribution of the individual-change-detection mechanism, we replaced the ICD branch while retaining the remaining LineRCA architecture and the complete entity-ranking protocol. As shown in Fig. X, Dm-BOCD obtained the highest value for 14 of the 15 reported metrics. Relative to BOCPD, Dm-BOCD improved PR@1 by {dm['PR@1'] - bocpd['PR@1']:.2f} percentage points, MAP@10 by {dm['MAP@10'] - bocpd['MAP@10']:.2f} percentage points, MRR by {dm['MRR'] - bocpd['MRR']:.2f} percentage points, and NDCG@10 by {dm['NDCG@10'] - bocpd['NDCG@10']:.2f} percentage points. The corresponding gains over MAD were {dm['PR@1'] - mad['PR@1']:.2f}, {dm['MAP@10'] - mad['MAP@10']:.2f}, {dm['MRR'] - mad['MRR']:.2f}, and {dm['NDCG@10'] - mad['NDCG@10']:.2f} percentage points. The only exception was PR@7, for which BOCPD exceeded Dm-BOCD by {bocpd['PR@7'] - dm['PR@7']:.2f} percentage points. These results indicate that the robust Dm-BOCD update provides a consistent benefit across early retrieval, average precision, reciprocal rank, and discounted cumulative gain, rather than improving only one ranking cutoff.

{caption}

Statistical reporting: all three branches use the same mixed-loss test split and candidate space. Each marker represents one official trained run; no standard deviation or confidence interval is claimed. Source data: `icd_branch_metric_profiles_source_data.csv`.
"""
    (OUTPUT / "manuscript_text_en.md").write_text(paragraph, encoding="utf-8")
    print(base)


if __name__ == "__main__":
    main()

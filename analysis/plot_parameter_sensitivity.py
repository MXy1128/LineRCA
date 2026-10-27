"""Create the manuscript parameter-sensitivity figure from real experiment logs."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results" / "parameter_sensitivity" / "metrics_by_run.csv"
OUTPUT = ROOT / "outputs" / "figures" / "parameter_sensitivity"

BLUE = "#356B9A"
ORANGE = "#D9822B"
GREY = "#7D8790"
GRID = "#D9DEE3"

SPECS = [
    ("window", "Temporal window length, $T$", 12.0),
    ("hidden", "Hidden dimension, $d$", 64.0),
    ("run_length", "Expected Dm-BOCD run length, $L_0$", 50.0),
    ("dynotears_quantile", "DyNOTEARS sparsification quantile, $q$", 0.85),
]
METRICS = [("MRR", "MRR", BLUE, "o"), ("NDCG@10", "NDCG@10", ORANGE, "s")]


def prepare_source_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Copy the shared default into each OFAT panel and aggregate across seeds."""
    records = []
    default = raw.loc[raw.Parameter.eq("default")]
    expected_seeds = sorted(raw.Seed.unique())
    if sorted(default.Seed.unique()) != expected_seeds:
        raise ValueError("Default configuration is missing for one or more seeds")
    for parameter, _, default_value in SPECS:
        panel = raw.loc[raw.Parameter.eq(parameter)].copy()
        base = default.copy()
        base["Parameter"] = parameter
        base["Value"] = default_value
        panel = pd.concat([panel, base], ignore_index=True)
        if panel.groupby("Value").Seed.nunique().min() != len(expected_seeds):
            raise ValueError(f"Incomplete seed coverage for {parameter}")
        for metric, _, _, _ in METRICS:
            summary = panel.groupby("Value")[metric].agg(["mean", "std", "count"]).reset_index()
            for row in summary.itertuples(index=False):
                records.append(
                    {
                        "parameter": parameter,
                        "value": row.Value,
                        "metric": metric,
                        "mean_percent": 100.0 * row.mean,
                        "sd_percent": 100.0 * row.std,
                        "n_seeds": int(row.count),
                        "is_default": bool(np.isclose(row.Value, default_value)),
                    }
                )
    return pd.DataFrame(records)


def main() -> None:
    if not INPUT.exists():
        raise FileNotFoundError(f"Run the sensitivity experiment first: {INPUT}")
    raw = pd.read_csv(INPUT)
    if raw.Seed.nunique() < 3:
        raise ValueError("The manuscript figure requires at least three completed seeds")
    source = prepare_source_data(raw)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source.to_csv(OUTPUT / "parameter_sensitivity_source_data.csv", index=False)

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
    axes = axes.ravel()
    global_min = source.mean_percent.sub(source.sd_percent).min()
    global_max = source.mean_percent.add(source.sd_percent).max()
    lower = np.floor((global_min - 1.0) / 2.0) * 2.0
    upper = np.ceil((global_max + 1.0) / 2.0) * 2.0

    for panel_index, (ax, (parameter, xlabel, default_value)) in enumerate(zip(axes, SPECS)):
        panel = source.loc[source.parameter.eq(parameter)]
        for metric, label, color, marker in METRICS:
            values = panel.loc[panel.metric.eq(metric)].sort_values("value")
            ax.errorbar(
                values.value,
                values.mean_percent,
                yerr=values.sd_percent,
                color=color,
                marker=marker,
                markersize=4.2,
                markerfacecolor="white",
                markeredgewidth=1.0,
                linewidth=1.35,
                capsize=2.3,
                capthick=0.8,
                label=label,
                zorder=3,
            )
            default_row = values.loc[np.isclose(values.value, default_value)].iloc[0]
            ax.scatter(
                [default_value], [default_row.mean_percent], s=32, marker=marker,
                facecolor=color, edgecolor="white", linewidth=0.7, zorder=4,
            )
        ax.axvline(default_value, color=GREY, linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
        ax.text(
            default_value, upper - 0.5, "default", ha="center", va="top",
            fontsize=6.3, color="#59636C",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.85},
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Entity-ranking score (%)" if panel_index % 2 == 0 else "")
        ax.set_ylim(lower, upper)
        ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
        ax.text(
            -0.10, 1.04, chr(ord("a") + panel_index), transform=ax.transAxes,
            fontsize=8.5, fontweight="bold", va="top", clip_on=False,
        )
        if parameter in {"window", "hidden", "run_length"}:
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True, nbins=5))
        else:
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter("%.2f"))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.015), ncol=2)
    fig.suptitle(
        "Parameter sensitivity on the mixed-loss benchmark",
        fontsize=9.5, fontweight="bold", y=1.055,
    )
    base = OUTPUT / "fig_parameter_sensitivity"
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    caption = (
        "Figure X | Parameter sensitivity of LineRCA on the mixed-loss dataset. "
        "Entity-level MRR and NDCG@10 are reported while varying (a) temporal "
        "window length, (b) hidden dimension, (c) expected Dm-BOCD run length, "
        "and (d) the DyNOTEARS edge-strength quantile. Only one parameter is "
        "changed at a time; dashed lines and filled markers indicate the default "
        "configuration. Points show the mean and error bars show ±1 s.d. over "
        f"{raw.Seed.nunique()} random seeds. All models use the chronological "
        "train/validation/test split and the same entity-level ranking protocol "
        "as the main comparison. Source data are provided with the figure."
    )
    (OUTPUT / "figure_caption_en.txt").write_text(caption, encoding="utf-8")
    print(base)


if __name__ == "__main__":
    main()

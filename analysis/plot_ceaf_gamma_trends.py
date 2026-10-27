"""Plot CEAF gamma trajectories from the three official full-model logs."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "figures" / "ceaf_gamma"
LOGS = {
    "Technical": ROOT / "logs" / "gat_entity_localization_technical_20260810_120010.log",
    "Non-technical": ROOT / "logs" / "gat_entity_localization_non_technical_20260810_121150.log",
    "Mixed": ROOT / "logs" / "gat_entity_localization_mixed_20260810_122555.log",
}
COLORS = {
    "Technical": "#3B7BBF",
    "Non-technical": "#E58B32",
    "Mixed": "#6F53A3",
}
ANNOTATION_OFFSETS = {
    "Technical": (5, 7),
    "Non-technical": (8, -23),
    "Mixed": (8, 15),
}
PATTERN = re.compile(
    r"MODEL/EPOCH (?P<epoch>\d+)/\d+ .*?val_loss=(?P<val>[0-9.]+) "
    r".*?gamma=(?P<gamma>[0-9.]+)"
)


def load_source_data() -> pd.DataFrame:
    rows = []
    for dataset, path in LOGS.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            match = PATTERN.search(line)
            if match:
                rows.append(
                    {
                        "dataset": dataset,
                        "epoch": int(match["epoch"]),
                        "gamma": float(match["gamma"]),
                        "validation_loss": float(match["val"]),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty or set(frame.dataset) != set(LOGS):
        raise RuntimeError("Failed to extract all three official gamma trajectories")
    frame["is_best_epoch"] = False
    for dataset, group in frame.groupby("dataset"):
        best_index = group.validation_loss.idxmin()
        frame.loc[best_index, "is_best_epoch"] = True
    return frame


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = load_source_data()
    data.to_csv(OUTPUT / "ceaf_gamma_source_data.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
        }
    )

    # Double-column width (183 mm), compact height for a single quantitative panel.
    fig, ax = plt.subplots(figsize=(183 / 25.4, 82 / 25.4))
    for dataset in ("Technical", "Non-technical", "Mixed"):
        group = data[data.dataset.eq(dataset)].sort_values("epoch")
        best = group[group.is_best_epoch].iloc[0]
        ax.plot(
            group.epoch,
            group.gamma,
            color=COLORS[dataset],
            linewidth=1.8,
            marker="o",
            markersize=2.7,
            markerfacecolor="white",
            markeredgewidth=0.7,
            label=dataset,
            zorder=3,
        )
        ax.scatter(
            best.epoch,
            best.gamma,
            s=35,
            color=COLORS[dataset],
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
        )
        ax.annotate(
            f"best epoch {int(best.epoch)}\nγ={best.gamma:.3f}",
            (best.epoch, best.gamma),
            xytext=ANNOTATION_OFFSETS[dataset],
            textcoords="offset points",
            color=COLORS[dataset],
            fontsize=6.8,
            linespacing=1.05,
        )

    ax.axhline(0.5, color="#888888", linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
    ax.text(0.5, 0.505, "initial equal weighting", color="#777777", fontsize=6.8)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel(r"CEAF fusion coefficient $\gamma$ (ICD weight)")
    ax.set_xlim(1, int(data.epoch.max()))
    ax.set_ylim(0.20, 0.52)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
    ax.legend(loc="lower left", ncol=3, handlelength=2.2, columnspacing=1.4)
    fig.tight_layout(pad=0.5)

    base = OUTPUT / "ceaf_gamma_trends"
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)
    print(data[data.is_best_epoch].to_string(index=False))


if __name__ == "__main__":
    main()

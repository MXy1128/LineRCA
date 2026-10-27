"""Draw training/validation loss curves for the three official full-model runs."""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "figures" / "loss_curves"
LOGS = {
    "Technical": ROOT / "logs" / "gat_entity_localization_technical_20260810_120010.log",
    "Non-technical": ROOT / "logs" / "gat_entity_localization_non_technical_20260810_121150.log",
    "Mixed": ROOT / "logs" / "gat_entity_localization_mixed_20260810_122555.log",
}
PATTERN = re.compile(
    r"MODEL/EPOCH (?P<epoch>\d+)/\d+ .*?train_loss=(?P<train>[0-9.]+) "
    r".*?val_loss=(?P<validation>[0-9.]+)"
)
TRAIN_COLOR = "#3B7BBF"
VALIDATION_COLOR = "#E58B32"


def source_data() -> pd.DataFrame:
    rows = []
    for dataset, path in LOGS.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            match = PATTERN.search(line)
            if match:
                rows.append(
                    {
                        "dataset": dataset,
                        "epoch": int(match["epoch"]),
                        "training_loss": float(match["train"]),
                        "validation_loss": float(match["validation"]),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty or set(frame.dataset) != set(LOGS):
        raise RuntimeError("Could not parse all official loss logs")
    frame["is_best_epoch"] = False
    for _, group in frame.groupby("dataset"):
        frame.loc[group.validation_loss.idxmin(), "is_best_epoch"] = True
    return frame


def style():
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Microsoft YaHei", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )


def draw_panel(ax, group: pd.DataFrame, title: str, show_legend=True):
    group = group.sort_values("epoch")
    best = group[group.is_best_epoch].iloc[0]
    ax.plot(group.epoch, group.training_loss, color=TRAIN_COLOR, linewidth=1.8,
            marker="o", markersize=2.4, markerfacecolor="white", markeredgewidth=0.65,
            label="Training loss")
    ax.plot(group.epoch, group.validation_loss, color=VALIDATION_COLOR, linewidth=1.8,
            marker="s", markersize=2.4, markerfacecolor="white", markeredgewidth=0.65,
            label="Validation loss")
    ax.axvline(best.epoch, color="#6F53A3", linewidth=1.0, linestyle=(0, (4, 3)),
               label=f"Best epoch ({int(best.epoch)})")
    ax.scatter(best.epoch, best.validation_loss, s=26, color=VALIDATION_COLOR,
               edgecolor="white", linewidth=0.7, zorder=5)
    ax.annotate(
        f"best val.={best.validation_loss:.3f}",
        (best.epoch, best.validation_loss),
        xytext=(5, 8), textcoords="offset points", color=VALIDATION_COLOR,
        fontsize=6.5,
    )
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim(1, int(group.epoch.max()))
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.65)
    if show_legend:
        ax.legend(loc="upper right", frameon=False)


def save(fig, base: Path):
    fig.savefig(base.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    data = source_data()
    data.to_csv(OUTPUT / "loss_curve_source_data.csv", index=False)
    style()

    fig, axes = plt.subplots(1, 3, figsize=(183 / 25.4, 58 / 25.4), constrained_layout=True)
    for index, dataset in enumerate(("Technical", "Non-technical", "Mixed")):
        draw_panel(axes[index], data[data.dataset.eq(dataset)], dataset, show_legend=index == 0)
        axes[index].text(-0.15, 1.04, chr(ord("a") + index), transform=axes[index].transAxes,
                         fontsize=9, fontweight="bold", va="bottom")
    save(fig, OUTPUT / "train_validation_loss_combined")
    plt.close(fig)

    for dataset in ("Technical", "Non-technical", "Mixed"):
        fig, ax = plt.subplots(figsize=(89 / 25.4, 68 / 25.4))
        draw_panel(ax, data[data.dataset.eq(dataset)], dataset)
        fig.tight_layout(pad=0.5)
        save(fig, OUTPUT / f"train_validation_loss_{dataset.lower().replace('-', '_')}")
        plt.close(fig)

    print(OUTPUT)
    print(data[data.is_best_epoch].to_string(index=False))


if __name__ == "__main__":
    main()

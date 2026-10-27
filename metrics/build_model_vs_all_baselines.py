"""Merge the current no-GRU LineRCA-GAT runs with every fair baseline."""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results" / "fair_hierarchical_comparison" / "all_models_all_datasets.csv"
GAT_RUNS = {
    "technical": ROOT / "results" / "gat_entity_localization" / "20260807_172913" / "metrics.csv",
    "non_technical": ROOT / "results" / "gat_entity_localization" / "20260807_173220_non_technical" / "metrics.csv",
    "mixed": ROOT / "results" / "gat_entity_localization" / "20260807_173220_mixed" / "metrics.csv",
}
METRICS = [
    "PR@1", "PR@3", "PR@5", "PR@7", "PR@10",
    "MAP@3", "MAP@5", "MAP@7", "MAP@10", "MRR",
    "NDCG@1", "NDCG@3", "NDCG@5", "NDCG@7", "NDCG@10",
]
COLS = ["Dataset", "Model", "Candidates", *METRICS]


def main() -> None:
    baseline = pd.read_csv(BASELINE)[COLS]
    current = []
    for dataset, path in GAT_RUNS.items():
        row = pd.read_csv(path).iloc[[0]].copy()
        row["Dataset"] = dataset
        row["Model"] = "LineRCA-GAT (ours)"
        current.append(row[COLS])

    combined = pd.concat([baseline, *current], ignore_index=True)
    combined = combined.sort_values(["Dataset", "PR@1"], ascending=[True, False])
    combined.insert(3, "PR@1 Rank", combined.groupby("Dataset")["PR@1"].rank(method="min", ascending=False).astype(int))

    counts = combined.groupby("Dataset")["Model"].nunique().to_dict()
    if counts != {"mixed": 15, "non_technical": 15, "technical": 15}:
        raise RuntimeError(f"unexpected model counts: {counts}")
    if combined[METRICS].isna().any().any():
        raise RuntimeError("comparison contains missing metrics")

    dest = ROOT / "results" / "model_vs_all_baselines"
    dest.mkdir(parents=True, exist_ok=True)
    combined.to_csv(dest / "all_results_raw.csv", index=False)

    percent = combined.copy()
    for col in METRICS:
        percent[col] = percent[col].map(lambda value: f"{value * 100:.2f}%")
    percent.to_csv(dest / "all_results_percent.csv", index=False)

    lines = []
    for dataset, table in percent.groupby("Dataset", sort=False):
        table = table.drop(columns="Dataset")
        header = "| " + " | ".join(table.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
        rows = ["| " + " | ".join(map(str, row)) + " |" for row in table.itertuples(index=False, name=None)]
        lines.extend([
            f"## {dataset}", "",
            header, separator, *rows, "",
        ])
    (dest / "all_results.md").write_text("\n".join(lines), encoding="utf-8")
    print(percent.to_string(index=False))
    print(f"\nSaved to: {dest}")


if __name__ == "__main__":
    main()

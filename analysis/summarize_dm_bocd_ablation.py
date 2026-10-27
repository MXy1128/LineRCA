"""Collect the final Dm-BOCD ablation runs into thesis-ready CSV tables."""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "results" / "gat_entity_localization"
OUTPUT = ROOT / "results" / "dm_bocd_ablation_final"

RUNS = {
    ("technical", "1-layer GAT"): "20260810_134023_technical",
    ("technical", "2-layer GAT (full)"): "20260810_120010_technical",
    ("technical", "3-layer GAT"): "20260810_135221_technical",
    ("technical", "No GAT (and no MCE)"): "20260810_135753_technical",
    ("technical", "No DyNOTEARS"): "20260810_140236_technical",
    ("technical", "No ICD (Dm-BOCD)"): "20260810_140736_technical",
    ("non_technical", "1-layer GAT"): "20260810_140922_non_technical",
    ("non_technical", "2-layer GAT (full)"): "20260810_121150_non_technical",
    ("non_technical", "3-layer GAT"): "20260810_141545_non_technical",
    ("non_technical", "No GAT (and no MCE)"): "20260810_142147_non_technical",
    ("non_technical", "No DyNOTEARS"): "20260810_143152_non_technical",
    ("non_technical", "No ICD (Dm-BOCD)"): "20260810_143800_non_technical",
    ("mixed", "1-layer GAT"): "20260810_143943_mixed",
    ("mixed", "2-layer GAT (full)"): "20260810_122555_mixed",
    ("mixed", "3-layer GAT"): "20260810_144730_mixed",
    ("mixed", "No GAT (and no MCE)"): "20260810_145326_mixed",
    ("mixed", "No DyNOTEARS"): "20260810_150059_mixed",
    ("mixed", "No ICD (Dm-BOCD)"): "20260810_151403_mixed",
}

OFFICIAL_FULL_BEST_EPOCH = {
    "technical": 9,
    "non_technical": 17,
    "mixed": 17,
}


def main():
    rows = []
    for (dataset, variant), directory in RUNS.items():
        row = pd.read_csv(RUN_ROOT / directory / "metrics.csv").iloc[0].to_dict()
        row.update(Dataset=dataset, Variant=variant, RunDirectory=directory)
        if pd.isna(row.get("Layers")):
            row["Layers"] = 2
        if variant == "2-layer GAT (full)" and pd.isna(row.get("BestEpoch")):
            row["BestEpoch"] = OFFICIAL_FULL_BEST_EPOCH[dataset]
        rows.append(row)

    frame = pd.DataFrame(rows)
    metric_columns = [
        column for column in frame
        if column.startswith(("PR@", "MAP@", "NDCG@"))
    ] + ["MRR"]
    leading = [
        "Dataset", "Variant", "Layers", "Epochs", "BestEpoch", "Candidates",
        "TestRooted", "Params", "Seconds", "RunDirectory",
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame[leading + metric_columns].to_csv(OUTPUT / "ablation_raw.csv", index=False)

    percent = frame[leading + metric_columns].copy()
    percent[metric_columns] = percent[metric_columns].astype(float).mul(100).round(2)
    percent.to_csv(OUTPUT / "ablation_percent.csv", index=False)

    mean = frame.groupby("Variant", as_index=False)[metric_columns].mean()
    mean[metric_columns] = mean[metric_columns].mul(100).round(2)
    mean.to_csv(OUTPUT / "cross_dataset_macro_average_percent.csv", index=False)

    raw_mean = frame.groupby("Variant")[metric_columns].mean()
    full = raw_mean.loc["2-layer GAT (full)"]
    comparisons = []
    for variant in raw_mean.index:
        if variant == "2-layer GAT (full)":
            continue
        delta = full - raw_mean.loc[variant]
        row = {
            "Comparison": f"Full minus {variant}",
            "ImprovedMetrics": int((delta > 0).sum()),
            "DecreasedMetrics": int((delta < 0).sum()),
            "TiedMetrics": int((delta == 0).sum()),
            "MeanDeltaPercentagePoints": round(float(delta.mean() * 100), 2),
        }
        row.update({column: round(float(delta[column] * 100), 2) for column in metric_columns})
        comparisons.append(row)
    pd.DataFrame(comparisons).to_csv(OUTPUT / "full_minus_variants_percent_points.csv", index=False)
    print(OUTPUT)
    print(mean.to_string(index=False))


if __name__ == "__main__":
    main()

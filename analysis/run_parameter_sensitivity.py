"""Run the publication parameter-sensitivity experiment for LineRCA.

The mixed-loss dataset is used because it is the most demanding benchmark: a
sample may contain both a technical and a non-technical root cause.  For each
seed, the expensive entity localisers and the continuous DyNOTEARS edge scores
are fitted once and reused across controlled one-factor-at-a-time sweeps.  The
neural model is retrained for every configuration and evaluated with the same
entity-level ranking protocol as the main experiment.

Outputs
-------
results/parameter_sensitivity/seed_<seed>/<run_id>/metrics.csv
logs/parameter_sensitivity/<run_id>_seed_<seed>.log
results/parameter_sensitivity/metrics_by_run.csv
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from metrics.ranking import localization_metrics
from model.data import load_data
from model.dynotears import fit_dynotears
from model.gat import LineRCA
from model.localization import (
    build_candidate_schema,
    combine_type_and_entity_scores,
    encode_entity_labels,
    extract_true_entities,
    fit_entity_localizers,
)
from model.preprocessing import build_localization_features
from model.run_gat_entity_localization import (
    causes_for,
    predict,
    train_type_model,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "simulated_pandapower_v2" / "mixed_line_loss.csv"
RESULT_ROOT = ROOT / "results" / "parameter_sensitivity"
LOG_ROOT = ROOT / "logs" / "parameter_sensitivity"


@dataclass(frozen=True)
class Configuration:
    parameter: str
    value: float
    window: int = 12
    hidden: int = 64
    run_length: float = 50.0
    dynotears_quantile: float = 0.85

    @property
    def run_id(self) -> str:
        value = f"{self.value:g}".replace(".", "p")
        return f"{self.parameter}_{value}"


def configurations() -> list[Configuration]:
    """Return unique OFAT settings; the default is evaluated once per sweep."""
    base = Configuration("default", 0.0)
    groups = {
        "window": [6, 9, 12, 18, 24],
        "hidden": [32, 48, 64, 96, 128],
        "run_length": [20, 35, 50, 75, 100],
        "dynotears_quantile": [0.70, 0.78, 0.85, 0.90, 0.95],
    }
    output = [base]
    for parameter, values in groups.items():
        for value in values:
            if (parameter == "window" and value == base.window) or (
                parameter == "hidden" and value == base.hidden
            ) or (parameter == "run_length" and value == base.run_length) or (
                parameter == "dynotears_quantile" and value == base.dynotears_quantile
            ):
                continue
            output.append(replace(base, parameter=parameter, value=float(value), **{parameter: value}))
    return output


def _quiet_logger(path: Path) -> logging.Logger:
    """Logger used for seed-level preprocessing that does not emit epochs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"sensitivity_prepare_{path.stem}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    return logger


def _threshold_adjacency(score: np.ndarray, quantile: float) -> torch.Tensor:
    positive = score[score > 1e-12]
    threshold = np.quantile(positive, quantile) if positive.size else np.inf
    adjacency = (score >= threshold).astype(np.float32)
    np.fill_diagonal(adjacency, 1.0)
    return torch.from_numpy(adjacency)


def _align_localization_rows(test_rows: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    lookup = {
        (str(row.timestamp), str(row.grid_id), str(row.event_id)): index
        for index, row in frame.iterrows()
    }
    return np.asarray(
        [lookup[(str(row.timestamp), str(row.grid_id), str(row.event_id))]
         for row in test_rows.itertuples(index=False)],
        dtype=int,
    )


def run_seed(
    seed: int,
    epochs: int,
    min_epochs: int,
    patience: int,
    dynotears_steps: int,
    entity_trees: int,
    torch_threads: int,
    smoke: bool,
) -> list[dict]:
    torch.set_num_threads(torch_threads)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    causes = causes_for("mixed")
    seed_root = RESULT_ROOT / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)

    prepare_logger = _quiet_logger(LOG_ROOT / f"prepare_seed_{seed}.log")
    prepare_logger.info(
        "RUN/PREPARE | dataset=mixed | seed=%d | entity_trees=%d | dynotears_steps=%d",
        seed, entity_trees, dynotears_steps,
    )
    frame, _, evidence, removed, localization_edges = build_localization_features(
        DATA_PATH, causes, window=8, seed=seed, dynotears_steps=30
    )
    entity_features = np.c_[
        evidence["topology"], evidence["robust"], evidence["consistency"]
    ]
    train_mask = frame.split.eq("train").to_numpy()
    localizers = fit_entity_localizers(
        entity_features, frame, causes, train_mask, seed, entity_trees, prepare_logger
    )
    candidates = build_candidate_schema("mixed", causes)
    prepare_logger.info(
        "RUN/PREPARED | rows=%d | invisible_removed=%d | localization_edges=%d | candidates=%d",
        len(frame), removed, localization_edges, len(candidates),
    )

    # The causal series is independent of the window tensor.  Fit continuous
    # edge strengths once; only the declared sparsification quantile is varied.
    base_bundle = load_data(DATA_PATH, window_size=12, batch_size=64, seed=seed)
    _, dynotears_score = fit_dynotears(
        base_bundle.causal_series, steps=dynotears_steps, quantile=0.85, seed=seed
    )
    bundle_cache = {12: base_bundle}

    settings = configurations()
    if smoke:
        settings = settings[:2]
    rows: list[dict] = []
    for setting in settings:
        result_dir = seed_root / setting.run_id
        metric_path = result_dir / "metrics.csv"
        if metric_path.exists():
            rows.append(pd.read_csv(metric_path).iloc[0].to_dict())
            continue
        result_dir.mkdir(parents=True, exist_ok=True)
        # Epoch-by-epoch information is retained in a dedicated file; avoiding
        # a console handler keeps parallel runs readable without losing detail.
        logger = _quiet_logger(LOG_ROOT / f"{setting.run_id}_seed_{seed}.log")
        logger.info(
            "RUN/START | dataset=mixed | seed=%d | parameter=%s | value=%s | device=cpu",
            seed, setting.parameter, setting.value,
        )
        logger.info("RUN/PARAMS | %s", json.dumps(asdict(setting), sort_keys=True))
        started = time.perf_counter()
        bundle = bundle_cache.get(setting.window)
        if bundle is None:
            bundle = load_data(
                DATA_PATH, window_size=setting.window, batch_size=64, seed=seed
            )
            bundle_cache[setting.window] = bundle
        dynamic = _threshold_adjacency(dynotears_score, setting.dynotears_quantile)
        adjacency = ((bundle.physical_adjacency + dynamic) > 0).float().to(device)
        active_indices = [bundle.labels.index(cause) for cause in causes]

        # Reset the random stream before every controlled run so that within a
        # seed the changed hyperparameter is the only intended difference.
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = LineRCA(
            bundle.num_nodes,
            bundle.node_dim,
            bundle.consistency_dim,
            len(causes),
            hidden_dim=setting.hidden,
            heads=4,
            layers=2,
            icd_method="dm_bocd",
            icd_hazard=1.0 / setting.run_length,
        ).to(device)
        parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        args = SimpleNamespace(
            epochs=epochs,
            min_epochs=min_epochs,
            patience=patience,
            ablation="full",
        )
        logger.info(
            "MODEL/START | params=%d | window=%d | hidden=%d | run_length=%.3f | "
            "hazard=%.6f | dynotears_quantile=%.3f | fused_edges=%d",
            parameter_count, setting.window, setting.hidden, setting.run_length,
            1.0 / setting.run_length, setting.dynotears_quantile,
            int(adjacency.sum().item() - bundle.num_nodes),
        )
        final_epoch, best_epoch = train_type_model(
            model, bundle, adjacency, active_indices, args, device, logger
        )
        type_scores, _, row_indices = predict(
            model, bundle.loaders["test"], adjacency, device, "full"
        )
        test_rows = bundle.loaders["test"].dataset.rows.iloc[row_indices].reset_index(drop=True)
        localization_rows = _align_localization_rows(test_rows, frame)
        entity_scores = combine_type_and_entity_scores(
            entity_features[localization_rows], type_scores, causes, candidates, localizers
        )
        true_entities = extract_true_entities(test_rows)
        entity_labels = encode_entity_labels(true_entities, candidates)
        metrics = localization_metrics(entity_labels, entity_scores)
        result = {
            "Dataset": "mixed",
            "Seed": seed,
            "Parameter": setting.parameter,
            "Value": setting.value,
            "Window": setting.window,
            "Hidden": setting.hidden,
            "ICDRunLength": setting.run_length,
            "ICDHazard": 1.0 / setting.run_length,
            "DyNOTEARSQuantile": setting.dynotears_quantile,
            "DynamicEdges": int(dynamic.sum().item() - bundle.num_nodes),
            "FusedEdges": int(adjacency.sum().item() - bundle.num_nodes),
            "Epochs": final_epoch,
            "BestEpoch": best_epoch,
            "Params": parameter_count,
            "DmOmega": float(model.icd.omega.detach()),
            "Seconds": time.perf_counter() - started,
            **metrics,
        }
        pd.DataFrame([result]).to_csv(metric_path, index=False)
        (result_dir / "config.json").write_text(
            json.dumps({**asdict(setting), "seed": seed, "epochs": epochs,
                        "min_epochs": min_epochs, "patience": patience,
                        "dynotears_steps": dynotears_steps,
                        "entity_trees": entity_trees}, indent=2),
            encoding="utf-8",
        )
        logger.info("MODEL/FINAL | %s", json.dumps(result))
        logger.info("RUN/END | result_dir=%s", result_dir)
        rows.append(result)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--dynotears-steps", type=int, default=120)
    parser.add_argument("--entity-trees", type=int, default=400)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    payload = dict(
        epochs=args.epochs,
        min_epochs=args.min_epochs,
        patience=args.patience,
        dynotears_steps=args.dynotears_steps,
        entity_trees=args.entity_trees,
        torch_threads=args.torch_threads,
        smoke=args.smoke,
    )
    all_rows: list[dict] = []
    if args.workers == 1 or len(args.seeds) == 1:
        for seed in args.seeds:
            all_rows.extend(run_seed(seed=seed, **payload))
    else:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(args.seeds))) as pool:
            futures = {pool.submit(run_seed, seed=seed, **payload): seed for seed in args.seeds}
            for future in as_completed(futures):
                seed = futures[future]
                rows = future.result()
                all_rows.extend(rows)
                print(f"completed seed {seed}: {len(rows)} configurations", flush=True)
    results = pd.DataFrame(all_rows).sort_values(["Parameter", "Value", "Seed"])
    results.to_csv(RESULT_ROOT / "metrics_by_run.csv", index=False)
    manifest = {
        "dataset": "mixed",
        "evaluation_level": "entity localization",
        "seeds": args.seeds,
        "n_configurations_per_seed": len(configurations()) if not args.smoke else 2,
        "training": {
            "epochs": args.epochs,
            "min_epochs": args.min_epochs,
            "patience": args.patience,
        },
        "default": asdict(Configuration("default", 0.0)),
        "one_factor_at_a_time": True,
        "metrics_file": str(RESULT_ROOT / "metrics_by_run.csv"),
    }
    (RESULT_ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(RESULT_ROOT / "metrics_by_run.csv")


if __name__ == "__main__":
    main()

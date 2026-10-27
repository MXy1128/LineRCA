"""Leakage-controlled preprocessing for the pandapower RCA benchmark.

The raw simulator exposes both measurements and intervention metadata.  This
module keeps labels out of the observation matrix, estimates all reference
statistics on the training split, removes non-observable labelled events, and
builds one row per (snapshot, candidate root-cause) for ranking experiments.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


CAUSES = [
    "overload", "line_resistance_increase", "low_source_voltage",
    "poor_power_factor", "reverse_power_flow", "topology_reconfiguration",
    "uniform_scaling", "meter_bypass", "peak_period_theft",
    "mean_replacement", "intermittent_dropout", "meter_bias",
]
TECH = set(CAUSES[:6])
NTL = set(CAUSES[6:])
LABEL_COLUMNS = {
    "technical_root_cause", "technical_root_element_type",
    "technical_root_element_index", "technical_severity", "ntl_root_cause",
    "ntl_root_bus", "ntl_severity", "technical_anomaly", "ntl_anomaly",
    "loss_type", "root_cause", "event_id", "simulation_seed", "dataset_kind",
}


def _ratio(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.divide(a, b, out=np.zeros_like(a), where=np.abs(b) > 1e-9)


def _robust_fit(frame: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series]:
    x = frame[cols].replace([np.inf, -np.inf], np.nan)
    center = x.median()
    scale = (x.sub(center).abs().median() * 1.4826).clip(lower=1e-6)
    return center, scale


def _observable_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive measurement features; intervention labels are never inputs."""
    d = raw.copy()
    d["timestamp"] = pd.to_datetime(d["timestamp"])
    d = d.sort_values(["grid_id", "timestamp"]).reset_index(drop=True)
    f = pd.DataFrame(index=d.index)
    demand = (d["feeder_input_kw"] - d["technical_loss_kw"]).clip(lower=1e-6)
    f["loss_ratio"] = _ratio(d["technical_loss_kw"], demand)
    f["reactive_ratio"] = np.abs(_ratio(d["reactive_load_kvar"], demand))
    f["voltage_drop"] = (1.0 - d["min_voltage_pu"]).clip(lower=0)
    f["voltage_rise"] = (d["max_voltage_pu"] - 1.0).clip(lower=0)
    f["max_loading"] = d["max_line_loading_pct"] / 100.0
    f["grid_import_ratio"] = _ratio(d["external_grid_p_kw"], demand)
    f["meter_gap_ratio"] = _ratio(d["feeder_input_kw"] - d["technical_loss_kw"] - d["metered_load_kw"], demand).clip(0, 1)

    line_loss = [c for c in d if c.startswith("line_") and c.endswith("_loss_kw")]
    line_loading = [c for c in d if c.startswith("line_") and c.endswith("_loading_pct")]
    ties = [f"line_{i:02d}_in_service" for i in range(32, 37) if f"line_{i:02d}_in_service" in d]
    ll = d[line_loss].clip(lower=0).to_numpy(float)
    f["line_loss_concentration"] = _ratio(ll.max(axis=1), ll.sum(axis=1))
    f["loading_concentration"] = d[line_loading].max(axis=1).to_numpy(float) / (d[line_loading].mean(axis=1).to_numpy(float) + 1e-6)
    f["closed_ties"] = d[ties].sum(axis=1).to_numpy(float) if ties else 0.0

    meter_cols = sorted(c for c in d if c.startswith("bus_") and c.endswith("_metered_load_kw"))
    meters = d[meter_cols].to_numpy(float)
    # IEEE-33 demand shares are calibrated from normal training controls only.
    normal_train = d["split"].eq("train") & d["technical_root_cause"].eq("normal") & d["ntl_root_cause"].eq("normal")
    ref = meters[normal_train.to_numpy()].sum(axis=0)
    ref = ref / max(ref.sum(), 1e-9)
    active = ref > 1e-8
    expected = demand.to_numpy()[:, None] * ref[None, :]
    meter_ratio = np.divide(meters, expected, out=np.ones_like(meters), where=expected > 1e-8)
    active_ratio = meter_ratio[:, active]
    f["min_meter_ratio"] = np.clip(active_ratio.min(axis=1), 0, 1.5)
    f["meter_ratio_std"] = np.clip(active_ratio.std(axis=1), 0, 2)
    f["zero_meter_count"] = (active_ratio < .02).sum(axis=1).astype(float)
    f["meter_min_share"] = np.clip(active_ratio.min(axis=1), 0, 1)
    f["peak_indicator"] = d["timestamp"].dt.hour.between(17, 22).astype(float)

    key = d["grid_id"]
    for col in ["meter_gap_ratio", "min_meter_ratio", "loss_ratio", "max_loading"]:
        f[f"delta_{col}"] = f[col].groupby(key).diff().abs().fillna(0)
        f[f"rollstd_{col}"] = f[col].groupby(key).transform(lambda s: s.rolling(6, min_periods=2).std()).fillna(0)
    f.insert(0, "_row", d.index)
    return d, f.replace([np.inf, -np.inf], np.nan).fillna(0)


SIGNATURES = {
    "overload": ["max_loading", "loading_concentration", "loss_ratio", "delta_max_loading"],
    "line_resistance_increase": ["loss_ratio", "line_loss_concentration", "voltage_drop"],
    "low_source_voltage": ["voltage_drop", "loss_ratio", "grid_import_ratio"],
    "poor_power_factor": ["reactive_ratio", "loss_ratio", "voltage_drop"],
    "reverse_power_flow": ["grid_import_ratio", "voltage_rise", "loss_ratio"],
    "topology_reconfiguration": ["closed_ties", "line_loss_concentration", "loading_concentration"],
    "uniform_scaling": ["meter_gap_ratio", "min_meter_ratio", "meter_ratio_std", "rollstd_min_meter_ratio"],
    "meter_bypass": ["meter_gap_ratio", "min_meter_ratio", "zero_meter_count"],
    "peak_period_theft": ["meter_gap_ratio", "min_meter_ratio", "peak_indicator", "delta_meter_gap_ratio"],
    "mean_replacement": ["meter_gap_ratio", "min_meter_ratio", "meter_ratio_std", "rollstd_meter_gap_ratio"],
    "intermittent_dropout": ["zero_meter_count", "min_meter_ratio", "delta_min_meter_ratio", "rollstd_min_meter_ratio"],
    "meter_bias": ["meter_gap_ratio", "min_meter_ratio", "meter_ratio_std"],
}


@dataclass
class ProcessReport:
    rows_in: int
    rows_out: int
    removed_invisible: int
    feature_columns: list[str]
    leakage_columns_excluded: list[str]


def build_candidate_table(raw: pd.DataFrame, dataset: str, min_severity: float = .02) -> tuple[pd.DataFrame, ProcessReport]:
    d, obs = _observable_features(raw)
    invisible = ((d["technical_root_cause"] != "normal") & (d["technical_severity"] <= min_severity)) | ((d["ntl_root_cause"] != "normal") & (d["ntl_severity"] <= min_severity))
    keep = ~invisible
    d, obs = d.loc[keep].reset_index(drop=True), obs.loc[keep].reset_index(drop=True)
    feature_cols = [c for c in obs if c != "_row"]
    normal_train = d["split"].eq("train") & d["technical_root_cause"].eq("normal") & d["ntl_root_cause"].eq("normal")
    center, scale = _robust_fit(obs.loc[normal_train], feature_cols)
    z = obs[feature_cols].sub(center).div(scale).clip(-12, 12)

    rows = []
    train = d["split"].eq("train")
    for cause in CAUSES:
        cols = SIGNATURES[cause]
        positive = train & (d["technical_root_cause"].eq(cause) if cause in TECH else d["ntl_root_cause"].eq(cause))
        # A train-only robust cause prototype turns heterogeneous physical units
        # into a comparable candidate compatibility score.
        proto = z.loc[positive, cols].median() if positive.any() else pd.Series(0.0, index=cols)
        spread = (z.loc[positive, cols].sub(proto).abs().median() * 1.4826).clip(lower=.35) if positive.any() else pd.Series(1.0, index=cols)
        distance = z[cols].sub(proto).div(spread).clip(-8, 8).pow(2).mean(axis=1)
        compatibility = np.exp(-0.5 * distance)
        anomaly = z[cols].abs().mean(axis=1).clip(0, 8) / 8
        family_signal = obs["loss_ratio"] if cause in TECH else obs["meter_gap_ratio"]
        part = pd.DataFrame({
            "scenario_id": dataset + "::" + d.index.astype(str),
            "split": d["split"].replace({"validation": "val"}),
            "candidate_object_type": "technical" if cause in TECH else "non_technical",
            "candidate_fault_type": cause,
            "type_prior": 1.0 / len(CAUSES),
            "object_anomaly_score": compatibility,
            "spatial_prior": anomaly,
            "type_anomaly_score": family_signal,
        })
        part["candidate_score"] = .70 * compatibility + .20 * anomaly + .10 * family_signal
        truth = d["technical_root_cause"].eq(cause) if cause in TECH else d["ntl_root_cause"].eq(cause)
        part["is_true_root"] = truth.astype("int8")
        rows.append(part)
    out = pd.concat(rows, ignore_index=True)
    report = ProcessReport(len(raw), len(d), int(invisible.sum()), feature_cols, sorted(LABEL_COLUMNS))
    return out, report


def save_processed(raw_path: Path, output_path: Path, dataset: str) -> ProcessReport:
    candidates, report = build_candidate_table(pd.read_csv(raw_path), dataset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(output_path, index=False)
    output_path.with_suffix(".report.json").write_text(json.dumps(report.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def prepare_raw_samples(raw: pd.DataFrame, min_severity: float = .02):
    """Return one feature row and one 12-D multi-hot target per raw snapshot.

    Unlike :func:`build_candidate_table`, this is the preferred direct-model
    interface: it never expands a snapshot into candidate rows.
    """
    d, obs = _observable_features(raw)
    invisible_tech = (d["technical_root_cause"] != "normal") & (d["technical_severity"] <= min_severity)
    invisible_ntl = (d["ntl_root_cause"] != "normal") & (d["ntl_severity"] <= min_severity)
    keep = ~(invisible_tech | invisible_ntl)
    d, obs = d.loc[keep].reset_index(drop=True), obs.loc[keep].reset_index(drop=True)
    feature_cols = [c for c in obs.columns if c != "_row"]
    train = d["split"].eq("train")
    center, scale = _robust_fit(obs.loc[train], feature_cols)
    x = obs[feature_cols].sub(center).div(scale).clip(-12, 12).fillna(0)
    y = np.zeros((len(d), len(CAUSES)), dtype=np.float32)
    for j, cause in enumerate(CAUSES):
        source = d["technical_root_cause"] if cause in TECH else d["ntl_root_cause"]
        y[:, j] = source.eq(cause).to_numpy(np.float32)
    meta = d[["timestamp", "grid_id", "split", "technical_root_cause", "ntl_root_cause"]].copy()
    meta["split"] = meta["split"].replace({"validation": "val"})
    report = {
        "rows_in": int(len(raw)), "rows_out": int(len(d)),
        "removed_invisible": int((~keep).sum()), "n_features": len(feature_cols),
        "feature_columns": feature_cols, "target_columns": CAUSES,
        "normal_rows": int((y.sum(axis=1) == 0).sum()),
        "single_root_rows": int((y.sum(axis=1) == 1).sum()),
        "mixed_root_rows": int((y.sum(axis=1) == 2).sum()),
        "scaler_fit_split": "train_only",
    }
    return x, y, meta, report

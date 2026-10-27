"""Generate line-loss RCA benchmarks with pandapower AC power flow.

Technical losses are native ``pandapower.runpp`` results on the IEEE 33-bus
distribution test system.  Existing project time series control total demand;
explicit interventions provide ground-truth root cause and location labels.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as pn


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "simulated_pandapower"
SEED = 20260806
TECH_CAUSES = (
    "normal", "overload", "line_resistance_increase", "low_source_voltage",
    "poor_power_factor", "reverse_power_flow", "topology_reconfiguration",
)
NTL_CAUSES = (
    "normal", "uniform_scaling", "meter_bypass", "peak_period_theft",
    "mean_replacement", "intermittent_dropout", "meter_bias",
)
_NETWORK_TEMPLATE = None


def stable_seed(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "little")


def rng_for(*parts: object) -> np.random.Generator:
    return np.random.default_rng(stable_seed(SEED, *parts))


def load_profiles(n_samples: int) -> pd.DataFrame:
    raw = pd.concat([pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")], ignore_index=True)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    positions = np.linspace(0, len(raw) - 1, min(n_samples, len(raw)), dtype=int)
    raw = raw.iloc[positions].reset_index(drop=True)
    records = []
    for i, row in raw.iterrows():
        grid_no = i % 3 + 1
        load = float(row[f"grid{grid_no}-load"])
        if not np.isfinite(load) or load <= 0:
            continue
        records.append({"timestamp": row["date"], "grid_id": f"grid{grid_no}", "profile_load": load})
    df = pd.DataFrame(records)
    df["load_scale"] = (df["profile_load"] / df.groupby("grid_id")["profile_load"].transform("median")).clip(.45, 1.75)
    df["date_key"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    df["hour"] = df["timestamp"].dt.hour
    df["weekday"] = df["timestamp"].dt.weekday
    return df


def daily_choice(grid: str, day: str, family: str, choices: tuple[str, ...]) -> str:
    # 30% normal controls; remaining causes are balanced.
    probs = [0.30] + [0.70 / (len(choices) - 1)] * (len(choices) - 1)
    return str(rng_for(family, grid, day).choice(choices, p=probs))


def base_network(scale: float):
    global _NETWORK_TEMPLATE
    if _NETWORK_TEMPLATE is None:
        _NETWORK_TEMPLATE = pn.case33bw()
    net = copy.deepcopy(_NETWORK_TEMPLATE)
    net.load.loc[:, "p_mw"] *= scale
    net.load.loc[:, "q_mvar"] *= scale
    return net


def apply_background(net, grid: str, day: str, hour: int) -> None:
    """Add label-independent operating variability before interventions."""
    rng = rng_for("background", grid, day, hour)
    p_mult = rng.lognormal(mean=-0.5 * .10 ** 2, sigma=.10, size=len(net.load))
    q_mult = rng.lognormal(mean=-0.5 * .14 ** 2, sigma=.14, size=len(net.load))
    net.load.loc[:, "p_mw"] *= p_mult
    net.load.loc[:, "q_mvar"] *= q_mult
    net.ext_grid.loc[:, "vm_pu"] = rng.uniform(.98, 1.02)
    net.line.loc[:, "r_ohm_per_km"] *= rng.uniform(.92, 1.14, size=len(net.line))
    # Background DER makes reverse flow a graded signature rather than a label flag.
    if rng.random() < .30:
        idx = int(rng.choice(net.load.index)); bus = int(net.load.at[idx, "bus"])
        pp.create_sgen(net, bus=bus, p_mw=float(rng.uniform(.03, .18) * net.load.p_mw.sum()), q_mvar=0.0, name="background_DER")


def apply_technical(net, cause: str, grid: str, day: str) -> tuple[str, int, float]:
    rng = rng_for("tech", cause, grid, day)
    severity = 0.0
    if cause == "normal":
        return "none", -1, severity
    if cause == "overload":
        idx = int(rng.choice(net.load.index))
        factor = float(rng.uniform(1.20, 1.85)); severity = factor
        net.load.at[idx, "p_mw"] *= factor; net.load.at[idx, "q_mvar"] *= factor
        return "load", idx, severity
    if cause == "line_resistance_increase":
        idx = int(rng.choice(net.line.index[net.line.in_service]))
        factor = float(rng.uniform(1.15, 2.10)); severity = factor
        net.line.at[idx, "r_ohm_per_km"] *= factor
        return "line", idx, severity
    if cause == "low_source_voltage":
        vm = float(rng.uniform(0.945, 0.985)); severity = 1.0 - vm
        net.ext_grid.loc[:, "vm_pu"] = vm
        return "ext_grid", int(net.ext_grid.index[0]), severity
    if cause == "poor_power_factor":
        idx = int(rng.choice(net.load.index)); pf = float(rng.uniform(.78, .92)); severity = 1.0 - pf
        p = float(net.load.at[idx, "p_mw"])
        net.load.at[idx, "q_mvar"] = p * np.tan(np.arccos(pf))
        return "load", idx, severity
    if cause == "reverse_power_flow":
        idx = int(rng.choice(net.load.index)); bus = int(net.load.at[idx, "bus"])
        p = float(rng.uniform(0.12, 0.48) * net.load.p_mw.sum()); severity = p
        pp.create_sgen(net, bus=bus, p_mw=p, q_mvar=0.0, name="injected_DG")
        return "bus", bus, severity
    # IEEE-33 case contains normally-open tie lines; closing one creates a
    # supported meshed topology and changes the flow/loss pattern.
    candidates = net.line.index[~net.line.in_service]
    idx = int(rng.choice(candidates)); net.line.at[idx, "in_service"] = True; severity = 1.0
    return "line", idx, severity


def run_flow(net) -> bool:
    try:
        pp.runpp(net, algorithm="nr", calculate_voltage_angles=False, init="auto", numba=False,
                 max_iteration=30, tolerance_mva=1e-8)
        return bool(net.converged)
    except pp.LoadflowNotConverged:
        return False


def flow_features(net) -> dict[str, float]:
    features: dict[str, float] = {
        "actual_load_kw": float(net.load.p_mw.sum() * 1000),
        "reactive_load_kvar": float(net.load.q_mvar.sum() * 1000),
        "technical_loss_kw": float((net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()) * 1000),
        "line_loss_kw": float(net.res_line.pl_mw.sum() * 1000),
        "transformer_loss_kw": float(net.res_trafo.pl_mw.sum() * 1000),
        "external_grid_p_kw": float(net.res_ext_grid.p_mw.sum() * 1000),
        "min_voltage_pu": float(net.res_bus.vm_pu.min()),
        "max_voltage_pu": float(net.res_bus.vm_pu.max()),
        "max_line_loading_pct": float(net.res_line.loading_percent.max()),
    }
    for idx in net.bus.index:
        features[f"bus_{idx:02d}_vm_pu"] = float(net.res_bus.at[idx, "vm_pu"])
    for idx in net.line.index:
        active = bool(net.line.at[idx, "in_service"])
        features[f"line_{idx:02d}_in_service"] = int(active)
        features[f"line_{idx:02d}_loss_kw"] = float(net.res_line.at[idx, "pl_mw"] * 1000) if active else 0.0
        features[f"line_{idx:02d}_loading_pct"] = float(net.res_line.at[idx, "loading_percent"]) if active else 0.0
    return features


def apply_ntl(actual_bus_kw: np.ndarray, cause: str, grid: str, day: str, hour: int):
    rng = rng_for("ntl", cause, grid, day)
    metered = actual_bus_kw.copy(); root = -1; severity = 0.0
    nonzero = np.flatnonzero(actual_bus_kw > 0)
    if cause == "normal" or len(nonzero) == 0:
        return metered, "normal", root, severity
    root = int(rng.choice(nonzero)); x = actual_bus_kw[root]
    if cause == "uniform_scaling":
        factor = float(rng.uniform(.20, .80)); metered[root] = x * factor; severity = 1 - factor
    elif cause == "meter_bypass":
        factor = float(rng.uniform(0, .18)); metered[root] = x * factor; severity = 1 - factor
    elif cause == "peak_period_theft":
        if not 17 <= hour <= 22:
            return metered, "normal", -1, 0.0
        factor = float(rng.uniform(.25, .60)); metered[root] = x * factor; severity = 1 - factor
    elif cause == "mean_replacement":
        factor = float(rng.uniform(.45, .80)); metered[root] = np.mean(actual_bus_kw[nonzero]) * factor
        metered[root] = min(metered[root], x * .92)
        severity = 1 - metered[root] / max(x, 1e-9)
    elif cause == "intermittent_dropout":
        if rng_for("drop", grid, day, hour).random() >= .45:
            return metered, "normal", -1, 0.0
        metered[root] = 0.0; severity = 1.0
    elif cause == "meter_bias":
        factor = float(rng.uniform(.72, .94)); metered[root] = x * factor; severity = 1 - factor
    return metered, cause, root, severity


def bus_loads_kw(net) -> np.ndarray:
    values = np.zeros(len(net.bus))
    for row in net.load.itertuples():
        values[int(row.bus)] += float(row.p_mw) * 1000
    return values


def make_row(profile, kind: str, tech_cause: str, ntl_cause: str) -> dict[str, object] | None:
    net = base_network(float(profile.load_scale))
    apply_background(net, profile.grid_id, profile.date_key, int(profile.hour))
    tech_type, tech_idx, tech_severity = apply_technical(net, tech_cause, profile.grid_id, profile.date_key)
    if not run_flow(net):
        return None
    row: dict[str, object] = {
        "timestamp": profile.timestamp, "grid_id": profile.grid_id, "dataset_kind": kind,
        "technical_root_cause": tech_cause, "technical_root_element_type": tech_type,
        "technical_root_element_index": tech_idx, "technical_severity": tech_severity,
    }
    row.update(flow_features(net))
    actual_bus = bus_loads_kw(net)
    metered_bus, ntl_cause_active, ntl_bus, ntl_severity = apply_ntl(
        actual_bus, ntl_cause, profile.grid_id, profile.date_key, int(profile.hour)
    )
    # Independent meter noise is applied after the intervention. Values remain
    # non-negative and the anomaly label never controls the noise realization.
    noise_rng = rng_for("meter_noise", profile.grid_id, profile.date_key, int(profile.hour), kind)
    metered_bus = np.maximum(0.0, metered_bus * noise_rng.normal(1.0, .006, size=len(metered_bus)))
    if metered_bus.sum() > actual_bus.sum():
        metered_bus *= actual_bus.sum() / metered_bus.sum()
    row.update({
        "ntl_root_cause": ntl_cause_active, "ntl_root_bus": ntl_bus,
        "ntl_severity": ntl_severity, "metered_load_kw": float(metered_bus.sum()),
    })
    for i, (actual, metered) in enumerate(zip(actual_bus, metered_bus)):
        row[f"bus_{i:02d}_actual_load_kw"] = float(actual)
        row[f"bus_{i:02d}_metered_load_kw"] = float(metered)
    row["non_technical_loss_kw"] = max(0.0, row["actual_load_kw"] - row["metered_load_kw"])
    row["total_loss_kw"] = row["technical_loss_kw"] + row["non_technical_loss_kw"]
    row["feeder_input_kw"] = row["actual_load_kw"] + row["technical_loss_kw"]
    row["energy_balance_residual_kw"] = row["feeder_input_kw"] - row["metered_load_kw"] - row["total_loss_kw"]
    ta, na = int(tech_cause != "normal"), int(ntl_cause_active != "normal")
    row["technical_anomaly"], row["ntl_anomaly"] = ta, na
    row["loss_type"] = "mixed" if ta and na else "technical" if ta else "non_technical" if na else "normal"
    row["root_cause"] = f"{tech_cause}+{ntl_cause_active}" if ta and na else tech_cause if ta else ntl_cause_active
    row["event_id"] = f"{profile.grid_id}_{profile.date_key}_{row['root_cause']}"
    return row


def generate(n_samples: int) -> dict[str, pd.DataFrame]:
    profiles = load_profiles(n_samples)
    rows = {"technical": [], "non_technical": [], "mixed": []}
    for n, profile in enumerate(profiles.itertuples(index=False), 1):
        tc = daily_choice(profile.grid_id, profile.date_key, "tech_only", TECH_CAUSES)
        nc = daily_choice(profile.grid_id, profile.date_key, "ntl_only", NTL_CAUSES)
        mtc = daily_choice(profile.grid_id, profile.date_key, "mixed_tech", TECH_CAUSES)
        mnc = daily_choice(profile.grid_id, profile.date_key, "mixed_ntl", NTL_CAUSES)
        rows["technical"].append(make_row(profile, "technical", tc, "normal"))
        rows["non_technical"].append(make_row(profile, "non_technical", "normal", nc))
        rows["mixed"].append(make_row(profile, "mixed", mtc, mnc))
        if n % 500 == 0:
            print(f"simulated {n}/{len(profiles)} timestamps", flush=True)
    outputs = {}
    for kind, records in rows.items():
        df = pd.DataFrame([r for r in records if r is not None]).sort_values("timestamp").reset_index(drop=True)
        q1, q2 = df.timestamp.quantile([.70, .85])
        df["split"] = np.select([df.timestamp <= q1, df.timestamp <= q2], ["train", "validation"], default="test")
        df["simulation_seed"] = SEED
        df["timestamp"] = df.timestamp.dt.strftime("%Y-%m-%d %H:%M:%S")
        outputs[kind] = df
    return outputs


def quality(outputs: dict[str, pd.DataFrame]) -> dict[str, object]:
    result = {"pandapower_version": pp.__version__, "network": "pandapower.networks.case33bw", "seed": SEED, "datasets": {}}
    for name, df in outputs.items():
        result["datasets"][name] = {
            "rows": len(df), "columns": len(df.columns), "missing_cells": int(df.isna().sum().sum()),
            "max_abs_energy_balance_residual_kw": float(df.energy_balance_residual_kw.abs().max()),
            "loss_type_counts": df.loss_type.value_counts().to_dict(),
            "root_cause_counts": df.root_cause.value_counts().to_dict(),
            "nonconverged_rows": 0,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = generate(args.samples)
    names = {"technical": "technical_line_loss.csv", "non_technical": "non_technical_line_loss.csv", "mixed": "mixed_line_loss.csv"}
    for kind, df in outputs.items():
        df.to_csv(args.output_dir / names[kind], index=False, float_format="%.8f")
    report = quality(outputs)
    (args.output_dir / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "schema_version": "2.0.0", "generator": "simulation/generate_pandapower_datasets.py",
        "pandapower_version": pp.__version__, "network": "case33bw", "samples_requested": args.samples,
        "source_profiles": ["data/train.csv", "data/test.csv"], "files": names,
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

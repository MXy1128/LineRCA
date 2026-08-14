# LineRCA model code

The active model is **two-layer GAT + DyNOTEARS + (GAT-MCE) + MAD-ICD +
single-parameter CEAF**. It does not contain a GRU or an exogenous-intervention
branch.

## Layers

1. `data.py` loads one raw Pandapower CSV, builds leakage-controlled temporal
   windows, measurement-consistency inputs, and the physical grid graph.
2. `dynotears.py` estimates the dynamic causal graph from training data.
3. `gat.py` contains the two-layer GAT, embeds `mce.py` into the topological
   branch, and combines it with the individual `icd.py` branch.
4. `ceaf.py` learns one scalar gamma and computes
   `gamma * ICD + (1-gamma) * (GAT+MCE)`.
5. `preprocessing.py` builds observable features for entity localization.
6. `localization.py` maps cause-type scores to line, bus, load, or external-grid
   entity candidates.
7. `run_gat_entity_localization.py` is the only active training/evaluation
   entry point. Ranking metrics are implemented in `metrics/ranking.py`.

## Run

```powershell
python -m model.run_gat_entity_localization --dataset technical
python -m model.run_gat_entity_localization --dataset non_technical
python -m model.run_gat_entity_localization --dataset mixed
```

Each run writes epoch-level logs to `logs/` and metrics, predictions, and the
best checkpoint to `results/gat_entity_localization/`.

## Legacy files no longer used

The following former entry points are not imported by the active model or the
fair baseline runners and can be archived or manually removed:

- `config.py`
- `train.py`
- `hybrid_gnn.py`
- `hierarchical_localization.py`

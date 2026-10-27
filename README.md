# LineRCA

LineRCA is a root-cause analysis pipeline for technical, non-technical, and
mixed electrical line-loss scenarios. The active model combines a two-layer
graph attention network (GAT), DyNOTEARS causal discovery, measurement
consistency evidence (MCE), MAD-based individual causal detection (ICD), and
single-parameter causal evidence adaptive fusion (CEAF).

## Repository contents

- `model/`: model, causal discovery, preprocessing, training, and entity-level
  localization code.
- `metrics/`: ranking metrics and comparison utilities.
- `simulation/`: Pandapower-based dataset generation and repair scripts.
- `analysis/`: sensitivity, ablation, loss-curve, and figure-analysis scripts.
- `data/`: raw and simulated datasets used by the code.

Manuscript files, temporary renders, IDE settings, logs, generated results,
and vendored third-party baseline repositories are intentionally excluded.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run the active model

From the repository root:

```bash
python -m model.run_gat_entity_localization --dataset technical
python -m model.run_gat_entity_localization --dataset non_technical
python -m model.run_gat_entity_localization --dataset mixed
```

Runs write logs to `logs/` and generated metrics, predictions, and checkpoints
to `results/gat_entity_localization/`. These runtime artifacts are ignored by
Git.

More implementation details are available in [`model/README.md`](model/README.md).

# Power Prediction on Alibaba GenTD26 Cluster Telemetry (Corrected)

One-step and multi-step power prediction using Transformer models on the Alibaba GenTD26 generative AI cluster dataset.

This corrected iteration uses the strict Fan et al. utilization-to-power conversion:

```text
P = n_gpus * (50 + 250 * gpu_util) / 1000
```

The idle term is charged to every powered GPU. `active_pod_ratio` remains a model feature, but it is not used in the power conversion. The 50/25/25 split remains the detailed stress-test environment, while the split-study and multi-step scripts also report 60/20/20 and 70/15/15 comparisons.

## Quick Start (Executables)

All operational scripts have been safely decoupled from backend architecture logic and securely placed in the `executables/` directory.

To fully reproduce this paper's analytics, run the scripts in the following exact order:

```bash
cd algorithmic_core/split_50_25_25_final_corrected/executables/

# 1. Run one-step sweep across data splits (populates cache & initial metrics)
python3 split_study.py

# 2. Extract final one-step 50/25/25 metrics & visualizations
python3 generate_main_results.py

# 3. Simulate forward horizon averaging (h=1 to h=24)
python3 multistep_averaged.py

# 4. Generate the multi-step power trajectory graph using averaged JSON logs
python3 generate_power_figures.py

# 5. Extract workload condition analytics (Volatile vs Calm transitions)
python3 locality_analysis.py
```

Subsequent runs natively pull from the `models/` cache repository to instantly regenerate graphics/markdown tables without triggering lengthy GPU training sequences.

## Clearing the Model Cache

Trained models `.pt` weights and `.joblib` feature scalers are proactively preserved. To flush the repository and retrain the 30+ architectural parameters manually from scratch (e.g. testing new configurations):

```bash
cd algorithmic_core/split_50_25_25_final_corrected
rm -rf models/onestep models/multistep
```

## Primary Outputs (`results/`)

Running the executables automatically constructs all referenced items natively inside the `results/` folder for immediate consumption.

| Generated Academic Target | Content |
| --- | --- |
| `power_table.md` | Combined one-step power metrics and exogenous-telemetry ablation table |
| `multistep_power_table.md` | Multi-step window-averaged power results across horizons |
| `locality_table.md` | Condition-specific tracking wins/losses at h=15 |
| `fig_onestep_power_overlay.png` | Graph: 10-Minute Rolling Error overlay tracking spikes |
| `fig_multistep_power_splits.png` | Graph: Trajectory error metrics degrading over forward averages |
| `fig_locality_best_worst.png` | Graph: Qualitative breakdown of Burst Interception metrics |

## Structural Core (`core/`)

Background calculation variables bounding data preparation are stored permanently under `core/`:

- **`pipeline.py`**: Isolates exactly 27 sequence features (including fractional differences, trailing telemetry metrics, QPS tracking). Converts GPU utilization to power with the strict Fan et al. idle-plus-dynamic linear model.
- **`models.py`**: Establishes 32-hidden-dimension PyTorch neural models.
- **`train.py`**: Orchestrates Gradient Clipping, Huber loss structures mapping residual variances, and Cosine Annealing logic.
- **`model_cache.py`**: Interacts dynamically with scaling features saving states continuously.

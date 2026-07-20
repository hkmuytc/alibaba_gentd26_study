# Power Prediction on Alibaba GenTD26 Cluster Telemetry

One-step and multi-step power prediction using Transformer models on the Alibaba GenTD26 generative AI cluster dataset.

## Quick Start

```bash
# 1. Run one-step sweep (trains 36 models, populates cache)
python3 split_study.py

# 2. Generate one-step main results (loads best model from cache)
python3 generate_main_results.py

# 3. Run multi-step sweep (trains 33 models, populates cache)
python3 multistep_averaged.py

# 4. Run locality analysis (loads all from cache)
python3 locality_analysis.py
```

Subsequent runs load all models from cache — **0 models trained**, only figures and tables regenerated.

## Model Cache

Trained models are saved in `models/` and reused across scripts:

```
models/
├── onestep/           ← 36 models (4 architectures × 3 windows × 3 splits)
│   ├── transformer_w60_70-15-15.pt   ← used by generate_main_results.py
│   └── metadata.json
└── multistep/         ← 33 models (11 horizons × 3 splits)
    ├── transformer_h12_70-15-15.pt   ← reused by locality_analysis.py
    └── metadata.json
```

To clear cache and retrain from scratch:
```bash
rm -rf models/onestep models/multistep
```

## Scripts

| Script | Purpose | Models | Outputs |
|--------|---------|--------|---------|
| `split_study.py` | One-step sweep: 4 architectures × 3 windows × 3 splits | 36 | `fig_onestep_split_*.png`, `split_study_tables.md` |
| `generate_main_results.py` | One-step main results (best config) | 1 (from cache) | `fig_onestep_power_overlay.png`, `power_table.md` |
| `multistep_averaged.py` | Multi-step averaged: h=1–24 × 3 splits | 33 | `fig_multistep_power_*.png`, `multistep_power_table.md` |
| `locality_analysis.py` | Locality deep-dive at h=12 | 11 (from cache) | `fig_multistep_power_growth.png`, `fig_locality_*.png`, `locality_table.md` |
| `generate_power_figures.py` | Multi-step splits comparison | 0 | `fig_multistep_power_splits.png` |

## Core Modules

| Module | Purpose |
|--------|---------|
| `pipeline.py` | Data loading, feature engineering, power estimation (Fan et al. 2007), train/val/test split |
| `models.py` | LSTM, GRU, Transformer, CNN-LSTM, Transformer-HistoryOnly architectures |
| `train.py` | Training loop (Huber loss, cosine LR, early stopping), evaluation, baselines |
| `model_cache.py` | Model save/load/compare for reproducibility |

## Figures

### One-Step
- `fig_onestep_power_overlay.png` — Actual vs predicted power with rolling MAE
- `fig_onestep_split_mae.png` — Power MAE across all configurations (includes history-only ablation)
- `fig_onestep_split_r2.png` — R² across all configurations
- `fig_onestep_ablation.png` — Ablation: full vs history-only Transformer vs persistence

### Multi-Step
- `fig_multistep_power_growth.png` — Power MAE, improvement %, and R² across h=1–24
- `fig_multistep_power_splits.png` — Power improvement across 3 data splits

### Locality (h=12)
- `fig_locality_per_sample_advantage.png` — Per-sample advantage with power trajectory
- `fig_locality_best_worst.png` — Best/worst cases ranked by model advantage

## Tables

| File | Content |
|------|---------|
| `power_table.md` | One-step: Transformer vs Persistence (MAE, RMSE, R², MAPE) |
| `ablation_table.md` | Ablation: full vs history-only Transformer vs persistence |
| `split_study_tables.md` | One-step sweep: all models × windows × splits |
| `multistep_power_table.md` | Multi-step: h=1–24 power MAE, R², improvement |
| `locality_table.md` | Locality: regime breakdown at h=12 (volatility, trend) |

## Configuration

- **Dataset**: Alibaba GenTD26 cluster telemetry, 1-minute bins, ~929 samples
- **Default split**: 70/15/15 (train/val/test), temporal ordering preserved
- **Window size**: 60 minutes
- **Features**: 27 (gpu_util, active_pod_ratio, qps_gen, qps_api, gpu_mem_frac, lags, rolling stats, fractional differencing, time-of-day)
- **Target**: gpu_util (one-step: y[t+1]; multi-step: mean(y[t+1..t+h]))
- **Prediction mode**: Residual (model predicts Δ from last observed value)
- **Power conversion**: P(kW) = n_gpus × (50 + 250 × gpu_util × active_pod_ratio) / 1000 (Fan et al. 2007)

# Algorithmic Core — Power Forecasting Experiments

This folder contains the full machine learning pipeline for predicting data center GPU power demand from the **Alibaba GenTD26 generative AI cluster trace**.

The dataset records how hundreds of GPU-equipped containers handle real generative AI workloads (image generation, text-to-image, LLM inference) over a 15-hour window. Our goal is to forecast the cluster's aggregate power consumption one or more minutes ahead, using only GPU utilization, memory usage, and request-rate telemetry.

Two experiment configurations are provided. They share identical code and models — the difference is which data split is treated as the primary evaluation environment.

| Folder | Primary Split | Purpose |
|---|---|---|
| `split_50_25_25_final/` | 50/25/25 | **Canonical.** The paper's main stress test — less training data, more test data. |
| `split_70_15_15/` | 70/15/15 | **Comparison.** A conventional ML split with more training data. |

Both folders internally evaluate all three splits (50/25/25, 60/20/20, 70/15/15) during the sweep scripts.

## What the Experiments Test

**Experiment 1 — One-Step Prediction**
Can a model predict next minute's GPU utilization better than simply repeating the last observed value? We compare four architectures (LSTM, GRU, Transformer, CNN-LSTM) across three window sizes (15, 30, 60 minutes) and three data splits — training 36 models total. The Transformer with a 60-minute window consistently outperforms all others.

**Experiment 2 — Multi-Step Averaged Forecasting**
Can the model forecast average power demand over increasingly distant horizons (1–24 minutes ahead)? We train 33 Transformer models (11 horizons × 3 splits) and measure how accuracy degrades as predictions reach further into the future.

**Experiment 3 — Locality Analysis**
When does the model win versus lose? We break down per-sample prediction advantage by workload regime: calm versus volatile utilization, and rising versus falling versus stable trends.

## How the Pipeline Works

```
Raw CSVs (data/raw/)
  │  GPU duty cycle, GPU memory, QPS (queries per second)
  ▼
Aggregation (60-second bins)
  │  Cluster-level: mean GPU util, std, active pods, memory fraction, QPS
  ▼
Feature Engineering (27 features)
  │  Lags, rolling means/stds, rate of change, fractional differencing,
  │  pod ratios, QPS telemetry, memory fraction, time-of-day (sin/cos)
  ▼
Sliding Window (60 steps) → Residual Prediction Mode
  │  Model predicts the *change* from the last value, not the absolute value
  ▼
Neural Network Training
  │  LSTM, GRU, Transformer, CNN-LSTM
  ▼
Power Conversion (Fan et al. 2007)
     P(kW) = n_gpus × (50 + 250 × gpu_util) / 1000
```

## Models

All models use 32 hidden dimensions and take 27 input features:

| Model | Architecture |
|---|---|
| LSTMForecaster | LSTM + LayerNorm + Linear |
| GRUForecaster | GRU + LayerNorm + Linear |
| TransformerForecaster | Sinusoidal PE + 2-layer Transformer Encoder (4 heads) + LayerNorm + Linear |
| CNNLSTMForecaster | 2× dilated Conv1D + GRU + LayerNorm + Linear |
| TransformerHistoryOnly | Same Transformer but only GPU utilization features (ablation, 10 features) |

## Reproducing All Results

Run the five scripts below **in order** from inside the experiment folder. All paths shown use `split_50_25_25_final/` (the canonical folder); substitute `split_70_15_15/` if you want the comparison results instead.

**macOS / Linux:**

```bash
cd algorithmic_core/split_50_25_25_final

# 1. One-step sweep: 4 models × 3 windows × 3 splits = 36 models trained and cached
python3 executables/split_study.py

# 2. One-step main results for best config (Transformer, w=60, 50/25/25) + ablation
python3 executables/generate_main_results.py

# 3. Multi-step sweep: 11 horizons × 3 splits = 33 models trained and cached
python3 executables/multistep_averaged.py

# 4. Multi-step comparison figure (reads the JSON produced by step 3)
python3 executables/generate_power_figures.py

# 5. Locality analysis: when does the model win or lose? (h=15 deep dive)
python3 executables/locality_analysis.py
```

**Windows (PowerShell, Python 3.11 and earlier):**

```powershell
cd algorithmic_core\split_50_25_25_final

python executables\split_study.py
python executables\generate_main_results.py
python executables\multistep_averaged.py
python executables\generate_power_figures.py
python executables\locality_analysis.py
```

**Windows (PowerShell, Python 3.12+):**

```powershell
cd algorithmic_core\split_50_25_25_final

py executables\split_study.py
py executables\generate_main_results.py
py executables\multistep_averaged.py
py executables\generate_power_figures.py
py executables\locality_analysis.py
```

On the first run, all models are trained from scratch (this takes time). On subsequent runs, models are loaded from the `models/` cache and only figures and tables are regenerated — much faster.

## Outputs

All results are written to the `results/` folder inside each experiment subfolder:

| File | Content |
|---|---|
| `power_table.md` | One-step power metrics and exogenous-telemetry ablation |
| `multistep_power_table.md` | Multi-step power results across horizons h=1–24 |
| `split_study_tables.md` | Full sweep: all models × windows × splits |
| `locality_table.md` | Workload-condition breakdown (volatility, trend) |
| `fig_onestep_power_overlay.png` | Actual vs predicted power with rolling error |
| `fig_onestep_ablation.png` | Full vs history-only Transformer vs persistence |
| `fig_onestep_split_mae.png` | Power MAE across all configurations |
| `fig_onestep_split_r2.png` | R² across all configurations |
| `fig_multistep_power_growth.png` | Power MAE and improvement % across horizons |
| `fig_multistep_power_splits.png` | Multi-step power improvement across 3 data splits |
| `fig_locality_per_sample_advantage.png` | Per-sample prediction advantage with power trajectory |
| `fig_locality_best_worst.png` | Best and worst cases ranked by model advantage |

## Clearing the Model Cache

To delete all cached models and retrain from scratch:

**macOS / Linux:**

```bash
rm -rf algorithmic_core/split_50_25_25_final/models/onestep
rm -rf algorithmic_core/split_50_25_25_final/models/multistep
```

**Windows (PowerShell):**

```powershell
Remove-Item -Recurse -Force algorithmic_core\split_50_25_25_final\models\onestep
Remove-Item -Recurse -Force algorithmic_core\split_50_25_25_final\models\multistep
```

Then rerun the five scripts above to retrain everything.

## Key Configuration

| Parameter | Value |
|---|---|
| Dataset | Alibaba GenTD26 cluster telemetry, 1-minute bins, ~929 samples |
| Window size | 60 minutes |
| Features | 27 (GPU util, lags, rolling stats, fractional differencing, QPS, pod ratios, memory, time-of-day) |
| Prediction mode | Residual — model predicts the change from last observed value |
| Power conversion | P(kW) = n_gpus × (50 + 250 × gpu_util) / 1000 (Fan et al. 2007) |
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) with CosineAnnealingLR |
| Loss | Huber loss (SmoothL1Loss) |
| Early stopping | patience=15, max 150 epochs |
| Gradient clipping | max norm = 1.0 |

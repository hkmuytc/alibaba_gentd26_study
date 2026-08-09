# Power Domain and Ablation Results (70/15/15, w=60, Transformer)

Power estimated via Fan et al. (2007): P = n_gpus × (50 + 250 × gpu_util) / 1000
The idle term is charged to every powered GPU; active-pod ratio is not used in the power conversion.

**Configuration**: 70/15/15 split, w=60, seed=0
**Full model**: 27 features (GPU utilization and derived statistics + exogenous workload telemetry)
**History-only model**: 10 features (GPU utilization, lags, rolling means/stds, rate of change, fractional differencing — no QPS, pod ratios, memory, or time-of-day)
**Persistence**: predict current value for all future steps

| Method | Feature Set | Features | MAE (kW) | RMSE (kW) | R² | MAPE (%) | vs Persistence MAE | vs Persistence R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Transformer (full)** | GPU history + workload telemetry | 27 | **0.2362** | **0.3121** | **0.5335** | **3.82** | **+5.7%** | **+0.1066** |
| Transformer (history only) | GPU history only | 10 | 0.3113 | 0.3881 | 0.2784 | 5.10 | -24.3% | -0.1485 |
| Persistence | Repeat last value | — | 0.2505 | 0.3459 | 0.4269 | 4.04 | — | — |

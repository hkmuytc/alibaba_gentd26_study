# Power Domain and Ablation Results (50/25/25, w=60, Transformer)

Power estimated via Fan et al. (2007): P = n_gpus × (50 + 250 × gpu_util) / 1000
The idle term is charged to every powered GPU; active-pod ratio is not used in the power conversion.

**Configuration**: 50/25/25 split, w=60, seed=0
**Full model**: 27 features (GPU utilization and derived statistics + exogenous workload telemetry)
**History-only model**: 10 features (GPU utilization, lags, rolling means/stds, rate of change, fractional differencing — no QPS, pod ratios, memory, or time-of-day)
**Persistence**: predict current value for all future steps

| Method | Feature Set | Features | MAE (kW) | RMSE (kW) | R² | MAPE (%) | vs Persistence MAE | vs Persistence R² |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Transformer (full)** | GPU history + workload telemetry | 27 | **0.2993** | **0.3971** | **0.5408** | **4.57** | **+7.0%** | **+0.0830** |
| Transformer (history only) | GPU history only | 10 | 0.3137 | 0.4181 | 0.4910 | 4.78 | +2.5% | +0.0332 |
| Persistence | Repeat last value | — | 0.3219 | 0.4315 | 0.4578 | 4.93 | — | — |

## Relative Improvements over Persistence

| Method | MAE | RMSE | R² | MAPE |
| --- | ---: | ---: | ---: | ---: |
| Transformer (full) | +7.0% | +8.0% | +0.0830 | +7.4% |
| Transformer (history only) | +2.5% | +3.1% | +0.0332 | +3.1% |

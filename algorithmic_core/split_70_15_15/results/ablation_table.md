## Ablation Study: Exogenous Workload Telemetry Contribution

**Configuration**: 70/15/15 split, w=60, seed=0
**Full model**: 27 features (GPU utilization and derived statistics + exogenous workload telemetry)
**History-only model**: 10 features (GPU utilization, lags, rolling means/stds, rate of change, fractional differencing — no QPS, pod ratios, memory, or time-of-day)
**Persistence**: predict current value for all future steps

| Model | Features | Power MAE (kW) | Power R² | vs Persistence MAE | vs Persistence R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Transformer (full)** | 27 | **0.0497** | **0.6124** | **+4.7%** | **+0.0600** |
| Transformer (history only) | 10 | 0.0546 | 0.6172 | -4.6% | +0.0648 |
| Persistence | — | 0.0522 | 0.5524 | — | — |

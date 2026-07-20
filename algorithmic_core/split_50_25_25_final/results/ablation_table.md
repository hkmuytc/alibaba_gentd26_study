## Ablation Study: Exogenous Workload Telemetry Contribution

**Configuration**: 50/25/25 split, w=60, seed=0
**Full model**: 27 features (GPU utilization and derived statistics + exogenous workload telemetry)
**History-only model**: 10 features (GPU utilization, lags, rolling means/stds, rate of change, fractional differencing — no QPS, pod ratios, memory, or time-of-day)
**Persistence**: predict current value for all future steps

| Model | Features | Power MAE (kW) | Power R² | vs Persistence MAE | vs Persistence R² |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Transformer (full)** | 27 | **0.0858** | **0.4952** | **+5.9%** | **+0.0593** |
| Transformer (history only) | 10 | 0.0883 | 0.4776 | +3.3% | +0.0416 |
| Persistence | — | 0.0912 | 0.4359 | — | — |

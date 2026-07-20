## Power Domain Results (70/15/15, w=60, Transformer)

Power estimated via Fan et al. (2007): P = n_gpus × (50 + 250 × gpu_util × active_pod_ratio) / 1000
Active pod ratio held at last known value. No future information used.

| Method | MAE (kW) | RMSE (kW) | R² | MAPE (%) |
| --- | ---: | ---: | ---: | ---: |
| **Transformer** | **0.0497** | **0.0859** | **0.6124** | **0.87** |
| Persistence | 0.0522 | 0.0923 | 0.5524 | 0.92 |
| Δ | +4.7% | +6.9% | +0.0600 | +4.9% |

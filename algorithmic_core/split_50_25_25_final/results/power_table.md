## Power Domain Results (50/25/25, w=60, Transformer)

Power estimated via Fan et al. (2007): P = n_gpus × (50 + 250 × gpu_util × active_pod_ratio) / 1000
Active pod ratio held at last known value. No future information used.

| Method | MAE (kW) | RMSE (kW) | R² | MAPE (%) |
| --- | ---: | ---: | ---: | ---: |
| **Transformer** | **0.0858** | **0.1479** | **0.4952** | **1.44** |
| Persistence | 0.0912 | 0.1563 | 0.4359 | 1.54 |
| Δ | +5.9% | +5.4% | +0.0593 | +6.1% |

## Multi-Step Window-Averaged Power Prediction Results

**Configuration**: 50/25/25 split, w=60, Transformer, residual prediction
**Target**: mean(y[t+1..t+h]) — average GPU utilization over next h minutes
**Persistence baseline**: y[t] — current value
**Power**: estimated via Fan et al. (2007), active pod ratio held at last known value

| h (min) | Model Power MAE (kW) | Persist Power MAE (kW) | Improvement | Model Power R² | Persist Power R² | Model Wins |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.0846 | 0.0917 | +7.8% | 0.5182 | 0.4464 | ✓ |
| 2 | 0.0902 | 0.0959 | +5.9% | 0.4599 | 0.3731 | ✓ |
| 3 | 0.0936 | 0.1008 | +7.1% | 0.3968 | 0.2789 | ✓ |
| 5 | 0.0998 | 0.1081 | +7.7% | 0.2938 | 0.1062 | ✓ |
| 8 | 0.1041 | 0.1133 | +8.0% | 0.1229 | -0.1567 | ✓ |
| 10 | 0.1120 | 0.1189 | +5.7% | -0.0663 | -0.3682 | ✓ |
| 12 | 0.1105 | 0.1233 | +10.4% | -0.0423 | -0.5609 | ✓ |
| 15 | 0.1112 | 0.1275 | +12.8% | -0.1846 | -0.7779 | ✓ |
| 18 | 0.1157 | 0.1300 | +10.9% | -0.3521 | -1.0020 | ✓ |
| 20 | 0.1255 | 0.1316 | +4.6% | -0.4662 | -1.1461 | ✓ |
| 24 | 0.1299 | 0.1343 | +3.3% | -0.7178 | -1.4706 | ✓ |

**Model wins: 11/11 horizons (power MAE)**

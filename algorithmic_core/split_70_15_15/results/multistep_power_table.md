## Multi-Step Window-Averaged Power Prediction Results

**Configuration**: 70/15/15 split, w=60, Transformer, residual prediction
**Target**: mean(y[t+1..t+h]) — average GPU utilization over next h minutes
**Persistence baseline**: y[t] — current value
**Power**: estimated via Fan et al. (2007), active pod ratio held at last known value

| h (min) | Model Power MAE (kW) | Persist Power MAE (kW) | Improvement | Model Power R² | Persist Power R² | Model Wins |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.2437 | 0.2440 | +0.1% | 0.5218 | 0.4502 | ✓ |
| 2 | 0.2406 | 0.2650 | +9.2% | 0.4251 | 0.3227 | ✓ |
| 3 | 0.2549 | 0.2897 | +12.0% | 0.3155 | 0.1536 | ✓ |
| 5 | 0.2891 | 0.3132 | +7.7% | -0.0339 | -0.1561 | ✓ |
| 8 | 0.3084 | 0.3111 | +0.9% | -0.4237 | -0.5576 | ✓ |
| 10 | 0.3089 | 0.3333 | +7.3% | -0.6197 | -1.0420 | ✓ |
| 12 | 0.3527 | 0.3527 | +0.0% | -0.8176 | -1.4512 | ✓ |
| 15 | 0.3848 | 0.3722 | -3.4% | -1.6696 | -2.0361 | ✗ |
| 18 | 0.3587 | 0.3951 | +9.2% | -1.7753 | -2.9704 | ✓ |
| 20 | 0.2695 | 0.4081 | +34.0% | -0.8022 | -3.7020 | ✓ |
| 24 | 0.3616 | 0.4140 | +12.7% | -2.8920 | -5.1919 | ✓ |

**Model wins: 10/11 horizons (power MAE)**

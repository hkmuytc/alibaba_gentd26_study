# Multi-Step Window-Averaged Power Prediction Results

**Configuration**: 50/25/25 split, w=60, Transformer, residual prediction
**Target**: mean(y[t+1..t+h]) — average GPU utilization over next h minutes
**Persistence baseline**: y[t] — current value
**Power**: strict Fan et al. utilization-to-power conversion; active-pod ratio is not used

| h (min) | Model Power MAE (kW) | Persist Power MAE (kW) | Improvement | Model Power R² | Persist Power R² | Model Wins |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.3227 | 0.3218 | -0.3% | 0.5088 | 0.4755 | ✗ |
| 2 | 0.3204 | 0.3131 | -2.3% | 0.4639 | 0.4631 | ✗ |
| 3 | 0.3252 | 0.3332 | +2.4% | 0.4257 | 0.3740 | ✓ |
| 5 | 0.3436 | 0.3584 | +4.1% | 0.3128 | 0.2087 | ✓ |
| 8 | 0.3950 | 0.3741 | -5.6% | -0.0653 | -0.0430 | ✗ |
| 10 | 0.4520 | 0.3903 | -15.8% | -0.4709 | -0.2288 | ✗ |
| 12 | 0.4588 | 0.4103 | -11.8% | -0.5479 | -0.4214 | ✗ |
| 15 | 0.4073 | 0.4227 | +3.6% | -0.4290 | -0.6820 | ✓ |
| 18 | 0.4315 | 0.4373 | +1.3% | -0.7269 | -1.0005 | ✓ |
| 20 | 0.5649 | 0.4474 | -26.2% | -1.9253 | -1.2365 | ✗ |
| 24 | 0.5645 | 0.4624 | -22.1% | -2.3507 | -1.7198 | ✗ |

## Model wins: 4/11 horizons (power MAE)

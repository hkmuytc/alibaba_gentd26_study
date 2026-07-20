## Multi-Step Window-Averaged Power Prediction Results

**Configuration**: 70/15/15 split, w=60, Transformer, residual prediction
**Target**: mean(y[t+1..t+h]) — average GPU utilization over next h minutes
**Persistence baseline**: y[t] — current value
**Power**: estimated via Fan et al. (2007), active pod ratio held at last known value

| h (min) | Model Power MAE (kW) | Persist Power MAE (kW) | Improvement | Model Power R² | Persist Power R² | Model Wins |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.0512 | 0.0523 | +2.2% | 0.6115 | 0.5617 | ✓ |
| 2 | 0.0558 | 0.0605 | +7.8% | 0.4202 | 0.3149 | ✓ |
| 3 | 0.0613 | 0.0676 | +9.4% | 0.2384 | 0.0483 | ✓ |
| 5 | 0.0632 | 0.0716 | +11.7% | 0.0713 | -0.3366 | ✓ |
| 8 | 0.0626 | 0.0703 | +10.8% | -0.1095 | -0.8275 | ✓ |
| 10 | 0.0687 | 0.0754 | +8.9% | -0.5773 | -1.2950 | ✓ |
| 12 | 0.0699 | 0.0783 | +10.8% | -0.4397 | -1.6868 | ✓ |
| 15 | 0.0786 | 0.0843 | +6.8% | -1.2895 | -2.2693 | ✓ |
| 18 | 0.0804 | 0.0903 | +11.0% | -1.6518 | -3.1123 | ✓ |
| 20 | 0.0709 | 0.0924 | +23.3% | -1.1151 | -3.6429 | ✓ |
| 24 | 0.0762 | 0.0941 | +19.0% | -2.0819 | -4.8929 | ✓ |

**Model wins: 11/11 horizons (power MAE)**

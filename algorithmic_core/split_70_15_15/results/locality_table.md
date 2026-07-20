## Locality Analysis: Power Prediction at h=12

**Configuration**: 70/15/15 split, w=60, Transformer, h=12
**Advantage** = |persistence error| − |model error| (positive = model better)

### By Workload Volatility (15-min rolling σ)

| Regime | Samples | Model Win Rate | Mean Advantage (kW) |
| --- | ---: | ---: | ---: |
| Calm (σ < 0.01) | 48 | 60.4% | +0.0050 |
| Moderate vol (0.01–0.03) | 82 | 51.2% | +0.0105 |

### By Workload Trend (Δ over 5 min)

| Regime | Samples | Model Win Rate | Mean Advantage (kW) |
| --- | ---: | ---: | ---: |
| Stable trend (|Δ₅| < 0.01) | 57 | 49.1% | +0.0031 |
| Rising trend (Δ₅ > 0.01) | 38 | 52.6% | +0.0220 |
| Falling trend (Δ₅ < −0.01) | 35 | 65.7% | +0.0026 |

### Overall

| Metric | Value |
| --- | --- |
| Total test samples | 130 |
| Model wins | 71/130 (55%) |
| Mean advantage | +0.0085 kW |

## Locality Analysis: Power Prediction at h=12

**Configuration**: 50/25/25 split, w=60, Transformer, h=12
**Advantage** = |persistence error| − |model error| (positive = model better)

### By Workload Volatility (15-min rolling σ)

| Regime | Samples | Model Win Rate | Mean Advantage (kW) |
| --- | ---: | ---: | ---: |
| Calm (σ < 0.01) | 57 | 19.3% | -0.0195 |
| Moderate vol (0.01–0.03) | 158 | 44.3% | +0.0245 |

### By Workload Trend (Δ over 5 min)

| Regime | Samples | Model Win Rate | Mean Advantage (kW) |
| --- | ---: | ---: | ---: |
| Stable trend (|Δ₅| < 0.01) | 86 | 29.1% | -0.0001 |
| Rising trend (Δ₅ > 0.01) | 68 | 57.4% | +0.0468 |
| Falling trend (Δ₅ < −0.01) | 61 | 27.9% | -0.0067 |

### Overall

| Metric | Value |
| --- | --- |
| Total test samples | 215 |
| Model wins | 81/215 (38%) |
| Mean advantage | +0.0129 kW |

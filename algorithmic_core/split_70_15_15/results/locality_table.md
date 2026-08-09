## Locality Analysis: Power Prediction at h=15

**Configuration**: 70/15/15 split, w=60, Transformer, h=15
**Advantage** = |persistence error| − |model error| (positive = model better)

### By Workload Volatility (15-min rolling σ)

| Regime | Samples | Model Win Rate | Mean Advantage (kW) |
| --- | ---: | ---: | ---: |
| Calm (σ < 0.01) | 48 | 56.2% | +0.0988 |
| Moderate vol (0.01–0.03) | 81 | 38.3% | -0.0787 |

### By Workload Trend (Δ over 5 min)

| Regime | Samples | Model Win Rate | Mean Advantage (kW) |
| --- | ---: | ---: | ---: |
| Stable trend (|Δ₅| < 0.01) | 57 | 47.4% | +0.0070 |
| Rising trend (Δ₅ > 0.01) | 38 | 52.6% | +0.0115 |
| Falling trend (Δ₅ < −0.01) | 34 | 32.4% | -0.0725 |

### Overall

| Metric | Value |
| --- | --- |
| Total test samples | 129 |
| Model wins | 58/129 (45%) |
| Mean advantage | -0.0126 kW |

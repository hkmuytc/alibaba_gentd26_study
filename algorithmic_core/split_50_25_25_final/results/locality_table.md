# Locality Analysis: Power Prediction at h=15

**Configuration**: 50/25/25 split, w=60, Transformer, h=15
**Advantage** = |persistence error| − |model error| (positive = model better)

## By Workload Volatility (15-min rolling σ)

| Regime | Samples | Model Win Rate | Mean Advantage (kW) |
| --- | ---: | ---: | ---: |
| Calm (σ < 0.01) | 57 | 19.3% | -0.1287 |
| Moderate vol (0.01–0.03) | 157 | 54.8% | +0.0677 |

## By Workload Trend (Δ over 5 min)

| Regime | Samples | Model Win Rate | Mean Advantage (kW) |
| --- | ---: | ---: | ---: |
| Stable trend (\|Δ₅\| < 0.01) | 87 | 34.5% | -0.0466 |
| Rising trend (Δ₅ > 0.01) | 67 | 62.7% | +0.1349 |
| Falling trend (Δ₅ < −0.01) | 60 | 41.7% | -0.0282 |

## Overall

| Metric | Value |
| --- | --- |
| Total test samples | 214 |
| Model wins | 97/214 (45%) |
| Mean advantage | +0.0154 kW |

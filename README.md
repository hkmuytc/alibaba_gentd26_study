# Data Center Power Demand Forecasting — Prototype

Interactive prototype for predicting short-term data center power demand using deep learning on Alibaba production cluster traces.

## Project Structure

```
prototype/
├── app.py                          # Streamlit dashboard (run this)
├── requirements.txt
├── config/
│   └── defaults.py                 # Default hyperparameters & constants
├── data/
│   ├── raw/                        # GenTD26 trace CSVs
│   ├── external/                   # Supplementary datasets (GPU v2020)
│   └── processed/                  # Aggregated & feature-engineered CSVs
├── models/
│   └── saved/                      # Trained model weights & metrics
├── src/
│   ├── data_processing/
│   │   └── pipeline.py             # Data loading, aggregation, power estimation
│   ├── models/
│   │   └── architectures.py        # LSTM, GRU, Transformer definitions
│   ├── evaluation/
│   │   └── trainer.py              # Training loop, metrics, save/load
│   └── dashboard/
└── notebooks/
```

## Quick Start

```bash
cd prototype

# Install dependencies
pip install -r requirements.txt

# Launch the interactive dashboard
streamlit run app.py
```

## Datasets

### Primary: Alibaba GenTD26 (GenAI Serving Trace)
- GPU duty cycle, GPU memory, container memory, QPS
- From a production Stable Diffusion serving cluster
- ~3 days of data at raw granularity

### Supplementary: Alibaba GPU v2020 Machine Metrics
- Per-machine CPU & GPU utilization
- ~80k records from a GPU training cluster
- Provides CPU workload comparison

## Dashboard Pages

1. **Data Explorer** — Load, process, and visualise datasets with power estimation
2. **Train Models** — Train LSTM/GRU/Transformer with configurable hyperparameters
3. **Model Comparison** — Side-by-side metrics (MAE, RMSE, MAPE, R²), bar charts, radar
4. **Predictions** — Overlay actual vs predicted, error distributions, scatter plots

## Power Estimation

Uses the Fan et al. (2007) linear power model:
```
P = P_idle + (P_max - P_idle) × utilisation
```

## Models

| Architecture | Description |
|---|---|
| LSTM | 2-layer LSTM with dropout, uses last hidden state |
| GRU | 2-layer GRU with dropout, lighter than LSTM |
| Transformer | Encoder-only with positional encoding and self-attention |

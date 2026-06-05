# Data Center Power Demand Forecasting

Interactive dashboard for predicting short-term data center power demand using deep learning on Alibaba production cluster traces.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/hkmuytc/alibaba_gentd26_study.git
cd alibaba_gentd26_study
```

### 2. Set up a Python environment

Requires **Python 3.9+**.

```bash
python -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Download the dataset

The raw data is not included in this repository. Run the provided script to download the [GenTD26 dataset](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI) from Alibaba's official GitHub:

```bash
bash download_data.sh
```

This downloads and extracts the following files into `data/raw/`:
- `pod_gpu_duty_cycle_anon.csv` — GPU utilisation per container
- `pod_gpu_memory_used_bytes_anon.csv` — GPU memory usage per container
- `pod_memory_util_anon.csv` — Container memory utilisation
- `qps.csv` — System query-per-second sampling
- `data_trace_processed.csv` — Pre-processed trace data

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

---

## Pre-trained Models

Two best-performing models are included in the repository under `models/saved/`:

| Model | Dataset | R² | MAE | RMSE |
|---|---|---|---|---|
| GRU (window=24) | GenTD26 | 0.587 | 0.176 | 0.206 |
| Transformer (window=24) | GPU v2020 | 0.976 | — | — |

To use them, go to the **Predictions** page in the dashboard and select the corresponding model.

---

## Dashboard Pages

1. **Data Explorer** — Load, process, and visualise datasets with power estimation
2. **Train Models** — Train LSTM/GRU/Transformer with configurable hyperparameters
3. **Model Comparison** — Side-by-side metrics (MAE, RMSE, MAPE, R²), bar charts, radar
4. **Predictions** — Overlay actual vs predicted, error distributions, scatter plots

---

## Project Structure

```
├── app.py                          # Streamlit entry point (run this)
├── requirements.txt
├── download_data.sh                # Script to download the GenTD26 dataset
├── config/
│   └── defaults.py                 # Hyperparameters & constants
├── data/
│   ├── raw/                        # GenTD26 trace CSVs (populated by download_data.sh)
│   ├── external/                   # Supplementary datasets (GPU v2020)
│   └── processed/                  # Aggregated & feature-engineered CSVs
├── models/
│   └── saved/                      # Trained model weights & metrics
└── src/
    ├── data_processing/
    │   └── pipeline.py             # Data loading, aggregation, power estimation
    ├── models/
    │   └── architectures.py        # LSTM, GRU, Transformer definitions
    ├── evaluation/
    │   └── trainer.py              # Training loop, metrics, save/load
    └── dashboard/                  # Streamlit page components
```

---

## Datasets

### Primary: Alibaba GenTD26 (GenAI Serving Trace)
- Source: https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI
- GPU duty cycle, GPU memory, container memory, QPS
- From a production Stable Diffusion serving cluster (~3 days at raw granularity)

### Supplementary: Alibaba GPU v2020 Machine Metrics
- Per-machine CPU & GPU utilisation (~80k records from a GPU training cluster)

---

## Power Estimation

Uses the Fan et al. (2007) linear power model:

```
P = P_idle + (P_max - P_idle) × utilisation
```

## Model Architectures

| Architecture | Description |
|---|---|
| LSTM | 2-layer LSTM with dropout, uses last hidden state |
| GRU | 2-layer GRU with dropout, lighter than LSTM |
| Transformer | Encoder-only with positional encoding and self-attention |

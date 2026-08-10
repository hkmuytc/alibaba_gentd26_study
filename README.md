# Data Center Power Demand Forecasting on Alibaba GenTD26

This repository predicts data center GPU power demand using deep learning models trained on the **Alibaba GenTD26 generative AI cluster trace**. It contains three interconnected deliverables:

1. **Algorithmic Core** — the full ML pipeline: data processing, model training (LSTM, GRU, Transformer, CNN-LSTM), and result generation across multiple experimental configurations
2. **Physical Validation (ECS)** — a real-hardware experiment on an Alibaba Cloud GPU instance that tests whether the model's power predictions hold up against measured physical power draw
3. **Streamlit Dashboard** — an interactive web application for exploring the data, stepping through predictions one sample at a time, and visualizing the physical validation results

The research shows that a Transformer model can forecast next-minute GPU utilization with meaningful accuracy, but that the standard linear model for converting utilization to power (Fan et al. 2007) systematically underestimates physical power consumption for serverless generative AI workloads.

---

## Windows Python Note

On Windows, the command to run Python depends on how it was installed:

- **Python 3.11 and earlier** (installed from python.org): use `python`
- **Python 3.12 and later** (installed from the Microsoft Store): use `py`

If `python --version` works in your terminal, use `python`. If it does not, use `py` instead.

---

## Quick Start — From Fresh Clone to Full Results

Follow these steps in order. Each section builds on the previous one.

### Step 1: Clone the Repository

**macOS / Linux:**

```bash
git clone https://github.com/hkmuytc/alibaba_gentd26_study.git
cd alibaba_gentd26_study
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/hkmuytc/alibaba_gentd26_study.git
cd alibaba_gentd26_study
```

### Step 2: Install Dependencies

Requires **Python 3.9 or later**.

**macOS / Linux:**

```bash
pip3 install -r requirements.txt
```

**Windows (PowerShell):**

```powershell
python -m pip install -r requirements.txt
```

Or if using Python 3.12+:

```powershell
py -m pip install -r requirements.txt
```

### Step 3: Download the Raw Data

The `data/` folder is git-ignored, so a fresh clone does not include the raw CSV files. Use the provided download script:

**macOS / Linux:**

```bash
bash download_data.sh
```

**Windows (Git Bash or WSL):**

```bash
bash download_data.sh
```

**Windows (manual):** Download and extract the five `.tar.gz` files from [Alibaba's GenTD26 trace release](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI) into `data/raw/`. The files are:

| File | Content |
|---|---|
| `pod_gpu_duty_cycle_anon.csv` | Per-container GPU utilization (0–100%) |
| `pod_gpu_memory_used_bytes_anon.csv` | Per-container GPU memory usage |
| `pod_memory_util_anon.csv` | Per-container system memory utilization |
| `qps.csv` | Queries per second (generative + API requests) |
| `data_trace_processed.csv` | Pre-processed combined trace |

### Step 4: Verify the Model Cache

The clone **already includes** trained model weights and scalers. No model download is needed. You should see files in:

```
algorithmic_core/split_50_25_25_final/models/onestep/   (36 models)
algorithmic_core/split_50_25_25_final/models/multistep/  (33 models)
```

On subsequent runs, scripts load from this cache instead of retraining — making reproduction fast.

---

## Experiment 1: Reproduce the Paper's Main Results

This is the primary reproducibility path. Five scripts in `algorithmic_core/split_50_25_25_final/executables/` generate every figure and table in the paper.

**macOS / Linux:**

```bash
# 1. One-step sweep: trains/loads 36 models (4 architectures × 3 windows × 3 splits)
python3 algorithmic_core/split_50_25_25_final/executables/split_study.py

# 2. One-step main results: best config (Transformer, w=60, 50/25/25) + ablation
python3 algorithmic_core/split_50_25_25_final/executables/generate_main_results.py

# 3. Multi-step sweep: trains/loads 33 models (11 horizons × 3 splits)
python3 algorithmic_core/split_50_25_25_final/executables/multistep_averaged.py

# 4. Multi-step comparison figure across all splits
python3 algorithmic_core/split_50_25_25_final/executables/generate_power_figures.py

# 5. Locality analysis: workload-condition breakdown at h=15
python3 algorithmic_core/split_50_25_25_final/executables/locality_analysis.py
```

**Windows (PowerShell, Python 3.11 and earlier):**

```powershell
python algorithmic_core\split_50_25_25_final\executables\split_study.py
python algorithmic_core\split_50_25_25_final\executables\generate_main_results.py
python algorithmic_core\split_50_25_25_final\executables\multistep_averaged.py
python algorithmic_core\split_50_25_25_final\executables\generate_power_figures.py
python algorithmic_core\split_50_25_25_final\executables\locality_analysis.py
```

**Windows (PowerShell, Python 3.12+):**

```powershell
py algorithmic_core\split_50_25_25_final\executables\split_study.py
py algorithmic_core\split_50_25_25_final\executables\generate_main_results.py
py algorithmic_core\split_50_25_25_final\executables\multistep_averaged.py
py algorithmic_core\split_50_25_25_final\executables\generate_power_figures.py
py algorithmic_core\split_50_25_25_final\executables\locality_analysis.py
```

### What Each Script Produces

All outputs are written to `algorithmic_core/split_50_25_25_final/results/`:

| Script | Outputs |
|---|---|
| `split_study.py` | `fig_onestep_split_mae.png`, `fig_onestep_split_r2.png`, `split_study_tables.md` |
| `generate_main_results.py` | `fig_onestep_power_overlay.png`, `fig_onestep_ablation.png`, `power_table.md` |
| `multistep_averaged.py` | `averaged_results.json`, `multistep_power_table.md` |
| `generate_power_figures.py` | `fig_multistep_power_splits.png` |
| `locality_analysis.py` | `fig_multistep_power_growth.png`, `fig_locality_per_sample_advantage.png`, `fig_locality_best_worst.png`, `locality_table.md` |

### Retrain From Scratch (Optional)

If you want to discard the cached models and retrain everything:

**macOS / Linux:**

```bash
rm -rf algorithmic_core/split_50_25_25_final/models/onestep
rm -rf algorithmic_core/split_50_25_25_final/models/multistep
```

**Windows (PowerShell):**

```powershell
Remove-Item -Recurse -Force algorithmic_core\split_50_25_25_final\models\onestep
Remove-Item -Recurse -Force algorithmic_core\split_50_25_25_final\models\multistep
```

Then rerun the five scripts. Retraining on CPU will be significantly slower than the cached path.

---

## Experiment 2: Physical Validation (ECS)

This experiment tests the model against real hardware power measurements. It works in two phases: telemetry is generated on an Alibaba Cloud GPU instance, then the AI evaluation and visualization are done locally.

### The Quick Path (Traces Already Recorded)

The telemetry files (`hardware_trace.csv` and `qps_trace.csv`) are already committed to the repository. To reproduce the paper's validation figure:

**macOS / Linux:**

```bash
cd ecs-realtime
python3 03_evaluate_power.py
python3 04_visualize_results.py
```

**Windows (PowerShell, Python 3.11 and earlier):**

```powershell
cd ecs-realtime
python 03_evaluate_power.py
python 04_visualize_results.py
```

**Windows (PowerShell, Python 3.12+):**

```powershell
cd ecs-realtime
py 03_evaluate_power.py
py 04_visualize_results.py
```

- `03_evaluate_power.py` — Processes the physical telemetry through the same pipeline used during training, loads the Transformer from `algorithmic_core/split_50_25_25_final/`, and prints a **Power Gap analysis** comparing predicted vs actual power.
- `04_visualize_results.py` — Generates `ecs-realtime/fig_full_paper_validation.png`, a 3-panel figure showing: (1) utilization forecast accuracy, (2) the gap between estimated and measured power, and (3) the workload context.

Optionally, calibrate an improved power model:

**macOS / Linux:**
```bash
cd ecs-realtime
python3 find_transform.py
```

**Windows (PowerShell, Python 3.11 and earlier):**
```powershell
cd ecs-realtime
python find_transform.py
```

**Windows (PowerShell, Python 3.12+):**
```powershell
cd ecs-realtime
py find_transform.py
```

This finds that the standard Fan model (50W idle + 250W range) has ~80W MAE, while an optimal affine fit (115W idle + 268W range) achieves ~11W MAE — an 87% improvement.

### The Full Path (Fresh Cloud Telemetry)

If you want to collect new physical telemetry from scratch on an ECS instance, see the [ecs-realtime/README.md](ecs-realtime/README.md) for the complete cloud execution workflow.

---

## Experiment 3: Streamlit Dashboard

The interactive dashboard lets you explore the data, step through model predictions one sample at a time, and visualize the physical validation results.

**macOS / Linux:**

```bash
streamlit run app.py
```

**Windows (PowerShell):**

```powershell
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

### Dashboard Pages

| Page | What It Shows |
|---|---|
| **Chapter 1: One-Step Anticipation** | Full test set predictions, plus an interactive step-by-step simulator where you drag a slider through each prediction and watch a running scoreboard of cumulative MAE and win rate. |
| **Chapter 2: Multi-Step Averaging** | Same structure but for forward-averaged horizons (h=1 to h=24 minutes ahead). A select-slider lets you explore how accuracy degrades at longer horizons. |
| **Bonus: Physical Validation (ECS)** | Loads the ECS telemetry and displays the actual GPU utilization, power draw, and QPS traces alongside the "Physical Reality Gap" analysis. |

### Dashboard Requirements

- The `data/raw/` folder must be populated (run `download_data.sh` first)
- The dashboard loads cached models directly from `algorithmic_core/split_50_25_25_final/models/` — no extra model download needed
- Compute device is auto-detected: MPS (Apple Silicon), CUDA (NVIDIA), or CPU

---

## Comparison Experiment: 70/15/15 Split

The `algorithmic_core/split_70_15_15/` folder is a comparison variant that uses a more conventional train/val/test split (70/15/15 instead of the aggressive 50/25/25). It runs the same code and the same five scripts, and reports all three splits internally. Use it to compare how results change when more training data is available.

To run the comparison experiment:

**macOS / Linux:**

```bash
python3 algorithmic_core/split_70_15_15/executables/split_study.py
python3 algorithmic_core/split_70_15_15/executables/generate_main_results.py
python3 algorithmic_core/split_70_15_15/executables/multistep_averaged.py
python3 algorithmic_core/split_70_15_15/executables/generate_power_figures.py
python3 algorithmic_core/split_70_15_15/executables/locality_analysis.py
```

**Windows (PowerShell, Python 3.11 and earlier):**

```powershell
python algorithmic_core\split_70_15_15\executables\split_study.py
python algorithmic_core\split_70_15_15\executables\generate_main_results.py
python algorithmic_core\split_70_15_15\executables\multistep_averaged.py
python algorithmic_core\split_70_15_15\executables\generate_power_figures.py
python algorithmic_core\split_70_15_15\executables\locality_analysis.py
```

**Windows (PowerShell, Python 3.12+):**

```powershell
py algorithmic_core\split_70_15_15\executables\split_study.py
py algorithmic_core\split_70_15_15\executables\generate_main_results.py
py algorithmic_core\split_70_15_15\executables\multistep_averaged.py
py algorithmic_core\split_70_15_15\executables\generate_power_figures.py
py algorithmic_core\split_70_15_15\executables\locality_analysis.py
```

---

## Repository Layout

```
alibaba_gentd26_study/
├── app.py                          ← Streamlit entry point
├── requirements.txt                ← Python dependencies
├── download_data.sh                ← Downloads raw Alibaba CSVs
├── algorithmic_core/
│   ├── README.md                   ← Experiment documentation
│   ├── split_50_25_25_final/       ← Canonical paper results
│   │   ├── core/                   ← Pipeline, models, training, caching
│   │   ├── executables/            ← 5 reproduction scripts
│   │   ├── models/                 ← Cached weights and scalers
│   │   └── results/                ← Generated figures and tables
│   └── split_70_15_15/             ← Comparison variant (same structure)
├── ecs-realtime/                   ← Physical validation experiment
│   ├── README.md                   ← ECS experiment documentation
│   ├── 03_evaluate_power.py        ← AI evaluation (run locally)
│   ├── 04_visualize_results.py     ← Figure generation (run locally)
│   ├── hardware_trace.csv          ← Recorded GPU telemetry
│   └── qps_trace.csv              ← Recorded request log
├── dashboard/                      ← Streamlit application
│   ├── pages.py                    ← All page logic
│   ├── inference.py                ← Model loading from cached weights
│   ├── nav.py                      ← Navigation structure
│   └── theme.py                    ← Visual styling
├── config/                         ← Default constants
├── data/                           ← Raw CSVs (git-ignored)
└── literature/                     ← Reference papers (PDF)
```

## Key Facts

| | |
|---|---|
| **Dataset** | Alibaba GenTD26 cluster-trace-v2026-GenAI |
| **Time span** | ~15 hours of cluster telemetry |
| **Time resolution** | 60-second bins (~929 samples after aggregation) |
| **Features** | 27 engineered features per timestep |
| **Best model** | Transformer (d_model=32, nhead=4, 2 layers), window=60 minutes |
| **Power model** | Fan et al. (2007): P(kW) = n_gpus × (50 + 250 × util) / 1000 |
| **Key finding** | Transformer predicts utilization well, but the linear util-to-power conversion misses ~80W of physical power in serverless GenAI workloads |

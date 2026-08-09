# Physical Validation — ECS Realtime Experiment

This experiment tests whether our Transformer power predictions hold up against **real hardware**. We replay a slice of the Alibaba GenTD26 generative AI workload trace on a rented Alibaba Cloud GPU instance, measure the actual physical power draw, and then compare it against the Transformer model's predictions.

## What This Proves

The algorithmic core trains a Transformer to predict GPU utilization and converts that to power via the standard Fan et al. (2007) linear model. This physical experiment reveals a critical gap: **the linear model systematically underestimates real power consumption** for serverless generative AI workloads, because it ignores memory-bound power spikes, VRAM retention during idle periods, PCIe bus energization, and thermal saturation effects during dense burst sequences.

## The Two-Phase Workflow

The experiment is split into a **cloud phase** (generating physical telemetry on an ECS GPU instance) and a **local phase** (running the AI evaluation and generating figures on your own machine). Only the local phase is needed to reproduce the paper's figures and tables.

---

## Phase 1: Cloud Execution (Alibaba ECS)

> This phase has already been completed. The output traces (`hardware_trace.csv` and `qps_trace.csv`) are committed to the repository. You only need to repeat this phase if you want to collect fresh telemetry.

**Infrastructure:** Alibaba Cloud `ecs.gn7i-c8g1.2xlarge` spot instance (NVIDIA A10, 24 GB VRAM).

**What happens on the server:**

1. `01_monitor_hardware.py` polls `nvidia-smi` every second, recording GPU utilization, memory usage, and power draw into `hardware_trace.csv` (12,243 readings at 1 Hz).

2. `02_replay_workload.py` loads Stable Diffusion XL onto the GPU and replays a 3-hour window of the GenTD26 trace (starting at minute 600). Each generative AI request from the trace is executed as a real inference job, paced to match the original arrival pattern. When the gap between requests exceeds 60 seconds, the pipeline is torn down and reloaded to simulate serverless cold starts. Each completed request is logged to `qps_trace.csv` (1,784 requests total).

**If you need to re-run the cloud phase:**

```bash
# Copy the ECS scripts and processed trace to the server
scp -r ./ecs-realtime root@<YOUR_ECS_IP>:~/
scp ./data/raw/data_trace_processed.csv root@<YOUR_ECS_IP>:~/ecs-realtime/

# SSH in, set up dependencies, and launch inside tmux
ssh root@<YOUR_ECS_IP>
bash ~/ecs-realtime/setup_ecs.sh
tmux new -s experiment
bash ~/ecs-realtime/run_experiment.sh
```

`setup_ecs.sh` installs Python, PyTorch, CUDA drivers, and the Hugging Face stack. `run_experiment.sh` launches both scripts and runs for approximately 3 hours.

---

## Phase 2: Local Analysis — Reproducing Figures and Tables

This is the phase that matters for result reproduction. Everything runs on your local machine using the pre-recorded telemetry and the trained Transformer model from `algorithmic_core/split_50_25_25_final/`.

### Prerequisites

Before running the evaluation scripts, make sure you have:

1. **Installed Python dependencies** from the project root:

   **macOS / Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

   **Windows (PowerShell):**
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   py -m pip install -r requirements.txt
   ```

2. **Downloaded the raw data** (needed for the pipeline's feature engineering):

   **macOS / Linux:**
   ```bash
   bash download_data.sh
   ```

   **Windows (Git Bash or WSL):**
   ```bash
   bash download_data.sh
   ```
   Or manually extract the five `.tar.gz` files from the [Alibaba GenTD26 release](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI) into `data/raw/`.

3. **Verified the trace files exist** in this folder:
   - `ecs-realtime/hardware_trace.csv` — 12,243 rows of GPU telemetry
   - `ecs-realtime/qps_trace.csv` — 1,784 rows of generative request logs

### Step 1: Evaluate the Power Gap

Run the evaluation script. It processes the physical telemetry through the same feature engineering pipeline used during training, loads the cached Transformer model from `algorithmic_core/split_50_25_25_final/models/onestep/`, runs inference, and compares the predicted power against the actual measured power.

**macOS / Linux:**
```bash
cd ecs-realtime
python3 03_evaluate_power.py
```

**Windows (PowerShell):**
```powershell
cd ecs-realtime
python 03_evaluate_power.py
```

**What it does:**
- Bins the hardware trace and QPS trace into 60-second intervals
- Engineers the same 27 features used during training (lags, rolling stats, fractional differencing, time-of-day embeddings, etc.)
- Loads the trained Transformer weights (`transformer_w60_50-25-25.pt`) and scalers
- Constructs sliding windows of 60 timesteps and predicts utilization residuals
- Converts predicted utilization to watts via the Fan et al. linear model
- Prints a **Power Gap analysis** to the console showing predicted vs actual average burst power

**Expected output:**
```
--- DISCUSSION RESULTS ---
Predicted Avg Burst Power (Linear Model): ~XXX W
Actual Avg Burst Power (Physical GPU):    ~XXX W
Unmodeled Power Gap:                      +XXX W
```

The positive gap demonstrates that the linear Fan model underestimates physical power.

### Step 2: Generate the Publication Figure

Run the visualization script. It produces a three-panel figure suitable for inclusion in the paper.

**macOS / Linux:**
```bash
python3 04_visualize_results.py
```

**Windows (PowerShell):**
```powershell
python 04_visualize_results.py
```

**Output:** `ecs-realtime/fig_full_paper_validation.png` (300 DPI, Times New Roman)

**What the three panels show:**

| Panel | Content |
|---|---|
| **Tier 1 — Utilization Forecast** | Transformer's predicted GPU utilization (teal dashed) versus actual measured utilization (grey solid). Shows the model tracks utilization patterns well. |
| **Tier 2 — Utilization-to-Power Gap** | Actual measured power (crimson) versus Fan-estimated power (orange dashed). The crimson shaded region between them is the **unmodeled power gap** — power that utilization alone cannot explain. |
| **Tier 3 — Replay Workload** | Bar chart of generative requests per minute, providing context for the utilization and power patterns above. |

The X-axis labels show both experiment minutes and the corresponding offset into the original GenTD26 trace (adding 660 = 60 warmup + 600 trace offset).

### Optional: Calibrate an Improved Power Model

The `find_transform.py` script fits an optimal affine transform between utilization and power, comparing it against the standard Fan model.

**macOS / Linux:**
```bash
python3 find_transform.py
```

**Windows (PowerShell):**
```powershell
python find_transform.py
```

**Expected finding:** The standard Fan model (base=50W, range=250W) has an MAE of ~80W. The optimal affine fit (base≈115W, range≈268W) achieves an MAE of ~11W — an 87% error reduction. The elevated base power is explained by VRAM retention, PCIe bus energization, and thermal saturation that persist even when GPU compute utilization drops to zero.

---

## How This Connects to the Algorithmic Core

The evaluation scripts import directly from `algorithmic_core/split_50_25_25_final/`:

- **Feature engineering:** `core.pipeline.engineer_features()` — the same function used during model training
- **Power estimation:** `core.pipeline.estimate_power_kw()` — the Fan et al. linear model
- **Model architecture:** `core.models.TransformerForecaster` — d_model=32, nhead=4, num_layers=2
- **Model weights:** `models/onestep/transformer_w60_50-25-25.pt` and its paired scalers

This ensures the physical evaluation uses exactly the same pipeline and model that produced the paper's main results.

## File Reference

| File | Purpose | Runs On |
|---|---|---|
| `01_monitor_hardware.py` | Polls `nvidia-smi` at 1 Hz, writes `hardware_trace.csv` | ECS (cloud) |
| `02_replay_workload.py` | Replays GenTD26 trace as Stable Diffusion XL jobs, writes `qps_trace.csv` | ECS (cloud) |
| `03_evaluate_power.py` | Processes telemetry, runs Transformer inference, prints Power Gap analysis | Local |
| `04_visualize_results.py` | Generates the 3-panel paper validation figure | Local |
| `find_transform.py` | Calibrates an improved affine power model | Local |
| `setup_ecs.sh` | Installs dependencies on the ECS instance | ECS (cloud) |
| `run_experiment.sh` | Orchestrates scripts 01 and 02 on the ECS instance | ECS (cloud) |
| `hardware_trace.csv` | Recorded GPU telemetry (12,243 rows, 1 Hz) | Output |
| `qps_trace.csv` | Recorded generative request log (1,784 rows) | Output |
| `fig_full_paper_validation.png` | The 3-panel publication figure | Output |

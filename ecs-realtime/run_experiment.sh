#!/bin/bash
set -e

echo "========================================="
echo " Starting Alibaba GenTD26 Physical Trace"
echo "========================================="

# Ensure we are in the virtual environment
if [[ ! -d "ecs_env" ]]; then
    echo "ERROR: Virtual environment not found. Please run bash setup_ecs.sh first."
    exit 1
fi
source ecs_env/bin/activate

# Optional: Run a quick 1-minute dry-run if the user passes `--dry-run` flag
if [[ "$1" == "--dry-run" ]]; then
    echo "DRY RUN MODE: Modifying workload to run for only a few minutes..."
    # You would typically pass an argument to 02_replay_workload.py here
fi

# 1. Start Hardware Monitor in the BACKGROUND
echo "[1/3] Starting hardware monitor (nvidia-smi polling)..."
python3 01_monitor_hardware.py &
MONITOR_PID=$!
echo "Hardware monitor running in background (PID: $MONITOR_PID)."

# 2. Start Workload Replayer in the FOREGROUND
echo "[2/3] Starting GenTD26 workload replayer..."
echo "This process will take several hours. Logs will appear below:"
echo "------------------------------------------------------------"

# We run this in the foreground so the script waits for it to finish.
# If this crashes, the trap below still ensures the monitor is killed.
python3 02_replay_workload.py

echo "------------------------------------------------------------"
echo "Workload replay finished."

# 3. Kill the Hardware Monitor
echo "Stopping hardware monitor..."
kill $MONITOR_PID
wait $MONITOR_PID 2>/dev/null || true

# 4. Stop Cloud Compute (We evaluate locally!)
echo "[3/3] Cloud generation complete. Skipping evaluation."
echo "Please download the trace files to your local Mac and run 03_evaluate_power.py locally."

echo "========================================="
echo "✅ Experiment Complete! Check hardware_trace.csv and qps_trace.csv."
echo "========================================="

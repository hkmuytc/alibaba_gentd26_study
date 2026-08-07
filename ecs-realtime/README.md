# Alibaba GenTD26 Physical Validation Experiment

This directory contains the runtime scripts designed to execute directly on a rented Alibaba Cloud `ecs.gn7i-c8g1.2xlarge` (NVIDIA A10) spot instance.

## Execution Blueprint (Hybrid Workflow)

This architecture is designed for the **Hybrid Workflow**. You generate the physical telemetry on the cloud to save compute costs, wait for it to finish, and then do the AI evaluation completely locally on your Mac.

### Phase 1: Cloud Execution (Alibaba ECS)

1. **Push the minimum files to your Cloud Server:**

   ```bash
   scp -r ./ecs-realtime root@<YOUR_ECS_IP>:~/
   scp -r ../../data/raw/data_trace_processed.csv root@<YOUR_ECS_IP>:~/ecs-realtime/
   ```

2. **Server Setup (`setup_ecs.sh`)**
   SSH into the Cloud Server and run the dependencies check:

   ```bash
   bash setup_ecs.sh
   ```

3. **Master Generation (`run_experiment.sh`)**
   Use tmux to prevent disconnects, and start the generation:

   ```bash
   tmux new -s experiment
   bash run_experiment.sh
   ```

   *This starts `01_monitor_hardware.py` in the background, and runs `02_replay_workload.py` in the foreground. It will run for 3 hours, save the traces, and automatically exit.*

### Phase 2: Local Analysis (Your Mac)

1. **Pull the Traces back to your Mac:**
   Once the cloud finishes, pull the two generated `.csv` files home:

   ```bash
   scp root@<YOUR_ECS_IP>:~/ecs-realtime/hardware_trace.csv ./ecs-realtime/
   scp root@<YOUR_ECS_IP>:~/ecs-realtime/qps_trace.csv ./ecs-realtime/
   ```

   *(You can now destroy the Spot Instance to save money).*

2. **Run the AI Evaluation (`03_evaluate_power.py`)**
   Now that the physical data is on your Mac, run the evaluating python script. It will trace up to `algorithmic_core/split_50_25_25_final_corrected/models/`, grab the corrected strict-Fan Transformer model + scalers, feed the physical data into it, and generate the final Power-Gap discussion output for your paper.

   ```bash
   python 03_evaluate_power.py
   ```

**Core Thesis Proof:** This pipeline demonstrates that while Transformers excellently predict utilization sequences, a linear `estimate_power_kw` translation fails in the physical world due to unmodeled IO-bound Serverless memory loading spikes and thermal-leakage during dense bursts.

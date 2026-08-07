import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "font.size": 14,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "legend.fontsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# Import the data processing logic from script 03
import importlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
eval_module = importlib.import_module("03_evaluate_power")
process_physical_data = eval_module.process_physical_data

def visualize_results():
    # 1. Process the raw CSVs into the 60-second bins
    hw_path = "hardware_trace.csv"
    qps_path = "qps_trace.csv"
    
    if not os.path.exists(hw_path) or not os.path.exists(qps_path):
        print(f"Error: Missing {hw_path} or {qps_path}. Please download them from ECS first.")
        return

    print("Loading physical telemetry...")
    features_df, actual_power = process_physical_data(hw_path, qps_path, offset_minutes=600)
    
    # 2. Run the AI Model evaluation to get AI predictions
    print("Running AI inference models...")
    evaluate_func = eval_module.evaluate
    valid_features_df, valid_actual_power, predicted_watts = evaluate_func(features_df, actual_power)

    # 3. Create the 3-Tier Visualization
    print("Generating graphical analysis...")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    time_axis = np.arange(len(valid_features_df))

    # --- TOP PLOT: Transformer vs Actual Utilization ---
    # To get back the AI's predicted utilization independently from the Watts, we reverse the math formula simply.
    # Strict Fan model for one GPU: Power(W) = 50 + 250 * util.
    # Thus: util = (Power - 50) / 250.
    implied_predicted_util = []
    for watts in predicted_watts:
        implied_predicted_util.append((watts - 50) / 250)
    
    ax1.plot(time_axis, valid_features_df['gpu_util'] * 100, label="Measured utilization", color='dimgray', linewidth=3)
    ax1.plot(time_axis, np.array(implied_predicted_util) * 100, label="Transformer prediction", color='teal', linestyle='--', linewidth=2.5)
    ax1.set_title("Tier 1: Utilization Forecast", pad=8)
    ax1.set_ylabel("Utilization (%)")
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc='lower right')
    ax1.set_ylim(-5, 110)

    # --- MIDDLE PLOT: The Linear Math Failure ---
    ax2.plot(time_axis, valid_actual_power, label="Measured power", color='crimson', linewidth=3)
    ax2.plot(time_axis, predicted_watts, label="Fan estimate", color='orange', linestyle='--', linewidth=2)
    # Highlight the gap
    ax2.fill_between(time_axis, valid_actual_power, predicted_watts, where=(valid_actual_power > predicted_watts), 
                     color='crimson', alpha=0.2, label="Unmodeled gap")
    
    ax2.set_title("Tier 2: Utilization-to-Power Gap", pad=8)
    ax2.set_ylabel("Power (Watts)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right')

    # --- BOTTOM PLOT: Workload Generation (Context) ---
    ax3.bar(time_axis, valid_features_df['qps_gen'], width=1.0, color='indigo', alpha=0.7, label="Generative requests")
    ax3.set_title("Tier 3: Replay Workload", pad=8)
    ax3.set_xlabel("Time (minutes after warmup)")
    ax3.set_ylabel("Generative QPS")
    
    # Configure custom X-axis tick labels to explicitly map back to the timeline
    total_minutes = len(time_axis)
    tick_positions = np.arange(0, total_minutes, step=30)  # Major tick every 30 mins
    # The chart begins at minute 61 of the physical trial.
    # To match the GenTD26 offset, we add `60` (warmup) + `600` (trace offset) = 660.
    tick_labels = [f"Exp Min: {t+60}\n(Trace Min: {t+660})" for t in tick_positions]
    ax3.set_xticks(tick_positions)
    ax3.set_xticklabels(tick_labels)

    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper right')

    plt.tight_layout()
    output_filename = "fig_full_paper_validation.png"
    plt.savefig(output_filename, dpi=300)
    print(f"✅ Master Visualization saved to: {output_filename}")

if __name__ == "__main__":
    visualize_results()

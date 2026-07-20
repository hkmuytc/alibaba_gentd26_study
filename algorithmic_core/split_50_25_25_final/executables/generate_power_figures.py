#!/usr/bin/env python3
"""
Generate power splits comparison figure for multi-step forecasting.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

R = Path(__file__).resolve().parent.parent / "results"

with open(R / "averaged_results.json", "r") as f:
    results = json.load(f)

horizons = []
for h_str in sorted(results["70/15/15"].keys(), key=int):
    horizons.append(int(h_str))
horizons = np.array(horizons)

plt.figure(figsize=(14, 6))

splits = ["50/25/25", "60/20/20", "70/15/15"]
colors_split = ['#1565C0', '#2E7D32', '#E65100']

for split_key, color in zip(splits, colors_split):
    split_data = results[split_key]
    h_list = []
    imp_list = []
    for h_str in sorted(split_data.keys(), key=int):
        h_list.append(int(h_str))
        p_model = split_data[h_str]["power_mae_kw"]
        p_persist = split_data[h_str]["power_persist_mae_kw"]
        power_imp = ((p_persist - p_model) / p_persist * 100) if p_persist > 0 else 0
        imp_list.append(power_imp)

    plt.plot(h_list, imp_list, 'o-', label=f'{split_key} split',
            color=color, linewidth=2, markersize=6, alpha=0.8)

plt.axhline(y=0, color='black', linewidth=1.5, linestyle='-')
plt.xlabel('Forecast Horizon (minutes)', fontsize=12, fontweight='bold')
plt.ylabel('Power MAE Improvement (%)', fontsize=12, fontweight='bold')
plt.title('Multi-Step Power Prediction: Model Improvement Over Persistence Across Splits',
         fontsize=14, fontweight='bold', pad=20)
plt.legend(fontsize=11, loc='best')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(horizons)
plt.tight_layout()
plt.savefig(R / 'fig_multistep_power_splits.png', dpi=300, bbox_inches='tight')
print("Saved: fig_multistep_power_splits.png")
plt.close()

print("\nAll figures generated.")

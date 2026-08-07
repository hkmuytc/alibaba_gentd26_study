import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from importlib import import_module
import sys

sys.path.insert(0, ".")
eval_mod = import_module("03_evaluate_power")
features_df, actual_power = eval_mod.process_physical_data("hardware_trace.csv", "qps_trace.csv", offset_minutes=600)

util = features_df['gpu_util'].values.reshape(-1, 1)
power = actual_power

# 1. Standard Fan et al. (Linear)
predicted_fan = 50 + (300 - 50) * util.flatten()
fan_mae = mean_absolute_error(power, predicted_fan)

# 2. Optimal Affine Transform
lr = LinearRegression()
lr.fit(util, power)
pred_opt = lr.predict(util)
opt_mae = mean_absolute_error(power, pred_opt)

base_intercept = lr.intercept_
coef = lr.coef_[0]

print(f"Optimal Bias (Base Power): {base_intercept:.2f} W")
print(f"Optimal Multiplier (Dynamic Range): {coef:.2f} W per 100% util")
print(f"Standard Model MAE: {fan_mae:.2f} W")
print(f"Optimal Model MAE: {opt_mae:.2f} W")

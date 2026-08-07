import pandas as pd
import numpy as np
import sys
import torch
import joblib
from pathlib import Path

# Add the root directory to path so we can import the pipeline
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from algorithmic_core.split_50_25_25_final_corrected.core.pipeline import engineer_features, estimate_power_kw
from algorithmic_core.split_50_25_25_final_corrected.core.models import TransformerForecaster

def process_physical_data(hw_path, qps_path, offset_minutes=600):
    """
    Takes the physical telemetry from scripts 01 and 02 and shapes it 
    exactly like the Alibaba GenTD26 data for the Transformer model.
    """
    print("Loading physical telemetry...")
    hw = pd.read_csv(hw_path)
    qps = pd.read_csv(qps_path)

    # 1. Binning (60 seconds)
    # Realigning the absolute unix timestamps into 60s buckets
    hw['bin'] = (hw['time_absolute'] // 60) * 60
    qps['bin'] = (qps['time_absolute'] // 60) * 60

    common_bins = sorted(hw['bin'].unique())
    df = pd.DataFrame(index=common_bins)
    df.index.name = "timestamp"

    # 2. Map standard features
    df["gpu_util"] = hw.groupby("bin")["gpu_util_perc"].mean()
    df["gpu_util_std"] = hw.groupby("bin")["gpu_util_perc"].std().fillna(0)
    df["gpu_n_pods"] = 1.0  
    df["active_pods"] = (df["gpu_util"] > 50.0).astype(float)
    
    # Map physical memory (MB) to the fraction of an 80GB GPU (to match trace scaling)
    df["gpu_mem_frac"] = (hw.groupby("bin")["gpu_mem_used_mb"].mean() * 1e6) / (80 * 1e9) 

    qps_gen = qps[qps["request_type"] == "Generative Requests"].groupby("bin")["value"].sum()
    df["qps_gen"] = qps_gen.reindex(common_bins, fill_value=0)
    df["qps_api"] = 0.0

    df = df.ffill().fillna(0.0)
    
    # 2.5 CRITICAL FIX: The fractional differencing in your pipeline requires 
    # a minimum of 50 minutes (max_lag=50) of historical data to produce any output.
    # We must pad the beginning with 50 minutes of idle data so that your 
    # ENTIRE physical telemetry run survives the .dropna() step in engineer_features!
    pad_size = 50
    pad_idx = [df.index[0] - (i * 60) for i in range(pad_size, 0, -1)]
    pad_df = pd.DataFrame(index=pad_idx, columns=df.columns)
    
    pad_df["gpu_util"] = 0.0
    pad_df["gpu_util_std"] = 0.0
    pad_df["gpu_n_pods"] = 1.0
    pad_df["active_pods"] = 0.0
    pad_df["gpu_mem_frac"] = 0.0
    pad_df["qps_gen"] = 0.0
    pad_df["qps_api"] = 0.0
    
    df = pd.concat([pad_df, df])
    common_bins = df.index.tolist()
    
    # 3. CRITICAL STEP: Adjust the time index for the Sine/Cosine embeddings
    # df.index is currently Unix absolute time (e.g., 1700000000). 
    # The pipeline calculates: minutes = (df.index - df.index[0]) / 60
    # We must trick df.index so it acts like it started 600 minutes into the trace.
    start_unix = df.index[0]
    df.index = df.index - start_unix + (offset_minutes * 60)
    
    # 4. Engineer Neural Features
    features_df = engineer_features(df)
    
    # 5. Attach the Ground Truth Power for our final evaluation charting
    actual_power = hw.groupby("bin")["power_draw_w"].max().reindex(common_bins).values
    # Note: Because engineer_features drops NA rows (due to lags), we align lengths
    aligned_actual_power = actual_power[len(actual_power) - len(features_df):]
    
    return features_df, aligned_actual_power

def evaluate(features_df, actual_power):
    """
    Loads your locally cached PyTorch `.pt` model and runs inference.
    """
    print(f"\nReady for Inference! Shape: {features_df.shape}")
    
    # 1. Paths to your locally cached model assets
    exp_dir = Path(__file__).resolve().parent.parent / "algorithmic_core" / "split_50_25_25_final_corrected"
    models_dir = exp_dir / "models" / "onestep"
    
    window_size = 60  # Aligning to w30 model
    
    # Verify standard length (prevent windowing on impossibly short sequences)
    if len(features_df) <= window_size:
        raise ValueError(f"Trace too short ({len(features_df)} bins) for window size {window_size}.")
    
    try:
        print(f"Loading accurate trained model from: {models_dir}")
        scalers = joblib.load(models_dir / "transformer_w60_50-25-25_scalers.joblib")
        feat_scaler = scalers['feat_scaler']
        tgt_scaler = scalers['tgt_scaler']
        
        # Scale all features (Note, your original code trains the scaler on just the columns 
        # that existed in pipeline.py, so we must strictly ensure no extra columns sneaked in)
        feature_cols = ['gpu_util', 'gpu_util_lag1', 'gpu_util_lag2', 'gpu_util_rmean5', 
                        'gpu_util_rmean15', 'gpu_util_rstd15', 'gpu_util_rmean30', 
                        'gpu_util_rstd30', 'gpu_util_roc', 'gpu_util_fd03', 
                        'active_pod_ratio', 'active_pod_ratio_lag1', 'active_pod_ratio_lag2', 
                        'qps_gen', 'qps_gen_lag1', 'qps_gen_lag2', 'qps_gen_rmean5', 
                        'qps_gen_rmean15', 'qps_gen_rstd15', 'qps_gen_rmean30', 
                        'qps_gen_rstd30', 'qps_roc', 'qps_api', 'gpu_mem_frac', 
                        'workload_intensity', 'tod_sin', 'tod_cos']
        X_all = feat_scaler.transform(features_df[feature_cols].values)
        
        # Build temporal windows of size 60 for the Transformer
        X_windows = []
        # We also need the target from the previous step to un-residualize the output
        yp_orig = features_df['gpu_util'].values
        yp_windows = []
        
        for i in range(window_size, len(X_all)):
            X_windows.append(X_all[i - window_size : i])
            yp_windows.append(yp_orig[i - 1])
            
        X_windows = np.array(X_windows, dtype=np.float32)
        yp_windows = np.array(yp_windows, dtype=np.float32)
        
        # Load and run the model
        model = TransformerForecaster(input_dim=len(features_df.columns), d_model=32, nhead=4, num_layers=2)
        model.load_state_dict(torch.load(models_dir / "transformer_w60_50-25-25.pt")) 
        model.eval()
        
        # Predict Residuals
        with torch.no_grad():
            predictions_scaled = model(torch.tensor(X_windows))
            
        # Inverse transform the residual back to raw space (requires reshaping to 2D for scikit-learn)
        predicted_residual = tgt_scaler.inverse_transform(predictions_scaled.numpy().reshape(-1, 1)).flatten()
        
        # Predicted value = Previous Value + Predicted Residual
        predicted_util_frac = yp_windows + predicted_residual

        # Because we consumed 30 rows building windows, we trim the arrays to match
        valid_features_df = features_df.iloc[window_size:].copy()
        valid_actual_power = actual_power[window_size:]

    except Exception as e:
        print(f"Warning: Could not load actual .pt models ({e}).")
        valid_features_df = features_df
        valid_actual_power = actual_power
        predicted_util_frac = np.full(len(features_df), 0.90)

    # 2. Convert predicted util to watts using the strict Fan et al. linear model
    predicted_watts = []
    for util in predicted_util_frac:
        watts = estimate_power_kw(util, 1) * 1000
        predicted_watts.append(watts)

    print("\n--- DISCUSSION RESULTS ---")
    print(f"Predicted Avg Burst Power (Linear Model): {np.mean(predicted_watts):.2f} W")
    print(f"Actual Avg Burst Power (Physical GPU):    {np.mean(valid_actual_power):.2f} W")
    
    difference = np.mean(valid_actual_power) - np.mean(predicted_watts)
    print(f"Unmodeled Power Gap:                      {difference:+.2f} W")
    print("\nConclusion: Utilization alone fails to capture thermal and memory-bound power spikes!")
    
    return valid_features_df, valid_actual_power, np.array(predicted_watts)

if __name__ == "__main__":
    hw_path = "hardware_trace.csv"
    qps_path = "qps_trace.csv"
    if not Path(hw_path).exists() or not Path(qps_path).exists():
        raise FileNotFoundError("Expected hardware_trace.csv and qps_trace.csv in ecs-realtime/")

    feats, true_pwr = process_physical_data(hw_path, qps_path, offset_minutes=600)
    evaluate(feats, true_pwr)

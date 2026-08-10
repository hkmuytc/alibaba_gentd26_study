import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent.parent.parent / "data" / "raw"

GPU_IDLE_W = 50
GPU_MAX_W = 300
GPU_MEM_GB = 80


def load_raw_signals():
    gpu = pd.read_csv(RAW_DIR / "pod_gpu_duty_cycle_anon.csv")
    gpu.columns = ["value", "timestamp_anon", "container_ip"]

    gmem = pd.read_csv(RAW_DIR / "pod_gpu_memory_used_bytes_anon.csv")
    gmem.columns = ["timestamp_anon", "value", "container_ip"]

    qps = pd.read_csv(RAW_DIR / "qps.csv")

    return gpu, gmem, qps


def aggregate_to_cluster(gpu, gmem, qps, bin_sec=60):
    gpu["bin"] = (gpu.timestamp_anon // bin_sec) * bin_sec
    gmem["bin"] = (gmem.timestamp_anon // bin_sec) * bin_sec
    qps["bin"] = (qps.timestamp_anon // bin_sec) * bin_sec

    gpu_agg = gpu.groupby("bin")["value"].agg(["mean", "std", "count"])
    gpu_agg.columns = ["gpu_util", "gpu_util_std", "gpu_n_pods"]

    active_agg = gpu[gpu.value > 0.5].groupby("bin").size()
    active_agg.name = "active_pods"

    gmem_agg = gmem.groupby("bin")["value"].mean() / (GPU_MEM_GB * 1e9)
    gmem_agg.name = "gpu_mem_frac"

    qps_gen = qps[qps.request_type == "Generative Requests"].groupby("bin")["value"].sum()
    qps_gen.name = "qps_gen"
    qps_api = qps[qps.request_type == "API Requests"].groupby("bin")["value"].sum()
    qps_api.name = "qps_api"

    common_bins = sorted(
        set(gpu_agg.index) & set(gmem_agg.index) & set(qps_gen.index)
    )

    df = pd.DataFrame(index=common_bins)
    df.index.name = "timestamp"
    df["gpu_util"] = gpu_agg["gpu_util"].reindex(common_bins)
    df["gpu_util_std"] = gpu_agg["gpu_util_std"].reindex(common_bins)
    df["gpu_n_pods"] = gpu_agg["gpu_n_pods"].reindex(common_bins)
    df["active_pods"] = active_agg.reindex(common_bins, fill_value=0)
    df["gpu_mem_frac"] = gmem_agg.reindex(common_bins)
    df["qps_gen"] = qps_gen.reindex(common_bins)
    df["qps_api"] = qps_api.reindex(common_bins, fill_value=0)

    df = df.ffill()
    return df


def fractional_difference(series, d=0.3, max_lag=50):
    """Causal fractional differencing: fd[t] uses only values up to time t."""
    weights = np.zeros(max_lag)
    weights[0] = 1.0
    for k in range(1, max_lag):
        weights[k] = -weights[k - 1] * (d - k + 1) / k
    vals = series.values
    result = np.full(len(vals), np.nan)
    for t in range(max_lag, len(vals)):
        result[t] = np.dot(weights, vals[t - max_lag + 1 : t + 1][::-1])
    return pd.Series(result, index=series.index)


def engineer_features(df):
    out = pd.DataFrame(index=df.index)

    # --- GPU utilization group (columns 0–9, contiguous for ablation) ---
    gpu = df["gpu_util"] / 100.0
    out["gpu_util"] = gpu
    out["gpu_util_lag1"] = gpu.shift(1)
    out["gpu_util_lag2"] = gpu.shift(2)
    out["gpu_util_rmean5"] = gpu.rolling(5, min_periods=1).mean()
    out["gpu_util_rmean15"] = gpu.rolling(15, min_periods=1).mean()
    out["gpu_util_rstd15"] = gpu.rolling(15, min_periods=1).std()
    out["gpu_util_rmean30"] = gpu.rolling(30, min_periods=1).mean()
    out["gpu_util_rstd30"] = gpu.rolling(30, min_periods=1).std()
    out["gpu_util_roc"] = gpu.diff(1)
    out["gpu_util_fd03"] = fractional_difference(gpu, d=0.3)

    # --- Exogenous workload telemetry (columns 10–26) ---
    out["active_pod_ratio"] = df["active_pods"] / df["gpu_n_pods"].clip(lower=1)
    out["active_pod_ratio_lag1"] = out["active_pod_ratio"].shift(1)
    out["active_pod_ratio_lag2"] = out["active_pod_ratio"].shift(2)
    out["qps_gen"] = df["qps_gen"]
    out["qps_gen_lag1"] = out["qps_gen"].shift(1)
    out["qps_gen_lag2"] = out["qps_gen"].shift(2)
    out["qps_gen_rmean5"] = out["qps_gen"].rolling(5, min_periods=1).mean()
    out["qps_gen_rmean15"] = out["qps_gen"].rolling(15, min_periods=1).mean()
    out["qps_gen_rstd15"] = out["qps_gen"].rolling(15, min_periods=1).std()
    out["qps_gen_rmean30"] = out["qps_gen"].rolling(30, min_periods=1).mean()
    out["qps_gen_rstd30"] = out["qps_gen"].rolling(30, min_periods=1).std()
    out["qps_roc"] = out["qps_gen"].diff(1)
    out["qps_api"] = df["qps_api"]
    out["gpu_mem_frac"] = df["gpu_mem_frac"]
    out["workload_intensity"] = out["active_pod_ratio"] * gpu

    minutes = (df.index - df.index[0]) / 60
    minute_of_day = minutes % 1440
    out["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    out["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)

    out = out.dropna()
    return out


def estimate_power_kw(gpu_util_frac, active_pod_ratio_or_n_gpus, n_gpus=None):
    """
    Strict GPU-only power estimation (Fan et al. 2007).
    """
    if n_gpus is None:
        n_gpus = active_pod_ratio_or_n_gpus
    return n_gpus * (GPU_IDLE_W + (GPU_MAX_W - GPU_IDLE_W) * gpu_util_frac) / 1000


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def prepare_data(feature_df, target_col="gpu_util", window_size=30,
                 train_ratio=0.70, val_ratio=0.15, batch_size=32,
                 residual_mode=True):
    """
    Prepare train/val/test DataLoaders with proper temporal split.

    The target column is always included as an input feature. This gives the
    model access to the most recent target value (e.g. gpu_util[i-1]) in the
    input window, which is critical for residual prediction. There is no data
    leakage because the window X[i-w:i] only contains rows up to i-1, while
    the target y[i] is at row i.

    In residual mode, the target scaler is fit on training RESIDUALS (y[i] - y[i-1]).
    """
    feature_cols = list(feature_df.columns)
    X_all = feature_df[feature_cols].values
    y_all = feature_df[target_col].values.astype(np.float32)

    n = len(X_all)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    warmup = window_size

    def build_windows(start, end):
        Xs, ys, yps = [], [], []
        for i in range(max(start, warmup), end):
            Xs.append(X_all[i - warmup : i])
            ys.append(y_all[i])
            yps.append(y_all[i - 1])
        if not Xs:
            return None, None, None
        return (np.array(Xs, dtype=np.float32),
                np.array(ys, dtype=np.float32),
                np.array(yps, dtype=np.float32))

    X_tr, y_tr, yp_tr = build_windows(0, train_end)
    X_va, y_va, yp_va = build_windows(train_end, val_end)
    X_te, y_te, yp_te = build_windows(val_end, n)

    feat_scaler = StandardScaler()
    n_feat = X_tr.shape[2]
    feat_scaler.fit(X_tr.reshape(-1, n_feat))

    tgt_scaler = StandardScaler()
    if residual_mode:
        tgt_scaler.fit((y_tr - yp_tr).reshape(-1, 1))
    else:
        tgt_scaler.fit(y_tr.reshape(-1, 1))

    def transform(X, y, yp):
        X_s = feat_scaler.transform(X.reshape(-1, n_feat)).reshape(X.shape)
        if residual_mode:
            y_s = tgt_scaler.transform((y - yp).reshape(-1, 1)).flatten()
        else:
            y_s = tgt_scaler.transform(y.reshape(-1, 1)).flatten()
        return X_s, y_s, yp.copy()

    X_tr_s, y_tr_s, yp_tr_s = transform(X_tr, y_tr, yp_tr)
    X_va_s, y_va_s, yp_va_s = transform(X_va, y_va, yp_va)
    X_te_s, y_te_s, yp_te_s = transform(X_te, y_te, yp_te)

    train_ds = TimeSeriesDataset(torch.tensor(X_tr_s), torch.tensor(y_tr_s))
    val_ds = TimeSeriesDataset(torch.tensor(X_va_s), torch.tensor(y_va_s))
    test_ds = TimeSeriesDataset(torch.tensor(X_te_s), torch.tensor(y_te_s))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "feat_scaler": feat_scaler,
        "tgt_scaler": tgt_scaler,
        "feature_cols": feature_cols,
        "window_size": window_size,
        "residual_mode": residual_mode,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "n_test": len(test_ds),
        "yp_test_orig": yp_te,
        "y_test_orig": y_te,
    }

import os
import pandas as pd
import numpy as np
from pathlib import Path

from .loading import clean_processed_power_frame, read_csv_portable

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
EXTERNAL_DIR = Path(__file__).resolve().parents[2] / "data" / "external"
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

# Fan et al. (2007) linear power model parameters
# P = P_idle + (P_max - P_idle) * u
# For GPUs (e.g. NVIDIA A100): P_idle ~ 50W, P_max ~ 300W per GPU
# For CPUs (typical server): P_idle ~ 100W, P_max ~ 250W per server
GPU_P_IDLE = 50.0   # watts
GPU_P_MAX = 300.0   # watts
CPU_P_IDLE = 100.0  # watts
CPU_P_MAX = 250.0   # watts

# Memory subsystem power (simplified model)
MEM_P_IDLE = 10.0   # watts
MEM_P_MAX = 40.0    # watts


def load_genai_gpu_duty_cycle(path=None):
    if path is None:
        path = RAW_DIR / "pod_gpu_duty_cycle_anon.csv"
    df = read_csv_portable(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"timestamp_anon": "timestamp", "value": "gpu_util", "container_ip": "container_id"})
    df["gpu_util"] = pd.to_numeric(df["gpu_util"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["gpu_util", "timestamp"])
    return df


def load_genai_gpu_memory(path=None):
    if path is None:
        path = RAW_DIR / "pod_gpu_memory_used_bytes_anon.csv"
    df = read_csv_portable(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"timestamp_anon": "timestamp", "value": "gpu_mem_bytes", "container_ip": "container_id"})
    df["gpu_mem_bytes"] = pd.to_numeric(df["gpu_mem_bytes"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["gpu_mem_bytes", "timestamp"])
    return df


def load_genai_pod_memory(path=None):
    if path is None:
        path = RAW_DIR / "pod_memory_util_anon.csv"
    df = read_csv_portable(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"timestamp_anon": "timestamp", "value": "mem_util", "container_ip": "container_id"})
    df["mem_util"] = pd.to_numeric(df["mem_util"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["mem_util", "timestamp"])
    return df


def load_genai_qps(path=None):
    if path is None:
        path = RAW_DIR / "qps.csv"
    df = read_csv_portable(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"timestamp_anon": "timestamp", "value": "qps", "container_ip": "container_id"})
    df["qps"] = pd.to_numeric(df["qps"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["qps", "timestamp"])
    return df


def load_gpu_v2020_machine_metrics(path=None):
    if path is None:
        path = EXTERNAL_DIR / "pai_machine_metric.csv"
    cols = [
        "worker_name", "machine", "start_time", "end_time",
        "machine_cpu_iowait", "machine_cpu_kernel", "machine_cpu_usr",
        "machine_gpu", "machine_load_1", "machine_net_receive",
        "machine_num_worker", "machine_cpu"
    ]
    df = read_csv_portable(path, names=cols, header=None)
    for c in ["start_time", "end_time", "machine_cpu", "machine_gpu",
              "machine_cpu_usr", "machine_cpu_kernel", "machine_cpu_iowait",
              "machine_load_1"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["start_time", "machine_cpu"])
    # Use midpoint of time window as timestamp (for raw data display)
    df["timestamp"] = (df["start_time"] + df["end_time"]) / 2
    return df


def spread_v2020_to_time_bins(df, freq_seconds=300, value_cols=None):
    """Spread v2020 instance-level lifetime averages across all time bins
    they overlap with, then average per bin.

    Each row in pai_machine_metric has (start_time, end_time, metric_value)
    where the metric is a lifetime average for that instance. We replicate
    each row into every time bin its [start_time, end_time) spans, then
    take the mean across all instances present in each bin.

    This produces a more honest temporal reconstruction than using the
    midpoint timestamp, though the within-instance variation is still lost.
    """
    if value_cols is None:
        value_cols = ["machine_cpu", "machine_gpu", "machine_cpu_usr",
                      "machine_cpu_kernel", "machine_cpu_iowait",
                      "machine_load_1", "machine_net_receive",
                      "machine_num_worker"]
    value_cols = [c for c in value_cols if c in df.columns]

    df = df.dropna(subset=["start_time", "end_time"]).copy()
    df = df[df["end_time"] > df["start_time"]].reset_index(drop=True)

    # Compute bin indices for start and end of each instance
    start_bins = (df["start_time"].values // freq_seconds).astype(np.int64)
    end_bins = (df["end_time"].values // freq_seconds).astype(np.int64)
    # Number of bins each instance spans
    n_bins = (end_bins - start_bins + 1).astype(np.int64)

    # Build expanded arrays: repeat each row's index n_bins times
    row_indices = np.repeat(np.arange(len(df)), n_bins)
    # Build the bin offset for each expanded row
    bin_offsets = np.concatenate([np.arange(n) for n in n_bins])
    time_bins = (start_bins[row_indices] + bin_offsets) * freq_seconds

    # Build expanded dataframe
    expanded = pd.DataFrame({"timestamp": time_bins})
    for c in value_cols:
        expanded[c] = df[c].values[row_indices]

    # Average per time bin
    cluster = expanded.groupby("timestamp")[value_cols].mean().reset_index()
    cluster = cluster.sort_values("timestamp").reset_index(drop=True)
    return cluster


def shared_time_anchor(*frames, time_col="timestamp"):
    """Return one common timestamp anchor so all sources bin onto the same grid."""
    anchors = []
    for frame in frames:
        if frame is None or time_col not in frame.columns:
            continue
        series = pd.to_numeric(frame[time_col], errors="coerce").dropna()
        if not series.empty:
            anchors.append(int(series.min()))

    if not anchors:
        raise ValueError(f"No valid `{time_col}` values were found to anchor time bins.")

    return min(anchors)


def aggregate_to_cluster_level(df, time_col="timestamp", value_cols=None,
                                freq_seconds=300, agg_func="mean", anchor_time=None):
    df = df.copy()
    # Normalise timestamps relative to one shared anchor so merged sources stay aligned.
    t_anchor = int(df[time_col].min()) if anchor_time is None else int(anchor_time)
    df["time_offset"] = df[time_col] - t_anchor
    df["time_bin"] = (df["time_offset"] // freq_seconds) * freq_seconds + t_anchor

    if value_cols is None:
        value_cols = [c for c in df.columns if c not in [time_col, "container_id", "worker_name", "machine", "time_offset", "time_bin", "start_time", "end_time"]]

    agg_dict = {c: agg_func for c in value_cols}
    cluster = df.groupby("time_bin").agg(agg_dict).reset_index()
    cluster = cluster.rename(columns={"time_bin": "timestamp"})
    cluster = cluster.sort_values("timestamp").reset_index(drop=True)
    return cluster


def estimate_power_gpu(gpu_util_fraction, n_gpus=1):
    """
    Fan et al. (2007) linear power model for GPU.
    gpu_util_fraction: utilization in [0, 1]
    Returns power in kW.
    """
    u = np.clip(gpu_util_fraction, 0, 1)
    power_watts = n_gpus * (GPU_P_IDLE + (GPU_P_MAX - GPU_P_IDLE) * u)
    return power_watts / 1000.0  # kW


def estimate_power_cpu(cpu_util_fraction, n_servers=1):
    """
    Fan et al. (2007) linear power model for CPU servers.
    cpu_util_fraction: utilization in [0, 1]
    Returns power in kW.
    """
    u = np.clip(cpu_util_fraction, 0, 1)
    power_watts = n_servers * (CPU_P_IDLE + (CPU_P_MAX - CPU_P_IDLE) * u)
    return power_watts / 1000.0  # kW


def estimate_power_memory(mem_util_fraction, n_units=1):
    """Linear power model for memory subsystem."""
    u = np.clip(mem_util_fraction, 0, 1)
    power_watts = n_units * (MEM_P_IDLE + (MEM_P_MAX - MEM_P_IDLE) * u)
    return power_watts / 1000.0  # kW


def build_genai_power_series(freq_seconds=300, n_gpus_cluster=100):
    """
    Full pipeline: load GenTD26 traces -> aggregate -> estimate power -> feature engineer.
    Returns a DataFrame with timestamp, features, and estimated power.

    n_gpus_cluster: assumed number of GPUs in the cluster (for scaling power estimate).
    """
    print("Loading GenTD26 traces...")
    gpu_duty = load_genai_gpu_duty_cycle()
    gpu_mem = load_genai_gpu_memory()
    pod_mem = load_genai_pod_memory()
    qps = load_genai_qps()
    anchor_time = shared_time_anchor(gpu_duty, gpu_mem, pod_mem, qps)

    print("Aggregating to cluster level...")
    gpu_agg = aggregate_to_cluster_level(
        gpu_duty,
        value_cols=["gpu_util"],
        freq_seconds=freq_seconds,
        anchor_time=anchor_time,
    )
    gpu_mem_agg = aggregate_to_cluster_level(
        gpu_mem,
        value_cols=["gpu_mem_bytes"],
        freq_seconds=freq_seconds,
        anchor_time=anchor_time,
    )
    pod_mem_agg = aggregate_to_cluster_level(
        pod_mem,
        value_cols=["mem_util"],
        freq_seconds=freq_seconds,
        anchor_time=anchor_time,
    )
    qps_agg = aggregate_to_cluster_level(
        qps,
        value_cols=["qps"],
        freq_seconds=freq_seconds,
        agg_func="sum",
        anchor_time=anchor_time,
    )

    # Merge all on timestamp
    df = gpu_agg
    for other in [gpu_mem_agg, pod_mem_agg, qps_agg]:
        df = pd.merge(df, other, on="timestamp", how="outer")
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Forward-fill then back-fill small gaps
    df = df.ffill().bfill()

    # Convert gpu_util from percentage to fraction if needed
    if df["gpu_util"].max() > 1.0:
        df["gpu_util_frac"] = df["gpu_util"] / 100.0
    else:
        df["gpu_util_frac"] = df["gpu_util"]

    # Convert mem_util from fraction (already 0-1 range in data)
    if df["mem_util"].max() > 1.0:
        df["mem_util_frac"] = df["mem_util"] / 100.0
    else:
        df["mem_util_frac"] = df["mem_util"]

    # Estimate power
    print("Estimating power consumption...")
    df["power_gpu_kw"] = estimate_power_gpu(df["gpu_util_frac"].values, n_gpus=n_gpus_cluster)
    df["power_mem_kw"] = estimate_power_memory(df["mem_util_frac"].values, n_units=n_gpus_cluster)
    df["power_total_kw"] = df["power_gpu_kw"] + df["power_mem_kw"]

    # Normalise GPU memory to a utilization metric (0-1)
    gpu_mem_max = df["gpu_mem_bytes"].max()
    if gpu_mem_max > 0:
        df["gpu_mem_util"] = df["gpu_mem_bytes"] / gpu_mem_max
    else:
        df["gpu_mem_util"] = 0.0

    # Feature engineering
    print("Engineering features...")
    df = add_temporal_features(df, time_col="timestamp")
    df = add_rolling_features(df, target_col="power_total_kw")
    df = add_rate_of_change(df, target_col="power_total_kw")

    df = df.dropna().reset_index(drop=True)
    return df


def build_v2020_power_series(freq_seconds=300, n_machines=100):
    """
    Build power series from the Alibaba GPU v2020 machine metrics.
    This dataset has both CPU and GPU utilization per machine.
    """
    print("Loading GPU v2020 machine metrics...")
    df = load_gpu_v2020_machine_metrics()

    print("Spreading instance metrics across time bins...")
    value_cols = ["machine_cpu", "machine_gpu", "machine_cpu_usr",
                  "machine_cpu_kernel", "machine_cpu_iowait", "machine_load_1"]
    cluster = spread_v2020_to_time_bins(df, freq_seconds=freq_seconds, value_cols=value_cols)

    # CPU and GPU utilization are in percentage (0-100)
    cluster["cpu_util_frac"] = cluster["machine_cpu"].clip(0, 100) / 100.0
    cluster["gpu_util_frac"] = cluster["machine_gpu"].clip(0, 100) / 100.0

    # Estimate power
    print("Estimating power consumption...")
    cluster["power_cpu_kw"] = estimate_power_cpu(cluster["cpu_util_frac"].values, n_servers=n_machines)
    cluster["power_gpu_kw"] = estimate_power_gpu(cluster["gpu_util_frac"].values, n_gpus=n_machines)
    cluster["power_total_kw"] = cluster["power_cpu_kw"] + cluster["power_gpu_kw"]

    # Feature engineering
    print("Engineering features...")
    cluster = add_temporal_features(cluster, time_col="timestamp")
    cluster = add_rolling_features(cluster, target_col="power_total_kw")
    cluster = add_rate_of_change(cluster, target_col="power_total_kw")

    cluster = cluster.dropna().reset_index(drop=True)
    return cluster


def add_temporal_features(df, time_col="timestamp"):
    """Add cyclical time-of-day and day-of-week encodings."""
    df = df.copy()
    # Convert to datetime
    dt = pd.to_datetime(df[time_col], unit="s")
    # Time of day in seconds
    seconds_in_day = dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second
    df["hour_sin"] = np.sin(2 * np.pi * seconds_in_day / 86400)
    df["hour_cos"] = np.cos(2 * np.pi * seconds_in_day / 86400)
    # Day of week
    day_of_week = dt.dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)
    df["hour"] = dt.dt.hour
    return df


def add_rolling_features(df, target_col="power_total_kw", windows=None):
    """Add rolling mean and std over specified windows."""
    if windows is None:
        windows = [12, 72]  # 12 * 5min = 1hr, 72 * 5min = 6hr
    for w in windows:
        df[f"{target_col}_roll_mean_{w}"] = df[target_col].rolling(window=w, min_periods=1).mean()
        df[f"{target_col}_roll_std_{w}"] = df[target_col].rolling(window=w, min_periods=1).std().fillna(0)
    return df


def add_rate_of_change(df, target_col="power_total_kw"):
    """Add rate of change (first difference) feature."""
    df = df.copy()
    df[f"{target_col}_roc"] = df[target_col].diff().fillna(0)
    return df


def aggregate_genai_sources(freq_seconds: int = 300) -> tuple:
    """Load all GenAI raw sources and merge them into a cluster-level time-binned DataFrame.

    Returns
    -------
    agg_df : pd.DataFrame
        Merged cluster-level DataFrame with columns:
        timestamp, gpu_util, gpu_mem_bytes, mem_util, qps.
    meta : dict
        Aggregation metadata: n_records, n_containers, n_bins.
    """
    gpu = load_genai_gpu_duty_cycle()
    gmem = load_genai_gpu_memory()
    pmem = load_genai_pod_memory()
    qps_raw = load_genai_qps()
    anchor_time = shared_time_anchor(gpu, gmem, pmem, qps_raw)

    gpu_agg = aggregate_to_cluster_level(
        gpu, value_cols=["gpu_util"], freq_seconds=freq_seconds, anchor_time=anchor_time,
    )
    gmem_agg = aggregate_to_cluster_level(
        gmem, value_cols=["gpu_mem_bytes"], freq_seconds=freq_seconds, anchor_time=anchor_time,
    )
    pmem_agg = aggregate_to_cluster_level(
        pmem, value_cols=["mem_util"], freq_seconds=freq_seconds, anchor_time=anchor_time,
    )
    qps_agg = aggregate_to_cluster_level(
        qps_raw, value_cols=["qps"], freq_seconds=freq_seconds, agg_func="sum", anchor_time=anchor_time,
    )

    agg_df = gpu_agg
    for other in [gmem_agg, pmem_agg, qps_agg]:
        agg_df = pd.merge(agg_df, other, on="timestamp", how="outer")
    agg_df = agg_df.sort_values("timestamp").ffill().bfill().reset_index(drop=True)

    meta = {
        "n_records": len(gpu),
        "n_containers": gpu["container_id"].nunique(),
        "n_bins": len(agg_df),
    }
    return agg_df, meta


def aggregate_v2020_sources(freq_seconds: int = 300, value_cols=None) -> tuple:
    """Load GPU v2020 machine metrics and aggregate to cluster-level time bins.

    Returns
    -------
    agg_df : pd.DataFrame
        Time-binned cluster averages for the specified value columns.
    meta : dict
        Aggregation metadata: n_records, n_machines, n_bins.
    """
    if value_cols is None:
        value_cols = [
            "machine_cpu", "machine_gpu", "machine_cpu_usr",
            "machine_cpu_kernel", "machine_cpu_iowait", "machine_load_1",
        ]
    raw = load_gpu_v2020_machine_metrics()
    agg_df = spread_v2020_to_time_bins(raw, freq_seconds=freq_seconds, value_cols=value_cols)
    meta = {
        "n_records": len(raw),
        "n_machines": raw["machine"].nunique(),
        "n_bins": len(agg_df),
    }
    return agg_df, meta


def estimate_cluster_power(
    agg_df,
    dataset: str,
    n_units: int,
    gpu_idle: float = GPU_P_IDLE,
    gpu_max: float = GPU_P_MAX,
    cpu_idle: float = CPU_P_IDLE,
    cpu_max: float = CPU_P_MAX,
    mem_idle: float = MEM_P_IDLE,
    mem_max: float = MEM_P_MAX,
):
    """Estimate cluster power consumption using the Fan et al. linear power model.

    Supports two dataset schemas:
      - ``"genai"``: uses ``gpu_util`` and ``mem_util`` columns.
      - ``"gpu_v2020"``: uses ``machine_cpu`` and ``machine_gpu`` columns.

    Parameters
    ----------
    agg_df : pd.DataFrame
        Cluster-level aggregated utilisation DataFrame.
    dataset : str
        Dataset identifier, one of ``"genai"`` or ``"gpu_v2020"``.
    n_units : int
        Number of GPUs (genai) or machines (gpu_v2020) in the cluster.
    gpu_idle, gpu_max : float
        Idle and max GPU power per unit (watts).
    cpu_idle, cpu_max : float
        Idle and max CPU power per unit (watts). Only used for ``"gpu_v2020"``.
    mem_idle, mem_max : float
        Idle and max memory power per unit (watts). Only used for ``"genai"``.

    Returns
    -------
    pd.DataFrame
        Copy of *agg_df* with added power columns including ``power_total_kw``.
    """
    power_df = agg_df.copy()

    if dataset == "genai":
        gpu_col = "gpu_util"
        power_df["gpu_util_frac"] = (
            power_df[gpu_col] / 100.0 if power_df[gpu_col].max() > 1.0 else power_df[gpu_col]
        )
        power_df["mem_util_frac"] = (
            power_df["mem_util"] / 100.0 if power_df["mem_util"].max() > 1.0 else power_df["mem_util"]
        )
        u_gpu = np.clip(power_df["gpu_util_frac"].values, 0, 1)
        u_mem = np.clip(power_df["mem_util_frac"].values, 0, 1)
        power_df["power_gpu_kw"] = n_units * (gpu_idle + (gpu_max - gpu_idle) * u_gpu) / 1000.0
        power_df["power_mem_kw"] = n_units * (mem_idle + (mem_max - mem_idle) * u_mem) / 1000.0
        power_df["power_total_kw"] = power_df["power_gpu_kw"] + power_df["power_mem_kw"]
        gpu_mem_max = power_df["gpu_mem_bytes"].max()
        power_df["gpu_mem_util"] = power_df["gpu_mem_bytes"] / gpu_mem_max if gpu_mem_max > 0 else 0.0

    elif dataset == "gpu_v2020":
        power_df["cpu_util_frac"] = power_df["machine_cpu"].clip(0, 100) / 100.0
        power_df["gpu_util_frac"] = power_df["machine_gpu"].clip(0, 100) / 100.0
        u_cpu = np.clip(power_df["cpu_util_frac"].values, 0, 1)
        u_gpu = np.clip(power_df["gpu_util_frac"].values, 0, 1)
        power_df["power_cpu_kw"] = n_units * (cpu_idle + (cpu_max - cpu_idle) * u_cpu) / 1000.0
        power_df["power_gpu_kw"] = n_units * (gpu_idle + (gpu_max - gpu_idle) * u_gpu) / 1000.0
        power_df["power_total_kw"] = power_df["power_cpu_kw"] + power_df["power_gpu_kw"]

    return power_df


def engineer_power_features(power_df, roll_windows=None):
    """Apply the full feature-engineering chain to a power-estimated DataFrame.

    Adds cyclical temporal encodings, rolling statistics, and rate-of-change
    features to *power_df*, then drops identifier columns and NaN rows.

    Parameters
    ----------
    power_df : pd.DataFrame
        Output of :func:`estimate_cluster_power` — must contain
        ``timestamp`` and ``power_total_kw``.
    roll_windows : list[int] | None
        Rolling window sizes in time-steps. Defaults to ``[12, 72]``
        (1 h and 6 h at 5-min bins).

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame with NaN rows removed.
    """
    from .loading import clean_processed_power_frame  # avoid circular at module level

    if roll_windows is None:
        roll_windows = [12, 72]

    feat_df = power_df.copy()
    feat_df = add_temporal_features(feat_df, time_col="timestamp")
    feat_df = add_rolling_features(feat_df, target_col="power_total_kw", windows=roll_windows)
    feat_df = add_rate_of_change(feat_df, target_col="power_total_kw")
    feat_df = clean_processed_power_frame(feat_df)
    feat_df = feat_df.dropna().reset_index(drop=True)
    return feat_df


def save_processed_dataset(feat_df, dataset_name: str, freq_seconds: int = 300) -> "Path":
    """Persist a processed feature DataFrame to ``data/processed/``.

    The file is named ``<dataset_name>_<freq_seconds>s.csv`` so that
    :func:`~src.data_processing.loading.load_processed_datasets` can
    discover it automatically on future runs.

    Returns
    -------
    pathlib.Path
        Absolute path of the saved CSV file.
    """
    save_dir = PROCESSED_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{dataset_name}_{freq_seconds}s.csv"
    feat_df.to_csv(save_path, index=False)
    return save_path


def process_and_save(dataset_name="genai", freq_seconds=300, **kwargs):
    """Process a dataset and save to the processed directory."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if dataset_name == "genai":
        df = build_genai_power_series(freq_seconds=freq_seconds, **kwargs)
    elif dataset_name == "gpu_v2020":
        df = build_v2020_power_series(freq_seconds=freq_seconds, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    df = clean_processed_power_frame(df)
    out_path = PROCESSED_DIR / f"{dataset_name}_{freq_seconds}s.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved processed data to {out_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    process_and_save("genai")
    process_and_save("gpu_v2020")

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.dashboard.nav import Page
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import torch
from st_aggrid import AgGrid, GridOptionsBuilder

from src.data_processing.pipeline import (
    load_genai_gpu_duty_cycle,
    load_genai_gpu_memory,
    load_genai_pod_memory,
    load_genai_qps,
    load_gpu_v2020_machine_metrics,
    aggregate_genai_sources,
    aggregate_v2020_sources,
    estimate_cluster_power,
    engineer_power_features,
    save_processed_dataset,
    GPU_P_IDLE, GPU_P_MAX, CPU_P_IDLE, CPU_P_MAX, MEM_P_IDLE, MEM_P_MAX,
    RAW_DIR, EXTERNAL_DIR,
)
from src.models.architectures import get_model, MODEL_REGISTRY, build_model_kwargs
from src.evaluation.trainer import prepare_data, train_model, evaluate_model, save_model
from src.data_processing.loading import load_processed_datasets
from src.models.inference import (
    autoregressive_forecast,
    forecast_metrics,
    format_model_bundle_caption,
    load_model_bundle,
)
from src.evaluation.symbolic_workflow import (
    SymbolicRunConfig,
    default_symbolic_exogenous,
    parse_positive_ints,
    run_symbolic_multistep,
    run_symbolic_regression,
    symbolic_exogenous_candidates,
)
from src.dashboard.theme import apply_dashboard_theme

GENAI_AVAILABLE = (RAW_DIR / "pod_gpu_duty_cycle_anon.csv").exists()
V2020_AVAILABLE = (EXTERNAL_DIR / "pai_machine_metric.csv").exists()
MODELS_DIR = PROJECT_ROOT / "models" / "saved"

DATASET_LABELS = {
    "genai": "GenTD26 (Primary)",
    "gpu_v2020": "GPU v2020 (Supplementary)",
}

def main():
    st.set_page_config(
        page_title="DC Power Demand Forecasting", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    apply_dashboard_theme()

    init_session_state()
    page = render_sidebar()

    render_selected_page(page)

def init_session_state():
    for key in [
        "raw_data", "aggregated_data", "power_data", "final_data",
        "training_results", "pipeline_params"
    ]:
        if key not in st.session_state:
            st.session_state[key] = {}
    
    if "device" not in st.session_state:
        st.session_state.device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )

def render_sidebar(page_title):
    st.sidebar.title(page_title)
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigation",
        Page.labels(),
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Compute:** `{st.session_state.device}`")
    st.sidebar.markdown(f"**GenTD26 data:** {'Ready' if GENAI_AVAILABLE else 'Missing'}")
    st.sidebar.markdown(f"**GPU v2020 data:** {'Ready' if V2020_AVAILABLE else 'Missing'}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Pipeline State**")
    for ds in ["genai", "gpu_v2020"]:
        if ds in st.session_state.final_data:
            st.sidebar.markdown(f"  `{ds}`: {len(st.session_state.final_data[ds])} rows")

    return page


def render_selected_page(page: str) -> None:
    _PAGE_HANDLERS: dict[str, callable] = {
        Page.RAW_DATA.value:        page_raw_data,
        Page.PROCESSING.value:      page_processing_pipeline,
        Page.TRAIN_EVALUATE.value:  page_train_and_evaluate,
        Page.FORECAST_REPLAY.value: page_forecast_replay,
        Page.SYMBOLIC.value:        page_symbolic_regression,
        Page.TRANSFER.value:        page_transfer_learning,
    }
    handler = _PAGE_HANDLERS.get(page)
    if handler is None:
        st.error(f"Unknown page: {page!r}")
        return
    handler()

def page_raw_data():
    st.title("Raw Data")

    dataset = st.selectbox(
        "Select Dataset", ["genai", "gpu_v2020"],
        format_func=lambda x: DATASET_LABELS[x]
    )

    if dataset == "genai":
        _show_genai_raw()
    elif dataset == "gpu_v2020":
        _show_v2020_raw()

def _show_genai_raw():
    if not GENAI_AVAILABLE:
        st.error("GenTD26 data files not found in data/raw/")
        st.stop()

    st.markdown("### GenTD26")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### GPU Duty Cycle")
        st.caption("How busy each GPU is (0-100%)")
        gpu_dataset = load_genai_gpu_duty_cycle()
        st.session_state.raw_data["genai_gpu"] = gpu_dataset
        ts_range = (gpu_dataset["timestamp"].max() - gpu_dataset["timestamp"].min()) / 3600
        st.caption(f"{len(gpu_dataset):,} records | {gpu_dataset['container_id'].nunique()} containers | {ts_range:.1f} hrs")
        _show_grid(gpu_dataset, column_tooltips=GENAI_COL_DESCRIPTIONS)

    with col2:
        st.markdown("#### GPU Memory Usage")
        st.caption("GPU memory in bytes per container")
        gpumem_dataset = load_genai_gpu_memory()
        st.session_state.raw_data["genai_gmem"] = gpumem_dataset
        st.caption(f"{len(gpumem_dataset):,} records | {gpumem_dataset['container_id'].nunique()} containers | max {gpumem_dataset['gpu_mem_bytes'].max() / 1e9:.1f} GB")
        _show_grid(gpumem_dataset, column_tooltips=GENAI_COL_DESCRIPTIONS)

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("#### Container Memory Utilisation")
        st.caption("System memory usage as a fraction (0-1)")
        mem_dataset = load_genai_pod_memory()
        st.session_state.raw_data["genai_pmem"] = mem_dataset
        st.caption(f"{len(mem_dataset):,} records | {mem_dataset['container_id'].nunique()} containers | avg {mem_dataset['mem_util'].mean():.2%}")
        _show_grid(mem_dataset, column_tooltips=GENAI_COL_DESCRIPTIONS)

    with col4:
        st.markdown("#### Queries Per Second (QPS)")
        st.caption("Incoming request rate to the serving system")
        qps_dataset = load_genai_qps()
        st.session_state.raw_data["genai_qps"] = qps_dataset
        st.caption(f"{len(qps_dataset):,} records | {qps_dataset['container_id'].nunique()} containers | max QPS {qps_dataset['qps'].max():.2f}")
        _show_grid(qps_dataset, column_tooltips=GENAI_COL_DESCRIPTIONS)

    st.markdown("---")
    st.markdown("#### Cluster-Level Utilisation Over Time")

    _visualize_5min_aggregate(gpu_dataset, mem_dataset, qps_dataset)

def _show_v2020_raw():
    if not V2020_AVAILABLE:
        st.error("GPU v2020 data not found in data/external/")
        st.stop()

    st.markdown("### GPU v2020")

    v2020 = load_gpu_v2020_machine_metrics()
    st.session_state.raw_data["gpu_v2020"] = v2020

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(v2020):,}")
    col2.metric("Unique Machines", f"{v2020['machine'].nunique()}")
    ts_range = (v2020["timestamp"].max() - v2020["timestamp"].min()) / 3600
    col3.metric("Time Span", f"{ts_range:.1f} hours")
    col4.metric("Avg CPU Util", f"{v2020['machine_cpu'].mean():.1f}%")

    st.caption("Hover over any column header to see its full description.")
    _show_grid(v2020, column_tooltips=V2020_COL_DESCRIPTIONS)

    st.markdown("---")
    st.markdown("#### Cluster-Level Utilisation Over Time")
    _visualize_v2020_5min_aggregate(v2020)

def page_processing_pipeline():
    st.title("Processing Pipeline")
    dataset = st.selectbox(
        "Dataset", ["genai", "gpu_v2020"],
        format_func=lambda x: DATASET_LABELS[x]
    )

    _step_aggregation(dataset)

    if dataset in st.session_state.aggregated_data:
        freq = st.session_state.pipeline_params.get(dataset, {}).get("freq", 300)
        _step_power_estimation(dataset, freq)

        if dataset in st.session_state.power_data:
            _step_feature_engineering(dataset, freq)

def _visualize_5min_aggregate(gpu_dataset, mem_dataset, qps_dataset):
    gpu_ts = gpu_dataset.groupby(gpu_dataset["timestamp"] // 300 * 300)["gpu_util"].mean().reset_index()
    mem_ts = mem_dataset.groupby(mem_dataset["timestamp"] // 300 * 300)["mem_util"].mean().reset_index()
    qps_ts = qps_dataset.groupby(qps_dataset["timestamp"] // 300 * 300)["qps"].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(gpu_ts["timestamp"], unit="s"),
        y=gpu_ts["gpu_util"], mode="lines", name="GPU Util (%)"
    ))
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(mem_ts["timestamp"], unit="s"),
        y=mem_ts["mem_util"] * 100, mode="lines", name="Mem Util (%)"
    ))
    fig.add_trace(go.Scatter(
        x=pd.to_datetime(qps_ts["timestamp"], unit="s"),
        y=qps_ts["qps"], mode="lines", name="QPS (total)", yaxis="y2"
    ))
    fig.update_layout(
        height=400,
        yaxis=dict(title="Utilisation %"),
        yaxis2=dict(title="QPS", overlaying="y", side="right"),
        legend=dict(x=0, y=1.12, orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

def _visualize_v2020_5min_aggregate(v2020):
    cpu_ts = v2020.groupby(v2020["timestamp"] // 300 * 300)["machine_cpu"].mean().reset_index()
    gpu_ts = v2020.groupby(v2020["timestamp"] // 300 * 300)["machine_gpu"].mean().reset_index()
    load_ts = v2020.groupby(v2020["timestamp"] // 300 * 300)["machine_load_1"].mean().reset_index()

    t_min = cpu_ts["timestamp"].min()
    cpu_hrs = (cpu_ts["timestamp"] - t_min) / 3600
    gpu_hrs = (gpu_ts["timestamp"] - t_min) / 3600
    load_hrs = (load_ts["timestamp"] - t_min) / 3600

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cpu_hrs, y=cpu_ts["machine_cpu"], mode="lines", name="CPU Util (%)"))
    fig.add_trace(go.Scatter(x=gpu_hrs, y=gpu_ts["machine_gpu"], mode="lines", name="GPU Util (%)"))
    fig.add_trace(go.Scatter(x=load_hrs, y=load_ts["machine_load_1"], mode="lines", name="Load Avg (1m)", yaxis="y2"))
    fig.update_layout(
        height=400,
        xaxis=dict(title="Elapsed Time (hours)"),
        yaxis=dict(title="Utilisation %"),
        yaxis2=dict(title="Load Average", overlaying="y", side="right"),
        legend=dict(x=0, y=1.12, orientation="h"),
    )
    st.plotly_chart(fig, use_container_width=True)

def _step_aggregation(dataset):
    st.markdown("---")
    st.subheader("Step 1: Aggregate to Cluster Level")

    col1, col2 = st.columns([2, 3])
    with col1:
        freq = st.select_slider(
            "Time bin width", [60, 120, 300, 600, 900, 1800],
            value=300, format_func=lambda x: f"{x//60} min"
        )

    if st.button("Run Aggregation", key="agg_btn"):
        with st.spinner("Aggregating..."):
            if dataset == "genai":
                agg_df = _aggregate_genai(freq)
            else:
                agg_df = _aggregate_v2020(freq)

            st.session_state.aggregated_data[dataset] = agg_df
            st.session_state.pipeline_params[dataset] = {"freq": freq}

    if dataset in st.session_state.aggregated_data:
        agg_df = st.session_state.aggregated_data[dataset]
        freq = st.session_state.pipeline_params[dataset]["freq"]

        with col2:
            st.metric("Output Rows", f"{len(agg_df):,}")
            st.metric("Columns", f"{len(agg_df.columns)}")
            duration_hrs = (agg_df["timestamp"].max() - agg_df["timestamp"].min()) / 3600
            st.metric("Time Span", f"{duration_hrs:.1f} hrs")

        _plot_aggregated(agg_df, dataset, freq)

        with st.expander("Aggregated Data Preview"):
            _show_grid(agg_df)


def _aggregate_genai(freq):
    agg_df, meta = aggregate_genai_sources(freq)
    st.info(
        f"Aggregated {meta['n_records']:,} records from {meta['n_containers']} containers "
        f"into {meta['n_bins']} time bins ({freq//60}-min)."
    )
    return agg_df

def _aggregate_v2020(freq):
    agg_df, meta = aggregate_v2020_sources(freq)
    st.info(
        f"Spread {meta['n_records']:,} instance records from {meta['n_machines']} machines "
        f"across their lifetimes into {meta['n_bins']} time bins ({freq//60}-min). "
        f"Each instance's metrics are replicated into every bin it overlaps."
    )
    return agg_df

def _plot_aggregated(agg_df, dataset, freq):
    x = pd.to_datetime(agg_df["timestamp"], unit="s")
    fig = go.Figure()
    if dataset == "genai":
        fig.add_trace(go.Scatter(x=x, y=agg_df["gpu_util"], name="GPU Util %", mode="lines"))
        fig.add_trace(go.Scatter(x=x, y=agg_df["mem_util"] * 100, name="Memory Util %", mode="lines"))
    else:
        fig.add_trace(go.Scatter(x=x, y=agg_df["machine_cpu"], name="CPU %", mode="lines"))
        fig.add_trace(go.Scatter(x=x, y=agg_df["machine_gpu"], name="GPU %", mode="lines"))
    fig.update_layout(
        title=f"Cluster-Level Metrics ({freq//60}-min bins)",
        yaxis_title="Utilisation %", height=350
    )
    st.plotly_chart(fig, use_container_width=True)

def _step_power_estimation(dataset, freq):
    st.markdown("---")
    st.subheader("Step 2: Estimate Power Consumption")

    agg_df = st.session_state.aggregated_data[dataset]

    col1, col2, col3 = st.columns(3)
    with col1:
        n_units = st.number_input("Number of GPUs/Machines in cluster", 10, 2000, 100, step=10)
    with col2:
        gpu_idle = st.number_input("GPU Idle Power (W)", 10, 200, int(GPU_P_IDLE), step=10)
        gpu_max = st.number_input("GPU Max Power (W)", 100, 700, int(GPU_P_MAX), step=10)
    with col3:
        if dataset == "gpu_v2020":
            cpu_idle = st.number_input("CPU Idle Power (W)", 10, 200, int(CPU_P_IDLE), step=10)
            cpu_max = st.number_input("CPU Max Power (W)", 100, 500, int(CPU_P_MAX), step=10)
        else:
            cpu_idle, cpu_max = CPU_P_IDLE, CPU_P_MAX
        mem_idle = st.number_input("Memory Idle Power (W)", 0, 50, int(MEM_P_IDLE), step=5)
        mem_max = st.number_input("Memory Max Power (W)", 10, 100, int(MEM_P_MAX), step=5)

    if st.button("Estimate Power", key="power_btn"):
        power_df = estimate_cluster_power(
            agg_df, dataset, n_units, gpu_idle, gpu_max,
            cpu_idle, cpu_max, mem_idle, mem_max
        )
        st.session_state.power_data[dataset] = power_df
        st.session_state.pipeline_params[dataset]["n_units"] = n_units
        st.success(
            f"Power estimated: {len(power_df)} steps, {n_units} units, "
            f"avg {power_df['power_total_kw'].mean():.1f} kW"
        )

    if dataset in st.session_state.power_data:
        _show_power_results(dataset)



def _show_power_results(dataset):
    power_df = st.session_state.power_data[dataset]
    n_units = st.session_state.pipeline_params[dataset].get("n_units", 100)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{len(power_df):,}")
    col2.metric("Avg Power", f"{power_df['power_total_kw'].mean():.1f} kW")
    col3.metric("Peak Power", f"{power_df['power_total_kw'].max():.1f} kW")
    col4.metric("Min Power", f"{power_df['power_total_kw'].min():.1f} kW")

    x = pd.to_datetime(power_df["timestamp"], unit="s")
    fig = go.Figure()
    power_cols = [c for c in power_df.columns if c.startswith("power_") and c != "power_total_kw"]
    for c in power_cols:
        label = c.replace("power_", "").replace("_kw", "").upper()
        fig.add_trace(go.Scatter(
            x=x, y=power_df[c], mode="lines", name=label, stackgroup="power"
        ))
    fig.add_trace(go.Scatter(
        x=x, y=power_df["power_total_kw"], mode="lines",
        name="Total", line=dict(color="black", width=2, dash="dash")
    ))
    fig.update_layout(title=f"Estimated Power ({n_units} units)", yaxis_title="Power (kW)", height=400)
    st.plotly_chart(fig, use_container_width=True)


def _step_feature_engineering(dataset, freq):
    st.markdown("---")
    st.subheader("Step 3: Feature Engineering")

    power_df = st.session_state.power_data[dataset]

    col1, col2 = st.columns(2)
    with col1:
        roll_window_1 = st.slider("Rolling Window 1 (steps)", 3, 36, 12)
        st.caption(f"= {roll_window_1 * freq // 60} minutes")
    with col2:
        roll_window_2 = st.slider("Rolling Window 2 (steps)", 12, 144, 72)
        st.caption(f"= {roll_window_2 * freq // 60} minutes")

    if st.button("Engineer Features", key="feat_btn"):
        feat_df = engineer_power_features(power_df, roll_windows=[roll_window_1, roll_window_2])
        save_path = save_processed_dataset(feat_df, dataset)

        st.session_state.final_data[dataset] = feat_df
        st.session_state.pipeline_params[dataset]["roll_windows"] = [roll_window_1, roll_window_2]

        new_cols = [c for c in feat_df.columns if c not in power_df.columns]
        st.success(f"Added {len(new_cols)} features -- final: "
                   f"{len(feat_df)} rows x {len(feat_df.columns)} columns. "
                   f"Saved to `{save_path.name}`.")

    if dataset in st.session_state.final_data:
        _show_feature_results(dataset)


def _show_feature_results(dataset):
    feat_df = st.session_state.final_data[dataset]

    st.markdown("#### Feature Correlation with Target")
    exclude = {"timestamp"}
    feature_cols = [c for c in feat_df.select_dtypes(include=[np.number]).columns
                    if c not in exclude and c != "power_total_kw"]
    corrs = feat_df[feature_cols + ["power_total_kw"]].corr()["power_total_kw"].drop("power_total_kw")
    corrs = corrs.sort_values(ascending=True)
    fig = px.bar(
        x=corrs.values, y=corrs.index, orientation="h",
        title="Feature Correlation with power_total_kw",
        labels={"x": "Pearson Correlation", "y": "Feature"},
        color=corrs.values, color_continuous_scale="RdBu_r", range_color=[-1, 1]
    )
    fig.update_layout(height=max(300, len(corrs) * 25))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Final Dataset Preview"):
        _show_grid(feat_df)
        st.download_button(
            "Download CSV", feat_df.to_csv(index=False),
            file_name=f"{dataset}_processed.csv", mime="text/csv"
        )

def page_sliding_window_explorer():
    st.title("Sliding Window Explorer")
    st.markdown(
        "Interactively visualise how the **sliding window approach** converts "
        "a time series into training samples for the forecasting models."
    )

    available = load_processed_datasets(st.session_state.final_data)

    if not available:
        st.warning(
            "No processed data yet. Run the **Processing Pipeline** first, "
            "or place a CSV in `data/processed/`."
        )
        st.stop()

    dataset_key = st.selectbox("Dataset", list(available.keys()))
    df = available[dataset_key].copy()

    target_col = "power_total_kw"
    exclude = {"timestamp", "container_id", "worker_name", "machine",
               "start_time", "end_time", "time_offset"}
    feature_cols = [
        c for c in df.columns
        if c not in exclude and c != target_col
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    # ---- parameters -------------------------------------------------
    st.markdown("---")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        window_size = st.slider(
            "Window size (steps)", 4, min(72, len(df) // 3), 24,
            help="Number of consecutive time steps the model sees as input."
        )
    with col_p2:
        train_ratio = st.slider("Train / Test split", 0.5, 0.9, 0.8, step=0.05)
    with col_p3:
        st.metric("Total rows", f"{len(df):,}")
        split_idx = int(len(df) * train_ratio)
        st.metric("Train / Test", f"{split_idx} / {len(df) - split_idx}")

    max_pos = len(df) - window_size - 1
    has_ts = "timestamp" in df.columns

    st.markdown("---")
    st.subheader("1 \u2014 How a sliding window creates one sample")
    st.markdown(
        "Drag the slider to move the window across the time series. "
        "The **blue shaded region** is the input the model receives; "
        "the **red dot** is the target value it must predict."
    )

    pos = st.slider("Window position", 0, max_pos, 0, key="win_pos")

    win_start = pos
    win_end = pos + window_size  # exclusive end of input window
    target_idx = win_end          # the value to predict

    x_vals = (
        pd.to_datetime(df["timestamp"], unit="s").values if has_ts
        else np.arange(len(df))
    )

    fig = go.Figure()
    # full series faintly
    fig.add_trace(go.Scatter(
        x=x_vals, y=df[target_col], mode="lines",
        line=dict(color="lightgrey", width=1), name="Full series", showlegend=True,
    ))
    # input window
    fig.add_trace(go.Scatter(
        x=x_vals[win_start:win_end], y=df[target_col].iloc[win_start:win_end],
        mode="lines+markers", line=dict(color="#1f77b4", width=2),
        marker=dict(size=4), name=f"Input window ({window_size} steps)",
    ))
    # shaded region
    fig.add_vrect(
        x0=x_vals[win_start], x1=x_vals[win_end - 1],
        fillcolor="rgba(31,119,180,0.10)", line_width=0,
    )
    # target point
    fig.add_trace(go.Scatter(
        x=[x_vals[target_idx]], y=[df[target_col].iloc[target_idx]],
        mode="markers", marker=dict(color="red", size=12, symbol="x"),
        name="Target (predict this)",
    ))
    # train/test boundary
    if has_ts:
        split_ts = pd.Timestamp(x_vals[split_idx]).isoformat()
        fig.add_vline(x=split_ts, line_dash="dash", line_color="green")
        fig.add_annotation(
            x=split_ts, y=1, yref="paper",
            text="Train | Test", showarrow=False, xanchor="left"
        )
    fig.update_layout(
        height=370,
        yaxis_title=target_col,
        xaxis_title="Time" if has_ts else "Step",
        margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    # context info
    region = "TRAIN" if target_idx < split_idx else "TEST"
    ts_label = ""
    if has_ts:
        t0 = pd.to_datetime(df["timestamp"].iloc[win_start], unit="s")
        t1 = pd.to_datetime(df["timestamp"].iloc[win_end - 1], unit="s")
        tt = pd.to_datetime(df["timestamp"].iloc[target_idx], unit="s")
        ts_label = f" &nbsp;|&nbsp; Input: **{t0} \u2192 {t1}** &nbsp;|&nbsp; Target time: **{tt}**"
    st.markdown(
        f"Position **{pos}** of {max_pos} &nbsp;|&nbsp; "
        f"Partition: **{region}**{ts_label}"
    )

    st.markdown("---")
    st.subheader("2 \u2014 Input feature matrix for this window")
    st.markdown(
        f"This is exactly what the model receives: a **{window_size} \u00d7 {len(feature_cols)}** "
        "matrix of feature values (one row per time step)."
    )

    feat_display = st.multiselect(
        "Features to show", feature_cols, default=feature_cols[:6],
        help="Select which features to display in the matrix and heatmap."
    )
    if not feat_display:
        feat_display = feature_cols[:6]

    window_df = df.iloc[win_start:win_end][feat_display].copy()
    window_df.index = [f"t-{window_size - 1 - i}" for i in range(window_size)]

    col_tbl, col_heat = st.columns([1, 1])
    with col_tbl:
        st.markdown("**Raw values**")
        st.dataframe(window_df.style.format("{:.4f}"), height=min(400, 35 * window_size + 40))
    with col_heat:
        st.markdown("**Heatmap (normalised per feature)**")
        normed = window_df.copy()
        for c in normed.columns:
            cmin, cmax = normed[c].min(), normed[c].max()
            normed[c] = (normed[c] - cmin) / (cmax - cmin + 1e-9)
        fig_heat = px.imshow(
            normed.values, x=normed.columns, y=normed.index,
            color_continuous_scale="Blues", aspect="auto",
            labels=dict(x="Feature", y="Time step", color="Normalised"),
        )
        fig_heat.update_layout(height=min(400, 25 * window_size + 80), margin=dict(t=10))
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown(
        f"**Target value:** `{df[target_col].iloc[target_idx]:.4f}` kW "
        f"&mdash; this is the single number the model must output."
    )

    st.markdown("---")
    st.subheader("3 \u2014 How consecutive samples overlap")
    st.markdown(
        "Each sample shifts the window forward by **one step**. "
        "Notice how neighbouring samples share most of their data \u2014 "
        "only one row enters and one row leaves."
    )

    n_show = st.slider("Number of consecutive samples to show", 2, 8, 4, key="n_samples")
    n_show = min(n_show, max_pos - pos + 1)

    fig_overlap = go.Figure()
    fig_overlap.add_trace(go.Scatter(
        x=x_vals, y=df[target_col], mode="lines",
        line=dict(color="lightgrey", width=1), name="Full series",
    ))
    colours = px.colors.qualitative.Set2
    for i in range(n_show):
        s = pos + i
        e = s + window_size
        t = e
        if t >= len(df):
            break
        c = colours[i % len(colours)]
        fig_overlap.add_trace(go.Scatter(
            x=x_vals[s:e], y=df[target_col].iloc[s:e],
            mode="lines", line=dict(color=c, width=2),
            name=f"Sample {i} (pos {s})",
        ))
        fig_overlap.add_trace(go.Scatter(
            x=[x_vals[t]], y=[df[target_col].iloc[t]],
            mode="markers", marker=dict(color=c, size=10, symbol="x"),
            showlegend=False,
        ))
    fig_overlap.update_layout(height=350, yaxis_title=target_col, margin=dict(t=30))
    st.plotly_chart(fig_overlap, use_container_width=True)

    st.info(
        f"From **{len(df)}** time steps with window size **{window_size}**, "
        f"we get **{len(df) - window_size}** training samples "
        f"({split_idx - window_size} train + {len(df) - split_idx} test)."
    )

    # ---- Section 4: predictions vs actuals (if training results exist)
    _sliding_window_predictions_section(dataset_key, df, target_col, x_vals,
                                        split_idx, window_size)


def _sliding_window_predictions_section(dataset_key, df, target_col, x_vals,
                                        split_idx, window_size):
    """Show predictions-vs-actuals if training results are available."""
    # Try to find results either from session state or from saved models
    results = None
    for key in st.session_state.training_results:
        if key in dataset_key or dataset_key in key:
            results = st.session_state.training_results[key]
            break

    # Also check for saved model metrics on disk
    saved_models = {}
    if MODELS_DIR.exists():
        for f in MODELS_DIR.glob("*_metrics.json"):
            name = f.stem.replace("_metrics", "")
            parts = name.split("_", 1)
            if len(parts) == 2:
                saved_models[parts[0]] = json.loads(f.read_text())

    if not results and not saved_models:
        st.markdown("---")
        st.subheader("4 \u2014 Predictions vs Actuals")
        st.info(
            "No training results yet. Go to **Train & Evaluate**, train a model, "
            "then come back here to step through predictions one by one."
        )
        return

    st.markdown("---")
    st.subheader("4 \u2014 Predictions vs Actuals")

    if results:
        model_names = list(results.keys())
        chosen = st.multiselect("Models to overlay", model_names, default=model_names)

        fig_pva = go.Figure()
        # actual test values
        test_start = split_idx + window_size
        x_test = x_vals[test_start:test_start + len(results[model_names[0]]["actuals"])]
        fig_pva.add_trace(go.Scatter(
            x=x_test, y=results[model_names[0]]["actuals"],
            mode="lines", line=dict(color="black", width=2), name="Actual",
        ))
        for mn in chosen:
            fig_pva.add_trace(go.Scatter(
                x=x_test, y=results[mn]["predictions"],
                mode="lines", name=f"{mn} Predicted",
            ))
        fig_pva.update_layout(
            height=400, yaxis_title=target_col + " (kW)",
            title="Model Predictions vs Actual Values (Test Set)",
        )
        st.plotly_chart(fig_pva, use_container_width=True)

        # Per-model error metrics side by side
        metrics_df = pd.DataFrame({mn: results[mn]["metrics"] for mn in chosen}).T
        st.markdown("**Test-set metrics**")
        st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)

    elif saved_models:
        st.markdown("Saved model metrics found on disk:")
        metrics_df = pd.DataFrame(saved_models).T
        st.dataframe(metrics_df.style.format("{:.4f}"), use_container_width=True)
        st.info("Train models in this session to see the predictions overlay chart.")

def page_forecast_replay():
    st.title("Forecast Replay")
    st.markdown(
        "Drag the **cutoff slider** to pick a point in time. "
        "The model predicts a short horizon forward **autoregressively** — "
        "each predicted window feeds back into the next. "
        "Loss is computed only over the predicted windows."
    )
    st.info(
        "No-peek rule used here: after the cutoff, the model does **not** read "
        "the hidden future GPU/QPS/memory rows from the dataset. It keeps the "
        "last known non-target inputs, then updates only the power-related "
        "history features with its own predictions.",
        icon="ℹ️",
    )

    available = load_processed_datasets(st.session_state.final_data)

    if not available:
        st.warning("No processed data. Complete the Processing Pipeline first.")
        st.stop()

    col_ds, col_mdl = st.columns(2)
    with col_ds:
        dataset_key = st.selectbox("Dataset", list(available.keys()), key="fr_ds")
    with col_mdl:
        model_name = st.selectbox("Model", list(MODEL_REGISTRY.keys()), key="fr_model")

    df = available[dataset_key].copy()
    target_col = "power_total_kw"

    if "timestamp" not in df.columns:
        st.error("Dataset must include a `timestamp` column.")
        st.stop()

    timestamps = pd.to_datetime(df["timestamp"], unit="s")
    freq_seconds = int(df["timestamp"].diff().median())

    col_win, col_hz = st.columns(2)
    with col_win:
        window_size = st.slider("Lookback window (steps)", 6, 72, 24, key="fr_win")
    with col_hz:
        max_horizon = min(24, len(df) // 4)
        forecast_horizon = st.slider(
            "Forecast horizon (steps)", 1, max_horizon, min(6, max_horizon),
            key="fr_horizon",
            help=f"Each step = {freq_seconds // 60} min. "
                 f"E.g. 6 steps = {6 * freq_seconds // 60} min ahead.",
        )

    # --- Auto-load saved model ---
    model_bundle = load_model_bundle(
        model_name, dataset_key, df, target_col, window_size,
    )
    if model_bundle is None:
        st.warning(
            f"No saved **{model_name}** model found. "
            f"Go to **Train & Evaluate** and train it first, then come back here."
        )
        _show_replay_preview(df, timestamps, target_col, int(len(df) * 0.8))
        return

    model = model_bundle["model"]
    feature_names = model_bundle["feature_names"]
    scaler = model_bundle["scaler"]
    target_scaler = model_bundle["target_scaler"]
    replay_df = model_bundle.get("dataframe", df)
    loaded_info = format_model_bundle_caption(model_bundle)
    if loaded_info:
        st.caption(loaded_info)
    if model_bundle.get("compat_warning"):
        st.warning(model_bundle["compat_warning"], icon="⚠️")
    if model_bundle.get("dim_mismatch"):
        st.warning(
            f"⚠️ Saved model was trained with **{model_bundle['input_dim']} features** "
            f"but the current dataset has a different count. "
            f"Predictions may be inaccurate — retrain the model to fix this.",
            icon="⚠️",
        )

    # --- Train / test split boundary ---
    train_ratio = 0.8
    split_idx = int(len(df) * train_ratio)

    # --- TIME-BASED cutoff slider ---
    st.markdown("---")
    min_cutoff_idx = window_size + 1
    max_cutoff_idx = len(df) - forecast_horizon - 1

    # Default to start of test region
    default_idx = max(split_idx, min_cutoff_idx)
    default_idx = min(default_idx, max_cutoff_idx)

    # Detect if dataset spans more than 1 day → use date+time format
    total_hours = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 3600
    slider_format = "MMM DD HH:mm" if total_hours > 36 else "HH:mm"

    cutoff_time = st.slider(
        "Cutoff time — drag to slide the forecast boundary",
        min_value=timestamps.iloc[min_cutoff_idx].to_pydatetime(),
        max_value=timestamps.iloc[max_cutoff_idx].to_pydatetime(),
        value=timestamps.iloc[default_idx].to_pydatetime(),
        format=slider_format,
        key="fr_cutoff_time",
    )
    cutoff_idx = int((timestamps - pd.Timestamp(cutoff_time)).abs().argmin())
    cutoff_idx = max(min_cutoff_idx, min(cutoff_idx, max_cutoff_idx))

    # Clamp forecast_horizon so we don't go past the dataset
    actual_horizon = min(forecast_horizon, len(df) - cutoff_idx)

    # --- Train/test region awareness ---
    in_train_region = cutoff_idx < split_idx
    # How many of the predicted steps fall in train vs test
    pred_end_idx = cutoff_idx + actual_horizon
    steps_in_train = max(0, min(split_idx, pred_end_idx) - cutoff_idx)
    steps_in_test = actual_horizon - steps_in_train

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Known data", f"{cutoff_idx:,} steps")
    c2.metric("Predicting", f"{actual_horizon} steps ({actual_horizon * freq_seconds // 60} min)")
    cutoff_fmt = "%b %d %H:%M" if total_hours > 36 else "%H:%M"
    c3.metric("Cutoff", timestamps.iloc[cutoff_idx].strftime(cutoff_fmt))
    region_label = "TRAIN" if in_train_region else "TEST"
    c4.metric("Region", region_label)

    if in_train_region:
        if steps_in_test > 0:
            st.warning(
                f"⚠️ The cutoff is in the **training region** (first {train_ratio:.0%} of data). "
                f"The model has seen these patterns during training — "
                f"{steps_in_train} of {actual_horizon} predicted steps overlap with training data. "
                f"Errors here are expected to be lower than on truly unseen data.",
                icon="⚠️",
            )
        else:
            st.warning(
                f"⚠️ The cutoff is in the **training region** (first {train_ratio:.0%} of data). "
                f"All {actual_horizon} predicted steps overlap with training data — "
                f"the model has seen these patterns. Move the slider past "
                f"**{timestamps.iloc[split_idx].strftime(cutoff_fmt)}** for a fair evaluation.",
                icon="⚠️",
            )

    # --- Autoregressive forecast (cached) ---
    cache = st.session_state.get("fr_cache")
    if (cache is None
            or cache.get("model_name") != model_name
            or cache.get("dataset_key") != dataset_key
            or cache.get("window_size") != window_size
            or cache.get("cutoff_idx") != cutoff_idx
            or cache.get("forecast_horizon") != actual_horizon):
        forecast_preds = autoregressive_forecast(
            model, replay_df, feature_names, target_col,
            scaler, target_scaler, window_size,
            cutoff_idx, actual_horizon, st.session_state.device,
        )
        st.session_state["fr_cache"] = {
            "model_name": model_name,
            "dataset_key": dataset_key,
            "window_size": window_size,
            "cutoff_idx": cutoff_idx,
            "forecast_horizon": actual_horizon,
            "forecast_preds": forecast_preds,
        }
    else:
        forecast_preds = cache["forecast_preds"]

    forecast_actuals = df[target_col].values[cutoff_idx : cutoff_idx + actual_horizon]
    forecast_times = timestamps.values[cutoff_idx : cutoff_idx + actual_horizon]
    forecast_starts = [pd.Timestamp(t) for t in forecast_times]
    forecast_ends = forecast_starts[1:] + [
        forecast_starts[-1] + pd.Timedelta(seconds=freq_seconds)
    ]
    forecast_mids = [
        start + (end - start) / 2 for start, end in zip(forecast_starts, forecast_ends)
    ]

    # --- Overlay chart with grid windows ---
    st.subheader("Forecast vs Reality")
    fig = go.Figure()

    # Full historical series (faint)
    fig.add_trace(go.Scatter(
        x=timestamps.values[:cutoff_idx],
        y=df[target_col].values[:cutoff_idx],
        mode="lines", name="Known",
        line=dict(color="steelblue", width=1.5),
    ))

    # Future actual values (beyond predicted horizon too, for context)
    # Start 1 step before cutoff so it connects to the Known line
    context_start = max(0, cutoff_idx - 1)
    context_end = min(cutoff_idx + actual_horizon + window_size, len(df))
    fig.add_trace(go.Scatter(
        x=timestamps.values[context_start:context_end],
        y=df[target_col].values[context_start:context_end],
        mode="lines", name="Actual (future)",
        line=dict(color="black", width=2),
    ))

    # Predicted steps: plot at bin midpoints so the final horizon is visible
    # without turning the whole forecast trace into a flat step chart.
    fig.add_trace(go.Scatter(
        x=forecast_mids,
        y=forecast_preds,
        mode="lines+markers", name=f"{model_name} prediction",
        line=dict(color="orangered", width=2.5),
        marker=dict(size=7, color="orangered"),
    ))

    # --- Grid overlay: shade each predicted window using vrect ---
    # vrect uses full y-extent automatically (no y_min/y_max needed)
    for i in range(actual_horizon):
        t_start = forecast_starts[i]
        t_end = forecast_ends[i]

        # Alternate colors, stronger alpha so they are clearly visible
        alpha = 0.15 if i % 2 == 0 else 0.08
        fill_color = f"rgba(255,69,0,{alpha})"

        fig.add_vrect(
            x0=t_start, x1=t_end,
            fillcolor=fill_color,
            line_width=1.5, line_color="rgba(255,69,0,0.5)",
            layer="below",
        )

        # Step label at the top of each cell
        mid_t = t_start + (t_end - t_start) / 2
        fig.add_annotation(
            x=mid_t, y=1.0, yref="paper", text=f"t+{i+1}",
            showarrow=False, font=dict(size=10, color="rgb(200,80,0)"),
            xanchor="center", yanchor="bottom",
        )

    # Cutoff line
    cut_ts = timestamps.iloc[cutoff_idx]
    fig.add_vline(x=cut_ts, line_dash="dash", line_color="crimson", line_width=2)
    fig.add_annotation(
        x=cut_ts, y=1, yref="paper", text="  Cutoff",
        showarrow=False, xanchor="left", font=dict(color="crimson", size=13),
    )

    # Train/test boundary line
    split_ts = timestamps.iloc[split_idx]
    fig.add_vline(x=split_ts, line_dash="dot", line_color="green", line_width=1.5)
    fig.add_annotation(
        x=split_ts, y=0.95, yref="paper", text=" Train | Test",
        showarrow=False, xanchor="left",
        font=dict(color="green", size=11),
    )

    # Lookback window shading
    lookback_start = max(0, cutoff_idx - window_size)
    fig.add_vrect(
        x0=timestamps.values[lookback_start], x1=cut_ts,
        fillcolor="rgba(31,119,180,0.08)",
        line_width=1, line_color="rgba(31,119,180,0.3)",
        layer="below",
    )
    fig.add_annotation(
        x=timestamps.values[lookback_start], y=1.0, yref="paper",
        text=f"Lookback ({window_size} steps)",
        showarrow=False, font=dict(size=9, color="rgba(31,119,180,0.8)"),
        xanchor="left", yanchor="bottom",
    )

    # Focus x-range: show lookback + prediction + a bit of future context
    x_range_start = timestamps.values[max(0, cutoff_idx - window_size - 5)]
    x_range_end = max(
        pd.Timestamp(timestamps.values[min(len(df) - 1, cutoff_idx + actual_horizon + 5)]),
        forecast_ends[-1],
    )

    # Y-axis: base on actual data range (don't let divergent predictions squash the view)
    visible_actuals = df[target_col].values[max(0, cutoff_idx - window_size):context_end]
    y_data_min = visible_actuals.min()
    y_data_max = visible_actuals.max()
    # Extend range to include predictions but cap the expansion
    y_all_min = min(y_data_min, forecast_preds.min())
    y_all_max = max(y_data_max, forecast_preds.max())
    y_data_range = y_data_max - y_data_min if y_data_max > y_data_min else 1.0
    # Allow at most 2x the actual data range as padding for divergent predictions
    y_min = max(y_all_min, y_data_min - y_data_range) * 0.95
    y_max = min(y_all_max, y_data_max + y_data_range) * 1.05

    fig.update_layout(
        height=480, yaxis_title="Power (kW)",
        dragmode="pan",
        xaxis=dict(range=[x_range_start, x_range_end]),
        yaxis=dict(range=[y_min, y_max]),
        legend=dict(
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray", borderwidth=1,
        ),
        margin=dict(t=40, b=30),
    )
    forecast_replay_plotly_config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtons": [["pan2d"], ["toImage"], ["resetScale2d"]],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "forecast_replay",
            "scale": 2,
        },
    }
    st.plotly_chart(
        fig,
        use_container_width=True,
        config=forecast_replay_plotly_config,
    )

    # --- Metrics (only over predicted windows) ---
    metrics = forecast_metrics(forecast_actuals, forecast_preds)
    mae = metrics["MAE"]
    rmse = metrics["RMSE"]
    mape = metrics["MAPE"]
    r2 = metrics["R2"]

    st.subheader("Prediction Error (over predicted horizon only)")

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("MAE", f"{mae:.2f} kW",
               help="Average absolute error over predicted steps")
    mc2.metric("RMSE", f"{rmse:.2f} kW",
               help="Root mean squared error over predicted steps")
    mc3.metric("MAPE", f"{mape:.1f}%",
               help="Mean absolute percentage error over predicted steps")
    mc4.metric("R²", f"{r2:.4f}",
               help="1.0 = perfect, 0.0 = no better than mean")

    avg_power = np.mean(forecast_actuals)
    region_note = ""
    if in_train_region and steps_in_train > 0:
        region_note = (
            f" ⚠️ {steps_in_train}/{actual_horizon} steps are in the "
            f"**training region** — errors may be optimistic."
        )
    st.caption(
        f"Avg actual power: **{avg_power:.1f} kW**. "
        f"Deviation: **{mae:.2f} kW** ({mape:.1f}%).{region_note}"
    )

    # --- Per-step error breakdown ---
    with st.expander("Per-step error breakdown"):
        step_errors = forecast_actuals - forecast_preds
        step_abs = np.abs(step_errors)

        step_df = pd.DataFrame({
            "Step": [f"t+{i+1}" for i in range(actual_horizon)],
            "Time": [pd.Timestamp(t).strftime("%H:%M") for t in forecast_times],
            "Actual (kW)": forecast_actuals,
            "Predicted (kW)": forecast_preds,
            "Error (kW)": step_errors,
            "|Error| (kW)": step_abs,
            "Region": ["TRAIN" if cutoff_idx + i < split_idx else "TEST"
                        for i in range(actual_horizon)],
        })
        st.dataframe(step_df.style.format({
            "Actual (kW)": "{:.2f}", "Predicted (kW)": "{:.2f}",
            "Error (kW)": "{:.2f}", "|Error| (kW)": "{:.2f}",
        }), use_container_width=True)

        # Bar chart of absolute error per step
        fig_bar = go.Figure()
        colors = ["#ff6347" if cutoff_idx + i < split_idx else "#4682b4"
                  for i in range(actual_horizon)]
        fig_bar.add_trace(go.Bar(
            x=[f"t+{i+1}" for i in range(actual_horizon)],
            y=step_abs, marker_color=colors,
            name="|Error|",
        ))
        fig_bar.update_layout(
            height=250, yaxis_title="|Error| (kW)",
            xaxis_title="Predicted Step",
            margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("Red bars = training region, Blue bars = test region.")


def _show_replay_preview(df, timestamps, target_col, cutoff_idx):
    """Show the dataset split preview before a model is loaded."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps.values[:cutoff_idx],
        y=df[target_col].values[:cutoff_idx],
        mode="lines", name="Before cutoff",
        line=dict(color="steelblue", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=timestamps.values[cutoff_idx:],
        y=df[target_col].values[cutoff_idx:],
        mode="lines", name="After cutoff (actual)",
        line=dict(color="black", width=2),
    ))
    cut_ts = timestamps.iloc[cutoff_idx]
    fig.add_vline(x=cut_ts, line_dash="dash", line_color="crimson")
    fig.update_layout(height=350, yaxis_title="Power (kW)", margin=dict(t=20))
    st.plotly_chart(fig, use_container_width=True)


def page_train_and_evaluate():
    st.title("Train & Evaluate Models")

    available = {k: v for k, v in st.session_state.final_data.items() if len(v) > 0}
    if not available:
        st.warning("No processed data. Complete the Processing Pipeline first.")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        dataset = st.selectbox("Train on", list(available.keys()))
        df = available[dataset]
        st.caption(f"{len(df)} rows x {len(df.columns)} columns")
    with col2:
        models_to_train = st.multiselect(
            "Models", list(MODEL_REGISTRY.keys()),
            default=list(MODEL_REGISTRY.keys())
        )

    freq = st.session_state.pipeline_params.get(dataset, {}).get("freq", 300)
    window_size, train_ratio, epochs, lr = _training_params(freq)
    hidden_dim, num_layers, dropout, patience = _architecture_params()

    split_idx = int(len(df) * train_ratio)
    _show_split_chart(df, split_idx)

    if st.button("Start Training", type="primary"):
        _run_training(
            df, dataset, models_to_train, window_size, train_ratio,
            epochs, lr, hidden_dim, num_layers, dropout, patience
        )

def _training_params(freq):
    st.markdown("### Data Split & Windowing")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        window_size = st.slider("Lookback Window (steps)", 6, 72, 24)
    with col2:
        train_ratio = st.slider("Train Ratio", 0.5, 0.9, 0.8, step=0.05)
    with col3:
        epochs = st.slider("Max Epochs", 10, 300, 100)
    with col4:
        lr = st.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)
    return window_size, train_ratio, epochs, lr

def _architecture_params():
    with st.expander("Architecture Settings", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            hidden_dim = st.selectbox("Hidden Dim", [32, 64, 128, 256], index=1)
        with col2:
            num_layers = st.selectbox("Num Layers", [1, 2, 3, 4], index=1)
        with col3:
            dropout = st.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)
        with col4:
            patience = st.slider("Early Stop Patience", 3, 30, 10)
    return hidden_dim, num_layers, dropout, patience

def _show_split_chart(df, split_idx):
    x = pd.to_datetime(df["timestamp"], unit="s")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x[:split_idx], y=df["power_total_kw"].iloc[:split_idx],
        mode="lines", name="Train", line=dict(color="blue")
    ))
    fig.add_trace(go.Scatter(
        x=x[split_idx:], y=df["power_total_kw"].iloc[split_idx:],
        mode="lines", name="Test", line=dict(color="orange")
    ))
    fig.add_vline(x=x.iloc[split_idx], line_dash="dash", line_color="red")
    fig.update_layout(title="Train / Test Split", yaxis_title="Power (kW)", height=300)
    st.plotly_chart(fig, use_container_width=True)

def _run_training(
    df, dataset, models_to_train, window_size, train_ratio,
    epochs, lr, hidden_dim, num_layers, dropout, patience
):
    device = st.session_state.device
    data = prepare_data(
        df, target_col="power_total_kw", window_size=window_size,
        train_ratio=train_ratio
    )
    results = {}

    for model_name in models_to_train:
        st.markdown("---")
        st.markdown(f"### Training: {model_name}")

        model_kwargs = build_model_kwargs(model_name, hidden_dim, num_layers, dropout)

        model = get_model(model_name, input_dim=data["input_dim"], **model_kwargs)
        param_count = sum(p.numel() for p in model.parameters())

        col1, col2, col3 = st.columns(3)
        col1.metric("Parameters", f"{param_count:,}")
        col2.metric("Input Features", data["input_dim"])
        col3.metric("Device", device)

        progress_bar = st.progress(0)
        loss_placeholder = st.empty()

        def _progress(update):
            progress_bar.progress(update["epoch"] / update["epochs"])
            should_render = (
                update["epoch"] % 5 == 0
                or update["epoch"] == 1
                or update["stopped"]
            )
            if not should_render:
                return
            if update["stopped"]:
                loss_placeholder.markdown(
                    f"**Early stopped at epoch {update['epoch']}** -- "
                    f"Best Val Loss: `{update['best_val_loss']:.6f}`"
                )
                return
            loss_placeholder.markdown(
                f"**Epoch {update['epoch']}/{update['epochs']}** -- "
                f"Train: `{update['train_loss']:.6f}` | "
                f"Val: `{update['val_loss']:.6f}` | "
                f"Best: `{update['best_val_loss']:.6f}` | "
                f"Wait: `{update['wait']}/{update['patience']}`"
            )

        history = train_model(
            model,
            data["train_loader"],
            val_loader=data["test_loader"],
            epochs=epochs,
            lr=lr,
            patience=patience,
            device=device,
            progress_cb=_progress,
        )

        preds, targets, metrics = evaluate_model(
            model, data["test_loader"],
            data["target_scaler"], device=device
        )
        save_model(model, model_name, dataset, metrics, history,
                   window_size=window_size, train_ratio=train_ratio,
                   input_dim=data["input_dim"])

        _show_training_results(model_name, metrics, history)

        results[model_name] = {
            "predictions": preds.tolist(),
            "actuals": targets.tolist(),
            "metrics": metrics,
            "history": history,
            "timestamps": data["timestamps_test"][:len(preds)].tolist()
                if data["timestamps_test"] is not None else None,
        }

    st.session_state.training_results[dataset] = results
    st.success(f"All {len(models_to_train)} models trained and saved.")

def _show_training_results(model_name, metrics, history):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{metrics['MAE']:.4f} kW")
    col2.metric("RMSE", f"{metrics['RMSE']:.4f} kW")
    col3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
    col4.metric("R2", f"{metrics['R2']:.4f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history["train_loss"], mode="lines", name="Train"))
    fig.add_trace(go.Scatter(y=history["val_loss"], mode="lines", name="Val", line=dict(dash="dash")))
    fig.update_layout(
        title=f"{model_name} Loss Curve", xaxis_title="Epoch",
        yaxis_title="MSE Loss", height=300
    )
    st.plotly_chart(fig, use_container_width=True)


# ===========================================================================
# Shared helpers for new analysis pages
# ===========================================================================
def _available_processed_datasets():
    """Return {label: DataFrame} including session state and saved CSVs on disk."""
    return load_processed_datasets(st.session_state.final_data)


def _require_dataset():
    av = _available_processed_datasets()
    if not av:
        st.warning("No processed data. Complete the Processing Pipeline first.")
        st.stop()
    return av


_SELECT_ALL_OPTION = "__select_all__"


def _multiselect_with_select_all(label, options, default, *, key, help_text=None):
    normalized_options = list(options)
    normalized_default = [option for option in default if option in normalized_options]

    if key in st.session_state:
        current_selection = [
            option
            for option in st.session_state[key]
            if option == _SELECT_ALL_OPTION or option in normalized_options
        ]
        if _SELECT_ALL_OPTION in current_selection:
            st.session_state[key] = normalized_options.copy()
        elif current_selection != st.session_state[key]:
            st.session_state[key] = current_selection

    if not normalized_options:
        return st.multiselect(label, normalized_options, default=normalized_default, key=key, help=help_text)

    widget_kwargs = {
        "label": label,
        "options": [_SELECT_ALL_OPTION, *normalized_options],
        "key": key,
        "format_func": lambda option: "Select All" if option == _SELECT_ALL_OPTION else option,
        "help": help_text,
    }
    if key not in st.session_state:
        widget_kwargs["default"] = normalized_default

    raw_selection = st.multiselect(**widget_kwargs)

    if _SELECT_ALL_OPTION not in raw_selection:
        return [option for option in raw_selection if option in normalized_options]
    return normalized_options


# ===========================================================================
# Page: Symbolic Regression
# ===========================================================================
def page_symbolic_regression():
    st.title("Symbolic Regression")
    av = _require_dataset()
    ds_label = st.selectbox("Dataset", list(av.keys()))
    df = av[ds_label]
    target_col = "power_total_kw"

    step_minutes = 5
    if "timestamp" in df.columns and len(df) > 1:
        freq_seconds = pd.Series(df["timestamp"]).diff().median()
        if pd.notna(freq_seconds) and float(freq_seconds) > 0:
            step_minutes = max(1, int(round(float(freq_seconds) / 60.0)))

    st.markdown("**Basic setup**")
    col1, col2 = st.columns(2)
    with col1:
        optimization_horizon = st.slider(
            "How far ahead should the formula train for? (steps)",
            1,
            60,
            12,
            help=f"Each step is about {step_minutes} minute(s). If you choose 12, the formula trains for steps 1 to 12 ahead.",
        )
        predict_residual = st.checkbox(
            "Learn change from the last known power",
            value=True,
            help="When ON, the model predicts how much power moves up or down, then adds that change to the last known power.",
        )
        lags_text = st.text_input(
            "Past power points to use (steps ago)",
            "1,2,3,6,12",
            help="1 means one step ago, 12 means twelve steps ago.",
        )
        roll_text = st.text_input(
            "Past average windows (steps)",
            "6,12",
            help="6 means the average of the last 6 steps, 12 means the average of the last 12 steps.",
        )
    with col2:
        exo = symbolic_exogenous_candidates(df)
        exo_pick = _multiselect_with_select_all(
            "Other inputs the formula may use",
            exo,
            default=default_symbolic_exogenous(exo),
            key="symbolic_exogenous_inputs",
            help_text="Examples: GPU usage, QPS, memory, or time-of-day features.",
        )
        funcs = _multiselect_with_select_all(
            "Math building blocks",
            ["add", "sub", "mul", "div", "sin", "cos", "log", "sqrt", "abs", "neg"],
            default=["add", "sub", "mul", "div", "sin"],
            key="symbolic_function_set",
            help_text="These are the math operations the symbolic search is allowed to use when building the formula.",
        )
        with st.expander("Optional: search settings"):
            train_ratio = st.slider(
                "How much data to use for training",
                0.5,
                0.9,
                0.8,
                step=0.05,
            )
            gens = st.slider("Search rounds", 5, 100, 20)
            pop = st.slider("Candidate formulas per round", 100, 2000, 500, step=100)
            parsimony = st.select_slider(
                "Prefer shorter formulas",
                [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                value=1e-3,
            )

    if st.button("Run Symbolic Regression", type="primary"):
        try:
            config = SymbolicRunConfig(
                lags=parse_positive_ints(lags_text, "lag"),
                rolling_means=parse_positive_ints(
                    roll_text, "rolling mean", allow_empty=True,
                ),
                extra_features=tuple(exo_pick),
                train_ratio=train_ratio,
                generations=gens,
                population_size=pop,
                parsimony=parsimony,
                function_set=tuple(funcs),
                predict_residual=predict_residual,
                optimization_horizon=optimization_horizon,
                train_across_horizon_window=True,
            )
            with st.spinner("Searching for a simple formula..."):
                cache_payload = run_symbolic_regression(df, config)
            cache_payload["dataset"] = ds_label
            st.session_state["symbolic_result"] = cache_payload
            st.session_state.pop("symbolic_multistep", None)
        except ImportError as e:
            st.error(f"{e}\n\nInstall with: `pip install gplearn`")
        except Exception as e:
            st.error(f"Failed: {e}")

    cache = st.session_state.get("symbolic_result")
    if not cache:
        st.info("Choose a setup, then click **Run Symbolic Regression**.")
        return

    res = cache["res"]
    symbolic_plotly_config = {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtons": [["pan2d"], ["toImage"], ["resetScale2d"]],
        "toImageButtonOptions": {
            "format": "png",
            "filename": "symbolic_regression",
            "scale": 2,
        },
    }
    from src.evaluation.symbolic import (
        describe_symbolic_feature,
        expression_feature_names,
        latex_expression,
        pretty_expression,
        raw_expression_mapping,
        replay_forecast_from_frame,
    )

    pretty = pretty_expression(res["expression"], res["feature_names"])
    pretty_latex = latex_expression(res["expression"], res["feature_names"])
    persistence_test_metrics = res.get("metrics_persistence_test")
    mae_skill = float("nan")
    if persistence_test_metrics and persistence_test_metrics.get("MAE"):
        mae_skill = (
            (persistence_test_metrics["MAE"] - res["metrics_test"]["MAE"])
            / persistence_test_metrics["MAE"]
        ) * 100.0

    used_features = expression_feature_names(res["expression"], res["feature_names"])
    if res.get("predict_residual") and res.get("baseline_col") and res["baseline_col"] not in used_features:
        used_features = [res["baseline_col"], *used_features]

    variable_rows = []
    for feature_name in used_features:
        row = describe_symbolic_feature(feature_name, step_minutes=step_minutes)
        symbol_latex = latex_expression(feature_name, [feature_name])
        variable_rows.append((symbol_latex, row["Meaning"]))

    st.markdown("---")
    st.subheader("Formula")
    if res.get("predict_residual"):
        st.latex(rf"\hat{{y}}_{{t+h}} = y_{{t-1}} + {pretty_latex}")
    else:
        st.latex(rf"\hat{{y}}_{{t+h}} = {pretty_latex}")
    if variable_rows:
        st.markdown("**Variables used**")
        for symbol_latex, meaning in variable_rows:
            sym_col, meaning_col = st.columns([1, 4])
            with sym_col:
                st.latex(symbol_latex)
            with meaning_col:
                st.markdown(meaning)

    st.subheader("Performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average miss", f"{res['metrics_test']['MAE']:.3f} kW")
    c2.metric("Average % miss", f"{res['metrics_test']['MAPE']:.2f}%")
    c3.metric(
        "Vs repeat-last baseline",
        "—" if pd.isna(mae_skill) else f"{mae_skill:+.1f}%",
    )

    timestamp_series = pd.Series(df["timestamp"])
    if pd.api.types.is_numeric_dtype(timestamp_series):
        timestamps = pd.to_datetime(timestamp_series, unit="s")
        freq_seconds = int(timestamp_series.diff().median()) if len(timestamp_series) > 1 else step_minutes * 60
    else:
        timestamps = pd.to_datetime(timestamp_series)
        freq_delta = timestamps.diff().median()
        freq_seconds = int(freq_delta.total_seconds()) if pd.notna(freq_delta) else step_minutes * 60

    symbolic_meta = cache.get("meta")
    split_idx = int(cache.get("split_idx", len(df) * cache["config"].train_ratio))
    if symbolic_meta and symbolic_meta.get("source_index") is not None:
        source_index = np.asarray(symbolic_meta["source_index"], dtype=int)
        horizon_values = np.asarray(symbolic_meta["horizon"], dtype=int)
        target_index = np.asarray(
            symbolic_meta.get("target_index", source_index + horizon_values - 1),
            dtype=int,
        )
        max_horizon_by_source = pd.Series(horizon_values).groupby(source_index).max().sort_index()
        replay_horizon_cap = int(min(max_horizon_by_source.max(), 24))

        if replay_horizon_cap >= 1:
            st.subheader("Symbolic Replay")

            replay_col1, replay_col2 = st.columns(2)
            with replay_col1:
                replay_horizon = st.slider(
                    "Replay horizon (steps)",
                    1,
                    replay_horizon_cap,
                    min(6, replay_horizon_cap),
                    key="symbolic_direct_replay_horizon",
                    help=f"Each step is about {step_minutes} minute(s).",
                )

            eligible_sources = max_horizon_by_source[max_horizon_by_source >= replay_horizon].index.to_numpy(dtype=int)
            preferred_sources = eligible_sources[eligible_sources >= split_idx]
            replay_sources = preferred_sources if len(preferred_sources) else eligible_sources

            if len(replay_sources) > 0:
                total_hours = (timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds() / 3600 if len(timestamps) > 1 else 0
                slider_format = "MMM DD HH:mm" if total_hours > 36 else "HH:mm"
                default_source_idx = int(replay_sources[0])
                replay_times = timestamps.iloc[replay_sources].reset_index(drop=True)

                with replay_col2:
                    replay_cutoff_time = st.slider(
                        "Replay cutoff time",
                        min_value=replay_times.iloc[0].to_pydatetime(),
                        max_value=replay_times.iloc[-1].to_pydatetime(),
                        value=timestamps.iloc[default_source_idx].to_pydatetime(),
                        format=slider_format,
                        key="symbolic_direct_replay_cutoff",
                    )

                replay_offset = (replay_times - pd.Timestamp(replay_cutoff_time)).abs()
                replay_source_idx = int(replay_sources[int(np.argmin(replay_offset.to_numpy()))])
                replay_result = replay_forecast_from_frame(
                    res,
                    df,
                    start_idx=replay_source_idx,
                    horizon=replay_horizon,
                    target_col=target_col,
                )

                replay_preds = replay_result["preds"]
                replay_actuals = replay_result["actuals"]
                replay_baseline = replay_result["persistence"]
                replay_target_idx = replay_result["target_indices"]
                replay_horizons = np.arange(1, len(replay_preds) + 1)

                if len(replay_preds) > 0:
                    replay_mae = float(np.mean(np.abs(replay_preds - replay_actuals)))
                    replay_baseline_mae = float(np.mean(np.abs(replay_baseline - replay_actuals)))
                    replay_skill = float("nan")
                    if replay_baseline_mae > 0:
                        replay_skill = ((replay_baseline_mae - replay_mae) / replay_baseline_mae) * 100.0

                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("This replay miss", f"{replay_mae:.3f} kW")
                    rc2.metric("Repeat-last miss", f"{replay_baseline_mae:.3f} kW")
                    rc3.metric(
                        "Vs repeat-last baseline",
                        "—" if pd.isna(replay_skill) else f"{replay_skill:+.1f}%",
                    )

                    forecast_starts = [pd.Timestamp(timestamps.iloc[idx]) for idx in replay_target_idx]
                    forecast_ends = forecast_starts[1:] + [
                        forecast_starts[-1] + pd.Timedelta(seconds=freq_seconds)
                    ]
                    forecast_mids = [
                        start + (end - start) / 2 for start, end in zip(forecast_starts, forecast_ends)
                    ]

                    lookback_steps = min(24, replay_source_idx)
                    context_start = max(0, replay_source_idx - 1)
                    context_end = min(len(df), replay_target_idx[-1] + 7)
                    x_range_start = timestamps.iloc[max(0, replay_source_idx - lookback_steps - 4)]
                    x_range_end = forecast_ends[-1]

                    fig_symbolic_replay = go.Figure()
                    fig_symbolic_replay.add_trace(go.Scatter(
                        x=timestamps.iloc[:replay_source_idx],
                        y=df[target_col].iloc[:replay_source_idx],
                        mode="lines",
                        name="Known",
                        line=dict(color="steelblue", width=1.5),
                    ))
                    fig_symbolic_replay.add_trace(go.Scatter(
                        x=timestamps.iloc[context_start:context_end],
                        y=df[target_col].iloc[context_start:context_end],
                        mode="lines",
                        name="Actual (future)",
                        line=dict(color="black", width=2),
                    ))
                    fig_symbolic_replay.add_trace(go.Scatter(
                        x=forecast_mids,
                        y=replay_preds,
                        mode="lines+markers",
                        name="Symbolic rollout",
                        line=dict(color="green", width=2.5),
                        marker=dict(size=7, color="green"),
                    ))
                    fig_symbolic_replay.add_trace(go.Scatter(
                        x=forecast_mids,
                        y=replay_baseline,
                        mode="lines+markers",
                        name="Repeat-last baseline",
                        line=dict(color="steelblue", width=2, dash="dash"),
                        marker=dict(size=6, color="steelblue"),
                    ))

                    for i, (t_start, t_end) in enumerate(zip(forecast_starts, forecast_ends)):
                        alpha = 0.15 if i % 2 == 0 else 0.08
                        fig_symbolic_replay.add_vrect(
                            x0=t_start,
                            x1=t_end,
                            fillcolor=f"rgba(34,139,34,{alpha})",
                            line_width=1.2,
                            line_color="rgba(34,139,34,0.45)",
                            layer="below",
                        )
                        mid_t = t_start + (t_end - t_start) / 2
                        fig_symbolic_replay.add_annotation(
                            x=mid_t,
                            y=1.0,
                            yref="paper",
                            text=f"t+{int(replay_horizons[i])}",
                            showarrow=False,
                            font=dict(size=10, color="rgb(34,139,34)"),
                            xanchor="center",
                            yanchor="bottom",
                        )

                    cut_ts = timestamps.iloc[replay_source_idx]
                    fig_symbolic_replay.add_vline(x=cut_ts, line_dash="dash", line_color="green", line_width=2)
                    fig_symbolic_replay.add_annotation(
                        x=cut_ts,
                        y=1,
                        yref="paper",
                        text="  Replay cutoff",
                        showarrow=False,
                        xanchor="left",
                        font=dict(color="green", size=12),
                    )

                    fig_symbolic_replay.update_layout(
                        title="One concrete step-by-step symbolic forecast from the unseen test region",
                        height=420,
                        dragmode="pan",
                        xaxis=dict(range=[x_range_start, x_range_end]),
                        yaxis_title="Power (kW)",
                        margin=dict(t=50, b=30),
                    )
                    st.plotly_chart(
                        fig_symbolic_replay,
                        use_container_width=True,
                        config=symbolic_plotly_config,
                    )

    metrics_by_horizon = res.get("metrics_by_horizon")
    persistence_by_horizon = res.get("persistence_by_horizon") or {}
    if metrics_by_horizon:
        st.subheader("Per-Step Error")

        horizon_rows = []
        for horizon in sorted(metrics_by_horizon):
            symbolic_metrics = metrics_by_horizon[horizon]
            baseline_metrics = persistence_by_horizon.get(horizon, {})
            horizon_rows.append({
                "Step ahead": int(horizon),
                "Minutes ahead": int(horizon) * step_minutes,
                "Symbolic MAE (kW)": float(symbolic_metrics["MAE"]),
                "Repeat-last MAE (kW)": float(baseline_metrics.get("MAE", np.nan)),
                "Symbolic RMSE (kW)": float(symbolic_metrics["RMSE"]),
            })

        horizon_df = pd.DataFrame(horizon_rows)
        fig_horizon = go.Figure()
        fig_horizon.add_trace(go.Scatter(
            x=horizon_df["Step ahead"],
            y=horizon_df["Symbolic MAE (kW)"],
            mode="lines+markers",
            name="Symbolic",
            line=dict(color="green", width=2),
            marker=dict(size=7, color="green"),
        ))
        if horizon_df["Repeat-last MAE (kW)"].notna().any():
            fig_horizon.add_trace(go.Scatter(
                x=horizon_df["Step ahead"],
                y=horizon_df["Repeat-last MAE (kW)"],
                mode="lines+markers",
                name="Repeat-last baseline",
                line=dict(color="steelblue", width=2, dash="dash"),
                marker=dict(size=6, color="steelblue"),
            ))
        fig_horizon.update_layout(
            title="Average miss at each future step",
            dragmode="pan",
            xaxis_title="Forecast step ahead",
            yaxis_title="MAE (kW)",
            height=320,
        )
        st.plotly_chart(fig_horizon, use_container_width=True, config=symbolic_plotly_config)

        if horizon_df["Repeat-last MAE (kW)"].notna().any():
            better_steps = int((horizon_df["Symbolic MAE (kW)"] < horizon_df["Repeat-last MAE (kW)"]).sum())
            total_steps = int(len(horizon_df))
            st.info(
                f"Plain reading: the symbolic formula beats the repeat-last baseline on "
                f"{better_steps} out of {total_steps} shown step(s).",
                icon="ℹ️",
            )

        with st.expander("Optional: per-step metric table"):
            st.dataframe(horizon_df.style.format({
                "Symbolic MAE (kW)": "{:.4f}",
                "Repeat-last MAE (kW)": "{:.4f}",
                "Symbolic RMSE (kW)": "{:.4f}",
            }), use_container_width=True)

    with st.expander("Optional: step-by-step no-peek rollout check"):
        st.markdown(
            "This check asks a different question: if we stop at one time point and keep "
            "predicting further ahead step by step, how quickly does the error grow?"
        )
        rollout_cap = max(1, min(24, len(cache["y"]) - cache["train_end"] - 2))
        rollout_col1, rollout_col2 = st.columns(2)
        with rollout_col1:
            rollout_horizon = st.slider(
                "How many future steps to roll forward",
                1,
                rollout_cap,
                min(6, rollout_cap),
                key="symbolic_rollout_horizon",
                help=f"Each step is about {step_minutes} minute(s).",
            )
        with rollout_col2:
            max_rollout_starts = max(1, min(30, len(cache["y"]) - cache["train_end"] - rollout_horizon))
            rollout_starts = st.slider(
                "How many starting points to average",
                1,
                max_rollout_starts,
                min(10, max_rollout_starts),
                key="symbolic_rollout_starts",
            )

        rollout_cache = st.session_state.get("symbolic_multistep")
        if (
            rollout_cache is None
            or rollout_cache.get("dataset") != cache["dataset"]
            or rollout_cache.get("expression") != res["expression"]
            or rollout_cache.get("horizon") != rollout_horizon
            or rollout_cache.get("n_starts") != rollout_starts
        ):
            rollout_result = run_symbolic_multistep(
                cache,
                horizon=rollout_horizon,
                n_starts=rollout_starts,
            )
            st.session_state["symbolic_multistep"] = {
                "dataset": cache["dataset"],
                "expression": res["expression"],
                "horizon": rollout_horizon,
                "n_starts": rollout_starts,
                "result": rollout_result,
            }
        else:
            rollout_result = rollout_cache["result"]

        st.info(rollout_result["rollout_exogenous_policy"], icon="ℹ️")

        rollout_mae = float(np.nanmean(rollout_result["mae_per_h"]))
        rollout_persistence_mae = float(np.nanmean(rollout_result["persistence_mae_per_h"]))
        rollout_skill = float("nan")
        if rollout_persistence_mae > 0:
            rollout_skill = (
                (rollout_persistence_mae - rollout_mae) / rollout_persistence_mae
            ) * 100.0

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("Average rollout miss", f"{rollout_mae:.3f} kW")
        rc2.metric("Repeat-last rollout miss", f"{rollout_persistence_mae:.3f} kW")
        rc3.metric(
            "Vs repeat-last baseline",
            "—" if pd.isna(rollout_skill) else f"{rollout_skill:+.1f}%",
        )

        fig_rollout = go.Figure()
        fig_rollout.add_trace(go.Scatter(
            x=rollout_result["horizon"],
            y=rollout_result["mae_per_h"],
            mode="lines+markers",
            name="Symbolic rollout",
            line=dict(color="green", width=2),
        ))
        fig_rollout.add_trace(go.Scatter(
            x=rollout_result["horizon"],
            y=rollout_result["persistence_mae_per_h"],
            mode="lines+markers",
            name="Repeat-last baseline",
            line=dict(color="steelblue", dash="dash"),
        ))
        fig_rollout.update_layout(
            title="How the error grows as we roll further into the future",
            dragmode="pan",
            xaxis_title="Forecast step ahead",
            yaxis_title="MAE (kW)",
            height=320,
        )
        st.plotly_chart(fig_rollout, use_container_width=True, config=symbolic_plotly_config)

        sample_rollout = rollout_result["sample_rollout"]
        sample_steps = [f"t+{i}" for i in range(1, len(sample_rollout["actuals"]) + 1)]
        fig_sample = go.Figure()
        fig_sample.add_trace(go.Scatter(
            x=sample_steps,
            y=sample_rollout["actuals"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="black", width=2),
        ))
        fig_sample.add_trace(go.Scatter(
            x=sample_steps,
            y=sample_rollout["preds"],
            mode="lines+markers",
            name="Symbolic rollout",
            line=dict(color="green", width=2),
        ))
        fig_sample.add_trace(go.Scatter(
            x=sample_steps,
            y=sample_rollout["persistence"],
            mode="lines+markers",
            name="Repeat-last baseline",
            line=dict(color="steelblue", dash="dash"),
        ))
        fig_sample.update_layout(
            title="One sample rollout",
            dragmode="pan",
            xaxis_title="Forecast step ahead",
            yaxis_title="Power (kW)",
            height=320,
        )
        st.plotly_chart(fig_sample, use_container_width=True, config=symbolic_plotly_config)

    with st.expander("Optional: technical details"):
        glossary_rows = [
            {
                "Symbol": "ŷ[t+h]",
                "Meaning": "Predicted cluster power h steps into the future.",
                "Source": "Model output",
            },
            {
                "Symbol": "y",
                "Meaning": "The target variable: cluster power demand (`power_total_kw`) in kW.",
                "Source": "Target variable",
            },
            {
                "Symbol": "h",
                "Meaning": "How many forecast steps ahead we are predicting.",
                "Source": "Forecast horizon",
            },
        ]
        seen_symbols = {row["Symbol"] for row in glossary_rows}
        for feature_name in used_features:
            row = describe_symbolic_feature(feature_name, step_minutes=step_minutes)
            if row["Symbol"] not in seen_symbols:
                glossary_rows.append(row)
                seen_symbols.add(row["Symbol"])

        if res.get("predict_residual"):
            st.markdown("**Technical math form**")
            st.latex(rf"\hat{{y}}_{{t+h}} = y_{{t-1}} + {pretty_latex}")
        else:
            st.markdown("**Technical math form**")
            st.latex(rf"\hat{{y}}_{{t+h}} = {pretty_latex}")

        st.markdown("**Variable guide**")
        st.table(pd.DataFrame(glossary_rows))

        st.markdown("**Train vs test metrics**")
        cmp_rows = {
            "Symbolic (train)": res["metrics_train"],
            "Symbolic (test)": res["metrics_test"],
        }
        if res.get("metrics_persistence_train"):
            cmp_rows["Persistence (train)"] = res["metrics_persistence_train"]
            cmp_rows["Persistence (test)"] = res["metrics_persistence_test"]
        cmp = pd.DataFrame(cmp_rows).T
        st.dataframe(cmp.style.format("{:.4f}"), use_container_width=True)

        fig_full = go.Figure()
        fig_full.add_trace(go.Scatter(y=res["actuals_test"], name="Actual", line=dict(color="black")))
        fig_full.add_trace(go.Scatter(y=res["preds_test"], name="Symbolic", line=dict(color="green")))
        if res.get("persistence_test") is not None:
            fig_full.add_trace(go.Scatter(
                y=res["persistence_test"],
                name="Repeat-last baseline",
                line=dict(color="steelblue", dash="dash"),
            ))
        fig_full.update_layout(
            title="Full test-set view",
            height=350,
            dragmode="pan",
            yaxis_title="Power (kW)",
        )
        st.plotly_chart(fig_full, use_container_width=True, config=symbolic_plotly_config)

        st.markdown("**Raw program expression**")
        st.code(res["expression"], language="text")
        mapping_rows = raw_expression_mapping(res["expression"], res["feature_names"])
        if mapping_rows:
            st.caption("Internal placeholders like X11 map to these actual variables:")
            st.table(pd.DataFrame(mapping_rows))

        st.markdown("**Comparison with saved/session deep models**")
        rows = [{
            "Model": "Symbolic" + (" (Δy)" if res.get("predict_residual") else ""),
            **res["metrics_test"],
            "Complexity": res["complexity"],
        }]
        sess = st.session_state.get("training_results", {})
        for k, v in sess.items():
            if k in cache["dataset"] or cache["dataset"] in k:
                for mn, r in v.items():
                    rows.append({"Model": f"{mn} (session)", **r["metrics"], "Complexity": "—"})
                break

        ds_short = "genai" if "genai" in cache["dataset"].lower() else "gpu_v2020"
        if MODELS_DIR.exists():
            for manifest_path in sorted(MODELS_DIR.glob(f"*_{ds_short}_w*_manifest.json")):
                try:
                    with open(manifest_path) as f:
                        m = json.load(f)
                    rows.append({
                        "Model": f"{m['model_name']} (saved {m.get('created','')})",
                        **m["metrics"],
                        "Complexity": "—",
                    })
                except Exception:
                    pass

        if len(rows) > 1:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.info("Train deep models to see them in this table.")


# ===========================================================================
# Page: Transfer Learning
# ===========================================================================
def page_transfer_learning():
    st.title("Transfer Learning — Pretrain → Fine-tune")
    st.markdown(
        "The historical comparison (pre/post-ChatGPT) on raw values is "
        "unreliable due to dataset incompatibilities. Reframe v2020 as "
        "**pretraining data**: train an encoder on it, then fine-tune on "
        "GenTD26. Compare against from-scratch training to show whether "
        "pretraining helped."
    )
    av = _require_dataset()
    keys = list(av.keys())
    if len(keys) < 2:
        st.warning("Need at least two processed datasets (one source + one target). "
                   "Process both `genai` and `gpu_v2020` in the Processing Pipeline first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        src_key = st.selectbox("Source (pretrain on)", keys,
                                index=next((i for i, k in enumerate(keys) if "v2020" in k), 0))
        tgt_key = st.selectbox("Target (fine-tune on)", keys,
                                index=next((i for i, k in enumerate(keys) if "genai" in k), 0))
    with col2:
        model_name = st.selectbox("Architecture", list(MODEL_REGISTRY.keys()), index=1)
        window_size = st.slider("Lookback window", 6, 48, 24)
        freeze = st.checkbox("Freeze encoder during fine-tune", value=False)

    pre_ep = st.slider("Pretrain epochs", 10, 200, 50)
    ft_ep = st.slider("Fine-tune epochs", 10, 200, 50)

    if src_key == tgt_key:
        st.warning("Source and target are the same — choose different datasets to test transfer.")

    if st.button("Run Transfer Learning", type="primary"):
        from src.evaluation.transfer import pretrain_then_finetune
        src_df = av[src_key]
        tgt_df = av[tgt_key]
        device = st.session_state.device
        prog = st.progress(0.0)
        status = st.empty()
        steps = {"Pretraining on source...": 0.33, "Fine-tuning on target...": 0.66,
                 "Training from scratch on target...": 1.0}
        def _cb(msg):
            status.markdown(f"**{msg}**")
            if msg in steps:
                prog.progress(steps[msg])
        try:
            res = pretrain_then_finetune(
                src_df, tgt_df, model_name=model_name,
                window_size=window_size,
                pretrain_epochs=pre_ep, finetune_epochs=ft_ep,
                device=device, freeze_encoder=freeze, progress_cb=_cb,
            )
            st.session_state["transfer_result"] = {"src": src_key, "tgt": tgt_key, "res": res,
                                                     "model": model_name}
            prog.progress(1.0); status.empty()
        except Exception as e:
            st.error(f"Failed: {e}")

    cache = st.session_state.get("transfer_result")
    if not cache:
        st.info("Configure and click **Run Transfer Learning**.")
        return

    res = cache["res"]
    st.markdown("---")
    st.subheader(f"Result — {cache['model']}: {cache['src']} → {cache['tgt']}")
    feats = res["common_features"]
    st.caption(f"Common features used ({len(feats)}): "
               f"`{', '.join(feats[:10])}{'...' if len(feats) > 10 else ''}`")

    cmp = pd.DataFrame({
        "From-scratch": res["from_scratch"]["metrics"],
        "Pretrain → Fine-tune": res["finetune"]["metrics"],
    }).T
    st.dataframe(cmp.style.format("{:.4f}"), use_container_width=True)

    improv = res["improvement"]
    cols = st.columns(3)
    cols[0].metric("Δ MAE", f"{improv.get('MAE', 0):+.3f} kW",
                    help="Positive = transfer helped (lower MAE for fine-tune).")
    cols[1].metric("Δ RMSE", f"{improv.get('RMSE', 0):+.3f} kW")
    cols[2].metric("Δ MAPE", f"{improv.get('MAPE', 0):+.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=res["from_scratch"]["history"]["val_loss"],
                              name="Scratch (val)", line=dict(color="orangered")))
    fig.add_trace(go.Scatter(y=res["finetune"]["history"]["val_loss"],
                              name="Fine-tune (val)", line=dict(color="seagreen")))
    fig.update_layout(title="Validation loss on target", xaxis_title="Epoch",
                      yaxis_title="MSE", height=350)
    st.plotly_chart(fig, use_container_width=True)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(y=res["finetune"]["targets"], name="Actual",
                                line=dict(color="black", width=2)))
    fig2.add_trace(go.Scatter(y=res["from_scratch"]["preds"], name="Scratch",
                                line=dict(color="orangered")))
    fig2.add_trace(go.Scatter(y=res["finetune"]["preds"], name="Fine-tuned",
                                line=dict(color="seagreen")))
    fig2.update_layout(title="Test-set predictions", height=350,
                        yaxis_title="Power (kW)")
    st.plotly_chart(fig2, use_container_width=True)


# Full human-readable descriptions for dataset columns (shown on hover)
V2020_COL_DESCRIPTIONS = {
    "worker_name": "Worker name — identifies an instance (container); join key with pai_instance_table",
    "machine": "Machine name (anonymised) — physical server hosting this instance",
    "start_time": "Instance start timestamp (seconds, offset for anonymisation)",
    "end_time": "Instance end timestamp (seconds, offset for anonymisation)",
    "machine_cpu_iowait": "Machine-level CPU I/O wait (%), averaged over instance lifetime",
    "machine_cpu_kernel": "Machine-level CPU kernel usage (%), averaged over instance lifetime",
    "machine_cpu_usr": "Machine-level CPU user-space usage (%), averaged over instance lifetime",
    "machine_gpu": "Machine-level GPU utilisation (%) — summed across all GPUs on the machine, averaged over instance lifetime",
    "machine_load_1": "Machine-level 1-minute load average, averaged over instance lifetime",
    "machine_net_receive": "Machine-level network bytes received, averaged over instance lifetime",
    "machine_num_worker": "Number of co-located instances (workers) on the machine, averaged over instance lifetime",
    "machine_cpu": "Machine-level overall CPU usage (%), averaged over instance lifetime",
    "timestamp": "Midpoint of (start_time, end_time) — used as the representative timestamp",
}

GENAI_COL_DESCRIPTIONS = {
    "timestamp": "Anonymised timestamp (seconds since epoch)",
    "container_id": "Container / pod IP (anonymised)",
    "gpu_util": "GPU duty cycle — how busy the GPU is (0–100%)",
    "gpu_mem_bytes": "GPU memory used (bytes)",
    "mem_util": "System (pod) memory utilisation (0–1 fraction)",
    "qps": "Queries per second — incoming request rate",
}


def _show_grid(df, page_size=15, height=400, column_tooltips=None):
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=page_size)
    gb.configure_default_column(filterable=True, sortable=True, resizable=True)

    # Use text filter with "contains" default on every column so users
    # can do substring searches on any column including numeric ones.
    for col in df.columns:
        extra = {}
        if column_tooltips and col in column_tooltips:
            extra["headerTooltip"] = column_tooltips[col]
        gb.configure_column(
            col,
            filter="agTextColumnFilter",
            filterParams={"defaultOption": "contains"},
            **extra,
        )

    grid_options = gb.build()
    # Allow native browser text selection so users can select + Ctrl-C
    grid_options["enableCellTextSelection"] = True
    grid_options["ensureDomOrder"] = True

    AgGrid(
        df,
        gridOptions=grid_options,
        height=height,
        fit_columns_on_grid_load=True,
        theme="streamlit",
    )

if __name__ == "__main__":
    main()

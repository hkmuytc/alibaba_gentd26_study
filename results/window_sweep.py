from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from results.support.window_sweep_cli import (
    load_cached_results,
    parse_args,
    print_data_and_cache_policy,
    print_run_configuration,
    resolve_device,
    resolve_training_budget,
)
from results.support.window_sweep_config import (
    FIXED_WINDOW,
    MAX_HORIZON,
    MODEL_NAMES,
    N_CUTOFFS,
    OUTPUT_DIR,
    WINDOW_SIZES,
    DATASET_NAME,
)
from results.support.window_sweep_experiments import (
    run_multistep_horizon_study,
    run_one_step_lookback_study,
)
from results.support.window_sweep_plots import (
    plot_heatmap,
    plot_horizon_results,
    plot_lookback_results,
    print_horizon_summary,
    print_lookback_summary,
)
from results.support.window_sweep_analysis import generate_analysis_outputs
from src.data_processing.pipeline import (
    aggregate_genai_sources,
    engineer_power_features,
    estimate_cluster_power,
    save_processed_dataset,
)


def main() -> None:
    args = parse_args()

    device = resolve_device()
    epochs, patience = resolve_training_budget()

    print_run_configuration(device, epochs, patience)
    print_data_and_cache_policy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = run_data_processing_stage(args)

    lookback_cache = OUTPUT_DIR / "lookback_results.json"
    horizon_cache = OUTPUT_DIR / "horizon_results.json"

    if args.plot_only:
        lookback_results = load_cached_results(lookback_cache)
        horizon_results = load_cached_results(horizon_cache)
    else:
        lookback_results = run_one_step_lookback_study(
            df,
            device,
            epochs,
            patience,
            lookback_cache,
            rebuild_cache=args.rebuild_study_results,
        )
        horizon_results = run_multistep_horizon_study(
            df,
            device,
            epochs,
            patience,
            horizon_cache,
            rebuild_cache=args.rebuild_study_results,
        )

    visualize_sweep_results(lookback_results, horizon_results)
    if df is not None:
        generate_analysis_outputs(df, device, epochs, patience, OUTPUT_DIR)
    print(f"\nDone.  All outputs in: {OUTPUT_DIR.relative_to(PROJECT_ROOT)}/")


def describe_aggregation_step(aggregation_meta: dict) -> None:
    """Print a short summary of the raw-to-cluster aggregation step."""
    print("\n[Data Processing: Aggregate Raw Signals]")
    print(f"  Dataset          : {DATASET_NAME}")
    print("  Sources          : gpu_util, gpu_mem_bytes, mem_util, qps")
    print("  Bin width        : 300 seconds")
    print(f"  Output rows      : {aggregation_meta['n_bins']:,}")
    print(f"  Input containers : {aggregation_meta['n_containers']:,}")


def describe_power_estimation_step(power_df, cluster_units: int) -> None:
    """Print a short summary of the power-estimation step."""
    print("\n[Data Processing: Estimate Power]")
    print(f"  Cluster units    : {cluster_units}")
    print("  Target variable  : power_total_kw")
    print(f"  Output rows      : {len(power_df):,}")


def run_data_processing_stage(args):
    """Run aggregation, power estimation, and feature engineering from raw data."""
    if args.plot_only:
        return None

    aggregated_cluster_data, aggregation_meta = aggregate_genai_sources(freq_seconds=300)
    describe_aggregation_step(aggregation_meta)

    power_estimated_data = estimate_cluster_power(
        aggregated_cluster_data,
        DATASET_NAME,
        args.cluster_units,
    )
    describe_power_estimation_step(power_estimated_data, args.cluster_units)

    engineered_feature_data = engineer_power_features(power_estimated_data)
    save_processed_snapshot(engineered_feature_data)
    return engineered_feature_data


def save_processed_snapshot(df) -> None:
    """Persist a processed CSV snapshot for inspection and reproducibility."""
    save_path = save_processed_dataset(df, DATASET_NAME, freq_seconds=300)
    print("\n[Data Processing: Engineer Features]")
    print(f"  Final rows       : {len(df):,}")
    print(f"  Final columns    : {len(df.columns):,}")
    print(f"  Saved snapshot   : {save_path.relative_to(PROJECT_ROOT)}")


def visualize_sweep_results(lookback_results, horizon_results) -> None:
    """Generate whichever figures have fresh or cached inputs available."""
    print("\n[Plots]")
    if lookback_results:
        plot_lookback_results(lookback_results, WINDOW_SIZES, OUTPUT_DIR)
        plot_heatmap(lookback_results, WINDOW_SIZES, OUTPUT_DIR)
    else:
        print("  (skipping lookback plots — no data)")

    if horizon_results:
        plot_horizon_results(horizon_results, MAX_HORIZON, OUTPUT_DIR)
    else:
        print("  (skipping horizon plot — no data)")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from src.api_clients import PolymarketClient
from src.build_dataset import (
    build_tidy_dataframe,
    build_volatility_analysis_tables,
    build_wide_dataframe,
    compute_calibration_by_horizon,
    compute_category_calibration_by_horizon,
    compute_category_horizon_error_metrics,
    compute_category_isotonic_metrics_by_horizon,
    compute_mae_by_horizon_global_volume_decile,
    compute_isotonic_calibration_by_horizon,
    compute_isotonic_gap_by_volume_decile,
    compute_isotonic_gap_by_volume_decile_price_bin,
    compute_isotonic_gap_threshold_summary,
    compute_isotonic_volume_interpretation_table,
    compute_lowess_calibration_by_horizon,
    compute_mae_change_distribution_by_global_decile,
    compute_price_change_distribution_by_global_decile,
    compute_staleness_by_horizon_and_volume_decile,
    compute_volume_error_control_analysis,
    compute_volume_error_joint_diagnostics,
)
from src.collect_markets import discover_top_volume_markets
from src.collect_snapshots import build_snapshot_rows
from src.config import DEFAULT_HORIZONS_MINUTES, PipelineConfig
from src.plots import (
    save_calibration_plot,
    save_category_calibration_plot,
    save_category_error_by_horizon_plot,
    save_category_isotonic_gap_by_horizon_plot,
    save_horizon_error_plot,
    save_isotonic_calibration_plot,
    save_lowess_calibration_plot,
    save_mae_change_distribution_explorer,
    save_mae_global_volume_bucket_plot,
    save_mae_by_volume_decile_plot,
    save_overlapping_bin_plot,
    save_price_change_distribution_explorer,
    save_signed_isotonic_gap_by_volume_decile_price_bin_plot,
    save_signed_error_by_price_bin_horizon_plot,
    save_signed_error_by_price_bin_plot,
    save_signed_error_by_volume_decile_price_bin_plot,
    save_staleness_by_volume_decile_plot,
    save_volume_error_control_coefficients_plot,
    save_volume_error_joint_diagnostics_plot,
    save_volatility_isotonic_gap_plot,
    save_volatility_realized_error_plot,
    save_isotonic_gap_by_volume_decile_plot,
    save_volume_bucket_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket mispricing initial pass")
    parser.add_argument("--target-markets", type=int, default=30000)
    parser.add_argument("--lookback-days", type=int, default=90)
    parser.add_argument(
        "--min-volume",
        type=float,
        default=50_000.0,
        help="Minimum Gamma volumeNum per market; 0 disables",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default=",".join(str(h) for h in DEFAULT_HORIZONS_MINUTES),
        help="Comma-separated minute horizons",
    )
    return parser.parse_args()


def _safe_output_path(path: Path) -> Path:
    try:
        with path.open("a", encoding="utf-8"):
            pass
        return path
    except PermissionError:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return path.with_name(f"{path.stem}_{ts}{path.suffix}")


def main() -> None:
    args = parse_args()
    horizons = tuple(int(h.strip()) for h in args.horizons.split(",") if h.strip())

    config = PipelineConfig(
        target_markets=args.target_markets,
        lookback_days=args.lookback_days,
        min_volume=args.min_volume,
        horizons_minutes=horizons,
    )
    config.ensure_output_dirs()

    start = time.perf_counter()
    client = PolymarketClient(
        timeout_seconds=config.request_timeout_seconds,
        max_retries=config.max_retries,
        backoff_base_seconds=config.backoff_base_seconds,
    )

    try:
        markets = discover_top_volume_markets(client, config)
        snapshots = build_snapshot_rows(client, markets, config)

        tidy_df = build_tidy_dataframe(snapshots)
        wide_df = build_wide_dataframe(tidy_df)
        calibration_df, horizon_metrics_df = compute_calibration_by_horizon(tidy_df)
        isotonic_points_df, isotonic_horizon_metrics_df = compute_isotonic_calibration_by_horizon(
            tidy_df
        )
        isotonic_gap_volume_decile_df = compute_isotonic_gap_by_volume_decile(isotonic_points_df)
        isotonic_gap_volume_decile_price_bin_df = compute_isotonic_gap_by_volume_decile_price_bin(
            isotonic_points_df, bin_width=0.1, stride=0.1
        )
        isotonic_gap_threshold_df = compute_isotonic_gap_threshold_summary(
            isotonic_gap_volume_decile_df
        )
        isotonic_interpretation_df = compute_isotonic_volume_interpretation_table(isotonic_points_df)
        lowess_points_df, lowess_horizon_metrics_df = compute_lowess_calibration_by_horizon(tidy_df)
        category_error_metrics_df = compute_category_horizon_error_metrics(tidy_df)
        category_calibration_df, category_horizon_metrics_df = compute_category_calibration_by_horizon(
            tidy_df
        )
        category_isotonic_metrics_df = compute_category_isotonic_metrics_by_horizon(tidy_df)
        volatility_market_df, volatility_bucket_df, volatility_threshold_df = (
            build_volatility_analysis_tables(wide_df, tidy_df, isotonic_points_df)
        )
        staleness_volume_df = compute_staleness_by_horizon_and_volume_decile(tidy_df)
        mae_global_volume_df = compute_mae_by_horizon_global_volume_decile(tidy_df)
        joint_diag_df = compute_volume_error_joint_diagnostics(tidy_df, wide_df)
        control_coef_df, control_model_df = compute_volume_error_control_analysis(tidy_df, wide_df)
        price_change_dist_df = compute_price_change_distribution_by_global_decile(wide_df)
        mae_change_dist_df = compute_mae_change_distribution_by_global_decile(wide_df)

        market_path = _safe_output_path(config.output_dir / "market_snapshots.csv")
        wide_path = _safe_output_path(config.output_dir / "market_snapshots_wide.csv")
        calibration_path = _safe_output_path(config.output_dir / "calibration_by_horizon.csv")
        horizon_metrics_path = _safe_output_path(config.output_dir / "horizon_metrics.csv")
        isotonic_points_path = _safe_output_path(
            config.output_dir / "isotonic_points_by_horizon.csv"
        )
        isotonic_horizon_metrics_path = _safe_output_path(
            config.output_dir / "isotonic_horizon_metrics.csv"
        )
        lowess_points_path = _safe_output_path(config.output_dir / "lowess_points_by_horizon.csv")
        lowess_horizon_metrics_path = _safe_output_path(
            config.output_dir / "lowess_horizon_metrics.csv"
        )
        category_error_metrics_path = _safe_output_path(
            config.output_dir / "category_horizon_error_metrics.csv"
        )
        category_calibration_path = _safe_output_path(
            config.output_dir / "category_calibration_by_horizon.csv"
        )
        category_horizon_metrics_path = _safe_output_path(
            config.output_dir / "category_horizon_calibration_metrics.csv"
        )
        category_isotonic_metrics_path = _safe_output_path(
            config.output_dir / "category_isotonic_horizon_metrics.csv"
        )
        isotonic_gap_volume_decile_path = _safe_output_path(
            config.output_dir / "isotonic_gap_by_volume_decile.csv"
        )
        isotonic_gap_volume_decile_price_bin_path = _safe_output_path(
            config.output_dir / "isotonic_gap_by_volume_decile_price_bin.csv"
        )
        isotonic_gap_threshold_path = _safe_output_path(
            config.output_dir / "isotonic_gap_threshold_summary.csv"
        )
        isotonic_interpretation_path = _safe_output_path(
            config.output_dir / "isotonic_volume_interpretation_table.csv"
        )
        volatility_market_path = _safe_output_path(config.output_dir / "volatility_market_analysis.csv")
        volatility_bucket_path = _safe_output_path(config.output_dir / "volatility_bucket_summary.csv")
        volatility_threshold_path = _safe_output_path(
            config.output_dir / "volatility_threshold_summary.csv"
        )
        staleness_volume_path = _safe_output_path(
            config.output_dir / "staleness_by_horizon_volume_decile.csv"
        )
        mae_global_volume_path = _safe_output_path(
            config.output_dir / "mae_by_horizon_global_volume_decile.csv"
        )
        joint_diag_path = _safe_output_path(
            config.output_dir / "volume_error_joint_diagnostics.csv"
        )
        control_coef_path = _safe_output_path(
            config.output_dir / "volume_error_control_coefficients.csv"
        )
        control_model_path = _safe_output_path(
            config.output_dir / "volume_error_control_model_summary.csv"
        )
        price_change_dist_path = _safe_output_path(
            config.output_dir / "price_change_distribution_by_global_decile.csv"
        )
        mae_change_dist_path = _safe_output_path(
            config.output_dir / "mae_change_distribution_by_global_decile.csv"
        )

        tidy_df.to_csv(market_path, index=False)
        wide_df.to_csv(wide_path, index=False)
        calibration_df.to_csv(calibration_path, index=False)
        horizon_metrics_df.to_csv(horizon_metrics_path, index=False)
        isotonic_points_df.to_csv(isotonic_points_path, index=False)
        isotonic_horizon_metrics_df.to_csv(isotonic_horizon_metrics_path, index=False)
        lowess_points_df.to_csv(lowess_points_path, index=False)
        lowess_horizon_metrics_df.to_csv(lowess_horizon_metrics_path, index=False)
        category_error_metrics_df.to_csv(category_error_metrics_path, index=False)
        category_calibration_df.to_csv(category_calibration_path, index=False)
        category_horizon_metrics_df.to_csv(category_horizon_metrics_path, index=False)
        category_isotonic_metrics_df.to_csv(category_isotonic_metrics_path, index=False)
        isotonic_gap_volume_decile_df.to_csv(isotonic_gap_volume_decile_path, index=False)
        isotonic_gap_volume_decile_price_bin_df.to_csv(
            isotonic_gap_volume_decile_price_bin_path, index=False
        )
        isotonic_gap_threshold_df.to_csv(isotonic_gap_threshold_path, index=False)
        isotonic_interpretation_df.to_csv(isotonic_interpretation_path, index=False)
        volatility_market_df.to_csv(volatility_market_path, index=False)
        volatility_bucket_df.to_csv(volatility_bucket_path, index=False)
        volatility_threshold_df.to_csv(volatility_threshold_path, index=False)
        staleness_volume_df.to_csv(staleness_volume_path, index=False)
        mae_global_volume_df.to_csv(mae_global_volume_path, index=False)
        joint_diag_df.to_csv(joint_diag_path, index=False)
        control_coef_df.to_csv(control_coef_path, index=False)
        control_model_df.to_csv(control_model_path, index=False)
        price_change_dist_df.to_csv(price_change_dist_path, index=False)
        mae_change_dist_df.to_csv(mae_change_dist_path, index=False)

        save_calibration_plot(calibration_df, config.plots_dir)
        save_category_calibration_plot(category_calibration_df, config.plots_dir)
        save_isotonic_calibration_plot(isotonic_points_df, config.plots_dir)
        save_lowess_calibration_plot(lowess_points_df, config.plots_dir)
        save_category_error_by_horizon_plot(category_error_metrics_df, config.plots_dir)
        save_category_isotonic_gap_by_horizon_plot(category_isotonic_metrics_df, config.plots_dir)
        save_isotonic_gap_by_volume_decile_plot(isotonic_gap_volume_decile_df, config.plots_dir)
        save_signed_isotonic_gap_by_volume_decile_price_bin_plot(
            isotonic_gap_volume_decile_price_bin_df, config.plots_dir
        )
        save_horizon_error_plot(horizon_metrics_df, config.plots_dir)
        save_volume_bucket_plot(tidy_df, config.plots_dir)
        save_mae_by_volume_decile_plot(tidy_df, config.plots_dir)
        save_staleness_by_volume_decile_plot(staleness_volume_df, config.plots_dir)
        save_mae_global_volume_bucket_plot(mae_global_volume_df, config.plots_dir)
        save_volume_error_joint_diagnostics_plot(joint_diag_df, config.plots_dir)
        save_volume_error_control_coefficients_plot(control_coef_df, config.plots_dir)
        save_price_change_distribution_explorer(price_change_dist_df, config.plots_dir)
        save_mae_change_distribution_explorer(mae_change_dist_df, config.plots_dir)
        save_overlapping_bin_plot(tidy_df, config.plots_dir, bin_width=0.02, stride=0.01)
        save_signed_error_by_price_bin_plot(tidy_df, config.plots_dir, bin_width=0.1, stride=0.02)
        save_signed_error_by_price_bin_horizon_plot(
            tidy_df, config.plots_dir, bin_width=0.1, stride=0.02
        )
        save_signed_error_by_volume_decile_price_bin_plot(
            tidy_df, config.plots_dir, bin_width=0.1, stride=0.1
        )
        save_volatility_isotonic_gap_plot(volatility_bucket_df, config.plots_dir)
        save_volatility_realized_error_plot(volatility_bucket_df, config.plots_dir)

        elapsed = time.perf_counter() - start
        diagnostics = {
            "run_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "markets_selected": len(markets),
            "snapshot_rows": len(snapshots),
            "gamma_requests": client.stats["gamma"].requests,
            "gamma_retries": client.stats["gamma"].retries,
            "gamma_failures": client.stats["gamma"].failures,
            "clob_requests": client.stats["clob"].requests,
            "clob_retries": client.stats["clob"].retries,
            "clob_failures": client.stats["clob"].failures,
            "config": asdict(config),
            "outputs": {
                "tidy_csv": str(market_path),
                "wide_csv": str(wide_path),
                "calibration_csv": str(calibration_path),
                "horizon_metrics_csv": str(horizon_metrics_path),
                "isotonic_points_csv": str(isotonic_points_path),
                "isotonic_horizon_metrics_csv": str(isotonic_horizon_metrics_path),
                "isotonic_gap_by_volume_decile_csv": str(isotonic_gap_volume_decile_path),
                "isotonic_gap_by_volume_decile_price_bin_csv": str(
                    isotonic_gap_volume_decile_price_bin_path
                ),
                "isotonic_gap_threshold_summary_csv": str(isotonic_gap_threshold_path),
                "isotonic_volume_interpretation_table_csv": str(isotonic_interpretation_path),
                "lowess_points_csv": str(lowess_points_path),
                "lowess_horizon_metrics_csv": str(lowess_horizon_metrics_path),
                "category_horizon_error_metrics_csv": str(category_error_metrics_path),
                "category_calibration_by_horizon_csv": str(category_calibration_path),
                "category_horizon_calibration_metrics_csv": str(category_horizon_metrics_path),
                "category_isotonic_horizon_metrics_csv": str(category_isotonic_metrics_path),
                "volatility_market_analysis_csv": str(volatility_market_path),
                "volatility_bucket_summary_csv": str(volatility_bucket_path),
                "volatility_threshold_summary_csv": str(volatility_threshold_path),
                "staleness_by_horizon_volume_decile_csv": str(staleness_volume_path),
                "mae_by_horizon_global_volume_decile_csv": str(mae_global_volume_path),
                "volume_error_joint_diagnostics_csv": str(joint_diag_path),
                "volume_error_control_coefficients_csv": str(control_coef_path),
                "volume_error_control_model_summary_csv": str(control_model_path),
                "price_change_distribution_by_global_decile_csv": str(price_change_dist_path),
                "mae_change_distribution_by_global_decile_csv": str(mae_change_dist_path),
                "plots_dir": str(config.plots_dir),
            },
        }

        diagnostics["config"]["output_dir"] = str(Path(diagnostics["config"]["output_dir"]))
        diagnostics["config"]["plots_dir"] = str(Path(diagnostics["config"]["plots_dir"]))

        diagnostics_path = config.output_dir / "run_diagnostics.json"
        diagnostics_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")

        print(json.dumps(diagnostics, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()


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
    build_wide_dataframe,
    compute_calibration_by_horizon,
)
from src.collect_markets import discover_top_volume_markets
from src.collect_snapshots import build_snapshot_rows
from src.config import PipelineConfig
from src.plots import (
    save_calibration_plot,
    save_horizon_error_plot,
    save_mae_by_volume_decile_plot,
    save_overlapping_bin_plot,
    save_signed_error_by_price_bin_horizon_plot,
    save_signed_error_by_price_bin_plot,
    save_signed_error_by_volume_decile_price_bin_plot,
    save_volume_bucket_plot,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket mispricing initial pass")
    parser.add_argument("--target-markets", type=int, default=10000)
    parser.add_argument("--lookback-days", type=int, default=30)
    parser.add_argument(
        "--horizons",
        type=str,
        default="10,30,60,120,240,360,1440,2880",
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

        market_path = _safe_output_path(config.output_dir / "market_snapshots.csv")
        wide_path = _safe_output_path(config.output_dir / "market_snapshots_wide.csv")
        calibration_path = _safe_output_path(config.output_dir / "calibration_by_horizon.csv")
        horizon_metrics_path = _safe_output_path(config.output_dir / "horizon_metrics.csv")

        tidy_df.to_csv(market_path, index=False)
        wide_df.to_csv(wide_path, index=False)
        calibration_df.to_csv(calibration_path, index=False)
        horizon_metrics_df.to_csv(horizon_metrics_path, index=False)

        save_calibration_plot(calibration_df, config.plots_dir)
        save_horizon_error_plot(horizon_metrics_df, config.plots_dir)
        save_volume_bucket_plot(tidy_df, config.plots_dir)
        save_mae_by_volume_decile_plot(tidy_df, config.plots_dir)
        save_overlapping_bin_plot(tidy_df, config.plots_dir, bin_width=0.1, stride=0.02)
        save_signed_error_by_price_bin_plot(tidy_df, config.plots_dir, bin_width=0.1, stride=0.02)
        save_signed_error_by_price_bin_horizon_plot(
            tidy_df, config.plots_dir, bin_width=0.1, stride=0.02
        )
        save_signed_error_by_volume_decile_price_bin_plot(
            tidy_df, config.plots_dir, bin_width=0.1, stride=0.1
        )

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


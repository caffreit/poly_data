from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px


def save_calibration_plot(calibration_df: pd.DataFrame, plots_dir: Path) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "calibration_horizon.html"
    output_png = plots_dir / "calibration_horizon.png"

    if calibration_df.empty:
        fig = px.scatter(title="No calibration data available")
    else:
        fig = px.scatter(
            calibration_df,
            x="predicted_mean",
            y="observed_yes_rate",
            color="horizon_min",
            size="n",
            hover_data=["prob_bin", "n", "volume_mean"],
            title="Reliability by Time-to-Resolution Horizon",
            labels={
                "predicted_mean": "Mean predicted YES probability",
                "observed_yes_rate": "Observed YES rate",
                "horizon_min": "Horizon (minutes)",
            },
        )
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])

    fig.write_html(output_html)
    try:
        fig.write_image(output_png)
    except Exception:
        # PNG export depends on runtime image backends; HTML is always produced.
        pass
    return output_html


def save_horizon_error_plot(horizon_metrics_df: pd.DataFrame, plots_dir: Path) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "error_by_horizon.html"
    output_png = plots_dir / "error_by_horizon.png"

    if horizon_metrics_df.empty:
        fig = px.line(title="No horizon metrics available")
    else:
        fig = px.line(
            horizon_metrics_df.sort_values("horizon_min"),
            x="horizon_min",
            y=["mae", "brier"],
            markers=True,
            title="Forecast Error vs Time-to-Resolution",
            labels={
                "horizon_min": "Horizon (minutes before close)",
                "value": "Error",
                "variable": "Metric",
            },
        )

    fig.write_html(output_html)
    try:
        fig.write_image(output_png)
    except Exception:
        pass
    return output_html


def save_volume_bucket_plot(tidy_df: pd.DataFrame, plots_dir: Path) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "mispricing_vs_volume.html"
    output_png = plots_dir / "mispricing_vs_volume.png"

    if tidy_df.empty:
        fig = px.bar(title="No volume/mispricing data available")
        fig.write_html(output_html)
        return output_html

    df = tidy_df.dropna(subset=["price_last_trade"]).copy()
    if df.empty:
        fig = px.bar(title="No price data available")
        fig.write_html(output_html)
        return output_html

    df["volume_bucket"] = pd.qcut(
        df["volume_total_market"].rank(method="first"),
        q=10,
        labels=[f"decile_{i}" for i in range(1, 11)],
    )
    bucketed = (
        df.groupby(["horizon_min", "volume_bucket"], observed=True)
        .agg(
            mae=("abs_error_last_trade", "mean"),
            n=("market_id", "count"),
        )
        .reset_index()
    )

    fig = px.bar(
        bucketed,
        x="volume_bucket",
        y="mae",
        color="horizon_min",
        barmode="group",
        hover_data=["n"],
        title="Average Absolute Mispricing by Volume Decile",
        labels={"mae": "Mean absolute error", "volume_bucket": "Volume decile"},
    )
    fig.write_html(output_html)
    try:
        fig.write_image(output_png)
    except Exception:
        pass
    return output_html


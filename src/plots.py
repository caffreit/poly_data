from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px

_VOLUME_DECILE_LABELS = (
    ["Decile 1 (lowest)"]
    + [f"Decile {i}" for i in range(2, 10)]
    + ["Decile 10 (highest)"]
)


def _overlapping_price_bin_starts(bin_width: float = 0.1, stride: float = 0.02) -> list[float]:
    starts: list[float] = []
    s = 0.0
    while s <= 1.0 - bin_width + 1e-12:
        starts.append(round(s, 10))
        s += stride
    return starts


def _mae_long_by_horizon_and_volume_decile(tidy_df: pd.DataFrame) -> pd.DataFrame:
    df = tidy_df.dropna(subset=["price_last_trade"]).copy()
    if df.empty:
        return df
    df["volume_bucket"] = pd.qcut(
        df["volume_total_market"].rank(method="first"),
        q=10,
        labels=_VOLUME_DECILE_LABELS,
    )
    return (
        df.groupby(["horizon_min", "volume_bucket"], observed=True)
        .agg(
            mae=("abs_error_last_trade", "mean"),
            n=("market_id", "count"),
        )
        .reset_index()
    )


def save_calibration_plot(calibration_df: pd.DataFrame, plots_dir: Path) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "calibration_horizon.html"
    output_png = plots_dir / "calibration_horizon.png"

    if calibration_df.empty:
        fig = px.line(title="No calibration data available")
    else:
        plot_df = calibration_df.sort_values(["horizon_min", "predicted_mean"]).copy()
        plot_df["horizon_label"] = plot_df["horizon_min"].map(lambda m: f"{m} min")
        fig = px.line(
            plot_df,
            x="predicted_mean",
            y="observed_yes_rate",
            color="horizon_label",
            markers=True,
            hover_data=["horizon_min", "prob_bin", "n", "volume_mean"],
            title="Reliability by Time-to-Resolution Horizon",
            labels={
                "predicted_mean": "Mean predicted YES probability",
                "observed_yes_rate": "Observed YES rate",
                "horizon_label": "Horizon",
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
        fig = px.line(title="No volume/mispricing data available")
        fig.write_html(output_html)
        return output_html

    bucketed = _mae_long_by_horizon_and_volume_decile(tidy_df)
    if bucketed.empty:
        fig = px.line(title="No price data available")
        fig.write_html(output_html)
        return output_html

    plot_df = bucketed.sort_values(["volume_bucket", "horizon_min"])
    fig = px.line(
        plot_df,
        x="horizon_min",
        y="mae",
        color="volume_bucket",
        markers=True,
        category_orders={"volume_bucket": list(_VOLUME_DECILE_LABELS)},
        hover_data=["n"],
        title="Mean Absolute Error vs Horizon by Volume Decile",
        labels={
            "horizon_min": "Horizon (minutes before close)",
            "mae": "Mean absolute error",
            "volume_bucket": "Volume decile",
        },
    )
    fig.update_xaxes(type="log")
    fig.write_html(output_html)
    try:
        fig.write_image(output_png)
    except Exception:
        pass
    return output_html


def save_mae_by_volume_decile_plot(tidy_df: pd.DataFrame, plots_dir: Path) -> Path:
    """MAE vs volume decile (x), one line per snapshot horizon (minutes before close)."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "mae_by_volume_decile.html"
    output_png = plots_dir / "mae_by_volume_decile.png"

    if tidy_df.empty:
        fig = px.line(title="No volume/mispricing data available")
        fig.write_html(output_html)
        return output_html

    bucketed = _mae_long_by_horizon_and_volume_decile(tidy_df)
    if bucketed.empty:
        fig = px.line(title="No price data available")
        fig.write_html(output_html)
        return output_html

    plot_df = bucketed.copy()
    plot_df["volume_bucket"] = pd.Categorical(
        plot_df["volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    plot_df = plot_df.sort_values(["horizon_min", "volume_bucket"])
    plot_df["horizon_label"] = plot_df["horizon_min"].map(lambda m: f"{int(m)} min")
    horizon_order = [
        f"{int(m)} min" for m in sorted(plot_df["horizon_min"].dropna().unique())
    ]

    fig = px.line(
        plot_df,
        x="volume_bucket",
        y="mae",
        color="horizon_label",
        markers=True,
        category_orders={
            "volume_bucket": list(_VOLUME_DECILE_LABELS),
            "horizon_label": horizon_order,
        },
        hover_data=["horizon_min", "n"],
        title="Mean Absolute Error vs Volume Decile by Horizon",
        labels={
            "volume_bucket": "Volume decile",
            "mae": "Mean absolute error",
            "horizon_label": "Horizon (minutes before close)",
        },
    )
    fig.write_html(output_html)
    try:
        fig.write_image(output_png)
    except Exception:
        pass
    return output_html


def save_signed_error_by_price_bin_plot(
    tidy_df: pd.DataFrame,
    plots_dir: Path,
    bin_width: float = 0.1,
    stride: float = 0.02,
) -> Path:
    """Mean signed error (price − outcome) vs overlapping price bins; one line per volume decile."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "signed_error_by_price_bin.html"
    output_png = plots_dir / "signed_error_by_price_bin.png"

    if tidy_df.empty:
        fig = px.line(title="No data available")
        fig.write_html(output_html)
        return output_html

    df = tidy_df.dropna(subset=["price_last_trade"]).copy()
    if df.empty:
        fig = px.line(title="No price data available")
        fig.write_html(output_html)
        return output_html

    df["signed_error"] = df["price_last_trade"] - df["outcome_yes"]
    df["volume_bucket"] = pd.qcut(
        df["volume_total_market"].rank(method="first"),
        q=10,
        labels=_VOLUME_DECILE_LABELS,
    )

    starts = _overlapping_price_bin_starts(bin_width, stride)
    rows: list[dict] = []
    for start in starts:
        end = min(1.0, start + bin_width)
        if end >= 1.0:
            in_bin = df[(df["price_last_trade"] >= start) & (df["price_last_trade"] <= end)]
        else:
            in_bin = df[(df["price_last_trade"] >= start) & (df["price_last_trade"] < end)]
        if in_bin.empty:
            continue
        for volume_bucket, sub in in_bin.groupby("volume_bucket", observed=True):
            if sub.empty:
                continue
            rows.append(
                {
                    "volume_bucket": volume_bucket,
                    "bin_start": start,
                    "bin_end": end,
                    "bin_mid": (start + end) / 2.0,
                    "mean_signed_error": float(sub["signed_error"].mean()),
                    "n": int(len(sub)),
                }
            )

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        fig = px.line(title="No overlapping-bin signed-error points available")
        fig.write_html(output_html)
        return output_html

    plot_df["volume_bucket"] = pd.Categorical(
        plot_df["volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    plot_df = plot_df.sort_values(["volume_bucket", "bin_mid"])

    fig = px.line(
        plot_df,
        x="bin_mid",
        y="mean_signed_error",
        color="volume_bucket",
        markers=True,
        category_orders={"volume_bucket": list(_VOLUME_DECILE_LABELS)},
        hover_data=["bin_start", "bin_end", "n"],
        title=(
            "Mean signed pricing error by overlapping price bins "
            f"(width={bin_width}, stride={stride}), by volume decile"
        ),
        labels={
            "bin_mid": "Price bin midpoint",
            "mean_signed_error": "Mean (YES price − outcome)",
            "volume_bucket": "Volume decile",
        },
    )
    fig.update_xaxes(range=[0, 1])
    fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(0,0,0,0.4)", line_width=1)
    fig.write_html(output_html)
    try:
        fig.write_image(output_png)
    except Exception:
        pass
    return output_html


def save_signed_error_by_price_bin_horizon_plot(
    tidy_df: pd.DataFrame,
    plots_dir: Path,
    bin_width: float = 0.1,
    stride: float = 0.02,
) -> Path:
    """Mean signed error vs overlapping price bins; one line per snapshot horizon (minutes before close)."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "signed_error_by_price_bin_horizon.html"
    output_png = plots_dir / "signed_error_by_price_bin_horizon.png"

    if tidy_df.empty:
        fig = px.line(title="No data available")
        fig.write_html(output_html)
        return output_html

    df = tidy_df.dropna(subset=["price_last_trade"]).copy()
    if df.empty:
        fig = px.line(title="No price data available")
        fig.write_html(output_html)
        return output_html

    df["signed_error"] = df["price_last_trade"] - df["outcome_yes"]
    starts = _overlapping_price_bin_starts(bin_width, stride)
    rows: list[dict] = []
    for horizon in sorted(df["horizon_min"].dropna().unique()):
        d_h = df[df["horizon_min"] == horizon]
        for start in starts:
            end = min(1.0, start + bin_width)
            if end >= 1.0:
                in_bin = d_h[
                    (d_h["price_last_trade"] >= start) & (d_h["price_last_trade"] <= end)
                ]
            else:
                in_bin = d_h[
                    (d_h["price_last_trade"] >= start) & (d_h["price_last_trade"] < end)
                ]
            if in_bin.empty:
                continue
            h = int(horizon)
            rows.append(
                {
                    "horizon_min": h,
                    "bin_start": start,
                    "bin_end": end,
                    "bin_mid": (start + end) / 2.0,
                    "mean_signed_error": float(in_bin["signed_error"].mean()),
                    "n": int(len(in_bin)),
                }
            )

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        fig = px.line(title="No overlapping-bin signed-error points available")
        fig.write_html(output_html)
        return output_html

    plot_df = plot_df.sort_values(["horizon_min", "bin_mid"])
    plot_df["horizon_label"] = plot_df["horizon_min"].map(lambda m: f"{int(m)} min")
    horizon_order = [f"{int(m)} min" for m in sorted(plot_df["horizon_min"].dropna().unique())]

    fig = px.line(
        plot_df,
        x="bin_mid",
        y="mean_signed_error",
        color="horizon_label",
        markers=True,
        category_orders={"horizon_label": horizon_order},
        hover_data=["horizon_min", "bin_start", "bin_end", "n"],
        title=(
            "Mean signed pricing error by overlapping price bins "
            f"(width={bin_width}, stride={stride}), by horizon"
        ),
        labels={
            "bin_mid": "Price bin midpoint",
            "mean_signed_error": "Mean (YES price − outcome)",
            "horizon_label": "Horizon (minutes before close)",
        },
    )
    fig.update_xaxes(range=[0, 1])
    fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(0,0,0,0.4)", line_width=1)
    fig.write_html(output_html)
    try:
        fig.write_image(output_png)
    except Exception:
        pass
    return output_html


def save_signed_error_by_volume_decile_price_bin_plot(
    tidy_df: pd.DataFrame,
    plots_dir: Path,
    bin_width: float = 0.1,
    stride: float = 0.1,
) -> Path:
    """Mean signed error vs volume decile (x); one line per price-bin midpoint (default stride=0.1)."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "signed_error_by_volume_decile_price_bin.html"
    output_png = plots_dir / "signed_error_by_volume_decile_price_bin.png"

    if tidy_df.empty:
        fig = px.line(title="No data available")
        fig.write_html(output_html)
        return output_html

    df = tidy_df.dropna(subset=["price_last_trade"]).copy()
    if df.empty:
        fig = px.line(title="No price data available")
        fig.write_html(output_html)
        return output_html

    df["signed_error"] = df["price_last_trade"] - df["outcome_yes"]
    df["volume_bucket"] = pd.qcut(
        df["volume_total_market"].rank(method="first"),
        q=10,
        labels=_VOLUME_DECILE_LABELS,
    )

    starts = _overlapping_price_bin_starts(bin_width, stride)
    rows: list[dict] = []
    for volume_bucket in _VOLUME_DECILE_LABELS:
        d_v = df[df["volume_bucket"] == volume_bucket]
        if d_v.empty:
            continue
        for start in starts:
            end = min(1.0, start + bin_width)
            if end >= 1.0:
                in_bin = d_v[
                    (d_v["price_last_trade"] >= start) & (d_v["price_last_trade"] <= end)
                ]
            else:
                in_bin = d_v[
                    (d_v["price_last_trade"] >= start) & (d_v["price_last_trade"] < end)
                ]
            if in_bin.empty:
                continue
            rows.append(
                {
                    "volume_bucket": volume_bucket,
                    "bin_start": start,
                    "bin_end": end,
                    "bin_mid": (start + end) / 2.0,
                    "mean_signed_error": float(in_bin["signed_error"].mean()),
                    "n": int(len(in_bin)),
                }
            )

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        fig = px.line(title="No volume-decile / price-bin signed-error points available")
        fig.write_html(output_html)
        return output_html

    plot_df["volume_bucket"] = pd.Categorical(
        plot_df["volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    bin_mid_order = sorted(plot_df["bin_mid"].unique())
    bin_label_order = [f"{m:.3f}" for m in bin_mid_order]
    plot_df["bin_label"] = plot_df["bin_mid"].map(lambda m: f"{float(m):.3f}")
    plot_df["bin_label"] = pd.Categorical(
        plot_df["bin_label"],
        categories=bin_label_order,
        ordered=True,
    )
    plot_df = plot_df.sort_values(["bin_label", "volume_bucket"])

    fig = px.line(
        plot_df,
        x="volume_bucket",
        y="mean_signed_error",
        color="bin_label",
        markers=True,
        category_orders={
            "volume_bucket": list(_VOLUME_DECILE_LABELS),
            "bin_label": bin_label_order,
        },
        hover_data=["bin_mid", "bin_start", "bin_end", "n"],
        title=(
            "Mean signed pricing error vs volume decile by price bin midpoint "
            f"(width={bin_width}, stride={stride})"
        ),
        labels={
            "volume_bucket": "Volume decile",
            "mean_signed_error": "Mean (YES price − outcome)",
            "bin_label": "Price bin midpoint",
        },
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(0,0,0,0.4)", line_width=1)
    fig.write_html(output_html)
    try:
        fig.write_image(output_png)
    except Exception:
        pass
    return output_html


def save_overlapping_bin_plot(
    tidy_df: pd.DataFrame,
    plots_dir: Path,
    bin_width: float = 0.1,
    stride: float = 0.02,
) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "overlapping_bin_outcome_rate.html"
    output_png = plots_dir / "overlapping_bin_outcome_rate.png"

    if tidy_df.empty:
        fig = px.line(title="No data available")
        fig.write_html(output_html)
        return output_html

    df = tidy_df.dropna(subset=["price_last_trade"]).copy()
    if df.empty:
        fig = px.line(title="No price data available")
        fig.write_html(output_html)
        return output_html

    starts = _overlapping_price_bin_starts(bin_width, stride)

    rows = []
    for horizon in sorted(df["horizon_min"].dropna().unique()):
        d_h = df[df["horizon_min"] == horizon]
        for start in starts:
            end = min(1.0, start + bin_width)
            if end >= 1.0:
                in_bin = d_h[(d_h["price_last_trade"] >= start) & (d_h["price_last_trade"] <= end)]
            else:
                in_bin = d_h[(d_h["price_last_trade"] >= start) & (d_h["price_last_trade"] < end)]
            n = len(in_bin)
            if n == 0:
                continue
            rows.append(
                {
                    "horizon_min": int(horizon),
                    "bin_start": start,
                    "bin_end": end,
                    "bin_mid": (start + end) / 2.0,
                    "n": n,
                    # outcome_yes=1 means the positive outcome resolved true.
                    "outcome_rate": float(in_bin["outcome_yes"].mean()),
                    "mean_price": float(in_bin["price_last_trade"].mean()),
                    "volume_mean": float(in_bin["volume_total_market"].mean()),
                }
            )

    overlap_df = pd.DataFrame(rows)
    if overlap_df.empty:
        fig = px.line(title="No overlapping-bin points available")
        fig.write_html(output_html)
        return output_html

    plot_df = overlap_df.sort_values(["horizon_min", "bin_mid"]).copy()
    plot_df["horizon_label"] = plot_df["horizon_min"].map(lambda m: f"{m} min")

    fig = px.line(
        plot_df,
        x="bin_mid",
        y="outcome_rate",
        color="horizon_label",
        markers=True,
        hover_data=["horizon_min", "bin_start", "bin_end", "n", "mean_price", "volume_mean"],
        title=f"Outcome Rate by Overlapping Price Bins (width={bin_width}, stride={stride})",
        labels={
            "bin_mid": "Price bin midpoint",
            "outcome_rate": "Fraction with resolved outcome = 1",
            "horizon_label": "Horizon",
            "volume_mean": "Mean market volume",
        },
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    fig.write_html(output_html)
    try:
        fig.write_image(output_png)
    except Exception:
        pass
    return output_html


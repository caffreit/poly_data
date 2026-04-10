from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.build_dataset import _normalized_category_labels

# HTML and static image export size (Plotly default ~700×500 is small for slides/reports).
FIGURE_WIDTH = 1600
FIGURE_HEIGHT = 1000


def _write_figure_outputs(fig, output_html: Path, output_png: Path | None = None) -> None:
    # Plotly 6 + Kaleido often exports 700×500 unless width/height are passed to write_image.
    fig.update_layout(
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        autosize=False,
    )
    fig.write_html(output_html, config={"responsive": False})
    if output_png is not None:
        try:
            fig.write_image(
                output_png,
                width=FIGURE_WIDTH,
                height=FIGURE_HEIGHT,
                scale=1,
            )
        except Exception:
            # PNG export depends on runtime image backends; HTML is always produced.
            pass


def _add_perfect_calibration_line(fig) -> None:
    """y = x from (0,0) to (1,1) for reliability / price-vs-outcome plots."""
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line={"dash": "dash", "color": "rgba(0,0,0,0.4)", "width": 1},
        layer="below",
    )


_VOLUME_DECILE_LABELS = (
    ["Decile 1 (lowest)"]
    + [f"Decile {i}" for i in range(2, 10)]
    + ["Decile 10 (highest)"]
)


def _volume_deciles_within_horizon(volume: pd.Series) -> pd.Series:
    """Equal-count volume deciles within one horizon (one row per market at that horizon)."""
    if volume.empty:
        return pd.Series(dtype=object, index=volume.index)
    r = volume.rank(method="first")
    n = len(volume)
    k = min(10, n)
    if k < 1:
        return pd.Series(index=volume.index, dtype=object)
    labels = list(_VOLUME_DECILE_LABELS[:k])
    return pd.qcut(r, q=k, labels=labels)


def _assign_volume_deciles_within_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with volume_bucket from quantiles of volume_total_market within each horizon_min."""
    out = df.copy()
    parts: list[pd.Series] = []
    for _, g in out.groupby("horizon_min", sort=False):
        parts.append(_volume_deciles_within_horizon(g["volume_total_market"]))
    out["volume_bucket"] = pd.concat(parts)
    return out


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
    df = _assign_volume_deciles_within_horizon(df)
    return (
        df.groupby(["horizon_min", "volume_bucket"], observed=True)
        .agg(
            mae=("abs_error_last_trade", "mean"),
            n=("market_id", "count"),
            decile_volume_min=("volume_total_market", "min"),
            decile_volume_max=("volume_total_market", "max"),
        )
        .reset_index()
    )


def _filter_categories_for_plot(
    df: pd.DataFrame,
    min_total_n: int = 250,
    category_col: str = "category",
    n_col: str = "n",
) -> pd.DataFrame:
    if df.empty or category_col not in df.columns:
        return df.copy()

    out = df.copy()
    out[category_col] = out[category_col].astype(str)
    if n_col in out.columns:
        totals = out.groupby(category_col, dropna=False)[n_col].sum().sort_values(ascending=False)
        keep = totals[totals >= min_total_n].index.tolist()
        if keep:
            out = out[out[category_col].isin(keep)].copy()
            out[category_col] = pd.Categorical(out[category_col], categories=keep, ordered=True)
        else:
            out = out.iloc[0:0].copy()
    return out


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
        _add_perfect_calibration_line(fig)

    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def _isotonic_reliability_binned(
    df: pd.DataFrame,
    group_keys: list[str],
    bin_width: float,
    stride: float,
) -> pd.DataFrame:
    """Overlapping bins on calibrated_prob; mean outcome_yes rate per bin within each group."""
    need = set(group_keys) | {"calibrated_prob", "outcome_yes"}
    if df.empty or not need.issubset(df.columns):
        return pd.DataFrame()

    d = df.dropna(subset=["calibrated_prob", "outcome_yes"]).copy()
    if d.empty:
        return pd.DataFrame()

    d["calibrated_prob"] = d["calibrated_prob"].clip(0, 1).astype(float)
    starts = _overlapping_price_bin_starts(bin_width, stride)
    rows: list[dict] = []
    for gkey, d_g in d.groupby(group_keys, dropna=False):
        keys = gkey if isinstance(gkey, tuple) else (gkey,)
        for start in starts:
            end = min(1.0, start + bin_width)
            if end >= 1.0:
                in_bin = d_g[
                    (d_g["calibrated_prob"] >= start) & (d_g["calibrated_prob"] <= end)
                ]
            else:
                in_bin = d_g[
                    (d_g["calibrated_prob"] >= start) & (d_g["calibrated_prob"] < end)
                ]
            if in_bin.empty:
                continue
            row = {k: v for k, v in zip(group_keys, keys)}
            row["bin_start"] = start
            row["bin_end"] = end
            row["bin_mid"] = (start + end) / 2.0
            row["mean_calibrated_prob"] = float(in_bin["calibrated_prob"].mean())
            row["outcome_rate"] = float(in_bin["outcome_yes"].mean())
            row["n"] = int(len(in_bin))
            rows.append(row)
    return pd.DataFrame(rows)


def save_isotonic_reliability_by_horizon_plot(
    isotonic_points_df: pd.DataFrame,
    plots_dir: Path,
    bin_width: float = 0.05,
    stride: float = 0.025,
) -> Path:
    """Reliability: x = isotonic calibrated probability (binned), y = fraction resolving YES, by horizon."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "isotonic_reliability_by_horizon.html"
    output_png = plots_dir / "isotonic_reliability_by_horizon.png"

    if isotonic_points_df.empty:
        fig = px.line(title="No isotonic reliability data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = _isotonic_reliability_binned(
        isotonic_points_df,
        ["horizon_min"],
        bin_width=bin_width,
        stride=stride,
    )
    if plot_df.empty:
        fig = px.line(title="No isotonic reliability bin aggregates available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = plot_df.sort_values(["horizon_min", "mean_calibrated_prob"])
    plot_df["horizon_label"] = plot_df["horizon_min"].map(lambda m: f"{int(m)} min")
    horizon_order = [f"{int(m)} min" for m in sorted(plot_df["horizon_min"].dropna().unique())]

    fig = px.line(
        plot_df,
        x="mean_calibrated_prob",
        y="outcome_rate",
        color="horizon_label",
        markers=True,
        category_orders={"horizon_label": horizon_order},
        hover_data=[
            "horizon_min",
            "bin_start",
            "bin_end",
            "n",
            "bin_mid",
        ],
        title=(
            "Reliability vs isotonic calibrated probability by horizon "
            f"(overlapping bins: width={bin_width}, stride={stride})"
        ),
        labels={
            "mean_calibrated_prob": "Isotonic calibrated probability (mean in bin)",
            "outcome_rate": "Fraction resolving YES (outcome = 1)",
            "horizon_label": "Horizon",
        },
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    _add_perfect_calibration_line(fig)
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_isotonic_reliability_by_category_plot(
    isotonic_points_df: pd.DataFrame,
    tidy_df: pd.DataFrame,
    plots_dir: Path,
    min_total_n: int = 250,
    bin_width: float = 0.05,
    stride: float = 0.025,
) -> Path:
    """Reliability by category; all horizons pooled (each row still uses that horizon's isotonic map)."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "isotonic_reliability_by_category.html"
    output_png = plots_dir / "isotonic_reliability_by_category.png"

    required_iso = {"market_id", "horizon_min", "calibrated_prob", "outcome_yes"}
    if isotonic_points_df.empty or not required_iso.issubset(isotonic_points_df.columns):
        fig = px.line(title="No isotonic reliability data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html
    if tidy_df.empty or "category" not in tidy_df.columns:
        fig = px.line(title="No category column on tidy data for isotonic reliability plot")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    cat_map = (
        tidy_df[["market_id", "horizon_min", "category"]]
        .drop_duplicates(subset=["market_id", "horizon_min"])
        .copy()
    )
    cat_map["category"] = _normalized_category_labels(cat_map["category"])
    merged = isotonic_points_df.merge(cat_map, on=["market_id", "horizon_min"], how="left")
    merged["category"] = _normalized_category_labels(merged["category"])

    base = merged.dropna(subset=["calibrated_prob", "outcome_yes"])
    cat_counts = base.groupby("category", dropna=False).size()
    keep_cats = cat_counts[cat_counts >= min_total_n].index.tolist()
    if not keep_cats:
        fig = px.line(title="No categories meet minimum sample threshold for plotting")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    merged = merged[merged["category"].isin(keep_cats)].copy()
    plot_df = _isotonic_reliability_binned(
        merged,
        ["category"],
        bin_width=bin_width,
        stride=stride,
    )
    if plot_df.empty:
        fig = px.line(title="No isotonic reliability bin aggregates available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    category_order = (
        base[base["category"].isin(keep_cats)]
        .groupby("category", dropna=False)
        .size()
        .sort_values(ascending=False)
        .index.tolist()
    )
    plot_df["category"] = pd.Categorical(
        plot_df["category"], categories=category_order, ordered=True
    )
    plot_df = plot_df.sort_values(["category", "mean_calibrated_prob"])

    fig = px.line(
        plot_df,
        x="mean_calibrated_prob",
        y="outcome_rate",
        color="category",
        markers=True,
        category_orders={"category": category_order},
        hover_data=["bin_start", "bin_end", "n", "bin_mid"],
        title=(
            f"Reliability vs isotonic calibrated probability by category "
            f"(all horizons pooled; categories with n >= {min_total_n}; "
            f"bins width={bin_width}, stride={stride})"
        ),
        labels={
            "mean_calibrated_prob": "Isotonic calibrated probability (mean in bin)",
            "outcome_rate": "Fraction resolving YES (outcome = 1)",
            "category": "Category",
        },
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    _add_perfect_calibration_line(fig)
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_isotonic_calibration_plot(isotonic_points_df: pd.DataFrame, plots_dir: Path) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "isotonic_calibration_horizon.html"
    output_png = plots_dir / "isotonic_calibration_horizon.png"

    if isotonic_points_df.empty:
        fig = px.line(title="No isotonic calibration data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = (
        isotonic_points_df.groupby(["horizon_min", "price_last_trade"], as_index=False)
        .agg(
            calibrated_prob=("calibrated_prob", "mean"),
            observed_yes_rate=("outcome_yes", "mean"),
            n=("outcome_yes", "count"),
        )
        .sort_values(["horizon_min", "price_last_trade"])
    )
    plot_df["horizon_label"] = plot_df["horizon_min"].map(lambda m: f"{int(m)} min")

    fig = px.line(
        plot_df,
        x="price_last_trade",
        y="calibrated_prob",
        color="horizon_label",
        markers=True,
        hover_data=["horizon_min", "observed_yes_rate", "n"],
        title="Isotonic Calibration by Time-to-Resolution Horizon",
        labels={
            "price_last_trade": "Raw YES price",
            "calibrated_prob": "Isotonic calibrated probability",
            "horizon_label": "Horizon",
        },
    )
    _add_perfect_calibration_line(fig)
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_lowess_calibration_plot(lowess_points_df: pd.DataFrame, plots_dir: Path) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "lowess_calibration_horizon.html"
    output_png = plots_dir / "lowess_calibration_horizon.png"

    if lowess_points_df.empty:
        fig = px.line(title="No LOWESS calibration data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = (
        lowess_points_df.groupby(["horizon_min", "price_last_trade"], as_index=False)
        .agg(
            lowess_prob=("lowess_prob", "mean"),
            observed_yes_rate=("outcome_yes", "mean"),
            n=("outcome_yes", "count"),
        )
        .sort_values(["horizon_min", "price_last_trade"])
    )
    plot_df["horizon_label"] = plot_df["horizon_min"].map(lambda m: f"{int(m)} min")

    fig = px.line(
        plot_df,
        x="price_last_trade",
        y="lowess_prob",
        color="horizon_label",
        markers=True,
        hover_data=["horizon_min", "observed_yes_rate", "n"],
        title="LOWESS Calibration by Time-to-Resolution Horizon",
        labels={
            "price_last_trade": "Raw YES price",
            "lowess_prob": "LOWESS calibrated probability",
            "horizon_label": "Horizon",
        },
    )
    _add_perfect_calibration_line(fig)
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    _write_figure_outputs(fig, output_html, output_png)
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

    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_category_error_by_horizon_plot(
    category_error_df: pd.DataFrame,
    plots_dir: Path,
    min_total_n: int = 250,
) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "category_error_by_horizon.html"
    output_png = plots_dir / "category_error_by_horizon.png"

    if category_error_df.empty:
        fig = px.line(title="No category horizon error data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = _filter_categories_for_plot(category_error_df, min_total_n=min_total_n)
    if plot_df.empty:
        fig = px.line(title="No categories meet minimum sample threshold for plotting")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = plot_df.sort_values(["category", "horizon_min"])
    fig = px.line(
        plot_df,
        x="horizon_min",
        y="mae",
        color="category",
        markers=True,
        hover_data={
            "n": True,
            "n_unique_markets": True,
            "brier": ":.4f",
            "mean_signed_error": ":.4f",
            "predicted_mean": ":.4f",
            "observed_yes_rate": ":.4f",
            "mean_volume": ":,.0f",
        },
        title=f"Category MAE vs Horizon (categories with n >= {min_total_n})",
        labels={
            "horizon_min": "Horizon (minutes before close)",
            "mae": "Mean absolute error",
            "category": "Category",
        },
    )
    fig.update_xaxes(type="log")
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_category_calibration_plot(
    category_calibration_df: pd.DataFrame,
    plots_dir: Path,
    min_total_n: int = 250,
) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "category_calibration_horizon.html"
    output_png = plots_dir / "category_calibration_horizon.png"

    if category_calibration_df.empty:
        fig = px.line(title="No category calibration data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = _filter_categories_for_plot(category_calibration_df, min_total_n=min_total_n)
    if plot_df.empty:
        fig = px.line(title="No categories meet minimum sample threshold for plotting")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = plot_df.sort_values(["horizon_min", "category", "predicted_mean"])
    plot_df["horizon_label"] = plot_df["horizon_min"].map(lambda m: f"{int(m)} min")
    horizon_order = sorted(plot_df["horizon_label"].dropna().unique(), key=lambda x: int(x.split()[0]))

    fig = px.line(
        plot_df,
        x="predicted_mean",
        y="observed_yes_rate",
        color="category",
        facet_col="horizon_label",
        facet_col_wrap=4,
        markers=True,
        category_orders={"horizon_label": horizon_order},
        hover_data=["prob_bin", "n", "volume_mean"],
        title=f"Category Calibration Curves by Horizon (categories with n >= {min_total_n})",
        labels={
            "predicted_mean": "Mean predicted YES probability",
            "observed_yes_rate": "Observed YES rate",
            "category": "Category",
            "horizon_label": "Horizon",
        },
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    _add_perfect_calibration_line(fig)
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_category_calibration_overall_plot(
    category_calibration_df: pd.DataFrame,
    plots_dir: Path,
    min_total_n: int = 250,
) -> Path:
    """Category calibration curves aggregated across all horizons."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "category_calibration_overall.html"
    output_png = plots_dir / "category_calibration_overall.png"

    if category_calibration_df.empty:
        fig = px.line(title="No category calibration data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    base_df = _filter_categories_for_plot(category_calibration_df, min_total_n=min_total_n)
    if base_df.empty:
        fig = px.line(title="No categories meet minimum sample threshold for plotting")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    def _weighted_mean(group: pd.DataFrame, value_col: str, weight_col: str = "n") -> float:
        vals = pd.to_numeric(group[value_col], errors="coerce")
        w = pd.to_numeric(group[weight_col], errors="coerce")
        valid = vals.notna() & w.notna() & (w > 0)
        if not valid.any():
            return float("nan")
        return float(np.average(vals[valid], weights=w[valid]))

    rows: list[dict] = []
    for (category, prob_bin), g in base_df.groupby(
        ["category", "prob_bin"], dropna=False, observed=True
    ):
        rows.append(
            {
                "category": category,
                "prob_bin": prob_bin,
                "n": int(pd.to_numeric(g["n"], errors="coerce").fillna(0).sum()),
                "predicted_mean": _weighted_mean(g, "predicted_mean"),
                "observed_yes_rate": _weighted_mean(g, "observed_yes_rate"),
                "volume_mean": _weighted_mean(g, "volume_mean"),
            }
        )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        fig = px.line(title="No aggregated category calibration data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    category_order = (
        plot_df.groupby("category", dropna=False)["n"].sum().sort_values(ascending=False).index.tolist()
    )
    plot_df["category"] = pd.Categorical(plot_df["category"], categories=category_order, ordered=True)
    plot_df = plot_df.sort_values(["category", "predicted_mean"])

    fig = px.line(
        plot_df,
        x="predicted_mean",
        y="observed_yes_rate",
        color="category",
        markers=True,
        category_orders={"category": category_order},
        hover_data={"prob_bin": True, "n": True, "volume_mean": ":,.0f"},
        title=f"Category Calibration Curves (All Horizons, categories with n >= {min_total_n})",
        labels={
            "predicted_mean": "Mean predicted YES probability",
            "observed_yes_rate": "Observed YES rate",
            "category": "Category",
        },
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(range=[0, 1])
    _add_perfect_calibration_line(fig)
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_category_isotonic_gap_by_horizon_plot(
    category_isotonic_metrics_df: pd.DataFrame,
    plots_dir: Path,
    min_total_n: int = 250,
) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "category_isotonic_gap_by_horizon.html"
    output_png = plots_dir / "category_isotonic_gap_by_horizon.png"

    if category_isotonic_metrics_df.empty:
        fig = px.line(title="No category isotonic metrics data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = _filter_categories_for_plot(category_isotonic_metrics_df, min_total_n=min_total_n)
    if plot_df.empty:
        fig = px.line(title="No categories meet minimum sample threshold for plotting")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = plot_df.sort_values(["category", "horizon_min"])
    fig = px.line(
        plot_df,
        x="horizon_min",
        y="mean_abs_isotonic_gap",
        color="category",
        markers=True,
        hover_data={
            "n": True,
            "n_unique_markets": True,
            "mean_signed_isotonic_gap": ":.4f",
            "share_positive_isotonic_gap": ":.3f",
            "raw_mae": ":.4f",
            "calibrated_mae": ":.4f",
            "raw_brier": ":.4f",
            "calibrated_brier": ":.4f",
            "mean_volume": ":,.0f",
        },
        title=f"Category Mean Absolute Isotonic Gap vs Horizon (categories with n >= {min_total_n})",
        labels={
            "horizon_min": "Horizon (minutes before close)",
            "mean_abs_isotonic_gap": "Mean absolute isotonic gap",
            "category": "Category",
        },
    )
    fig.update_xaxes(type="log")
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_volume_bucket_plot(tidy_df: pd.DataFrame, plots_dir: Path) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "mispricing_vs_volume.html"
    output_png = plots_dir / "mispricing_vs_volume.png"

    if tidy_df.empty:
        fig = px.line(title="No volume/mispricing data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    bucketed = _mae_long_by_horizon_and_volume_decile(tidy_df)
    if bucketed.empty:
        fig = px.line(title="No price data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = bucketed.sort_values(["volume_bucket", "horizon_min"])
    fig = px.line(
        plot_df,
        x="horizon_min",
        y="mae",
        color="volume_bucket",
        markers=True,
        category_orders={"volume_bucket": list(_VOLUME_DECILE_LABELS)},
        hover_data={
            "n": True,
            "decile_volume_min": ":,.0f",
            "decile_volume_max": ":,.0f",
        },
        title="Mean Absolute Error vs Horizon by Volume Decile",
        labels={
            "horizon_min": "Horizon (minutes before close)",
            "mae": "Mean absolute error",
            "volume_bucket": "Volume decile",
            "decile_volume_min": "Decile volume (min)",
            "decile_volume_max": "Decile volume (max)",
        },
    )
    fig.update_xaxes(type="log")
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_mae_by_volume_decile_plot(tidy_df: pd.DataFrame, plots_dir: Path) -> Path:
    """MAE vs volume decile (x), one line per horizon; deciles are equal-count within each horizon."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "mae_by_volume_decile.html"
    output_png = plots_dir / "mae_by_volume_decile.png"

    if tidy_df.empty:
        fig = px.line(title="No volume/mispricing data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    bucketed = _mae_long_by_horizon_and_volume_decile(tidy_df)
    if bucketed.empty:
        fig = px.line(title="No price data available")
        _write_figure_outputs(fig, output_html, None)
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
        hover_data={
            "horizon_min": True,
            "n": True,
            "decile_volume_min": ":,.0f",
            "decile_volume_max": ":,.0f",
        },
        title="Mean Absolute Error vs Volume Decile by Horizon",
        labels={
            "volume_bucket": "Volume decile",
            "mae": "Mean absolute error",
            "horizon_label": "Horizon (minutes before close)",
            "decile_volume_min": "Decile volume (min)",
            "decile_volume_max": "Decile volume (max)",
        },
    )
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_staleness_by_volume_decile_plot(staleness_df: pd.DataFrame, plots_dir: Path) -> Path:
    """Mean staleness vs horizon, one line per within-horizon volume decile."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "staleness_by_volume_decile_horizon.html"
    output_png = plots_dir / "staleness_by_volume_decile_horizon.png"

    if staleness_df.empty:
        fig = px.line(title="No staleness-by-volume data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    required = {"horizon_min", "volume_bucket", "mean_staleness_min"}
    if not required.issubset(set(staleness_df.columns)):
        fig = px.line(title="Staleness summary columns are unavailable")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = staleness_df.dropna(subset=["horizon_min", "volume_bucket"]).copy()
    if plot_df.empty:
        fig = px.line(title="No staleness points available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df["volume_bucket"] = pd.Categorical(
        plot_df["volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    plot_df = plot_df.sort_values(["volume_bucket", "horizon_min"])

    fig = px.line(
        plot_df,
        x="horizon_min",
        y="mean_staleness_min",
        color="volume_bucket",
        markers=True,
        category_orders={"volume_bucket": list(_VOLUME_DECILE_LABELS)},
        hover_data={
            "n_rows": True,
            "n_price_available": True,
            "n_staleness_available": True,
            "median_staleness_min": ":.2f",
            "p90_staleness_min": ":.2f",
            "decile_volume_min": ":,.0f",
            "decile_volume_max": ":,.0f",
        },
        title="Mean Price Staleness vs Horizon by Volume Decile",
        labels={
            "horizon_min": "Horizon (minutes before close)",
            "mean_staleness_min": "Mean staleness (minutes since last trade)",
            "volume_bucket": "Volume decile",
            "decile_volume_min": "Decile volume (min)",
            "decile_volume_max": "Decile volume (max)",
        },
    )
    fig.update_xaxes(type="log")
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_mae_global_volume_bucket_plot(global_bucket_df: pd.DataFrame, plots_dir: Path) -> Path:
    """MAE vs horizon by global (cohort-stable) volume decile."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "mae_vs_horizon_global_volume_decile.html"
    output_png = plots_dir / "mae_vs_horizon_global_volume_decile.png"

    if global_bucket_df.empty:
        fig = px.line(title="No global-volume MAE data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    required = {"horizon_min", "global_volume_bucket", "mae"}
    if not required.issubset(set(global_bucket_df.columns)):
        fig = px.line(title="Global-volume MAE columns are unavailable")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = global_bucket_df.dropna(subset=["horizon_min", "global_volume_bucket", "mae"]).copy()
    if plot_df.empty:
        fig = px.line(title="No global-volume MAE points available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df["global_volume_bucket"] = pd.Categorical(
        plot_df["global_volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )

    has_snapshot_date = "snapshot_date" in plot_df.columns
    if has_snapshot_date:
        date_series = pd.to_datetime(plot_df["snapshot_date"], errors="coerce")
        plot_df = plot_df.loc[date_series.notna()].copy()
        plot_df["snapshot_date"] = date_series.loc[date_series.notna()].dt.strftime("%Y-%m-%d")
        has_snapshot_date = not plot_df.empty

    if not has_snapshot_date:
        plot_df = plot_df.sort_values(["global_volume_bucket", "horizon_min"])
        fig = px.line(
            plot_df,
            x="horizon_min",
            y="mae",
            color="global_volume_bucket",
            markers=True,
            category_orders={"global_volume_bucket": list(_VOLUME_DECILE_LABELS)},
            hover_data={
                "n": True,
                "global_volume_min": ":,.0f",
                "global_volume_max": ":,.0f",
            },
            title="Mean Absolute Error vs Horizon by Global Volume Decile",
            labels={
                "horizon_min": "Horizon (minutes before close)",
                "mae": "Mean absolute error",
                "global_volume_bucket": "Global volume decile",
                "global_volume_min": "Global decile volume (min)",
                "global_volume_max": "Global decile volume (max)",
            },
        )
        fig.update_xaxes(type="log")
        _write_figure_outputs(fig, output_html, output_png)
        return output_html

    # Static PNG summary across full date range (weighted by row counts).
    full_range = (
        plot_df.groupby(["horizon_min", "global_volume_bucket"], observed=True)
        .agg(
            n=("n", "sum"),
            mae_weighted_sum=("mae", lambda s: float((s * plot_df.loc[s.index, "n"]).sum())),
            global_volume_min=("global_volume_min", "min"),
            global_volume_max=("global_volume_max", "max"),
        )
        .reset_index()
    )
    full_range["mae"] = full_range["mae_weighted_sum"] / full_range["n"].clip(lower=1)
    full_range = full_range.sort_values(["global_volume_bucket", "horizon_min"])
    png_fig = px.line(
        full_range,
        x="horizon_min",
        y="mae",
        color="global_volume_bucket",
        markers=True,
        category_orders={"global_volume_bucket": list(_VOLUME_DECILE_LABELS)},
        hover_data={
            "n": True,
            "global_volume_min": ":,.0f",
            "global_volume_max": ":,.0f",
        },
        title="Mean Absolute Error vs Horizon by Global Volume Decile",
        labels={
            "horizon_min": "Horizon (minutes before close)",
            "mae": "Mean absolute error",
            "global_volume_bucket": "Global volume decile",
            "global_volume_min": "Global decile volume (min)",
            "global_volume_max": "Global decile volume (max)",
        },
    )
    png_fig.update_xaxes(type="log")
    try:
        png_fig.write_image(
            output_png,
            width=FIGURE_WIDTH,
            height=FIGURE_HEIGHT,
            scale=1,
        )
    except Exception:
        pass

    records_cols = [
        "snapshot_date",
        "horizon_min",
        "global_volume_bucket",
        "mae",
        "n",
        "global_volume_min",
        "global_volume_max",
    ]
    records = plot_df[records_cols].copy()
    records["horizon_min"] = records["horizon_min"].astype(float)
    records["mae"] = records["mae"].astype(float)
    records["n"] = records["n"].fillna(0).astype(float)
    records["global_volume_min"] = records["global_volume_min"].fillna(0).astype(float)
    records["global_volume_max"] = records["global_volume_max"].fillna(0).astype(float)
    records["global_volume_bucket"] = records["global_volume_bucket"].astype(str)

    date_min = str(records["snapshot_date"].min())
    date_max = str(records["snapshot_date"].max())

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>MAE vs Horizon by Global Volume Decile</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; }}
    .controls {{ display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }}
    label {{ font-size: 14px; }}
    input[type="date"] {{ padding: 4px 6px; }}
    button {{ padding: 6px 10px; cursor: pointer; }}
    .hint {{ margin: 8px 0 14px 0; font-size: 13px; color: #444; }}
  </style>
</head>
<body>
  <h2>Mean Absolute Error vs Horizon by Global Volume Decile</h2>
  <div class="controls">
    <label>From:
      <input type="date" id="fromDate" />
    </label>
    <label>To:
      <input type="date" id="toDate" />
    </label>
    <button id="resetBtn" type="button">Reset</button>
  </div>
  <div id="hint" class="hint"></div>
  <div id="plot" style="width: 100%; max-width: 1600px; height: 1000px;"></div>

  <script>
    const rows = {json.dumps(records.to_dict("records"))};
    const bucketOrder = {json.dumps(list(_VOLUME_DECILE_LABELS))};
    const fromInput = document.getElementById('fromDate');
    const toInput = document.getElementById('toDate');
    const resetBtn = document.getElementById('resetBtn');
    const hint = document.getElementById('hint');
    const minDate = '{date_min}';
    const maxDate = '{date_max}';

    fromInput.min = minDate;
    fromInput.max = maxDate;
    toInput.min = minDate;
    toInput.max = maxDate;
    fromInput.value = minDate;
    toInput.value = maxDate;

    function inRange(dateStr, fromStr, toStr) {{
      return (!fromStr || dateStr >= fromStr) && (!toStr || dateStr <= toStr);
    }}

    function aggregate(filteredRows) {{
      const grouped = new Map();
      for (const r of filteredRows) {{
        const key = `${{r.global_volume_bucket}}|${{r.horizon_min}}`;
        if (!grouped.has(key)) {{
          grouped.set(key, {{
            global_volume_bucket: r.global_volume_bucket,
            horizon_min: Number(r.horizon_min),
            n: 0,
            mae_weighted_sum: 0,
            global_volume_min: Number(r.global_volume_min),
            global_volume_max: Number(r.global_volume_max),
          }});
        }}
        const g = grouped.get(key);
        const n = Number(r.n) || 0;
        const mae = Number(r.mae) || 0;
        g.n += n;
        g.mae_weighted_sum += mae * n;
        g.global_volume_min = Math.min(g.global_volume_min, Number(r.global_volume_min) || 0);
        g.global_volume_max = Math.max(g.global_volume_max, Number(r.global_volume_max) || 0);
      }}
      const out = [];
      grouped.forEach((g) => {{
        if (g.n > 0) {{
          out.push({{
            global_volume_bucket: g.global_volume_bucket,
            horizon_min: g.horizon_min,
            mae: g.mae_weighted_sum / g.n,
            n: g.n,
            global_volume_min: g.global_volume_min,
            global_volume_max: g.global_volume_max,
          }});
        }}
      }});
      return out;
    }}

    function buildTraces(points) {{
      const byBucket = new Map();
      for (const p of points) {{
        if (!byBucket.has(p.global_volume_bucket)) byBucket.set(p.global_volume_bucket, []);
        byBucket.get(p.global_volume_bucket).push(p);
      }}
      const traces = [];
      for (const bucket of bucketOrder) {{
        const pts = byBucket.get(bucket) || [];
        if (!pts.length) continue;
        pts.sort((a, b) => a.horizon_min - b.horizon_min);
        traces.push({{
          type: 'scatter',
          mode: 'lines+markers',
          name: bucket,
          x: pts.map(p => p.horizon_min),
          y: pts.map(p => p.mae),
          customdata: pts.map(p => [p.n, p.global_volume_min, p.global_volume_max]),
          hovertemplate:
            'Horizon: %{{x}} min<br>' +
            'MAE: %{{y:.4f}}<br>' +
            'n: %{{customdata[0]:.0f}}<br>' +
            'Global decile volume min: %{{customdata[1]:,.0f}}<br>' +
            'Global decile volume max: %{{customdata[2]:,.0f}}<extra>%{{fullData.name}}</extra>',
        }});
      }}
      return traces;
    }}

    function render() {{
      const fromDate = fromInput.value;
      const toDate = toInput.value;
      const filtered = rows.filter(r => inRange(r.snapshot_date, fromDate, toDate));
      const aggregated = aggregate(filtered);
      const traces = buildTraces(aggregated);

      hint.textContent = `Date range: ${{fromDate || minDate}} to ${{toDate || maxDate}} | Rows used: ${{filtered.length}}`;

      const layout = {{
        title: 'Mean Absolute Error vs Horizon by Global Volume Decile',
        xaxis: {{
          title: 'Horizon (minutes before close)',
          type: 'log'
        }},
        yaxis: {{ title: 'Mean absolute error' }},
        width: {FIGURE_WIDTH},
        height: {FIGURE_HEIGHT},
        autosize: false,
      }};
      Plotly.react('plot', traces, layout, {{ responsive: true }});
    }}

    fromInput.addEventListener('change', render);
    toInput.addEventListener('change', render);
    resetBtn.addEventListener('click', () => {{
      fromInput.value = minDate;
      toInput.value = maxDate;
      render();
    }});

    render();
  </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")
    return output_html


def save_volume_error_joint_diagnostics_plot(joint_df: pd.DataFrame, plots_dir: Path) -> Path:
    """Facet plot for MAE, staleness, and volatility by horizon and volume decile."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "volume_error_joint_diagnostics.html"
    output_png = plots_dir / "volume_error_joint_diagnostics.png"

    if joint_df.empty:
        fig = px.line(title="No joint volume-error diagnostics available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    required = {"horizon_min", "volume_bucket", "mae", "mean_staleness_min", "mean_volatility_std"}
    if not required.issubset(set(joint_df.columns)):
        fig = px.line(title="Joint diagnostics columns are unavailable")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = joint_df.dropna(subset=["horizon_min", "volume_bucket"]).copy()
    if plot_df.empty:
        fig = px.line(title="No joint diagnostics points available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    metric_map = {
        "mae": "Realized MAE",
        "mean_staleness_min": "Mean staleness (min)",
        "mean_volatility_std": "Mean market volatility (std)",
    }
    long_df = plot_df.melt(
        id_vars=[
            "horizon_min",
            "volume_bucket",
            "n",
            "decile_volume_min",
            "decile_volume_max",
            "median_staleness_min",
            "p90_staleness_min",
            "mean_volatility_range",
        ],
        value_vars=list(metric_map.keys()),
        var_name="metric",
        value_name="metric_value",
    )
    long_df["metric_label"] = long_df["metric"].map(metric_map).fillna(long_df["metric"])
    long_df["volume_bucket"] = pd.Categorical(
        long_df["volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    long_df = long_df.sort_values(["metric_label", "volume_bucket", "horizon_min"])

    fig = px.line(
        long_df,
        x="horizon_min",
        y="metric_value",
        color="volume_bucket",
        markers=True,
        facet_col="metric_label",
        facet_col_wrap=3,
        category_orders={
            "volume_bucket": list(_VOLUME_DECILE_LABELS),
            "metric_label": [metric_map["mae"], metric_map["mean_staleness_min"], metric_map["mean_volatility_std"]],
        },
        hover_data={
            "n": True,
            "decile_volume_min": ":,.0f",
            "decile_volume_max": ":,.0f",
            "median_staleness_min": ":.2f",
            "p90_staleness_min": ":.2f",
            "mean_volatility_range": ":.4f",
        },
        title="Joint Volume Diagnostics: MAE, Staleness, and Volatility",
        labels={
            "horizon_min": "Horizon (minutes before close)",
            "metric_value": "Value",
            "volume_bucket": "Volume decile",
            "metric_label": "Metric",
        },
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(type="log")
    fig.update_yaxes(matches=None)
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_volume_error_control_coefficients_plot(control_coef_df: pd.DataFrame, plots_dir: Path) -> Path:
    """Coefficient chart for focused terms in baseline vs controlled models."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "volume_error_control_coefficients.html"
    output_png = plots_dir / "volume_error_control_coefficients.png"

    if control_coef_df.empty:
        fig = px.bar(title="No control-analysis coefficients available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    required = {"model", "term", "coefficient"}
    if not required.issubset(set(control_coef_df.columns)):
        fig = px.bar(title="Control-analysis columns are unavailable")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = control_coef_df.copy()
    if "is_focus_term" in plot_df.columns:
        plot_df = plot_df[plot_df["is_focus_term"] == True]
    plot_df = plot_df[plot_df["term"] != "intercept"]
    if plot_df.empty:
        fig = px.bar(title="No focused control-analysis coefficients available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    term_order = ["log_volume", "log_horizon", "log1p_staleness", "volatility_std"]
    present_terms = [t for t in term_order if t in set(plot_df["term"])]

    fig = px.bar(
        plot_df,
        x="term",
        y="coefficient",
        color="model",
        barmode="group",
        category_orders={"term": present_terms},
        hover_data={"std_error": ":.6f", "t_stat": ":.3f", "n_obs": True, "r2": ":.4f"},
        title="Controlled Analysis Coefficients for Volume-Error Mechanism",
        labels={"term": "Regressor", "coefficient": "OLS coefficient", "model": "Model"},
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(0,0,0,0.4)", line_width=1)
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_price_change_distribution_explorer(
    price_change_df: pd.DataFrame,
    plots_dir: Path,
    tail_threshold: float = 0.1,
    bins: int = 80,
) -> Path:
    """Standalone interactive histogram explorer for horizon-pair price changes."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "price_change_distribution_explorer.html"
    output_png = plots_dir / "price_change_distribution_explorer.png"

    required = {
        "global_volume_bucket",
        "from_horizon_min",
        "to_horizon_min",
        "price_change",
    }
    if price_change_df.empty or not required.issubset(set(price_change_df.columns)):
        fig = px.histogram(title="No price-change distribution data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    df = price_change_df.dropna(
        subset=["global_volume_bucket", "from_horizon_min", "to_horizon_min", "price_change"]
    ).copy()
    if df.empty:
        fig = px.histogram(title="No valid price-change rows available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    df["global_volume_bucket"] = pd.Categorical(
        df["global_volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    df = df.sort_values(["global_volume_bucket", "from_horizon_min", "to_horizon_min"])

    bin_edges = np.linspace(-1.0, 1.0, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    nested: dict[str, dict[str, dict[str, dict[str, float | int | list[int] | list[float]]]]] = {}
    for (bucket, from_h, to_h), g in df.groupby(
        ["global_volume_bucket", "from_horizon_min", "to_horizon_min"], observed=True
    ):
        if pd.isna(bucket):
            continue
        values = g["price_change"].astype(float).to_numpy()
        if values.size == 0:
            continue
        counts, _ = np.histogram(values, bins=bin_edges)
        stats = {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q10": float(np.quantile(values, 0.10)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "q90": float(np.quantile(values, 0.90)),
            "share_tail": float(np.mean(np.abs(values) >= float(tail_threshold))),
            "counts": counts.tolist(),
        }
        bucket_key = str(bucket)
        from_key = str(int(from_h))
        to_key = str(int(to_h))
        nested.setdefault(bucket_key, {}).setdefault(from_key, {})[to_key] = stats

    if not nested:
        fig = px.histogram(title="No grouped price-change data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    deciles = [d for d in _VOLUME_DECILE_LABELS if d in nested]
    if not deciles:
        deciles = sorted(list(nested.keys()))
    default_decile = deciles[min(4, len(deciles) - 1)]

    default_from_values = sorted(
        [int(k) for k in nested[default_decile].keys()],
        reverse=True,
    )
    default_from = default_from_values[0]
    default_to_options = sorted(
        {int(k) for k in nested[default_decile][str(default_from)].keys()},
        reverse=True,
    )
    default_to = default_to_options[0]

    default_stats = nested[default_decile][str(default_from)][str(default_to)]
    default_fig = go.Figure(
        data=[
            go.Bar(
                x=bin_centers,
                y=default_stats["counts"],
                marker_color="rgba(55, 126, 184, 0.75)",
                hovertemplate="Price change: %{x:.3f}<br>Count: %{y}<extra></extra>",
            )
        ]
    )
    default_fig.update_layout(
        title=(
            f"Price-change distribution: {default_decile}, "
            f"{default_from}m -> {default_to}m"
        ),
        xaxis_title="Price change (price_to - price_from)",
        yaxis_title="Market count",
        bargap=0.02,
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        autosize=False,
    )
    default_fig.add_vline(x=0.0, line_dash="dash", line_color="rgba(0,0,0,0.4)", line_width=1)
    try:
        default_fig.write_image(
            output_png,
            width=FIGURE_WIDTH,
            height=FIGURE_HEIGHT,
            scale=1,
        )
    except Exception:
        pass

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Price-Change Distribution Explorer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; }}
    .controls {{ display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }}
    .statline {{ margin: 8px 0 14px 0; font-size: 14px; }}
    label {{ font-size: 14px; }}
    select {{ padding: 4px 6px; min-width: 190px; }}
  </style>
</head>
<body>
  <h2>Price-Change Distribution Explorer (Global Volume Deciles)</h2>
  <div class="controls">
    <label>Global volume decile:
      <select id="decileSelect"></select>
    </label>
    <label>From horizon (minutes):
      <select id="fromSelect"></select>
    </label>
    <label>To horizon (minutes):
      <select id="toSelect"></select>
    </label>
  </div>
  <div id="statline" class="statline"></div>
  <div id="histogram" style="width: 100%; max-width: 1600px; height: 1000px;"></div>

  <script>
    const dataMap = {json.dumps(nested)};
    const binCenters = {json.dumps(bin_centers.tolist())};
    const tailThreshold = {float(tail_threshold)};
    const decileOrder = {json.dumps(deciles)};

    const decileSelect = document.getElementById('decileSelect');
    const fromSelect = document.getElementById('fromSelect');
    const toSelect = document.getElementById('toSelect');
    const statline = document.getElementById('statline');

    function setOptions(selectEl, values, selectedValue) {{
      selectEl.innerHTML = '';
      values.forEach(v => {{
        const opt = document.createElement('option');
        opt.value = String(v);
        opt.textContent = String(v);
        if (String(v) === String(selectedValue)) opt.selected = true;
        selectEl.appendChild(opt);
      }});
    }}

    function sortedNumericKeys(obj, desc=true) {{
      return Object.keys(obj).map(v => Number(v)).sort((a, b) => desc ? b - a : a - b);
    }}

    function refreshFromOptions() {{
      const decile = decileSelect.value;
      const fromVals = sortedNumericKeys(dataMap[decile], true);
      const current = Number(fromSelect.value);
      const next = fromVals.includes(current) ? current : fromVals[0];
      setOptions(fromSelect, fromVals, next);
      refreshToOptions();
    }}

    function refreshToOptions() {{
      const decile = decileSelect.value;
      const fromH = fromSelect.value;
      const toVals = sortedNumericKeys(dataMap[decile][fromH], true);
      const current = Number(toSelect.value);
      const next = toVals.includes(current) ? current : toVals[0];
      setOptions(toSelect, toVals, next);
      render();
    }}

    function render() {{
      const decile = decileSelect.value;
      const fromH = fromSelect.value;
      const toH = toSelect.value;
      const stats = dataMap?.[decile]?.[fromH]?.[toH];
      if (!stats) return;

      const trace = {{
        type: 'bar',
        x: binCenters,
        y: stats.counts,
        marker: {{ color: 'rgba(55, 126, 184, 0.75)' }},
        hovertemplate: 'Price change: %{{x:.3f}}<br>Count: %{{y}}<extra></extra>',
      }};
      const layout = {{
        title: `Price-change distribution: ${{decile}}, ${{fromH}}m -> ${{toH}}m`,
        xaxis: {{ title: 'Price change (price_to - price_from)' }},
        yaxis: {{ title: 'Market count' }},
        bargap: 0.02,
        shapes: [{{
          type: 'line',
          x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper',
          line: {{ dash: 'dash', color: 'rgba(0,0,0,0.4)', width: 1 }}
        }}],
      }};
      Plotly.react('histogram', [trace], layout, {{ responsive: true }});

      statline.textContent =
        `n=${{stats.n}} | mean=${{stats.mean.toFixed(4)}} | median=${{stats.median.toFixed(4)}} | ` +
        `q10=${{stats.q10.toFixed(4)}} | q25=${{stats.q25.toFixed(4)}} | ` +
        `q75=${{stats.q75.toFixed(4)}} | q90=${{stats.q90.toFixed(4)}} | ` +
        `share(|delta|>=${{tailThreshold.toFixed(2)}})=${{(100 * stats.share_tail).toFixed(2)}}%`;
    }}

    setOptions(decileSelect, decileOrder, decileOrder[Math.min(4, decileOrder.length - 1)]);
    refreshFromOptions();

    decileSelect.addEventListener('change', refreshFromOptions);
    fromSelect.addEventListener('change', refreshToOptions);
    toSelect.addEventListener('change', render);
  </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")
    return output_html


def save_mae_change_distribution_explorer(
    mae_change_df: pd.DataFrame,
    plots_dir: Path,
    tail_threshold: float = 0.1,
    bins: int = 80,
) -> Path:
    """Standalone interactive histogram explorer for horizon-pair MAE changes."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "mae_change_distribution_explorer.html"
    output_png = plots_dir / "mae_change_distribution_explorer.png"

    required = {
        "global_volume_bucket",
        "from_horizon_min",
        "to_horizon_min",
        "mae_change",
    }
    if mae_change_df.empty or not required.issubset(set(mae_change_df.columns)):
        fig = px.histogram(title="No MAE-change distribution data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    df = mae_change_df.dropna(
        subset=["global_volume_bucket", "from_horizon_min", "to_horizon_min", "mae_change"]
    ).copy()
    if df.empty:
        fig = px.histogram(title="No valid MAE-change rows available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    df["global_volume_bucket"] = pd.Categorical(
        df["global_volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    df = df.sort_values(["global_volume_bucket", "from_horizon_min", "to_horizon_min"])

    bin_edges = np.linspace(-1.0, 1.0, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    nested: dict[str, dict[str, dict[str, dict[str, float | int | list[int] | list[float]]]]] = {}
    for (bucket, from_h, to_h), g in df.groupby(
        ["global_volume_bucket", "from_horizon_min", "to_horizon_min"], observed=True
    ):
        if pd.isna(bucket):
            continue
        values = g["mae_change"].astype(float).to_numpy()
        if values.size == 0:
            continue
        counts, _ = np.histogram(values, bins=bin_edges)
        stats = {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "q10": float(np.quantile(values, 0.10)),
            "q25": float(np.quantile(values, 0.25)),
            "q75": float(np.quantile(values, 0.75)),
            "q90": float(np.quantile(values, 0.90)),
            "share_tail": float(np.mean(np.abs(values) >= float(tail_threshold))),
            "counts": counts.tolist(),
        }
        bucket_key = str(bucket)
        from_key = str(int(from_h))
        to_key = str(int(to_h))
        nested.setdefault(bucket_key, {}).setdefault(from_key, {})[to_key] = stats

    if not nested:
        fig = px.histogram(title="No grouped MAE-change data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    deciles = [d for d in _VOLUME_DECILE_LABELS if d in nested]
    if not deciles:
        deciles = sorted(list(nested.keys()))
    default_decile = deciles[min(4, len(deciles) - 1)]

    default_from_values = sorted(
        [int(k) for k in nested[default_decile].keys()],
        reverse=True,
    )
    default_from = default_from_values[0]
    default_to_options = sorted(
        {int(k) for k in nested[default_decile][str(default_from)].keys()},
        reverse=True,
    )
    default_to = default_to_options[0]

    default_stats = nested[default_decile][str(default_from)][str(default_to)]
    default_fig = go.Figure(
        data=[
            go.Bar(
                x=bin_centers,
                y=default_stats["counts"],
                marker_color="rgba(55, 126, 184, 0.75)",
                hovertemplate="MAE change: %{x:.3f}<br>Count: %{y}<extra></extra>",
            )
        ]
    )
    default_fig.update_layout(
        title=(
            f"MAE-change distribution: {default_decile}, "
            f"{default_from}m -> {default_to}m"
        ),
        xaxis_title="MAE change (abs_error_to - abs_error_from)",
        yaxis_title="Market count",
        bargap=0.02,
        width=FIGURE_WIDTH,
        height=FIGURE_HEIGHT,
        autosize=False,
    )
    default_fig.add_vline(x=0.0, line_dash="dash", line_color="rgba(0,0,0,0.4)", line_width=1)
    try:
        default_fig.write_image(
            output_png,
            width=FIGURE_WIDTH,
            height=FIGURE_HEIGHT,
            scale=1,
        )
    except Exception:
        pass

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>MAE-Change Distribution Explorer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; }}
    .controls {{ display: flex; gap: 16px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }}
    .statline {{ margin: 8px 0 14px 0; font-size: 14px; }}
    label {{ font-size: 14px; }}
    select {{ padding: 4px 6px; min-width: 190px; }}
  </style>
</head>
<body>
  <h2>MAE-Change Distribution Explorer (Global Volume Deciles)</h2>
  <div class="controls">
    <label>Global volume decile:
      <select id="decileSelect"></select>
    </label>
    <label>From horizon (minutes):
      <select id="fromSelect"></select>
    </label>
    <label>To horizon (minutes):
      <select id="toSelect"></select>
    </label>
  </div>
  <div id="statline" class="statline"></div>
  <div id="histogram" style="width: 100%; max-width: 1600px; height: 1000px;"></div>

  <script>
    const dataMap = {json.dumps(nested)};
    const binCenters = {json.dumps(bin_centers.tolist())};
    const tailThreshold = {float(tail_threshold)};
    const decileOrder = {json.dumps(deciles)};

    const decileSelect = document.getElementById('decileSelect');
    const fromSelect = document.getElementById('fromSelect');
    const toSelect = document.getElementById('toSelect');
    const statline = document.getElementById('statline');

    function setOptions(selectEl, values, selectedValue) {{
      selectEl.innerHTML = '';
      values.forEach(v => {{
        const opt = document.createElement('option');
        opt.value = String(v);
        opt.textContent = String(v);
        if (String(v) === String(selectedValue)) opt.selected = true;
        selectEl.appendChild(opt);
      }});
    }}

    function sortedNumericKeys(obj, desc=true) {{
      return Object.keys(obj).map(v => Number(v)).sort((a, b) => desc ? b - a : a - b);
    }}

    function refreshFromOptions() {{
      const decile = decileSelect.value;
      const fromVals = sortedNumericKeys(dataMap[decile], true);
      const current = Number(fromSelect.value);
      const next = fromVals.includes(current) ? current : fromVals[0];
      setOptions(fromSelect, fromVals, next);
      refreshToOptions();
    }}

    function refreshToOptions() {{
      const decile = decileSelect.value;
      const fromH = fromSelect.value;
      const toVals = sortedNumericKeys(dataMap[decile][fromH], true);
      const current = Number(toSelect.value);
      const next = toVals.includes(current) ? current : toVals[0];
      setOptions(toSelect, toVals, next);
      render();
    }}

    function render() {{
      const decile = decileSelect.value;
      const fromH = fromSelect.value;
      const toH = toSelect.value;
      const stats = dataMap?.[decile]?.[fromH]?.[toH];
      if (!stats) return;

      const trace = {{
        type: 'bar',
        x: binCenters,
        y: stats.counts,
        marker: {{ color: 'rgba(55, 126, 184, 0.75)' }},
        hovertemplate: 'MAE change: %{{x:.3f}}<br>Count: %{{y}}<extra></extra>',
      }};
      const layout = {{
        title: `MAE-change distribution: ${{decile}}, ${{fromH}}m -> ${{toH}}m`,
        xaxis: {{ title: 'MAE change (abs_error_to - abs_error_from)' }},
        yaxis: {{ title: 'Market count' }},
        bargap: 0.02,
        shapes: [{{
          type: 'line',
          x0: 0, x1: 0, y0: 0, y1: 1, yref: 'paper',
          line: {{ dash: 'dash', color: 'rgba(0,0,0,0.4)', width: 1 }}
        }}],
      }};
      Plotly.react('histogram', [trace], layout, {{ responsive: true }});

      statline.textContent =
        `n=${{stats.n}} | mean=${{stats.mean.toFixed(4)}} | median=${{stats.median.toFixed(4)}} | ` +
        `q10=${{stats.q10.toFixed(4)}} | q25=${{stats.q25.toFixed(4)}} | ` +
        `q75=${{stats.q75.toFixed(4)}} | q90=${{stats.q90.toFixed(4)}} | ` +
        `share(|delta|>=${{tailThreshold.toFixed(2)}})=${{(100 * stats.share_tail).toFixed(2)}}%`;
    }}

    setOptions(decileSelect, decileOrder, decileOrder[Math.min(4, decileOrder.length - 1)]);
    refreshFromOptions();

    decileSelect.addEventListener('change', refreshFromOptions);
    fromSelect.addEventListener('change', refreshToOptions);
    toSelect.addEventListener('change', render);
  </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")
    return output_html


def save_crypto_price_slice_explorer(
    tidy_df: pd.DataFrame,
    plots_dir: Path,
    *,
    default_price_low: float = 0.49,
    default_price_high: float = 0.51,
    default_num_bins: int = 10,
) -> Path:
    """Interactive histogram of last-trade prices in a zoomable range for Crypto markets by horizon.

    The page embeds raw price/outcome pairs per horizon; bin edges are chosen in the browser via
    either a fixed bin count or a fixed bin width inside [price_low, price_high].
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "crypto_price_slice_explorer.html"
    output_png = plots_dir / "crypto_price_slice_explorer.png"

    required = {"category", "horizon_min", "price_last_trade", "outcome_yes"}
    if tidy_df.empty or not required.issubset(set(tidy_df.columns)):
        fig = px.bar(title="No snapshot data available for crypto price slice explorer")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    df = tidy_df.dropna(subset=["horizon_min", "price_last_trade", "outcome_yes"]).copy()
    if df.empty:
        fig = px.bar(title="No valid price/outcome rows for crypto price slice explorer")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    df["category"] = _normalized_category_labels(df.get("category", pd.Series(index=df.index, dtype=object)))
    df = df[df["category"].astype(str) == "Crypto"].copy()
    if df.empty:
        fig = px.bar(title="No Crypto category rows in snapshot data")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    df["horizon_min"] = df["horizon_min"].astype(int)
    df["price_last_trade"] = df["price_last_trade"].astype(float).clip(0.0, 1.0)
    df["outcome_yes"] = df["outcome_yes"].astype(float)

    data_by_horizon: dict[str, dict[str, list[float]]] = {}
    horizons_sorted: list[int] = []
    for h, g in df.groupby("horizon_min", sort=True):
        hi = int(h)
        horizons_sorted.append(hi)
        data_by_horizon[str(hi)] = {
            "p": g["price_last_trade"].tolist(),
            "y": g["outcome_yes"].tolist(),
        }

    if not data_by_horizon:
        fig = px.bar(title="No per-horizon crypto data to embed")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    default_h = min(horizons_sorted)
    payload = json.dumps(data_by_horizon, separators=(",", ":"))
    horizons_json = json.dumps([str(h) for h in horizons_sorted])

    # Default PNG: histogram + observed YES rate by bin for the default slice.
    def _slice_histogram(
        prices: np.ndarray,
        outcomes: np.ndarray,
        lo: float,
        hi: float,
        n_bins: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        m = (prices >= lo) & (prices <= hi)
        pv, ov = prices[m], outcomes[m]
        if pv.size == 0:
            return np.array([]), np.array([]), np.array([]), np.array([])
        edges = np.linspace(lo, hi, int(n_bins) + 1)
        centers = (edges[:-1] + edges[1:]) / 2.0
        counts = np.zeros(n_bins, dtype=int)
        sums = np.zeros(n_bins, dtype=float)
        idx = np.minimum(
            np.searchsorted(edges, pv, side="right") - 1,
            n_bins - 1,
        )
        idx = np.clip(idx, 0, n_bins - 1)
        for i in range(n_bins):
            sel = idx == i
            counts[i] = int(np.sum(sel))
            sums[i] = float(np.sum(ov[sel])) if np.any(sel) else 0.0
        rates = np.divide(
            sums,
            np.maximum(counts, 1),
            out=np.zeros_like(sums, dtype=float),
            where=counts > 0,
        )
        return centers, counts.astype(float), rates, edges

    d0 = data_by_horizon[str(default_h)]
    p0 = np.asarray(d0["p"], dtype=float)
    y0 = np.asarray(d0["y"], dtype=float)
    xc, cnt, rate, _edges = _slice_histogram(
        p0, y0, default_price_low, default_price_high, default_num_bins
    )
    if xc.size > 0:
        fig_png = go.Figure(
            data=[
                go.Bar(
                    x=xc,
                    y=cnt,
                    name="Count",
                    marker_color="rgba(55, 126, 184, 0.7)",
                    yaxis="y",
                    hovertemplate="Bin center: %{x:.4f}<br>Count: %{y}<extra></extra>",
                ),
                go.Scatter(
                    x=xc,
                    y=rate,
                    name="Observed YES rate",
                    mode="lines+markers",
                    yaxis="y2",
                    line={"color": "rgba(214, 39, 40, 0.9)", "width": 2},
                    marker={"size": 8},
                    hovertemplate="Bin center: %{x:.4f}<br>YES rate: %{y:.4f}<extra></extra>",
                ),
            ]
        )
        fig_png.add_trace(
            go.Scatter(
                x=[default_price_low, default_price_high],
                y=[default_price_low, default_price_high],
                name="Perfect calibration",
                mode="lines",
                yaxis="y2",
                line={"dash": "dash", "color": "rgba(0,0,0,0.35)", "width": 1},
                hoverinfo="skip",
            )
        )
        fig_png.update_layout(
            title=(
                f"Crypto last-trade price slice (default): {default_h} min, "
                f"[{default_price_low}, {default_price_high}], {default_num_bins} bins"
            ),
            xaxis_title="Price (last trade)",
            yaxis=dict(title="Market count", side="left", showgrid=True),
            yaxis2=dict(
                title="Observed YES rate",
                side="right",
                overlaying="y",
                range=[0, 1],
                showgrid=False,
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            bargap=0.08,
            width=FIGURE_WIDTH,
            height=FIGURE_HEIGHT,
            autosize=False,
        )
        try:
            fig_png.write_image(
                output_png,
                width=FIGURE_WIDTH,
                height=FIGURE_HEIGHT,
                scale=1,
            )
        except Exception:
            pass

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Crypto price slice explorer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 16px; }}
    .controls {{ display: flex; gap: 14px; align-items: flex-end; flex-wrap: wrap; margin-bottom: 10px; }}
    .controls label {{ font-size: 14px; display: flex; flex-direction: column; gap: 4px; }}
    .controls input[type="number"], .controls select {{ padding: 4px 6px; min-width: 100px; }}
    .mode {{ flex-direction: row; align-items: center; gap: 10px; flex-wrap: wrap; }}
    .mode label {{ flex-direction: row; align-items: center; gap: 6px; }}
    .statline {{ margin: 6px 0 12px 0; font-size: 14px; max-width: 1600px; }}
    #plot {{ width: 100%; max-width: 1600px; height: 1000px; }}
  </style>
</head>
<body>
  <h2>Crypto markets: price slice explorer (by time-to-close horizon)</h2>
  <p style="max-width:960px;font-size:14px;">
    Crypto rows only. Choose a horizon, a price window (e.g. 0.49–0.51), and either a number of equal-width
    bins inside that window or a fixed bin width. The bars are counts per bin; the red trace is the observed
    YES rate (secondary axis). The dashed line is y = x for calibration reference on that axis.
  </p>
  <div class="controls">
    <label>Horizon (minutes)
      <select id="horizonSelect"></select>
    </label>
    <label>Price low
      <input type="number" id="priceLow" step="0.0001" min="0" max="1" value="{default_price_low}" />
    </label>
    <label>Price high
      <input type="number" id="priceHigh" step="0.0001" min="0" max="1" value="{default_price_high}" />
    </label>
    <div class="mode">
      <span style="font-size:14px;">Bins:</span>
      <label><input type="radio" name="binMode" id="modeCount" value="count" checked /> Fixed count</label>
      <label><input type="radio" name="binMode" id="modeWidth" value="width" /> Fixed width</label>
    </div>
    <label>Number of bins
      <input type="number" id="numBins" step="1" min="2" max="500" value="{default_num_bins}" />
    </label>
    <label>Bin width
      <input type="number" id="binWidth" step="0.0001" min="1e-6" value="0.002" />
    </label>
    <button type="button" id="applyBtn" style="padding:8px 14px;">Update plot</button>
  </div>
  <div id="statline" class="statline"></div>
  <div id="plot"></div>

  <script>
    const dataByHorizon = {payload};
    const horizonOrder = {horizons_json};

    const horizonSelect = document.getElementById('horizonSelect');
    const priceLowEl = document.getElementById('priceLow');
    const priceHighEl = document.getElementById('priceHigh');
    const modeCountEl = document.getElementById('modeCount');
    const numBinsEl = document.getElementById('numBins');
    const binWidthEl = document.getElementById('binWidth');
    const applyBtn = document.getElementById('applyBtn');
    const statline = document.getElementById('statline');

    function setHorizonOptions() {{
      horizonSelect.innerHTML = '';
      horizonOrder.forEach(h => {{
        const opt = document.createElement('option');
        opt.value = h;
        opt.textContent = h + ' min';
        horizonSelect.appendChild(opt);
      }});
    }}

    function buildEdgesFromCount(lo, hi, n) {{
      const edges = [];
      for (let i = 0; i <= n; i++) {{
        edges.push(lo + (i / n) * (hi - lo));
      }}
      return edges;
    }}

    function buildEdgesFromWidth(lo, hi, w) {{
      const edges = [lo];
      const eps = 1e-12;
      while (edges[edges.length - 1] < hi - eps) {{
        const next = edges[edges.length - 1] + w;
        if (next >= hi) {{
          edges.push(hi);
          break;
        }}
        edges.push(next);
      }}
      if (edges[edges.length - 1] < hi - eps) edges.push(hi);
      return edges;
    }}

    function upperBound(arr, x) {{
      let lo = 0, hi = arr.length;
      while (lo < hi) {{
        const mid = (lo + hi) >> 1;
        if (arr[mid] <= x) lo = mid + 1;
        else hi = mid;
      }}
      return lo;
    }}

    function histogramInSlice(pArr, yArr, lo, hi, edges) {{
      const nBin = edges.length - 1;
      const counts = new Array(nBin).fill(0);
      const sumY = new Array(nBin).fill(0);
      let nIn = 0;
      let sumP = 0;
      let sumAllY = 0;
      for (let i = 0; i < pArr.length; i++) {{
        const p = pArr[i];
        const y = yArr[i];
        if (p < lo || p > hi) continue;
        nIn++;
        sumP += p;
        sumAllY += y;
        let b = upperBound(edges, p) - 1;
        if (b < 0) b = 0;
        if (b >= nBin) b = nBin - 1;
        counts[b]++;
        sumY[b] += y;
      }}
      const centers = [];
      const rates = [];
      for (let bj = 0; bj < nBin; bj++) {{
        centers.push((edges[bj] + edges[bj + 1]) / 2);
        rates.push(counts[bj] > 0 ? sumY[bj] / counts[bj] : null);
      }}
      return {{
        nIn, nTotal: pArr.length, sumP, sumAllY, counts, centers, rates, edges, nBin
      }};
    }}

    function render() {{
      const h = horizonSelect.value;
      const pack = dataByHorizon[h];
      if (!pack) {{
        statline.textContent = 'No data for this horizon.';
        return;
      }}
      const lo = parseFloat(priceLowEl.value);
      const hi = parseFloat(priceHighEl.value);
      if (!(lo < hi) || lo < 0 || hi > 1) {{
        statline.textContent = 'Need 0 ≤ price low < price high ≤ 1.';
        Plotly.purge('plot');
        return;
      }}

      let edges;
      let modeLabel;
      if (modeCountEl.checked) {{
        let n = parseInt(numBinsEl.value, 10);
        if (!Number.isFinite(n)) n = 10;
        n = Math.max(2, Math.min(500, n));
        edges = buildEdgesFromCount(lo, hi, n);
        modeLabel = `${{n}} equal-width bins`;
      }} else {{
        let w = parseFloat(binWidthEl.value);
        if (!(w > 0)) {{
          statline.textContent = 'Bin width must be positive.';
          Plotly.purge('plot');
          return;
        }}
        if (w > hi - lo) {{
          statline.textContent = 'Bin width should not exceed (price high − price low).';
          Plotly.purge('plot');
          return;
        }}
        edges = buildEdgesFromWidth(lo, hi, w);
        modeLabel = `${{edges.length - 1}} bins of width ${{w}}`;
      }}

      const {{ nIn, nTotal, sumP, sumAllY, counts, centers, rates }} = histogramInSlice(
        pack.p, pack.y, lo, hi, edges
      );

      const rateNumeric = rates.map(v => (v === null ? null : v));

      const traceBar = {{
        type: 'bar',
        x: centers,
        y: counts,
        name: 'Count',
        marker: {{ color: 'rgba(55, 126, 184, 0.7)' }},
        yaxis: 'y',
        hovertemplate: 'Bin center: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>',
      }};
      const traceRate = {{
        type: 'scatter',
        x: centers,
        y: rateNumeric,
        name: 'Observed YES rate',
        mode: 'lines+markers',
        yaxis: 'y2',
        connectgaps: false,
        line: {{ color: 'rgba(214, 39, 40, 0.9)', width: 2 }},
        marker: {{ size: 8 }},
        hovertemplate: 'Bin center: %{{x:.4f}}<br>YES rate: %{{y:.4f}}<extra></extra>',
      }};
      const traceDiag = {{
        type: 'scatter',
        x: [lo, hi],
        y: [lo, hi],
        name: 'Perfect calibration',
        mode: 'lines',
        yaxis: 'y2',
        line: {{ dash: 'dash', color: 'rgba(0,0,0,0.35)', width: 1 }},
        hoverinfo: 'skip',
      }};

      const layout = {{
        title: `Crypto: horizon ${{h}} min | window [${{lo.toFixed(4)}}, ${{hi.toFixed(4)}}] | ${{modeLabel}}`,
        xaxis: {{ title: 'Price (last trade)' }},
        yaxis: {{ title: 'Market count', side: 'left', showgrid: true }},
        yaxis2: {{
          title: 'Observed YES rate',
          side: 'right',
          overlaying: 'y',
          range: [0, 1],
          showgrid: false,
        }},
        legend: {{ orientation: 'h', yanchor: 'bottom', y: 1.02, xanchor: 'right', x: 1 }},
        bargap: 0.08,
        width: {FIGURE_WIDTH},
        height: {FIGURE_HEIGHT},
      }};

      Plotly.react('plot', [traceBar, traceRate, traceDiag], layout, {{ responsive: true }});

      const meanP = nIn > 0 ? sumP / nIn : NaN;
      const meanY = nIn > 0 ? sumAllY / nIn : NaN;
      statline.textContent =
        `n in window=${{nIn}} (of ${{nTotal}} at this horizon) | mean price=${{meanP.toFixed(4)}} | ` +
        `mean outcome=${{meanY.toFixed(4)}} | bins=${{counts.length}}`;
    }}

    setHorizonOptions();
    horizonSelect.addEventListener('change', render);
    applyBtn.addEventListener('click', render);
    modeCountEl.addEventListener('change', render);
    document.getElementById('modeWidth').addEventListener('change', render);

    let debounce = null;
    function debouncedRender() {{
      if (debounce) clearTimeout(debounce);
      debounce = setTimeout(render, 280);
    }}
    [priceLowEl, priceHighEl, numBinsEl, binWidthEl].forEach(el => {{
      el.addEventListener('input', debouncedRender);
    }});

    render();
  </script>
</body>
</html>
"""
    output_html.write_text(html, encoding="utf-8")
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
        _write_figure_outputs(fig, output_html, None)
        return output_html

    df = tidy_df.dropna(subset=["price_last_trade"]).copy()
    if df.empty:
        fig = px.line(title="No price data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    df["signed_error"] = df["price_last_trade"] - df["outcome_yes"]

    starts = _overlapping_price_bin_starts(bin_width, stride)
    rows: list[dict] = []
    horizon_order: list[str] = []
    for horizon in sorted(df["horizon_min"].dropna().unique()):
        h = int(horizon)
        horizon_order.append(f"{h} min")
        d_h = df[df["horizon_min"] == horizon].copy()
        d_h["volume_bucket"] = _volume_deciles_within_horizon(d_h["volume_total_market"])
        decile_bounds = (
            d_h.groupby("volume_bucket", observed=True)["volume_total_market"]
            .agg(decile_volume_min="min", decile_volume_max="max")
        )
        decile_bounds_map = {
            vb: (float(r["decile_volume_min"]), float(r["decile_volume_max"]))
            for vb, r in decile_bounds.iterrows()
        }
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
            for volume_bucket, sub in in_bin.groupby("volume_bucket", observed=True):
                if sub.empty:
                    continue
                dv_min, dv_max = decile_bounds_map[volume_bucket]
                rows.append(
                    {
                        "horizon_min": h,
                        "horizon_label": f"{h} min",
                        "volume_bucket": volume_bucket,
                        "bin_start": start,
                        "bin_end": end,
                        "bin_mid": (start + end) / 2.0,
                        "mean_signed_error": float(sub["signed_error"].mean()),
                        "n": int(len(sub)),
                        "decile_volume_min": dv_min,
                        "decile_volume_max": dv_max,
                    }
                )

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        fig = px.line(title="No overlapping-bin signed-error points available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df["volume_bucket"] = pd.Categorical(
        plot_df["volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    plot_df = plot_df.sort_values(["horizon_min", "volume_bucket", "bin_mid"])

    fig = px.line(
        plot_df,
        x="bin_mid",
        y="mean_signed_error",
        color="volume_bucket",
        markers=True,
        facet_col="horizon_label",
        facet_col_wrap=4,
        category_orders={
            "volume_bucket": list(_VOLUME_DECILE_LABELS),
            "horizon_label": horizon_order,
        },
        hover_data={
            "horizon_min": True,
            "bin_start": True,
            "bin_end": True,
            "n": True,
            "decile_volume_min": ":,.0f",
            "decile_volume_max": ":,.0f",
        },
        title=(
            "Mean signed pricing error by overlapping price bins "
            f"(width={bin_width}, stride={stride}), by volume decile "
            "(deciles within each horizon)"
        ),
        labels={
            "bin_mid": "Price bin midpoint",
            "mean_signed_error": "Mean (YES price − outcome)",
            "volume_bucket": "Volume decile",
            "decile_volume_min": "Decile volume (min)",
            "decile_volume_max": "Decile volume (max)",
        },
    )
    fig.update_xaxes(range=[0, 1])
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(0,0,0,0.4)", line_width=1)
    _write_figure_outputs(fig, output_html, output_png)
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
        _write_figure_outputs(fig, output_html, None)
        return output_html

    df = tidy_df.dropna(subset=["price_last_trade"]).copy()
    if df.empty:
        fig = px.line(title="No price data available")
        _write_figure_outputs(fig, output_html, None)
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
        _write_figure_outputs(fig, output_html, None)
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
    _write_figure_outputs(fig, output_html, output_png)
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
        _write_figure_outputs(fig, output_html, None)
        return output_html

    df = tidy_df.dropna(subset=["price_last_trade"]).copy()
    if df.empty:
        fig = px.line(title="No price data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    df["signed_error"] = df["price_last_trade"] - df["outcome_yes"]

    starts = _overlapping_price_bin_starts(bin_width, stride)
    rows: list[dict] = []
    horizon_order: list[str] = []
    for horizon in sorted(df["horizon_min"].dropna().unique()):
        h = int(horizon)
        horizon_order.append(f"{h} min")
        d_h = df[df["horizon_min"] == horizon].copy()
        d_h["volume_bucket"] = _volume_deciles_within_horizon(d_h["volume_total_market"])
        for volume_bucket in _VOLUME_DECILE_LABELS:
            d_v = d_h[d_h["volume_bucket"] == volume_bucket]
            if d_v.empty:
                continue
            decile_volume_min = float(d_v["volume_total_market"].min())
            decile_volume_max = float(d_v["volume_total_market"].max())
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
                        "horizon_min": h,
                        "horizon_label": f"{h} min",
                        "volume_bucket": volume_bucket,
                        "bin_start": start,
                        "bin_end": end,
                        "bin_mid": (start + end) / 2.0,
                        "mean_signed_error": float(in_bin["signed_error"].mean()),
                        "n": int(len(in_bin)),
                        "decile_volume_min": decile_volume_min,
                        "decile_volume_max": decile_volume_max,
                    }
                )

    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        fig = px.line(title="No volume-decile / price-bin signed-error points available")
        _write_figure_outputs(fig, output_html, None)
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
    plot_df = plot_df.sort_values(["horizon_min", "bin_label", "volume_bucket"])

    fig = px.line(
        plot_df,
        x="volume_bucket",
        y="mean_signed_error",
        color="bin_label",
        markers=True,
        facet_col="horizon_label",
        facet_col_wrap=4,
        category_orders={
            "volume_bucket": list(_VOLUME_DECILE_LABELS),
            "bin_label": bin_label_order,
            "horizon_label": horizon_order,
        },
        hover_data={
            "horizon_min": True,
            "bin_mid": True,
            "bin_start": True,
            "bin_end": True,
            "n": True,
            "decile_volume_min": ":,.0f",
            "decile_volume_max": ":,.0f",
        },
        title=(
            "Mean signed pricing error vs volume decile by price bin midpoint "
            f"(width={bin_width}, stride={stride}); deciles within each horizon"
        ),
        labels={
            "volume_bucket": "Volume decile",
            "mean_signed_error": "Mean (YES price − outcome)",
            "bin_label": "Price bin midpoint",
            "decile_volume_min": "Decile volume (min)",
            "decile_volume_max": "Decile volume (max)",
        },
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(0,0,0,0.4)", line_width=1)
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_overlapping_bin_plot(
    tidy_df: pd.DataFrame,
    plots_dir: Path,
    bin_width: float = 0.02,
    stride: float = 0.01,
) -> Path:
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "overlapping_bin_outcome_rate.html"
    output_png = plots_dir / "overlapping_bin_outcome_rate.png"

    if tidy_df.empty:
        fig = px.line(title="No data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    df = tidy_df.dropna(subset=["price_last_trade"]).copy()
    if df.empty:
        fig = px.line(title="No price data available")
        _write_figure_outputs(fig, output_html, None)
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
        _write_figure_outputs(fig, output_html, None)
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
    _add_perfect_calibration_line(fig)
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_isotonic_gap_by_volume_decile_plot(
    isotonic_gap_volume_decile_df: pd.DataFrame, plots_dir: Path
) -> Path:
    """Mean absolute isotonic gap vs volume decile, one line per horizon."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "isotonic_gap_by_volume_decile.html"
    output_png = plots_dir / "isotonic_gap_by_volume_decile.png"

    if isotonic_gap_volume_decile_df.empty:
        fig = px.line(title="No isotonic gap by volume-decile data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = isotonic_gap_volume_decile_df.copy()
    plot_df["volume_bucket"] = pd.Categorical(
        plot_df["volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    plot_df = plot_df.sort_values(["horizon_min", "volume_bucket"])
    plot_df["horizon_label"] = plot_df["horizon_min"].map(lambda m: f"{int(m)} min")
    horizon_order = [f"{int(m)} min" for m in sorted(plot_df["horizon_min"].dropna().unique())]

    fig = px.line(
        plot_df,
        x="volume_bucket",
        y="mean_abs_isotonic_gap",
        color="horizon_label",
        markers=True,
        category_orders={
            "volume_bucket": list(_VOLUME_DECILE_LABELS),
            "horizon_label": horizon_order,
        },
        hover_data={
            "horizon_min": True,
            "n": True,
            "median_abs_isotonic_gap": ":.4f",
            "mean_signed_isotonic_gap": ":.4f",
            "share_positive_isotonic_gap": ":.3f",
            "decile_volume_min": ":,.0f",
            "decile_volume_max": ":,.0f",
        },
        title="Mean Absolute Isotonic Gap vs Volume Decile by Horizon",
        labels={
            "volume_bucket": "Volume decile",
            "mean_abs_isotonic_gap": "Mean absolute isotonic gap",
            "horizon_label": "Horizon (minutes before close)",
            "decile_volume_min": "Decile volume (min)",
            "decile_volume_max": "Decile volume (max)",
        },
    )
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_signed_isotonic_gap_by_volume_decile_price_bin_plot(
    isotonic_gap_volume_decile_price_bin_df: pd.DataFrame,
    plots_dir: Path,
) -> Path:
    """Mean signed isotonic gap vs volume decile, one line per price-bin midpoint."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "signed_isotonic_gap_by_volume_decile_price_bin.html"
    output_png = plots_dir / "signed_isotonic_gap_by_volume_decile_price_bin.png"

    if isotonic_gap_volume_decile_price_bin_df.empty:
        fig = px.line(title="No isotonic gap by volume-decile and price-bin data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = isotonic_gap_volume_decile_price_bin_df.copy()
    plot_df["volume_bucket"] = pd.Categorical(
        plot_df["volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    plot_df["horizon_label"] = plot_df["horizon_min"].map(lambda m: f"{int(m)} min")
    horizon_order = [f"{int(m)} min" for m in sorted(plot_df["horizon_min"].dropna().unique())]
    bin_mid_order = sorted(plot_df["bin_mid"].unique())
    bin_label_order = [f"{m:.3f}" for m in bin_mid_order]
    plot_df["bin_label"] = plot_df["bin_mid"].map(lambda m: f"{float(m):.3f}")
    plot_df["bin_label"] = pd.Categorical(
        plot_df["bin_label"],
        categories=bin_label_order,
        ordered=True,
    )
    plot_df = plot_df.sort_values(["horizon_min", "bin_label", "volume_bucket"])

    fig = px.line(
        plot_df,
        x="volume_bucket",
        y="mean_signed_isotonic_gap",
        color="bin_label",
        markers=True,
        facet_col="horizon_label",
        facet_col_wrap=4,
        category_orders={
            "volume_bucket": list(_VOLUME_DECILE_LABELS),
            "bin_label": bin_label_order,
            "horizon_label": horizon_order,
        },
        hover_data={
            "horizon_min": True,
            "bin_mid": True,
            "bin_start": True,
            "bin_end": True,
            "n": True,
            "mean_abs_isotonic_gap": ":.4f",
            "decile_volume_min": ":,.0f",
            "decile_volume_max": ":,.0f",
        },
        title=(
            "Mean signed isotonic gap vs volume decile by price-bin midpoint "
            "(deciles within each horizon)"
        ),
        labels={
            "volume_bucket": "Volume decile",
            "mean_signed_isotonic_gap": "Mean signed isotonic gap (price - isotonic fair value)",
            "bin_label": "Price bin midpoint",
            "decile_volume_min": "Decile volume (min)",
            "decile_volume_max": "Decile volume (max)",
        },
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.add_hline(y=0.0, line_dash="dash", line_color="rgba(0,0,0,0.4)", line_width=1)
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_volatility_isotonic_gap_plot(volatility_bucket_df: pd.DataFrame, plots_dir: Path) -> Path:
    """Mean absolute isotonic gap vs volatility bucket, one line per volatility proxy."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "volatility_vs_isotonic_gap.html"
    output_png = plots_dir / "volatility_vs_isotonic_gap.png"

    if volatility_bucket_df.empty:
        fig = px.line(title="No volatility/isotonic-gap data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    required = {"volatility_bucket", "volatility_metric", "mean_abs_isotonic_gap"}
    if not required.issubset(set(volatility_bucket_df.columns)):
        fig = px.line(title="Volatility isotonic-gap columns are unavailable")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = volatility_bucket_df.dropna(subset=["volatility_bucket"]).copy()
    if plot_df.empty:
        fig = px.line(title="No volatility buckets available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    if "bucket_rank" in plot_df.columns:
        plot_df = plot_df.sort_values(["volatility_metric", "bucket_rank"])
        bucket_order = list(plot_df.sort_values("bucket_rank")["volatility_bucket"].astype(str).drop_duplicates())
    else:
        plot_df = plot_df.sort_values(["volatility_metric", "volatility_bucket"])
        bucket_order = list(plot_df["volatility_bucket"].astype(str).drop_duplicates())

    fig = px.line(
        plot_df,
        x="volatility_bucket",
        y="mean_abs_isotonic_gap",
        color="volatility_metric",
        markers=True,
        category_orders={"volatility_bucket": bucket_order},
        hover_data={
            "n_markets": True,
            "volatility_min": ":.4f",
            "volatility_max": ":.4f",
            "volatility_mean": ":.4f",
            "mean_abs_error_last_trade": ":.4f",
            "mean_brier_last_trade": ":.4f",
        },
        title="Mean Absolute Isotonic Gap vs Volatility Bucket",
        labels={
            "volatility_bucket": "Volatility bucket",
            "mean_abs_isotonic_gap": "Mean absolute isotonic gap",
            "volatility_metric": "Volatility proxy",
        },
    )
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


def save_volatility_realized_error_plot(volatility_bucket_df: pd.DataFrame, plots_dir: Path) -> Path:
    """Realized error targets vs volatility bucket for each volatility proxy."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    output_html = plots_dir / "volatility_vs_realized_error.html"
    output_png = plots_dir / "volatility_vs_realized_error.png"

    if volatility_bucket_df.empty:
        fig = px.line(title="No volatility/realized-error data available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    required = {
        "volatility_bucket",
        "volatility_metric",
        "mean_abs_error_last_trade",
        "mean_brier_last_trade",
    }
    if not required.issubset(set(volatility_bucket_df.columns)):
        fig = px.line(title="Volatility realized-error columns are unavailable")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    plot_df = volatility_bucket_df.dropna(subset=["volatility_bucket"]).copy()
    if plot_df.empty:
        fig = px.line(title="No volatility buckets available")
        _write_figure_outputs(fig, output_html, None)
        return output_html

    if "bucket_rank" in plot_df.columns:
        plot_df = plot_df.sort_values(["volatility_metric", "bucket_rank"])
        bucket_order = list(plot_df.sort_values("bucket_rank")["volatility_bucket"].astype(str).drop_duplicates())
    else:
        plot_df = plot_df.sort_values(["volatility_metric", "volatility_bucket"])
        bucket_order = list(plot_df["volatility_bucket"].astype(str).drop_duplicates())

    long_df = plot_df.melt(
        id_vars=["volatility_bucket", "volatility_metric", "n_markets", "volatility_min", "volatility_max"],
        value_vars=["mean_abs_error_last_trade", "mean_brier_last_trade"],
        var_name="target_metric",
        value_name="target_value",
    )
    metric_labels = {
        "mean_abs_error_last_trade": "Realized MAE (mean abs error)",
        "mean_brier_last_trade": "Realized Brier",
    }
    long_df["target_metric_label"] = long_df["target_metric"].map(metric_labels).fillna(long_df["target_metric"])

    fig = px.line(
        long_df,
        x="volatility_bucket",
        y="target_value",
        color="volatility_metric",
        markers=True,
        facet_col="target_metric_label",
        category_orders={
            "volatility_bucket": bucket_order,
            "target_metric_label": [metric_labels["mean_abs_error_last_trade"], metric_labels["mean_brier_last_trade"]],
        },
        hover_data={
            "n_markets": True,
            "volatility_min": ":.4f",
            "volatility_max": ":.4f",
        },
        title="Realized Error vs Volatility Bucket",
        labels={
            "volatility_bucket": "Volatility bucket",
            "target_value": "Value",
            "volatility_metric": "Volatility proxy",
            "target_metric_label": "Target metric",
        },
    )
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    _write_figure_outputs(fig, output_html, output_png)
    return output_html


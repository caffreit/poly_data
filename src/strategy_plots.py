from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

FIGURE_WIDTH = 1600
FIGURE_HEIGHT = 1000

# Sliding bins: centers every 0.025, each bin spans [center - 0.025, center + 0.025] (width 0.05).
_SLIDING_BIN_STEP = 0.025
_SLIDING_BIN_HALF_WIDTH = 0.025
_SLIDING_BIN_CENTERS = np.arange(
    _SLIDING_BIN_HALF_WIDTH, 1.0, _SLIDING_BIN_STEP, dtype=float
)


def _sliding_bin_aggregate_by_side(
    trades: pd.DataFrame,
    *,
    x_col: str,
    mean_pnl: bool,
) -> pd.DataFrame:
    """One row per (bin_center, side) with mean PnL or mean hit rate % and trade count."""
    required = {x_col, "pnl", "side"}
    if trades.empty or not required.issubset(trades.columns):
        return pd.DataFrame(columns=["bin_center", "side", "y", "n"])

    t = trades.dropna(subset=[x_col, "pnl", "side"]).copy()
    t[x_col] = pd.to_numeric(t[x_col], errors="coerce")
    t = t.dropna(subset=[x_col])
    t = t[t["side"].isin(["YES", "NO"])]
    if t.empty:
        return pd.DataFrame(columns=["bin_center", "side", "y", "n"])

    if mean_pnl:
        value = t["pnl"].astype(float)
    else:
        value = (t["pnl"].astype(float) > 0).astype(float) * 100.0

    rows: list[dict[str, object]] = []
    x = t[x_col].to_numpy(dtype=float)
    sides = t["side"].astype(str).to_numpy()
    v = value.to_numpy(dtype=float)

    for center in _SLIDING_BIN_CENTERS:
        lo = max(0.0, center - _SLIDING_BIN_HALF_WIDTH)
        hi = min(1.0, center + _SLIDING_BIN_HALF_WIDTH)
        in_bin = (x >= lo) & (x <= hi)
        for side in ("YES", "NO"):
            m = in_bin & (sides == side)
            n = int(m.sum())
            if n == 0:
                continue
            rows.append(
                {
                    "bin_center": float(center),
                    "side": side,
                    "y": float(np.mean(v[m])),
                    "n": n,
                }
            )

    return pd.DataFrame(rows)


def _save_sliding_bin_side_line_plot(
    trades: pd.DataFrame,
    out_dir: Path,
    *,
    x_col: str,
    x_title: str,
    y_title: str,
    title: str,
    stem: str,
    mean_pnl: bool,
) -> list[Path]:
    out: list[Path] = []
    agg = _sliding_bin_aggregate_by_side(trades, x_col=x_col, mean_pnl=mean_pnl)
    if agg.empty:
        return out

    fig = go.Figure()
    for side, label in (("YES", "Yes"), ("NO", "No")):
        sub = agg[agg["side"] == side].sort_values("bin_center")
        if sub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=sub["bin_center"],
                y=sub["y"],
                mode="lines+markers",
                name=label,
                customdata=np.stack([sub["n"].to_numpy()], axis=1),
                hovertemplate=(
                    f"{x_title}: %{{x:.3f}}<br>"
                    f"{y_title}: %{{y:.4f}}<br>"
                    "Trades in bin: %{customdata[0]:.0f}<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend_title="Side",
    )
    fig.update_xaxes(range=[0.0, 1.0], dtick=0.05)
    html = out_dir / f"{stem}.html"
    png = out_dir / f"{stem}.png"
    _write_figure_outputs(fig, html, png)
    out.append(html)
    return out


def _write_figure_outputs(fig, output_html: Path, output_png: Path | None = None) -> None:
    fig.update_layout(width=FIGURE_WIDTH, height=FIGURE_HEIGHT, autosize=False)
    fig.write_html(output_html, config={"responsive": False})
    if output_png is not None:
        try:
            fig.write_image(output_png, width=FIGURE_WIDTH, height=FIGURE_HEIGHT, scale=1)
        except Exception:
            pass


def _read_strategy_artifacts(strategy_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_path = strategy_dir / "sports_trade_log.csv"
    equity_path = strategy_dir / "sports_equity_curve.csv"
    if not trade_path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {trade_path}")
    if not equity_path.is_file():
        raise FileNotFoundError(f"Missing required artifact: {equity_path}")

    trades = pd.read_csv(trade_path)
    equity = pd.read_csv(equity_path)
    if "entry_time" in trades.columns:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    if "end_time" in trades.columns:
        trades["end_time"] = pd.to_datetime(trades["end_time"], utc=True, errors="coerce")
    if "resolution_time" in trades.columns:
        trades["resolution_time"] = pd.to_datetime(trades["resolution_time"], utc=True, errors="coerce")
    if "event_time" in equity.columns:
        equity["event_time"] = pd.to_datetime(equity["event_time"], utc=True, errors="coerce")
    return trades, equity


def save_strategy_plots(strategy_dir: Path, plots_dir: Path | None = None) -> list[Path]:
    trades, equity = _read_strategy_artifacts(strategy_dir)
    out_dir = plots_dir if plots_dir is not None else (strategy_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    saved.extend(_save_wallet_and_drawdown_plots(equity, out_dir))
    saved.extend(_save_pnl_timeline_plot(trades, out_dir))
    saved.extend(_save_trade_pnl_distribution_plot(trades, out_dir))
    saved.extend(_save_edge_vs_pnl_plot(trades, out_dir))
    saved.extend(_save_side_performance_plot(trades, out_dir))
    saved.extend(_save_sliding_bin_market_price_plots(trades, out_dir))
    saved.extend(_save_sliding_bin_isotonic_prob_plots(trades, out_dir))
    return saved


def _save_sliding_bin_market_price_plots(trades: pd.DataFrame, out_dir: Path) -> list[Path]:
    out: list[Path] = []
    if trades.empty or "market_price" not in trades.columns:
        return out
    out.extend(
        _save_sliding_bin_side_line_plot(
            trades,
            out_dir,
            x_col="market_price",
            x_title="Market price (YES)",
            y_title="Mean PnL (USD)",
            title="Mean PnL vs market price (0.05-wide bins, centers every 0.025)",
            stem="sports_mean_pnl_vs_market_price_sliding_bin",
            mean_pnl=True,
        )
    )
    out.extend(
        _save_sliding_bin_side_line_plot(
            trades,
            out_dir,
            x_col="market_price",
            x_title="Market price (YES)",
            y_title="Hit rate (%)",
            title="Hit rate vs market price (0.05-wide bins, centers every 0.025)",
            stem="sports_hit_rate_vs_market_price_sliding_bin",
            mean_pnl=False,
        )
    )
    return out


def _save_sliding_bin_isotonic_prob_plots(trades: pd.DataFrame, out_dir: Path) -> list[Path]:
    out: list[Path] = []
    if trades.empty or "model_prob" not in trades.columns:
        return out
    out.extend(
        _save_sliding_bin_side_line_plot(
            trades,
            out_dir,
            x_col="model_prob",
            x_title="Isotonic probability (calibrated P(YES))",
            y_title="Mean PnL (USD)",
            title="Mean PnL vs isotonic probability (0.05-wide bins, centers every 0.025)",
            stem="sports_mean_pnl_vs_isotonic_prob_sliding_bin",
            mean_pnl=True,
        )
    )
    out.extend(
        _save_sliding_bin_side_line_plot(
            trades,
            out_dir,
            x_col="model_prob",
            x_title="Isotonic probability (calibrated P(YES))",
            y_title="Hit rate (%)",
            title="Hit rate vs isotonic probability (0.05-wide bins, centers every 0.025)",
            stem="sports_hit_rate_vs_isotonic_prob_sliding_bin",
            mean_pnl=False,
        )
    )
    return out


def _save_wallet_and_drawdown_plots(equity: pd.DataFrame, out_dir: Path) -> list[Path]:
    out: list[Path] = []
    if equity.empty:
        return out
    eq = equity.dropna(subset=["event_time"]).sort_values("event_time").copy()
    if eq.empty:
        return out

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq["event_time"], y=eq["wallet_value"], mode="lines", name="Wallet value"))
    fig.add_trace(go.Scatter(x=eq["event_time"], y=eq["cash"], mode="lines", name="Cash", opacity=0.65))
    fig.add_trace(go.Scatter(x=eq["event_time"], y=eq["locked_stake"], mode="lines", name="Locked stake", opacity=0.65))
    fig.update_layout(
        title="Wallet Components Over Time",
        xaxis_title="Time",
        yaxis_title="USD",
        legend_title="Series",
    )
    wallet_html = out_dir / "sports_wallet_over_time.html"
    wallet_png = out_dir / "sports_wallet_over_time.png"
    _write_figure_outputs(fig, wallet_html, wallet_png)
    out.append(wallet_html)

    eq["running_peak"] = eq["wallet_value"].cummax()
    eq["drawdown"] = (eq["wallet_value"] / eq["running_peak"]) - 1.0
    dd_fig = px.area(eq, x="event_time", y="drawdown", title="Drawdown Over Time")
    dd_fig.update_layout(xaxis_title="Time", yaxis_title="Drawdown")
    dd_fig.update_yaxes(tickformat=".1%")
    dd_html = out_dir / "sports_drawdown_over_time.html"
    dd_png = out_dir / "sports_drawdown_over_time.png"
    _write_figure_outputs(dd_fig, dd_html, dd_png)
    out.append(dd_html)
    return out


def _save_pnl_timeline_plot(trades: pd.DataFrame, out_dir: Path) -> list[Path]:
    out: list[Path] = []
    if trades.empty:
        return out
    resolution_col = "resolution_time" if "resolution_time" in trades.columns else "end_time"
    t = trades.dropna(subset=[resolution_col]).copy()
    if t.empty:
        return out
    t["resolution_time_plot"] = (
        pd.to_datetime(t[resolution_col], utc=True, errors="coerce").dt.tz_convert("UTC").dt.tz_localize(None)
    )
    t = t.dropna(subset=["resolution_time_plot"])
    if t.empty:
        return out
    by_resolution = (
        t.groupby("resolution_time_plot", as_index=False)
        .agg(
            pnl_at_resolution=("pnl", "sum"),
            trades_resolved=("pnl", "count"),
        )
        .sort_values("resolution_time_plot")
    )
    by_resolution["cum_pnl"] = by_resolution["pnl_at_resolution"].cumsum()
    by_resolution["resolution_label"] = by_resolution["resolution_time_plot"].dt.strftime("%Y-%m-%d %H:%M UTC")
    x_values = np.array([ts.to_pydatetime() for ts in by_resolution["resolution_time_plot"]])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=by_resolution["cum_pnl"],
            mode="lines+markers",
            line={"shape": "hv"},
            marker={"size": 6},
            name="Cumulative PnL",
            customdata=np.stack(
                [by_resolution["pnl_at_resolution"], by_resolution["trades_resolved"]],
                axis=1,
            ),
            hovertemplate=(
                "Resolution: %{x|%Y-%m-%d %H:%M UTC}<br>"
                "Cumulative PnL: %{y:.2f} USD<br>"
                "PnL at timestamp: %{customdata[0]:.2f} USD<br>"
                "Trades resolved: %{customdata[1]:.0f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=f"Cumulative PnL Over Time ({len(by_resolution)} resolution timestamps)",
        xaxis_title="Resolution time (UTC)",
        yaxis_title="USD",
    )
    fig.update_xaxes(
        type="date",
        tickformat="%Y-%m-%d\n%H:%M",
        hoverformat="%Y-%m-%d %H:%M:%S",
        range=[x_values.min(), x_values.max()],
    )
    html = out_dir / "sports_cumulative_pnl_over_time.html"
    png = out_dir / "sports_cumulative_pnl_over_time.png"
    _write_figure_outputs(fig, html, png)
    out.append(html)
    return out


def _save_trade_pnl_distribution_plot(trades: pd.DataFrame, out_dir: Path) -> list[Path]:
    out: list[Path] = []
    if trades.empty:
        return out
    fig = px.histogram(
        trades,
        x="pnl",
        color="side" if "side" in trades.columns else None,
        nbins=60,
        barmode="overlay",
        title="Per-Trade PnL Distribution",
    )
    fig.update_layout(xaxis_title="PnL (USD)", yaxis_title="Trade count")
    html = out_dir / "sports_trade_pnl_distribution.html"
    png = out_dir / "sports_trade_pnl_distribution.png"
    _write_figure_outputs(fig, html, png)
    out.append(html)
    return out


def _save_edge_vs_pnl_plot(trades: pd.DataFrame, out_dir: Path) -> list[Path]:
    out: list[Path] = []
    required = {"edge", "pnl", "stake"}
    if trades.empty or not required.issubset(trades.columns):
        return out
    fig = px.scatter(
        trades,
        x="edge",
        y="pnl",
        color="side" if "side" in trades.columns else None,
        size="stake",
        title="Edge vs Realized PnL",
        hover_data=["market_id", "market_price", "model_prob"],
    )
    fig.update_layout(xaxis_title="Model edge", yaxis_title="PnL (USD)")
    html = out_dir / "sports_edge_vs_pnl.html"
    png = out_dir / "sports_edge_vs_pnl.png"
    _write_figure_outputs(fig, html, png)
    out.append(html)
    return out


def _save_side_performance_plot(trades: pd.DataFrame, out_dir: Path) -> list[Path]:
    out: list[Path] = []
    if trades.empty or "side" not in trades.columns:
        return out

    t = trades.copy()
    t["win"] = (t["pnl"] > 0).astype(float)
    grouped = (
        t.groupby("side", dropna=False)
        .agg(
            trades=("pnl", "count"),
            total_pnl=("pnl", "sum"),
            avg_pnl=("pnl", "mean"),
            hit_rate=("win", "mean"),
            avg_edge=("edge", "mean"),
            avg_roi=("roi_on_stake", "mean"),
        )
        .reset_index()
    )
    grouped["hit_rate_pct"] = grouped["hit_rate"] * 100.0
    grouped["avg_roi_pct"] = grouped["avg_roi"] * 100.0

    fig = go.Figure()
    fig.add_trace(go.Bar(x=grouped["side"], y=grouped["total_pnl"], name="Total PnL (USD)"))
    fig.add_trace(go.Bar(x=grouped["side"], y=grouped["hit_rate_pct"], name="Hit rate (%)", yaxis="y2"))
    fig.add_trace(go.Bar(x=grouped["side"], y=grouped["avg_roi_pct"], name="Avg ROI (%)", yaxis="y2"))
    fig.update_layout(
        title="Performance by Trade Side",
        xaxis_title="Side",
        yaxis=dict(title="Total PnL (USD)"),
        yaxis2=dict(title="Percent metrics", overlaying="y", side="right"),
        barmode="group",
    )
    html = out_dir / "sports_side_performance.html"
    png = out_dir / "sports_side_performance.png"
    _write_figure_outputs(fig, html, png)
    out.append(html)
    return out

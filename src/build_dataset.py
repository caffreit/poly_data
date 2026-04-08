from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from statsmodels.nonparametric.smoothers_lowess import lowess

from .collect_snapshots import SnapshotRow

_VOLUME_DECILE_LABELS = (
    ["Decile 1 (lowest)"]
    + [f"Decile {i}" for i in range(2, 10)]
    + ["Decile 10 (highest)"]
)
_PRICE_HORIZON_COL_PATTERN = re.compile(r"^price_(\d+)m$")


def build_tidy_dataframe(rows: List[SnapshotRow]) -> pd.DataFrame:
    frame = pd.DataFrame([row.__dict__ for row in rows])
    if frame.empty:
        return frame

    frame["end_time"] = pd.to_datetime(frame["end_time"], utc=True)
    frame["snapshot_time"] = pd.to_datetime(frame["snapshot_time"], utc=True)
    frame["price_available"] = frame["price_last_trade"].notna().astype(int)
    frame["abs_error_last_trade"] = (frame["price_last_trade"] - frame["outcome_yes"]).abs()
    frame["brier_last_trade"] = (frame["price_last_trade"] - frame["outcome_yes"]) ** 2
    return frame


def build_wide_dataframe(tidy_df: pd.DataFrame) -> pd.DataFrame:
    if tidy_df.empty:
        return tidy_df

    base_cols = [
        "market_id",
        "condition_id",
        "question",
        "category",
        "token_id_yes",
        "outcome_label_positive",
        "outcome_label_negative",
        "end_time",
        "volume_total_market",
        "outcome_yes",
        "is_binary_yes_no",
    ]
    base_df = tidy_df[base_cols].drop_duplicates(subset=["market_id"]).set_index("market_id")

    prices_wide = (
        tidy_df.pivot_table(
            index="market_id",
            columns="horizon_min",
            values="price_last_trade",
            aggfunc="first",
        )
        .add_prefix("price_")
        .add_suffix("m")
    )

    stale_wide = (
        tidy_df.pivot_table(
            index="market_id",
            columns="horizon_min",
            values="price_staleness_min",
            aggfunc="first",
        )
        .add_prefix("staleness_")
        .add_suffix("m")
    )

    merged = base_df.join(prices_wide, how="left").join(stale_wide, how="left").reset_index()

    horizon_price_cols = _sorted_price_horizon_columns(merged)
    if not horizon_price_cols:
        return merged

    price_cols = [col for _, col in horizon_price_cols]
    prices_matrix = merged[price_cols].apply(pd.to_numeric, errors="coerce")
    merged["horizon_count_used"] = prices_matrix.notna().sum(axis=1).astype(int)
    merged["volatility_std"] = prices_matrix.std(axis=1, ddof=0)
    merged["volatility_range"] = prices_matrix.max(axis=1) - prices_matrix.min(axis=1)

    for (h1, c1), (h2, c2) in zip(horizon_price_cols[:-1], horizon_price_cols[1:]):
        merged[f"delta_price_{h1}m_{h2}m"] = merged[c1] - merged[c2]

    return merged


def _sorted_price_horizon_columns(df: pd.DataFrame) -> list[tuple[int, str]]:
    cols: list[tuple[int, str]] = []
    for col in df.columns:
        match = _PRICE_HORIZON_COL_PATTERN.match(str(col))
        if match:
            cols.append((int(match.group(1)), str(col)))
    return sorted(cols, key=lambda t: t[0])


def _decile_labels(k: int) -> list[str]:
    if k <= 1:
        return ["Decile 1 (lowest)"]
    if k == 10:
        return list(_VOLUME_DECILE_LABELS)
    labels = ["Decile 1 (lowest)"]
    labels.extend(f"Decile {i}" for i in range(2, k))
    labels.append(f"Decile {k} (highest)")
    return labels


def _quantile_bucket_labels(values: pd.Series, max_buckets: int = 10) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=object, index=values.index)
    valid = values.dropna()
    if valid.empty:
        return pd.Series(index=values.index, dtype=object)

    ranked = valid.rank(method="first")
    k = min(max_buckets, len(valid))
    labels = _decile_labels(k)
    bucketed = pd.qcut(ranked, q=k, labels=labels)

    out = pd.Series(index=values.index, dtype=object)
    out.loc[valid.index] = bucketed.astype(str)
    return out


def _assign_volume_deciles_within_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of df with volume_bucket from quantiles of volume_total_market within each horizon."""
    out = df.copy()
    parts: list[pd.Series] = []
    for _, g in out.groupby("horizon_min", sort=False):
        parts.append(_volume_deciles_within_horizon(g["volume_total_market"]))
    out["volume_bucket"] = pd.concat(parts)
    return out


def _normalized_category_labels(series: pd.Series, null_label: str = "Uncategorized") -> pd.Series:
    """Normalize category labels while preserving source categories."""
    out = series.copy()
    out = out.astype(object).where(out.notna(), "")
    out = out.astype(str).str.strip()
    out = out.where(out != "", null_label)
    return out


def compute_staleness_by_horizon_and_volume_decile(tidy_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize price staleness by horizon and within-horizon volume decile."""
    if tidy_df.empty:
        return tidy_df.copy()

    df = tidy_df.dropna(subset=["horizon_min", "volume_total_market"]).copy()
    if df.empty:
        return df

    df = _assign_volume_deciles_within_horizon(df)
    grouped = (
        df.groupby(["horizon_min", "volume_bucket"], observed=True)
        .agg(
            n_rows=("market_id", "count"),
            n_price_available=("price_last_trade", lambda s: int(s.notna().sum())),
            n_staleness_available=("price_staleness_min", lambda s: int(s.notna().sum())),
            mean_staleness_min=("price_staleness_min", "mean"),
            median_staleness_min=("price_staleness_min", "median"),
            p90_staleness_min=("price_staleness_min", lambda s: float(s.quantile(0.90))),
            decile_volume_min=("volume_total_market", "min"),
            decile_volume_max=("volume_total_market", "max"),
        )
        .reset_index()
    )
    grouped["volume_bucket"] = pd.Categorical(
        grouped["volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    return grouped.sort_values(["horizon_min", "volume_bucket"]).reset_index(drop=True)


def compute_mae_by_horizon_global_volume_decile(tidy_df: pd.DataFrame) -> pd.DataFrame:
    """MAE by horizon for global (cohort-stable) volume deciles."""
    if tidy_df.empty:
        return tidy_df.copy()

    df = tidy_df.dropna(
        subset=["market_id", "horizon_min", "volume_total_market", "price_last_trade", "abs_error_last_trade"]
    ).copy()
    if df.empty:
        return df
    if "snapshot_time" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_time"], utc=True, errors="coerce").dt.date

    market_volume = df[["market_id", "volume_total_market"]].drop_duplicates(subset=["market_id"]).copy()
    market_volume["global_volume_bucket"] = _volume_deciles_within_horizon(market_volume["volume_total_market"])
    df = df.merge(market_volume[["market_id", "global_volume_bucket"]], on="market_id", how="left")

    group_cols = ["horizon_min", "global_volume_bucket"]
    if "snapshot_date" in df.columns and df["snapshot_date"].notna().any():
        group_cols = ["snapshot_date"] + group_cols
    grouped = (
        df.groupby(group_cols, observed=True)
        .agg(
            mae=("abs_error_last_trade", "mean"),
            n=("market_id", "count"),
            global_volume_min=("volume_total_market", "min"),
            global_volume_max=("volume_total_market", "max"),
        )
        .reset_index()
    )
    grouped["global_volume_bucket"] = pd.Categorical(
        grouped["global_volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    sort_cols = ["horizon_min", "global_volume_bucket"]
    if "snapshot_date" in grouped.columns:
        sort_cols = ["snapshot_date"] + sort_cols
    return grouped.sort_values(sort_cols).reset_index(drop=True)


def compute_volume_error_joint_diagnostics(tidy_df: pd.DataFrame, wide_df: pd.DataFrame) -> pd.DataFrame:
    """Joint horizon-volume diagnostics across MAE, staleness, and volatility."""
    if tidy_df.empty:
        return tidy_df.copy()

    df = tidy_df.dropna(
        subset=["market_id", "horizon_min", "volume_total_market", "price_last_trade", "abs_error_last_trade"]
    ).copy()
    if df.empty:
        return df

    vol_cols = [c for c in ("market_id", "volatility_std", "volatility_range", "horizon_count_used") if c in wide_df.columns]
    if vol_cols:
        vol_df = wide_df[vol_cols].drop_duplicates(subset=["market_id"])
        df = df.merge(vol_df, on="market_id", how="left")

    df = _assign_volume_deciles_within_horizon(df)
    grouped = (
        df.groupby(["horizon_min", "volume_bucket"], observed=True)
        .agg(
            n=("market_id", "count"),
            mae=("abs_error_last_trade", "mean"),
            mean_staleness_min=("price_staleness_min", "mean"),
            median_staleness_min=("price_staleness_min", "median"),
            p90_staleness_min=("price_staleness_min", lambda s: float(s.quantile(0.90))),
            mean_volatility_std=("volatility_std", "mean"),
            mean_volatility_range=("volatility_range", "mean"),
            mean_horizon_count_used=("horizon_count_used", "mean"),
            decile_volume_min=("volume_total_market", "min"),
            decile_volume_max=("volume_total_market", "max"),
        )
        .reset_index()
    )
    grouped["volume_bucket"] = pd.Categorical(
        grouped["volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    return grouped.sort_values(["horizon_min", "volume_bucket"]).reset_index(drop=True)


def _ols_fit_table(y: pd.Series, x: pd.DataFrame, model_name: str) -> tuple[pd.DataFrame, dict]:
    """Lightweight OLS via least squares with approximate standard errors."""
    x_num = x.astype(float)
    y_num = y.astype(float)
    valid = x_num.notna().all(axis=1) & y_num.notna()
    x_num = x_num.loc[valid]
    y_num = y_num.loc[valid]
    if x_num.empty:
        return pd.DataFrame(), {"model": model_name, "n_obs": 0, "r2": np.nan, "rmse": np.nan}

    x_mat = x_num.to_numpy(dtype=float)
    y_vec = y_num.to_numpy(dtype=float)

    coef, _, _, _ = np.linalg.lstsq(x_mat, y_vec, rcond=None)
    fitted = x_mat @ coef
    resid = y_vec - fitted
    n_obs = len(y_vec)
    n_params = x_mat.shape[1]
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y_vec - y_vec.mean()) ** 2))
    r2 = np.nan if sst <= 0 else 1.0 - (sse / sst)
    rmse = float(np.sqrt(sse / max(n_obs, 1)))

    dof = max(n_obs - n_params, 1)
    sigma2 = sse / dof
    xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
    se = np.sqrt(np.clip(np.diag(xtx_inv) * sigma2, a_min=0.0, a_max=None))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stat = np.where(se > 0, coef / se, np.nan)

    coef_df = pd.DataFrame(
        {
            "model": model_name,
            "term": x_num.columns,
            "coefficient": coef,
            "std_error": se,
            "t_stat": t_stat,
            "n_obs": n_obs,
            "r2": r2,
            "rmse": rmse,
        }
    )
    model_stats = {"model": model_name, "n_obs": n_obs, "n_params": n_params, "r2": r2, "rmse": rmse}
    return coef_df, model_stats


def compute_volume_error_control_analysis(
    tidy_df: pd.DataFrame, wide_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Control analysis for abs error vs volume with staleness/volatility/horizon controls."""
    if tidy_df.empty:
        empty = pd.DataFrame()
        return empty, empty

    df = tidy_df.dropna(
        subset=["market_id", "horizon_min", "volume_total_market", "abs_error_last_trade"]
    ).copy()
    if df.empty:
        empty = pd.DataFrame()
        return empty, empty

    vol_cols = [c for c in ("market_id", "volatility_std") if c in wide_df.columns]
    if vol_cols:
        vol_df = wide_df[vol_cols].drop_duplicates(subset=["market_id"])
        df = df.merge(vol_df, on="market_id", how="left")

    df["log_volume"] = np.log(df["volume_total_market"].clip(lower=1.0))
    df["log_horizon"] = np.log(df["horizon_min"].clip(lower=1.0))
    df["log1p_staleness"] = np.log1p(df["price_staleness_min"].clip(lower=0.0))

    y = df["abs_error_last_trade"].astype(float)

    x_base = pd.DataFrame(
        {
            "intercept": 1.0,
            "log_volume": df["log_volume"],
            "log_horizon": df["log_horizon"],
        }
    )
    coef_base, stats_base = _ols_fit_table(y, x_base, model_name="baseline")

    horizon_fe = pd.get_dummies(df["horizon_min"].astype(int), prefix="h", drop_first=True).astype(float)
    x_control = pd.concat(
        [
            pd.DataFrame(
                {
                    "intercept": 1.0,
                    "log_volume": df["log_volume"],
                    "log_horizon": df["log_horizon"],
                    "log1p_staleness": df["log1p_staleness"],
                    "volatility_std": df["volatility_std"],
                }
            ),
            horizon_fe,
        ],
        axis=1,
    )
    coef_control, stats_control = _ols_fit_table(y, x_control, model_name="with_controls")

    coef_df = pd.concat([coef_base, coef_control], ignore_index=True)
    if coef_df.empty:
        return coef_df, pd.DataFrame([stats_base, stats_control])

    focus_terms = {"intercept", "log_volume", "log_horizon", "log1p_staleness", "volatility_std"}
    coef_df["is_focus_term"] = coef_df["term"].isin(focus_terms)
    model_df = pd.DataFrame([stats_base, stats_control])
    return coef_df, model_df


def compute_price_change_distribution_by_global_decile(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Per-market horizon-pair price changes with stable global volume deciles."""
    if wide_df.empty:
        return wide_df.copy()

    required = {"market_id", "volume_total_market"}
    if not required.issubset(set(wide_df.columns)):
        return pd.DataFrame()

    horizon_price_cols = _sorted_price_horizon_columns(wide_df)
    if len(horizon_price_cols) < 2:
        return pd.DataFrame()

    market_volume = wide_df[["market_id", "volume_total_market"]].dropna(
        subset=["market_id", "volume_total_market"]
    )
    if market_volume.empty:
        return pd.DataFrame()
    market_volume = market_volume.drop_duplicates(subset=["market_id"]).copy()
    market_volume["global_volume_bucket"] = _volume_deciles_within_horizon(
        market_volume["volume_total_market"]
    )

    use_cols = ["market_id"] + [col for _, col in horizon_price_cols]
    base = wide_df[use_cols].drop_duplicates(subset=["market_id"]).copy()
    base = base.merge(
        market_volume[["market_id", "global_volume_bucket"]],
        on="market_id",
        how="inner",
    )
    if base.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for from_h, from_col in horizon_price_cols:
        for to_h, to_col in horizon_price_cols:
            if from_h <= to_h:
                continue
            pair_df = base[
                ["market_id", "global_volume_bucket", from_col, to_col]
            ].dropna(subset=[from_col, to_col])
            if pair_df.empty:
                continue
            out = pair_df.rename(
                columns={
                    from_col: "price_from",
                    to_col: "price_to",
                }
            ).copy()
            out["from_horizon_min"] = int(from_h)
            out["to_horizon_min"] = int(to_h)
            out["price_change"] = out["price_to"] - out["price_from"]
            out["abs_price_change"] = out["price_change"].abs()
            rows.append(
                out[
                    [
                        "market_id",
                        "global_volume_bucket",
                        "from_horizon_min",
                        "to_horizon_min",
                        "price_from",
                        "price_to",
                        "price_change",
                        "abs_price_change",
                    ]
                ]
            )

    if not rows:
        return pd.DataFrame()

    out_df = pd.concat(rows, ignore_index=True)
    out_df["global_volume_bucket"] = pd.Categorical(
        out_df["global_volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    return out_df.sort_values(
        ["global_volume_bucket", "from_horizon_min", "to_horizon_min", "market_id"]
    ).reset_index(drop=True)


def compute_mae_change_distribution_by_global_decile(wide_df: pd.DataFrame) -> pd.DataFrame:
    """Per-market absolute-error changes between horizon pairs for global deciles."""
    if wide_df.empty:
        return wide_df.copy()

    required = {"market_id", "volume_total_market", "outcome_yes"}
    if not required.issubset(set(wide_df.columns)):
        return pd.DataFrame()

    horizon_price_cols = _sorted_price_horizon_columns(wide_df)
    if len(horizon_price_cols) < 2:
        return pd.DataFrame()

    market_volume = wide_df[["market_id", "volume_total_market"]].dropna(
        subset=["market_id", "volume_total_market"]
    )
    if market_volume.empty:
        return pd.DataFrame()
    market_volume = market_volume.drop_duplicates(subset=["market_id"]).copy()
    market_volume["global_volume_bucket"] = _volume_deciles_within_horizon(
        market_volume["volume_total_market"]
    )

    use_cols = ["market_id", "outcome_yes"] + [col for _, col in horizon_price_cols]
    base = wide_df[use_cols].drop_duplicates(subset=["market_id"]).copy()
    base = base.merge(
        market_volume[["market_id", "global_volume_bucket"]],
        on="market_id",
        how="inner",
    )
    if base.empty:
        return pd.DataFrame()

    rows: list[pd.DataFrame] = []
    for from_h, from_col in horizon_price_cols:
        for to_h, to_col in horizon_price_cols:
            if from_h <= to_h:
                continue
            pair_df = base[
                ["market_id", "global_volume_bucket", "outcome_yes", from_col, to_col]
            ].dropna(subset=["outcome_yes", from_col, to_col])
            if pair_df.empty:
                continue
            out = pair_df.rename(
                columns={
                    from_col: "price_from",
                    to_col: "price_to",
                }
            ).copy()
            out["from_horizon_min"] = int(from_h)
            out["to_horizon_min"] = int(to_h)
            out["abs_error_from"] = (out["price_from"] - out["outcome_yes"]).abs()
            out["abs_error_to"] = (out["price_to"] - out["outcome_yes"]).abs()
            out["mae_change"] = out["abs_error_to"] - out["abs_error_from"]
            out["abs_mae_change"] = out["mae_change"].abs()
            rows.append(
                out[
                    [
                        "market_id",
                        "global_volume_bucket",
                        "from_horizon_min",
                        "to_horizon_min",
                        "price_from",
                        "price_to",
                        "outcome_yes",
                        "abs_error_from",
                        "abs_error_to",
                        "mae_change",
                        "abs_mae_change",
                    ]
                ]
            )

    if not rows:
        return pd.DataFrame()

    out_df = pd.concat(rows, ignore_index=True)
    out_df["global_volume_bucket"] = pd.Categorical(
        out_df["global_volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    return out_df.sort_values(
        ["global_volume_bucket", "from_horizon_min", "to_horizon_min", "market_id"]
    ).reset_index(drop=True)


def build_volatility_analysis_tables(
    wide_df: pd.DataFrame,
    tidy_df: pd.DataFrame,
    isotonic_points_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build per-market and bucketed volatility analyses for edge/error proxies."""
    if wide_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    volatility_cols = ["market_id", "volatility_std", "volatility_range", "horizon_count_used"]
    present_vol_cols = [c for c in volatility_cols if c in wide_df.columns]
    per_market = wide_df[present_vol_cols].copy()
    if "horizon_count_used" not in per_market.columns:
        per_market["horizon_count_used"] = np.nan

    tidy_market = pd.DataFrame(columns=["market_id"])
    if not tidy_df.empty and "market_id" in tidy_df.columns:
        tidy_market = (
            tidy_df.groupby("market_id", dropna=False)
            .agg(
                n_tidy_rows=("market_id", "count"),
                mean_abs_error_last_trade=("abs_error_last_trade", "mean"),
                median_abs_error_last_trade=("abs_error_last_trade", "median"),
                mean_brier_last_trade=("brier_last_trade", "mean"),
                median_brier_last_trade=("brier_last_trade", "median"),
            )
            .reset_index()
        )

    isotonic_market = pd.DataFrame(columns=["market_id"])
    if (
        not isotonic_points_df.empty
        and "market_id" in isotonic_points_df.columns
        and "price_last_trade" in isotonic_points_df.columns
        and "calibrated_prob" in isotonic_points_df.columns
    ):
        iso = isotonic_points_df.copy()
        iso["isotonic_gap"] = iso["price_last_trade"] - iso["calibrated_prob"]
        iso["abs_isotonic_gap"] = iso["isotonic_gap"].abs()
        isotonic_market = (
            iso.groupby("market_id", dropna=False)
            .agg(
                n_isotonic_rows=("market_id", "count"),
                mean_abs_isotonic_gap=("abs_isotonic_gap", "mean"),
                median_abs_isotonic_gap=("abs_isotonic_gap", "median"),
                mean_signed_isotonic_gap=("isotonic_gap", "mean"),
                share_positive_isotonic_gap=("isotonic_gap", lambda s: float((s > 0).mean())),
            )
            .reset_index()
        )

    per_market = per_market.merge(tidy_market, on="market_id", how="left")
    per_market = per_market.merge(isotonic_market, on="market_id", how="left")

    bucket_rows: list[dict] = []
    threshold_rows: list[dict] = []
    for volatility_metric in ("volatility_std", "volatility_range"):
        if volatility_metric not in per_market.columns:
            continue

        d = per_market.dropna(subset=[volatility_metric]).copy()
        if d.empty:
            continue
        d["volatility_bucket"] = _quantile_bucket_labels(d[volatility_metric], max_buckets=10)
        d = d.dropna(subset=["volatility_bucket"])
        if d.empty:
            continue

        bucket_summary = (
            d.groupby("volatility_bucket", dropna=False)
            .agg(
                n_markets=("market_id", "count"),
                volatility_min=(volatility_metric, "min"),
                volatility_max=(volatility_metric, "max"),
                volatility_mean=(volatility_metric, "mean"),
                volatility_median=(volatility_metric, "median"),
                mean_abs_isotonic_gap=("mean_abs_isotonic_gap", "mean"),
                median_abs_isotonic_gap=("mean_abs_isotonic_gap", "median"),
                mean_abs_error_last_trade=("mean_abs_error_last_trade", "mean"),
                median_abs_error_last_trade=("mean_abs_error_last_trade", "median"),
                mean_brier_last_trade=("mean_brier_last_trade", "mean"),
                median_brier_last_trade=("mean_brier_last_trade", "median"),
            )
            .reset_index()
        )
        labels = _decile_labels(min(10, len(d)))
        bucket_summary["volatility_bucket"] = pd.Categorical(
            bucket_summary["volatility_bucket"],
            categories=labels,
            ordered=True,
        )
        bucket_summary = bucket_summary.sort_values("volatility_bucket").reset_index(drop=True)
        bucket_summary["volatility_metric"] = volatility_metric
        bucket_summary["bucket_rank"] = np.arange(1, len(bucket_summary) + 1, dtype=int)

        for row in bucket_summary.to_dict("records"):
            bucket_rows.append(row)

        low_labels = set(labels[: min(3, len(labels))])
        high_labels = set(labels[max(0, len(labels) - 3) :])
        low_mask = bucket_summary["volatility_bucket"].astype(str).isin(low_labels)
        high_mask = bucket_summary["volatility_bucket"].astype(str).isin(high_labels)

        row = {
            "volatility_metric": volatility_metric,
            "n_total_markets": int(bucket_summary["n_markets"].sum()),
            "low_bucket_labels": ", ".join(sorted(low_labels)),
            "high_bucket_labels": ", ".join(sorted(high_labels)),
            "high_minus_low_mean_abs_isotonic_gap": np.nan,
            "high_minus_low_mean_abs_error_last_trade": np.nan,
            "high_minus_low_mean_brier_last_trade": np.nan,
            "slope_mean_abs_isotonic_gap_by_bucket": np.nan,
            "slope_mean_abs_error_last_trade_by_bucket": np.nan,
            "slope_mean_brier_last_trade_by_bucket": np.nan,
        }
        if low_mask.any() and high_mask.any():
            low_iso = bucket_summary.loc[low_mask, "mean_abs_isotonic_gap"].mean()
            high_iso = bucket_summary.loc[high_mask, "mean_abs_isotonic_gap"].mean()
            row["high_minus_low_mean_abs_isotonic_gap"] = float(high_iso - low_iso)

            low_mae = bucket_summary.loc[low_mask, "mean_abs_error_last_trade"].mean()
            high_mae = bucket_summary.loc[high_mask, "mean_abs_error_last_trade"].mean()
            row["high_minus_low_mean_abs_error_last_trade"] = float(high_mae - low_mae)

            low_brier = bucket_summary.loc[low_mask, "mean_brier_last_trade"].mean()
            high_brier = bucket_summary.loc[high_mask, "mean_brier_last_trade"].mean()
            row["high_minus_low_mean_brier_last_trade"] = float(high_brier - low_brier)

        x = bucket_summary["bucket_rank"].astype(float).to_numpy()
        for metric_name, out_name in (
            ("mean_abs_isotonic_gap", "slope_mean_abs_isotonic_gap_by_bucket"),
            ("mean_abs_error_last_trade", "slope_mean_abs_error_last_trade_by_bucket"),
            ("mean_brier_last_trade", "slope_mean_brier_last_trade_by_bucket"),
        ):
            y = bucket_summary[metric_name].astype(float)
            valid = y.notna()
            if valid.sum() > 1:
                row[out_name] = float(np.polyfit(x[valid.to_numpy()], y[valid].to_numpy(), 1)[0])
        threshold_rows.append(row)

    bucket_df = pd.DataFrame(bucket_rows)
    threshold_df = pd.DataFrame(threshold_rows)
    return per_market, bucket_df, threshold_df


def compute_calibration_by_horizon(
    tidy_df: pd.DataFrame,
    bins: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if tidy_df.empty:
        return tidy_df, tidy_df

    df = tidy_df.dropna(subset=["price_last_trade"]).copy()
    if df.empty:
        return df, df

    edges = [x / bins for x in range(bins + 1)]
    labels = [f"[{edges[i]:.1f},{edges[i+1]:.1f})" for i in range(bins)]
    labels[-1] = f"[{edges[-2]:.1f},{edges[-1]:.1f}]"

    df["prob_bin"] = pd.cut(
        df["price_last_trade"].clip(0, 1),
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    ).astype(str)

    calibration = (
        df.groupby(["horizon_min", "prob_bin"], dropna=False)
        .agg(
            n=("market_id", "count"),
            predicted_mean=("price_last_trade", "mean"),
            observed_yes_rate=("outcome_yes", "mean"),
            volume_mean=("volume_total_market", "mean"),
        )
        .reset_index()
    )

    horizon_metrics = (
        df.groupby("horizon_min")
        .agg(
            n=("market_id", "count"),
            mae=("abs_error_last_trade", "mean"),
            brier=("brier_last_trade", "mean"),
            observed_yes_rate=("outcome_yes", "mean"),
            predicted_mean=("price_last_trade", "mean"),
        )
        .reset_index()
    )
    return calibration, horizon_metrics


def compute_category_horizon_error_metrics(tidy_df: pd.DataFrame) -> pd.DataFrame:
    """Raw forecast error metrics by category and horizon."""
    if tidy_df.empty:
        return tidy_df.copy()

    df = tidy_df.dropna(
        subset=["horizon_min", "price_last_trade", "outcome_yes", "abs_error_last_trade", "brier_last_trade"]
    ).copy()
    if df.empty:
        return df

    df["category"] = _normalized_category_labels(df.get("category", pd.Series(index=df.index, dtype=object)))
    df["signed_error_last_trade"] = df["price_last_trade"] - df["outcome_yes"]

    grouped = (
        df.groupby(["category", "horizon_min"], dropna=False)
        .agg(
            n=("market_id", "count"),
            n_unique_markets=("market_id", "nunique"),
            predicted_mean=("price_last_trade", "mean"),
            observed_yes_rate=("outcome_yes", "mean"),
            mae=("abs_error_last_trade", "mean"),
            brier=("brier_last_trade", "mean"),
            mean_signed_error=("signed_error_last_trade", "mean"),
            mean_volume=("volume_total_market", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(["horizon_min", "n"], ascending=[True, False]).reset_index(drop=True)


def compute_category_calibration_by_horizon(
    tidy_df: pd.DataFrame,
    bins: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Binned calibration and horizon metrics by category."""
    if tidy_df.empty:
        return tidy_df, tidy_df

    df = tidy_df.dropna(subset=["horizon_min", "price_last_trade", "outcome_yes"]).copy()
    if df.empty:
        return df, df

    df["category"] = _normalized_category_labels(df.get("category", pd.Series(index=df.index, dtype=object)))

    edges = [x / bins for x in range(bins + 1)]
    labels = [f"[{edges[i]:.1f},{edges[i+1]:.1f})" for i in range(bins)]
    labels[-1] = f"[{edges[-2]:.1f},{edges[-1]:.1f}]"
    df["prob_bin"] = pd.cut(
        df["price_last_trade"].clip(0, 1),
        bins=edges,
        labels=labels,
        include_lowest=True,
        right=False,
    ).astype(str)

    calibration = (
        df.groupby(["category", "horizon_min", "prob_bin"], dropna=False)
        .agg(
            n=("market_id", "count"),
            predicted_mean=("price_last_trade", "mean"),
            observed_yes_rate=("outcome_yes", "mean"),
            volume_mean=("volume_total_market", "mean"),
        )
        .reset_index()
    )
    horizon_metrics = (
        df.groupby(["category", "horizon_min"], dropna=False)
        .agg(
            n=("market_id", "count"),
            n_unique_markets=("market_id", "nunique"),
            mae=("abs_error_last_trade", "mean"),
            brier=("brier_last_trade", "mean"),
            observed_yes_rate=("outcome_yes", "mean"),
            predicted_mean=("price_last_trade", "mean"),
            volume_mean=("volume_total_market", "mean"),
        )
        .reset_index()
    )
    calibration = calibration.sort_values(["horizon_min", "category", "predicted_mean"]).reset_index(drop=True)
    horizon_metrics = horizon_metrics.sort_values(["horizon_min", "n"], ascending=[True, False]).reset_index(
        drop=True
    )
    return calibration, horizon_metrics


def compute_category_isotonic_metrics_by_horizon(tidy_df: pd.DataFrame) -> pd.DataFrame:
    """Category and horizon isotonic calibration diagnostics."""
    if tidy_df.empty:
        return tidy_df.copy()

    df = tidy_df.dropna(subset=["horizon_min", "price_last_trade", "outcome_yes"]).copy()
    if df.empty:
        return df

    df["category"] = _normalized_category_labels(df.get("category", pd.Series(index=df.index, dtype=object)))
    rows: list[dict] = []
    for (category, horizon), g in df.groupby(["category", "horizon_min"], dropna=False):
        if g.empty:
            continue
        x = g["price_last_trade"].clip(0, 1).astype(float).to_numpy()
        y = g["outcome_yes"].astype(float).to_numpy()
        if len(x) == 0:
            continue

        model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        try:
            calibrated = model.fit_transform(x, y)
        except ValueError:
            calibrated = np.full_like(x, fill_value=float(np.mean(y)))

        raw_err = x - y
        cal_err = calibrated - y
        isotonic_gap = x - calibrated
        rows.append(
            {
                "category": category,
                "horizon_min": int(horizon),
                "n": int(len(g)),
                "n_unique_markets": int(g["market_id"].nunique()),
                "raw_predicted_mean": float(np.mean(x)),
                "calibrated_predicted_mean": float(np.mean(calibrated)),
                "observed_yes_rate": float(np.mean(y)),
                "raw_mae": float(np.mean(np.abs(raw_err))),
                "raw_brier": float(np.mean(raw_err**2)),
                "calibrated_mae": float(np.mean(np.abs(cal_err))),
                "calibrated_brier": float(np.mean(cal_err**2)),
                "mean_signed_isotonic_gap": float(np.mean(isotonic_gap)),
                "mean_abs_isotonic_gap": float(np.mean(np.abs(isotonic_gap))),
                "share_positive_isotonic_gap": float(np.mean(isotonic_gap > 0)),
                "mean_volume": float(g["volume_total_market"].mean()),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["horizon_min", "n"], ascending=[True, False]).reset_index(drop=True)


def compute_isotonic_calibration_by_horizon(
    tidy_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if tidy_df.empty:
        return tidy_df, tidy_df

    df = tidy_df.dropna(subset=["price_last_trade", "outcome_yes"]).copy()
    if df.empty:
        return df, df

    points_rows = []
    metrics_rows = []
    for horizon in sorted(df["horizon_min"].dropna().unique()):
        horizon_df = df[df["horizon_min"] == horizon].copy()
        if horizon_df.empty:
            continue

        x = horizon_df["price_last_trade"].clip(0, 1).astype(float)
        y = horizon_df["outcome_yes"].astype(float)

        model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        calibrated = model.fit_transform(x, y)

        points_rows.extend(
            {
                "market_id": row["market_id"],
                "horizon_min": int(horizon),
                "price_last_trade": float(price),
                "calibrated_prob": float(cal_prob),
                "outcome_yes": float(outcome),
                "volume_total_market": float(row["volume_total_market"]),
            }
            for row, price, cal_prob, outcome in zip(
                horizon_df.to_dict("records"),
                x,
                calibrated,
                y,
            )
        )

        raw_err = x - y
        cal_err = calibrated - y
        metrics_rows.append(
            {
                "horizon_min": int(horizon),
                "n": int(len(horizon_df)),
                "raw_predicted_mean": float(x.mean()),
                "calibrated_predicted_mean": float(calibrated.mean()),
                "observed_yes_rate": float(y.mean()),
                "raw_mae": float(raw_err.abs().mean()),
                "raw_brier": float((raw_err**2).mean()),
                "calibrated_mae": float(pd.Series(cal_err).abs().mean()),
                "calibrated_brier": float((cal_err**2).mean()),
            }
        )

    isotonic_points = pd.DataFrame(points_rows)
    isotonic_horizon_metrics = pd.DataFrame(metrics_rows)
    return isotonic_points, isotonic_horizon_metrics


def compute_lowess_calibration_by_horizon(
    tidy_df: pd.DataFrame,
    frac: float = 0.2,
    it: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if tidy_df.empty:
        return tidy_df, tidy_df

    df = tidy_df.dropna(subset=["price_last_trade", "outcome_yes"]).copy()
    if df.empty:
        return df, df

    points_rows = []
    metrics_rows = []
    for horizon in sorted(df["horizon_min"].dropna().unique()):
        horizon_df = df[df["horizon_min"] == horizon].copy()
        if horizon_df.empty:
            continue

        x = horizon_df["price_last_trade"].clip(0, 1).astype(float).to_numpy()
        y = horizon_df["outcome_yes"].astype(float).to_numpy()

        grouped = (
            pd.DataFrame({"x": x, "y": y})
            .groupby("x", as_index=False)
            .agg(y=("y", "mean"))
            .sort_values("x")
        )
        x_unique = grouped["x"].to_numpy()
        y_unique = grouped["y"].to_numpy()
        if len(x_unique) < 2:
            y_smooth = np.full_like(y_unique, fill_value=float(np.mean(y)))
            x_smooth = x_unique
        else:
            smoothed = lowess(
                endog=y_unique,
                exog=x_unique,
                frac=frac,
                it=it,
                return_sorted=True,
            )
            x_smooth = smoothed[:, 0]
            y_smooth = np.clip(smoothed[:, 1], 0.0, 1.0)
            if np.isnan(y_smooth).any():
                y_smooth = np.nan_to_num(y_smooth, nan=float(np.mean(y)))

        order = np.argsort(x)
        lowess_prob_sorted = np.interp(x[order], x_smooth, y_smooth)
        lowess_prob = np.empty_like(lowess_prob_sorted)
        lowess_prob[order] = lowess_prob_sorted

        points_rows.extend(
            {
                "market_id": row["market_id"],
                "horizon_min": int(horizon),
                "price_last_trade": float(price),
                "lowess_prob": float(low_prob),
                "outcome_yes": float(outcome),
                "volume_total_market": float(row["volume_total_market"]),
            }
            for row, price, low_prob, outcome in zip(
                horizon_df.to_dict("records"),
                x,
                lowess_prob,
                y,
            )
        )

        raw_err = x - y
        lowess_err = lowess_prob - y
        metrics_rows.append(
            {
                "horizon_min": int(horizon),
                "n": int(len(horizon_df)),
                "raw_predicted_mean": float(np.mean(x)),
                "lowess_predicted_mean": float(np.mean(lowess_prob)),
                "observed_yes_rate": float(np.mean(y)),
                "raw_mae": float(np.mean(np.abs(raw_err))),
                "raw_brier": float(np.mean(raw_err**2)),
                "lowess_mae": float(np.mean(np.abs(lowess_err))),
                "lowess_brier": float(np.mean(lowess_err**2)),
                "lowess_frac": float(frac),
            }
        )

    lowess_points = pd.DataFrame(points_rows)
    lowess_horizon_metrics = pd.DataFrame(metrics_rows)
    return lowess_points, lowess_horizon_metrics


def _volume_deciles_within_horizon(volume: pd.Series) -> pd.Series:
    """Equal-count volume deciles within one horizon."""
    if volume.empty:
        return pd.Series(dtype=object, index=volume.index)
    ranked = volume.rank(method="first")
    n = len(volume)
    k = min(10, n)
    if k < 1:
        return pd.Series(index=volume.index, dtype=object)
    return pd.qcut(ranked, q=k, labels=_VOLUME_DECILE_LABELS[:k])


def _overlapping_price_bin_starts(bin_width: float = 0.1, stride: float = 0.1) -> list[float]:
    starts: list[float] = []
    s = 0.0
    while s <= 1.0 - bin_width + 1e-12:
        starts.append(round(s, 10))
        s += stride
    return starts


def compute_isotonic_gap_by_volume_decile(isotonic_points_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize isotonic mispricing by within-horizon volume decile."""
    if isotonic_points_df.empty:
        return isotonic_points_df.copy()

    df = isotonic_points_df.dropna(
        subset=["horizon_min", "price_last_trade", "calibrated_prob", "volume_total_market"]
    ).copy()
    if df.empty:
        return df

    df["isotonic_gap"] = df["price_last_trade"] - df["calibrated_prob"]
    df["abs_isotonic_gap"] = df["isotonic_gap"].abs()
    parts: list[pd.Series] = []
    for _, g in df.groupby("horizon_min", sort=False):
        parts.append(_volume_deciles_within_horizon(g["volume_total_market"]))
    df["volume_bucket"] = pd.concat(parts)

    summary = (
        df.groupby(["horizon_min", "volume_bucket"], observed=True)
        .agg(
            n=("market_id", "count"),
            mean_abs_isotonic_gap=("abs_isotonic_gap", "mean"),
            median_abs_isotonic_gap=("abs_isotonic_gap", "median"),
            mean_signed_isotonic_gap=("isotonic_gap", "mean"),
            share_positive_isotonic_gap=("isotonic_gap", lambda s: float((s > 0).mean())),
            decile_volume_min=("volume_total_market", "min"),
            decile_volume_max=("volume_total_market", "max"),
        )
        .reset_index()
    )
    summary["volume_bucket"] = pd.Categorical(
        summary["volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    return summary.sort_values(["horizon_min", "volume_bucket"]).reset_index(drop=True)


def compute_isotonic_gap_by_volume_decile_price_bin(
    isotonic_points_df: pd.DataFrame,
    bin_width: float = 0.1,
    stride: float = 0.1,
) -> pd.DataFrame:
    """Aggregate isotonic gap by horizon, within-horizon volume decile, and overlapping price bins."""
    if isotonic_points_df.empty:
        return isotonic_points_df.copy()

    df = isotonic_points_df.dropna(
        subset=["horizon_min", "price_last_trade", "calibrated_prob", "volume_total_market"]
    ).copy()
    if df.empty:
        return df

    df["isotonic_gap"] = df["price_last_trade"] - df["calibrated_prob"]
    df["abs_isotonic_gap"] = df["isotonic_gap"].abs()

    starts = _overlapping_price_bin_starts(bin_width=bin_width, stride=stride)
    rows: list[dict] = []
    for horizon in sorted(df["horizon_min"].dropna().unique()):
        h = int(horizon)
        d_h = df[df["horizon_min"] == horizon].copy()
        d_h["volume_bucket"] = _volume_deciles_within_horizon(d_h["volume_total_market"])
        decile_bounds = (
            d_h.groupby("volume_bucket", observed=True)["volume_total_market"]
            .agg(decile_volume_min="min", decile_volume_max="max")
            .to_dict("index")
        )
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
                bounds = decile_bounds[volume_bucket]
                rows.append(
                    {
                        "horizon_min": h,
                        "volume_bucket": volume_bucket,
                        "bin_start": start,
                        "bin_end": end,
                        "bin_mid": (start + end) / 2.0,
                        "n": int(len(sub)),
                        "mean_signed_isotonic_gap": float(sub["isotonic_gap"].mean()),
                        "mean_abs_isotonic_gap": float(sub["abs_isotonic_gap"].mean()),
                        "decile_volume_min": float(bounds["decile_volume_min"]),
                        "decile_volume_max": float(bounds["decile_volume_max"]),
                    }
                )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["volume_bucket"] = pd.Categorical(
        out["volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    return out.sort_values(["horizon_min", "volume_bucket", "bin_mid"]).reset_index(drop=True)


def compute_isotonic_gap_threshold_summary(
    isotonic_gap_volume_decile_df: pd.DataFrame,
) -> pd.DataFrame:
    """Descriptive threshold comparisons for low-vs-high volume deciles by horizon."""
    if isotonic_gap_volume_decile_df.empty:
        return isotonic_gap_volume_decile_df.copy()

    df = isotonic_gap_volume_decile_df.copy()
    if "volume_bucket" not in df.columns:
        return pd.DataFrame()

    rows: list[dict] = []
    low_labels = {"Decile 1 (lowest)", "Decile 2", "Decile 3"}
    high_labels = {"Decile 8", "Decile 9", "Decile 10 (highest)"}

    for horizon in sorted(df["horizon_min"].dropna().unique()):
        d_h = df[df["horizon_min"] == horizon].copy()
        if d_h.empty:
            continue
        w = d_h["n"].astype(float)
        abs_gap = d_h["mean_abs_isotonic_gap"].astype(float)
        low_mask = d_h["volume_bucket"].astype(str).isin(low_labels)
        high_mask = d_h["volume_bucket"].astype(str).isin(high_labels)
        decile1_mask = d_h["volume_bucket"].astype(str).eq("Decile 1 (lowest)")
        others_mask = ~decile1_mask

        low_high_diff = np.nan
        if low_mask.any() and high_mask.any():
            low_mean = np.average(abs_gap[low_mask], weights=w[low_mask])
            high_mean = np.average(abs_gap[high_mask], weights=w[high_mask])
            low_high_diff = float(low_mean - high_mean)

        decile1_vs_rest_diff = np.nan
        if decile1_mask.any() and others_mask.any():
            d1_mean = np.average(abs_gap[decile1_mask], weights=w[decile1_mask])
            rest_mean = np.average(abs_gap[others_mask], weights=w[others_mask])
            decile1_vs_rest_diff = float(d1_mean - rest_mean)

        d_ordered = d_h.sort_values("volume_bucket")
        x = np.arange(1, len(d_ordered) + 1, dtype=float)
        y = d_ordered["mean_abs_isotonic_gap"].astype(float).to_numpy()
        slope = float(np.polyfit(x, y, 1)[0]) if len(y) > 1 else np.nan

        rows.append(
            {
                "horizon_min": int(horizon),
                "n_total": int(d_h["n"].sum()),
                "low123_minus_high810_mean_abs_gap": low_high_diff,
                "decile1_minus_rest_mean_abs_gap": decile1_vs_rest_diff,
                "decile_trend_slope": slope,
            }
        )

    return pd.DataFrame(rows).sort_values("horizon_min").reset_index(drop=True)


def compute_isotonic_volume_interpretation_table(
    isotonic_points_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Per-horizon, per-volume-decile interpretation table with uncertainty bands.

    Uses a normal-approximation 95% CI around mean absolute isotonic gap:
    mean +/- 1.96 * std / sqrt(n), where std is sample std (ddof=1).
    """
    if isotonic_points_df.empty:
        return isotonic_points_df.copy()

    df = isotonic_points_df.dropna(
        subset=["horizon_min", "price_last_trade", "calibrated_prob", "volume_total_market"]
    ).copy()
    if df.empty:
        return df

    df["isotonic_gap"] = df["price_last_trade"] - df["calibrated_prob"]
    df["abs_isotonic_gap"] = df["isotonic_gap"].abs()

    parts: list[pd.Series] = []
    for _, g in df.groupby("horizon_min", sort=False):
        parts.append(_volume_deciles_within_horizon(g["volume_total_market"]))
    df["volume_bucket"] = pd.concat(parts)

    grouped = (
        df.groupby(["horizon_min", "volume_bucket"], observed=True)
        .agg(
            n=("market_id", "count"),
            mean_abs_isotonic_gap=("abs_isotonic_gap", "mean"),
            std_abs_isotonic_gap=("abs_isotonic_gap", "std"),
            mean_signed_isotonic_gap=("isotonic_gap", "mean"),
            share_positive_isotonic_gap=("isotonic_gap", lambda s: float((s > 0).mean())),
            decile_volume_min=("volume_total_market", "min"),
            decile_volume_max=("volume_total_market", "max"),
        )
        .reset_index()
    )

    n = grouped["n"].astype(float)
    std = grouped["std_abs_isotonic_gap"].fillna(0.0).astype(float)
    se = std / np.sqrt(n.clip(lower=1.0))
    grouped["mean_abs_gap_ci95_low"] = (grouped["mean_abs_isotonic_gap"] - 1.96 * se).clip(lower=0.0)
    grouped["mean_abs_gap_ci95_high"] = grouped["mean_abs_isotonic_gap"] + 1.96 * se

    grouped["volume_bucket"] = pd.Categorical(
        grouped["volume_bucket"],
        categories=_VOLUME_DECILE_LABELS,
        ordered=True,
    )
    grouped = grouped.sort_values(["horizon_min", "volume_bucket"]).reset_index(drop=True)

    grouped["rank_abs_gap_within_horizon"] = (
        grouped.groupby("horizon_min")["mean_abs_isotonic_gap"].rank(method="dense", ascending=True).astype(int)
    )
    grouped["is_lowest_abs_gap_within_horizon"] = (
        grouped.groupby("horizon_min")["mean_abs_isotonic_gap"].transform("min")
        == grouped["mean_abs_isotonic_gap"]
    )
    grouped["is_highest_abs_gap_within_horizon"] = (
        grouped.groupby("horizon_min")["mean_abs_isotonic_gap"].transform("max")
        == grouped["mean_abs_isotonic_gap"]
    )
    return grouped


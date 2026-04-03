from __future__ import annotations

from typing import List, Tuple

import pandas as pd

from .collect_snapshots import SnapshotRow


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
    return merged


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


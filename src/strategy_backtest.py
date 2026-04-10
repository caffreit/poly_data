from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


TradeSide = Literal["YES", "NO"]


@dataclass(frozen=True)
class StrategyConfig:
    snapshots_csv: Path = Path("output") / "market_snapshots.csv"
    output_dir: Path = Path("output") / "strategy"
    category: str = "Sports"
    horizon_min: int = 120
    train_ratio: float = 0.7
    starting_bankroll: float = 1000.0
    entry_threshold: float = 0.02
    kelly_fraction: float = 0.25
    max_bet_fraction: float = 0.10
    fixed_stake_usd: float = 25.0
    min_price: float = 0.01
    max_price: float = 0.99

    def ensure_output_dir(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)


def _normalize_category(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def load_strategy_frame(config: StrategyConfig) -> tuple[pd.DataFrame, dict[str, int]]:
    df = pd.read_csv(config.snapshots_csv)
    coverage: dict[str, int] = {"rows_total": int(len(df))}
    if df.empty:
        return df, coverage

    df["category_norm"] = df.get("category", pd.Series(index=df.index, dtype=object)).map(_normalize_category)
    df["end_time"] = pd.to_datetime(df["end_time"], utc=True, errors="coerce")
    if "closed_time" in df.columns:
        df["closed_time"] = pd.to_datetime(df["closed_time"], utc=True, errors="coerce")
    else:
        df["closed_time"] = pd.NaT
    df["snapshot_time"] = pd.to_datetime(df["snapshot_time"], utc=True, errors="coerce")

    df = df[df["category_norm"] == _normalize_category(config.category)].copy()
    coverage["rows_after_category"] = int(len(df))

    df = df[df["horizon_min"] == config.horizon_min].copy()
    coverage["rows_after_horizon"] = int(len(df))

    required_cols = ["price_last_trade", "outcome_yes", "snapshot_time", "end_time"]
    df = df.dropna(subset=required_cols).copy()
    coverage["rows_after_required_fields"] = int(len(df))

    df["price_last_trade"] = pd.to_numeric(df["price_last_trade"], errors="coerce")
    df["outcome_yes"] = pd.to_numeric(df["outcome_yes"], errors="coerce")
    df = df.dropna(subset=["price_last_trade", "outcome_yes"])
    coverage["rows_after_numeric_cast"] = int(len(df))

    price_mask = (df["price_last_trade"] >= config.min_price) & (df["price_last_trade"] <= config.max_price)
    df = df[price_mask].copy()
    coverage["rows_after_price_band"] = int(len(df))

    df = df.sort_values(["end_time", "snapshot_time", "market_id"]).reset_index(drop=True)
    coverage["rows_final"] = int(len(df))
    coverage["unique_markets_final"] = int(df["market_id"].nunique()) if "market_id" in df.columns else int(len(df))
    return df, coverage


def split_train_test(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return df.copy(), df.copy()

    unique_end = np.sort(df["end_time"].dropna().unique())
    if len(unique_end) < 2:
        return df.copy(), df.iloc[0:0].copy()

    raw_idx = int(np.floor(len(unique_end) * train_ratio)) - 1
    cutoff_idx = min(max(raw_idx, 0), len(unique_end) - 2)
    cutoff = unique_end[cutoff_idx]

    train_df = df[df["end_time"] <= cutoff].copy()
    test_df = df[df["end_time"] > cutoff].copy()
    return train_df, test_df


def fit_isotonic(train_df: pd.DataFrame) -> IsotonicRegression:
    x = train_df["price_last_trade"].clip(0, 1).astype(float).to_numpy()
    y = train_df["outcome_yes"].astype(float).to_numpy()
    model = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
    model.fit(x, y)
    return model


def _kelly_fraction_for_contract(model_prob: pd.Series, contract_price: pd.Series) -> pd.Series:
    safe_denom = (1.0 - contract_price).where((contract_price > 0.0) & (contract_price < 1.0))
    raw = (model_prob - contract_price) / safe_denom
    return raw.fillna(0.0)


def _build_signals(
    df: pd.DataFrame,
    *,
    model_prob_col: str,
    entry_threshold: float,
    kelly_fraction: float,
    max_bet_fraction: float,
    fixed_stake_usd: float | None = None,
) -> pd.DataFrame:
    out = df.copy()
    out["market_price"] = out["price_last_trade"].clip(0, 1).astype(float)
    out["model_prob"] = out[model_prob_col].clip(0, 1).astype(float)
    out["edge"] = out["model_prob"] - out["market_price"]
    out["trade_side"] = ""

    yes_mask = out["edge"] > entry_threshold
    no_mask = out["edge"] < -entry_threshold
    out.loc[yes_mask, "trade_side"] = "YES"
    out.loc[no_mask, "trade_side"] = "NO"

    out["kelly_raw"] = 0.0
    yes_rows = out["trade_side"] == "YES"
    no_rows = out["trade_side"] == "NO"
    out.loc[yes_rows, "kelly_raw"] = _kelly_fraction_for_contract(
        out.loc[yes_rows, "model_prob"], out.loc[yes_rows, "market_price"]
    )
    out.loc[no_rows, "kelly_raw"] = _kelly_fraction_for_contract(
        1.0 - out.loc[no_rows, "model_prob"], 1.0 - out.loc[no_rows, "market_price"]
    )
    out["kelly_raw"] = out["kelly_raw"].clip(lower=0.0)

    if fixed_stake_usd is None:
        out["bet_fraction"] = (out["kelly_raw"] * kelly_fraction).clip(upper=max_bet_fraction)
        out["fixed_stake_usd"] = np.nan
    else:
        out["bet_fraction"] = np.nan
        out["fixed_stake_usd"] = float(fixed_stake_usd)
    return out


def _simulate_wallet(
    signals_df: pd.DataFrame,
    *,
    starting_bankroll: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades = signals_df[signals_df["trade_side"].isin(["YES", "NO"])].copy()
    trades = trades.sort_values(["snapshot_time", "resolution_time", "market_id"]).reset_index(drop=True)
    if trades.empty:
        empty = trades.iloc[0:0].copy()
        equity = pd.DataFrame(
            [{"event_time": pd.NaT, "event_type": "start", "wallet_value": float(starting_bankroll), "cash": float(starting_bankroll), "locked_stake": 0.0}]
        )
        return empty, equity

    pending_resolutions: list[dict[str, object]] = []
    cash = float(starting_bankroll)
    trade_logs: list[dict[str, object]] = []
    equity_logs: list[dict[str, object]] = [
        {
            "event_time": pd.NaT,
            "event_type": "start",
            "wallet_value": float(starting_bankroll),
            "cash": float(starting_bankroll),
            "locked_stake": 0.0,
        }
    ]

    def _settle_until(current_time: pd.Timestamp) -> None:
        nonlocal cash, pending_resolutions
        if not pending_resolutions:
            return
        due_positions = [p for p in pending_resolutions if pd.Timestamp(p["resolution_time"]) <= current_time]
        if not due_positions:
            return
        due_positions = sorted(due_positions, key=lambda p: pd.Timestamp(p["resolution_time"]))
        due_times = sorted({pd.Timestamp(p["resolution_time"]) for p in due_positions})
        for due_time in due_times:
            due_now = [p for p in pending_resolutions if pd.Timestamp(p["resolution_time"]) == due_time]
            settle_now = float(sum(float(p["settle_value"]) for p in due_now))
            cash += settle_now
            for pos in due_now:
                pos["cash_after_resolution"] = cash
            pending_resolutions = [p for p in pending_resolutions if pd.Timestamp(p["resolution_time"]) != due_time]
            locked_after = float(sum(float(p["stake"]) for p in pending_resolutions))
            equity_logs.append(
                {
                    "event_time": due_time,
                    "event_type": "resolution",
                    "wallet_value": cash + locked_after,
                    "cash": cash,
                    "locked_stake": locked_after,
                }
            )

    for _, row in trades.iterrows():
        entry_time = pd.Timestamp(row["snapshot_time"])
        _settle_until(entry_time)

        locked_stake = sum(float(p["stake"]) for p in pending_resolutions)
        wallet_before = cash + locked_stake

        if pd.notna(row.get("fixed_stake_usd")):
            desired_stake = float(row["fixed_stake_usd"])
            sizing_method = "fixed_usd"
        else:
            desired_stake = float(wallet_before * row["bet_fraction"])
            sizing_method = "fractional_kelly"

        stake = max(0.0, min(desired_stake, cash))
        if stake <= 0:
            continue

        cash_before = cash
        cash -= stake
        side = str(row["trade_side"])
        market_price = float(row["market_price"])
        outcome_yes = int(round(float(row["outcome_yes"])))
        if side == "YES":
            settle_value = stake / market_price if outcome_yes == 1 else 0.0
            contract_price = market_price
        else:
            contract_price = 1.0 - market_price
            settle_value = stake / contract_price if outcome_yes == 0 else 0.0
        pnl = settle_value - stake

        position = {
            "market_id": row.get("market_id", np.nan),
            "entry_time": entry_time,
            "end_time": pd.Timestamp(row["end_time"]),
            "closed_time": pd.Timestamp(row["closed_time"]) if pd.notna(row.get("closed_time")) else pd.NaT,
            "resolution_time": pd.Timestamp(row["resolution_time"]),
            "side": side,
            "stake": float(stake),
            "settle_value": float(settle_value),
            "pnl": float(pnl),
            "market_price": market_price,
            "model_prob": float(row["model_prob"]),
            "edge": float(row["edge"]),
            "kelly_raw": float(row.get("kelly_raw", np.nan)),
            "bet_fraction": float(row.get("bet_fraction", np.nan)) if pd.notna(row.get("bet_fraction")) else np.nan,
            "sizing_method": sizing_method,
            "cash_before_entry": float(cash_before),
            "wallet_before_entry": float(wallet_before),
            "outcome_yes": outcome_yes,
        }
        trade_logs.append(position)
        pending_resolutions.append(position)
        equity_logs.append(
            {
                "event_time": entry_time,
                "event_type": "entry",
                "wallet_value": cash + sum(float(p["stake"]) for p in pending_resolutions),
                "cash": cash,
                "locked_stake": sum(float(p["stake"]) for p in pending_resolutions),
            }
        )

    _settle_until(pd.Timestamp.max.tz_localize("UTC"))
    trade_df = pd.DataFrame(trade_logs)
    if not trade_df.empty:
        trade_df["roi_on_stake"] = trade_df["pnl"] / trade_df["stake"]
    equity_df = pd.DataFrame(equity_logs).sort_values(["event_time", "event_type"], na_position="first").reset_index(
        drop=True
    )
    return trade_df, equity_df


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_peak = equity.cummax()
    drawdown = (equity / running_peak) - 1.0
    return float(drawdown.min())


def _prediction_metrics(y_true: pd.Series, y_prob: pd.Series) -> dict[str, float]:
    err = y_prob - y_true
    mae = float(np.mean(np.abs(err)))
    brier = float(np.mean(err**2))
    return {"mae": mae, "brier": brier}


def run_benchmark(config: StrategyConfig) -> dict[str, object]:
    config.ensure_output_dir()
    frame, coverage = load_strategy_frame(config)
    train_df, test_df = split_train_test(frame, config.train_ratio)
    coverage["rows_train"] = int(len(train_df))
    coverage["rows_test"] = int(len(test_df))

    if train_df.empty or test_df.empty:
        summary = {
            "status": "insufficient_data",
            "reason": "Train/test split left an empty partition.",
            "coverage": coverage,
            "config": _serialize_config(config),
        }
        _write_outputs(config, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), summary)
        return summary

    iso_model = fit_isotonic(train_df)
    test_eval = test_df.copy()
    test_eval["resolution_time"] = test_eval["closed_time"].where(test_eval["closed_time"].notna(), test_eval["end_time"])
    test_eval["calibrated_prob"] = iso_model.predict(test_eval["price_last_trade"].clip(0, 1).astype(float))
    test_eval["raw_prob"] = test_eval["price_last_trade"].clip(0, 1).astype(float)

    calibrated_signals = _build_signals(
        test_eval,
        model_prob_col="calibrated_prob",
        entry_threshold=config.entry_threshold,
        kelly_fraction=config.kelly_fraction,
        max_bet_fraction=config.max_bet_fraction,
    )
    raw_signals = _build_signals(
        test_eval,
        model_prob_col="raw_prob",
        entry_threshold=config.entry_threshold,
        kelly_fraction=config.kelly_fraction,
        max_bet_fraction=config.max_bet_fraction,
    )
    fixed_stake_signals = _build_signals(
        test_eval,
        model_prob_col="calibrated_prob",
        entry_threshold=config.entry_threshold,
        kelly_fraction=config.kelly_fraction,
        max_bet_fraction=config.max_bet_fraction,
        fixed_stake_usd=config.fixed_stake_usd,
    )

    trade_log_df, equity_curve_df = _simulate_wallet(calibrated_signals, starting_bankroll=config.starting_bankroll)
    fixed_trade_df, fixed_equity_df = _simulate_wallet(fixed_stake_signals, starting_bankroll=config.starting_bankroll)
    raw_trade_df, _ = _simulate_wallet(raw_signals, starting_bankroll=config.starting_bankroll)

    final_wallet = float(equity_curve_df["wallet_value"].iloc[-1]) if not equity_curve_df.empty else config.starting_bankroll
    fixed_final_wallet = float(fixed_equity_df["wallet_value"].iloc[-1]) if not fixed_equity_df.empty else config.starting_bankroll
    max_dd = _max_drawdown(equity_curve_df["wallet_value"]) if not equity_curve_df.empty else 0.0

    calibrated_metrics = _prediction_metrics(test_eval["outcome_yes"].astype(float), test_eval["calibrated_prob"].astype(float))
    raw_metrics = _prediction_metrics(test_eval["outcome_yes"].astype(float), test_eval["raw_prob"].astype(float))

    summary = {
        "status": "ok",
        "config": _serialize_config(config),
        "coverage": coverage,
        "results": {
            "strategy": {
                "trades": int(len(trade_log_df)),
                "hit_rate": float((trade_log_df["pnl"] > 0).mean()) if not trade_log_df.empty else 0.0,
                "avg_edge": float(trade_log_df["edge"].mean()) if not trade_log_df.empty else 0.0,
                "total_pnl": float(final_wallet - config.starting_bankroll),
                "final_wallet": final_wallet,
                "total_return": float((final_wallet / config.starting_bankroll) - 1.0),
                "max_drawdown": max_dd,
            },
            "baseline_no_trade": {
                "final_wallet": float(config.starting_bankroll),
                "total_return": 0.0,
            },
            "baseline_fixed_stake": {
                "trades": int(len(fixed_trade_df)),
                "final_wallet": float(fixed_final_wallet),
                "total_return": float((fixed_final_wallet / config.starting_bankroll) - 1.0),
            },
            "baseline_raw_market_prob_strategy": {
                "trades": int(len(raw_trade_df)),
                "note": "Same edge-threshold rule using raw market probability as model probability.",
            },
            "prediction_quality_test": {
                "calibrated": calibrated_metrics,
                "raw_market_prob": raw_metrics,
            },
        },
    }

    _write_outputs(config, trade_log_df, equity_curve_df, test_eval, summary)
    return summary


def _serialize_config(config: StrategyConfig) -> dict[str, object]:
    data = asdict(config)
    data["snapshots_csv"] = str(config.snapshots_csv)
    data["output_dir"] = str(config.output_dir)
    return data


def _write_outputs(
    config: StrategyConfig,
    trade_log_df: pd.DataFrame,
    equity_curve_df: pd.DataFrame,
    test_eval_df: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    trade_log_df.to_csv(config.output_dir / "sports_trade_log.csv", index=False)
    equity_curve_df.to_csv(config.output_dir / "sports_equity_curve.csv", index=False)
    test_eval_df.to_csv(config.output_dir / "sports_test_predictions.csv", index=False)
    summary_json = config.output_dir / "sports_strategy_summary.json"
    summary_csv = config.output_dir / "sports_strategy_summary.csv"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    flat_summary = pd.json_normalize(summary, sep=".")
    flat_summary.to_csv(summary_csv, index=False)

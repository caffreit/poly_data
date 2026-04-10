from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.strategy_backtest import StrategyConfig, run_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sports strategy benchmark backtest")
    parser.add_argument(
        "--snapshots-csv",
        type=Path,
        default=Path("output") / "market_snapshots.csv",
        help="Path to market snapshot CSV from the pipeline",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output") / "strategy",
        help="Directory for benchmark outputs",
    )
    parser.add_argument("--category", type=str, default="Sports")
    parser.add_argument("--horizon-min", type=int, default=120)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--starting-bankroll", type=float, default=1000.0)
    parser.add_argument("--entry-threshold", type=float, default=0.02)
    parser.add_argument("--kelly-fraction", type=float, default=0.25)
    parser.add_argument("--max-bet-fraction", type=float, default=0.10)
    parser.add_argument("--fixed-stake-usd", type=float, default=25.0)
    parser.add_argument("--min-price", type=float, default=0.01)
    parser.add_argument("--max-price", type=float, default=0.99)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = StrategyConfig(
        snapshots_csv=args.snapshots_csv,
        output_dir=args.output_dir,
        category=args.category,
        horizon_min=args.horizon_min,
        train_ratio=args.train_ratio,
        starting_bankroll=args.starting_bankroll,
        entry_threshold=args.entry_threshold,
        kelly_fraction=args.kelly_fraction,
        max_bet_fraction=args.max_bet_fraction,
        fixed_stake_usd=args.fixed_stake_usd,
        min_price=args.min_price,
        max_price=args.max_price,
    )
    summary = run_benchmark(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

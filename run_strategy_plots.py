from __future__ import annotations

import argparse
from pathlib import Path

from src.strategy_plots import save_strategy_plots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate plots from sports strategy outputs")
    parser.add_argument(
        "--strategy-dir",
        type=Path,
        default=Path("output") / "strategy",
        help="Directory containing sports strategy CSV outputs",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory for generated plots (default: <strategy-dir>/plots)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = save_strategy_plots(strategy_dir=args.strategy_dir, plots_dir=args.plots_dir)
    for path in paths:
        print(path)
    print(f"Generated {len(paths)} strategy plots.")


if __name__ == "__main__":
    main()

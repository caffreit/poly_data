from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# 10 / 30 min, then hourly from 1h through 24h before close, plus 48h.
DEFAULT_HORIZONS_MINUTES: Tuple[int, ...] = (10, 30) + tuple(range(60, 1441, 60)) + (2880,)


@dataclass(frozen=True)
class PipelineConfig:
    lookback_days: int = 90
    target_markets: int = 30000
    # Minimum Gamma volumeNum (same units as API). Use 0 to disable.
    min_volume: float = 50_000.0
    horizons_minutes: Tuple[int, ...] = DEFAULT_HORIZONS_MINUTES
    gamma_page_size: int = 200
    # Enough pages to reach target_markets after filters (page size 200; early exit at 2x target).
    gamma_max_pages: int = 200
    clob_history_interval: str = "max"
    clob_history_fidelity: int | None = None
    request_timeout_seconds: float = 25.0
    max_retries: int = 4
    backoff_base_seconds: float = 0.4
    output_dir: Path = field(default_factory=lambda: Path("output"))
    plots_dir: Path = field(default_factory=lambda: Path("output") / "plots")

    def ensure_output_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

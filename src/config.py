from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class PipelineConfig:
    lookback_days: int = 30
    target_markets: int = 100
    horizons_minutes: Tuple[int, ...] = (10, 30, 60, 120, 240, 360)
    gamma_page_size: int = 200
    gamma_max_pages: int = 50
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

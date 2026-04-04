# AGENTS.md

## Cursor Cloud specific instructions

### Overview

This is a Python data pipeline that analyzes Polymarket prediction market mispricing. It fetches closed binary markets from public Polymarket APIs, computes calibration/error metrics, and generates interactive Plotly charts.

### Running the pipeline

```
python3 run_pipeline.py --target-markets 5 --lookback-days 30 --horizons "60,1440"
```

Use small `--target-markets` values (5–20) for quick dev iterations; the default of 10000 takes a long time due to API calls. Outputs go to `output/` (CSVs) and `output/plots/` (HTML + PNG charts).

### Key caveats

- **No local services required** — no database, Docker, or web server. The pipeline is a batch script that exits after writing outputs.
- **No secrets or auth needed** — the Polymarket Gamma and CLOB APIs are public.
- **No automated test suite exists** — verify correctness by running the pipeline with a small target count and inspecting outputs.
- **Linting**: no linter config is present in the repo. Use `python3 -m py_compile <file>` for syntax checks.
- **kaleido PNG export** requires a Chromium binary. `kaleido_get_chrome` or `choreo_get_chrome` (from `~/.local/bin`) fetches it on first use. Failures in PNG export are silently caught; HTML plots are always generated.
- **Rate limiting**: the Polymarket APIs return HTTP 429 under load; the client retries with exponential backoff automatically.

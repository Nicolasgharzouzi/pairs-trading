# Pairs Trading

Statistical arbitrage on cointegrated stock pairs: mean-reversion on the OLS spread with z-score entry/exit, a Hurst regime filter, and walk-forward out-of-sample validation.

## Final portfolio

| Pair | Entry | Exit | Role |
|------|-------|------|------|
| GS / MS | ±2.5 | ±0.0 | Best OOS performer |
| MOH / UNH | ±1.5 | ±0.5 | Diversifier (healthcare) |

BAC/PNC and VLO/XOM were screened and tested but dropped after weak walk-forward results.

## Setup

```bash
pip install -r requirements.txt
```

## Pipeline

```bash
python data.py          # screen pairs → pairs.csv
python optimise.py      # grid-search entry/exit z per pair
python backtest.py      # in-sample backtest → results_*.csv
python walkforward.py   # per-pair out-of-sample validation
python portfolio.py     # equal-weight portfolio OOS (final result)
python tearsheet.py     # per-pair performance charts
python metrics.py       # summary table from results_*.csv
```

## Strategy (short)

1. Fit hedge ratio via OLS (train window only in walk-forward)
2. Spread = stock₁ − β × stock₂; 30-day z-score
3. Enter at ±entry_z, exit at ±exit_z
4. Skip new entries when rolling Hurst ≥ 0.55 (trending regime)
5. $10k capital, 0.15% cost per trade

## Config

All settings live in `config.py` (`PAIRS`, `CAPITAL`, `USE_REGIME`, windows, thresholds).

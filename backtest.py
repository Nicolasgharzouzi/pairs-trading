import pandas as pd
import numpy as np
from data import download_prices
from signals import compute_spread, compute_zscore, generate_signals

PAIRS = [
    ('BAC', 'PNC', 2.0, 0.0),
    ('GS',  'MS',  2.5, 0.0),
]

CAPITAL      = 10_000
TOTAL_COST   = 0.0015


def backtest_pair(prices, pair):
    s1_name, s2_name, entry_z, exit_z = pair
    s1, s2 = prices[s1_name], prices[s2_name]

    spread, hedge_ratio = compute_spread(s1, s2)
    zscore             = compute_zscore(spread)
    signal             = generate_signals(zscore, entry_z=entry_z, exit_z=exit_z)

    r1 = s1.pct_change()
    r2 = s2.pct_change()

    strategy_returns = signal.shift(1) * (r1 - hedge_ratio * r2)
    trade_entries    = signal.diff().abs() > 0
    strategy_returns[trade_entries] -= TOTAL_COST

    pnl            = strategy_returns * CAPITAL
    cumulative_pnl = pnl.cumsum()

    results = pd.DataFrame({
        'signal':         signal,
        'zscore':         zscore,
        'daily_pnl':      pnl,
        'cumulative_pnl': cumulative_pnl,
    }).dropna()

    print(f"\n=== {s1_name} / {s2_name} ===")
    print(f"  Entry z     : ±{entry_z}")
    print(f"  Exit z      : ±{exit_z}")
    print(f"  Total trades: {int((signal.diff().abs() > 0).sum())}")
    print(f"  Total P&L   : ${cumulative_pnl.iloc[-1]:,.2f}")
    print(f"  Best day    : ${pnl.max():,.2f}")
    print(f"  Worst day   : ${pnl.min():,.2f}")

    return results


if __name__ == "__main__":
    prices      = download_prices()
    all_results = {}

    for pair in PAIRS:
        results = backtest_pair(prices, pair)
        all_results[f"{pair[0]}_{pair[1]}"] = results

    for name, df in all_results.items():
        df.to_csv(f"results_{name}.csv")
        print(f"\nSaved: results_{name}.csv")
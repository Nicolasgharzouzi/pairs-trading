import pandas as pd
import numpy as np
from data import download_prices
from signals import compute_spread, compute_zscore, generate_signals

PAIRS = [
    ('MS',  'WFC'),
    ('KO',  'PEP'),
]

CAPITAL        = 10_000   # dollars allocated per pair
COST_BPS       = 10       # transaction cost: 10 basis points per trade (0.1%)
SLIPPAGE_BPS   = 5        # slippage: 5 basis points per trade (0.05%)
TOTAL_COST     = (COST_BPS + SLIPPAGE_BPS) / 10_000  # combined cost per trade


def backtest_pair(prices, pair):
    s1_name, s2_name = pair
    s1, s2 = prices[s1_name], prices[s2_name]

    spread, hedge_ratio = compute_spread(s1, s2)
    zscore             = compute_zscore(spread)
    signal             = generate_signals(zscore)

    # Daily returns of each stock
    r1 = s1.pct_change()
    r2 = s2.pct_change()

    # Strategy return:
    # When signal = +1: long s1, short s2
    # When signal = -1: short s1, long s2
    # Hedge ratio keeps the position dollar-neutral
    strategy_returns = signal.shift(1) * (r1 - hedge_ratio * r2)

    # Detect trade entries (signal changes)
    trade_entries = signal.diff().abs() > 0

    # Subtract transaction costs on every entry/exit
    strategy_returns[trade_entries] -= TOTAL_COST

    # Dollar P&L
    pnl = strategy_returns * CAPITAL

    # Cumulative P&L
    cumulative_pnl = pnl.cumsum()

    # Build results DataFrame
    results = pd.DataFrame({
        'signal':         signal,
        'zscore':         zscore,
        'daily_pnl':      pnl,
        'cumulative_pnl': cumulative_pnl,
    }).dropna()

    print(f"\n=== {s1_name} / {s2_name} ===")
    print(f"  Total trades   : {trade_entries.sum()}")
    print(f"  Total P&L      : ${cumulative_pnl.iloc[-1]:,.2f}")
    print(f"  Best day       : ${pnl.max():,.2f}")
    print(f"  Worst day      : ${pnl.min():,.2f}")

    return results


if __name__ == "__main__":
    prices  = download_prices()
    all_results = {}

    for pair in PAIRS:
        results = backtest_pair(prices, pair)
        all_results[f"{pair[0]}_{pair[1]}"] = results

    # Save results to CSV for use in metrics
    for name, df in all_results.items():
        df.to_csv(f"results_{name}.csv")
        print(f"\nSaved: results_{name}.csv")
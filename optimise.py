import pandas as pd
import numpy as np
from data import download_prices
from signals import compute_spread, compute_zscore, generate_signals
from metrics import sharpe_ratio

PAIRS = [
    ('BAC', 'PNC'),
    ('GS',  'MS'),
]

CAPITAL = 10_000
TOTAL_COST = 0.0015

ENTRY_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]
EXIT_THRESHOLDS  = [0.0, 0.25, 0.5, 0.75, 1.0]


def backtest_with_params(prices, pair, entry_z, exit_z):
    s1_name, s2_name = pair
    s1, s2 = prices[s1_name], prices[s2_name]

    spread, hedge_ratio = compute_spread(s1, s2)
    zscore             = compute_zscore(spread)
    signal             = generate_signals(zscore, entry_z=entry_z, exit_z=exit_z)

    r1 = s1.pct_change()
    r2 = s2.pct_change()

    strategy_returns = signal.shift(1) * (r1 - hedge_ratio * r2)
    trade_entries    = signal.diff().abs() > 0
    strategy_returns[trade_entries] -= TOTAL_COST

    daily_returns = strategy_returns.dropna()
    return sharpe_ratio(daily_returns)


def optimise_pair(prices, pair):
    print(f"\nOptimising {pair[0]}/{pair[1]}...")
    results = []

    for entry_z in ENTRY_THRESHOLDS:
        for exit_z in EXIT_THRESHOLDS:
            if exit_z >= entry_z:
                continue
            sharpe = backtest_with_params(prices, pair, entry_z, exit_z)
            results.append({
                'entry_z': entry_z,
                'exit_z':  exit_z,
                'sharpe':  round(sharpe, 3)
            })

    df = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    print(df.to_string(index=False))

    best = df.iloc[0]
    print(f"\n  Best: entry=±{best['entry_z']} exit=±{best['exit_z']} Sharpe={best['sharpe']}")
    return best


if __name__ == "__main__":
    prices = download_prices()

    for pair in PAIRS:
        optimise_pair(prices, pair)
import pandas as pd

from config import (
    CAPITAL,
    ENTRY_THRESHOLDS,
    EXIT_THRESHOLDS,
    PAIRS,
)
from data import download_prices
from engine import run_backtest
from metrics import sharpe_ratio


def backtest_with_params(prices, pair, entry_z, exit_z):
    s1_name, s2_name = pair
    s1, s2 = prices[s1_name], prices[s2_name]

    result = run_backtest(s1, s2, entry_z, exit_z)
    return sharpe_ratio(result.strategy_returns.dropna())


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
                'sharpe':  round(sharpe, 3),
            })

    df = pd.DataFrame(results).sort_values('sharpe', ascending=False)
    print(df.to_string(index=False))

    best = df.iloc[0]
    print(f"\n  Best: entry=±{best['entry_z']} exit=±{best['exit_z']} Sharpe={best['sharpe']}")
    return best


if __name__ == "__main__":
    prices = download_prices()

    for pair in PAIRS:
        optimise_pair(prices, (pair[0], pair[1]))

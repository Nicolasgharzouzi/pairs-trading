import pandas as pd
import numpy as np

from config import (
    CAPITAL,
    ENTRY_THRESHOLDS,
    EXIT_THRESHOLDS,
    PAIRS,
    TEST_DAYS,
    TRAIN_DAYS,
)
from data import download_prices
from engine import fit_hedge_ratio, run_backtest
from metrics import max_drawdown, sharpe_ratio, win_rate


def run_window(s1, s2, hedge_ratio, entry_z, exit_z):
    """Run backtest on a price window with given parameters."""
    result = run_backtest(s1, s2, entry_z, exit_z, hedge_ratio=hedge_ratio)
    return result.daily_pnl.dropna()


def optimise_on_train(s1_train, s2_train, hedge_ratio):
    """
    Find best entry/exit thresholds using only training data.
    This is the key fix — thresholds are chosen before seeing test data.
    """
    best_sharpe = -np.inf
    best_entry  = 2.0
    best_exit   = 0.0

    for entry_z in ENTRY_THRESHOLDS:
        for exit_z in EXIT_THRESHOLDS:
            if exit_z >= entry_z:
                continue

            daily_pnl = run_window(s1_train, s2_train, hedge_ratio, entry_z, exit_z)

            if len(daily_pnl) == 0:
                continue

            sharpe = sharpe_ratio(daily_pnl / CAPITAL)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_entry  = entry_z
                best_exit   = exit_z

    return best_entry, best_exit


def walk_forward(prices, pair):
    s1_name, s2_name = pair
    print(f"\n=== Walk-Forward: {s1_name}/{s2_name} ===")

    s1       = prices[s1_name]
    s2       = prices[s2_name]
    all_dates = prices.index
    n         = len(all_dates)

    windows  = []
    all_pnl  = []
    start    = 0
    window_num = 1

    while start + TRAIN_DAYS + TEST_DAYS <= n:
        train_start = all_dates[start]
        train_end   = all_dates[start + TRAIN_DAYS - 1]
        test_start  = all_dates[start + TRAIN_DAYS]
        test_end    = all_dates[min(start + TRAIN_DAYS + TEST_DAYS - 1, n - 1)]

        # Training data only
        s1_train = s1[train_start:train_end]
        s2_train = s2[train_start:train_end]

        # Step 1: fit hedge ratio on training data
        hedge_ratio = fit_hedge_ratio(s1_train, s2_train)

        # Step 2: optimise thresholds on training data
        best_entry, best_exit = optimise_on_train(s1_train, s2_train, hedge_ratio)

        # Step 3: apply to test data — never seen before
        s1_test = s1[test_start:test_end]
        s2_test = s2[test_start:test_end]

        daily_pnl = run_window(s1_test, s2_test, hedge_ratio, best_entry, best_exit)

        period_sharpe = sharpe_ratio(daily_pnl / CAPITAL)
        period_pnl    = daily_pnl.cumsum().iloc[-1] if len(daily_pnl) > 0 else 0

        print(f"  Window {window_num}: "
              f"test {test_start.date()} to {test_end.date()} | "
              f"entry=±{best_entry} exit=±{best_exit} | "
              f"P&L: ${period_pnl:,.0f} | Sharpe: {period_sharpe:.3f}")

        windows.append({
            'window':     window_num,
            'test_start': test_start,
            'test_end':   test_end,
            'entry_z':    best_entry,
            'exit_z':     best_exit,
            'pnl':        period_pnl,
            'sharpe':     period_sharpe,
        })

        all_pnl.append(daily_pnl)
        start      += TEST_DAYS
        window_num += 1

    # Combine all out-of-sample windows
    combined_pnl   = pd.concat(all_pnl)
    total_pnl      = combined_pnl.cumsum().iloc[-1]
    overall_sharpe = sharpe_ratio(combined_pnl / CAPITAL)
    overall_wr     = win_rate(combined_pnl)
    overall_dd     = max_drawdown(combined_pnl.cumsum())

    print(f"\n  Overall out-of-sample results:")
    print(f"  Total P&L    : ${total_pnl:,.2f}")
    print(f"  Sharpe Ratio : {overall_sharpe:.3f}")
    print(f"  Max Drawdown : ${overall_dd:,.2f}")
    print(f"  Win Rate     : {overall_wr:.1f}%")

    return pd.DataFrame(windows), combined_pnl


if __name__ == "__main__":
    prices = download_prices()

    for pair in PAIRS:
        walk_forward(prices, (pair[0], pair[1]))
import pandas as pd
import numpy as np
from data import download_prices
from signals import compute_spread, compute_zscore, generate_signals
from metrics import sharpe_ratio, max_drawdown, win_rate

# Best parameters from optimisation
PAIRS = [
    ('BAC', 'PNC', 2.0, 0.0),
    ('GS',  'MS',  2.5, 0.0),
]

CAPITAL    = 10_000
TOTAL_COST = 0.0015

# Walk-forward settings
TRAIN_DAYS = 504   # 2 years of training data
TEST_DAYS  = 126   # 6 months of test data


def backtest_window(prices, pair, train_start, train_end, test_start, test_end):
    """
    Train on one window, test on the next.
    The model never sees the test data during training.
    """
    s1_name, s2_name, entry_z, exit_z = pair
    s1, s2 = prices[s1_name], prices[s2_name]

    # Training window — used to fit the hedge ratio only
    s1_train = s1[train_start:train_end]
    s2_train = s2[train_start:train_end]

    # Fit hedge ratio on training data only
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    model       = OLS(s1_train, add_constant(s2_train)).fit()
    hedge_ratio = model.params.iloc[1]

    # Test window — never seen during training
    s1_test = s1[test_start:test_end]
    s2_test = s2[test_start:test_end]

    # Compute spread and signals on test data using training hedge ratio
    spread = s1_test - hedge_ratio * s2_test
    zscore = compute_zscore(spread)
    signal = generate_signals(zscore, entry_z=entry_z, exit_z=exit_z)

    # Calculate returns
    r1 = s1_test.pct_change()
    r2 = s2_test.pct_change()

    strategy_returns = signal.shift(1) * (r1 - hedge_ratio * r2)
    trade_entries    = signal.diff().abs() > 0
    strategy_returns[trade_entries] -= TOTAL_COST

    daily_pnl      = strategy_returns.dropna() * CAPITAL
    cumulative_pnl = daily_pnl.cumsum()

    return daily_pnl, cumulative_pnl


def walk_forward(prices, pair):
    s1_name, s2_name, entry_z, exit_z = pair
    print(f"\n=== Walk-Forward: {s1_name}/{s2_name} ===")

    all_dates  = prices.index
    n          = len(all_dates)
    windows    = []
    all_pnl    = []

    start = 0
    window_num = 1

    while start + TRAIN_DAYS + TEST_DAYS <= n:
        train_start = all_dates[start]
        train_end   = all_dates[start + TRAIN_DAYS - 1]
        test_start  = all_dates[start + TRAIN_DAYS]
        test_end    = all_dates[min(start + TRAIN_DAYS + TEST_DAYS - 1, n - 1)]

        daily_pnl, cumulative_pnl = backtest_window(
            prices, pair, train_start, train_end, test_start, test_end
        )

        period_sharpe = sharpe_ratio(daily_pnl / CAPITAL)
        period_pnl    = cumulative_pnl.iloc[-1] if len(cumulative_pnl) > 0 else 0

        print(f"  Window {window_num}: train {train_start.date()} to {train_end.date()} "
              f"| test {test_start.date()} to {test_end.date()} "
              f"| P&L: ${period_pnl:,.0f} | Sharpe: {period_sharpe:.3f}")

        windows.append({
            'window':      window_num,
            'test_start':  test_start,
            'test_end':    test_end,
            'pnl':         period_pnl,
            'sharpe':      period_sharpe,
        })

        all_pnl.append(daily_pnl)
        start      += TEST_DAYS
        window_num += 1

    # Combine all test windows
    combined_pnl    = pd.concat(all_pnl)
    total_pnl       = combined_pnl.cumsum().iloc[-1]
    overall_sharpe  = sharpe_ratio(combined_pnl / CAPITAL)
    overall_wr      = win_rate(combined_pnl)
    overall_dd      = max_drawdown(combined_pnl.cumsum())

    print(f"\n  Overall out-of-sample results:")
    print(f"  Total P&L    : ${total_pnl:,.2f}")
    print(f"  Sharpe Ratio : {overall_sharpe:.3f}")
    print(f"  Max Drawdown : ${overall_dd:,.2f}")
    print(f"  Win Rate     : {overall_wr:.1f}%")

    return pd.DataFrame(windows), combined_pnl


if __name__ == "__main__":
    prices = download_prices()

    for pair in PAIRS:
        walk_forward(prices, pair)
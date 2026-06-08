import pandas as pd

from config import CAPITAL, PAIRS, TEST_DAYS, TRAIN_DAYS
from data import download_prices
from engine import fit_hedge_ratio, run_backtest
from metrics import max_drawdown, sharpe_ratio, win_rate


def backtest_window(prices, pair, train_start, train_end, test_start, test_end):
    """
    Train on one window, test on the next.
    The model never sees the test data during training.
    """
    s1_name, s2_name, entry_z, exit_z = pair
    s1, s2 = prices[s1_name], prices[s2_name]

    s1_train = s1[train_start:train_end]
    s2_train = s2[train_start:train_end]
    hedge_ratio = fit_hedge_ratio(s1_train, s2_train)

    s1_test = s1[test_start:test_end]
    s2_test = s2[test_start:test_end]

    result = run_backtest(
        s1_test, s2_test, entry_z, exit_z, hedge_ratio=hedge_ratio
    )
    return result.daily_pnl.dropna(), result.cumulative_pnl.dropna()


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

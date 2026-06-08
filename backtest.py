from data import download_prices
from config import PAIRS
from engine import run_backtest_pair


def backtest_pair(prices, pair):
    result = run_backtest_pair(prices, pair)

    print(f"\n=== {result.stock_1} / {result.stock_2} ===")
    print(f"  Entry z     : ±{result.entry_z}")
    print(f"  Exit z      : ±{result.exit_z}")
    print(f"  Total trades: {result.total_trades}")
    print(f"  Total P&L   : ${result.cumulative_pnl.iloc[-1]:,.2f}")
    print(f"  Best day    : ${result.daily_pnl.max():,.2f}")
    print(f"  Worst day   : ${result.daily_pnl.min():,.2f}")

    return result.to_dataframe()


if __name__ == "__main__":
    prices      = download_prices()
    all_results = {}

    for pair in PAIRS:
        results = backtest_pair(prices, pair)
        all_results[f"{pair[0]}_{pair[1]}"] = results

    for name, df in all_results.items():
        df.to_csv(f"results_{name}.csv")
        print(f"\nSaved: results_{name}.csv")

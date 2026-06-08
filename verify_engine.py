"""
Sanity check: engine output should match the saved backtest CSVs.
Run after any change to engine.py or config.py.
"""
import pandas as pd

from config import PAIRS
from data import download_prices
from engine import run_backtest_pair

# yfinance can return slightly different prices between downloads
TOLERANCE = 1.0


def verify_pair(prices, pair):
    name = f"{pair[0]}_{pair[1]}"
    csv_path = f"results_{name}.csv"

    result = run_backtest_pair(prices, pair)
    engine_df = result.to_dataframe()

    try:
        saved_df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    except FileNotFoundError:
        print(f"  SKIP {name}: no {csv_path} — run backtest.py first")
        return None

    merged = engine_df.join(saved_df, lsuffix='_engine', rsuffix='_saved', how='inner')
    pnl_diff = (merged['cumulative_pnl_engine'] - merged['cumulative_pnl_saved']).abs().max()

    if pnl_diff <= TOLERANCE:
        print(f"  PASS {name}: cumulative P&L matches (max diff ${pnl_diff:.4f})")
        return True

    print(f"  FAIL {name}: cumulative P&L diverged by ${pnl_diff:.4f}")
    return False


def verify_config_wired(prices):
    """Changing CAPITAL in config should scale P&L linearly."""
    import config
    pair = PAIRS[0]

    original = config.CAPITAL
    config.CAPITAL = original * 2
    try:
        high = run_backtest_pair(prices, pair).cumulative_pnl.iloc[-1]
        config.CAPITAL = original
        low = run_backtest_pair(prices, pair).cumulative_pnl.iloc[-1]
    finally:
        config.CAPITAL = original

    ratio = high / low if low != 0 else 0
    if abs(ratio - 2.0) < 0.001:
        print(f"  PASS config: doubling CAPITAL scales P&L 2x ({low:,.2f} -> {high:,.2f})")
        return True

    print(f"  FAIL config: expected 2x P&L scaling, got {ratio:.4f}x")
    return False


if __name__ == "__main__":
    print("Verifying shared engine...\n")
    prices = download_prices()

    print("1. Config wiring")
    config_ok = verify_config_wired(prices)

    print("\n2. Engine vs saved CSVs (run backtest.py first)")
    outcomes = [verify_pair(prices, pair) for pair in PAIRS]
    checked = [o for o in outcomes if o is not None]

    if not checked:
        print("  SKIP: no saved results. Run: python backtest.py")
        csv_ok = None
    else:
        csv_ok = all(checked)

    print()
    if config_ok and (csv_ok is None or csv_ok):
        print("All checks passed.")
    else:
        print("Some checks failed — review engine.py / config.py.")

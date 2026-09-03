"""
Final deliverable: walk-forward portfolio across all PAIRS.

Each pair gets an equal share of CAPITAL. Daily P&L is summed into one
out-of-sample equity curve — the honest portfolio result.
"""
import pandas as pd
import matplotlib.pyplot as plt

from config import CAPITAL, PAIRS, USE_REGIME
from data import download_prices
from metrics import max_drawdown, profit_factor, sharpe_ratio, win_rate
from walkforward import walk_forward


def run_portfolio(prices):
    n_pairs = len(PAIRS)
    capital_per_pair = CAPITAL / n_pairs
    scale = capital_per_pair / CAPITAL  # walk_forward uses full CAPITAL

    print(f"\n=== Portfolio Walk-Forward ===")
    print(f"  Pairs          : {n_pairs}")
    print(f"  Total capital  : ${CAPITAL:,}")
    print(f"  Capital / pair : ${capital_per_pair:,.0f}")
    print(f"  Regime filter  : {USE_REGIME}")

    pair_summaries = []
    portfolio_pnl = None

    for pair in PAIRS:
        name = f"{pair[0]}/{pair[1]}"
        _, combined_pnl = walk_forward(prices, (pair[0], pair[1]))

        # Scale so total book = CAPITAL (equal weight)
        pair_pnl = combined_pnl * scale

        if portfolio_pnl is None:
            portfolio_pnl = pair_pnl.copy()
        else:
            portfolio_pnl = portfolio_pnl.add(pair_pnl, fill_value=0)

        pair_summaries.append({
            'Pair':   name,
            'P&L':    pair_pnl.cumsum().iloc[-1],
            'Sharpe': sharpe_ratio(pair_pnl / capital_per_pair),
            'Max DD': max_drawdown(pair_pnl.cumsum()),
        })

    cumulative = portfolio_pnl.cumsum()
    total_pnl  = cumulative.iloc[-1]
    sharpe     = sharpe_ratio(portfolio_pnl / CAPITAL)
    mdd        = max_drawdown(cumulative)
    wr         = win_rate(portfolio_pnl)
    pf         = profit_factor(portfolio_pnl)

    print("\n=== Pair Contributions (equal-weight) ===")
    summary = pd.DataFrame(pair_summaries).set_index('Pair')
    print(summary.to_string(
        formatters={
            'P&L':    lambda x: f"${x:,.0f}",
            'Sharpe': lambda x: f"{x:.3f}",
            'Max DD': lambda x: f"${x:,.0f}",
        }
    ))

    print("\n=== PORTFOLIO OUT-OF-SAMPLE RESULTS ===")
    print(f"  Total P&L      : ${total_pnl:,.2f}")
    print(f"  Total Return   : {total_pnl / CAPITAL * 100:.1f}%")
    print(f"  Sharpe Ratio   : {sharpe:.3f}")
    print(f"  Max Drawdown   : ${mdd:,.2f}")
    print(f"  Win Rate       : {wr:.1f}%")
    print(f"  Profit Factor  : {pf:.3f}")

    results = pd.DataFrame({
        'daily_pnl':      portfolio_pnl,
        'cumulative_pnl': cumulative,
    })
    results.to_csv("results_portfolio.csv")
    print("\nSaved: results_portfolio.csv")

    # Equity curve
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(cumulative.index, cumulative.values, color='#2196f3', linewidth=1.5)
    ax.fill_between(cumulative.index, 0, cumulative.values,
                     where=cumulative.values >= 0, alpha=0.15, color='#00e676')
    ax.fill_between(cumulative.index, 0, cumulative.values,
                     where=cumulative.values < 0, alpha=0.15, color='#ff1744')
    ax.axhline(0, color='#555555', linewidth=0.8, linestyle='--')
    ax.set_title(
        f"Portfolio OOS Equity — {n_pairs} pairs | "
        f"P&L ${total_pnl:,.0f} | Sharpe {sharpe:.2f}"
    )
    ax.set_ylabel("Cumulative P&L ($)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("portfolio_equity.png", dpi=150)
    plt.close()
    print("Saved: portfolio_equity.png")

    return results, summary


if __name__ == "__main__":
    prices = download_prices()
    run_portfolio(prices)

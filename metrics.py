import pandas as pd
import numpy as np

TRADING_DAYS = 252


def sharpe_ratio(daily_returns):
    """
    Risk-adjusted return.
    Above 1.0 acceptable, above 2.0 strong.
    """
    if daily_returns.std() == 0:
        return 0
    return (daily_returns.mean() / daily_returns.std()) * np.sqrt(TRADING_DAYS)


def max_drawdown(cumulative_pnl):
    """
    Largest peak-to-trough loss.
    How much could you have lost at the worst point?
    """
    rolling_max = cumulative_pnl.cummax()
    drawdown    = cumulative_pnl - rolling_max
    return drawdown.min()


def calmar_ratio(daily_returns, cumulative_pnl):
    """
    Annualised return divided by max drawdown.
    Return per unit of risk taken.
    """
    annual_return = daily_returns.mean() * TRADING_DAYS
    mdd           = abs(max_drawdown(cumulative_pnl))
    if mdd == 0:
        return 0
    return annual_return / mdd


def win_rate(daily_pnl):
    """Percentage of days with positive P&L."""
    winning_days = (daily_pnl > 0).sum()
    total_days   = (daily_pnl != 0).sum()
    if total_days == 0:
        return 0
    return winning_days / total_days * 100


def profit_factor(daily_pnl):
    """
    Total gains divided by total losses.
    Above 1.0 means you make more than you lose.
    """
    gains  = daily_pnl[daily_pnl > 0].sum()
    losses = abs(daily_pnl[daily_pnl < 0].sum())
    if losses == 0:
        return float('inf')
    return gains / losses


def compute_all_metrics(results, pair_name):
    daily_returns  = results['daily_pnl'] / 10_000
    cumulative_pnl = results['cumulative_pnl']
    daily_pnl      = results['daily_pnl']

    metrics = {
        'Pair':          pair_name,
        'Total P&L':     f"${cumulative_pnl.iloc[-1]:,.2f}",
        'Sharpe Ratio':  round(sharpe_ratio(daily_returns), 3),
        'Max Drawdown':  f"${max_drawdown(cumulative_pnl):,.2f}",
        'Calmar Ratio':  round(calmar_ratio(daily_returns, cumulative_pnl), 3),
        'Win Rate':      f"{win_rate(daily_pnl):.1f}%",
        'Profit Factor': round(profit_factor(daily_pnl), 3),
        'Total Trades':  int((results['signal'].diff().abs() > 0).sum()),
    }

    return metrics


if __name__ == "__main__":
    pairs = ['BAC_PNC', 'GS_MS']
    all_metrics = []

    for pair in pairs:
        df = pd.read_csv(f"results_{pair}.csv", index_col=0, parse_dates=True)
        m  = compute_all_metrics(df, pair.replace('_', '/'))
        all_metrics.append(m)

    summary = pd.DataFrame(all_metrics).set_index('Pair')

    print("\n=== Strategy Performance Summary ===\n")
    print(summary.T.to_string())
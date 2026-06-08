import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from data import download_prices
from signals import compute_spread, compute_zscore, generate_signals
from metrics import sharpe_ratio, max_drawdown, win_rate, profit_factor

PAIRS = [
    ('BAC', 'PNC', 2.0, 0.0),
    ('GS',  'MS',  2.5, 0.0),
]

CAPITAL    = 10_000
TOTAL_COST = 0.0015


def run_backtest(prices, pair):
    s1_name, s2_name, entry_z, exit_z = pair
    s1, s2 = prices[s1_name], prices[s2_name]

    spread, hedge_ratio = compute_spread(s1, s2)
    zscore             = compute_zscore(spread)
    signal             = generate_signals(zscore, entry_z=entry_z, exit_z=exit_z)

    r1 = s1.pct_change()
    r2 = s2.pct_change()

    strategy_returns = signal.shift(1) * (r1 - hedge_ratio * r2)
    trade_entries    = signal.diff().abs() > 0
    strategy_returns[trade_entries] -= TOTAL_COST

    daily_pnl      = strategy_returns.dropna() * CAPITAL
    cumulative_pnl = daily_pnl.cumsum()

    return daily_pnl, cumulative_pnl, zscore, signal


def plot_tearsheet(prices, pair):
    s1_name, s2_name, entry_z, exit_z = pair
    s1, s2 = prices[s1_name], prices[s2_name]

    daily_pnl, cumulative_pnl, zscore, signal = run_backtest(prices, pair)

    # Compute metrics
    returns       = daily_pnl / CAPITAL
    sharpe        = sharpe_ratio(returns)
    mdd           = max_drawdown(cumulative_pnl)
    wr            = win_rate(daily_pnl)
    pf            = profit_factor(daily_pnl)
    total_trades  = int((signal.diff().abs() > 0).sum())
    total_return  = cumulative_pnl.iloc[-1] / CAPITAL * 100

    # Drawdown series
    rolling_max   = cumulative_pnl.cummax()
    drawdown      = cumulative_pnl - rolling_max

    # Build figure
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#0f0f0f')
    gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    title = f"Strategy Tearsheet — {s1_name} / {s2_name}"
    fig.suptitle(title, fontsize=16, color='white', fontweight='bold', y=0.98)

    label_color  = '#aaaaaa'
    value_color  = 'white'
    grid_color   = '#2a2a2a'
    accent_green = '#00e676'
    accent_red   = '#ff1744'
    accent_blue  = '#2196f3'

    def style_ax(ax):
        ax.set_facecolor('#1a1a1a')
        ax.tick_params(colors=label_color, labelsize=8)
        ax.xaxis.label.set_color(label_color)
        ax.yaxis.label.set_color(label_color)
        ax.title.set_color(value_color)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        ax.grid(True, color=grid_color, linewidth=0.5, alpha=0.7)

    # --- Panel 1: Cumulative P&L ---
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(cumulative_pnl.index, cumulative_pnl.values, color=accent_green, linewidth=1.5)
    ax1.fill_between(cumulative_pnl.index, 0, cumulative_pnl.values,
                     where=cumulative_pnl.values >= 0, alpha=0.15, color=accent_green)
    ax1.fill_between(cumulative_pnl.index, 0, cumulative_pnl.values,
                     where=cumulative_pnl.values < 0, alpha=0.15, color=accent_red)
    ax1.axhline(0, color='#555555', linewidth=0.8, linestyle='--')
    ax1.set_title("Cumulative P&L", fontsize=11)
    ax1.set_ylabel("P&L ($)", fontsize=9)
    style_ax(ax1)

    # --- Panel 2: Drawdown ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.fill_between(drawdown.index, drawdown.values, 0, color=accent_red, alpha=0.6)
    ax2.set_title("Drawdown", fontsize=11)
    ax2.set_ylabel("P&L ($)", fontsize=9)
    style_ax(ax2)

    # --- Panel 3: Daily P&L distribution ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(daily_pnl.values, bins=50, color=accent_blue, alpha=0.8, edgecolor='none')
    ax3.axvline(0, color='white', linewidth=0.8, linestyle='--')
    ax3.set_title("Daily P&L Distribution", fontsize=11)
    ax3.set_xlabel("P&L ($)", fontsize=9)
    style_ax(ax3)

    # --- Panel 4: Z-score ---
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(zscore.index, zscore.values, color=accent_blue, linewidth=0.8, alpha=0.9)
    ax4.axhline( entry_z, color=accent_red,   linewidth=0.8, linestyle='--')
    ax4.axhline(-entry_z, color=accent_green, linewidth=0.8, linestyle='--')
    ax4.axhline(0, color='#555555', linewidth=0.5, linestyle=':')
    ax4.set_title("Z-Score", fontsize=11)
    style_ax(ax4)

    # --- Panel 5: Metrics table ---
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.set_facecolor('#1a1a1a')
    ax5.axis('off')

    metrics = [
        ("Total Return",   f"{total_return:.1f}%"),
        ("Sharpe Ratio",   f"{sharpe:.3f}"),
        ("Max Drawdown",   f"${mdd:,.0f}"),
        ("Win Rate",       f"{wr:.1f}%"),
        ("Profit Factor",  f"{pf:.3f}"),
        ("Total Trades",   f"{total_trades}"),
        ("Capital",        f"${CAPITAL:,}"),
    ]

    for i, (label, value) in enumerate(metrics):
        y = 0.92 - i * 0.13
        ax5.text(0.05, y, label, transform=ax5.transAxes,
                 color=label_color, fontsize=10)
        ax5.text(0.95, y, value, transform=ax5.transAxes,
                 color=value_color, fontsize=10, fontweight='bold', ha='right')

    ax5.set_title("Performance Metrics", fontsize=11, color=value_color)
    for spine in ax5.spines.values():
        spine.set_edgecolor('#333333')

    filename = f"tearsheet_{s1_name}_{s2_name}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
    plt.close()
    print(f"Saved: {filename}")


if __name__ == "__main__":
    prices = download_prices()

    for pair in PAIRS:
        plot_tearsheet(prices, pair)

    print("\nAll tearsheets generated.")
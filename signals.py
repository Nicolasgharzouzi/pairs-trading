import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from data import download_prices

# The pairs we selected
PAIRS = [
    ('MS',  'WFC'),
    ('KO',  'PEP'),
]

# Signal parameters
WINDOW      = 30   # rolling window for z-score (days)
ENTRY_Z     =  2.0 # open a trade when z-score exceeds this
EXIT_Z      =  0.5 # close the trade when z-score reverts to this


def compute_spread(s1, s2):
    """
    Regress s1 on s2 to find the hedge ratio.
    Spread = s1 - hedge_ratio * s2
    This gives us a (hopefully) stationary series to trade.
    """
    model = OLS(s1, add_constant(s2)).fit()
    hedge_ratio = model.params.iloc[1]
    spread = s1 - hedge_ratio * s2
    return spread, hedge_ratio


def compute_zscore(spread, window=WINDOW):
    """
    Rolling z-score of the spread.
    z = (spread - rolling_mean) / rolling_std
    Tells us how many standard deviations away from 'normal' we are.
    """
    mean = spread.rolling(window).mean()
    std  = spread.rolling(window).std()
    zscore = (spread - mean) / std
    return zscore


def generate_signals(zscore, entry_z=ENTRY_Z, exit_z=EXIT_Z):
    """
    Signal logic:
      +1 = long the spread  (z-score too low,  spread will rise)
      -1 = short the spread (z-score too high, spread will fall)
       0 = flat (no position)
    """
    signal = pd.Series(0, index=zscore.index)
    position = 0

    for i in range(len(zscore)):
        z = zscore.iloc[i]
        if np.isnan(z):
            continue

        if position == 0:
            if z < -entry_z:
                position = 1    # spread too low, go long
            elif z > entry_z:
                position = -1   # spread too high, go short

        elif position == 1:
            if z > -exit_z:
                position = 0    # spread reverted, exit

        elif position == -1:
            if z < exit_z:
                position = 0    # spread reverted, exit

        signal.iloc[i] = position

    return signal


def plot_pair(prices, pair):
    s1_name, s2_name = pair
    s1, s2 = prices[s1_name], prices[s2_name]

    spread, hedge_ratio = compute_spread(s1, s2)
    zscore  = compute_zscore(spread)
    signal  = generate_signals(zscore)

    print(f"\n{s1_name}/{s2_name} — hedge ratio: {hedge_ratio:.4f}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Pairs Signal: {s1_name} / {s2_name}", fontsize=14)

    # Plot 1: Normalised prices
    axes[0].plot(s1 / s1.iloc[0], label=s1_name)
    axes[0].plot(s2 / s2.iloc[0], label=s2_name)
    axes[0].set_ylabel("Normalised Price")
    axes[0].legend()

    # Plot 2: Spread
    axes[1].plot(spread, color='purple')
    axes[1].axhline(spread.mean(), color='black', linestyle='--', linewidth=0.8)
    axes[1].set_ylabel("Spread")

    # Plot 3: Z-score + signals
    axes[2].plot(zscore, color='steelblue')
    axes[2].axhline( ENTRY_Z, color='red',   linestyle='--', linewidth=0.8, label=f'+{ENTRY_Z}σ entry')
    axes[2].axhline(-ENTRY_Z, color='green', linestyle='--', linewidth=0.8, label=f'-{ENTRY_Z}σ entry')
    axes[2].axhline( EXIT_Z,  color='orange',linestyle=':',  linewidth=0.8)
    axes[2].axhline(-EXIT_Z,  color='orange',linestyle=':',  linewidth=0.8, label=f'±{EXIT_Z}σ exit')
    axes[2].fill_between(zscore.index, signal * ENTRY_Z, 0, alpha=0.15, color='gray', label='Position')
    axes[2].set_ylabel("Z-Score")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"signal_{s1_name}_{s2_name}.png", dpi=150)
    plt.show()
    print(f"  Chart saved: signal_{s1_name}_{s2_name}.png")


if __name__ == "__main__":
    prices = download_prices()

    for pair in PAIRS:
        plot_pair(prices, pair)
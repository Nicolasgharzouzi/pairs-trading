import numpy as np
import pandas as pd


def hurst_exponent(series, max_lag=20):
    """
    Compute the Hurst exponent of a time series.
    H < 0.5 → mean-reverting (good for pairs trading)
    H = 0.5 → random walk
    H > 0.5 → trending (avoid trading)
    """
    lags = range(2, max_lag)
    tau  = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]

    reg  = np.polyfit(np.log(lags), np.log(tau), 1)
    return reg[0]


def compute_regime(spread, window=60, threshold=0.55):
    """
    Rolling Hurst exponent on the spread.
    Returns a boolean Series: True = mean-reverting = safe to trade.
    
    We use a 60-day rolling window — enough history to measure regime
    but responsive enough to detect changes.
    
    Threshold of 0.55 gives some buffer above 0.5 random walk boundary.
    """
    regime = pd.Series(False, index=spread.index)

    for i in range(window, len(spread)):
        window_data = spread.iloc[i - window:i].values
        try:
            h = hurst_exponent(window_data)
            regime.iloc[i] = h < threshold
        except:
            regime.iloc[i] = False

    return regime
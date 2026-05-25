import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

START = "2019-01-01"
END   = "2023-12-31"

# Handpicked liquid S&P 500 stocks by sector - 80 stocks
# More stocks = more pairs = better chance of finding strong relationships
TICKERS = [
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD', 'MPC', 'VLO',
    # Banks
    'JPM', 'BAC', 'WFC', 'C', 'USB', 'PNC', 'TFC', 'MTB',
    # Financial services
    'GS', 'MS', 'BLK', 'SCHW', 'AXP', 'COF', 'DFS', 'SYF',
    # Consumer staples
    'KO', 'PEP', 'MDLZ', 'STZ', 'GIS', 'K', 'SJM', 'MKC',
    # Retail
    'WMT', 'TGT', 'COST', 'DG', 'DLTR', 'KR', 'SYY', 'BJ',
    # Pharma
    'JNJ', 'PFE', 'MRK', 'ABBV', 'BMY', 'LLY', 'AMGN', 'GILD',
    # Tech
    'MSFT', 'GOOGL', 'META', 'ORCL', 'IBM', 'CSCO', 'TXN', 'QCOM',
    # Industrials
    'HON', 'MMM', 'GE', 'CAT', 'DE', 'EMR', 'ETN', 'ROK',
    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL',
    # Healthcare
    'UNH', 'CVS', 'CI', 'HUM', 'MCK', 'ABC', 'CAH', 'MOH',
]


def download_prices(tickers=TICKERS, start=START, end=END):
    print(f"Downloading price data for {len(tickers)} stocks...")
    raw    = yf.download(tickers, start=start, end=end)
    prices = raw['Close'].dropna(axis=1, thresh=int(0.95 * len(raw)))
    prices = prices.dropna()
    print(f"Got {len(prices)} days of data for {len(prices.columns)} stocks\n")
    return prices


def compute_half_life(spread):

    spread_lag  = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    spread_lag  = spread_lag.iloc[1:]
    spread_diff = spread_diff.iloc[1:]

    model       = OLS(spread_diff, add_constant(spread_lag)).fit()
    half_life   = -np.log(2) / model.params.iloc[1]
    return half_life


def find_cointegrated_pairs(prices, pvalue_threshold=0.05, min_hl=5, max_hl=40):
    """
    Screen all pair combinations for:
    1. Cointegration (p-value < 0.05)
    2. Tradeable half-life (5 to 40 days)
    """
    tickers = list(prices.columns)
    results = []
    total   = len(list(combinations(tickers, 2)))

    print(f"Testing {total} pairs...")

    for i, (t1, t2) in enumerate(combinations(tickers, 2)):
        if i % 500 == 0:
            print(f"  {i}/{total} tested...")

        s1, s2 = prices[t1], prices[t2]

        # Step 1: cointegration test
        _, pvalue, _ = coint(s1, s2)
        if pvalue >= pvalue_threshold:
            continue

        # Step 2: compute spread
        model        = OLS(s1, add_constant(s2)).fit()
        hedge_ratio  = model.params.iloc[1]
        spread       = s1 - hedge_ratio * s2

        # Step 3: half-life filter
        try:
            hl = compute_half_life(spread)
        except:
            continue

        if not (min_hl <= hl <= max_hl):
            continue
        
        results.append({
            'stock_1':     t1,
            'stock_2':     t2,
            'p_value':     round(pvalue, 4),
            'half_life':   round(hl, 1),
            'hedge_ratio': round(hedge_ratio, 4),
        })

    df = pd.DataFrame(results).sort_values(['p_value', 'half_life']).reset_index(drop=True)
    return df


if __name__ == "__main__":
    prices = download_prices()
    pairs  = find_cointegrated_pairs(prices)

    print(f"\n✓ {len(pairs)} high quality pairs found:\n")
    print(pairs.to_string(index=False))

    # Save for use in other modules
    pairs.to_csv("pairs.csv", index=False)
    print("\nSaved: pairs.csv")
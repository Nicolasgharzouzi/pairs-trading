import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings('ignore')

# Larger universe — more pairs = better chance of finding real relationships
TICKERS = [
    'XOM', 'CVX', 'COP', 'SLB',        # Energy
    'JPM', 'BAC', 'WFC', 'C',           # Banks
    'KO',  'PEP', 'MDLZ', 'STZ',       # Consumer staples
    'MSFT','GOOGL','META','ORCL',       # Tech
    'GS',  'MS',  'BLK', 'SCHW',       # Financial services
    'JNJ', 'PFE', 'MRK', 'ABBV',       # Pharma
    'WMT', 'TGT', 'COST', 'DG',        # Retail
]

START = "2020-01-01"
END   = "2023-12-31"


def download_prices(tickers=TICKERS, start=START, end=END):
    print("Downloading price data...")
    raw = yf.download(tickers, start=start, end=end)
    prices = raw['Close'].dropna()
    print(f"Got {len(prices)} days of data for {len(prices.columns)} stocks\n")
    return prices


def find_cointegrated_pairs(prices, pvalue_threshold=0.05):
    tickers = list(prices.columns)
    results = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            t1, t2 = tickers[i], tickers[j]
            score, pvalue, _ = coint(prices[t1], prices[t2])
            results.append({
                'stock_1': t1,
                'stock_2': t2,
                'p_value': round(pvalue, 4),
                'cointegrated': pvalue < pvalue_threshold
            })

    df = pd.DataFrame(results).sort_values('p_value').reset_index(drop=True)
    return df


if __name__ == "__main__":
    prices = download_prices()

    pairs = find_cointegrated_pairs(prices)

    good = pairs[pairs['cointegrated']]
    print(f"✓ {len(good)} cointegrated pairs found:\n")
    print(good[['stock_1', 'stock_2', 'p_value']].to_string(index=False))
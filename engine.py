from dataclasses import dataclass

import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

import config
from regime import compute_regime
from signals import compute_zscore, generate_signals


@dataclass
class BacktestResult:
    stock_1: str
    stock_2: str
    entry_z: float
    exit_z: float
    hedge_ratio: float
    spread: pd.Series
    zscore: pd.Series
    signal: pd.Series
    strategy_returns: pd.Series
    daily_pnl: pd.Series
    cumulative_pnl: pd.Series

    def to_dataframe(self):
        return pd.DataFrame({
            'signal':         self.signal,
            'zscore':         self.zscore,
            'daily_pnl':      self.daily_pnl,
            'cumulative_pnl': self.cumulative_pnl,
        }).dropna()

    @property
    def total_trades(self):
        return int((self.signal.diff().abs() > 0).sum())


def fit_hedge_ratio(s1, s2):
    model = OLS(s1, add_constant(s2)).fit()
    return model.params.iloc[1]


def run_backtest(
    s1,
    s2,
    entry_z,
    exit_z,
    hedge_ratio=None,
    capital=None,
    total_cost=None,
    zscore_window=None,
):
    """
    Core backtest engine. Pass pre-fitted hedge_ratio for walk-forward
    out-of-sample windows; omit it to fit on the full s1/s2 series.
    """
    if capital is None:
        capital = config.CAPITAL
    if total_cost is None:
        total_cost = config.TOTAL_COST
    if zscore_window is None:
        zscore_window = config.ZSCORE_WINDOW

    if hedge_ratio is None:
        hedge_ratio = fit_hedge_ratio(s1, s2)

    spread = s1 - hedge_ratio * s2
    zscore = compute_zscore(spread, window=zscore_window)
    regime = compute_regime(spread)
    signal = generate_signals(zscore, entry_z=entry_z, exit_z=exit_z, regime=regime)

    r1 = s1.pct_change()
    r2 = s2.pct_change()

    strategy_returns = signal.shift(1) * (r1 - hedge_ratio * r2)
    trade_entries    = signal.diff().abs() > 0
    strategy_returns[trade_entries] -= total_cost

    daily_pnl      = strategy_returns * capital
    cumulative_pnl = daily_pnl.cumsum()

    return BacktestResult(
        stock_1='',
        stock_2='',
        entry_z=entry_z,
        exit_z=exit_z,
        hedge_ratio=hedge_ratio,
        spread=spread,
        zscore=zscore,
        signal=signal,
        strategy_returns=strategy_returns,
        daily_pnl=daily_pnl,
        cumulative_pnl=cumulative_pnl,
    )


def run_backtest_pair(prices, pair, hedge_ratio=None):
    """Run backtest for a (stock_1, stock_2, entry_z, exit_z) tuple."""
    s1_name, s2_name, entry_z, exit_z = pair
    s1, s2 = prices[s1_name], prices[s2_name]

    result = run_backtest(s1, s2, entry_z, exit_z, hedge_ratio=hedge_ratio)
    result.stock_1 = s1_name
    result.stock_2 = s2_name
    return result

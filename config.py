# Data range
START = "2019-01-01"
END   = "2023-12-31"

# Backtest settings
CAPITAL    = 10_000
TOTAL_COST = 0.0015
ZSCORE_WINDOW = 30

# Default signal thresholds (used by signals.py)
DEFAULT_ENTRY_Z = 2.0
DEFAULT_EXIT_Z  = 0.0

# Final portfolio: pairs with positive walk-forward OOS results
# (BAC/PNC and VLO/XOM were flat/negative OOS and dropped)
PAIRS = [
    ('GS',  'MS',  2.5, 0.0),
    ('MOH', 'UNH', 1.5, 0.5),
]

# Block new entries when spread is in a trending regime (Hurst filter)
USE_REGIME = True

# Optimisation grid
ENTRY_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]
EXIT_THRESHOLDS  = [0.0, 0.25, 0.5, 0.75, 1.0]

# Walk-forward settings
TRAIN_DAYS = 504   # ~2 years
TEST_DAYS  = 126   # ~6 months

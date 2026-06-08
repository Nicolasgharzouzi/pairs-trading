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

# Pairs with optimised entry/exit z-scores
PAIRS = [
    ('BAC', 'PNC', 2.0, 0.0),
    ('GS',  'MS',  2.5, 0.0),
]

# Optimisation grid
ENTRY_THRESHOLDS = [1.0, 1.5, 2.0, 2.5, 3.0]
EXIT_THRESHOLDS  = [0.0, 0.25, 0.5, 0.75, 1.0]

# Walk-forward settings
TRAIN_DAYS = 504   # ~2 years
TEST_DAYS  = 126   # ~6 months

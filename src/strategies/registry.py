# src/strategies/registry.py
from dataclasses import dataclass

@dataclass
class StratParams:
    name: str
    ema_fast: int
    ema_slow: int
    bb_window: int
    bb_std: float
    ml_min_prob: float
    dollar_cap: int

# a few presets (tweak later)
ARMS = [
    StratParams("high_tight", ema_fast=3, ema_slow=15, bb_window=20, bb_std=1.4, ml_min_prob=0.50, dollar_cap=20000),
    StratParams("high_loose", ema_fast=3, ema_slow=15, bb_window=20, bb_std=1.2, ml_min_prob=0.48, dollar_cap=20000),
    StratParams("medium",     ema_fast=5, ema_slow=20, bb_window=20, bb_std=1.6, ml_min_prob=0.55, dollar_cap=12000),
]

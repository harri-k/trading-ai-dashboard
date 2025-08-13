# src/rl/bandit_selector.py
import json, math, random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

@dataclass
class ArmStats:
    n: int = 0
    reward_sum: float = 0.0
    reward_sq: float = 0.0
    def mean(self): return self.reward_sum / self.n if self.n > 0 else 0.0
    def var(self):
        if self.n < 2: return 0.0
        m = self.mean()
        return max(0.0, self.reward_sq/self.n - m*m)

@dataclass
class ContextualBandit:
    """Thompson-style sampler per (symbol, vol-bucket) context."""
    store: Path
    arms: List[str]
    buckets: Dict[str, Dict[str, ArmStats]] = field(default_factory=dict)

    def _bucket_key(self, symbol: str, atr_pct: float) -> str:
        b = "low" if atr_pct < 1 else ("med" if atr_pct < 3 else "high")
        return f"{symbol}:{b}"

    def select(self, symbol: str, atr_pct: float) -> str:
        key = self._bucket_key(symbol, atr_pct)
        stats = self.buckets.setdefault(key, {a: ArmStats() for a in self.arms})
        # sample score ~ N(mean, var/(n+1)) with sane default variance
        best = None
        best_score = -1e9
        for arm, s in stats.items():
            mu = s.mean()
            sigma = (s.var()/max(1, s.n))**0.5 if s.n > 0 and s.var() > 0 else 0.05
            sample = random.gauss(mu, max(1e-3, sigma))
            if sample > best_score:
                best_score, best = sample, arm
        return best

    def update(self, symbol: str, atr_pct: float, arm: str, reward: float):
        key = self._bucket_key(symbol, atr_pct)
        stats = self.buckets.setdefault(key, {a: ArmStats() for a in self.arms})
        s = stats[arm]
        s.n += 1
        s.reward_sum += reward
        s.reward_sq += reward * reward
        self._persist()

    def _persist(self):
        self.store.parent.mkdir(parents=True, exist_ok=True)
        blob = {k: {a: v.__dict__ for a, v in d.items()} for k, d in self.buckets.items()}
        self.store.write_text(json.dumps(blob))

    @classmethod
    def load(cls, store: Path, arms: List[str]):
        if store.exists():
            blob = json.loads(store.read_text())
            buckets = {k: {a: ArmStats(**ad) for a, ad in d.items()} for k, d in blob.items()}
            return cls(store=store, arms=arms, buckets=buckets)
        return cls(store=store, arms=arms)

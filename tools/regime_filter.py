"""RegimeFilter: online 2-state (calm / storm) HMM forward filter on per-second counts.

Round 2 switches between a calm state (round1-like) and storms (jumps every ~10 s, vol x3, more IOCs, wider spread);
storms last ~60 s, calm stretches ~160 s, and a switch is not announced in advance, so the useful forecast is "which
state are we in now" plus persistence (docs/DATA_ANALYSIS.md). Feed it one observation per second; it returns
P(storm). O(1) per update.

Evidence per second (all observable by a strategy):
- far:  new market limit orders placed >= 4 ticks behind their side's best price (S has moved away from the book)
- ioc:  taker events by the market (callbacks with market-taker trades)
- wide: 1 if the spread was > 1 tick at the sample, else 0
- lim (optional, lim_rate set): new market limit orders seen that second. Fewer arrive in a storm, and already 10 s
  before its first jump, because orders that would cross the book are dropped when S moves fast.
Likelihoods are Poisson (far, ioc) and Bernoulli (wide) per state, multiplied and tempered by TEMPER (< 1, since the
three are correlated and treating them as independent makes the filter overconfident).

Defaults: fitted on round2_public (scripts/regime_features.py). Imports: math only.
"""

import math


class RegimeFilter:
    def __init__(self, calm_dur_s=158.0, storm_dur_s=61.0, far_rate=(0.36, 10.5), ioc_rate=(8.06, 10.4),
                 wide_p=(0.074, 0.32), temper=0.25, p0=0.28, lim_rate=None):
        self.a01 = 1.0 / calm_dur_s  # per-second switch probabilities
        self.a10 = 1.0 / storm_dur_s
        self.far = far_rate
        self.ioc = ioc_rate
        self.wide = wide_p
        self.temper = temper
        self.p = p0
        self.lim = lim_rate

    def _loglik(self, state: int, far: float, ioc: float, wide: int, lim: float) -> float:
        lf, li = self.far[state], self.ioc[state]
        lw = self.wide[state]
        ll = (far * math.log(lf) - lf + ioc * math.log(li) - li
              + (math.log(lw) if wide else math.log(1.0 - lw)))
        if self.lim is not None and lim >= 0:
            ll += lim * math.log(self.lim[state]) - self.lim[state]
        return ll

    def update(self, far: float, ioc: float, wide: int, lim: float = -1.0) -> float:
        """One second of evidence; returns P(storm)."""
        p = self.p
        prior = p * (1.0 - self.a10) + (1.0 - p) * self.a01
        d = self.temper * (self._loglik(1, far, ioc, wide, lim) - self._loglik(0, far, ioc, wide, lim))
        d = max(-30.0, min(30.0, d))
        odds = prior / (1.0 - prior) * math.exp(d)
        self.p = odds / (1.0 + odds)
        self.p = min(max(self.p, 1e-6), 1.0 - 1e-6)
        return self.p

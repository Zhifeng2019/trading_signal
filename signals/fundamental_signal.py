"""
Fundamental signal generator.

Maps fundamental data (valuation, quality, earnings momentum) into
frequency-band signals.

Rationale for band assignment
------------------------------
MACRO band   : Valuation (P/E, P/B) reverts over multi-year cycles driven
               by interest rate regimes, earnings cycles.
BUSINESS band: Earnings momentum and EPS surprises play out over 1-4 quarters.
SECTOR band  : Relative quality (ROE vs sector) drives medium-term
               sector-rotation signals.
TACTICAL band: Near-term earnings revision momentum.
NOISE band   : Fundamental data is too low-frequency to contribute here
               → minimal weight.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from trading_signals.data.fundamental_data import FundamentalDataFetcher
from trading_signals.config import SignalConfig


class FundamentalSignalGenerator:
    """
    Produces per-band fundamental signals in [-1, +1].

    Parameters
    ----------
    config : SignalConfig
    fundamental_data : FundamentalDataFetcher, optional
        Shared fetcher instance.
    """

    def __init__(
        self,
        config: SignalConfig,
        fundamental_data: Optional[FundamentalDataFetcher] = None,
    ):
        self.config = config
        self._fundamental = fundamental_data or FundamentalDataFetcher()

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def compute(self, ticker: str) -> dict[str, float]:
        """
        Compute fundamental signals for all frequency bands.

        Returns
        -------
        dict mapping band name → signal in [-1, +1].
        Also includes raw sub-scores for diagnostics.
        """
        # Raw factor scores
        val_score = self._fundamental.valuation_signal(ticker)
        qual_score = self._fundamental.quality_signal(ticker)
        earn_score = self._fundamental.earnings_momentum_signal(ticker)

        # Beta-adjusted risk preference
        metrics = self._fundamental.get_metrics(ticker)
        beta = float(metrics.get("beta", 1.0) or 1.0)
        beta_score = self._beta_adjustment(beta)

        # Assign to bands
        signals = {
            "macro":    float(np.clip(0.60 * val_score + 0.40 * qual_score, -1.0, 1.0)),
            "business": float(np.clip(0.50 * earn_score + 0.30 * val_score + 0.20 * qual_score, -1.0, 1.0)),
            "sector":   float(np.clip(0.50 * qual_score + 0.30 * earn_score + 0.20 * val_score, -1.0, 1.0)),
            "tactical": float(np.clip(0.70 * earn_score + 0.30 * qual_score, -1.0, 1.0)),
            "noise":    0.0,   # fundamentals don't contribute to noise band
        }

        # Diagnostics
        signals["valuation_score"] = float(val_score)
        signals["quality_score"] = float(qual_score)
        signals["earnings_score"] = float(earn_score)
        signals["beta"] = beta
        signals["beta_score"] = float(beta_score)

        return signals

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _beta_adjustment(self, beta: float) -> float:
        """
        Convert beta into a risk-preference modifier.

        For an aggressive return target (high target Sharpe), higher-beta
        stocks are preferred (+1).  For a conservative target, lower-beta
        stocks are preferred (+1).

        Returns a value in [-1, +1] where +1 = matches the investor's risk
        preference.
        """
        target_sharpe = self.config.target_sharpe

        # Aggressive investors (target Sharpe > 0.8) like high beta
        # Conservative investors (target Sharpe < 0.4) prefer low beta
        if target_sharpe >= 0.8:
            return float(np.clip(_linear_score(beta, low=0.8, high=1.5), -1.0, 1.0))
        elif target_sharpe <= 0.4:
            return float(np.clip(_linear_score(-beta, low=-1.5, high=-0.5), -1.0, 1.0))
        else:
            # Neutral – prefer beta ≈ 1
            return float(np.clip(-np.abs(beta - 1.0) * 2, -1.0, 0.0))


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────

def _linear_score(value: float, low: float, high: float) -> float:
    mid = (low + high) / 2
    half_range = (high - low) / 2 + 1e-9
    return float(np.clip((value - mid) / half_range, -1.0, 1.0))

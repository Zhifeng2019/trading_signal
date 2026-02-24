"""
Macro signal generator.

Translates the macro regime score into per-band signals.

Band assignment rationale
--------------------------
MACRO band    : The macro regime *is* the macro band – full weight here.
BUSINESS band : Macro drives earnings cycles with a 1-2 quarter lag.
SECTOR band   : Credit spreads and VIX influence sector rotation over weeks.
TACTICAL band : VIX changes affect risk appetite day-to-day.
NOISE band    : Macro does not explain intra-week price noise.

Regime modulation
------------------
A positive macro regime (expansion) amplifies bullish technical/fundamental
signals and attenuates bearish ones.  A negative regime (contraction) does
the opposite.  This is implemented in SignalCombiner as a multiplicative
modifier on the combined signal.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from trading_signals.data.macro_data import MacroDataFetcher
from trading_signals.config import SignalConfig


class MacroSignalGenerator:
    """
    Produces per-band macro signals in [-1, +1].

    Parameters
    ----------
    config : SignalConfig
    macro_data : MacroDataFetcher, optional
    """

    def __init__(
        self,
        config: SignalConfig,
        macro_data: Optional[MacroDataFetcher] = None,
    ):
        self.config = config
        self._macro = macro_data or MacroDataFetcher()
        self._cached_regime: Optional[dict] = None

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def compute(self) -> dict[str, float]:
        """
        Compute macro signals for all frequency bands.

        Returns
        -------
        dict mapping band name → signal in [-1, +1],
        plus sub-component scores.
        """
        regime = self._get_regime()
        composite = regime["composite"]
        yield_score = regime.get("yield_curve", 0.0)
        credit_score = regime.get("credit", 0.0)
        inflation_score = regime.get("inflation", 0.0)
        monetary_score = regime.get("monetary_policy", 0.0)
        growth_score = regime.get("growth", 0.0)

        # Assign macro signals to frequency bands
        # Lower-frequency bands get more macro content
        signals = {
            "macro":    float(np.clip(composite, -1.0, 1.0)),
            "business": float(np.clip(
                0.50 * composite + 0.30 * yield_score + 0.20 * growth_score,
                -1.0, 1.0
            )),
            "sector":   float(np.clip(
                0.40 * credit_score + 0.30 * composite + 0.30 * monetary_score,
                -1.0, 1.0
            )),
            "tactical": float(np.clip(
                0.50 * credit_score + 0.30 * monetary_score + 0.20 * composite,
                -1.0, 1.0
            )),
            "noise":    0.0,
        }

        # Sub-components for diagnostics
        signals.update({
            "composite":       composite,
            "yield_curve":     yield_score,
            "credit":          credit_score,
            "inflation":       inflation_score,
            "monetary_policy": monetary_score,
            "growth":          growth_score,
        })

        return signals

    def regime_multiplier(self) -> float:
        """
        Return a multiplier [0.5, 1.5] that can be applied to other signals.

        Positive macro regime → amplify bullish signals (multiplier > 1).
        Negative macro regime → attenuate or flip (multiplier < 1).
        """
        regime = self._get_regime()
        composite = regime["composite"]
        # Map [-1, +1] → [0.5, 1.5]
        return float(1.0 + 0.5 * composite)

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _get_regime(self) -> dict:
        if self._cached_regime is None:
            try:
                self._cached_regime = self._macro.regime_score()
            except Exception:
                self._cached_regime = {"composite": 0.0}
        return self._cached_regime

    def invalidate_cache(self):
        """Force a fresh regime fetch on next call."""
        self._cached_regime = None

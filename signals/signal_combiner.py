"""
Signal combiner – the heart of the system.

This module takes per-band signals from technical, fundamental, and macro
generators and produces a single composite signal that reflects the
investor's specific objective.

Horizon-to-frequency-band weight mapping
-----------------------------------------
The core insight is that a signal for an investor with horizon H (trading days)
should primarily reflect cycles with period ≈ H.  We model this as a Gaussian
kernel in log-period space:

    w(band | H) ∝ exp( -[log10(H) - log10(T_band)]² / (2·σ²) )

where T_band is the characteristic period of each band.

This means:
  H =  5 days  → weights concentrated on TACTICAL and NOISE
  H = 21 days  → weights on SECTOR and TACTICAL
  H = 63 days  → weights on BUSINESS and SECTOR
  H = 252 days → weights on MACRO and BUSINESS

Return-target risk adjustment
------------------------------
A higher target return implies a higher required Sharpe, which in turn
requires taking on more risk (higher-volatility, shorter-cycle components).
We implement this as a shift of the Gaussian centre towards shorter periods:

    log10(H_effective) = log10(H) - risk_shift × target_sharpe_excess

Source-to-band weight mixing
-----------------------------
Within each band, the three signal sources (technical, fundamental, macro)
contribute with band-specific weights defined in config.BAND_SOURCE_WEIGHTS.
These reflect which data source is most informative at each timescale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from trading_signals.config import (
    SignalConfig,
    FREQ_BANDS,
    BAND_LOG_PERIOD_CENTRE,
    BAND_LOG_PERIOD_WIDTH,
    BAND_SOURCE_WEIGHTS,
    RISK_SIGMA_SHIFT,
    TRADING_DAYS_PER_YEAR,
)
from trading_signals.data.market_data import MarketDataFetcher
from trading_signals.data.fundamental_data import FundamentalDataFetcher
from trading_signals.data.macro_data import MacroDataFetcher
from trading_signals.signals.technical_signal import TechnicalSignalGenerator
from trading_signals.signals.fundamental_signal import FundamentalSignalGenerator
from trading_signals.signals.macro_signal import MacroSignalGenerator


# ──────────────────────────────────────────────────────────────────────────────
# Output dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class TickerSignal:
    """Complete signal output for a single ticker."""
    ticker: str
    composite_score: float      # Final [-1, +1] signal
    direction: str              # 'BUY' | 'SELL' | 'HOLD'
    confidence: float           # [0, 1] – how strong/consistent the signal is
    horizon_days: int
    target_return: float

    # Per-band signals (raw, before horizon weighting)
    band_scores: dict[str, float]

    # Per-source signals (averaged across bands for each source)
    source_scores: dict[str, float]

    # Band weights for the given horizon/target
    band_weights: dict[str, float]

    # Diagnostic sub-scores
    technical_detail: dict
    fundamental_detail: dict
    macro_detail: dict

    def summary(self) -> str:
        lines = [
            f"Ticker          : {self.ticker}",
            f"Signal          : {self.direction}  ({self.composite_score:+.3f})",
            f"Confidence      : {self.confidence:.1%}",
            f"Horizon         : {self.horizon_days} days  "
            f"(~{self.horizon_days/21:.1f} months)",
            f"Target return   : {self.target_return:.1%} p.a.",
            "",
            "Band scores (raw → weighted):",
        ]
        for band, raw in self.band_scores.items():
            w = self.band_weights.get(band, 0.0)
            lines.append(f"  {band:<10} {raw:+.3f}  ×  {w:.3f}")
        lines.append("")
        lines.append("Source scores:")
        for src, score in self.source_scores.items():
            lines.append(f"  {src:<15} {score:+.3f}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# Main combiner
# ──────────────────────────────────────────────────────────────────────────────

class SignalCombiner:
    """
    Combines technical, fundamental, and macro signals into a single
    investment signal parameterised by horizon and return target.

    Parameters
    ----------
    config : SignalConfig
    fred_api_key : str, optional
        FRED API key for macro data.
    """

    def __init__(
        self,
        config: SignalConfig,
        fred_api_key: Optional[str] = None,
    ):
        self.config = config

        # Shared data fetchers
        self._market = MarketDataFetcher(config.lookback_years)
        self._fundamental = FundamentalDataFetcher()
        self._macro_data = MacroDataFetcher(fred_api_key)

        # Signal generators
        self._tech_gen = TechnicalSignalGenerator(config, self._market)
        self._fund_gen = FundamentalSignalGenerator(config, self._fundamental)
        self._macro_gen = MacroSignalGenerator(config, self._macro_data)

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def compute(self, ticker: str) -> TickerSignal:
        """
        Compute the composite investment signal for a single ticker.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.

        Returns
        -------
        TickerSignal
        """
        # 1. Fetch per-source signals for all bands
        tech_signals = self._tech_gen.compute(ticker)
        fund_signals = self._fund_gen.compute(ticker)
        macro_signals = self._macro_gen.compute()

        # 2. Compute horizon-based band weights
        band_weights = self._horizon_band_weights()

        # 3. Mix sources within each band and weight by horizon
        band_scores: dict[str, float] = {}
        source_weighted_scores: dict[str, list] = {
            "technical": [], "fundamental": [], "macro": []
        }

        for band in FREQ_BANDS:
            src_weights = BAND_SOURCE_WEIGHTS[band]
            tech_s = tech_signals.get(band, 0.0)
            fund_s = fund_signals.get(band, 0.0)
            macro_s = macro_signals.get(band, 0.0)

            # Weighted source mix within this band
            band_raw = (
                src_weights["technical"] * tech_s
                + src_weights["fundamental"] * fund_s
                + src_weights["macro"] * macro_s
            )
            band_scores[band] = float(np.clip(band_raw, -1.0, 1.0))

            # Accumulate source scores (weighted by band weight for later average)
            w = band_weights[band]
            source_weighted_scores["technical"].append(tech_s * w)
            source_weighted_scores["fundamental"].append(fund_s * w)
            source_weighted_scores["macro"].append(macro_s * w)

        # 4. Horizon-weighted composite
        composite = sum(
            band_weights[b] * band_scores[b] for b in FREQ_BANDS
        )

        # 5. Macro regime multiplier (amplifies/attenuates based on expansion/contraction)
        regime_mult = self._macro_gen.regime_multiplier()
        composite = float(np.clip(composite * regime_mult, -1.0, 1.0))

        # 6. Forward-projection bonus (FFT projection at the target horizon)
        fwd_proj = tech_signals.get("forward_projection", 0.0)
        composite = float(np.clip(0.75 * composite + 0.25 * fwd_proj, -1.0, 1.0))

        # 7. Confidence: agreement across bands (std dev of weighted band scores)
        band_score_values = [band_scores[b] for b in FREQ_BANDS]
        signal_std = float(np.std(band_score_values))
        idio_score = tech_signals.get("idiosyncratic_score", 0.5)
        entropy = tech_signals.get("spectral_entropy", 0.5)
        # High agreement + low entropy + high idiosyncratic → high confidence
        confidence = float(np.clip(
            (1 - signal_std) * (1 - 0.5 * entropy) * (0.5 + 0.5 * idio_score),
            0.0, 1.0
        ))

        # 8. Direction
        direction = self._direction(composite, confidence)

        # Source average scores
        source_scores = {
            src: float(np.sum(vals)) for src, vals in source_weighted_scores.items()
        }

        return TickerSignal(
            ticker=ticker,
            composite_score=composite,
            direction=direction,
            confidence=confidence,
            horizon_days=self.config.horizon_days,
            target_return=self.config.target_return_annual,
            band_scores=band_scores,
            source_scores=source_scores,
            band_weights=band_weights,
            technical_detail=tech_signals,
            fundamental_detail=fund_signals,
            macro_detail=macro_signals,
        )

    def band_weights_for_horizon(
        self, horizon_days: int, target_return: float
    ) -> dict[str, float]:
        """
        Utility: compute band weights for an arbitrary horizon and return target
        without computing the full signal.
        """
        tmp_config = SignalConfig(
            horizon_days=horizon_days,
            target_return_annual=target_return,
            risk_free_rate=self.config.risk_free_rate,
        )
        return _horizon_band_weights(tmp_config)

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _horizon_band_weights(self) -> dict[str, float]:
        return _horizon_band_weights(self.config)

    @staticmethod
    def _direction(score: float, confidence: float) -> str:
        threshold = max(0.05, 0.15 * (1 - confidence))
        if score > threshold:
            return "BUY"
        elif score < -threshold:
            return "SELL"
        return "HOLD"


# ──────────────────────────────────────────────────────────────────────────────
# Standalone weight computation (used by visualisation too)
# ──────────────────────────────────────────────────────────────────────────────

def _horizon_band_weights(config: SignalConfig) -> dict[str, float]:
    """
    Compute Gaussian frequency-band weights given investment horizon and
    return target.

    A higher target Sharpe shifts the Gaussian towards shorter periods
    (higher risk / higher return components).
    """
    log_horizon = np.log10(float(config.horizon_days))

    # Risk-adjustment: high target Sharpe → shift towards shorter cycles
    excess_sharpe = max(config.target_sharpe - 0.5, 0.0)   # neutral Sharpe ≈ 0.5
    risk_shift = RISK_SIGMA_SHIFT * excess_sharpe
    log_horizon_effective = log_horizon - risk_shift        # shift towards shorter periods

    weights_raw = {}
    for band, centre in BAND_LOG_PERIOD_CENTRE.items():
        dist = (log_horizon_effective - centre) / BAND_LOG_PERIOD_WIDTH
        weights_raw[band] = float(np.exp(-0.5 * dist ** 2))

    # Normalise
    total = sum(weights_raw.values()) + 1e-12
    return {b: w / total for b, w in weights_raw.items()}

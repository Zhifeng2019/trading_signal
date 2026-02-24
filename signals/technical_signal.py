"""
Technical signal generator.

Combines FFT/wavelet frequency-domain signals with volume-weighted analysis
to produce a band-level technical signal for each frequency band.

Key ideas
---------
1. **Phase signal**: Where is the price cycle relative to its trough/peak?
   Derived from the FFT phase of the dominant component in each band.

2. **Volume confirmation**: High-volume moves in the direction of the cycle
   are more reliable.  We compute a volume-weighted spectral signal.

3. **Cross-band momentum**: Positive alignment across multiple bands
   (e.g. price is rising in both the SECTOR and TACTICAL bands) strengthens
   the signal.

4. **Idiosyncratic vs systematic frequency content**:
   By comparing the stock's spectrum to the cross-asset average spectrum,
   we separate idiosyncratic cycles (more predictable, stock-specific) from
   systematic cycles (correlated with market-wide moves, harder to exploit).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from trading_signals.data.market_data import MarketDataFetcher
from trading_signals.signals.frequency_decomp import FrequencyDecomposer, DecompositionResult
from trading_signals.config import FREQ_BANDS, SignalConfig


class TechnicalSignalGenerator:
    """
    Produces per-band technical signals in [-1, +1] for a single ticker.

    Parameters
    ----------
    config : SignalConfig
    market_data : MarketDataFetcher, optional
        Shared fetcher instance to avoid duplicate downloads.
    decomposer : FrequencyDecomposer, optional
        Shared decomposer instance.
    """

    def __init__(
        self,
        config: SignalConfig,
        market_data: Optional[MarketDataFetcher] = None,
        decomposer: Optional[FrequencyDecomposer] = None,
    ):
        self.config = config
        self._market = market_data or MarketDataFetcher(config.lookback_years)
        self._decomposer = decomposer or FrequencyDecomposer(
            n_dominant=config.n_dominant_freqs,
            use_wavelets=config.use_wavelets,
        )

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def compute(self, ticker: str) -> dict[str, float]:
        """
        Compute technical signals for all frequency bands.

        Returns
        -------
        dict mapping band name → signal in [-1, +1], plus extra diagnostics:
            forward_projection  : FFT projection over investment horizon
            spectral_entropy    : 0 = periodic, 1 = random noise
            idiosyncratic_score : 0 = fully systematic, 1 = fully idiosyncratic
        """
        returns = self._market.get_returns(ticker)

        # Core FFT decomposition on price returns
        decomp = self._decomposer.decompose(returns, ticker=ticker)

        # Volume-weighted decomposition for cross-confirmation
        vw_returns = self._market.get_volume_weighted_returns(ticker)
        vw_decomp = self._decomposer.decompose(vw_returns, ticker=f"{ticker}_VW")

        # Forward projection at the target investment horizon
        forward = self._decomposer.project_forward(returns, self.config.horizon_days)

        # Idiosyncratic score: compare spectrum to cross-asset basket
        idio_score = self._idiosyncratic_score(ticker, returns)

        # Cross-band alignment bonus: if neighbouring bands agree, boost signal
        result: dict[str, float] = {}
        band_names = list(FREQ_BANDS.keys())
        raw_band_signals = {b: decomp.get_band_signal(b) for b in band_names}

        for band in band_names:
            fft_sig = decomp.get_band_signal(band)
            vw_sig = vw_decomp.get_band_signal(band)
            wavelet_sig = decomp.wavelet_band_signals.get(band, 0.0)

            # Weighted average: FFT 50%, volume-weighted FFT 30%, wavelet 20%
            combined = 0.50 * fft_sig + 0.30 * vw_sig + 0.20 * wavelet_sig

            # Agreement bonus with adjacent bands
            alignment = self._cross_band_alignment(band, raw_band_signals, band_names)

            result[band] = float(np.clip(combined * (1 + 0.25 * alignment), -1.0, 1.0))

        result["forward_projection"] = forward
        result["spectral_entropy"] = decomp.spectral_entropy
        result["idiosyncratic_score"] = idio_score
        return result

    def dominant_cycles_summary(self, ticker: str) -> pd.DataFrame:
        """
        Return a DataFrame summarising the dominant spectral cycles.
        Useful for inspection and visualisation.
        """
        returns = self._market.get_returns(ticker)
        decomp = self._decomposer.decompose(returns, ticker=ticker)
        rows = []
        for comp in decomp.dominant_components:
            rows.append({
                "period_days": round(comp.period_days, 1),
                "period_months": round(comp.period_days / 21, 1),
                "amplitude": round(comp.amplitude * 100, 3),   # as % return
                "phase_deg": round(np.degrees(comp.phase), 1),
                "phase_signal": round(comp.phase_signal, 3),
                "power_fraction": round(comp.power_fraction * 100, 2),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("power_fraction", ascending=False)
        return df

    def get_decomposition(self, ticker: str) -> DecompositionResult:
        """Return the raw DecompositionResult for advanced use."""
        returns = self._market.get_returns(ticker)
        return self._decomposer.decompose(returns, ticker=ticker)

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _idiosyncratic_score(
        self,
        ticker: str,
        returns: pd.Series,
    ) -> float:
        """
        Estimate what fraction of spectral power is idiosyncratic (not shared
        with the cross-asset basket).

        Method: compute R² of regressing the stock's return spectrum onto the
        first principal component of the cross-asset spectrum.
        Higher residual power → more idiosyncratic.
        """
        cross_tickers = [t for t in self.config.cross_asset_tickers if t != ticker]
        try:
            basket = self._market.get_cross_asset_returns(cross_tickers, ticker)
        except Exception:
            return 0.5   # default: 50% idiosyncratic

        aligned = basket.dropna()
        if len(aligned) < 32:
            return 0.5

        stock_ret = aligned[ticker].values if ticker in aligned.columns else returns.values
        basket_ret = aligned[[c for c in aligned.columns if c != ticker]].values

        # FFT of stock and basket columns
        stock_fft_amp = np.abs(np.fft.rfft(stock_ret)) ** 2
        basket_fft_amp = np.abs(np.fft.rfft(basket_ret, axis=0)) ** 2

        # Mean basket power
        mean_basket = basket_fft_amp.mean(axis=1)

        # R² of stock power on basket power
        corr = np.corrcoef(stock_fft_amp, mean_basket)[0, 1]
        systematic_r2 = float(np.clip(corr ** 2, 0.0, 1.0))
        return float(1.0 - systematic_r2)

    @staticmethod
    def _cross_band_alignment(
        band: str,
        band_signals: dict[str, float],
        band_names: list[str],
    ) -> float:
        """
        Return a value in [-1, +1] representing agreement of adjacent bands.
        +1 = all neighbours point the same direction as this band.
        -1 = neighbours contradict this band.
        """
        idx = band_names.index(band)
        neighbours = []
        if idx > 0:
            neighbours.append(band_signals[band_names[idx - 1]])
        if idx < len(band_names) - 1:
            neighbours.append(band_signals[band_names[idx + 1]])
        if not neighbours:
            return 0.0
        this_sig = band_signals[band]
        alignment = np.mean([np.sign(this_sig) * np.sign(n) for n in neighbours])
        return float(alignment)

"""
Frequency-domain decomposition of financial time series.

Core algorithms
---------------
1. FFT-based spectral decomposition
   - Extract dominant cycles, their amplitudes and phases
   - Band-pass filter to isolate macro / business / sector / tactical frequency bands
   - Project dominant cycles forward to generate a predictive signal

2. Discrete Wavelet Transform (DWT) decomposition
   - Multi-resolution analysis across dyadic frequency bands
   - Each level corresponds to a specific timescale
   - Approximation = long-run trend; detail coefficients = shorter cycles

3. Cross-spectrum analysis (inter-asset phase coherence)
   - Identify which stocks lead/lag at a given frequency
   - High coherence at high frequencies → systematic risk (correlated moves)
   - Low coherence → idiosyncratic (diversifiable) cycles

Signal convention
-----------------
  Positive output → price expected to rise over the target horizon
  Negative output → price expected to fall
  Range           → [-1, +1] (normalised)

Key insight (Frequency-Horizon mapping)
----------------------------------------
An investor with horizon H (trading days) is primarily affected by price
cycles with period ≈ H.  The FFT decomposes the return series into
independent sinusoidal cycles.  If the current phase of the H-period cycle
is near its trough and the amplitude is large, a buy signal is generated.
The exact weight given to each frequency band is controlled by the
horizon-to-frequency mapping in config.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import windows as sig_windows

try:
    import pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _PYWT_AVAILABLE = False

from trading_signals.config import (
    FREQ_BANDS,
    BAND_PERIODS,
    WAVELET_NAME,
    WAVELET_MAX_LEVEL,
    TRADING_DAYS_PER_YEAR,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SpectralComponent:
    """A single dominant spectral component extracted from the FFT."""
    frequency: float      # cycles per trading day
    period_days: float    # = 1 / frequency
    amplitude: float      # peak-to-trough half-range of this cycle in return units
    phase: float          # current phase in radians ∈ [-π, π]
    power_fraction: float # fraction of total spectral power in this component

    @property
    def phase_signal(self) -> float:
        """
        [-1, +1] signal derived from current phase.

        The idea: a sinusoidal cycle  A·cos(2π·f·t + φ) has its trough at
        phase = π and its peak at phase = 0.  The expected *forward* return
        over the next quarter-cycle (≈ period/4 days) is proportional to
        the negative of the cosine of the current phase.

          phase ≈ π  (at trough, about to rise)  → signal = +1
          phase ≈ 0  (at peak, about to fall)    → signal = -1
        """
        return float(-np.cos(self.phase))

    @property
    def momentum_signal(self) -> float:
        """
        Rate of change of the cycle: positive when cycle is ascending.
        = sin(phase), so +1 when phase is at the middle of the up-leg.
        """
        return float(np.sin(self.phase))


@dataclass
class BandSignal:
    """Aggregated signal for a single frequency band."""
    band_name: str
    raw_signal: float           # phase-based signal weighted by amplitude
    amplitude_score: float      # how strong are cycles in this band (0–1)
    dominant_period: float      # dominant period in days
    components: list[SpectralComponent] = field(default_factory=list)

    @property
    def signal(self) -> float:
        """Amplitude-weighted phase signal, clipped to [-1, +1]."""
        return float(np.clip(self.raw_signal * self.amplitude_score, -1.0, 1.0))


@dataclass
class DecompositionResult:
    """Full decomposition output for a single ticker."""
    ticker: str
    band_signals: dict[str, BandSignal]          # keyed by band name
    dominant_components: list[SpectralComponent] # top-N across all bands
    spectral_entropy: float                      # 0 = pure tone, 1 = white noise
    wavelet_band_signals: dict[str, float]       # DWT band signals (if available)

    def get_band_signal(self, band: str) -> float:
        """Return the signal for a named frequency band."""
        bs = self.band_signals.get(band)
        return bs.signal if bs is not None else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Main decomposer class
# ──────────────────────────────────────────────────────────────────────────────

class FrequencyDecomposer:
    """
    Decomposes a return series into frequency components and produces
    band-level signals.

    Parameters
    ----------
    n_dominant : int
        Number of dominant spectral components to track.
    use_wavelets : bool
        Whether to supplement FFT with discrete wavelet decomposition.
    window : str
        Windowing function applied before FFT to reduce spectral leakage.
        Options: 'hann', 'blackman', 'hamming', 'none'.
    """

    def __init__(
        self,
        n_dominant: int = 5,
        use_wavelets: bool = True,
        window: str = "hann",
    ):
        self.n_dominant = n_dominant
        self.use_wavelets = use_wavelets and _PYWT_AVAILABLE
        self.window_name = window

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def decompose(
        self,
        returns: pd.Series,
        ticker: str = "UNKNOWN",
    ) -> DecompositionResult:
        """
        Full frequency decomposition of a daily return series.

        Parameters
        ----------
        returns : pd.Series
            Daily log-return series (no NaNs).
        ticker : str
            Label for the result.

        Returns
        -------
        DecompositionResult
        """
        r = np.asarray(returns.dropna(), dtype=float)
        if len(r) < 32:
            raise ValueError(f"Need at least 32 data points, got {len(r)}")

        # Apply window to reduce leakage
        r_windowed = self._apply_window(r)

        # Compute FFT
        freqs, power, phases, amplitudes = self._compute_fft(r_windowed)

        # Extract per-band signals
        band_signals = {}
        for band_name, (f_lo, f_hi) in FREQ_BANDS.items():
            band_signals[band_name] = self._band_signal(
                band_name, freqs, power, phases, amplitudes, f_lo, f_hi
            )

        # Extract global top-N dominant components
        dominant = self._dominant_components(freqs, power, phases, amplitudes)

        # Spectral entropy
        entropy = self._spectral_entropy(power)

        # Wavelet decomposition
        wavelet_signals = {}
        if self.use_wavelets:
            wavelet_signals = self._wavelet_decompose(r)

        return DecompositionResult(
            ticker=ticker,
            band_signals=band_signals,
            dominant_components=dominant,
            spectral_entropy=entropy,
            wavelet_band_signals=wavelet_signals,
        )

    def cross_spectrum(
        self,
        returns_a: pd.Series,
        returns_b: pd.Series,
        band: str = "tactical",
    ) -> dict[str, float]:
        """
        Compute cross-spectral coherence and phase lead/lag between two series.

        A positive phase_lead_a means series A leads series B at the target
        frequency band → if A is rising, B is expected to follow.

        Returns
        -------
        dict with keys: coherence, phase_lead_a, predictive_signal
        """
        # Align
        aligned = pd.DataFrame({"a": returns_a, "b": returns_b}).dropna()
        if len(aligned) < 32:
            return {"coherence": 0.0, "phase_lead_a": 0.0, "predictive_signal": 0.0}

        ra = self._apply_window(aligned["a"].values)
        rb = self._apply_window(aligned["b"].values)

        fa = np.fft.rfft(ra)
        fb = np.fft.rfft(rb)
        freqs = np.fft.rfftfreq(len(ra))

        f_lo, f_hi = FREQ_BANDS[band]
        mask = (freqs >= f_lo) & (freqs <= f_hi) if f_hi > 0 else (freqs <= f_lo)

        cross = fa[mask] * np.conj(fb[mask])
        power_a = np.abs(fa[mask]) ** 2
        power_b = np.abs(fb[mask]) ** 2

        # Coherence: |cross|² / (power_a * power_b)
        denom = np.sqrt(power_a.sum() * power_b.sum()) + 1e-12
        coherence = float(np.abs(cross.sum()) / denom)

        # Mean phase difference
        phase_diff = float(np.angle(cross.mean())) if cross.size > 0 else 0.0

        # Predictive signal: A leads B when phase_diff > 0 and A is currently rising
        a_recent_trend = float(np.sign(ra[-5:].mean()))
        predictive_signal = float(
            np.clip(coherence * np.sin(phase_diff) * a_recent_trend, -1.0, 1.0)
        )

        return {
            "coherence": coherence,
            "phase_lead_a": phase_diff,
            "predictive_signal": predictive_signal,
        }

    def project_forward(
        self,
        returns: pd.Series,
        horizon_days: int,
        n_components: int = 10,
    ) -> float:
        """
        Project the dominant spectral components forward by `horizon_days` days
        and return the expected total return as a normalised score in [-1, +1].

        This is a *deterministic* extrapolation of current cycles – it says
        "if these cycles continue at their current phase and amplitude, where
        will the signal be in horizon_days days?"  It does NOT account for
        cycle decay or regime change.
        """
        r = np.asarray(returns.dropna(), dtype=float)
        if len(r) < 32:
            return 0.0

        r_windowed = self._apply_window(r)
        freqs, power, phases, amplitudes = self._compute_fft(r_windowed)

        # Focus on frequencies relevant to the horizon (band centred on 1/horizon)
        f_target = 1.0 / horizon_days
        # Allow ±1 octave
        f_lo = f_target / 2
        f_hi = min(f_target * 2, 0.5)
        mask = (freqs >= f_lo) & (freqs <= f_hi)

        if not mask.any():
            return 0.0

        # Top-n by power within band
        band_power = np.where(mask, power, 0.0)
        top_idx = np.argsort(band_power)[::-1][:n_components]

        # Reconstruct future signal: Σ A_k · cos(2π·f_k·(t+H) + φ_k) – current
        t_now = len(r)
        projected = sum(
            amplitudes[i] * np.cos(2 * np.pi * freqs[i] * (t_now + horizon_days) + phases[i])
            for i in top_idx
        )
        current = sum(
            amplitudes[i] * np.cos(2 * np.pi * freqs[i] * t_now + phases[i])
            for i in top_idx
        )
        total_amp = sum(amplitudes[i] for i in top_idx) + 1e-12

        # Normalise by total amplitude of selected components
        score = (projected - current) / total_amp
        return float(np.clip(score, -1.0, 1.0))

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _apply_window(self, r: np.ndarray) -> np.ndarray:
        n = len(r)
        if self.window_name == "hann":
            w = sig_windows.hann(n)
        elif self.window_name == "blackman":
            w = sig_windows.blackman(n)
        elif self.window_name == "hamming":
            w = sig_windows.hamming(n)
        else:
            w = np.ones(n)
        # Normalise window energy so amplitude estimates remain comparable
        w = w / (w.mean() + 1e-12)
        return r * w

    def _compute_fft(
        self, r: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns
        -------
        freqs      : positive frequency grid (cycles/day)
        power      : one-sided power spectrum
        phases     : phase at each frequency (radians)
        amplitudes : one-sided amplitude spectrum (units of return)
        """
        n = len(r)
        fft_vals = np.fft.rfft(r)
        freqs = np.fft.rfftfreq(n)

        amplitudes = np.abs(fft_vals) / n * 2   # two-sided → one-sided conversion
        phases = np.angle(fft_vals)
        power = amplitudes ** 2

        return freqs, power, phases, amplitudes

    def _band_signal(
        self,
        band_name: str,
        freqs: np.ndarray,
        power: np.ndarray,
        phases: np.ndarray,
        amplitudes: np.ndarray,
        f_lo: float,
        f_hi: float,
    ) -> BandSignal:
        """Extract signal for a specific frequency band."""
        if f_hi == 0.0:
            mask = freqs <= f_lo
        else:
            mask = (freqs >= f_lo) & (freqs < f_hi)

        if not mask.any():
            return BandSignal(band_name, 0.0, 0.0, BAND_PERIODS[band_name])

        band_power = power[mask]
        band_phases = phases[mask]
        band_amps = amplitudes[mask]
        band_freqs = freqs[mask]

        total_power = power.sum() + 1e-12
        band_total_power = band_power.sum()

        # Amplitude score: fraction of total power in this band (0→1)
        amplitude_score = float(np.clip(band_total_power / total_power * 3, 0.0, 1.0))

        # Dominant frequency in band
        dom_idx = np.argmax(band_power)
        dom_freq = float(band_freqs[dom_idx])
        dom_period = 1.0 / dom_freq if dom_freq > 1e-9 else BAND_PERIODS[band_name]

        # Power-weighted phase signal across all components in band
        weights = band_power / (band_total_power + 1e-12)
        phase_signals = -np.cos(band_phases)        # trough → +1, peak → -1
        raw_signal = float(np.dot(weights, phase_signals))

        # Top components for inspection
        top_n = min(3, mask.sum())
        top_idx = np.argsort(band_power)[::-1][:top_n]
        global_idx = np.where(mask)[0][top_idx]
        components = [
            SpectralComponent(
                frequency=float(freqs[i]),
                period_days=1.0 / float(freqs[i]) if freqs[i] > 1e-9 else 9999,
                amplitude=float(amplitudes[i]),
                phase=float(phases[i]),
                power_fraction=float(power[i] / total_power),
            )
            for i in global_idx
        ]

        return BandSignal(
            band_name=band_name,
            raw_signal=raw_signal,
            amplitude_score=amplitude_score,
            dominant_period=dom_period,
            components=components,
        )

    def _dominant_components(
        self,
        freqs: np.ndarray,
        power: np.ndarray,
        phases: np.ndarray,
        amplitudes: np.ndarray,
    ) -> list[SpectralComponent]:
        """Return the top-N dominant spectral components across all bands."""
        # Exclude DC component (freq=0)
        valid = freqs > 1e-9
        idx_sorted = np.argsort(power)[::-1]
        valid_sorted = [i for i in idx_sorted if valid[i]]
        top_idx = valid_sorted[: self.n_dominant]
        total_power = power.sum() + 1e-12
        return [
            SpectralComponent(
                frequency=float(freqs[i]),
                period_days=1.0 / float(freqs[i]),
                amplitude=float(amplitudes[i]),
                phase=float(phases[i]),
                power_fraction=float(power[i] / total_power),
            )
            for i in top_idx
        ]

    def _spectral_entropy(self, power: np.ndarray) -> float:
        """
        Normalised spectral entropy.
        0 = pure sinusoid (all power at one frequency)
        1 = white noise (power evenly spread across all frequencies)
        """
        p = power[power > 0]
        if len(p) == 0:
            return 1.0
        p_norm = p / p.sum()
        entropy = -np.sum(p_norm * np.log(p_norm + 1e-12))
        max_entropy = np.log(len(p))
        return float(entropy / max_entropy) if max_entropy > 0 else 1.0

    def _wavelet_decompose(self, r: np.ndarray) -> dict[str, float]:
        """
        Discrete Wavelet Transform decomposition.

        Returns a dict mapping frequency-band names to signals in [-1, +1].
        DWT level k captures periods in [2^k, 2^(k+1)] days.

        Level → approximate band mapping:
            1  : 2–4   days     → noise
            2  : 4–8   days     → noise/tactical
            3  : 8–16  days     → tactical
            4  : 16–32 days     → tactical/sector
            5  : 32–64 days     → sector
            6  : 64–128 days    → business
            7  : 128–256 days   → business/macro
            8  : >256  days     → macro (approximation)
        """
        if not _PYWT_AVAILABLE:
            return {}

        r = np.asarray(r, dtype=np.float64).copy()  # pywt requires a writable buffer
        max_level = min(WAVELET_MAX_LEVEL, pywt.dwt_max_level(len(r), WAVELET_NAME))
        coeffs = pywt.wavedec(r, WAVELET_NAME, level=max_level)
        # coeffs[0] = approximation (lowest freq), coeffs[1..] = details (high → low freq)

        level_to_band = {
            1: "noise",
            2: "noise",
            3: "tactical",
            4: "tactical",
            5: "sector",
            6: "business",
            7: "business",
            8: "macro",
        }

        band_energies: dict[str, float] = {b: 0.0 for b in FREQ_BANDS}
        band_signals: dict[str, float] = {b: 0.0 for b in FREQ_BANDS}
        band_counts: dict[str, int] = {b: 0 for b in FREQ_BANDS}

        # Detail coefficients (coeffs[1] = highest freq)
        for level_idx, coef in enumerate(coeffs[1:], start=1):
            band = level_to_band.get(level_idx, "noise")
            energy = float(np.sum(coef ** 2))
            band_energies[band] += energy
            # Signal: sign of the mean of recent coefficients
            recent = coef[-min(4, len(coef)):]
            sig = float(np.tanh(recent.mean() / (np.std(coef) + 1e-12) * 2))
            band_signals[band] += sig
            band_counts[band] += 1

        # Approximation → macro
        approx = coeffs[0]
        energy = float(np.sum(approx ** 2))
        band_energies["macro"] += energy
        recent_approx = approx[-min(4, len(approx)):]
        sig_macro = float(np.tanh(recent_approx.mean() / (np.std(approx) + 1e-12) * 2))
        band_signals["macro"] += sig_macro
        band_counts["macro"] += 1

        # Average within each band
        result = {}
        for band in FREQ_BANDS:
            n = band_counts[band]
            result[band] = float(band_signals[band] / n) if n > 0 else 0.0

        return result

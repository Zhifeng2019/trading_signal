"""
Configuration for the trading signal generation system.

Frequency band definitions and their relationship to investment horizons:

  MACRO band      (period > 252 days)  : Economic cycles, interest rate cycles
  BUSINESS band   (63 - 252 days)      : Earnings cycles, sector rotations
  SECTOR band     (21 - 63 days)       : Industry-level momentum
  TACTICAL band   (5 - 21 days)        : Short-term price patterns
  NOISE band      (<5 days)            : Market microstructure, cross-asset correlations

Investment horizon maps to frequency bands via a Gaussian kernel in log-frequency space,
so a 6-month horizon emphasises both BUSINESS and SECTOR bands, not just one.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple

# ──────────────────────────────────────────────
# Frequency band definitions (cycles per trading day)
# ──────────────────────────────────────────────

FREQ_BANDS: Dict[str, Tuple[float, float]] = {
    "macro":    (0.0,    1 / 252),   # period > 252 days
    "business": (1 / 252, 1 / 63),  # 63 – 252 days
    "sector":   (1 / 63,  1 / 21),  # 21 – 63 days
    "tactical": (1 / 21,  1 / 5),   # 5 – 21 days
    "noise":    (1 / 5,   0.5),     # < 5 days (Nyquist limit for daily data)
}

# Human-readable characteristic periods (trading days) per band
BAND_PERIODS: Dict[str, float] = {
    "macro":    504,   # ~2 years
    "business": 126,   # ~6 months
    "sector":   42,    # ~2 months
    "tactical": 10,    # ~2 weeks
    "noise":    2,     # 2 days
}

# ──────────────────────────────────────────────
# Horizon → band weight mapping
# Weights are computed from a Gaussian in log-frequency space.
# These are the log10(period) centres and widths for each band.
# ──────────────────────────────────────────────

BAND_LOG_PERIOD_CENTRE: Dict[str, float] = {
    "macro":    2.70,   # log10(504)
    "business": 2.10,   # log10(126)
    "sector":   1.62,   # log10(42)
    "tactical": 1.00,   # log10(10)
    "noise":    0.30,   # log10(2)
}

BAND_LOG_PERIOD_WIDTH: float = 0.60   # half-sigma in log10(period) space

# ──────────────────────────────────────────────
# Signal source weights per band
# How much each data source contributes to a given frequency band.
# ──────────────────────────────────────────────

BAND_SOURCE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "macro":    {"macro": 0.60, "fundamental": 0.30, "technical": 0.10},
    "business": {"fundamental": 0.50, "macro": 0.25, "technical": 0.25},
    "sector":   {"technical": 0.45, "fundamental": 0.35, "macro": 0.20},
    "tactical": {"technical": 0.70, "fundamental": 0.20, "macro": 0.10},
    "noise":    {"technical": 0.85, "fundamental": 0.10, "macro": 0.05},
}

# ──────────────────────────────────────────────
# Return-target risk adjustment
# Higher target return → allow more weight on volatile (high-freq) bands.
# Modelled as a sigmoid shift along the log-period axis.
# ──────────────────────────────────────────────

RISK_SIGMA_SHIFT: float = 0.40   # log10-period units per unit of excess Sharpe demand

# ──────────────────────────────────────────────
# Default data parameters
# ──────────────────────────────────────────────

TRADING_DAYS_PER_YEAR: int = 252
DEFAULT_LOOKBACK_YEARS: int = 5
MIN_DATA_POINTS: int = 252           # require at least 1 year for FFT

# ──────────────────────────────────────────────
# Macro FRED series IDs
# ──────────────────────────────────────────────

FRED_SERIES: Dict[str, str] = {
    "yield_10y":      "DGS10",      # 10-year Treasury yield
    "yield_2y":       "DGS2",       # 2-year Treasury yield
    "yield_3m":       "DTB3",       # 3-month T-bill
    "fed_funds":      "FEDFUNDS",   # Fed funds effective rate
    "hy_spread":      "BAMLH0A0HYM2",  # ICE BofA US HY spread
    "ig_spread":      "BAMLC0A0CM",    # ICE BofA US IG spread
    "vix":            "VIXCLS",     # CBOE VIX
    "cpi_yoy":        "CPIAUCSL",   # CPI (all items)
    "pce_yoy":        "PCEPI",      # PCE price index
    "ism_mfg":        "MANEMP",     # Manufacturing employment (proxy for PMI direction)
    "unemployment":   "UNRATE",     # Unemployment rate
    "gdp_growth":     "A191RL1Q225SBEA",  # Real GDP growth, quarterly
}

# ──────────────────────────────────────────────
# Wavelet parameters
# ──────────────────────────────────────────────

WAVELET_NAME: str = "db4"        # Daubechies-4 wavelet
WAVELET_MAX_LEVEL: int = 8       # decomposition levels (covers up to 256-day cycles)

# ──────────────────────────────────────────────
# Signal scoring
# ──────────────────────────────────────────────

@dataclass
class SignalConfig:
    horizon_days: int = 63             # investment horizon in trading days
    target_return_annual: float = 0.12 # annualised target return (e.g. 0.12 = 12 %)
    risk_free_rate: float = 0.045      # annualised risk-free rate
    lookback_years: int = DEFAULT_LOOKBACK_YEARS
    n_dominant_freqs: int = 5          # top-N dominant frequencies to track
    use_wavelets: bool = True          # supplement FFT with DWT decomposition
    cross_asset_tickers: list = field(default_factory=lambda: [
        "SPY", "QQQ", "IWM", "TLT", "GLD", "USO", "UUP"
    ])

    @property
    def target_sharpe(self) -> float:
        excess = self.target_return_annual - self.risk_free_rate
        return max(excess / 0.15, 0.0)   # assume 15% vol baseline

    @property
    def lookback_days(self) -> int:
        return self.lookback_years * TRADING_DAYS_PER_YEAR

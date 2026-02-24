"""
Macro data fetcher – FRED economic indicators.

Derives a macro regime score in [-1, +1] that feeds primarily into the
MACRO and BUSINESS frequency bands.

Regime interpretation
---------------------
  +1 : Strong expansion – risk-on, equities favoured
   0 : Neutral / uncertain
  -1 : Contraction / credit stress – risk-off

Dimensions considered
---------------------
  1. Yield curve slope (10Y – 2Y)    → leading recession indicator
  2. Credit spreads (HY OAS)         → risk appetite
  3. Monetary policy stance          → Fed funds vs neutral rate
  4. Inflation regime                → CPI trend
  5. Growth momentum                 → GDP trend / unemployment rate
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd

try:
    from fredapi import Fred
    _FRED_AVAILABLE = True
except ImportError:
    _FRED_AVAILABLE = False

from trading_signals.config import FRED_SERIES, TRADING_DAYS_PER_YEAR


class MacroDataFetcher:
    """
    Fetches FRED series and computes a macro regime signal.

    Parameters
    ----------
    fred_api_key : str, optional
        FRED API key.  If None the class will attempt to read
        ``FRED_API_KEY`` from the environment.  If FRED is unavailable
        a neutral (0.0) signal is returned with a warning.
    """

    def __init__(self, fred_api_key: Optional[str] = None):
        self._fred: Optional["Fred"] = None
        if _FRED_AVAILABLE:
            import os
            key = fred_api_key or os.environ.get("FRED_API_KEY", "")
            if key:
                try:
                    self._fred = Fred(api_key=key)
                except Exception:
                    pass
        if self._fred is None:
            warnings.warn(
                "FRED API not available.  Install 'fredapi' and set FRED_API_KEY "
                "to enable macro signals.  Returning neutral score (0.0).",
                stacklevel=2,
            )

    # ──────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────

    def get_series(self, series_id: str, lookback_years: int = 5) -> pd.Series:
        """Fetch a FRED series as a daily-frequency pd.Series."""
        if self._fred is None:
            return pd.Series(dtype=float)
        try:
            raw = self._fred.get_series(series_id)
            raw = raw.dropna().sort_index()
            raw.index = pd.to_datetime(raw.index)
            cutoff = pd.Timestamp.today() - pd.DateOffset(years=lookback_years)
            return raw[raw.index >= cutoff]
        except Exception as e:
            warnings.warn(f"Could not fetch FRED series '{series_id}': {e}")
            return pd.Series(dtype=float)

    def get_yield_curve(self) -> pd.DataFrame:
        """10Y yield, 2Y yield, and their spread (10Y–2Y)."""
        y10 = self.get_series(FRED_SERIES["yield_10y"])
        y2 = self.get_series(FRED_SERIES["yield_2y"])
        df = pd.DataFrame({"yield_10y": y10, "yield_2y": y2}).dropna()
        df["spread_10y_2y"] = df["yield_10y"] - df["yield_2y"]
        return df

    def get_credit_spreads(self) -> pd.DataFrame:
        """HY and IG option-adjusted spreads."""
        hy = self.get_series(FRED_SERIES["hy_spread"])
        ig = self.get_series(FRED_SERIES["ig_spread"])
        df = pd.DataFrame({"hy_spread": hy, "ig_spread": ig}).dropna()
        return df

    def get_vix(self) -> pd.Series:
        return self.get_series(FRED_SERIES["vix"])

    # ──────────────────────────────────────────
    # Regime signals in [-1, +1]
    # ──────────────────────────────────────────

    def yield_curve_signal(self) -> float:
        """
        10Y–2Y spread signal.
        Positive spread (+) → expansion; inverted (–) → recession risk.
        """
        if self._fred is None:
            return 0.0
        df = self.get_yield_curve()
        if df.empty:
            return 0.0
        spread = df["spread_10y_2y"].iloc[-1]
        # Map: spread > +1.5 → +1, spread < -0.5 → -1
        return float(np.clip(_linear_score(spread, low=-0.5, high=1.5), -1.0, 1.0))

    def credit_spread_signal(self) -> float:
        """
        HY spread signal.
        Low spreads → risk-on (+1); wide spreads → risk-off (-1).
        """
        if self._fred is None:
            return 0.0
        df = self.get_credit_spreads()
        if df.empty:
            return 0.0
        hy = df["hy_spread"].iloc[-1]
        # Map: <3.0 → +1, >8.0 → -1
        return float(np.clip(_linear_score(-hy, low=-8.0, high=-3.0), -1.0, 1.0))

    def monetary_policy_signal(self) -> float:
        """
        Fed funds relative to its own 2-year moving average.
        Cutting cycle (+), hiking cycle (-).
        """
        if self._fred is None:
            return 0.0
        ff = self.get_series(FRED_SERIES["fed_funds"])
        if len(ff) < 24:
            return 0.0
        # Monthly data – 24 months window
        ma = ff.rolling(24).mean()
        diff = ff - ma
        recent_diff = diff.iloc[-1]
        # Negative diff = rates below average = dovish = positive for equities
        return float(np.clip(_linear_score(-recent_diff, low=-2.0, high=2.0), -1.0, 1.0))

    def inflation_signal(self) -> float:
        """
        CPI trend signal.
        Moderate inflation (<3 %) rising slightly → neutral/positive.
        High or accelerating inflation → negative (Fed will hike).
        """
        if self._fred is None:
            return 0.0
        cpi = self.get_series(FRED_SERIES["cpi_yoy"])
        if len(cpi) < 13:
            return 0.0
        # Compute YoY change of CPI index
        cpi_yoy = cpi.pct_change(12) * 100  # approximate YoY %
        cpi_yoy = cpi_yoy.dropna()
        if cpi_yoy.empty:
            return 0.0
        level = cpi_yoy.iloc[-1]
        trend = cpi_yoy.diff(3).iloc[-1]   # 3-month acceleration
        level_score = _linear_score(-level, low=-5.0, high=-1.5)   # lower CPI = better
        trend_score = _linear_score(-trend, low=-0.5, high=0.5)
        return float(np.clip(0.6 * level_score + 0.4 * trend_score, -1.0, 1.0))

    def growth_signal(self) -> float:
        """
        Unemployment rate trend as a proxy for economic growth direction.
        Falling unemployment → expansion (+1).
        """
        if self._fred is None:
            return 0.0
        ur = self.get_series(FRED_SERIES["unemployment"])
        if len(ur) < 12:
            return 0.0
        # 3-month change in unemployment rate
        change_3m = ur.diff(3).iloc[-1]
        # Falling unemployment (negative change) = positive
        return float(np.clip(_linear_score(-change_3m, low=-1.0, high=1.0), -1.0, 1.0))

    def regime_score(self) -> dict[str, float]:
        """
        Composite macro regime score and sub-components.

        Returns a dict with keys:
            composite        : weighted overall score in [-1, +1]
            yield_curve      : yield-curve sub-score
            credit           : credit-spread sub-score
            monetary_policy  : monetary-policy sub-score
            inflation        : inflation sub-score
            growth           : growth sub-score
        """
        components = {
            "yield_curve":     (self.yield_curve_signal(),     0.30),
            "credit":          (self.credit_spread_signal(),   0.25),
            "monetary_policy": (self.monetary_policy_signal(), 0.20),
            "inflation":       (self.inflation_signal(),       0.15),
            "growth":          (self.growth_signal(),          0.10),
        }
        composite = sum(score * weight for score, weight in components.values())
        return {
            "composite": float(np.clip(composite, -1.0, 1.0)),
            **{k: float(v[0]) for k, v in components.items()},
        }

    def get_macro_feature_vector(
        self,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """
        Returns a daily time-series DataFrame of raw macro features
        (aligned to business days) for use in time-series models.
        Features: yield_10y, yield_2y, spread_10y_2y, hy_spread,
                  fed_funds, cpi_yoy (monthly interpolated).
        """
        if self._fred is None:
            return pd.DataFrame()

        series_keys = [
            ("yield_10y",   FRED_SERIES["yield_10y"]),
            ("yield_2y",    FRED_SERIES["yield_2y"]),
            ("hy_spread",   FRED_SERIES["hy_spread"]),
            ("fed_funds",   FRED_SERIES["fed_funds"]),
            ("vix",         FRED_SERIES["vix"]),
        ]
        frames = {}
        for name, sid in series_keys:
            s = self.get_series(sid, lookback_years=lookback_days // TRADING_DAYS_PER_YEAR + 1)
            if not s.empty:
                frames[name] = s

        if not frames:
            return pd.DataFrame()

        df = pd.DataFrame(frames)
        df = df.resample("B").last().ffill()
        df["spread_10y_2y"] = df.get("yield_10y", np.nan) - df.get("yield_2y", np.nan)

        cutoff = pd.Timestamp.today() - pd.offsets.BDay(lookback_days)
        return df[df.index >= cutoff].dropna(how="all")


# ──────────────────────────────────────────────
# Utilities (duplicated locally to keep module self-contained)
# ──────────────────────────────────────────────

def _linear_score(value: float, low: float, high: float) -> float:
    mid = (low + high) / 2
    half_range = (high - low) / 2 + 1e-9
    return float(np.clip((value - mid) / half_range, -1.0, 1.0))

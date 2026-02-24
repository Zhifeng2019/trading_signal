"""
Fundamental data fetcher – financial ratios, earnings, and quality metrics.

Sources: yfinance (info, financials, balance_sheet, cashflow).
Data is cross-sectional (point-in-time per ticker) and updated quarterly.

Signals derived here feed primarily into the BUSINESS and SECTOR frequency bands.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


# Map yfinance keys to our canonical names
_INFO_MAP = {
    # Valuation
    "trailingPE":          "pe_trailing",
    "forwardPE":           "pe_forward",
    "priceToBook":         "pb",
    "enterpriseToEbitda":  "ev_ebitda",
    "enterpriseToRevenue": "ev_revenue",
    "priceToSalesTrailing12Months": "ps",
    # Profitability / Quality
    "returnOnEquity":      "roe",
    "returnOnAssets":      "roa",
    "grossMargins":        "gross_margin",
    "operatingMargins":    "op_margin",
    "profitMargins":       "net_margin",
    # Growth
    "revenueGrowth":       "revenue_growth_yoy",
    "earningsGrowth":      "earnings_growth_yoy",
    "earningsQuarterlyGrowth": "earnings_growth_qoq",
    # Leverage / Solvency
    "debtToEquity":        "debt_to_equity",
    "currentRatio":        "current_ratio",
    "quickRatio":          "quick_ratio",
    # Dividend
    "dividendYield":       "div_yield",
    "payoutRatio":         "payout_ratio",
    # Market
    "beta":                "beta",
    "marketCap":           "market_cap",
    "sector":              "sector",
    "industry":            "industry",
}


class FundamentalDataFetcher:
    """
    Fetches and normalises fundamental data for a single ticker.

    All ratio metrics are returned as plain floats (NaN if unavailable).
    The ``score`` methods convert raw ratios into [-1, +1] signals using
    sector-relative z-scores where possible, or absolute thresholds.
    """

    def __init__(self):
        pass   # stateless; caching is at the yf.Ticker level

    # ──────────────────────────────────────────
    # Raw data
    # ──────────────────────────────────────────

    def get_metrics(self, ticker: str) -> dict[str, Any]:
        """Return dict of canonical fundamental metrics."""
        info = self._get_info(ticker)
        result: dict[str, Any] = {}
        for yf_key, canon_key in _INFO_MAP.items():
            result[canon_key] = info.get(yf_key, np.nan)
        # Ensure numeric types where expected
        for k, v in result.items():
            if k not in ("sector", "industry"):
                try:
                    result[k] = float(v) if v is not None else np.nan
                except (TypeError, ValueError):
                    result[k] = np.nan
        return result

    def get_earnings_history(self, ticker: str) -> pd.DataFrame:
        """Quarterly EPS actuals vs estimates → surprise series."""
        t = yf.Ticker(ticker)
        try:
            earnings = t.earnings_dates
            if earnings is None or earnings.empty:
                return pd.DataFrame()
            earnings = earnings.copy()
            earnings.index = pd.to_datetime(earnings.index)
            earnings = earnings.sort_index()
            if "Reported EPS" in earnings.columns and "EPS Estimate" in earnings.columns:
                earnings["surprise"] = (
                    earnings["Reported EPS"] - earnings["EPS Estimate"]
                ) / earnings["EPS Estimate"].abs().clip(lower=1e-6)
            return earnings
        except Exception:
            return pd.DataFrame()

    def get_revenue_trend(self, ticker: str) -> pd.Series:
        """Annual revenue from income statement (most-recent 4 periods)."""
        t = yf.Ticker(ticker)
        try:
            fin = t.financials
            if fin is None or fin.empty:
                return pd.Series(dtype=float)
            if "Total Revenue" in fin.index:
                rev = fin.loc["Total Revenue"].sort_index()
                return rev.astype(float)
        except Exception:
            pass
        return pd.Series(dtype=float)

    # ──────────────────────────────────────────
    # Fundamental signals in [-1, +1]
    # ──────────────────────────────────────────

    def valuation_signal(self, ticker: str) -> float:
        """
        Cheap = +1, Expensive = -1.
        Uses forward P/E, P/B, EV/EBITDA.  Missing metrics are dropped.
        Thresholds are broad market heuristics; ideally replace with
        sector-relative z-scores when running a whole universe.
        """
        m = self.get_metrics(ticker)
        scores = []

        # Forward P/E: <15 is cheap, >30 is expensive
        if not np.isnan(m.get("pe_forward", np.nan)):
            pe = m["pe_forward"]
            if pe > 0:
                scores.append(_sigmoid_score(pe, cheap_val=15, expensive_val=30))

        # P/B: <1.5 cheap, >4 expensive
        if not np.isnan(m.get("pb", np.nan)):
            pb = m["pb"]
            if pb > 0:
                scores.append(_sigmoid_score(pb, cheap_val=1.5, expensive_val=4.0))

        # EV/EBITDA: <10 cheap, >20 expensive
        if not np.isnan(m.get("ev_ebitda", np.nan)):
            ev = m["ev_ebitda"]
            if ev > 0:
                scores.append(_sigmoid_score(ev, cheap_val=10, expensive_val=20))

        return float(np.nanmean(scores)) if scores else 0.0

    def quality_signal(self, ticker: str) -> float:
        """
        High quality = +1, Low quality = -1.
        ROE, margins, low leverage.
        """
        m = self.get_metrics(ticker)
        scores = []

        # ROE: >15% high quality, <0% low
        if not np.isnan(m.get("roe", np.nan)):
            scores.append(_linear_score(m["roe"], low=0.0, high=0.20))

        # Operating margin: >20% good, <0% bad
        if not np.isnan(m.get("op_margin", np.nan)):
            scores.append(_linear_score(m["op_margin"], low=0.0, high=0.25))

        # Debt/Equity: <50% low leverage (good), >200% high leverage (bad)
        if not np.isnan(m.get("debt_to_equity", np.nan)):
            de = m["debt_to_equity"]
            if de >= 0:
                scores.append(_linear_score(-de, low=-200, high=-0))

        return float(np.nanmean(scores)) if scores else 0.0

    def earnings_momentum_signal(self, ticker: str) -> float:
        """
        Positive earnings surprises and growth = +1.
        Returns signal in [-1, +1].
        """
        m = self.get_metrics(ticker)
        scores = []

        # YoY earnings growth
        if not np.isnan(m.get("earnings_growth_yoy", np.nan)):
            g = m["earnings_growth_yoy"]
            scores.append(_linear_score(g, low=-0.20, high=0.30))

        # QoQ earnings growth (more recent)
        if not np.isnan(m.get("earnings_growth_qoq", np.nan)):
            gq = m["earnings_growth_qoq"]
            scores.append(_linear_score(gq, low=-0.10, high=0.20))

        # Most-recent earnings surprise
        history = self.get_earnings_history(ticker)
        if not history.empty and "surprise" in history.columns:
            recent = history["surprise"].dropna()
            if len(recent) >= 1:
                avg_surprise = recent.iloc[-4:].mean() if len(recent) >= 4 else recent.mean()
                scores.append(_linear_score(avg_surprise, low=-0.10, high=0.10))

        return float(np.nanmean(scores)) if scores else 0.0

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    @lru_cache(maxsize=64)
    def _get_info(self, ticker: str) -> dict:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = yf.Ticker(ticker)
            try:
                return t.info or {}
            except Exception:
                return {}


# ──────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────

def _sigmoid_score(value: float, cheap_val: float, expensive_val: float) -> float:
    """Map value → [-1, +1]: cheap_val → +1, expensive_val → -1."""
    mid = (cheap_val + expensive_val) / 2
    scale = (expensive_val - cheap_val) / 2
    normalised = (value - mid) / (scale + 1e-9)
    return float(-np.tanh(normalised))   # negative because higher ratio = more expensive


def _linear_score(value: float, low: float, high: float) -> float:
    """Linearly map [low, high] → [-1, +1], clipped."""
    mid = (low + high) / 2
    half_range = (high - low) / 2 + 1e-9
    return float(np.clip((value - mid) / half_range, -1.0, 1.0))

"""
Market data fetcher – OHLCV via yfinance.

Provides:
  - Daily adjusted OHLCV
  - Log returns and volume-weighted returns
  - Rolling realised volatility
  - Cross-asset return matrix (used by the frequency decomposer
    to separate idiosyncratic from systematic frequency components)
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from trading_signals.config import TRADING_DAYS_PER_YEAR


class MarketDataFetcher:
    """Thin wrapper around yfinance with caching and derived series."""

    def __init__(self, lookback_years: int = 5):
        self.lookback_years = lookback_years

    # ──────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────

    def get_ohlcv(self, ticker: str) -> pd.DataFrame:
        """Return adjusted OHLCV DataFrame indexed by date."""
        raw = self._download(ticker)
        return raw[["Open", "High", "Low", "Close", "Volume"]].copy()

    def get_returns(
        self,
        ticker: str,
        method: str = "log",
        fillna: bool = True,
    ) -> pd.Series:
        """
        Compute daily returns.

        Parameters
        ----------
        method : 'log' | 'simple'
        fillna : forward-fill any missing values before computing returns
        """
        close = self._download(ticker)["Close"]
        if fillna:
            close = close.ffill()
        if method == "log":
            ret = np.log(close / close.shift(1)).dropna()
        else:
            ret = close.pct_change().dropna()
        ret.name = ticker
        return ret

    def get_volume(self, ticker: str) -> pd.Series:
        raw = self._download(ticker)
        vol = raw["Volume"].astype(float)
        vol.name = ticker
        return vol

    def get_volume_weighted_returns(self, ticker: str) -> pd.Series:
        """Returns multiplied by normalised volume (amplifies high-volume moves)."""
        rets = self.get_returns(ticker)
        vol = self.get_volume(ticker).reindex(rets.index).ffill()
        vol_norm = vol / vol.rolling(21).mean().clip(lower=1e-9)
        vw = (rets * vol_norm).dropna()
        vw.name = ticker
        return vw

    def get_realised_vol(
        self,
        ticker: str,
        window: int = 21,
        annualise: bool = True,
    ) -> pd.Series:
        """Rolling realised volatility."""
        rets = self.get_returns(ticker)
        rv = rets.rolling(window).std()
        if annualise:
            rv = rv * np.sqrt(TRADING_DAYS_PER_YEAR)
        rv.name = ticker
        return rv.dropna()

    def get_cross_asset_returns(
        self,
        tickers: list[str],
        primary_ticker: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch log-returns for a basket of tickers and align on common dates.
        Used by the frequency decomposer to identify systematic vs idiosyncratic
        frequency components.

        If primary_ticker is given it is always included and the DataFrame
        is aligned to its date index.
        """
        all_tickers = list(dict.fromkeys(
            ([primary_ticker] if primary_ticker else []) + tickers
        ))
        frames = {}
        for t in all_tickers:
            try:
                frames[t] = self.get_returns(t)
            except Exception:
                warnings.warn(f"Could not fetch {t}, skipping.")
        df = pd.DataFrame(frames).dropna(how="all")
        if primary_ticker and primary_ticker in df.columns:
            df = df.dropna(subset=[primary_ticker])
        return df

    def get_price_momentum(
        self,
        ticker: str,
        windows: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Multi-period price momentum (total return over window).
        Classic factor: past 12-1 month return, 6-1 month, etc.
        """
        if windows is None:
            windows = [5, 21, 63, 126, 252]
        close = self._download(ticker)["Close"].ffill()
        result = {}
        for w in windows:
            result[f"mom_{w}d"] = close.pct_change(w)
        return pd.DataFrame(result).dropna()

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    @lru_cache(maxsize=64)
    def _download(self, ticker: str) -> pd.DataFrame:
        period = f"{self.lookback_years}y"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for '{ticker}'")
        # Flatten MultiIndex columns produced by newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df

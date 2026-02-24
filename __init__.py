"""
Trading Signal Generation System
=================================
A frequency-domain approach to investment signal generation for publicly
listed equities.  Signals are parameterised by investment horizon and
target return so that the same underlying data produces different
recommendations depending on the investor's objective.

Quick start
-----------
>>> from trading_signals import SignalPipeline, SignalConfig
>>> cfg = SignalConfig(horizon_days=63, target_return_annual=0.15)
>>> pipeline = SignalPipeline(cfg)
>>> result = pipeline.run(["AAPL", "MSFT", "GOOGL"])
>>> print(result.ranked_table())
"""

from trading_signals.config import SignalConfig, FREQ_BANDS, BAND_PERIODS
from trading_signals.signals.signal_combiner import SignalCombiner
from trading_signals.portfolio.signal_ranker import SignalRanker


class SignalPipeline:
    """End-to-end pipeline: fetch data → decompose → score → rank."""

    def __init__(self, config: SignalConfig | None = None):
        self.config = config or SignalConfig()
        self._combiner = SignalCombiner(self.config)
        self._ranker = SignalRanker(self.config)

    def run(self, tickers: list[str]):
        signals = {}
        for ticker in tickers:
            try:
                signals[ticker] = self._combiner.compute(ticker)
            except Exception as exc:
                print(f"[WARN] {ticker}: {exc}")
        return self._ranker.rank(signals)


__all__ = [
    "SignalPipeline",
    "SignalConfig",
    "FREQ_BANDS",
    "BAND_PERIODS",
]

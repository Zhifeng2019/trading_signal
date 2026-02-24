"""
Signal ranker – converts a dict of TickerSignal objects into an ordinal
ranking and an optional simple long/short portfolio.

Ranking methodology
--------------------
1. Sort by composite_score (descending for longs).
2. Break ties using confidence × |composite_score|.
3. Apply minimum-confidence filter: signals with confidence < threshold
   are demoted to HOLD regardless of direction.
4. Compute cross-sectional z-scores so that relative strength across
   stocks is visible even when all absolute signals are positive.

Portfolio construction (simple, not optimised)
----------------------------------------------
- Top-N by signal score → LONG basket
- Bottom-N by signal score → SHORT basket
- Weights are proportional to |composite_score| × confidence
  (not volatility-adjusted – add a covariance matrix for a proper
  mean-variance optimiser)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from trading_signals.config import SignalConfig
from trading_signals.signals.signal_combiner import TickerSignal


@dataclass
class RankingResult:
    """Output of SignalRanker.rank()."""
    signals: dict[str, TickerSignal]
    ranked_tickers: list[str]             # sorted best → worst
    scores: pd.Series                     # composite scores, indexed by ticker
    z_scores: pd.Series                   # cross-sectional z-scores
    confidence: pd.Series

    long_basket: list[str]
    short_basket: list[str]
    long_weights: dict[str, float]
    short_weights: dict[str, float]

    def ranked_table(self) -> pd.DataFrame:
        """Return a human-readable ranked DataFrame."""
        rows = []
        for t in self.ranked_tickers:
            sig = self.signals[t]
            rows.append({
                "ticker":     t,
                "direction":  sig.direction,
                "score":      round(sig.composite_score, 4),
                "z_score":    round(self.z_scores.get(t, 0.0), 2),
                "confidence": f"{sig.confidence:.1%}",
                "horizon":    f"{sig.horizon_days}d",
                "target_ret": f"{sig.target_return:.1%}",
            })
        return pd.DataFrame(rows).set_index("ticker")

    def portfolio_table(self) -> pd.DataFrame:
        """Return a combined long/short portfolio DataFrame."""
        rows = []
        for t, w in self.long_weights.items():
            rows.append({"ticker": t, "side": "LONG", "weight": round(w, 4),
                         "score": round(self.scores[t], 4)})
        for t, w in self.short_weights.items():
            rows.append({"ticker": t, "side": "SHORT", "weight": round(w, 4),
                         "score": round(self.scores[t], 4)})
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["side", "weight"], ascending=[True, False])
        return df


class SignalRanker:
    """
    Ranks a universe of TickerSignal objects and builds a simple
    long/short portfolio.

    Parameters
    ----------
    config : SignalConfig
    long_n : int
        Number of stocks in the long basket.
    short_n : int
        Number of stocks in the short basket.
    min_confidence : float
        Signals with confidence below this threshold are demoted to HOLD.
    """

    def __init__(
        self,
        config: SignalConfig,
        long_n: int = 5,
        short_n: int = 5,
        min_confidence: float = 0.20,
    ):
        self.config = config
        self.long_n = long_n
        self.short_n = short_n
        self.min_confidence = min_confidence

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def rank(self, signals: dict[str, TickerSignal]) -> RankingResult:
        """
        Rank a dict of {ticker: TickerSignal} and return a RankingResult.
        """
        if not signals:
            return self._empty_result()

        # Apply confidence filter
        scores = {}
        confidence = {}
        for ticker, sig in signals.items():
            score = sig.composite_score
            conf = sig.confidence
            if conf < self.min_confidence:
                score = 0.0   # demote to neutral
            scores[ticker] = score
            confidence[ticker] = conf

        scores_series = pd.Series(scores)
        confidence_series = pd.Series(confidence)

        # Cross-sectional z-scores
        if scores_series.std() > 1e-9:
            z_scores = (scores_series - scores_series.mean()) / scores_series.std()
        else:
            z_scores = scores_series * 0.0

        # Sort by score (descending)
        ranked = scores_series.sort_values(ascending=False).index.tolist()

        # Long/short baskets
        long_basket, long_weights = self._build_basket(
            scores_series, ranked, top=True, n=self.long_n
        )
        short_basket, short_weights = self._build_basket(
            scores_series, ranked, top=False, n=self.short_n
        )

        return RankingResult(
            signals=signals,
            ranked_tickers=ranked,
            scores=scores_series,
            z_scores=z_scores,
            confidence=confidence_series,
            long_basket=long_basket,
            short_basket=short_basket,
            long_weights=long_weights,
            short_weights=short_weights,
        )

    # ──────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────

    def _build_basket(
        self,
        scores: pd.Series,
        ranked: list[str],
        top: bool,
        n: int,
    ) -> tuple[list[str], dict[str, float]]:
        """Pick top-N or bottom-N and return normalised weights."""
        if top:
            candidates = [t for t in ranked if scores[t] > 0.0][:n]
        else:
            candidates = [t for t in reversed(ranked) if scores[t] < 0.0][:n]

        if not candidates:
            return [], {}

        raw_weights = {t: abs(scores[t]) for t in candidates}
        total = sum(raw_weights.values()) + 1e-12
        weights = {t: w / total for t, w in raw_weights.items()}
        return candidates, weights

    @staticmethod
    def _empty_result() -> RankingResult:
        empty_series = pd.Series(dtype=float)
        return RankingResult(
            signals={},
            ranked_tickers=[],
            scores=empty_series,
            z_scores=empty_series,
            confidence=empty_series,
            long_basket=[],
            short_basket=[],
            long_weights={},
            short_weights={},
        )

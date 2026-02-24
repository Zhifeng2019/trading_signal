"""
Visualisation utilities for the trading signal system.

Charts available
----------------
1. plot_spectrum            – Power spectral density of a return series with
                              frequency-band overlays and dominant cycle markers
2. plot_band_decomposition  – Reconstructed price with each frequency band
                              filtered out separately (stacked subplots)
3. plot_signal_radar        – Radar chart showing per-band signal strength
4. plot_horizon_weights     – How band weights shift across different horizons
5. plot_ranked_signals      – Horizontal bar chart of composite scores for a universe
6. plot_cycle_phase         – Phasor diagram showing where each stock sits in its cycle
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend safe for scripts
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

from trading_signals.config import (
    FREQ_BANDS,
    BAND_PERIODS,
    BAND_LOG_PERIOD_CENTRE,
    SignalConfig,
    TRADING_DAYS_PER_YEAR,
)
from trading_signals.signals.frequency_decomp import FrequencyDecomposer
from trading_signals.signals.signal_combiner import _horizon_band_weights, TickerSignal
from trading_signals.portfolio.signal_ranker import RankingResult


# Colour palette for frequency bands
BAND_COLORS = {
    "macro":    "#2196F3",   # blue
    "business": "#4CAF50",   # green
    "sector":   "#FF9800",   # orange
    "tactical": "#F44336",   # red
    "noise":    "#9E9E9E",   # grey
}


def _require_mpl():
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for visualisation. Install it with: pip install matplotlib")


class SignalVisualizer:
    """Collection of plotting methods for signal analysis."""

    def __init__(self, figsize_default: tuple = (14, 8)):
        _require_mpl()
        self.figsize_default = figsize_default
        self._decomposer = FrequencyDecomposer(n_dominant=10)

    # ──────────────────────────────────────────
    # 1. Power spectrum
    # ──────────────────────────────────────────

    def plot_spectrum(
        self,
        returns: pd.Series,
        ticker: str = "",
        ax: Optional[plt.Axes] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot one-sided power spectral density with frequency-band overlays.

        X-axis: period in trading days (log scale)
        Y-axis: spectral power (log scale)
        """
        r = np.asarray(returns.dropna(), dtype=float)
        n = len(r)
        w = np.hanning(n)
        w /= w.mean()
        fft_vals = np.fft.rfft(r * w)
        freqs = np.fft.rfftfreq(n)
        power = (np.abs(fft_vals) / n) ** 2

        # Convert to period (skip DC)
        valid = freqs > 0
        periods = 1.0 / freqs[valid]
        power_v = power[valid]

        fig, ax_main = (plt.subplots(1, 1, figsize=self.figsize_default)
                        if ax is None else (ax.figure, ax))

        ax_main.plot(periods, power_v, color="#37474F", linewidth=0.8, label="PSD")

        # Shade frequency bands
        for band, (f_lo, f_hi) in FREQ_BANDS.items():
            p_hi = 1 / f_lo if f_lo > 0 else periods.max() * 2
            p_lo = 1 / f_hi if f_hi > 0 else periods.min() * 0.5
            ax_main.axvspan(
                p_lo, p_hi,
                alpha=0.15,
                color=BAND_COLORS[band],
                label=f"{band.capitalize()} ({int(p_lo)}–{int(p_hi)}d)",
            )

        # Mark dominant peaks
        decomp = self._decomposer.decompose(returns, ticker=ticker)
        for comp in decomp.dominant_components:
            ax_main.axvline(comp.period_days, color="black", linestyle="--",
                            linewidth=0.7, alpha=0.6)
            ax_main.text(comp.period_days, power_v.max() * 0.7,
                         f"{comp.period_days:.0f}d",
                         fontsize=7, rotation=90, va="top", ha="right", alpha=0.7)

        ax_main.set_xscale("log")
        ax_main.set_yscale("log")
        ax_main.set_xlabel("Period (trading days)")
        ax_main.set_ylabel("Spectral Power")
        ax_main.set_title(f"Power Spectrum – {ticker or 'Return Series'}")
        ax_main.legend(fontsize=8, ncol=2)
        ax_main.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ──────────────────────────────────────────
    # 2. Band-filtered price decomposition
    # ──────────────────────────────────────────

    def plot_band_decomposition(
        self,
        returns: pd.Series,
        ticker: str = "",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Show how the return series decomposes into each frequency band.
        Each subplot shows the band-filtered signal overlaid on the raw returns.
        """
        r = np.asarray(returns.dropna(), dtype=float)
        n = len(r)
        dates = returns.dropna().index

        fft_vals = np.fft.rfft(r)
        freqs = np.fft.rfftfreq(n)

        bands = list(FREQ_BANDS.keys())
        fig, axes = plt.subplots(len(bands), 1, figsize=(14, 3 * len(bands)),
                                 sharex=True)
        fig.suptitle(f"Frequency-Band Decomposition – {ticker}", fontsize=13)

        for i, band in enumerate(bands):
            f_lo, f_hi = FREQ_BANDS[band]
            mask = np.zeros(len(freqs), dtype=bool)
            if f_hi == 0:
                mask[freqs <= f_lo] = True
            else:
                mask[(freqs >= f_lo) & (freqs < f_hi)] = True

            filtered_fft = np.zeros_like(fft_vals)
            filtered_fft[mask] = fft_vals[mask]
            filtered = np.fft.irfft(filtered_fft, n=n)

            ax = axes[i]
            ax.fill_between(dates, 0, r, alpha=0.2, color="grey", label="Raw returns")
            ax.plot(dates, filtered, color=BAND_COLORS[band], linewidth=1.2,
                    label=f"{band.capitalize()} component")
            ax.axhline(0, color="black", linewidth=0.5)
            ax.set_ylabel("Return")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Date")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ──────────────────────────────────────────
    # 3. Signal radar chart
    # ──────────────────────────────────────────

    def plot_signal_radar(
        self,
        signal: TickerSignal,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Radar/spider chart showing per-band signal strength for one ticker.
        """
        bands = list(FREQ_BANDS.keys())
        n = len(bands)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]   # close the polygon

        values = [signal.band_scores[b] for b in bands]
        values += values[:1]

        fig, ax = plt.subplots(1, 1, figsize=(7, 7),
                               subplot_kw={"polar": True})

        ax.plot(angles, values, "o-", linewidth=2, color="#1565C0")
        ax.fill(angles, values, alpha=0.25, color="#1565C0")

        # Draw reference circles at ±0.5 and ±1
        for r_val, style in [(0.5, "--"), (1.0, "-")]:
            circle = [r_val] * (n + 1)
            ax.plot(angles, circle, style, linewidth=0.5, color="grey", alpha=0.5)
            circle_neg = [-r_val] * (n + 1)
            ax.plot(angles, circle_neg, style, linewidth=0.5, color="grey", alpha=0.5)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([b.capitalize() for b in bands], fontsize=10)
        ax.set_ylim(-1, 1)
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticklabels(["-1", "-0.5", "0", "0.5", "1"], fontsize=7)
        ax.set_title(
            f"{signal.ticker}  |  {signal.direction}  ({signal.composite_score:+.3f})\n"
            f"Horizon: {signal.horizon_days}d  |  Confidence: {signal.confidence:.1%}",
            fontsize=11, pad=15,
        )
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ──────────────────────────────────────────
    # 4. Band weight vs horizon
    # ──────────────────────────────────────────

    def plot_horizon_weights(
        self,
        target_returns: list[float] | None = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Show how frequency-band weights evolve as investment horizon increases,
        for multiple target-return levels.
        """
        if target_returns is None:
            target_returns = [0.08, 0.15, 0.25]

        horizons = [1, 2, 5, 10, 21, 42, 63, 126, 252, 504]
        bands = list(FREQ_BANDS.keys())

        fig, axes = plt.subplots(1, len(target_returns),
                                 figsize=(5 * len(target_returns), 5),
                                 sharey=True)
        if len(target_returns) == 1:
            axes = [axes]

        fig.suptitle("Frequency-Band Weights vs Investment Horizon", fontsize=13)

        for ax, tgt_ret in zip(axes, target_returns):
            band_weights_matrix = {b: [] for b in bands}
            for h in horizons:
                cfg = SignalConfig(horizon_days=h, target_return_annual=tgt_ret)
                w = _horizon_band_weights(cfg)
                for b in bands:
                    band_weights_matrix[b].append(w[b])

            bottom = np.zeros(len(horizons))
            for band in bands:
                vals = np.array(band_weights_matrix[band])
                ax.bar(
                    range(len(horizons)), vals, bottom=bottom,
                    color=BAND_COLORS[band], label=band.capitalize(), alpha=0.85,
                )
                bottom += vals

            ax.set_xticks(range(len(horizons)))
            ax.set_xticklabels([str(h) for h in horizons], rotation=45, ha="right")
            ax.set_xlabel("Horizon (trading days)")
            ax.set_ylabel("Weight")
            ax.set_title(f"Target return: {tgt_ret:.0%}")
            ax.set_ylim(0, 1)
            ax.grid(True, axis="y", alpha=0.3)

        handles = [
            mpatches.Patch(color=BAND_COLORS[b], label=b.capitalize())
            for b in bands
        ]
        axes[-1].legend(handles=handles, loc="upper right", fontsize=8)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ──────────────────────────────────────────
    # 5. Universe ranking bar chart
    # ──────────────────────────────────────────

    def plot_ranked_signals(
        self,
        result: RankingResult,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Horizontal bar chart of composite signal scores for a universe.
        """
        scores = result.scores.sort_values()
        colors = [
            BAND_COLORS["business"] if s > 0 else BAND_COLORS["tactical"]
            for s in scores.values
        ]

        fig, ax = plt.subplots(figsize=(10, max(4, len(scores) * 0.5)))
        bars = ax.barh(scores.index, scores.values, color=colors, alpha=0.8)

        # Confidence annotations
        for bar, ticker in zip(bars, scores.index):
            conf = result.confidence.get(ticker, 0.0)
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{conf:.0%}",
                va="center", fontsize=8, color="grey",
            )

        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Composite Signal Score")
        ax.set_title("Universe Signal Ranking")
        ax.grid(True, axis="x", alpha=0.3)

        # Annotate long/short baskets
        for t in result.long_basket:
            if t in scores.index:
                idx = list(scores.index).index(t)
                ax.get_yticklabels()[idx].set_color(BAND_COLORS["business"])
                ax.get_yticklabels()[idx].set_fontweight("bold")
        for t in result.short_basket:
            if t in scores.index:
                idx = list(scores.index).index(t)
                ax.get_yticklabels()[idx].set_color(BAND_COLORS["tactical"])
                ax.get_yticklabels()[idx].set_fontweight("bold")

        green_patch = mpatches.Patch(color=BAND_COLORS["business"], label="Long basket")
        red_patch = mpatches.Patch(color=BAND_COLORS["tactical"], label="Short basket")
        ax.legend(handles=[green_patch, red_patch], fontsize=9)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

    # ──────────────────────────────────────────
    # 6. Cycle phase phasor diagram
    # ──────────────────────────────────────────

    def plot_cycle_phases(
        self,
        signals: dict[str, TickerSignal],
        band: str = "business",
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Phasor diagram: each stock is plotted on a unit circle according to
        the phase of its dominant cycle in the target frequency band.

        Stocks near the bottom (phase ≈ π) are at cycle troughs → buy zone.
        Stocks near the top (phase ≈ 0) are at cycle peaks → sell zone.
        """
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"aspect": "equal"})

        # Draw circle
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), "k-", linewidth=0.8, alpha=0.3)
        ax.axhline(0, color="grey", linewidth=0.5, alpha=0.4)
        ax.axvline(0, color="grey", linewidth=0.5, alpha=0.4)

        # Zone annotations
        ax.text(0, 1.12, "PEAK (Sell zone)", ha="center", va="bottom",
                fontsize=9, color="red")
        ax.text(0, -1.12, "TROUGH (Buy zone)", ha="center", va="top",
                fontsize=9, color="green")

        for ticker, sig in signals.items():
            # Get dominant component in target band
            tech_detail = sig.technical_detail
            band_score = sig.band_scores.get(band, 0.0)

            # Phase is encoded in the band score as -cos(phase), so:
            # band_score = +1 → phase ≈ π (trough), x=-1, y≈0
            # band_score = -1 → phase ≈ 0 (peak), x=+1, y≈0
            # We approximate phase from band_score
            phase = np.arccos(-band_score)
            x = np.cos(phase)
            y = np.sin(phase)

            color = (BAND_COLORS["business"] if band_score > 0.1
                     else BAND_COLORS["tactical"] if band_score < -0.1
                     else "grey")
            ax.scatter(x, y, s=80, color=color, zorder=5, alpha=0.85)
            ax.annotate(
                ticker, (x, y),
                textcoords="offset points", xytext=(6, 4),
                fontsize=8, fontweight="bold",
            )
            ax.annotate("", (x, y), (0, 0),
                        arrowprops={"arrowstyle": "->", "color": color,
                                    "lw": 1.0, "alpha": 0.5})

        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-1.4, 1.4)
        ax.set_title(f"Cycle Phase Diagram – {band.capitalize()} Band", fontsize=12)
        ax.axis("off")

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig

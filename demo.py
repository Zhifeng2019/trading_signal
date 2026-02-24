"""
demo.py – End-to-end demonstration of the trading signal generation system.

Usage
-----
    python demo.py

Environment variables
---------------------
    FRED_API_KEY   : (optional) FRED API key for macro indicators.
                     Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html
                     Macro signals will be neutral (0) without this key.

What this script does
---------------------
1. Runs the signal pipeline for two different investor profiles:
      A) Long-horizon / conservative  (H=252d, target=10%)
      B) Short-horizon / aggressive   (H=21d,  target=25%)

2. Prints ranked tables for each profile.

3. Generates visualisation charts:
      - Power spectrum for each stock
      - Frequency-band decomposition
      - Signal radar for each stock × profile
      - Band-weight shift across horizons
      - Universe ranking bar chart
      - Cycle phase phasor diagram
"""

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from trading_signals import SignalPipeline, SignalConfig
from trading_signals.data.market_data import MarketDataFetcher
from trading_signals.signals.frequency_decomp import FrequencyDecomposer
from trading_signals.signals.signal_combiner import _horizon_band_weights
from trading_signals.visualization.plots import SignalVisualizer

# ──────────────────────────────────────────────────────────────────────────────
# Universe
# ──────────────────────────────────────────────────────────────────────────────

UNIVERSE = [
    "AAPL",   # Apple       – Large-cap tech
    "MSFT",   # Microsoft   – Large-cap tech
    "GOOGL",  # Alphabet    – Large-cap tech
    "JPM",    # JPMorgan    – Financials
    "XOM",    # ExxonMobil  – Energy
    "JNJ",    # J&J         – Healthcare
    "AMZN",   # Amazon      – Consumer / Cloud
    "NVDA",   # Nvidia      – Semis / AI
    "PG",     # P&G         – Consumer staples
    "BA",     # Boeing      – Industrials
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Helper: section header
# ──────────────────────────────────────────────────────────────────────────────

def section(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# ──────────────────────────────────────────────────────────────────────────────
# Part 1: Show how band weights shift with horizon and target return
# ──────────────────────────────────────────────────────────────────────────────

def demo_band_weights():
    section("Band Weights vs Horizon × Target Return")

    scenarios = [
        (5,   0.30, "5d / 30%"),
        (21,  0.25, "21d / 25%"),
        (63,  0.15, "63d / 15%"),
        (252, 0.10, "252d / 10%"),
    ]
    bands = ["macro", "business", "sector", "tactical", "noise"]

    rows = []
    for h, tr, label in scenarios:
        cfg = SignalConfig(horizon_days=h, target_return_annual=tr)
        w = _horizon_band_weights(cfg)
        row = {"Scenario": label}
        row.update({b.capitalize(): f"{w[b]:.3f}" for b in bands})
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Scenario")
    print(df.to_string())
    print()
    print("Interpretation:")
    print("  - As horizon increases, weight shifts from Tactical/Sector → Business/Macro")
    print("  - Higher target return shifts weight towards shorter cycles (more risk)")


# ──────────────────────────────────────────────────────────────────────────────
# Part 2: Spectral analysis of a single stock
# ──────────────────────────────────────────────────────────────────────────────

def demo_spectral_analysis(ticker: str = "AAPL"):
    section(f"Spectral Analysis – {ticker}")

    fetcher = MarketDataFetcher(lookback_years=5)
    decomposer = FrequencyDecomposer(n_dominant=8, use_wavelets=True)

    print(f"Fetching {ticker} returns...")
    returns = fetcher.get_returns(ticker)
    print(f"  {len(returns)} daily observations  "
          f"({returns.index[0].date()} → {returns.index[-1].date()})")

    result = decomposer.decompose(returns, ticker=ticker)
    print(f"\nSpectral entropy: {result.spectral_entropy:.3f}  "
          f"(0 = pure cycle, 1 = white noise)")

    print("\nDominant spectral components:")
    print(f"  {'Period (d)':>10}  {'Period (mo)':>11}  "
          f"{'Amplitude (%)':>13}  {'Phase (°)':>9}  "
          f"{'Signal':>8}  {'Power (%)':>9}")
    print("  " + "-" * 70)
    for comp in result.dominant_components:
        print(
            f"  {comp.period_days:>10.1f}  "
            f"{comp.period_days/21:>11.1f}  "
            f"{comp.amplitude*100:>13.4f}  "
            f"{np.degrees(comp.phase):>9.1f}  "
            f"{comp.phase_signal:>8.3f}  "
            f"{comp.power_fraction*100:>9.2f}"
        )

    print("\nPer-band signals (FFT):")
    for band, bs in result.band_signals.items():
        print(f"  {band:<12} signal={bs.signal:+.3f}  "
              f"amplitude_score={bs.amplitude_score:.3f}  "
              f"dominant_period={bs.dominant_period:.0f}d")

    if result.wavelet_band_signals:
        print("\nPer-band signals (DWT wavelet):")
        for band, sig in result.wavelet_band_signals.items():
            print(f"  {band:<12} {sig:+.3f}")

    return returns, result


# ──────────────────────────────────────────────────────────────────────────────
# Part 3: Run full pipeline for two investor profiles
# ──────────────────────────────────────────────────────────────────────────────

def demo_full_pipeline():
    section("Full Signal Pipeline")

    profiles = [
        SignalConfig(
            horizon_days=252,
            target_return_annual=0.10,
            lookback_years=5,
        ),
        SignalConfig(
            horizon_days=21,
            target_return_annual=0.25,
            lookback_years=5,
        ),
    ]

    profile_names = [
        "Long-horizon / Conservative  (H=252d, target=10% p.a.)",
        "Short-horizon / Aggressive   (H=21d,  target=25% p.a.)",
    ]

    results = {}
    for cfg, name in zip(profiles, profile_names):
        print(f"\nProfile: {name}")
        print(f"  Running pipeline for {len(UNIVERSE)} stocks...")

        pipeline = SignalPipeline(cfg)
        ranking = pipeline.run(UNIVERSE)

        print(f"\n  {'─'*60}")
        print(f"  Ranked table:")
        print(ranking.ranked_table().to_string())

        print(f"\n  Long basket:  {ranking.long_basket}")
        if ranking.long_weights:
            for t, w in ranking.long_weights.items():
                print(f"    {t}: {w:.1%}")

        print(f"\n  Short basket: {ranking.short_basket}")
        if ranking.short_weights:
            for t, w in ranking.short_weights.items():
                print(f"    {t}: {w:.1%}")

        results[name] = ranking

    return results, profiles


# ──────────────────────────────────────────────────────────────────────────────
# Part 4: Detailed signal breakdown for one ticker × both profiles
# ──────────────────────────────────────────────────────────────────────────────

def demo_signal_detail(ticker: str = "AAPL"):
    section(f"Detailed Signal Breakdown – {ticker}")

    profiles = [
        SignalConfig(horizon_days=252, target_return_annual=0.10),
        SignalConfig(horizon_days=21,  target_return_annual=0.25),
    ]

    from trading_signals.signals.signal_combiner import SignalCombiner

    for cfg in profiles:
        combiner = SignalCombiner(cfg)
        sig = combiner.compute(ticker)
        print()
        print(sig.summary())

    print()
    print("Key observation:")
    print("  The same stock produces DIFFERENT signals depending on the investor's")
    print("  horizon and return target, because different frequency bands are")
    print("  weighted differently.")


# ──────────────────────────────────────────────────────────────────────────────
# Part 5: Generate charts
# ──────────────────────────────────────────────────────────────────────────────

def demo_charts(ranking_results: dict, profiles: list):
    section("Generating Charts")

    viz = SignalVisualizer()
    fetcher = MarketDataFetcher(lookback_years=5)

    # 4a. Band weight vs horizon
    print("  Plotting band weights vs horizon...")
    fig = viz.plot_horizon_weights(
        target_returns=[0.08, 0.15, 0.25],
        save_path=os.path.join(OUTPUT_DIR, "band_weights_vs_horizon.png"),
    )
    import matplotlib.pyplot as plt
    plt.close(fig)

    # 4b. Spectrum for AAPL
    ticker = "AAPL"
    print(f"  Plotting power spectrum for {ticker}...")
    returns = fetcher.get_returns(ticker)
    fig = viz.plot_spectrum(
        returns, ticker=ticker,
        save_path=os.path.join(OUTPUT_DIR, f"spectrum_{ticker}.png"),
    )
    plt.close(fig)

    # 4c. Band decomposition for AAPL
    print(f"  Plotting band decomposition for {ticker}...")
    fig = viz.plot_band_decomposition(
        returns, ticker=ticker,
        save_path=os.path.join(OUTPUT_DIR, f"decomposition_{ticker}.png"),
    )
    plt.close(fig)

    # 4d. Ranking chart for each profile
    profile_labels = ["long_horizon", "short_horizon"]
    for label, ranking in zip(profile_labels, ranking_results.values()):
        print(f"  Plotting ranking chart ({label})...")
        fig = viz.plot_ranked_signals(
            ranking,
            save_path=os.path.join(OUTPUT_DIR, f"ranking_{label}.png"),
        )
        plt.close(fig)

    # 4e. Cycle phase phasor for long-horizon profile
    print("  Plotting cycle phase diagram (business band)...")
    first_ranking = list(ranking_results.values())[0]
    fig = viz.plot_cycle_phases(
        first_ranking.signals, band="business",
        save_path=os.path.join(OUTPUT_DIR, "cycle_phases_business.png"),
    )
    plt.close(fig)

    # 4f. Radar chart for AAPL (both profiles)
    from trading_signals.signals.signal_combiner import SignalCombiner
    for label, cfg in zip(profile_labels, profiles):
        combiner = SignalCombiner(cfg)
        sig = combiner.compute(ticker)
        print(f"  Plotting radar chart for {ticker} ({label})...")
        fig = viz.plot_signal_radar(
            sig,
            save_path=os.path.join(OUTPUT_DIR, f"radar_{ticker}_{label}.png"),
        )
        plt.close(fig)

    print(f"\n  All charts saved to: {OUTPUT_DIR}/")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  TRADING SIGNAL GENERATION SYSTEM  –  FREQUENCY DOMAIN APPROACH")
    print("=" * 70)
    print()
    print("Architecture overview:")
    print("  Data sources   →  FFT + DWT decomposition  →  Per-band signals")
    print("  (OHLCV, fundamentals, macro)                  (macro / business /")
    print("                                                  sector / tactical / noise)")
    print("                  ↓")
    print("  Horizon-dependent band weights  (Gaussian in log-period space)")
    print("                  ↓")
    print("  Weighted combination  →  Composite score  →  BUY / HOLD / SELL")

    # Run demos
    demo_band_weights()

    returns, decomp_result = demo_spectral_analysis("AAPL")

    ranking_results, profiles = demo_full_pipeline()

    demo_signal_detail("AAPL")

    try:
        demo_charts(ranking_results, profiles)
    except Exception as e:
        print(f"\n[WARN] Chart generation failed: {e}")
        print("  Install matplotlib and run again.")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

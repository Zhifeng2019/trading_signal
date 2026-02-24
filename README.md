# Trading Signal Generation System

A frequency-domain approach to investment signal generation for publicly listed equities. Signals are parameterised by **investment horizon** and **target return**, so the same underlying data produces different recommendations depending on the investor's objective.

---

## Architecture

```
Data Sources          Decomposition            Per-Band Signals
─────────────         ─────────────────        ──────────────────────
OHLCV (yfinance)  →  FFT + DWT (wavelet)  →   macro
Fundamentals      →  Band-pass filtering  →   business
Macro (FRED)      →  Cross-spectrum       →   sector
                                          →   tactical
                                          →   noise
                          ↓
          Horizon-dependent band weights
          (Gaussian kernel in log-period space)
                          ↓
          Weighted combination → Composite score → BUY / HOLD / SELL
```

---

## Frequency Bands

| Band     | Period Range     | Represents                          |
|----------|-----------------|-------------------------------------|
| Macro    | > 252 days      | Economic cycles, interest rate cycles |
| Business | 63 – 252 days   | Earnings cycles, sector rotations   |
| Sector   | 21 – 63 days    | Industry-level momentum             |
| Tactical | 5 – 21 days     | Short-term price patterns           |
| Noise    | < 5 days        | Market microstructure               |

The investor's **horizon** maps to these bands via a Gaussian kernel in log-frequency space. A 6-month horizon emphasises both BUSINESS and SECTOR bands, not just one.

---

## Signal Sources

Each frequency band draws from three data sources with band-specific weights:

| Band     | Technical | Fundamental | Macro |
|----------|-----------|-------------|-------|
| Macro    | 10%       | 30%         | 60%   |
| Business | 25%       | 50%         | 25%   |
| Sector   | 45%       | 35%         | 20%   |
| Tactical | 70%       | 20%         | 10%   |
| Noise    | 85%       | 10%         | 5%    |

---

## Project Structure

```
trading_signals/
├── __init__.py              # SignalPipeline – end-to-end entry point
├── config.py                # SignalConfig, frequency band definitions
├── requirements.txt
├── demo.py                  # Full demonstration script
│
├── data/
│   ├── market_data.py       # OHLCV, returns, volume via yfinance
│   ├── fundamental_data.py  # Valuation, quality, earnings via yfinance
│   └── macro_data.py        # FRED economic indicators, regime score
│
├── signals/
│   ├── frequency_decomp.py  # FFT + DWT decomposition, cross-spectrum
│   ├── technical_signal.py  # Per-band technical signals
│   ├── fundamental_signal.py# Per-band fundamental signals
│   ├── macro_signal.py      # Per-band macro signals
│   └── signal_combiner.py   # Horizon-weighted composite signal
│
├── portfolio/
│   └── signal_ranker.py     # Cross-sectional ranking, long/short baskets
│
└── visualization/
    └── plots.py             # Power spectrum, radar, phasor, ranking charts
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Optional:**
- `PyWavelets` – enables DWT decomposition (recommended)
- `fredapi` + `FRED_API_KEY` env var – enables macro signals from FRED

---

## Quick Start

```python
from trading_signals import SignalPipeline, SignalConfig

# Define investor profile
cfg = SignalConfig(
    horizon_days=63,            # ~3 month horizon
    target_return_annual=0.15,  # 15% target return
)

# Run pipeline
pipeline = SignalPipeline(cfg)
result = pipeline.run(["AAPL", "MSFT", "GOOGL", "NVDA", "JPM"])

# View ranked table
print(result.ranked_table())

# View portfolio
print(result.portfolio_table())
```

### Two investor profiles compared

```python
from trading_signals.signals.signal_combiner import SignalCombiner, SignalConfig

# Conservative, long-horizon
conservative = SignalConfig(horizon_days=252, target_return_annual=0.10)

# Aggressive, short-horizon
aggressive = SignalConfig(horizon_days=21, target_return_annual=0.25)

for cfg in [conservative, aggressive]:
    combiner = SignalCombiner(cfg)
    sig = combiner.compute("AAPL")
    print(sig.summary())
```

The **same stock** produces different signals for different investor profiles, because the horizon-dependent band weights change which frequency components dominate.

---

## Key Concepts

### FFT Phase Signal
Each sinusoidal cycle `A·cos(2π·f·t + φ)` produces a directional signal based on its current phase:
- Phase ≈ π (at trough, about to rise) → **+1 (buy)**
- Phase ≈ 0 (at peak, about to fall) → **−1 (sell)**

### Horizon → Band Weights
Band weights follow a Gaussian in log-period space centred on the investor's horizon:

```
w(band | H) ∝ exp( −[log₁₀(H) − log₁₀(T_band)]² / (2·σ²) )
```

A higher target return shifts the Gaussian towards shorter periods (higher-risk components).

### Macro Regime Multiplier
The macro regime score (yield curve, credit spreads, Fed policy, inflation, growth) acts as a multiplier `[0.5, 1.5]` on the composite signal:
- Expansion → amplifies bullish signals
- Contraction → attenuates or flips signals

### Confidence Score
Confidence reflects agreement across bands, spectral entropy, and idiosyncratic content:
```
confidence = (1 − band_std) × (1 − 0.5 × entropy) × (0.5 + 0.5 × idio_score)
```

---

## Running the Demo

```bash
# Optional: set FRED API key for macro signals
export FRED_API_KEY=your_key_here

python demo.py
```

The demo:
1. Shows band weights shifting across horizons and return targets
2. Runs spectral analysis on AAPL (dominant cycles, phases, band signals)
3. Runs the full pipeline for 10 stocks under two investor profiles
4. Generates charts saved to `output/`

### Demo Universe
`AAPL`, `MSFT`, `GOOGL`, `JPM`, `XOM`, `JNJ`, `AMZN`, `NVDA`, `PG`, `BA`

---

## Visualisations

| Chart | Description |
|-------|-------------|
| `plot_spectrum` | Power spectral density with band overlays and dominant cycle markers |
| `plot_band_decomposition` | Return series filtered into each frequency band |
| `plot_signal_radar` | Spider chart of per-band signal strength |
| `plot_horizon_weights` | How band weights shift across horizons and return targets |
| `plot_ranked_signals` | Horizontal bar chart of composite scores for a universe |
| `plot_cycle_phases` | Phasor diagram showing where each stock sits in its cycle |

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `FRED_API_KEY` | Optional | FRED API key for macro indicators. Get a free key at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html). Without this, macro signals default to neutral (0). |

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `numpy`, `scipy`, `pandas` | Core numerics |
| `yfinance` | Market & fundamental data |
| `fredapi` | FRED macro data (optional) |
| `PyWavelets` | DWT decomposition (optional) |
| `scikit-learn` | PCA for cross-asset analysis |
| `matplotlib`, `seaborn` | Visualisation |

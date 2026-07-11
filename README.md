# Stocks GA — NN Training and Prediction Done Before the GA

This covers everything built **before** the GA: data fetching, indicator/feature
engineering, NN training, and per-stock inference. GA itslef hasn't been touched, and you'll pick up
from where I left off.

## Key changes from our original idea:

The original idea was for the NN to directly predict the "optimal crossover
rate" for a stock. That label doesn't actually exist anywhere in the data —
there's no way to know the "optimal" rate without already running the GA many
times, which defeats the purpose. To keep this learnable with data we actually
have, we split it into two steps:

1. **NN predicts a real, computable quantity**: forward realized volatility
   (the stock's annualized volatility over the *next* ~21 trading days,
   computed from historical returns). This is a legitimate supervised
   learning target — no GA runs required to generate it.
2. **A separate, hardcoded (non-learned) function** linearly maps that
   predicted volatility into a crossover rate within a fixed valid range
   (default `[0.6, 0.9]`, you can change freely to what you deem appropriate). Higher predicted volatility leads to a higher crossover
   rate. This mapping lives in `model/predict.py::map_to_crossover_rate` and
   is easy to inspect/adjust without retraining anything.

Other adjustments made along the way:

- **Training data ≠ experiment data.** The NN is trained once on a broad
  universe of ~69 tickers across ~8 years (our initial idea of only 1 year was not enough for generalization), completely
  separate from the single stock + ~1 year window used at experiment time.
  Train once (`train_nn.py`), reuse forever (`run_experiment.py`).
- **Two separate roles for indicators (RSI, SMA, etc.):**
  - *Feature-representation indicators* — fixed, standard periods (RSI-14,
    SMA-50/200, etc.), used only as NN input to describe "what kind of
    stock is this." Never touched by the GA.
  - *Chromosome genes* — variable-period versions of the same indicator
    concepts, evolved by the GA as part of the trading strategy. These are
    a completely separate set of values, defined in `CONFIG.chromosome`.
- **NN features are normalized**, not raw price/indicator levels (ratios,
  percentages, z-scores — e.g. `price_to_sma_short` instead of raw SMA).
  Necessary because the training universe spans wildly different price
  scales (like a $15 stock vs. an $800 stock) — raw levels aren't comparable
  across tickers and would hurt generalization.
- **Chromosome gene ranges are static/hardcoded**, not stock-dependent —
  they encode logical validity (e.g. a percentage can't exceed 100), not
  something tailored per stock.
- **All config centralized in `config.py`** as dataclasses, including a
  `ChromosomeConfig` (gene specs + relational constraints) and a `GAConfig`
  placeholder — both intended for you to import from directly rather than
  redefining elsewhere.

## What to run

## Necessary Libraries:
- yfinance
- pandas
- numpy
- scikit-learn
- joblib
- tensorflow

### 1. `train_nn.py` — run once (offline, before any experiments)

Builds the training dataset across the full ticker universe, trains the
volatility-prediction NN, evaluates it, and saves the model + scaler under model/.

```bash
python train_nn.py
```

This takes a while (69 tickers × 8 years of data + training). Only needs to
be re-run if you want to retrain on fresher data or change the model/features.

**Watch the printed validation metrics** (`r2`, `mae`,
`improvement_over_baseline_mae_pct`). If `improvement_over_baseline_mae_pct`
is at or below 0, the model isn't beating "always guess the average
volatility" — worth knowing before trusting its output downstream.

Saves:
- `model/saved/volatility_predictor.keras`
- `model/saved/feature_scaler.pkl`

### 2. `run_experiment.py` — run per stock, per experiment

Loads the already-trained model/scaler (no retraining) and runs the full
pipeline for **one specific stock**:

```bash
python run_experiment.py AAPL
```

or programmatically:

```python
from run_experiment import run_experiment
result = run_experiment("AAPL")
```

Returns a dict:

| Key | Description |
|---|---|
| `ticker` | the ticker requested |
| `window_start`, `window_end` | the ~1-year experiment date window used |
| `ohlcv_with_indicators` | DataFrame — OHLCV + all feature-representation indicators for the window (buffer rows already stripped, no NaNs) |
| `predicted_volatility` | NN's predicted forward volatility for this stock |
| `crossover_rate` | the value to feed into the GA, derived from `predicted_volatility` |

**This is the handoff point.** `result["ohlcv_with_indicators"]` and
`result["crossover_rate"]` are what the GA consumes — one gives you the
stock's data to build a fitness function against, the other is the
GA-level parameter that's supposed to vary by stock.

## Where to plug into the GA

- **`CONFIG.chromosome.genes`** — the 14 chromosome gene specs (name, type,
  min/max) we designed. Use these to generate/validate individuals.
- **`CONFIG.chromosome.constraints`** — relational constraints (e.g.
  `sma_short_window < sma_long_window`) to enforce at generation time, on
  top of the per-gene ranges.
- **`CONFIG.ga`** — placeholder GA parameters (population size, generations,
  mutation rate, etc.). Crossover rate is deliberately *not* a static field
  here since it's produced per-stock by `run_experiment.py`, not fixed.

Fitness function is not yet defined, I'll leave that up to you for how to determine that for a chromosome since you'll be designing the GA.
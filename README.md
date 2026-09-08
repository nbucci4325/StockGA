# Stocks GA — NN Training and Prediction Done Before the GA

## Authors
- [Subquat Siddiqui](https://github.com/Subquat-Siddiqui)
- [Nico Buccilli](https://github.com/nbucci4325)

## Project Overview

This project evolves individual stock trading strategies using a Genetic
Algorithm (GA), guided by a Neural Network (NN) trained across a broad
universe of stocks. For a given ticker, the NN predicts that stock's forward
realized volatility, which is then mapped to a stock-specific crossover rate
used by the GA. Each GA run evolves a population of 14-gene chromosomes
(technical-indicator periods, signal thresholds, and risk-management
parameters) against a backtested-return fitness function, ultimately
producing a best-fit trading strategy and a live BUY/SELL/HOLD decision for
that stock.

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

### Necessary Libraries:
- yfinance
- pandas
- numpy
- scikit-learn
- joblib
- tensorflow

### Python Version
This project was run on Python 3.12.

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

## GA Component
Evolves per-stock trading strategy parameters using the crossover rate predicted by the NN pipeline, then backtests the result and produces a live trading decision.

Chromosone:
14 genes (defined in CONFIG.chromosone, config.py): SMA short/long windows, RSI period + oversold/overbought thresholds, MACD fast/slow/signal periods, Bollinger window + num_std, stop-loss %, take-profit %, position-size %, and signal_combination_weight. Three relational constraints are enforced (e.g. sma_short_window < sma_long_window) via a repair step after crossover/mutation -- genes are rounded to the nearest valid value and clamped to their own bounds rather than swapped outright (that would violate other genes' ranges).

Fitness Function:
For each chromosone, indicators are recomputed at _that_ chromosones own evolved periods onthe stock's OHLCV window. These four signals are computed daily, each normalized to [-1 , 1]:
| Signal | Formula |
|---|---|
| sma_signal | clip((sma_short - sma_long) / sma_long / 0.05, -1, 1) |
| macd_signal | clip(macd_hist / close / 0.05, -1, 1) |
| rsi_signal | linear map of RSI onto [-1 , 1] using the chromosone's own oversold/overbought (oversold -> +1, overbought -> -1) |
| bb_signal | linear map of %B onto [-1 , 1] using the chromosone's own bands (lower band -> +1, upper band -> -1) |

These blend into one combined signal via the chromosone's signal_combination_weight (w). This controls whether the evolved strategy leans trend-following (w -> 1) or reverting (w -> 0).

Backtest rules: long-only, one position at a time. Enter when combined > 0.3; exit when combined < -0.3, or when price hits the chromosone's stop_loss_pct / take_profit_pct from entry (whichever comes first overrides the signal). position_size_pct of current cash is deployed on each entry, starting from $100,000. 

Fitness = total/cumulative return over the backtest window (final equity / initial - 1)

NOTE: Fitness is scored in-sample against what its optimized on. Treat a high fitness as "fit this historical window well", NOT a guarantee of future performance. 

GA Operators:

Selection: torunament (k = 3, from CONFIG.ga.tournament_k)
Crossover (uniform (each gene independently from parent A or B), gated by the NN-predicted crossover_rate (unfix, per-stock)
Mutation: random-reset per gene at CONFIG.ga.mutation_rate
Elitism: top CONFIG.ga.elisism carried over unchanged each generation
Population/generations: CONFIG.ga.population_size (100) x CONFIG.ga.max)gen (300)

### HOW TO RUN

```python
python ga.py APPL

# OPTIONAL second arg: your entry price, if you're already holding a position
# (needed so stop-loss/take-profit evaluate against the real entry, not flat)
python ga.py APPL 187.50

# APPL is simply a placeholder for showcasing purposes
```

Each ga.py run for a ticker produces/appends to:

| File | Contents |
|---|---|
| results.csv | One row per run: ticker, crossover rate used, best fitness, and all 14 best-chromosone gene values. Appended, so multiple tickers build one comparison table. |
| history_<TICKER>.csv | Best fitness per generation, for convergence plots. Overwritten per run. |
| decisions.csv | One row per run: date, price, combined signal, and the resulting BUY/SELL/HOLD decision with reason. Appended. |

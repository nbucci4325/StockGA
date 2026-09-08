# Stock Trading Strategy Evolution with a Neural Network + Genetic Algorithm
 
## Author
- [Subquat Siddiqui](https://github.com/your-username)
- [Nico Buccilli](https://github.com/nbucci4325)

---
 
## What This Project Does
 
This project automatically designs a **trading strategy for a single stock**, tuned specifically to that stock's own behavior, and outputs a live **BUY / SELL / HOLD** decision.
 
It does this in two stages:
 
1. **A Neural Network (NN)** looks at a stock and predicts how volatile that stock is likely to be over the *next* ~21 trading days.
2. **A Genetic Algorithm (GA)** uses that volatility prediction to help evolve a population of candidate trading strategies for that stock — testing thousands of variations of technical-indicator settings and risk rules against the stock's real historical price data — until it converges on the best-performing strategy it found.
The core idea: **volatile stocks and calm stocks shouldn't necessarily be traded with the same strategy-evolution settings.** The NN's job is to quantify *how volatile* a given stock currently looks, and feed that information into how aggressively the GA explores new strategies for it (via a "crossover rate," explained below).
 
---
 
## Why a Neural Network Is Involved at All
 
The original plan was simpler: have the NN directly predict the "ideal" GA setting for a stock. That turned out to be impossible to train, for a basic reason — **there's no existing correct answer to learn from.** You can't know the "ideal" GA setting for a stock without already running the GA many times to find out, which defeats the purpose of predicting it in advance.
 
Instead, this project splits the problem into two parts:
 
1. The NN predicts something **real and measurable**: the stock's forward realized volatility (computed directly from historical price returns — no GA runs needed to generate this as a training label).
2. A **separate, simple, hand-written formula** (not learned by any model) converts that predicted volatility into a GA setting called the *crossover rate*, within a fixed range (default `0.6`–`0.9`). Higher predicted volatility → higher crossover rate. This conversion lives in `model/predict.py::map_to_crossover_rate` and can be adjusted directly without retraining anything.
---
 
## Key Design Decisions (and why they were made)
 
- **The NN is trained once, broadly, and reused for every stock.** It's trained on ~69 different tickers across ~8 years of data (an earlier attempt using only 1 year of data wasn't enough for the model to generalize well). This training step (`train_nn.py`) is separate from actually using the model — once trained, the same model is reused for any individual stock via `run_experiment.py`.
- **Technical indicators serve two different, separate purposes** in this project, which is easy to conflate:
  - *As NN input features* — fixed, standard indicator settings (e.g. RSI over 14 days, SMA over 50/200 days) used only to describe "what kind of stock is this" to the neural network. These never change.
  - *As GA chromosome genes* — a separate, variable set of the same *kinds* of indicators (RSI period, SMA windows, etc.), which the GA is actually allowed to evolve and experiment with as part of a candidate trading strategy.
- **NN inputs are normalized, not raw values** (e.g. a ratio like `price ÷ 50-day SMA`, rather than the SMA's raw dollar value). This matters because the training data spans wildly different stocks — a $15 stock and an $800 stock — and raw price levels aren't comparable across them.
- **The valid ranges for each GA gene are fixed, not stock-specific.** These ranges just enforce basic logical validity (e.g. a percentage can't exceed 100%), not something that should vary per stock.
- **All configuration lives in one place** (`config.py`), including the GA's gene definitions and constraints, so nothing important is hardcoded elsewhere in the codebase.
---
 
## Required Libraries
 
- yfinance
- pandas
- numpy
- scikit-learn
- joblib
- tensorflow
**Python version used:** 3.12
 
---
 
## How to Run This
 
### Step 1 — Train the Neural Network (run once)
 
This builds the training dataset across all ~69 tickers, trains the model that predicts forward volatility, evaluates it, and saves the trained model.
 
```bash
python train_nn.py
```
 
This takes a while, since it's pulling and processing 8 years of data across many tickers. You only need to re-run this if you want to retrain on newer data or change the model itself.
 
**Check the printed validation metrics** (`r2`, `mae`, and especially `improvement_over_baseline_mae_pct`) once it finishes. If `improvement_over_baseline_mae_pct` is at or below 0, the model isn't actually beating a naive "always guess the average volatility" prediction — worth knowing before trusting anything downstream.
 
This saves two files:
- `model/saved/volatility_predictor.keras` (the trained model)
- `model/saved/feature_scaler.pkl` (needed to preprocess new data the same way the model was trained on)
### Step 2 — Run the Experiment for a Specific Stock
 
This loads the already-trained model (no retraining involved) and runs the volatility prediction + crossover-rate calculation for **one stock you choose**:
 
```bash
python run_experiment.py AAPL
```
 
or, if calling it directly from Python code:
 
```python
from run_experiment import run_experiment
result = run_experiment("AAPL")
```
 
This returns a dictionary with:
 
| Key | What it is |
|---|---|
| `ticker` | The stock symbol you asked for |
| `window_start`, `window_end` | The ~1-year date window used for this experiment |
| `ohlcv_with_indicators` | A table of that stock's price data plus all the fixed technical indicators, ready to use |
| `predicted_volatility` | The NN's predicted forward volatility for this stock |
| `crossover_rate` | The GA setting derived from that volatility prediction |
 
**This dictionary is the handoff point to the GA** — `ohlcv_with_indicators` is the data the GA tests strategies against, and `crossover_rate` is the per-stock setting that controls how the GA evolves.
 
### Step 3 — Run the Genetic Algorithm
 
This actually evolves the trading strategy for the stock and produces a live decision:
 
```bash
python ga.py AAPL
 
# Optional second argument: your current entry price, if you're already holding
# a position in this stock (needed so stop-loss/take-profit rules are checked
# against your real entry price, not a blank slate)
python ga.py AAPL 187.50
```
 
(`AAPL` here is just an example ticker — swap in whatever stock you're evaluating.)
 
---
 
## What the Genetic Algorithm Is Actually Evolving
 
Each candidate trading strategy (a "chromosome" in GA terminology) is made up of **14 genes**, defined in `config.py`: SMA short/long windows, RSI period plus its oversold/overbought thresholds, MACD's fast/slow/signal periods, Bollinger Band settings, stop-loss %, take-profit %, position-size %, and a signal-blending weight.
 
Three logical rules are enforced on these genes after each round of crossover/mutation (e.g. the short SMA window must always be smaller than the long SMA window) — genes that break these rules get nudged back into valid ranges rather than swapped wholesale, which would risk breaking a different rule instead.
 
### How a Strategy's Performance Is Scored
 
For each candidate strategy, the four underlying trading signals (trend via SMA, momentum via MACD, RSI, and Bollinger Bands) are calculated daily using that specific strategy's own evolved indicator settings, then combined into a single signal each day. The strategy's `signal_combination_weight` gene controls whether it leans toward trend-following or mean-reversion behavior.
 
The strategy is then backtested against the stock's real historical prices: it buys when the combined signal is strongly positive, sells when it turns strongly negative (or hits its own stop-loss/take-profit levels first), starting from a hypothetical $100,000. Its final score ("fitness") is simply its total return over that backtest period.
 
**Important caveat:** this fitness score reflects how well a strategy performed on the *exact historical data it was tested against* — a high score means "this fit the past well," not a guarantee it'll perform the same way going forward.
 
### How the GA Searches for Good Strategies
 
- **Selection:** tournament selection (the best of 3 randomly chosen candidates advances)
- **Crossover:** each gene is independently inherited from one of two parent strategies, at a rate controlled by the NN-predicted, per-stock crossover rate discussed above
- **Mutation:** a small chance (set in config) for any given gene to be randomly reset to a new value
- **Elitism:** the best-performing strategies from each generation are carried forward unchanged, so progress is never lost
- **Scale:** by default, a population of 100 candidate strategies evolved over 300 generations
### What Each Run Produces
 
| File | What's in it |
|---|---|
| `results.csv` | One row per run — the ticker, the crossover rate used, the best fitness score found, and all 14 of the winning strategy's gene values. New runs are added to this file, so it builds into a comparison table across multiple stocks over time. |
| `history_<TICKER>.csv` | The best fitness score seen at each generation, for that run — useful for plotting how the GA converged over time. This file is overwritten each time you rerun a given ticker. |
| `decisions.csv` | One row per run — the date, price, combined signal value, and the resulting BUY/SELL/HOLD decision with a short reason. New runs are added to this file. |

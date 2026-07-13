"""
    - Two signal families are blended by the chromosome's own
      signal_combination_weight (w):
          trend      = (sma_signal + macd_signal) / 2
          reversion  = (rsi_signal + bb_signal) / 2
          combined   = w * trend + (1 - w) * reversion
    - sma_signal / macd_signal are normalized by GENE_NORMALIZATION_K
      (0.05) and clipped to [-1, 1].
    - rsi_signal / bb_signal are linear maps of RSI / %B onto [-1, 1]
      using the chromosome's own oversold/overbought and band values.
    - Enter long when combined > SIGNAL_THRESHOLD, exit when it drops
      below -SIGNAL_THRESHOLD, overridden at any time by the
      chromosome's stop_loss_pct / take_profit_pct.
    - Long-only, one position at a time; position_size_pct of current
      cash is deployed on each entry.
    - Fitness = total/cumulative return over the backtest window.

Output: run_ga_for_stock() results are written to two CSVs (see
save_result_to_csv / save_history_to_csv):
    - results.csv  -- one row per run: ticker, crossover_rate_used,
      best_fitness, and every gene of the best chromosome. Appended to,
      so multiple tickers build up one comparison table.
    - history_<ticker>.csv -- best fitness per generation for that run,
      for convergence plotting.
"""

import csv
import os
import random
from datetime import date

import numpy as np
import pandas as pd

from config import CONFIG, GeneSpec
from data.fetcher import fetch_ohclv_with_buffer, get_experiment_date_range
from features.indicators import (
    compute_sma,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
)
from run_experiment import run_experiment

Chromosome = dict[str, float]

GENE_NORMALIZATION_K = 0.05
SIGNAL_THRESHOLD = 0.3
INITIAL_CAPITAL = 100_000.0

# Output locations for CSV results.
RESULTS_CSV_PATH = "results.csv"
HISTORY_CSV_DIR = "."

_GENE_SPECS: dict[str, GeneSpec] = {spec.name: spec for spec in CONFIG.chromosome.genes}
_GENE_NAMES: tuple[str, ...] = tuple(spec.name for spec in CONFIG.chromosome.genes)


def random_gene_value(spec: GeneSpec, rng: random.Random) -> float:
    """Draws a single gene value uniformly within its configured range."""
    if spec.dtype == "int":
        return float(rng.randint(int(spec.low), int(spec.high)))
    return rng.uniform(spec.low, spec.high)


def _clamp(value: float, spec: GeneSpec) -> float:
    return min(max(value, spec.low), spec.high)


def repair_constraints(chrom: Chromosome) -> Chromosome:
    for c in CONFIG.chromosome.constraints:
        left_spec, right_spec = _GENE_SPECS[c.left], _GENE_SPECS[c.right]
        left_val, right_val = chrom[c.left], chrom[c.right]

        satisfied = {
            "<": left_val < right_val,
            "<=": left_val <= right_val,
            ">": left_val > right_val,
            ">=": left_val >= right_val,
        }[c.op]
        if satisfied:
            continue

        step_left = 1.0 if left_spec.dtype == "int" else (left_spec.high - left_spec.low) * 0.01
        step_right = 1.0 if right_spec.dtype == "int" else (right_spec.high - right_spec.low) * 0.01

        candidate_right = _clamp(left_val + step_right, right_spec)
        if candidate_right > left_val:
            right_val = candidate_right
        else:
            left_val = _clamp(right_val - step_left, left_spec)

        chrom[c.left], chrom[c.right] = left_val, right_val

    return chrom


def random_chromosome(rng: random.Random) -> Chromosome:
    chrom = {spec.name: random_gene_value(spec, rng) for spec in CONFIG.chromosome.genes}
    return repair_constraints(chrom)

# Uniform crossover
def crossover(
    parent1: Chromosome, parent2: Chromosome, crossover_rate: float, rng: random.Random
) -> tuple[Chromosome, Chromosome]:
    if rng.random() >= crossover_rate:
        return dict(parent1), dict(parent2)

    child1: Chromosome = {}
    child2: Chromosome = {}
    for spec in CONFIG.chromosome.genes:
        if rng.random() < 0.5:
            child1[spec.name], child2[spec.name] = parent1[spec.name], parent2[spec.name]
        else:
            child1[spec.name], child2[spec.name] = parent2[spec.name], parent1[spec.name]
    return child1, child2


def mutate(chrom: Chromosome, mutation_rate: float, rng: random.Random) -> Chromosome:
    mutated = dict(chrom)
    for spec in CONFIG.chromosome.genes:
        if rng.random() < mutation_rate:
            mutated[spec.name] = random_gene_value(spec, rng)
    return mutated


def tournament_select(
    ranked: list[tuple[Chromosome, float]], k: int, rng: random.Random
) -> Chromosome:
    contestants = rng.sample(ranked, k)
    return max(contestants, key=lambda pair: pair[1])[0]


def _chromosome_indicators(chrom: Chromosome, buffered_df: pd.DataFrame) -> pd.DataFrame:
    """Recomputes SMA/RSI/MACD/BBands at the chromosome's OWN evolved
    periods (distinct from the fixed periods used for NN features)."""
    sma_short = compute_sma(buffered_df, int(chrom["sma_short_window"]))
    sma_long = compute_sma(buffered_df, int(chrom["sma_long_window"]))
    rsi = compute_rsi(buffered_df, int(chrom["rsi_period"]))
    macd_df = compute_macd(
        buffered_df,
        fast=int(chrom["macd_fast_period"]),
        slow=int(chrom["macd_slow_period"]),
        signal=int(chrom["macd_signal_period"]),
    )
    bb_df = compute_bollinger_bands(
        buffered_df, window=int(chrom["bbands_window"]), num_std=chrom["bbands_num_std"]
    )

    out = pd.DataFrame(index=buffered_df.index)
    out["close"] = buffered_df["close"]
    out["sma_short"] = sma_short
    out["sma_long"] = sma_long
    out["rsi"] = rsi
    out["macd_hist"] = macd_df["macd_hist"]
    out["bb_upper"] = bb_df["bb_upper"]
    out["bb_lower"] = bb_df["bb_lower"]
    return out


def _combined_signal(chrom: Chromosome, ind_df: pd.DataFrame) -> pd.Series:
    k = GENE_NORMALIZATION_K

    sma_signal = ((ind_df["sma_short"] - ind_df["sma_long"]) / ind_df["sma_long"] / k).clip(-1, 1)
    macd_signal = (ind_df["macd_hist"] / ind_df["close"] / k).clip(-1, 1)

    oversold, overbought = chrom["rsi_oversold"], chrom["rsi_overbought"]
    rsi_signal = (2 * (overbought - ind_df["rsi"]) / (overbought - oversold) - 1).clip(-1, 1)

    band_range = (ind_df["bb_upper"] - ind_df["bb_lower"]).replace(0, np.nan)
    percent_b = ((ind_df["close"] - ind_df["bb_lower"]) / band_range).fillna(0.5)
    bb_signal = (1 - 2 * percent_b).clip(-1, 1)

    w = chrom["signal_combination_weight"]
    trend = (sma_signal + macd_signal) / 2
    reversion = (rsi_signal + bb_signal) / 2
    return w * trend + (1 - w) * reversion


def _backtest_total_return(chrom: Chromosome, prices: np.ndarray, signals: np.ndarray) -> float:
    cash = INITIAL_CAPITAL
    shares = 0.0
    in_position = False
    entry_price = 0.0

    size_pct = chrom["position_size_pct"]
    sl_pct = chrom["stop_loss_pct"]
    tp_pct = chrom["take_profit_pct"]

    for price, sig in zip(prices, signals):
        if in_position:
            hit_stop = price <= entry_price * (1 - sl_pct)
            hit_target = price >= entry_price * (1 + tp_pct)
            signal_exit = sig < -SIGNAL_THRESHOLD
            if hit_stop or hit_target or signal_exit:
                cash += shares * price
                shares = 0.0
                in_position = False
        elif sig > SIGNAL_THRESHOLD:
            invest = cash * size_pct
            shares = invest / price
            cash -= invest
            entry_price = price
            in_position = True

    final_equity = cash + shares * prices[-1]
    return final_equity / INITIAL_CAPITAL - 1.0


def evaluate_chromosome(chrom: Chromosome, buffered_df: pd.DataFrame, window_start) -> float:
    """Fitness function: total return of the chromosome's backtest over
    the experiment window (buffer rows excluded)."""
    ind_df = _chromosome_indicators(chrom, buffered_df)
    window = ind_df.loc[str(window_start):].dropna()

    if window.empty:
        return -1.0

    signals = _combined_signal(chrom, window)
    return _backtest_total_return(chrom, window["close"].values, signals.values)


def run_ga(
    buffered_df: pd.DataFrame, window_start, crossover_rate: float, seed: int = 42
) -> tuple[Chromosome, float, list[float]]:
    rng = random.Random(seed)
    ga_cfg = CONFIG.ga

    population: list[tuple[Chromosome, float | None]] = [
        (random_chromosome(rng), None) for _ in range(ga_cfg.population_size)
    ]
    best_fitness_by_gen: list[float] = []

    for gen in range(ga_cfg.max_gen):
        population = [
            (c, evaluate_chromosome(c, buffered_df, window_start) if f is None else f)
            for c, f in population
        ]
        ranked = sorted(population, key=lambda pair: pair[1], reverse=True)
        best_fitness_by_gen.append(ranked[0][1])

        if gen % 25 == 0 or gen == ga_cfg.max_gen - 1:
            mean_fitness = sum(f for _, f in ranked) / len(ranked)
            print(f"[ga] gen {gen:>3}: best={ranked[0][1]:.4f}  mean={mean_fitness:.4f}")

        next_population = [(dict(c), f) for c, f in ranked[: ga_cfg.elitism]]
        while len(next_population) < ga_cfg.population_size:
            p1 = tournament_select(ranked, ga_cfg.tournament_k, rng)
            p2 = tournament_select(ranked, ga_cfg.tournament_k, rng)
            c1, c2 = crossover(p1, p2, crossover_rate, rng)
            c1 = repair_constraints(mutate(c1, ga_cfg.mutation_rate, rng))
            c2 = repair_constraints(mutate(c2, ga_cfg.mutation_rate, rng))
            next_population.append((c1, None))
            if len(next_population) < ga_cfg.population_size:
                next_population.append((c2, None))

        population = next_population

    final_ranked = sorted(
        (
            (c, evaluate_chromosome(c, buffered_df, window_start) if f is None else f)
            for c, f in population
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )
    best_chrom, best_fitness = final_ranked[0]
    return best_chrom, best_fitness, best_fitness_by_gen


def run_ga_for_stock(ticker: str, as_of: date = None, seed: int = 42) -> dict:
    exp_result = run_experiment(ticker, as_of)

    window_start, window_end = get_experiment_date_range(as_of)
    buffered_df = fetch_ohclv_with_buffer(ticker, window_start, window_end)

    print(f"[ga] Running GA for {ticker} (crossover_rate={exp_result['crossover_rate']:.4f})...")
    best_chrom, best_fitness, history = run_ga(
        buffered_df, window_start, exp_result["crossover_rate"], seed=seed
    )

    return {
        "ticker": ticker,
        "best_chromosome": best_chrom,
        "best_fitness": best_fitness,
        "fitness_history": history,
        "crossover_rate_used": exp_result["crossover_rate"],
    }


def save_result_to_csv(result: dict, path: str = RESULTS_CSV_PATH) -> None:
    fieldnames = ["ticker", "crossover_rate_used", "best_fitness", *_GENE_NAMES]
    file_exists = os.path.isfile(path)

    row = {
        "ticker": result["ticker"],
        "crossover_rate_used": result["crossover_rate_used"],
        "best_fitness": result["best_fitness"],
        **result["best_chromosome"],
    }

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"[ga] Appended result for {result['ticker']} to {path}")


def save_history_to_csv(result: dict, directory: str = HISTORY_CSV_DIR) -> str:
    path = os.path.join(directory, f"history_{result['ticker']}.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["generation", "best_fitness"])
        for gen, fitness in enumerate(result["fitness_history"]):
            writer.writerow([gen, fitness])

    print(f"[ga] Wrote fitness history for {result['ticker']} to {path}")
    return path


if __name__ == "__main__":
    import sys

    ticker_arg = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    result = run_ga_for_stock(ticker_arg)

    print(f"\n[ga] Done. Best total return for {result['ticker']}: "
          f"{result['best_fitness']:.2%}")

    save_result_to_csv(result)
    save_history_to_csv(result)
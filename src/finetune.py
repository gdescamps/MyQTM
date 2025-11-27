import json
import os

import cma
import numpy as np

from src.benchmark import run_benchmark
from src.utils.plot import plot_portfolio_metrics

best_perf = -np.inf


def cmaes_grid_search_benchmark(
    n_calls=50,
    early_stop_rounds=None,
    cma_dropout_round=1,
    cma_dropout=0,
    space=None,
    bench_start_date=None,
    bench_end_date=None,
    init_capital=10000,
    model_path=None,
    data_path=None,
    random_state=42,
    x0=None,
    cma_std=0.2,
    local_log=None,
):
    """
    Hyperparameter optimization using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).

    Args:
        n_calls: Maximum number of iterations
        early_stop_rounds: Number of rounds without improvement before stopping
        cma_dropout_round: Number of rounds to run for robustness testing
        cma_dropout: Number of stocks to randomly remove during dropout rounds
        space: Parameter space bounds (skopt-style)
        bench_start_date: Benchmark start date
        bench_end_date: Benchmark end date
        init_capital: Initial capital for backtesting
        model_path: Path to model files
        data_path: Path to data directory
        random_state: Random seed
        x0: Initial point for CMA-ES
        cma_std: Initial standard deviation for CMA-ES
        local_log: Logging configuration object

    Returns:
        Tuple of (best_params, best_score, best_plot, best_positions)
    """
    global best_perf
    best_perf = -np.inf
    global best_plot
    best_plot = None
    global best_positions
    best_positions = None
    global last_params
    last_params = None

    # Convert skopt space → CMA-ES bounds
    lower_bounds = [dim.low for dim in space]
    upper_bounds = [dim.high for dim in space]
    bounds = [lower_bounds, upper_bounds]

    # Starting point = center of bounds
    if x0 is None:
        x0 = [(lo + hi) / 2 for lo, hi in zip(lower_bounds, upper_bounds)]

    # Initialize CMA-ES optimization strategy
    es = cma.CMAEvolutionStrategy(
        x0,
        cma_std,
        {"bounds": bounds, "seed": random_state},
    )

    def objective(params):
        """
        Objective function to minimize (negative performance score).
        """
        (
            long_open_prob_thresa,
            long_close_prob_thresa,
            short_open_prob_thresa,
            short_close_prob_thresa,
            long_open_prob_thresb,
            long_close_prob_thresb,
            short_open_prob_thresb,
            short_close_prob_thresb,
            long_prob_powera,
            short_prob_powera,
            long_prob_powerb,
            short_prob_powerb,
        ) = params
        global last_params
        global best_perf
        global best_plot
        global best_positions

        last_params = params

        # Validate parameter constraints: opening threshold must be greater than closing threshold
        if long_open_prob_thresa <= long_close_prob_thresa:
            return 1e6
        if short_open_prob_thresa <= short_close_prob_thresa:
            return 1e6

        if long_open_prob_thresb <= long_close_prob_thresb:
            return 1e6
        if short_open_prob_thresb <= short_close_prob_thresb:
            return 1e6

        # Run multiple evaluations with optional dropout for robustness
        perfs = []
        metrics_list = []
        for i in range(cma_dropout_round):
            remove_stocks = 0 if i == 0 else cma_dropout
            # Run benchmark with current parameters
            metrics, positions, _ = run_benchmark(
                BENCH_START_DATE=bench_start_date,
                BENCH_END_DATE=bench_end_date,
                INIT_CAPITAL=init_capital,
                LONG_OPEN_PROB_THRESA=long_open_prob_thresa,
                LONG_CLOSE_PROB_THRESA=long_close_prob_thresa,
                SHORT_OPEN_PROB_THRESA=short_open_prob_thresa,
                SHORT_CLOSE_PROB_THRESA=short_close_prob_thresa,
                LONG_OPEN_PROB_THRESB=long_open_prob_thresb,
                LONG_CLOSE_PROB_THRESB=long_close_prob_thresb,
                SHORT_OPEN_PROB_THRESB=short_open_prob_thresb,
                SHORT_CLOSE_PROB_THRESB=short_close_prob_thresb,
                LONG_PROB_POWERA=long_prob_powera,
                SHORT_PROB_POWERA=short_prob_powera,
                LONG_PROB_POWERB=long_prob_powerb,
                SHORT_PROB_POWERB=short_prob_powerb,
                MODEL_PATH=model_path,
                data_path=data_path,
                remove_stocks=remove_stocks,
            )
            perf = metrics["portfolio"]["perf"]
            perfs.append(perf)
            metrics_list.append(metrics)

        # Combine performance scores: base + mean + minimum
        perf = perfs[0] + np.mean(perfs) + np.min(perfs)

        # Track and save best performance found so far
        if perf > best_perf:
            best_perf = perf
            best_positions = positions
            if local_log is not None:
                plot, _ = plot_portfolio_metrics(metrics_list)
                png_path = os.path.join(local_log.output_dir_time, "current.png")
                plot.save(png_path)

                # Save current best parameters
                with open(
                    os.path.join(local_log.output_dir_time, "current_params.json"), "w"
                ) as f:
                    json.dump(params.tolist(), f, indent=4)
                # Save current best positions
                with open(
                    os.path.join(local_log.output_dir_time, "current_positions.json"),
                    "w",
                ) as f:
                    json.dump(positions, f, indent=4)

        return -perf  # CMA-ES minimizes

    no_improve = 0
    best_score = float("inf")

    # Main optimization loop
    for i in range(n_calls):
        solutions = es.ask()
        scores = [objective(x) for x in solutions]
        es.tell(solutions, scores)
        es.disp()
        min_score = min(scores)
        # Check for improvement and apply early stopping if needed
        if min_score < best_score:
            best_score = min_score
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= early_stop_rounds:
            print(f"Early stopping at iteration {i}")
            break

    # Extract best parameters found
    xbest = es.result[0]
    xbest = [float(x) for x in xbest]

    return xbest, es.result[1], best_plot, best_positions

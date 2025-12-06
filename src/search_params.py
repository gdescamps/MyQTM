"""
search_params.py

This module is responsible for performing parameter search and optimization using the CMA-ES algorithm. It includes functions for running optimization processes, benchmarking, and saving results.

Functions:
- run_single_random_state(): Executes a single optimization process for a given random state.
- sort_perfs(): Sorts and saves the top-performing parameter sets based on performance metrics.

Main Execution:
- Loads environment variables and initializes configurations.
- Runs the optimization process in parallel using multiprocessing.
- Archives and saves the results of the optimization.
"""

import json
import multiprocessing
import os
import random
import shutil
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

import config
from src.benchmark import compute_nasdaq_data, run_benchmark
from src.cuda import auto_select_gpu
from src.finetune import cmaes_grid_search_benchmark
from src.path import get_project_root
from src.plot import plot_portfolio_metrics
from src.printlog import PrintLog, PrintLogProcess


def run_single_random_state(
    random_state,
    output_dir_time,
    init_space,
    init_x0,
    init_cma_std,
    cma_loops,
    early_stop_rounds,
    cma_dropout_round,
    cma_dropout,
):
    """
    Executes a single CMA-ES optimization process for a given random state.

    Args:
        random_state (int): Random seed for reproducibility.
        output_dir_time (str): Directory to save output files.
        init_space (list): Initial parameter space for optimization.
        init_x0 (list): Initial parameter values.
        init_cma_std (float): Initial standard deviation for CMA-ES.
        cma_loops (int): Number of optimization iterations.
        early_stop_rounds (int): Early stopping rounds for optimization.
        cma_dropout_round (int): Dropout rounds for CMA-ES.
        cma_dropout (float): Dropout rate for CMA-ES.

    Returns:
        None
    """
    # Initialize process-specific logger
    process_log = PrintLogProcess(
        output_dir_time=output_dir_time, process_id=random_state
    )

    # Run CMA-ES grid search optimization
    xbest, fbest, _, positions = cmaes_grid_search_benchmark(
        n_calls=cma_loops,
        early_stop_rounds=early_stop_rounds,
        cma_dropout_round=cma_dropout_round,
        cma_dropout=cma_dropout,
        space=init_space,
        file_bench_end_date=config.TEST_END_DATE,
        bench_start_date=config.TEST_START_DATE,
        bench_end_date=config.CMAES_END_DATE,
        init_capital=config.INITIAL_CAPITAL,
        model_path=config.TRAIN_DIR,
        data_path=None,
        random_state=random_state,
        x0=init_x0,
        cma_std=init_cma_std,
        local_log=None,
    )
    # Extract optimized trading parameters
    (
        long_open_prob_thresa,
        long_close_prob_thresa,
        short_open_prob_thresa,
        short_close_prob_thresa,
        long_open_prob_thresb,
        long_close_prob_thresb,
        short_open_prob_thresb,
        short_close_prob_thresb,
        increase_positions_count,
    ) = list(xbest)

    # Initialize performance tracking lists
    performances = []
    metrics_list = []
    random.seed(42)

    # Run benchmark with varying stock dropout rates
    for attempt in range(config.CMA_STOCKS_DROP_OUT_ROUND):
        remove_stocks = 0 if attempt == 0 else config.CMA_STOCKS_DROP_OUT
        metrics, _, _ = run_benchmark(
            FILE_BENCH_END_DATE=config.TEST_END_DATE,
            BENCH_START_DATE=config.TEST_START_DATE,
            BENCH_END_DATE=config.CMAES_END_DATE,
            INIT_CAPITAL=config.INITIAL_CAPITAL,
            LONG_OPEN_PROB_THRES_A=long_open_prob_thresa,
            LONG_CLOSE_PROB_THRES_A=long_close_prob_thresa,
            SHORT_OPEN_PROB_THRES_A=short_open_prob_thresa,
            SHORT_CLOSE_PROB_THRES_A=short_close_prob_thresa,
            LONG_OPEN_PROB_THRES_B=long_open_prob_thresb,
            LONG_CLOSE_PROB_THRES_B=long_close_prob_thresb,
            SHORT_OPEN_PROB_THRES_B=short_open_prob_thresb,
            SHORT_CLOSE_PROB_THRES_B=short_close_prob_thresb,
            INCREASE_POSITIONS_COUNT=increase_positions_count,
            MODEL_PATH=config.TRAIN_DIR,
            data_path=None,
            remove_stocks=remove_stocks,
        )
        perf = metrics["portfolio"]["perf"]
        performances.append(perf)

    for attempt in range(config.CMA_STOCKS_DROP_OUT_ROUND):
        remove_stocks = 0 if attempt == 0 else config.CMA_STOCKS_DROP_OUT
        metrics, _, _ = run_benchmark(
            FILE_BENCH_END_DATE=config.TEST_END_DATE,
            BENCH_START_DATE=config.TEST_START_DATE,
            BENCH_END_DATE=config.TEST_END_DATE,
            INIT_CAPITAL=config.INITIAL_CAPITAL,
            LONG_OPEN_PROB_THRES_A=long_open_prob_thresa,
            LONG_CLOSE_PROB_THRES_A=long_close_prob_thresa,
            SHORT_OPEN_PROB_THRES_A=short_open_prob_thresa,
            SHORT_CLOSE_PROB_THRES_A=short_close_prob_thresa,
            LONG_OPEN_PROB_THRES_B=long_open_prob_thresb,
            LONG_CLOSE_PROB_THRES_B=long_close_prob_thresb,
            SHORT_OPEN_PROB_THRES_B=short_open_prob_thresb,
            SHORT_CLOSE_PROB_THRES_B=short_close_prob_thresb,
            INCREASE_POSITIONS_COUNT=increase_positions_count,
            MODEL_PATH=config.TRAIN_DIR,
            data_path=None,
            remove_stocks=remove_stocks,
            force_reload=True,
        )
        metrics_list.append(metrics)

    if not metrics_list:
        return

    # Save combined metrics plot
    png_path = os.path.join(local_log.output_dir_time, f"best_{random_state}.png")

    nasdaq_metrics = compute_nasdaq_data(
        BENCH_START_DATE=config.TEST_START_DATE,
        BENCH_END_DATE=config.TEST_END_DATE,
        MODEL_PATH=config.TRAIN_DIR,
        data_path=None,
    )

    plot_all, _ = plot_portfolio_metrics(metrics_list, nasdaq_metrics)
    plot_all.save(png_path)

    # Calculate average performance across all runs
    global_perf = np.mean(performances)
    process_log.print("--------------------------------------------------------------")
    process_log.print(f"ret.fun: {fbest}")
    process_log.print(f"params: {str(list(xbest))}")
    process_log.print(f"global perf = {global_perf}")

    # Save optimized parameters to JSON file
    best_param = list(xbest)
    params_path = os.path.join(local_log.output_dir_time, f"params_{random_state}.json")
    with open(params_path, "w") as f:
        json.dump(best_param, f, indent=2)

    # Save global performance metric to JSON file
    perf_path = os.path.join(local_log.output_dir_time, f"perf_{random_state}.json")
    with open(perf_path, "w") as f:
        json.dump(global_perf, f, indent=2)


def sort_perfs(random_states, SEARCH_DIR):
    """
    Sorts and saves the top-performing parameter sets based on performance metrics.

    Args:
        random_states (list): List of random states used in the optimization process.
        SEARCH_DIR (str): Directory containing the performance results.

    Returns:
        None
    """
    # Collect and sort results by performance
    perfs = []
    for random_state in random_states:
        perf_path = os.path.join(SEARCH_DIR, f"perf_{random_state}.json")
        if os.path.exists(perf_path):
            with open(perf_path, "r") as f:
                try:
                    perf = json.load(f)
                    perfs.append({"perf": perf, "random_state": random_state})
                except json.JSONDecodeError:
                    # Fichier vide ou corrompu, on ignore
                    continue
    # Sort performance list in descending order by performance score
    perfs.sort(key=lambda x: x["perf"], reverse=True)

    # Copy top results with ranking labels
    for i, entry in enumerate(perfs):
        random_state = entry["random_state"]
        shutil.copy(
            os.path.join(SEARCH_DIR, f"params_{random_state}.json"),
            os.path.join(SEARCH_DIR, f"top{i+1}_params.json"),
        )
        shutil.copy(
            os.path.join(SEARCH_DIR, f"best_{random_state}.png"),
            os.path.join(SEARCH_DIR, f"top{i+1}_best.png"),
        )
        if 4 >= i >= 0:  # Copy best top 5 overall in same place
            shutil.copy(
                os.path.join(SEARCH_DIR, f"best_{random_state}.png"),
                os.path.join("./outputs", f"top{i+1}_best.png"),
            )
            shutil.copy(
                os.path.join(SEARCH_DIR, f"params_{random_state}.json"),
                os.path.join("./outputs", f"top{i+1}_params.json"),
            )


if __name__ == "__main__":
    """
    Main execution block for the parameter search pipeline. It performs the following steps:
    1. Loads environment variables and configurations.
    2. Initializes the optimization process with multiprocessing.
    3. Runs the optimization iteratively, refining the parameter search.
    4. Archives and saves the results of the optimization.
    """

    # Load environment variables from .env file
    # This ensures sensitive information like API keys is securely loaded into the environment.
    load_dotenv()

    # Set the path to the data directory
    # Create a directory to store downloaded data if it doesn't already exist.
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    # Automatically select GPU with sufficient memory
    auto_select_gpu(threshold_mb=500)

    # Set random seed for reproducibility
    seed = 42

    # Generate list of random states for parallel processes
    random_states = list(range(seed, seed + config.CMA_PROCESSES))

    # Load optimization configuration
    init_space = config.INIT_SPACE

    for iter in range(config.CMA_RECURSIVE):

        # Initialize process logger
        local_log = PrintLog(extra_name="_cma", enable=False)

        # Define output directory for search results
        SEARCH_DIR = local_log.output_dir_time

        processes = []

        if iter == 0:

            # Load optimization configuration
            init_x0 = config.INIT_X0
            init_cma_std = config.INIT_CMA_STD

            # Start parallel processes for current batch
            for random_state in random_states:
                p = multiprocessing.Process(
                    target=run_single_random_state,
                    args=(
                        random_state,
                        local_log.output_dir_time,
                        init_space,
                        init_x0,
                        init_cma_std,
                        config.CMA_LOOPS,
                        config.CMA_EARLY_STOP_ROUNDS,
                        1,
                        0,
                    ),
                )
                processes.append(p)

                if len(processes) >= config.CMA_PARALLEL_PROCESSES:
                    # Start all processes in the current batch
                    for p in processes:
                        p.start()

                    # Wait for all processes in batch to complete
                    for p in processes:
                        p.join()
                    processes = []

                    sort_perfs(random_states, SEARCH_DIR)

        else:

            for top in range(
                1,
                max(
                    config.CMA_PARALLEL_PROCESSES + 1,
                    int(config.CMA_PROCESSES / (2 * iter)) + 1,
                ),
            ):

                top_params_path = os.path.join(config.CMA_DIR, f"top{top}_params.json")
                with open(top_params_path, "r") as f:
                    best_param = json.load(f)
                init_x0 = best_param
                init_cma_std = config.INIT_CMA_STD / (iter + 1)  # Reduce std for finer

                p = multiprocessing.Process(
                    target=run_single_random_state,
                    args=(
                        random_states[top - 1],
                        local_log.output_dir_time,
                        init_space,
                        init_x0,
                        init_cma_std,
                        int(config.CMA_LOOPS),
                        int(config.CMA_EARLY_STOP_ROUNDS),
                        1,
                        0,
                    ),
                )
                processes.append(p)

                if len(processes) >= config.CMA_PARALLEL_PROCESSES:
                    # Start all processes in the current batch
                    for p in processes:
                        p.start()

                    # Wait for all processes in batch to complete
                    for p in processes:
                        p.join()
                    processes = []

                    sort_perfs(random_states, SEARCH_DIR)

        for p in processes:
            p.start()

        # Wait for all processes in batch to complete
        for p in processes:
            p.join()

        sort_perfs(random_states, SEARCH_DIR)

        # Archive final results
        local_log.copy_last()

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
import subprocess
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from skopt.space import Real

import config
from src.benchmark import compute_nasdaq_data, run_benchmark
from src.cuda import auto_select_gpu
from src.finetune import cmaes_grid_search_benchmark
from src.path import get_project_root
from src.plot import plot_portfolio_metrics
from src.printlog import PrintLog, PrintLogProcess


def clamp_x0_to_space(x0, space):
    if not x0:
        return x0
    clipped = []
    for value, dim in zip(x0, space):
        if isinstance(dim, Real):
            low = dim.low
            high = dim.high
            clipped.append(min(max(float(value), low), high))
    return clipped


def load_thresholds(train_dir):
    thresholds_paths = [
        Path(get_project_root()) / train_dir / "best_thresholds.json",
        Path(get_project_root()) / train_dir / "best_threshold.json",
    ]
    for thresholds_path in thresholds_paths:
        if thresholds_path.exists():
            with open(thresholds_path, "r") as f:
                payload = json.load(f)
            labels = payload.get("labels")
            thresholds = payload.get("thresholds")
            if labels and thresholds:
                return dict(zip(labels, thresholds))
            if thresholds:
                default_labels = ("A_long", "B_long", "A_short", "B_short")
                return dict(zip(default_labels, thresholds))
    return None


def override_open_threshold_bounds(space, thresholds, margin=0.05):
    if not thresholds:
        return space
    label_to_space_name = {
        "A_long": "long_open_prob_thres_A",
        "B_long": "long_open_prob_thres_B",
        "A_short": "short_open_prob_thres_A",
        "B_short": "short_open_prob_thres_B",
    }
    overrides = {
        label_to_space_name[label]: float(value)
        for label, value in thresholds.items()
        if label in label_to_space_name
    }
    if not overrides:
        return space
    updated_space = []
    for dim in space:
        if isinstance(dim, Real) and dim.name in overrides:
            center = overrides[dim.name]
            updated_space.append(
                Real(center - margin, center + margin, name=dim.name)
            )
        else:
            updated_space.append(dim)
    return updated_space


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
        file_bench_end_date=config.BENCHMARK_END_DATE,
        bench_start_date=config.BENCHMARK_START_DATE,
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
        long_pos_count,
        short_pos_count,
        long_pos_pow,
        short_pos_pow,
    ) = list(xbest)

    # Initialize performance tracking lists
    performances = []
    metrics_list = []
    random.seed(42)

    # Run benchmark with varying stock dropout rates
    for attempt in range(config.CMA_STOCKS_DROP_OUT_ROUND):
        remove_stocks = 0 if attempt == 0 else config.CMA_STOCKS_DROP_OUT
        metrics, _, _ = run_benchmark(
            FILE_BENCH_END_DATE=config.BENCHMARK_END_DATE,
            BENCH_START_DATE=config.BENCHMARK_START_DATE,
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
            LONG_POS_COUNT=long_pos_count,
            SHORT_POS_COUNT=short_pos_count,
            LONG_POS_POW=long_pos_pow,
            SHORT_POS_POW=short_pos_pow,
            MODEL_PATH=config.TRAIN_DIR,
            data_path=None,
            remove_stocks=remove_stocks,
            attempt=attempt,
        )
        perf = metrics["portfolio"]["perf"]
        performances.append(perf)

    positions = None
    for attempt in range(config.CMA_STOCKS_DROP_OUT_ROUND):
        remove_stocks = 0 if attempt == 0 else config.CMA_STOCKS_DROP_OUT
        metrics, pos, _ = run_benchmark(
            FILE_BENCH_END_DATE=config.BENCHMARK_END_DATE,
            BENCH_START_DATE=config.BENCHMARK_START_DATE,
            BENCH_END_DATE=config.BENCHMARK_END_DATE,
            INIT_CAPITAL=config.INITIAL_CAPITAL,
            LONG_OPEN_PROB_THRES_A=long_open_prob_thresa,
            LONG_CLOSE_PROB_THRES_A=long_close_prob_thresa,
            SHORT_OPEN_PROB_THRES_A=short_open_prob_thresa,
            SHORT_CLOSE_PROB_THRES_A=short_close_prob_thresa,
            LONG_OPEN_PROB_THRES_B=long_open_prob_thresb,
            LONG_CLOSE_PROB_THRES_B=long_close_prob_thresb,
            SHORT_OPEN_PROB_THRES_B=short_open_prob_thresb,
            SHORT_CLOSE_PROB_THRES_B=short_close_prob_thresb,
            LONG_POS_COUNT=long_pos_count,
            SHORT_POS_COUNT=short_pos_count,
            LONG_POS_POW=long_pos_pow,
            SHORT_POS_POW=short_pos_pow,
            MODEL_PATH=config.TRAIN_DIR,
            data_path=None,
            remove_stocks=remove_stocks,
            attempt=attempt,
            force_reload=True,
        )
        metrics_list.append(metrics)
        if remove_stocks == 0:
            positions = pos

    if not metrics_list:
        return

    # Save combined metrics plot
    png_path = os.path.join(local_log.output_dir_time, f"best_{random_state}.png")

    nasdaq_metrics = compute_nasdaq_data(
        BENCH_START_DATE=config.BENCHMARK_START_DATE,
        BENCH_END_DATE=config.BENCHMARK_END_DATE,
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

    position_path = os.path.join(
        local_log.output_dir_time, f"positions_{random_state}.json"
    )
    with open(position_path, "w") as f:
        json.dump(positions, f, indent=2)


def sort_perfs(random_states, SEARCH_DIR):
    """
    Sorts and saves the top-performing parameter sets based on performance metrics.

    Args:
        random_states (list): List of random states used in the optimization process.
        SEARCH_DIR (str): Directory containing the performance results.

    Returns:
        None
    """

    def safe_json_load(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def copy_if_exists(src, dst):
        if src.exists():
            shutil.copy2(src, dst)

    def run_git(args, cwd):
        try:
            return subprocess.check_output(
                ["git"] + args, cwd=cwd, stderr=subprocess.STDOUT, text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            return f"unavailable: {exc}"

    def write_code_log(path):
        repo_root = str(get_project_root())
        branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_root)
        last_commit_sha = run_git(["rev-parse", "HEAD"], repo_root)
        last_commit = run_git(
            ["log", "-1", "--pretty=format:%H %an %ad %s"], repo_root
        )
        last_commits_messages = run_git(
            ["log", "-5", "--pretty=format:%h %s"], repo_root
        )
        status = run_git(["status", "-sb"], repo_root)
        diff_stat = run_git(["diff", "--stat"], repo_root)
        diff_numstat = run_git(["diff", "--numstat"], repo_root)
        with open(path, "w") as f:
            f.write(f"branch: {branch}\n")
            f.write(f"last_commit_sha1: {last_commit_sha}\n")
            f.write(f"last_commit: {last_commit}\n")
            f.write("last_5_commit_messages:\n")
            f.write(f"{last_commits_messages}\n")
            f.write("status:\n")
            f.write(f"{status}\n")
            f.write("diff_stat:\n")
            f.write(f"{diff_stat}\n")
            f.write("diff_numstat:\n")
            f.write(f"{diff_numstat}\n")

    all_dir = Path(config.ALL_DIR)
    best_dir = Path(config.BEST_DIR)
    all_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    run_id = Path(SEARCH_DIR).name
    for random_state in random_states:
        perf_path = Path(SEARCH_DIR) / f"perf_{random_state}.json"
        perf = safe_json_load(perf_path)
        if perf is None:
            continue

        uid = f"{run_id}_{random_state}"
        copy_if_exists(perf_path, all_dir / f"perf_{uid}.json")
        copy_if_exists(
            Path(SEARCH_DIR) / f"params_{random_state}.json",
            all_dir / f"params_{uid}.json",
        )
        copy_if_exists(
            Path(SEARCH_DIR) / f"positions_{random_state}.json",
            all_dir / f"positions_{uid}.json",
        )
        copy_if_exists(
            Path(SEARCH_DIR) / f"best_{random_state}.png",
            all_dir / f"best_{uid}.png",
        )

        train_log = Path(config.TRAIN_DIR) / "print.log"
        copy_if_exists(train_log, all_dir / f"model_{uid}.log")

        code_log_path = all_dir / f"code_{uid}.log"
        write_code_log(code_log_path)

    for existing in best_dir.glob("top_*"):
        if existing.is_file():
            existing.unlink()

    all_perfs = []
    for perf_path in all_dir.glob("perf_*.json"):
        perf = safe_json_load(perf_path)
        if perf is None:
            continue
        uid = perf_path.stem.replace("perf_", "", 1)
        all_perfs.append({"perf": perf, "uid": uid})

    all_perfs.sort(key=lambda x: x["perf"], reverse=True)
    for i, entry in enumerate(all_perfs):
        rank = i + 1
        uid = entry["uid"]
        copy_if_exists(
            all_dir / f"perf_{uid}.json",
            best_dir / f"top_{rank}_perf.json",
        )
        copy_if_exists(
            all_dir / f"params_{uid}.json",
            best_dir / f"top_{rank}_params.json",
        )
        copy_if_exists(
            all_dir / f"positions_{uid}.json",
            best_dir / f"top_{rank}_positions.json",
        )
        copy_if_exists(
            all_dir / f"model_{uid}.log",
            best_dir / f"top_{rank}_model.log",
        )
        copy_if_exists(
            all_dir / f"code_{uid}.log",
            best_dir / f"top_{rank}_code.log",
        )
        copy_if_exists(
            all_dir / f"best_{uid}.png",
            best_dir / f"top_{rank}.png",
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
    thresholds = load_thresholds(config.TRAIN_DIR)
    init_space = override_open_threshold_bounds(config.INIT_SPACE, thresholds)

    for iter in range(config.CMA_RECURSIVE):

        # Initialize process logger
        local_log = PrintLog(extra_name="_cma", enable=False)

        # Define output directory for search results
        SEARCH_DIR = local_log.output_dir_time

        processes = []

        if iter == 0:

            # Load optimization configuration
            init_x0 = clamp_x0_to_space(config.INIT_X0, init_space)
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
                init_x0 = clamp_x0_to_space(best_param, init_space)
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

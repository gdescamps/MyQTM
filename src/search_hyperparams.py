import json
import os
import random

import numpy as np

import config
from src.benchmark import plot_portfolio_metrics, run_benchmark
from src.finetune import cmaes_grid_search_benchmark
from src.utils.printlog import PrintLog

if __name__ == "__main__":
    M = 1
    seed = 42
    local_log = PrintLog(extra_name="_cmaes")
    best_perf = -np.inf
    for random_state in range(seed, seed + M):
        xbest, fbest, positions = cmaes_grid_search_benchmark(
            n_calls=config.CMA_LOOPS,
            space=config.INIT_SPACE,
            bench_start_date=config.TEST_START_DATE,
            bench_end_date=config.TEST_END_DATE,
            init_capital=config.INITIAL_CAPITAL,
            model_path=config.TRAIN_DIR,
            random_state=random_state,
            x0=config.INIT_X0,
            cma_std=config.INIT_CMA_STD,
            early_stop_rounds=config.CMA_EARLY_STOP_ROUNDS,
            local_log=local_log,
        )
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
        ) = list(xbest)

        performances = []
        metrics_list = []
        plot_first = None
        random.seed(42)
        for remove_stocks in [0, config.CMA_STOCKS_DROP_OUT]:
            for attempt in range(config.CMA_STOCKS_DROP_OUT_ROUND):
                metrics, plot, _, remove_stocks_list = run_benchmark(
                    BENCH_START_DATE=config.TEST_START_DATE,
                    BENCH_END_DATE=config.TEST_END_DATE,
                    INIT_CAPITAL=config.INITIAL_CAPITAL,
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
                    MODEL_PATH=config.TRAIN_DIR,
                    remove_stocks=remove_stocks,
                )
                perf = metrics["portfolio"]["perf"]

                performances.append(perf)
                metrics_list.append(metrics)
                if remove_stocks == 0:
                    png_path = os.path.join(
                        local_log.output_dir_time, f"backtest_{random_state}.png"
                    )
                    plot_first = plot
                    plot_first.save(png_path)
                    break

        png_path = os.path.join(
            local_log.output_dir_time, f"backtest_{random_state}_all.png"
        )
        plot_all, _ = plot_portfolio_metrics(metrics_list)
        plot_all.save(png_path)
        global_perf = np.mean(performances)

        positions_path = os.path.join(
            local_log.output_dir_time, f"positions_{random_state}.json"
        )
        with open(positions_path, "w") as f:
            json.dump(positions, f, indent=2)

        hyperparams = list(xbest)
        hyperparams_path = os.path.join(
            local_log.output_dir_time, f"hyperparams_{random_state}.json"
        )
        with open(hyperparams_path, "w") as f:
            json.dump(hyperparams, f, indent=2)

        perf_path = os.path.join(local_log.output_dir_time, f"perf_{random_state}.json")
        with open(perf_path, "w") as f:
            json.dump(global_perf, f, indent=2)

        if global_perf > best_perf:
            best_perf = global_perf
            best_hyperparams_path = os.path.join(
                local_log.output_dir_time, "best_hyperparams.json"
            )
            with open(best_hyperparams_path, "w") as f:
                json.dump(hyperparams, f, indent=2)

    local_log.copy_last()

import json
import os
import random

from config import (
    CMA_STOCKS_DROP_OUT,
    CMA_STOCKS_DROP_OUT_ROUND,
    HYPERPARAMS_DIR,
    INITIAL_CAPITAL,
    TEST_END_DATE,
    TEST_START_DATE,
    TRAIN_DIR,
)
from src.benchmark import run_benchmark
from src.utils.plot import plot_portfolio_metrics
from src.utils.printlog import PrintLog

if __name__ == "__main__":

    local_log = PrintLog(extra_name="_backtest")

    assert (
        TRAIN_DIR is not None
    ), "TRAIN_DIR must be set in config.py, use train.py to train a model first"
    assert (
        HYPERPARAMS_DIR is not None
    ), "PARAMS_DIR must be set in config.py, use search_params.py to search model first"

    with open(os.path.join(HYPERPARAMS_DIR, "best_hyperparams.json"), "r") as f:
        xbest = json.load(f)

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

    returns = []
    max_drawdowns = []
    ulcer_indexes = []
    performances = []
    metrics_list = []

    random.seed(42)
    for remove_stocks in [0, CMA_STOCKS_DROP_OUT]:
        for attempt in range(CMA_STOCKS_DROP_OUT_ROUND):
            metrics, plot, positions, remove_stocks_list = run_benchmark(
                BENCH_START_DATE=TEST_START_DATE,
                BENCH_END_DATE=TEST_END_DATE,
                INIT_CAPITAL=INITIAL_CAPITAL,
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
                MODEL_PATH=TRAIN_DIR,
                remove_stocks=remove_stocks,
            )
            metrics_list.append(metrics)

            if remove_stocks == 0:
                png_path = os.path.join(local_log.output_dir_time, "backtest.png")
                plot.save(png_path)

                positions_path = os.path.join(
                    local_log.output_dir_time, "backtest_positions.json"
                )
                with open(positions_path, "w") as f:
                    json.dump(positions, f, indent=2)
                break

    plot, metrics_text = plot_portfolio_metrics(metrics_list)
    png_path = os.path.join(
        local_log.output_dir_time,
        "backtest_all.png",
    )
    plot.save(png_path)

    with local_log:
        print(metrics_text)

    local_log.copy_last()

import json
import os
import random
import shutil
from pathlib import Path

from dotenv import load_dotenv

from benchmark import compute_nasdaq_data, run_benchmark
from config import (
    BENCHMARK_END_DATE,
    BENCHMARK_START_DATE,
    CMA_DIR,
    CMA_STOCKS_DROP_OUT,
    CMA_STOCKS_DROP_OUT_ROUND,
    INITIAL_CAPITAL,
    TRAIN_DIR,
)
from src.path import get_project_root
from src.plot import plot_portfolio_metrics
from src.printlog import PrintLog

if __name__ == "__main__":

    # Load environment variables from .env file
    # This ensures sensitive information like API keys is securely loaded into the environment.
    load_dotenv()

    # Set the path to the data directory
    # Create a directory to store downloaded data if it doesn't already exist.
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    local_log = PrintLog(extra_name="_analyse", enable=False)

    benchmark_end_date = BENCHMARK_END_DATE
    # current_date = datetime.now(ZoneInfo("Europe/Paris")).strftime("%Y-%m-%d")
    # if os.path.exists(os.path.join(data_path, f"{current_date}_benchmark_XY.csv")):
    #     benchmark_end_date = current_date

    with local_log:
        print(f"Analyse until {benchmark_end_date} :")

    top = 1

    if os.path.exists(os.path.join(CMA_DIR, f"top{top}_params.json")):

        with open(os.path.join(CMA_DIR, f"top{top}_params.json"), "r") as f:
            XBEST = json.load(f)

        (
            # max_positions,
            long_open_prob_thres_a,
            long_close_prob_thres_a,
            short_open_prob_thres_a,
            short_close_prob_thres_a,
            long_open_prob_thres_b,
            long_close_prob_thres_b,
            short_open_prob_thres_b,
            short_close_prob_thres_b,
            increase_position_count,
        ) = list(XBEST)

        returns = []
        max_drawdowns = []
        ulcer_indexes = []
        performances = []
        metrics_list = []

        random.seed(42)
        positions = None
        for index in range(1):
            remove_stocks = 0 if index == 0 else CMA_STOCKS_DROP_OUT
            metrics, pos, remove_stocks_list = run_benchmark(
                FILE_BENCH_END_DATE=benchmark_end_date,
                BENCH_START_DATE=BENCHMARK_START_DATE,
                BENCH_END_DATE=benchmark_end_date,
                INIT_CAPITAL=INITIAL_CAPITAL,
                LONG_OPEN_PROB_THRES_A=long_open_prob_thres_a,
                LONG_CLOSE_PROB_THRES_A=long_close_prob_thres_a,
                SHORT_OPEN_PROB_THRES_A=short_open_prob_thres_a,
                SHORT_CLOSE_PROB_THRES_A=short_close_prob_thres_a,
                LONG_OPEN_PROB_THRES_B=long_open_prob_thres_b,
                LONG_CLOSE_PROB_THRES_B=long_close_prob_thres_b,
                SHORT_OPEN_PROB_THRES_B=short_open_prob_thres_b,
                SHORT_CLOSE_PROB_THRES_B=short_close_prob_thres_b,
                INCREASE_POSITIONS_COUNT=increase_position_count,
                MODEL_PATH=TRAIN_DIR,
                data_path=None,
                remove_stocks=remove_stocks,
            )
            metrics_list.append(metrics)
            if remove_stocks == 0:
                positions = pos

        nasdaq_metrics = compute_nasdaq_data(
            BENCH_START_DATE=BENCHMARK_START_DATE,
            BENCH_END_DATE=benchmark_end_date,
            MODEL_PATH=TRAIN_DIR,
            data_path=None,
        )

        json_path = os.path.join(
            local_log.output_dir_time,
            f"top{top}_positions.json",
        )
        with open(json_path, "w") as f:
            json.dump(positions, f, indent=2)

        plot, metrics_text = plot_portfolio_metrics(metrics_list, nasdaq_metrics)
        png_path = os.path.join(
            local_log.output_dir_time,
            f"top{top}_bench.png",
        )
        plot.save(png_path)

        if 5 >= top >= 1:  # Copy best top 5 overall in same place
            shutil.copy(
                png_path,
                os.path.join("./outputs", f"top{top}_best.png"),
            )
            shutil.copy(
                json_path,
                os.path.join("./outputs", f"top{top}_positions.json"),
            )
            shutil.copy(
                os.path.join(CMA_DIR, f"top{top}_params.json"),
                os.path.join("./outputs", f"top{top}_params.json"),
            )

        with local_log:
            print(f"Benchmark results for top{top}_best:")
            print(metrics_text)
    local_log.copy_last()

import cma
import numpy as np

import src.config as config
from src.benchmark import run_benchmark


def cmaes_grid_search_benchmark(
    n_calls=50,
    space=None,
    bench_start_date=None,
    bench_end_date=None,
    init_capital=10000,
    model_path=None,
    random_state=42,
    x0=None,
    cma_std=0.2,
    early_stop_rounds=config.CMA_EARLY_STOP_ROUNDS,
    local_log=None,
):
    global best_perf
    best_perf = -np.inf
    global best_positions
    best_positions = None

    # Convert skopt space → CMA-ES bounds
    lower_bounds = [dim.low for dim in space]
    upper_bounds = [dim.high for dim in space]
    bounds = [lower_bounds, upper_bounds]

    # Point de départ = centre des bornes
    if x0 is None:
        x0 = [(lo + hi) / 2 for lo, hi in zip(lower_bounds, upper_bounds)]

    es = cma.CMAEvolutionStrategy(
        x0,
        cma_std,
        {"bounds": bounds, "seed": random_state},
    )

    def objective(params):
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
        global best_perf
        global best_positions

        if long_open_prob_thresa <= long_close_prob_thresa:
            return 1e6
        if short_open_prob_thresa <= short_close_prob_thresa:
            return 1e6

        if long_open_prob_thresb <= long_close_prob_thresb:
            return 1e6
        if short_open_prob_thresb <= short_close_prob_thresb:
            return 1e6

        # random_value = np.random.randint(0, 5)
        perfs = []
        metrics_list = []
        for i in range(config.CMA_STOCKS_DROP_OUT_ROUND):
            remove_stocks = config.CMA_STOCKS_DROP_OUT
            if i == 0:
                remove_stocks = 0
            metrics, _, positions, _ = run_benchmark(
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
                remove_stocks=remove_stocks,
            )
            perf = metrics["portfolio"]["perf"]
            perfs.append(perf)
            metrics_list.append(metrics)

        perf = perfs[0] + np.mean(perfs) + np.min(perfs)

        if perf > best_perf:
            best_perf = perf
            best_positions = positions
        return -perf  # CMA-ES minimise

    no_improve = 0
    best_score = float("inf")

    for i in range(n_calls):
        solutions = es.ask()
        scores = [objective(x) for x in solutions]
        es.tell(solutions, scores)
        es.disp()
        min_score = min(scores)
        if min_score < best_score:
            best_score = min_score
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= early_stop_rounds:
            print(f"Early stopping at iteration {i}")
            break

    xbest = es.result[0]
    xbest = [float(x) for x in xbest]

    return xbest, es.result[1], best_positions

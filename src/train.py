"""
train.py

This module contains the training pipeline for the project. It includes functions for loading data, preprocessing, training models, and saving results.

Functions:
- main(): Entry point for the training pipeline.
- load_data(): Loads and preprocesses the training data.
- train_model(): Trains the machine learning model.
- save_results(): Saves the training results to the outputs directory.

"""

# Import necessary libraries
import io
import json
import os
from itertools import product
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from tqdm import tqdm
from xgboost.callback import TrainingCallback

import config
from src.cuda import auto_select_gpu
from src.interval import get_interval_type
from src.path import get_project_root
from src.printlog import PrintLog
from src.trade import build_trade_data


class EvalF1Callback(TrainingCallback):
    """
    Custom XGBoost callback to evaluate F1 score during training and implement early stopping.
    """

    def __init__(self, dvalid, y_valid, patience=100, log=False):
        self.dvalid = dvalid
        self.y_valid = (
            y_valid.values.ravel() if hasattr(y_valid, "values") else np.array(y_valid)
        )
        self.f1_scores = []
        self.best_f1 = -np.inf
        self.best_iter = 0
        self.patience = patience
        self.wait = 0
        self.log = log

    def after_iteration(self, model, epoch, evals_log):
        # Compute F1 score on validation set
        y_pred_prob = model.predict(self.dvalid)
        y_pred = np.argmax(y_pred_prob, axis=1)
        f1 = f1_score(self.y_valid, y_pred, average="macro")
        self.f1_scores.append(f1)
        # Update best F1 and track patience counter
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_iter = epoch
            self.wait = 0
        else:
            self.wait += 1
        # Stop training if no improvement after patience rounds
        if self.wait >= self.patience:
            if self.log:
                print(
                    f"best F1 at iteration {self.best_iter+1} (F1={self.best_f1:.4f})"
                )
            return True  # stop training
        return False  # continue training


def compute_f1(trade_data, threshold, label="_A_Long", close_class=None, do_plot=False):
    """
    Compute F1 score on trade data filtered by probability threshold.

    Args:
        trade_data (dict): Trade data containing predictions and ground truth labels.
        threshold (float): Probability threshold for filtering predictions.
        label (str): Label suffix for output files.
        close_class (int): Class label to be considered as close (for binary classification).
        do_plot (bool): Whether to plot the F1 scores.

    Returns:
        tuple: Mean F1 score across the evaluated time periods and total positions count.
    """
    dates_f1 = []
    values_f1 = []
    mask_blue = []
    mask_red = []
    mask_green = []
    mask_orange = []
    total_count = 0

    sorted_keys = list(sorted(trade_data.keys()))

    for index, current_date in enumerate(sorted_keys):
        item = trade_data[current_date]
        y_pred = []
        y_truth = []
        y_prob = []
        y_index = []
        for stock in item:
            y_pred.append(item[stock]["class"])
            y_truth.append(item[stock]["Y"])
            y_prob.append(max(item[stock]["ybull"], item[stock]["ybear"]))

        y_index = np.array(y_index)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)
        y_truth = np.array(y_truth)

        # Filter predictions by probability threshold
        mask = y_prob >= threshold
        total_count += int(np.sum(mask))

        y_pred_filtered = y_pred[mask]
        y_truth_filtered = y_truth[mask]

        if close_class is not None:
            y_pred_filtered = (y_pred_filtered != close_class).astype(int)
            y_truth_filtered = (y_truth_filtered != close_class).astype(int)

        if len(y_truth_filtered) > 0:
            f1 = f1_score(y_truth_filtered, y_pred_filtered, average="macro")
            values_f1.append(f1)
            period = get_interval_type(current_date)
            dates_f1.append(pd.to_datetime(current_date, format="%Y-%m-%d"))
            # Color code based on interval type (A, B, C, D)
            if "A" in period:
                mask_red.append(False)
                mask_blue.append(True)
                mask_green.append(False)
                mask_orange.append(False)
            elif "B" in period:
                mask_red.append(True)
                mask_blue.append(False)
                mask_green.append(False)
                mask_orange.append(False)
            elif "C" in period:
                mask_red.append(False)
                mask_green.append(True)
                mask_blue.append(False)
                mask_orange.append(False)
            elif "D" in period:
                mask_red.append(False)
                mask_green.append(True)
                mask_blue.append(False)
                mask_orange.append(True)

    values_f1 = np.array(values_f1)
    dates_f1 = np.array(dates_f1)
    mean_f1 = float(np.mean(values_f1)) if len(values_f1) else 0.0
    std_f1 = float(np.std(values_f1)) if len(values_f1) else 0.0
    max_f1 = float(np.max(values_f1)) if len(values_f1) else 0.0
    min_f1 = float(np.min(values_f1)) if len(values_f1) else 0.0

    # Plot F1 scores with color coding if requested
    if do_plot and len(values_f1) == 0:
        return mean_f1, total_count

    if do_plot:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_yscale("log")

        ax1.plot(
            dates_f1[mask_blue],
            values_f1[mask_blue],
            color="blue",
            marker=".",
            linestyle="None",
            markersize=4,
        )
        ax1.plot(
            dates_f1[mask_red],
            values_f1[mask_red],
            color="red",
            marker=".",
            linestyle="None",
            markersize=4,
        )
        ax1.plot(
            dates_f1[mask_green],
            values_f1[mask_green],
            color="green",
            marker=".",
            linestyle="None",
            markersize=4,
        )
        ax1.plot(
            dates_f1[mask_orange],
            values_f1[mask_orange],
            color="orange",
            marker=".",
            linestyle="None",
            markersize=4,
        )

        ymin = np.min(values_f1) * 0.9
        ymax = np.max(values_f1) * 1.1

        ax1.set_ylim(bottom=ymin, top=ymax)
        ax1.set_ylabel("F1 Value", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")

        metrics_text = (
            f"F1 target: {config.F1_TARGET}\n"
            f"  threshold: {threshold:.4f}\n"
            f"  positions count: {total_count}\n"
            f"  mean f1: {mean_f1:.3f}\n"
            f"  std f1: {std_f1:.3f}\n"
            f"  max f1: {max_f1:.3f}\n"
            f"  min f1: {min_f1:.3f}\n"
        )
        ax1.text(
            0.01,
            0.99,
            metrics_text,
            transform=ax1.transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        image = Image.open(buf)
        image.save(
            os.path.join(
                local_log.output_dir_time,
                f"best_transitions_{label}.png",
            )
        )
        buf.close()
    return mean_f1, total_count


def search_threshold_for_f1_target(
    trade_data,
    f1_target=config.F1_TARGET,
    threshold_min=0.65,
    threshold_max=0.95,
    threshold_step=config.F1_THRESHOLD_STEP,
    step_multipliers=(50000, 10000, 5000, 1000, 500, 100, 50, 10, 5, 2, 1),
):
    """
    Search for threshold that maximizes positions count while meeting the F1 target,
    using a coarse-to-fine sweep that narrows bounds between passes.

    Args:
        trade_data (dict): Trade data containing predictions.
        f1_target (float): Target mean F1 score threshold.
        threshold_min (float): Minimum threshold value.
        threshold_max (float): Maximum threshold value.
        threshold_step (float): Finest threshold step size.
        step_multipliers (tuple): Coarse-to-fine step multipliers.

    Returns:
        tuple: Optimal threshold, count, mean F1, and whether target was met.
    """
    best_threshold = threshold_min
    best_count = -1
    best_mean_f1 = -np.inf
    hit_target = False

    fallback_threshold = threshold_min
    fallback_count = -1
    fallback_mean_f1 = -np.inf

    search_min = threshold_min
    search_max = threshold_max

    for step_multiplier in step_multipliers:
        step = threshold_step * step_multiplier
        if step <= 0 or search_min >= search_max:
            continue

        local_best_threshold = search_min
        local_best_count = -1
        local_best_mean_f1 = -np.inf
        local_hit_target = False

        local_fallback_threshold = search_min
        local_fallback_count = -1
        local_fallback_mean_f1 = -np.inf

        # Progress bar helps track the threshold sweep during training.
        thresholds = np.arange(search_min, search_max, step)
        for threshold in tqdm(
            thresholds,
            desc=f"Threshold search (F1>={f1_target}) step={step:.6f}",
            leave=False,
        ):
            mean_f1, count = compute_f1(trade_data, threshold, do_plot=False)
            if mean_f1 >= f1_target and count > local_best_count:
                local_best_threshold = threshold
                local_best_count = count
                local_best_mean_f1 = mean_f1
                local_hit_target = True
                # First threshold that meets target gives the max count.
                break
            if mean_f1 > local_fallback_mean_f1 or (
                np.isclose(mean_f1, local_fallback_mean_f1)
                and count > local_fallback_count
            ):
                local_fallback_threshold = threshold
                local_fallback_count = count
                local_fallback_mean_f1 = mean_f1

        if local_hit_target:
            best_threshold = local_best_threshold
            best_count = local_best_count
            best_mean_f1 = local_best_mean_f1
            hit_target = True
            center = local_best_threshold
        else:
            center = local_fallback_threshold

        if local_fallback_mean_f1 > fallback_mean_f1 or (
            np.isclose(local_fallback_mean_f1, fallback_mean_f1)
            and local_fallback_count > fallback_count
        ):
            fallback_threshold = local_fallback_threshold
            fallback_count = local_fallback_count
            fallback_mean_f1 = local_fallback_mean_f1

        # Narrow bounds around the best coarse threshold for the next pass.
        search_min = max(threshold_min, center - step)
        search_max = min(threshold_max, center + step)

    if hit_target:
        return best_threshold, best_count, best_mean_f1, True
    return fallback_threshold, fallback_count, fallback_mean_f1, False


def split_trade_data(trade_data):
    """
    Split trade data into long and short positions by interval type (A/B).

    Args:
        trade_data (dict): Trade data containing predictions and ground truth labels.

    Returns:
        tuple: Dictionaries containing split trade data (long/short, A/B).
    """
    sorted_keys = list(sorted(trade_data.keys()))
    trade_dataA_long = {}
    trade_dataA_short = {}
    trade_dataB_long = {}
    trade_dataB_short = {}

    for index, current_date in enumerate(sorted_keys):
        type = get_interval_type(current_date)
        item = trade_data[current_date]
        item_long = {}
        item_short = {}

        item = trade_data[current_date]
        for stock in item:
            if item[stock]["index_short"] == 0:
                item_short[stock] = item[stock]
            if item[stock]["index_long"] == 0:
                item_long[stock] = item[stock]

        # Assign to appropriate interval and direction
        if "A" in type:
            trade_dataA_long[current_date] = item_long
            trade_dataA_short[current_date] = item_short
        elif "B" in type:
            trade_dataB_long[current_date] = item_long
            trade_dataB_short[current_date] = item_short
        if "C" in type:
            trade_dataA_long[current_date] = item_long
            trade_dataA_short[current_date] = item_short
        elif "D" in type:
            trade_dataB_long[current_date] = item_long
            trade_dataB_short[current_date] = item_short
    return (
        trade_dataA_long,
        trade_dataA_short,
        trade_dataB_long,
        trade_dataB_short,
    )


def compute_positions_score(labels, counts, target_hits, long_weight=0.6, short_weight=0.4):
    """
    Compute a weighted positions score by separating long and short segments.

    Returns:
        tuple: positions_score, positions_mean, positions_std
    """
    target_counts = [count if hit else 0 for count, hit in zip(counts, target_hits)]
    long_counts = [
        count for count, label in zip(target_counts, labels) if "Long" in label
    ]
    short_counts = [
        count for count, label in zip(target_counts, labels) if "Short" in label
    ]

    long_mean = float(np.mean(long_counts)) if len(long_counts) else 0.0
    long_std = float(np.std(long_counts)) if len(long_counts) else 0.0
    short_mean = float(np.mean(short_counts)) if len(short_counts) else 0.0
    short_std = float(np.std(short_counts)) if len(short_counts) else 0.0

    long_score = long_mean - long_std
    short_score = short_mean - short_std
    positions_score = long_weight * long_score + short_weight * short_score

    positions_mean = float(np.mean(target_counts)) if len(target_counts) else 0.0
    positions_std = float(np.std(target_counts)) if len(target_counts) else 0.0
    return positions_score, positions_mean, positions_std


def plot_top_f1(do_plot=False):
    """
    Evaluate F1 scores and target thresholds across all interval types.

    Args:
        do_plot (bool): Whether to plot the F1 scores.

    Returns:
        tuple: F1 scores, thresholds, counts, target hits, and labels.
    """
    # Evaluate F1 with thresholds that maximize positions above the target
    trade_data = build_trade_data(
        model_path=Path(get_project_root()) / local_log.output_dir_time,
        data_path=data_path,
        file_date_str=config.BENCHMARK_END_DATE,
        start_date=pd.to_datetime(config.BENCHMARK_START_DATE, format="%Y-%m-%d"),
        end_date=pd.to_datetime(config.TRAIN_END_DATE, format="%Y-%m-%d"),
    )
    (
        trade_dataA_long,
        trade_dataA_short,
        trade_dataB_long,
        trade_dataB_short,
    ) = split_trade_data(trade_data)

    labels = ["A_Long", "B_Long", "A_Short", "B_Short"]
    segments = [
        (trade_dataA_long, "_A_Long"),
        (trade_dataB_long, "_B_Long"),
        (trade_dataA_short, "_A_Short"),
        (trade_dataB_short, "_B_Short"),
    ]

    thresholds = []
    counts = []
    f1s = []
    target_hits = []

    for trade_data_segment, label_suffix in segments:
        threshold, count, mean_f1, hit_target = search_threshold_for_f1_target(
            trade_data_segment
        )
        if do_plot:
            mean_f1, count = compute_f1(
                trade_data_segment,
                threshold,
                label=label_suffix,
                do_plot=True,
            )
        thresholds.append(float(threshold))
        counts.append(int(count))
        f1s.append(float(mean_f1))
        target_hits.append(bool(hit_target))

    return f1s, thresholds, counts, target_hits, labels


def save_thresholds_json(output_dir, labels, thresholds, counts, target_hits, f1s):
    """Save best thresholds to a JSON file for downstream CMA-ES initialization."""
    data = {}
    for label, threshold, count, hit, f1 in zip(
        labels, thresholds, counts, target_hits, f1s
    ):
        data[label] = {
            "threshold": float(threshold),
            "count": int(count),
            "f1": float(f1),
            "hit_target": bool(hit),
            "f1_target": float(config.F1_TARGET),
        }
    thresholds_path = os.path.join(output_dir, "thresholds.json")
    with open(thresholds_path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":

    # Load environment variables from .env file
    # This ensures sensitive information like API keys is securely loaded into the environment
    load_dotenv()

    # Set the path to the data directory and create it if it doesn't already exist
    data_path = Path(get_project_root()) / "data" / "fmp_data"

    # Automatically select GPU based on available memory
    auto_select_gpu(threshold_mb=500)

    # Load training and test datasets for each interval
    df_part1A_X = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part1A_X.csv")
    df_part1A_Y = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part1A_Y.csv")
    df_part1B_X = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part1B_X.csv")
    df_part1B_Y = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part1B_Y.csv")
    df_part2A_X = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part2A_X.csv")
    df_part2A_Y = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part2A_Y.csv")
    df_part2B_X = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part2B_X.csv")
    df_part2B_Y = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part2B_Y.csv")
    df_part3A_X = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part3A_X.csv")
    df_part3A_Y = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part3A_Y.csv")
    df_part3B_X = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part3B_X.csv")
    df_part3B_Y = pd.read_csv(data_path / f"{config.BENCHMARK_END_DATE}_part3B_Y.csv")

    # Drop unnecessary columns
    df_part1A_X.drop(columns=["index", "open", "stock_name"], inplace=True)
    df_part1B_X.drop(columns=["index", "open", "stock_name"], inplace=True)
    df_part2A_X.drop(columns=["index", "open", "stock_name"], inplace=True)
    df_part2B_X.drop(columns=["index", "open", "stock_name"], inplace=True)
    df_part3A_X.drop(columns=["index", "open", "stock_name"], inplace=True)
    df_part3B_X.drop(columns=["index", "open", "stock_name"], inplace=True)

    # Number of iterations for XGBoost model training
    n_estimators = 1000

    # Shuffle datasets for random sampling
    df_part1A_X, df_part1A_Y = shuffle(df_part1A_X, df_part1A_Y, random_state=42)
    df_part1B_X, df_part1B_Y = shuffle(df_part1B_X, df_part1B_Y, random_state=42)
    df_part2A_X, df_part2A_Y = shuffle(df_part2A_X, df_part2A_Y, random_state=42)
    df_part2B_X, df_part2B_Y = shuffle(df_part2B_X, df_part2B_Y, random_state=42)
    df_part3A_X, df_part3A_Y = shuffle(df_part3A_X, df_part3A_Y, random_state=42)
    df_part3B_X, df_part3B_Y = shuffle(df_part3B_X, df_part3B_Y, random_state=42)

    # Concatenate all interval data
    df_partA_X = pd.concat([df_part1A_X, df_part2A_X, df_part3A_X])
    df_partA_Y = pd.concat([df_part1A_Y, df_part2A_Y, df_part3A_Y])
    df_partB_X = pd.concat([df_part1B_X, df_part2B_X, df_part3B_X])
    df_partB_Y = pd.concat([df_part1B_Y, df_part2B_Y, df_part3B_Y])

    # Create XGBoost DMatrix objects for training and validation
    dtestA = xgb.DMatrix(df_partA_X, label=df_partA_Y.values.ravel())
    dtestB = xgb.DMatrix(df_partB_X, label=df_partB_Y.values.ravel())

    dtrain1A = xgb.DMatrix(df_part1A_X, label=df_part1A_Y.values.ravel())
    dtrain2A = xgb.DMatrix(df_part2A_X, label=df_part2A_Y.values.ravel())
    dtrain3A = xgb.DMatrix(df_part3A_X, label=df_part3A_Y.values.ravel())
    dtrain1B = xgb.DMatrix(df_part1B_X, label=df_part1B_Y.values.ravel())
    dtrain2B = xgb.DMatrix(df_part2B_X, label=df_part2B_Y.values.ravel())
    dtrain3B = xgb.DMatrix(df_part3B_X, label=df_part3B_Y.values.ravel())

    dtrainA = xgb.DMatrix(df_partA_X, label=df_partA_Y.values.ravel())
    dtrainB = xgb.DMatrix(df_partB_X, label=df_partB_Y.values.ravel())

    # XGBoost hyperparameters
    params = {
        "tree_method": "hist",
        "device": "cuda",
        "objective": "multi:softprob",
        "num_class": len(np.unique(df_part1A_Y)),
        "max_depth": 5,
        "learning_rate": 0.01,
        "subsample": 0.6,
        "colsample_bytree": 0.7,
        "gamma": 4,
        "min_child_weight": 5,
        "reg_alpha": 0.4,
        "reg_lambda": 4,
        "eval_metric": "mlogloss",
    }

    # Define hyperparameter search grid
    param_grid = config.PARAM_GRID.copy()

    # Generate all combinations of hyperparameters (warning: can be very long if grid is large)
    grid = list(
        product(
            param_grid["patience"],
            param_grid["max_depth"],
            param_grid["learning_rate"],
            param_grid["subsample"],
            param_grid["colsample_bytree"],
            param_grid["gamma"],
            param_grid["min_child_weight"],
            param_grid["reg_alpha"],
            param_grid["reg_lambda"],
            param_grid["mean_std_power"],
            param_grid["top_features"],
        )
    )

    local_log = PrintLog(extra_name="_train", enable=False)

    best_positions_score = -np.inf
    best_positions_mean = -np.inf
    best_importance_df_sorted_by_std_mean = None
    best_modela = None
    best_modelb = None
    best_selected_features = None
    best_f1_callbacka_best_iter = None
    best_f1_callbackb_best_iter = None

    # Train initial models for each interval part
    dtestA = xgb.DMatrix(df_partA_X, label=df_partA_Y.values.ravel())
    dtestB = xgb.DMatrix(df_partB_X, label=df_partB_Y.values.ravel())

    patience = 100
    params_grid = params.copy()

    # Train models for interval A parts (1, 2, 3)
    f1_callback = EvalF1Callback(dtestB, df_partB_Y, patience=patience)

    best_model1 = xgb.train(
        params_grid,
        dtrain1A,
        num_boost_round=n_estimators,
        evals=[(dtrain1A, "train"), (dtestB, "eval")],
        callbacks=[f1_callback],
        verbose_eval=False,
    )
    importance1 = best_model1.get_score(importance_type="weight")

    f1_callback = EvalF1Callback(dtestB, df_partB_Y, patience=patience)
    best_model2 = xgb.train(
        params_grid,
        dtrain2A,
        num_boost_round=n_estimators,
        evals=[(dtrain2A, "train"), (dtestB, "eval")],
        callbacks=[f1_callback],
        verbose_eval=False,
    )
    importance2 = best_model2.get_score(importance_type="weight")

    f1_callback = EvalF1Callback(dtestB, df_partB_Y, patience=patience)
    best_model3 = xgb.train(
        params_grid,
        dtrain3A,
        num_boost_round=n_estimators,
        evals=[(dtrain3A, "train"), (dtestB, "eval")],
        callbacks=[f1_callback],
        verbose_eval=False,
    )
    importance3 = best_model3.get_score(importance_type="weight")

    # Train models for interval B parts (1, 2, 3)
    f1_callback = EvalF1Callback(dtestA, df_partA_Y, patience=patience)
    best_model4 = xgb.train(
        params_grid,
        dtrain1B,
        num_boost_round=n_estimators,
        evals=[(dtrain1B, "train"), (dtestA, "eval")],
        callbacks=[f1_callback],
        verbose_eval=False,
    )
    importance4 = best_model4.get_score(importance_type="weight")

    f1_callback = EvalF1Callback(dtestA, df_partA_Y, patience=patience)
    best_model5 = xgb.train(
        params_grid,
        dtrain2B,
        num_boost_round=n_estimators,
        evals=[(dtrain2B, "train"), (dtestA, "eval")],
        callbacks=[f1_callback],
        verbose_eval=False,
    )
    importance5 = best_model5.get_score(importance_type="weight")

    f1_callback = EvalF1Callback(dtestA, df_partA_Y, patience=patience)
    best_model6 = xgb.train(
        params_grid,
        dtrain3B,
        num_boost_round=n_estimators,
        evals=[(dtrain3B, "train"), (dtestA, "eval")],
        callbacks=[f1_callback],
        verbose_eval=False,
    )
    importance6 = best_model6.get_score(importance_type="weight")

    # Build feature importance DataFrame correctly
    importance_df = pd.DataFrame(
        {
            "feature": df_partA_X.columns,
            "interval_1": [importance1.get(f, 0) for f in df_partA_X.columns],
            "interval_2": [importance2.get(f, 0) for f in df_partA_X.columns],
            "interval_3": [importance3.get(f, 0) for f in df_partA_X.columns],
            "interval_4": [importance4.get(f, 0) for f in df_partB_X.columns],
            "interval_5": [importance5.get(f, 0) for f in df_partB_X.columns],
            "interval_6": [importance6.get(f, 0) for f in df_partB_X.columns],
        }
    )

    # Replace NaN with 0 (safety check)
    importance_df = importance_df.fillna(0)
    # Normalize each column by its maximum (excluding 'feature' column)
    cols = [
        "interval_1",
        "interval_2",
        "interval_3",
        "interval_4",
        "interval_5",
        "interval_6",
    ]
    importance_df[cols] = importance_df[cols] / importance_df[cols].max()

    # Filter out features that have zero importance in any interval
    importance_df = importance_df[(importance_df[cols] != 0).all(axis=1)]
    # Compute mean, standard deviation, and mean/std ratio
    importance_df["mean"] = importance_df[cols].mean(axis=1)
    importance_df["std"] = importance_df[cols].std(axis=1)

    # Grid search over hyperparameters
    for (
        patience,
        max_depth,
        learning_rate,
        subsample,
        colsample_bytree,
        gamma,
        min_child_weight,
        reg_alpha,
        reg_lambda,
        mean_std_power,
        top_features,
    ) in tqdm(grid):

        params_grid = params.copy()
        params_grid["max_depth"] = max_depth
        params_grid["learning_rate"] = learning_rate
        params_grid["subsample"] = subsample
        params_grid["colsample_bytree"] = colsample_bytree
        params_grid["gamma"] = gamma
        params_grid["min_child_weight"] = min_child_weight
        params_grid["reg_alpha"] = reg_alpha
        params_grid["reg_lambda"] = reg_lambda

        # Compute mean/std ratio for feature selection
        importance_df["mean/std"] = importance_df["mean"] / (
            importance_df["std"] ** mean_std_power
        )
        importance_df_sorted_by_std_mean = importance_df.sort_values(
            by="mean/std", ascending=False
        )

        # Select top features based on mean/std ratio
        importance_df_best = importance_df_sorted_by_std_mean[:top_features]
        selected_features = list(importance_df_best["feature"])

        df_partB_X_selected = df_partB_X[selected_features]
        df_partA_X_selected = df_partA_X[selected_features]
        df_partB_test_X_selected = df_partB_X[selected_features]
        df_partA_test_X_selected = df_partA_X[selected_features]

        dtrain = xgb.DMatrix(df_partA_X_selected, label=df_partA_Y.values.ravel())
        dtest = xgb.DMatrix(df_partB_test_X_selected, label=df_partB_Y.values.ravel())

        f1_callbacka = EvalF1Callback(dtest, df_partB_Y, patience=patience)
        modela = xgb.train(
            params_grid,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dtest, "eval")],
            callbacks=[f1_callbacka],
            verbose_eval=False,
        )

        df_test_y_pred = np.argmax(
            modela.predict(dtest, iteration_range=(0, f1_callbacka.best_iter + 1)),
            axis=1,
        )
        f1A = f1_score(df_partB_Y, df_test_y_pred, average="macro")

        df_partB_X_selected = df_partB_X[selected_features]
        df_partA_X_selected = df_partA_X[selected_features]

        df_partB_test_X_selected = df_partB_X[selected_features]
        df_partA_test_X_selected = df_partA_X[selected_features]

        dtrain = xgb.DMatrix(df_partB_X_selected, label=df_partB_Y.values.ravel())
        dtest = xgb.DMatrix(df_partA_test_X_selected, label=df_partA_Y.values.ravel())

        f1_callbackb = EvalF1Callback(dtest, df_partA_Y, patience=patience)
        modelb = xgb.train(
            params_grid,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dtest, "eval")],
            callbacks=[f1_callbackb],
            verbose_eval=False,
        )

        df_test_y_pred = np.argmax(
            modelb.predict(dtest, iteration_range=(0, f1_callbackb.best_iter + 1)),
            axis=1,
        )

        f1B = f1_score(df_partA_Y, df_test_y_pred, average="macro")

        # Save current models and selected features
        importance_df_sorted_by_std_mean.to_csv(
            os.path.join(
                local_log.output_dir_time, "importance_df_sorted_by_std_mean.csv"
            ),
            index=False,
        )
        selected_features_path = os.path.join(
            local_log.output_dir_time, "selected_featuresA.json"
        )
        with open(selected_features_path, "w") as f:
            json.dump(selected_features, f, indent=2)
        selected_features_path = os.path.join(
            local_log.output_dir_time, "selected_featuresB.json"
        )
        with open(selected_features_path, "w") as f:
            json.dump(selected_features, f, indent=2)
        model_path = os.path.join(local_log.output_dir_time, "best_modelA.pkl")
        joblib.dump(modela, model_path)
        ntree_limit_path = os.path.join(
            local_log.output_dir_time, "best_modelA_ntree_limit.json"
        )
        with open(ntree_limit_path, "w") as f:
            json.dump({"ntree_limit": int(f1_callbacka.best_iter + 1)}, f)
        model_path = os.path.join(local_log.output_dir_time, "best_modelB.pkl")
        joblib.dump(modelb, model_path)
        ntree_limit_path = os.path.join(
            local_log.output_dir_time, "best_modelB_ntree_limit.json"
        )
        with open(ntree_limit_path, "w") as f:
            json.dump({"ntree_limit": int(f1_callbackb.best_iter + 1)}, f)

        # Evaluate on full test set with target-driven thresholds
        f1s, thresholds, counts, target_hits, labels = plot_top_f1(do_plot=False)
        positions_score, positions_mean, positions_std = compute_positions_score(
            labels, counts, target_hits
        )
        mean_f1_all = float(np.mean(f1s)) if len(f1s) else 0.0
        f1_str = [f"{f1:.2f}" for f1 in f1s]
        thresholds_str = ", ".join(
            f"{label}={threshold:.4f}" for label, threshold in zip(labels, thresholds)
        )
        counts_str = ", ".join(
            f"{label}={count}" for label, count in zip(labels, counts)
        )
        target_hit_str = ", ".join(
            f"{label}={'ok' if hit else 'miss'}"
            for label, hit in zip(labels, target_hits)
        )

        # Update best model if positions score improved.
        if positions_score > best_positions_score or (
            np.isclose(positions_score, best_positions_score)
            and positions_mean > best_positions_mean
        ):
            # plot_top_f1(do_plot=True)
            save_thresholds_json(
                local_log.output_dir_time,
                labels,
                thresholds,
                counts,
                target_hits,
                f1s,
            )

            # Save best variables
            best_positions_score = positions_score
            best_positions_mean = positions_mean
            best_importance_df_sorted_by_std_mean = (
                importance_df_sorted_by_std_mean.copy()
            )
            best_modela = modela.copy()
            best_modelb = modelb.copy()
            best_selected_features = selected_features.copy()
            best_f1_callbacka_best_iter = f1_callbacka.best_iter
            best_f1_callbackb_best_iter = f1_callbackb.best_iter

            with local_log:
                print("\n")
                print(f"Best model params: {params_grid}")
                print(
                    f"Selected features: {len(selected_features)}/{len(df_partB_X.columns)}"
                )
                print(f"max_depth: {max_depth}")
                print(f"mean_std_power: {mean_std_power}")
                print(f"F1 scores: {f1_str}")
                print(f"F1_TARGET={config.F1_TARGET} thresholds: {thresholds_str}")
                print(f"Positions count: {counts_str}")
                print(f"Target hit: {target_hit_str}")
                print(
                    "Best Positions score: "
                    f"{positions_score:.2f} (mean={positions_mean:.2f}, std={positions_std:.2f})"
                )
        else:
            print(f"F1 scores: {f1_str}")
            print(f"F1_TARGET={config.F1_TARGET} thresholds: {thresholds_str}")
            print(f"Positions count: {counts_str}")
            pass
    # Save best final model
    if best_importance_df_sorted_by_std_mean is not None:
        best_importance_df_sorted_by_std_mean.to_csv(
            os.path.join(
                local_log.output_dir_time, "importance_df_sorted_by_std_mean.csv"
            ),
            index=False,
        )
        selected_features_path = os.path.join(
            local_log.output_dir_time, "selected_featuresA.json"
        )
        with open(selected_features_path, "w") as f:
            json.dump(best_selected_features, f, indent=2)

        selected_features_path = os.path.join(
            local_log.output_dir_time, "selected_featuresB.json"
        )
        with open(selected_features_path, "w") as f:
            json.dump(best_selected_features, f, indent=2)

        model_path = os.path.join(local_log.output_dir_time, "best_modelA.pkl")
        joblib.dump(best_modela, model_path)
        ntree_limit_path = os.path.join(
            local_log.output_dir_time, "best_modelA_ntree_limit.json"
        )
        with open(ntree_limit_path, "w") as f:
            json.dump({"ntree_limit": int(best_f1_callbacka_best_iter + 1)}, f)
        model_path = os.path.join(local_log.output_dir_time, "best_modelB.pkl")
        joblib.dump(best_modelb, model_path)

        ntree_limit_path = os.path.join(
            local_log.output_dir_time, "best_modelB_ntree_limit.json"
        )
        with open(ntree_limit_path, "w") as f:
            json.dump({"ntree_limit": int(best_f1_callbackb_best_iter + 1)}, f)

    # Second pass

    importanceA = best_modela.get_score(importance_type="weight")
    importanceB = best_modelb.get_score(importance_type="weight")

    importance = {}

    for key in set(importanceA.keys()).union(set(importanceB.keys())):
        importance[key] = {
            "mean": (importanceA.get(key, 0) + importanceB.get(key, 0)) / 2,
            "std": float(
                np.std(
                    [importanceA.get(key, 0), importanceB.get(key, 0)],
                    ddof=1,
                )
            ),
            "feature": key,
        }

    # Define hyperparameter search grid
    param_grid = config.PARAM_GRID.copy()

    grid = list(
        product(
            param_grid["patience"],
            param_grid["max_depth"],
            param_grid["learning_rate"],
            param_grid["subsample"],
            param_grid["colsample_bytree"],
            param_grid["gamma"],
            param_grid["min_child_weight"],
            param_grid["reg_alpha"],
            param_grid["reg_lambda"],
            param_grid["mean_std_power_2nd"],
            list(
                range(
                    int(0.6 * len(importance)),
                    len(importance),
                    1,
                )
            ),
        )
    )

    # Grid search over hyperparameters
    for (
        patience,
        max_depth,
        learning_rate,
        subsample,
        colsample_bytree,
        gamma,
        min_child_weight,
        reg_alpha,
        reg_lambda,
        mean_std_power_2nd,
        top_features,
    ) in tqdm(grid):

        params_grid = params.copy()
        params_grid["max_depth"] = max_depth
        params_grid["learning_rate"] = learning_rate
        params_grid["subsample"] = subsample
        params_grid["colsample_bytree"] = colsample_bytree
        params_grid["gamma"] = gamma
        params_grid["min_child_weight"] = min_child_weight
        params_grid["reg_alpha"] = reg_alpha
        params_grid["reg_lambda"] = reg_lambda

        sorted_importance = sorted(
            importance.items(),
            key=lambda x: x[1]["mean"] / ((0.01 + x[1]["std"]) ** mean_std_power_2nd),
            reverse=True,
        )

        sorted_selected_features = [item[0] for item in sorted_importance]

        selected_features = sorted_selected_features[:top_features]

        df_partB_X_selected = df_partB_X[selected_features]
        df_partA_X_selected = df_partA_X[selected_features]
        df_partB_test_X_selected = df_partB_X[selected_features]
        df_partA_test_X_selected = df_partA_X[selected_features]

        dtrain = xgb.DMatrix(df_partA_X_selected, label=df_partA_Y.values.ravel())
        dtest = xgb.DMatrix(df_partB_test_X_selected, label=df_partB_Y.values.ravel())

        f1_callbacka = EvalF1Callback(dtest, df_partB_Y, patience=patience)
        modela = xgb.train(
            params_grid,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dtest, "eval")],
            callbacks=[f1_callbacka],
            verbose_eval=False,
        )

        df_test_y_pred = np.argmax(
            modela.predict(dtest, iteration_range=(0, f1_callbacka.best_iter + 1)),
            axis=1,
        )
        f1A = f1_score(df_partB_Y, df_test_y_pred, average="macro")

        df_partB_X_selected = df_partB_X[selected_features]
        df_partA_X_selected = df_partA_X[selected_features]

        df_partB_test_X_selected = df_partB_X[selected_features]
        df_partA_test_X_selected = df_partA_X[selected_features]

        dtrain = xgb.DMatrix(df_partB_X_selected, label=df_partB_Y.values.ravel())
        dtest = xgb.DMatrix(df_partA_test_X_selected, label=df_partA_Y.values.ravel())

        f1_callbackb = EvalF1Callback(dtest, df_partA_Y, patience=patience)
        modelb = xgb.train(
            params_grid,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dtest, "eval")],
            callbacks=[f1_callbackb],
            verbose_eval=False,
        )

        df_test_y_pred = np.argmax(
            modelb.predict(dtest, iteration_range=(0, f1_callbackb.best_iter + 1)),
            axis=1,
        )

        f1B = f1_score(df_partA_Y, df_test_y_pred, average="macro")

        # Save current models and selected features
        selected_features_path = os.path.join(
            local_log.output_dir_time, "selected_featuresA.json"
        )
        with open(selected_features_path, "w") as f:
            json.dump(selected_features, f, indent=2)
        selected_features_path = os.path.join(
            local_log.output_dir_time, "selected_featuresB.json"
        )
        with open(selected_features_path, "w") as f:
            json.dump(selected_features, f, indent=2)
        model_path = os.path.join(local_log.output_dir_time, "best_modelA.pkl")
        joblib.dump(modela, model_path)
        ntree_limit_path = os.path.join(
            local_log.output_dir_time, "best_modelA_ntree_limit.json"
        )
        with open(ntree_limit_path, "w") as f:
            json.dump({"ntree_limit": int(f1_callbacka.best_iter + 1)}, f)
        model_path = os.path.join(local_log.output_dir_time, "best_modelB.pkl")
        joblib.dump(modelb, model_path)
        ntree_limit_path = os.path.join(
            local_log.output_dir_time, "best_modelB_ntree_limit.json"
        )
        with open(ntree_limit_path, "w") as f:
            json.dump({"ntree_limit": int(f1_callbackb.best_iter + 1)}, f)

        # Evaluate on full test set with target-driven thresholds
        f1s, thresholds, counts, target_hits, labels = plot_top_f1(do_plot=False)
        positions_score, positions_mean, positions_std = compute_positions_score(
            labels, counts, target_hits
        )
        mean_f1_all = float(np.mean(f1s)) if len(f1s) else 0.0
        f1_str = [f"{f1:.2f}" for f1 in f1s]
        thresholds_str = ", ".join(
            f"{label}={threshold:.4f}" for label, threshold in zip(labels, thresholds)
        )
        counts_str = ", ".join(
            f"{label}={count}" for label, count in zip(labels, counts)
        )
        target_hit_str = ", ".join(
            f"{label}={'ok' if hit else 'miss'}"
            for label, hit in zip(labels, target_hits)
        )

        # Update best model if positions score improved.
        if positions_score > best_positions_score or (
            np.isclose(positions_score, best_positions_score)
            and positions_mean > best_positions_mean
        ):
            # plot_top_f1(do_plot=True)
            save_thresholds_json(
                local_log.output_dir_time,
                labels,
                thresholds,
                counts,
                target_hits,
                f1s,
            )

            best_positions_score = positions_score
            best_positions_mean = positions_mean
            best_modela = modela.copy()
            best_modelb = modelb.copy()

            best_selected_features = selected_features.copy()
            best_f1_callbacka_best_iter = f1_callbacka.best_iter
            best_f1_callbackb_best_iter = f1_callbackb.best_iter

            with local_log:
                print("\n")
                print(f"Best model params: {params_grid}")
                print(f"Selected features: {len(selected_features)}")
                print(f"max_depth: {max_depth}")
                print(f"F1 scores: {f1_str}")
                print(f"F1_TARGET={config.F1_TARGET} thresholds: {thresholds_str}")
                print(f"Positions count: {counts_str}")
                print(f"Target hit: {target_hit_str}")
                print(
                    "Best Positions score: "
                    f"{positions_score:.2f} (mean={positions_mean:.2f}, std={positions_std:.2f})"
                )
        else:
            print(f"F1 scores: {f1_str}")
            print(f"F1_TARGET={config.F1_TARGET} thresholds: {thresholds_str}")
            print(f"Positions count: {counts_str}")
            pass
    # Save best final model
    if best_modela is not None:
        selected_features_path = os.path.join(
            local_log.output_dir_time, "selected_featuresA.json"
        )
        with open(selected_features_path, "w") as f:
            json.dump(best_selected_features, f, indent=2)

        selected_features_path = os.path.join(
            local_log.output_dir_time, "selected_featuresB.json"
        )
        with open(selected_features_path, "w") as f:
            json.dump(best_selected_features, f, indent=2)

        model_path = os.path.join(local_log.output_dir_time, "best_modelA.pkl")
        joblib.dump(best_modela, model_path)
        ntree_limit_path = os.path.join(
            local_log.output_dir_time, "best_modelA_ntree_limit.json"
        )
        with open(ntree_limit_path, "w") as f:
            json.dump({"ntree_limit": int(best_f1_callbacka_best_iter + 1)}, f)
        model_path = os.path.join(local_log.output_dir_time, "best_modelB.pkl")
        joblib.dump(best_modelb, model_path)

        ntree_limit_path = os.path.join(
            local_log.output_dir_time, "best_modelB_ntree_limit.json"
        )
        with open(ntree_limit_path, "w") as f:
            json.dump({"ntree_limit": int(best_f1_callbackb_best_iter + 1)}, f)

    local_log.copy_last()

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
import argparse
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
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_sample_weight
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


def compute_perf(trade_data, threshold):
    gain = 1.0
    gain_max = 1.0
    gain_min_norm = 1.0
    count = 0
    loss_count = 0
    opportunity_days = []
    opportunity_gains = []
    sorted_keys = list(sorted(trade_data.keys()))
    for index, current_date in enumerate(sorted_keys):
        item = trade_data[current_date]
        candidates = []
        for stock in item:
            if "gain" not in item[stock]:
                continue
            yprob = 0.0
            if item[stock]["index_long"] == 0:
                yprob = item[stock]["ybull"]
            elif item[stock]["index_short"] == 0:
                yprob = item[stock]["ybear"]
            if yprob >= threshold:
                candidates.append((yprob, item[stock]["gain"]))

        if not candidates:
            opportunity_gains.append(gain)
            continue

        candidates.sort(key=lambda entry: entry[0], reverse=True)
        selected = candidates[:3]
        for _, trade_gain in selected:
            gain *= 1.0 + trade_gain
            if gain > gain_max:
                gain_max = gain
            if gain / gain_max < gain_min_norm:
                gain_min_norm = gain / gain_max
            if trade_gain < 0:
                loss_count += 1
            count += 1
        opportunity_days.append(index)
        opportunity_gains.append(gain)

    gain_per_trade = 1
    if count:
        gain_per_trade = gain ** (1 / count)

    span_factor = 0.0
    if opportunity_days:
        span = max(opportunity_days) - min(opportunity_days)
        total_span = max(len(sorted_keys) - 1, 1)
        span_factor = span / total_span

    regularity = 0.0
    if len(opportunity_days) > 20:
        gt = opportunity_gains[-1]
        gpt = gt ** (1 / len(opportunity_gains))
        norm_error = []
        g = 1.0
        for g_ in opportunity_gains:
            g *= gpt
            norm_error.append(min(1.0, abs(g - g_) / g))
        regularity = 1.0 - np.mean(norm_error)
        regularity = max(0.1, regularity) ** 3.0

    score = 0
    if gain_min_norm < 0.65:
        gain_min_norm = gain_min_norm**3.0

    if gain_per_trade > 1.005:
        score = (
            regularity
            * span_factor
            * (gain_per_trade ** len(opportunity_days))
            * gain_min_norm
        )

    dd = -(1.0 - gain_min_norm)

    return (
        gain_per_trade,
        dd,
        len(opportunity_days),
        score,
        regularity,
        opportunity_gains,
    )


def search_threshold_for_perf(
    trade_data,
    threshold_min=0.50,
    threshold_max=0.97,
    threshold_step=config.F1_THRESHOLD_STEP,
):
    best_score = 0
    best_count = 0
    best_gain_per_trade = 0
    best_threshold = 1.0
    best_dd = 0
    best_regularity = 0.0
    best_opportunity_gains = []

    thresholds = np.arange(threshold_min, threshold_max, threshold_step)
    for threshold in tqdm(thresholds, leave=False):
        (
            gain_per_trade,
            dd,
            count,
            score,
            regularity,
            opportunity_gains,
        ) = compute_perf(trade_data, threshold)
        if score > best_score:
            best_score = score
            best_count = count
            best_dd = dd
            best_threshold = threshold
            best_gain_per_trade = gain_per_trade
            best_regularity = regularity
            best_opportunity_gains = list(opportunity_gains)

    return (
        best_score,
        best_threshold,
        best_gain_per_trade,
        best_dd,
        best_count,
        best_regularity,
        best_opportunity_gains,
    )


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
    trade_dataB_long = {}

    for _, current_date in enumerate(sorted_keys):
        type = get_interval_type(current_date)
        item = trade_data[current_date]
        item_long = {}

        item = trade_data[current_date]
        for stock in item:
            if item[stock]["index_long"] == 0:
                item_long[stock] = item[stock]

        # Assign to appropriate interval and direction
        if "A" in type:
            trade_dataA_long[current_date] = item_long
        elif "B" in type:
            trade_dataB_long[current_date] = item_long
        if "C" in type:
            trade_dataA_long[current_date] = item_long
        elif "D" in type:
            trade_dataB_long[current_date] = item_long
    return (
        trade_dataA_long,
        trade_dataB_long,
    )


def save_opportunity_gains_plot(output_dir, label, opportunity_gains, feature_count):
    if not opportunity_gains:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"opportunity_gains_{label}_nfeat{int(feature_count)}.png"
    path = output_dir / filename

    plt.figure(figsize=(10, 4))
    plt.plot(opportunity_gains, color="#1f77b4", linewidth=1.5)
    plt.title(f"Opportunity gains ({label}) - nfeat={int(feature_count)}")
    plt.xlabel("Opportunity index")
    plt.ylabel("Gain")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def search_perfs_threshold(model_path, data_path, output_dir, feature_count, labels):
    trade_data = build_trade_data(
        model_path=model_path,
        data_path=data_path,
        file_date_str=config.BENCHMARK_END_DATE,
        start_date=pd.to_datetime(config.BENCHMARK_START_DATE, format="%Y-%m-%d"),
        end_date=pd.to_datetime(config.TRAIN_END_DATE, format="%Y-%m-%d"),
    )
    (
        trade_dataA_long,
        trade_dataB_long,
    ) = split_trade_data(trade_data)

    segments = [
        trade_dataA_long,
        trade_dataB_long,
    ]

    thresholds = []
    counts = []
    gain_per_trades = []
    dds = []
    scores = []
    regularities = []
    opportunity_gains_segments = []

    for trade_data_segment in segments:
        (
            score,
            threshold,
            gain_per_trade,
            dd,
            count,
            regularity,
            opportunity_gains,
        ) = search_threshold_for_perf(trade_data_segment)
        counts.append(int(count))
        gain_per_trades.append(float(gain_per_trade))
        dds.append(float(dd))
        thresholds.append(float(threshold))
        scores.append(score)
        regularities.append(float(regularity))
        opportunity_gains_segments.append(opportunity_gains)

    for label, opportunity_gains in zip(labels, opportunity_gains_segments):
        save_opportunity_gains_plot(output_dir, label, opportunity_gains, feature_count)

    return (
        scores,
        counts,
        gain_per_trades,
        dds,
        thresholds,
        regularities,
    )


THRESHOLD_SEGMENT_LABELS = ("A_long", "B_long")


def format_threshold_details(
    labels,
    thresholds,
    counts,
    gain_per_trades,
    dds,
    regularities,
):
    lines = []
    for label, threshold, count, gain_per_trade, dd, regularity in zip(
        labels,
        thresholds,
        counts,
        gain_per_trades,
        dds,
        regularities,
    ):
        lines.append(
            f"{label}: thres={threshold:.3f}, count={int(count)}, "
            f"gain={gain_per_trade:.3f}, dd={dd:.3f}, reg={regularity:.3f}"
        )
    return "\n".join(lines)


def save_best_thresholds(output_dir, labels, thresholds):
    thresholds_payload = {
        "labels": list(labels),
        "thresholds": [float(threshold) for threshold in thresholds],
    }
    latest_path = os.path.join(output_dir, "best_thresholds.json")
    with open(latest_path, "w") as f:
        json.dump(thresholds_payload, f, indent=2)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train XGBoost models.")
    parser.add_argument(
        "--max_depth",
        type=int,
        default=None,
        help="Override XGBoost max_depth in PARAM_GRID and base params.",
    )
    args = parser.parse_args()
    max_depth_override = args.max_depth

    if max_depth_override is not None:
        config.PARAM_GRID["max_depth"] = [max_depth_override]

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

    df_part1A_weights = compute_sample_weight(
        class_weight="balanced", y=df_part1A_Y.astype(int)
    )
    df_part1B_weights = compute_sample_weight(
        class_weight="balanced", y=df_part1B_Y.astype(int)
    )
    df_part2A_weights = compute_sample_weight(
        class_weight="balanced", y=df_part2A_Y.astype(int)
    )
    df_part2B_weights = compute_sample_weight(
        class_weight="balanced", y=df_part2B_Y.astype(int)
    )
    df_part3A_weights = compute_sample_weight(
        class_weight="balanced", y=df_part3A_Y.astype(int)
    )
    df_part3B_weights = compute_sample_weight(
        class_weight="balanced", y=df_part3B_Y.astype(int)
    )

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

    df_partA_weights = compute_sample_weight(class_weight="balanced", y=df_partA_Y)
    df_partB_weights = compute_sample_weight(class_weight="balanced", y=df_partB_Y)

    # Create XGBoost DMatrix objects for training and validation
    dtestA = xgb.DMatrix(df_partA_X, label=df_partA_Y.values.ravel())
    dtestB = xgb.DMatrix(df_partB_X, label=df_partB_Y.values.ravel())

    dtrain1A = xgb.DMatrix(
        df_part1A_X, label=df_part1A_Y.values.ravel(), weight=df_part1A_weights
    )
    dtrain2A = xgb.DMatrix(
        df_part2A_X, label=df_part2A_Y.values.ravel(), weight=df_part2A_weights
    )
    dtrain3A = xgb.DMatrix(
        df_part3A_X, label=df_part3A_Y.values.ravel(), weight=df_part3A_weights
    )
    dtrain1B = xgb.DMatrix(
        df_part1B_X, label=df_part1B_Y.values.ravel(), weight=df_part1B_weights
    )
    dtrain2B = xgb.DMatrix(
        df_part2B_X, label=df_part2B_Y.values.ravel(), weight=df_part2B_weights
    )
    dtrain3B = xgb.DMatrix(
        df_part3B_X, label=df_part3B_Y.values.ravel(), weight=df_part3B_weights
    )

    dtrainA = xgb.DMatrix(
        df_partA_X, label=df_partA_Y.values.ravel(), weight=df_partA_weights
    )
    dtrainB = xgb.DMatrix(
        df_partB_X, label=df_partB_Y.values.ravel(), weight=df_partB_weights
    )

    # XGBoost hyperparameters
    params = {
        "tree_method": "hist",
        "device": "cuda",
        "objective": "multi:softprob",
        "num_class": len(np.unique(df_part1A_Y)),
        "max_depth": max_depth_override if max_depth_override is not None else 5,
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

    best_score = -np.inf
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

        dtrain = xgb.DMatrix(
            df_partA_X_selected,
            label=df_partA_Y.values.ravel(),
            weight=df_partA_weights,
        )
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

        dtrain = xgb.DMatrix(
            df_partB_X_selected,
            label=df_partB_Y.values.ravel(),
            weight=df_partB_weights,
        )
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

        model_path = Path(get_project_root()) / local_log.output_dir_time
        (
            scores,
            counts,
            gain_per_trades,
            dds,
            thresholds,
            regularities,
        ) = search_perfs_threshold(
            model_path,
            data_path,
            local_log.output_dir_time,
            len(selected_features),
            THRESHOLD_SEGMENT_LABELS,
        )
        score = float(np.prod(scores))

        if score > best_score:
            best_score = score
            best_importance_df_sorted_by_std_mean = (
                importance_df_sorted_by_std_mean.copy()
            )
            best_modela = modela.copy()
            best_modelb = modelb.copy()
            best_selected_features = selected_features.copy()
            best_f1_callbacka_best_iter = f1_callbacka.best_iter
            best_f1_callbackb_best_iter = f1_callbackb.best_iter
            save_best_thresholds(
                local_log.output_dir_time,
                THRESHOLD_SEGMENT_LABELS,
                thresholds,
            )

            with local_log:
                print("\n")
                print(f"Best model params: {params_grid}")
                print(
                    f"Selected features: {len(selected_features)}/{len(df_partB_X.columns)}"
                )
                print(f"max_depth: {max_depth}")
                print(f"mean_std_power: {mean_std_power}")
                print(f"Best score: {best_score:.3f}")
                print("Threshold details:")
                print(
                    format_threshold_details(
                        THRESHOLD_SEGMENT_LABELS,
                        thresholds,
                        counts,
                        gain_per_trades,
                        dds,
                        regularities,
                    )
                )
        else:
            print(" ")
            print("Threshold details:")
            print(f"Score: {score:.4f}")
            print(
                format_threshold_details(
                    THRESHOLD_SEGMENT_LABELS,
                    thresholds,
                    counts,
                    gain_per_trades,
                    dds,
                    regularities,
                )
            )
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

    local_log.copy_last()

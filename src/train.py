import json
import os
from itertools import product

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import f1_score
from sklearn.utils import shuffle
from xgboost.callback import TrainingCallback

import config
from src.utils.printlog import PrintLog


class EvalF1Callback(TrainingCallback):
    def __init__(self, dvalid, y_valid, patience=100):
        self.dvalid = dvalid
        self.y_valid = (
            y_valid.values.ravel() if hasattr(y_valid, "values") else np.array(y_valid)
        )
        self.f1_scores = []
        self.best_f1 = -np.inf
        self.best_iter = 0
        self.patience = patience
        self.wait = 0

    def after_iteration(self, model, epoch, evals_log):
        y_pred_prob = model.predict(self.dvalid)
        y_pred = np.argmax(y_pred_prob, axis=1)
        f1 = f1_score(self.y_valid, y_pred, average="macro")
        self.f1_scores.append(f1)
        if f1 > self.best_f1:
            self.best_f1 = f1
            self.best_iter = epoch
            self.wait = 0
        else:
            self.wait += 1
        if self.wait >= self.patience:
            return True  # stop training
        return False  # continue training


if __name__ == "__main__":

    # Load environment variables from .env file
    # This ensures sensitive information like API keys is securely loaded into the environment.
    load_dotenv()

    local_log = PrintLog(extra_name="_train")

    df_part1A_X = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part1A_X.csv")
    df_part1A_Y = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part1A_Y.csv")
    df_part1B_X = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part1B_X.csv")
    df_part1B_Y = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part1B_Y.csv")
    df_part2A_X = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part2A_X.csv")
    df_part2A_Y = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part2A_Y.csv")
    df_part2B_X = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part2B_X.csv")
    df_part2B_Y = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part2B_Y.csv")
    df_part3A_X = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part3A_X.csv")
    df_part3A_Y = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part3A_Y.csv")
    df_part3B_X = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part3B_X.csv")
    df_part3B_Y = pd.read_csv(config.DATA_DIR / f"{config.TRADE_END_DATE}_part3B_Y.csv")

    df_part1A_X.drop(columns=["index", "open", "stock_name"], inplace=True)
    df_part1B_X.drop(columns=["index", "open", "stock_name"], inplace=True)
    df_part2A_X.drop(columns=["index", "open", "stock_name"], inplace=True)
    df_part2B_X.drop(columns=["index", "open", "stock_name"], inplace=True)
    df_part3A_X.drop(columns=["index", "open", "stock_name"], inplace=True)
    df_part3B_X.drop(columns=["index", "open", "stock_name"], inplace=True)

    n_estimators = 1000  # Number of iterations for XGBoost model training

    df_part1A_X, df_part1A_Y = shuffle(df_part1A_X, df_part1A_Y, random_state=42)
    df_part1B_X, df_part1B_Y = shuffle(df_part1B_X, df_part1B_Y, random_state=42)
    df_part2A_X, df_part2A_Y = shuffle(df_part2A_X, df_part2A_Y, random_state=42)
    df_part2B_X, df_part2B_Y = shuffle(df_part2B_X, df_part2B_Y, random_state=42)
    df_part3A_X, df_part3A_Y = shuffle(df_part3A_X, df_part3A_Y, random_state=42)
    df_part3B_X, df_part3B_Y = shuffle(df_part3B_X, df_part3B_Y, random_state=42)

    df_partA_X = pd.concat([df_part1A_X, df_part2A_X, df_part3A_X])
    df_partA_Y = pd.concat([df_part1A_Y, df_part2A_Y, df_part3A_Y])
    df_partB_X = pd.concat([df_part1B_X, df_part2B_X, df_part3B_X])
    df_partB_Y = pd.concat([df_part1B_Y, df_part2B_Y, df_part3B_Y])

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

    params = {
        "tree_method": "hist",
        "objective": "multi:softprob",
        "num_class": len(np.unique(df_part1A_Y)),
        "max_depth": 10,
        "learning_rate": 0.015,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "gamma": 5,
        "min_child_weight": 3,
        "reg_alpha": 0.3,
        "reg_lambda": 5,
        "eval_metric": "mlogloss",
    }

    # Define the search grid with more parameters
    param_grid = config.XGBOOST_HYPERPARAM_GRID_SEARCH.copy()

    # Generate all combinations of the grid (beware, this can become very long if the grid is large)
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
            param_grid["top_features"],
        )
    )

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
        top_features,
    ) in grid:

        # model A
        dtestA = xgb.DMatrix(df_partA_X, label=df_partA_Y.values.ravel())
        dtestB = xgb.DMatrix(df_partB_X, label=df_partB_Y.values.ravel())

        params_grid = params.copy()
        params_grid["max_depth"] = max_depth
        params_grid["learning_rate"] = learning_rate
        params_grid["subsample"] = subsample
        params_grid["colsample_bytree"] = colsample_bytree
        params_grid["gamma"] = gamma
        params_grid["min_child_weight"] = min_child_weight
        params_grid["reg_alpha"] = reg_alpha
        params_grid["reg_lambda"] = reg_lambda

        f1_callback = EvalF1Callback(dtestB, df_partB_Y, patience=patience)

        # print("Training model on interval 1")
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
        # print("Training model on interval 2")
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
        # print("Training model on interval 3")
        best_model3 = xgb.train(
            params_grid,
            dtrain3A,
            num_boost_round=n_estimators,
            evals=[(dtrain3A, "train"), (dtestB, "eval")],
            callbacks=[f1_callback],
            verbose_eval=False,
        )
        importance3 = best_model3.get_score(importance_type="weight")

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

        # Construction of the importance DataFrame
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

        # Replace NaN with 0 (safety)
        importance_df = importance_df.fillna(0)
        # Normalize each column by its max (except 'feature' column)
        cols = [
            "interval_1",
            "interval_2",
            "interval_3",
            "interval_4",
            "interval_5",
            "interval_6",
        ]
        importance_df[cols] = importance_df[cols] / importance_df[cols].max()

        # This line will filter out features that have zero importance in any interval
        importance_df = importance_df[(importance_df[cols] != 0).all(axis=1)]
        # compute mean, std and std/mean
        importance_df["mean"] = importance_df[cols].mean(axis=1)
        importance_df["std"] = importance_df[cols].std(axis=1)
        importance_df["std/mean"] = importance_df["std"] / importance_df["mean"]
        importance_df_sorted_by_std_mean = importance_df.sort_values(
            by="std/mean", ascending=True
        )
        # Save the sorted importance dataframe to a CSV file
        importance_df_sorted_by_std_mean.to_csv(
            os.path.join(
                local_log.output_dir_time, "importance_df_sorted_by_std_mean.csv"
            ),
            index=False,
        )
        # Filter features based on std/mean threshold and keep those with low std/mean between intervals
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
        with local_log:
            print(f"F1 Score A: {f1A}")
            print(f"F1 Score B: {f1B}")

        # Save models and selected features
        selected_features_path = os.path.join(
            local_log.output_dir_time, "selected_features.json"
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

    local_log.copy_last()

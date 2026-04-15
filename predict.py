import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

PREDICT_FEATURES = [
    "volume",
    "return_1d", "return_5d", "return_20d",
    "vol_10d", "vol_20d",
    "sma_10", "sma_20",
    "dist_sma10", "dist_sma20",
    "rsi_14", "volume_zscore", "hl_range",
    "lag_1", "lag_2", "lag_3",
    "day_of_week", "month", "volume_trend"
]

QUANTILES = [0.1, 0.5, 0.9]


def train_prediction_models(train: pd.DataFrame, val: pd.DataFrame):
    """
    Trains 3 LightGBM models — one per quantile.
    Target is now next day return (%), not raw price.
    """
    X_train = train[PREDICT_FEATURES]
    y_train = train["target_return"].values

    X_val   = val[PREDICT_FEATURES]
    y_val   = val["target_return"].values

    # fit scaler on train 
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=PREDICT_FEATURES,
        index=train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=PREDICT_FEATURES,
        index=val.index
    )

    models = {}

    for q in QUANTILES:
        print(f"  Training quantile {q}...")
        model = LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=200,
            learning_rate=0.03,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            random_state=42,
            verbose=-1
        )
        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
        )
        models[q] = model

    return scaler, models


def predict(df: pd.DataFrame, scaler: StandardScaler, models: dict) -> pd.DataFrame:
    """
    Returns predicted returns (lower, median, upper)
    plus the actual return for comparison.
    """
    X = df[PREDICT_FEATURES]
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=PREDICT_FEATURES,
        index=df.index
    )

    results = pd.DataFrame(index=df.index)
    if "target_return" in df.columns:
        results["actual_return"] = df["target_return"].values
    else:
        results["actual_return"] = None
    results["lower"]         = models[0.1].predict(X_scaled)
    results["median"]        = models[0.5].predict(X_scaled)
    results["upper"]         = models[0.9].predict(X_scaled)

    # fix quantile crossing
    results["lower"] = results[["lower", "median"]].min(axis=1)
    results["upper"] = results[["upper", "median"]].max(axis=1)

    return results


def returns_to_prices(results: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts predicted returns back to price predictions.
    today_close × (1 + predicted_return) = predicted price
    """
    today_close = raw_df["Close"].reindex(results.index)

    prices = pd.DataFrame(index=results.index)
    prices["actual"] = today_close * (1 + results["actual_return"])
    prices["lower"]  = today_close * (1 + results["lower"])
    prices["median"] = today_close * (1 + results["median"])
    prices["upper"]  = today_close * (1 + results["upper"])

    return prices


def evaluate(results: pd.DataFrame, raw_df: pd.DataFrame, split_name: str):
    """
    Converts returns to prices then evaluates MAE, RMSE, coverage.
    """
    prices = returns_to_prices(results, raw_df)

    mae  = mean_absolute_error(prices["actual"], prices["median"])
    rmse = mean_squared_error(prices["actual"],  prices["median"]) ** 0.5

    inside = (
        (prices["actual"] >= prices["lower"]) &
        (prices["actual"] <= prices["upper"])
    )
    coverage = inside.mean() * 100

    print(f"  {split_name} — MAE: ${mae:.2f}  RMSE: ${rmse:.2f}  Coverage: {coverage:.1f}%")


def save_prediction_models(scaler, models, save_dir="models/prediction"):
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler, f"{save_dir}/scaler.pkl")
    for q, model in models.items():
        joblib.dump(model, f"{save_dir}/lgbm_q{int(q*100)}.pkl")
    print(f"  Saved prediction models to {save_dir}/")


def load_prediction_models(save_dir="models/prediction"):
    scaler = joblib.load(f"{save_dir}/scaler.pkl")
    models = {}
    for q in QUANTILES:
        models[q] = joblib.load(f"{save_dir}/lgbm_q{int(q*100)}.pkl")
    return scaler, models
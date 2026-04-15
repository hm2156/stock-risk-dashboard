import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# features fed into the prediction model
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
    Trains 3 separate LightGBM models,one per quantile.
    """
    X_train = train[PREDICT_FEATURES]
    y_train = train["target_close"].values
    X_val   = val[PREDICT_FEATURES]
    y_val   = val["target_close"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)

    models = {}

    for q in QUANTILES:
        print(f"Training quantile {q}...")

        model = LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=200,
            learning_rate=0.03,    # was 0.05 — slower learning = more generalization
            num_leaves=15,
            min_child_samples=30,  # was 20 — even stricter
            subsample=0.7,         # was 0.8 — see even less data per tree
            colsample_bytree=0.7,  # was 0.8
            reg_lambda=2.0,        # was 1.0 — stronger penalty
            random_state=42,
            verbose=-1
        )

        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
        )

        models[q] = model
        print(f"  Done.")

    return scaler, models


def predict(df: pd.DataFrame, scaler: StandardScaler, models: dict) -> pd.DataFrame:
    X        = df[PREDICT_FEATURES]  # keep as DataFrame, not numpy array
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=PREDICT_FEATURES,    # preserve column names through the scaler
        index=df.index
    )

    results = pd.DataFrame(index=df.index)
    results["actual"] = df["target_close"].values
    results["lower"]  = models[0.1].predict(X_scaled)
    results["median"] = models[0.5].predict(X_scaled)
    results["upper"]  = models[0.9].predict(X_scaled)

    # fix quantile crossing
    results["lower"] = results[["lower", "median"]].min(axis=1)
    results["upper"] = results[["upper", "median"]].max(axis=1)

    return results

def evaluate(results: pd.DataFrame, split_name: str):
    """
    Prints MAE and RMSE for the median prediction.
    Also prints how often the actual price fell inside the predicted range.
    """
    mae  = mean_absolute_error(results["actual"], results["median"])
    rmse = mean_squared_error(results["actual"],  results["median"]) ** 0.5

    # coverage — what % of actual prices landed inside our lower/upper band
    inside = (
        (results["actual"] >= results["lower"]) &
        (results["actual"] <= results["upper"])
    )
    coverage = inside.mean() * 100

    print(f"\n{split_name} results:")
    print(f"  MAE:      ${mae:.2f}")
    print(f"  RMSE:     ${rmse:.2f}")
    print(f"  Coverage: {coverage:.1f}%  (actual price inside predicted range)")


def save_prediction_models(scaler, models, save_dir="models/prediction"):
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler, f"{save_dir}/scaler.pkl")
    for q, model in models.items():
        joblib.dump(model, f"{save_dir}/lgbm_q{int(q*100)}.pkl")
    print(f"\nSaved models to {save_dir}/")


def load_prediction_models(save_dir="models/prediction"):
    scaler = joblib.load(f"{save_dir}/scaler.pkl")
    models = {}
    for q in QUANTILES:
        models[q] = joblib.load(f"{save_dir}/lgbm_q{int(q*100)}.pkl")
    return scaler, models
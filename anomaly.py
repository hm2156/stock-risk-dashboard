import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

# features fed into the anomaly detector
ANOMALY_FEATURES = [
    "return_1d", "return_5d", "return_20d",
    "vol_10d", "vol_20d",
    "dist_sma10", "dist_sma20",
    "rsi_14", "volume_zscore", "hl_range"
]

def train_anomaly_model(train: pd.DataFrame, contamination: float = 0.05):
    """
    Returns the fitted scaler and model.
    """
    X_train = train[ANOMALY_FEATURES].values

    # fit scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # fit isolation forest
    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    model.fit(X_train_scaled)

    return scaler, model


def score_anomalies(df: pd.DataFrame, scaler: StandardScaler, model: IsolationForest) -> pd.Series:
    """
    Scores any split (train, val, or test) using the already-fitted
    scaler and model. Returns a risk score between 0 and 100.
    Higher = more anomalous = higher risk.
    """
    X = df[ANOMALY_FEATURES].values
    X_scaled = scaler.transform(X)  # transform

    # raw scores are negative
    raw_scores = model.score_samples(X_scaled)

    # flip and normalize to 0-100
    flipped = -raw_scores
    min_s, max_s = flipped.min(), flipped.max()
    normalized = (flipped - min_s) / (max_s - min_s) * 100

    return pd.Series(normalized, index=df.index, name="risk_score")


def save_anomaly_model(scaler, model, save_dir="models/anomaly"):
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(scaler, f"{save_dir}/scaler.pkl")
    joblib.dump(model,  f"{save_dir}/isolation_forest.pkl")
    print(f"Saved scaler and model to {save_dir}/")


def load_anomaly_model(save_dir="models/anomaly"):
    scaler = joblib.load(f"{save_dir}/scaler.pkl")
    model  = joblib.load(f"{save_dir}/isolation_forest.pkl")
    return scaler, model
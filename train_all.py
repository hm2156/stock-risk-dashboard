import pandas as pd
import os
from features import add_features
from split import time_split
from anomaly import train_anomaly_model, score_anomalies, save_anomaly_model
from predict import train_prediction_models, predict, evaluate, save_prediction_models

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "BAC", "GS",
    "XOM", "CVX", "JNJ", "PFE", "WMT",
    "COST", "SPY", "QQQ", "XLF", "XLE"
]

def train_ticker(ticker: str):
    print(f"\n{'='*40}")
    print(f"  {ticker}")
    print(f"{'='*40}")

    # load raw data
    path = f"data/raw/{ticker}.parquet"
    if not os.path.exists(path):
        print(f"  Skipping — no data file found at {path}")
        return None

    df   = pd.read_parquet(path)        # keep raw df in scope
    feat = add_features(df)
    train, val, test = time_split(feat)

    print(f"  Rows — train: {len(train)}, val: {len(val)}, test: {len(test)}")

    # anomaly model
    print(f"  Training anomaly model...")
    scaler_a, model_a = train_anomaly_model(train)
    save_anomaly_model(scaler_a, model_a, save_dir=f"models/anomaly/{ticker}")

    # prediction models
    print(f"  Training prediction models...")
    scaler_p, models_p = train_prediction_models(train, val)
    save_prediction_models(scaler_p, models_p, save_dir=f"models/prediction/{ticker}")

    # evaluate — pass raw df for price conversion
    val_results = predict(val, scaler_p, models_p)
    evaluate(val_results, df, f"Val ({ticker})")

    return val_results

if __name__ == "__main__":
    results = {}

    for ticker in TICKERS:
        results[ticker] = train_ticker(ticker)

    print(f"\n{'='*40}")
    print("  All tickers done")
    print(f"{'='*40}")
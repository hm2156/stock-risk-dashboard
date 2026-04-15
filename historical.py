import pandas as pd
import numpy as np
from features import add_features
from split import time_split
from anomaly import load_anomaly_model, score_anomalies
from predict import load_prediction_models, predict, returns_to_prices
import os

def get_historical_scores(ticker: str) -> pd.DataFrame:
    """
    Returns a dataframe with daily risk scores and price predictions
    for the full val + test period of a given ticker.
    This is used to power the historical charts on the stock detail page.
    """
    path = f"data/raw/{ticker}.parquet"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data for {ticker}")

    raw  = pd.read_parquet(path)
    feat = add_features(raw)
    train, val, test = time_split(feat)

    # load trained models
    scaler_a, model_a  = load_anomaly_model(save_dir=f"models/anomaly/{ticker}")
    scaler_p, models_p = load_prediction_models(save_dir=f"models/prediction/{ticker}")

    # score val + test combined
    val_test = pd.concat([val, test])

    # risk scores
    risk_scores = score_anomalies(val_test, scaler_a, model_a)

    # price predictions
    pred_results  = predict(val_test, scaler_p, models_p)
    price_preds   = returns_to_prices(pred_results, raw)

    # combine everything
    result = pd.DataFrame(index=val_test.index)
    result["close"]       = val_test["close"]
    result["risk_score"]  = risk_scores
    result["risk_label"]  = result["risk_score"].apply(get_risk_label)
    result["pred_lower"]  = price_preds["lower"].round(2)
    result["pred_median"] = price_preds["median"].round(2)
    result["pred_upper"]  = price_preds["upper"].round(2)
    result["actual"]      = price_preds["actual"].round(2)
    result["inside_band"] = (
        (result["actual"] >= result["pred_lower"]) &
        (result["actual"] <= result["pred_upper"])
    )

    return result


def get_risk_label(score: float) -> str:
    if score >= 75:
        return "HIGH"
    elif score >= 40:
        return "ELEVATED"
    else:
        return "NORMAL"


def get_top_anomaly_dates(ticker: str, n: int = 5) -> pd.DataFrame:
    """
    Returns the top N riskiest historical dates for a ticker
    with their risk scores and what the price did that day.
    """
    hist = get_historical_scores(ticker)
    top  = hist.nlargest(n, "risk_score")[["risk_score", "close", "actual", "risk_label"]]
    top.index = top.index.date
    return top


def get_all_historical(tickers: list) -> dict:
    """
    Loads historical scores for all tickers.
    Returns a dict of {ticker: dataframe}.
    """
    results = {}
    for ticker in tickers:
        try:
            results[ticker] = get_historical_scores(ticker)
        except Exception as e:
            print(f"Error loading {ticker}: {e}")
    return results
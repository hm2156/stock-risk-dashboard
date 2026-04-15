import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from features import add_features
from anomaly import load_anomaly_model, score_anomalies, ANOMALY_FEATURES
from predict import load_prediction_models, predict, returns_to_prices
def get_live_features(ticker: str) -> tuple:
    end   = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=120)).strftime("%Y-%m-%d")

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    raw.columns = [col[0] if isinstance(col, tuple) else col for col in raw.columns]
    raw = raw[["Open", "High", "Low", "Close", "Volume"]]

    feat = add_features(raw, live=True)

    print(f"  Feature date: {feat.index[0].date()}")
    print(f"  Raw data latest: {raw.index[-1].date()}")

    return feat, raw


def predict_today(ticker: str) -> dict:
    """
    Returns today's risk score and tomorrow's predicted
    price range for a given ticker.
    """
    print(f"Fetching live data for {ticker}...")

    today_feat, raw = get_live_features(ticker)
    today_date = today_feat.index[0].date()

    scaler_a, model_a = load_anomaly_model(save_dir=f"models/anomaly/{ticker}")
    scaler_p, models_p = load_prediction_models(save_dir=f"models/prediction/{ticker}")

    risk_scores = score_anomalies(today_feat, scaler_a, model_a)
    risk_score  = round(risk_scores.iloc[0], 1)

    pred_results = predict(today_feat, scaler_p, models_p)

    price_preds = returns_to_prices(pred_results, raw)

    today_close = raw["Close"].iloc[-1]

    result = {
        "ticker":      ticker,
        "as_of_date":  str(today_date),
        "today_close": round(float(today_close), 2),
        "risk_score":  risk_score,
        "risk_label":  get_risk_label(risk_score),
        "pred_lower":  round(float(price_preds["lower"].iloc[0]), 2),
        "pred_median": round(float(price_preds["median"].iloc[0]), 2),
        "pred_upper":  round(float(price_preds["upper"].iloc[0]), 2),
    }

    return result


def get_risk_label(score: float) -> str:
    if score >= 75:
        return "HIGH"
    elif score >= 40:
        return "ELEVATED"
    else:
        return "NORMAL"


def predict_all(tickers: list) -> pd.DataFrame:
    """
    Runs predict_today for all tickers and returns
    a clean summary dataframe.
    """
    rows = []
    for ticker in tickers:
        try:
            result = predict_today(ticker)
            rows.append(result)
        except Exception as e:
            print(f"  Error on {ticker}: {e}")

    return pd.DataFrame(rows)
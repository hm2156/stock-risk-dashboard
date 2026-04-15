import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes raw OHLCV dataframe, returns a new dataframe
    with technical indicator features.
    """
    feat = pd.DataFrame(index=df.index)

    feat["close"] = df["Close"]
    feat["volume"] = df["Volume"]

    # daily return
    feat["return_1d"] = df["Close"].pct_change()

    # rolling returns
    feat["return_5d"]= df["Close"].pct_change(5)
    feat["return_20d"]= df["Close"].pct_change(20)

    # rolling volatility (std of daily returns)
    feat["vol_10d"] = feat["return_1d"].rolling(10).std()
    feat["vol_20d"] = feat["return_1d"].rolling(20).std()

    # simple moving averages
    feat["sma_10"]  = df["Close"].rolling(10).mean()
    feat["sma_20"]  = df["Close"].rolling(20).mean()

    # distance from moving average (normalized)
    feat["dist_sma10"] = (df["Close"] - feat["sma_10"]) / feat["sma_10"]
    feat["dist_sma20"]= (df["Close"] - feat["sma_20"]) / feat["sma_20"]

    # RSI (14-day)
    feat["rsi_14"]= compute_rsi(df["Close"], 14)

    # volume z-score 
    feat["volume_zscore"] = (
        (df["Volume"] - df["Volume"].rolling(20).mean()) /
        df["Volume"].rolling(20).std()
    )

    # high-low range as % of close (intraday volatility)
    feat["hl_range"]= (df["High"] - df["Low"]) / df["Close"]

    feat = feat.shift(1)

    # lag features — last 3 days of raw close
    feat["lag_1"] = df["Close"].shift(1)
    feat["lag_2"] = df["Close"].shift(2)
    feat["lag_3"] = df["Close"].shift(3)

    # calendar features
    feat["day_of_week"] = pd.Series(df.index.dayofweek, index=df.index).astype(float)
    feat["month"]       = pd.Series(df.index.month,     index=df.index).astype(float)

    # volume trend — is volume rising or falling over last 5 days?
    feat["volume_trend"] = (
        df["Volume"].rolling(5).mean() /
        df["Volume"].rolling(20).mean()
    )

    # target: next day's closing price
    feat["target_close"] = df["Close"].shift(-1)

    # drop rows with NaN
    feat = feat.dropna()

    return feat


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))
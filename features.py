import pandas as pd
import numpy as np

def add_features(df: pd.DataFrame, live: bool = False) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    # --- all your existing feature code stays exactly the same ---
    feat["close"]        = df["Close"]
    feat["volume"]       = df["Volume"]
    feat["return_1d"]    = df["Close"].pct_change()
    feat["return_5d"]    = df["Close"].pct_change(5)
    feat["return_20d"]   = df["Close"].pct_change(20)
    feat["vol_10d"]      = feat["return_1d"].rolling(10).std()
    feat["vol_20d"]      = feat["return_1d"].rolling(20).std()
    feat["sma_10"]       = df["Close"].rolling(10).mean()
    feat["sma_20"]       = df["Close"].rolling(20).mean()
    feat["dist_sma10"]   = (df["Close"] - feat["sma_10"]) / feat["sma_10"]
    feat["dist_sma20"]   = (df["Close"] - feat["sma_20"]) / feat["sma_20"]
    feat["rsi_14"]       = compute_rsi(df["Close"], 14)
    feat["volume_zscore"] = (
        (df["Volume"] - df["Volume"].rolling(20).mean()) /
        df["Volume"].rolling(20).std()
    )
    feat["hl_range"]     = (df["High"] - df["Low"]) / df["Close"]
    feat["lag_1"]        = df["Close"].shift(0)
    feat["lag_2"]        = df["Close"].shift(1)
    feat["lag_3"]        = df["Close"].shift(2)
    feat["day_of_week"]  = pd.Series(df.index.dayofweek, index=df.index).astype(float)
    feat["month"]        = pd.Series(df.index.month, index=df.index).astype(float)
    feat["volume_trend"] = (
        df["Volume"].rolling(5).mean() /
        df["Volume"].rolling(20).mean()
    )

    # leakage prevention — shift all features forward by 1
    feat = feat.shift(1)

    # target
    feat["target_return"] = df["Close"].pct_change(-1) * -1

    if live:
        # in live mode — drop NaNs from rolling windows but KEEP the last row
        # even though target_return is NaN (we don't need it for prediction)
        feature_cols = [c for c in feat.columns if c != "target_return"]
        feat = feat.dropna(subset=feature_cols)
        # take only the last row and drop target
        feat = feat.iloc[[-1]].drop(columns=["target_return"])
    else:
        # normal training mode — drop all NaN rows including target
        feat = feat.dropna()

    return feat


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))
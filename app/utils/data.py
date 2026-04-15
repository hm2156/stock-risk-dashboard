import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import pandas as pd
from live import predict_all, predict_today
from historical import get_historical_scores, get_top_anomaly_dates

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "BAC", "GS",
    "XOM", "CVX", "JNJ", "PFE", "WMT",
    "COST", "SPY", "QQQ", "XLF", "XLE"
]

SECTOR_MAP = {
    "AAPL": "Tech",  "MSFT": "Tech",  "GOOGL": "Tech",
    "AMZN": "Tech",  "NVDA": "Tech",  "META": "Tech",
    "TSLA": "Tech",  "JPM": "Finance","BAC": "Finance",
    "GS":   "Finance","XOM": "Energy","CVX": "Energy",
    "JNJ":  "Health","PFE": "Health", "WMT": "Retail",
    "COST": "Retail","SPY": "ETF",    "QQQ": "ETF",
    "XLF":  "ETF",   "XLE": "ETF"
}

MODEL_PERFORMANCE = {
    "AAPL": {"mae": 1.90, "coverage": 75.3},
    "MSFT": {"mae": 3.95, "coverage": 80.1},
    "GOOGL":{"mae": 2.17, "coverage": 71.9},
    "AMZN": {"mae": 1.85, "coverage": 87.7},
    "NVDA": {"mae": 1.60, "coverage": 61.0},
    "META": {"mae": 7.06, "coverage": 78.8},
    "TSLA": {"mae": 4.67, "coverage": 78.8},
    "JPM":  {"mae": 1.58, "coverage": 76.0},
    "BAC":  {"mae": 0.39, "coverage": 72.6},
    "GS":   {"mae": 3.88, "coverage": 69.2},
    "XOM":  {"mae": 1.04, "coverage": 78.8},
    "CVX":  {"mae": 1.26, "coverage": 84.9},
    "JNJ":  {"mae": 1.02, "coverage": 68.5},
    "PFE":  {"mae": 0.30, "coverage": 65.8},
    "WMT":  {"mae": 0.43, "coverage": 67.1},
    "COST": {"mae": 6.73, "coverage": 76.7},
    "SPY":  {"mae": 2.94, "coverage": 66.4},
    "QQQ":  {"mae": 3.29, "coverage": 74.0},
    "XLF":  {"mae": 0.28, "coverage": 78.1},
    "XLE":  {"mae": 0.35, "coverage": 82.2},
}


@st.cache_data(ttl=3600)
def load_live_data() -> pd.DataFrame:
    """
    Fetches live predictions for all 20 tickers.
    Cached for 1 hour — refreshes automatically or on button press.
    """
    df = predict_all(TICKERS)
    df["sector"] = df["ticker"].map(SECTOR_MAP)
    return df


@st.cache_data(ttl=3600)
def load_historical_data(ticker: str) -> pd.DataFrame:
    """
    Loads historical risk scores and predictions for a ticker.
    Cached for 1 hour.
    """
    return get_historical_scores(ticker)


@st.cache_data(ttl=3600)
def load_top_anomalies(ticker: str) -> pd.DataFrame:
    """
    Returns top 5 riskiest days for a ticker.
    """
    return get_top_anomaly_dates(ticker)


@st.cache_data(ttl=3600)
def load_all_historical() -> dict:
    """
    Loads historical data for all tickers.
    Used for portfolio analytics page.
    """
    all_data = {}
    for ticker in TICKERS:
        try:
            all_data[ticker] = get_historical_scores(ticker)
        except Exception as e:
            print(f"Error loading {ticker}: {e}")
    return all_data


def get_risk_color(label: str) -> str:
    colors = {
        "HIGH":     "#E24B4A",
        "ELEVATED": "#EF9F27",
        "NORMAL":   "#1D9E75"
    }
    return colors.get(label, "#888780")


def clear_cache():
    st.cache_data.clear()
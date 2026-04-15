import yfinance as yf
import pandas as pd
import os

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "JPM", "BAC", "GS",
    "XOM", "CVX", "JNJ", "PFE", "WMT",
    "COST", "SPY", "QQQ", "XLF", "XLE"
]

START = "2022-01-01"
END   = "2025-01-01"

def download_data(tickers, start, end, save_dir="data/raw"):
    os.makedirs(save_dir, exist_ok=True)

    for ticker in tickers:
        print(f"Downloading {ticker}...")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True)

        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        df.to_parquet(f"{save_dir}/{ticker}.parquet")
        print(f"  Saved {len(df)} rows")

if __name__ == "__main__":
    download_data(TICKERS, START, END)
    print("\nDone! All tickers saved to data/raw/")
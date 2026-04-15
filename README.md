# AlphaWatch

ML-powered market risk intelligence platform. Monitors 20 major stocks 
in real time using anomaly detection and quantile regression to flag 
unusual behavior and predict tomorrow's closing price range.

**Live demo:** [alphawatch.streamlit.app](https://alphawatch.streamlit.app/)

---

## Models

### Anomaly detection : Isolation Forest

Detect unusual market behavior by identifying days that are statistically   
isolated from normal trading patterns across 21 engineered features.  
Outputs a daily risk score (0–100) per ticker.

- Fitted on training data only - never val or test
- Validated against real market events (earnings, macro shocks, crashes)
- Every flagged date in validation mapped to a real, explainable event

### Price prediction : LightGBM quantile regression

Three separate LightGBM models trained per ticker — one each for the 
10th, 50th, and 90th percentile — outputting a price confidence range 
rather than a single point estimate.

Predicts next-day **return** (not raw price) to handle regime changes 
across different market environments. Converts back to price at inference.


| Metric            | Value                                 |
| ----------------- | ------------------------------------- |
| Avg val coverage  | 74.5% across 20 tickers               |
| Best coverage     | AMZN 87.7%, CVX 84.9%, MSFT 80.1%     |
| Avg val MAE       | ~$2.50                                |
| Quantile crossing | Fixed via post-processing enforcement |


---

## Feature engineering

21 features engineered from raw OHLCV data:


| Category        | Features                                           |
| --------------- | -------------------------------------------------- |
| Returns         | 1d, 5d, 20d price returns                          |
| Volatility      | 10d, 20d rolling std of returns                    |
| Moving averages | SMA 10, SMA 20, distance from each                 |
| Momentum        | RSI (14-day)                                       |
| Volume          | Raw volume, volume z-score, 5d vs 20d volume trend |
| Price lags      | Yesterday, 2 days ago, 3 days ago close            |
| Calendar        | Day of week, month                                 |
| Intraday        | High-low range as % of close                       |


---

## Data leakage prevention

All features shifted by 1 day — model never sees same-day or future data.
Verified with 8 independent checks:

- Zero overlap across train / val / test splits
- Target return matches next-day raw return exactly (5 spot checks)
- Lag features match correct historical closes (3 spot checks)
- Scaler and Isolation Forest fitted on train only

---

## Tech stack


| Layer      | Tools                         |
| ---------- | ----------------------------- |
| Data       | yfinance, pandas, pyarrow     |
| ML         | scikit-learn, LightGBM, numpy |
| Dashboard  | Streamlit, Plotly             |
| Deployment | Docker                        |


---

## Setup

Trained models are already included in the repo.
You only need to run `ingest.py` and `train_all.py` if you want 
to pull fresh data or retrain from scratch.

```bash
git clone https://github.com/hm2156/stock-risk-dashboard
cd stock-risk-dashboard
pip install -r requirements.txt
python ingest.py        # optional , download latest data
python train_all.py     # optional, retrain all models (~3 mins)
streamlit run app/main.py
```


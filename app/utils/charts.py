import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np

FEATURE_LABELS = {
    "volume":        "Trading volume",
    "return_1d":     "1-day price return",
    "return_5d":     "5-day price return",
    "return_20d":    "20-day price return",
    "vol_10d":       "10-day volatility",
    "vol_20d":       "20-day volatility",
    "sma_10":        "10-day moving average",
    "sma_20":        "20-day moving average",
    "dist_sma10":    "Distance from 10-day avg",
    "dist_sma20":    "Distance from 20-day avg",
    "rsi_14":        "RSI (momentum indicator)",
    "volume_zscore": "Volume anomaly score",
    "hl_range":      "Intraday high-low range",
    "lag_1":         "Yesterday's close",
    "lag_2":         "2 days ago close",
    "lag_3":         "3 days ago close",
    "day_of_week":   "Day of week",
    "month":         "Month of year",
    "volume_trend":  "Volume trend (5d vs 20d)",
}

def price_prediction_chart(hist: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Price chart with prediction band overlaid.
    Shows actual close, median prediction, and lower/upper band.
    """
    fig = go.Figure()

    # prediction band (shaded area)
    fig.add_trace(go.Scatter(
        x=hist.index.tolist() + hist.index.tolist()[::-1],
        y=hist["pred_upper"].tolist() + hist["pred_lower"].tolist()[::-1],
        fill="toself",
        fillcolor="rgba(55, 138, 221, 0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="Prediction band",
        hoverinfo="skip"
    ))

    # median prediction line
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist["pred_median"],
        line=dict(color="#378ADD", width=1.5, dash="dash"),
        name="Predicted median"
    ))

    # actual close line
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist["actual"],
        line=dict(color="#1D9E75", width=2),
        name="Actual close"
    ))

    fig.update_layout(
        title=f"{ticker} — price vs prediction",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        height=400,
        margin=dict(l=40, r=20, t=50, b=40)
    )

    return fig


def risk_score_chart(hist: pd.DataFrame, ticker: str) -> go.Figure:
    """
    Risk score history chart with color coded zones.
    """
    fig = go.Figure()

    # add colored zone backgrounds
    fig.add_hrect(y0=0,  y1=40, fillcolor="#1D9E75", opacity=0.08, line_width=0)
    fig.add_hrect(y0=40, y1=75, fillcolor="#EF9F27", opacity=0.08, line_width=0)
    fig.add_hrect(y0=75, y1=100,fillcolor="#E24B4A", opacity=0.08, line_width=0)

    # risk score line
    fig.add_trace(go.Scatter(
        x=hist.index,
        y=hist["risk_score"],
        line=dict(color="#534AB7", width=2),
        fill="tozeroy",
        fillcolor="rgba(83, 74, 183, 0.1)",
        name="Risk score"
    ))

    # threshold lines
    fig.add_hline(y=40, line_dash="dot", line_color="#EF9F27",
                  annotation_text="Elevated", annotation_position="right")
    fig.add_hline(y=75, line_dash="dot", line_color="#E24B4A",
                  annotation_text="High", annotation_position="right")

    fig.update_layout(
        title=f"{ticker} — risk score history",
        xaxis_title="Date",
        yaxis_title="Risk score",
        yaxis=dict(range=[0, 105]),
        height=300,
        margin=dict(l=40, r=60, t=50, b=40)
    )

    return fig


def feature_importance_chart(ticker: str) -> go.Figure:
    """
    Feature importance from the median LightGBM model.
    """
    import joblib
    model = joblib.load(f"models/prediction/{ticker}/lgbm_q50.pkl")
    importance = model.feature_importances_
    features   = model.feature_name_

    df = pd.DataFrame({
        "feature":    features,
        "importance": importance
    }).sort_values("importance", ascending=True).tail(10)

    df["label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])

    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["label"],
        orientation="h",
        marker_color="#534AB7",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.0f}<extra></extra>"
    ))
    fig.update_layout(
        title=f"{ticker} — top 10 feature importance",
        xaxis_title="Importance",
        height=350,
        margin=dict(l=40, r=20, t=50, b=40)
    )

    return fig


def correlation_heatmap(all_historical: dict) -> go.Figure:
    """
    Rolling correlation heatmap across all tickers.
    Uses actual closing prices.
    """
    closes = pd.DataFrame({
        ticker: hist["actual"]
        for ticker, hist in all_historical.items()
    })

    corr = closes.corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdBu",
        zmid=0,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=9),
        hovertemplate="<b>%{x} vs %{y}</b><br>Correlation: %{z}<extra></extra>"
    ))

    fig.update_layout(
        title="Portfolio correlation heatmap",
        height=550,
        margin=dict(l=40, r=20, t=50, b=40)
    )

    return fig


def sector_risk_chart(live_df: pd.DataFrame) -> go.Figure:
    """
    Average risk score by sector bar chart.
    """
    sector_risk = (
        live_df.groupby("sector")["risk_score"]
        .mean()
        .round(1)
        .sort_values(ascending=False)
        .reset_index()
    )

    colors = [
        "#E24B4A" if r >= 75 else "#EF9F27" if r >= 40 else "#1D9E75"
        for r in sector_risk["risk_score"]
    ]

    fig = go.Figure(go.Bar(
        x=sector_risk["sector"],
        y=sector_risk["risk_score"],
        marker_color=colors,
        text=sector_risk["risk_score"],
        textposition="outside"
    ))

    fig.update_layout(
        title="Average risk score by sector",
        yaxis=dict(range=[0, 105]),
        height=350,
        margin=dict(l=40, r=20, t=50, b=40)
    )

    return fig


def anomaly_timeline_chart(all_historical: dict) -> go.Figure:
    """
    Shows which tickers have been flagged HIGH risk most
    frequently over the historical period.
    """
    counts = {}
    for ticker, hist in all_historical.items():
        counts[ticker] = (hist["risk_label"] == "HIGH").sum()

    df = pd.DataFrame({
        "ticker": list(counts.keys()),
        "high_risk_days": list(counts.values())
    }).sort_values("high_risk_days", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["high_risk_days"],
        y=df["ticker"],
        orientation="h",
        marker_color="#E24B4A"
    ))

    fig.update_layout(
        title="Historical high risk days per ticker",
        xaxis_title="Number of HIGH risk days",
        height=500,
        margin=dict(l=60, r=20, t=50, b=40)
    )

    return fig


def model_performance_chart(performance: dict) -> go.Figure:
    """
    MAE and coverage per ticker side by side.
    """
    tickers   = list(performance.keys())
    mae       = [performance[t]["mae"] for t in tickers]
    coverage  = [performance[t]["coverage"] for t in tickers]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=tickers,
        y=mae,
        name="MAE ($)",
        marker_color="#378ADD",
        opacity=0.8
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=tickers,
        y=coverage,
        name="Coverage (%)",
        line=dict(color="#1D9E75", width=2),
        mode="lines+markers"
    ), secondary_y=True)

    fig.add_hline(
        y=80, line_dash="dot",
        line_color="#1D9E75",
        annotation_text="80% target",
        secondary_y=True
    )

    fig.update_layout(
        title="Model performance — MAE and coverage by ticker",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", y=-0.2)
    )

    fig.update_yaxes(title_text="MAE ($)", secondary_y=False)
    fig.update_yaxes(title_text="Coverage (%)", secondary_y=True)

    return fig
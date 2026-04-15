import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
from app.utils.data import load_all_historical, load_live_data, MODEL_PERFORMANCE
from app.utils.charts import (
    correlation_heatmap,
    sector_risk_chart,
    anomaly_timeline_chart,
    model_performance_chart
)


def render():
    st.title("AlphaWatch Portfolio analytics")
    with st.expander("How to read this page"):
        st.markdown("""
        **Sector risk today**
        Average risk score across all tickers in each sector.
        A high bar means multiple stocks in that sector are showing unusual behavior simultaneously —
        which could indicate a sector-wide event (regulation, macro, earnings season).

        **Portfolio correlation heatmap**
        Shows how closely each pair of stocks moves together historically.
        - **+1.0 (dark red)** : move in perfect lockstep
        - **0.0 (white)** : no relationship
        - **-1.0 (dark blue)** : move in opposite directions
        High correlation across your whole portfolio means less diversification —
        if one stock drops, correlated ones likely drop too.

        **Historical anomaly frequency**
        How many days each ticker was flagged as HIGH risk over the historical period.
        Stocks with many HIGH risk days are inherently more volatile and unpredictable.
        This is useful for understanding which positions carry more uncertainty.

        **Model performance by ticker**
        - Blue bars: MAE (mean absolute error) : lower is better
        - Green line: coverage — higher is better, target is 80%
        Tickers where coverage is well below 80% had regime changes
        (e.g. NVDA's AI-driven rally) that were hard to predict from prior data.
        """)
    st.caption("Historical risk analysis, correlations, and model performance across all 20 tickers")

    with st.spinner("Loading portfolio data..."):
        all_hist = load_all_historical()
        live_df  = load_live_data()

    # --- sector risk ---
    st.subheader("Sector risk today")
    st.caption("Average anomaly risk score per sector based on today's live data. Higher = more unusual collective behavior.")
    st.plotly_chart(sector_risk_chart(live_df), use_container_width=True)

    st.divider()

    # --- correlation heatmap ---
    st.subheader("Portfolio correlation")
    st.caption("Pearson correlation of daily closing prices across the val + test period (Oct 2023 – Dec 2024). Helps identify diversification gaps.")
    st.plotly_chart(correlation_heatmap(all_hist), use_container_width=True)

    st.divider()

    # --- anomaly timeline ---
    st.subheader("Historical anomaly frequency")
    st.caption("Number of days each ticker was flagged HIGH risk by the Isolation Forest model. More days = more historically volatile stock.")
    st.plotly_chart(anomaly_timeline_chart(all_hist), use_container_width=True)

    st.divider()

    # --- model performance ---
    st.subheader("Model performance by ticker")
    st.caption("MAE = average dollar error on median prediction. Coverage = % of actual prices inside the 10th–90th percentile band. Target coverage is 80%.")
    st.plotly_chart(model_performance_chart(MODEL_PERFORMANCE), use_container_width=True)
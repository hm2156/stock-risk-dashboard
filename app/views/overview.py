import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
import pandas as pd
from app.utils.data import load_live_data, get_risk_color, clear_cache, SECTOR_MAP


def render():
    st.title("AlphaWatch overview")
    st.caption("Live risk scores and next-day price predictions across 20 stocks")

    with st.expander("How to read this dashboard"):
        st.markdown("""
        **What is this?**
        This dashboard monitors 20 major stocks in real time using machine learning.
        It gives you two things for each stock:

        **1. Risk score (0–100)**
        Calculated by an anomaly detection model (Isolation Forest) that watches for
        unusual behavior like volume spikes, volatility surges, price momentum extremes.
        - **0–40 NORMAL** : stock is behaving as expected
        - **40–75 ELEVATED** : some unusual activity detected, worth watching
        - **75–100 HIGH** : significant anomaly detected, similar to past market events

        **2. Tomorrow's predicted close (10th · 50th · 90th percentile)**
        A machine learning model (LightGBM) predicts tomorrow's closing price as a range:
        - **LOW** : there's a 10% chance the price closes below this
        - **MID** : the median estimate, the model's best single guess
        - **HIGH** : there's a 10% chance the price closes above this
        - A wider range means more uncertainty. A narrow range means more confidence.

        **Important:** These are statistical predictions, not financial advice.
        The model was validated on historical data with ~74% average coverage.
        """)

    # refresh button
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("Refresh"):
            clear_cache()
            st.session_state["data_loaded"] = False
            st.rerun()

    # load data with proper loading state
    if not st.session_state.get("data_loaded"):
        with st.spinner(""):
            placeholder = st.empty()
            placeholder.markdown(
                """
                <div style='text-align:center; padding: 3rem;'>
                    <h3>Loading live market data...</h3>
                    <p style='color:gray'>Fetching prices and computing risk scores for 20 tickers</p>
                    <p style='color:gray; font-size:13px'>This takes about 20-30 seconds on first load</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            df = load_live_data()
            st.session_state["live_df"] = df
            st.session_state["data_loaded"] = True
            placeholder.empty()
    else:
        df = st.session_state["live_df"]

    # --- summary metrics ---
    st.subheader("Today's summary")
    m1, m2, m3, m4 = st.columns(4)

    high_count     = (df["risk_label"] == "HIGH").sum()
    elevated_count = (df["risk_label"] == "ELEVATED").sum()
    avg_risk       = df["risk_score"].mean()
    avg_coverage   = 74.5

    m1.metric("Total tickers", len(df))
    m1.caption("Stocks being monitored")
    m2.metric("High risk today", int(high_count))
    m2.caption("Anomalous behavior detected")
    m3.metric("Elevated risk", int(elevated_count))
    m3.caption("Worth keeping an eye on")
    m4.metric("Avg model coverage", f"{avg_coverage}%")
    m4.caption("% of actual prices inside predicted band")

    st.divider()

    # --- filter controls ---
    st.subheader("All tickers")
    col1, col2 = st.columns([2, 2])

    with col1:
        sector_filter = st.selectbox(
            "Filter by sector",
            ["All"] + sorted(df["sector"].unique().tolist())
        )
    with col2:
        risk_filter = st.selectbox(
            "Filter by risk",
            ["All", "HIGH", "ELEVATED", "NORMAL"]
        )

    # apply filters
    filtered = df.copy()
    if sector_filter != "All":
        filtered = filtered[filtered["sector"] == sector_filter]
    if risk_filter != "All":
        filtered = filtered[filtered["risk_label"] == risk_filter]

    # sort by risk score
    filtered = filtered.sort_values("risk_score", ascending=False)

    # --- column headers ---
    st.divider()
    h1, h2, h3, h4, h5, h6 = st.columns([1, 1, 1, 2, 3, 1])
    h1.markdown("**Ticker**")
    h2.markdown("**Price**")
    h3.markdown("**Sector**")
    h4.markdown("**Risk score**")
    h5.markdown("**Tomorrow's predicted close** *(10th · 50th · 90th percentile)*")
    h6.markdown("")
    st.divider()

    for _, row in filtered.iterrows():
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 2, 3, 1])

        risk_color = get_risk_color(row["risk_label"])

        col1.markdown(f"**{row['ticker']}**")
        col2.markdown(f"${row['today_close']:,.2f}")
        col3.markdown(
            f"<span style='color:var(--text-color); opacity:0.6'>{row['sector']}</span>",
            unsafe_allow_html=True
        )
        col4.markdown(
            f"<span style='color:{risk_color}; font-weight:600'>"
            f"{row['risk_score']:.1f}</span>"
            f"<span style='color:{risk_color}; font-size:12px'> /100 · {row['risk_label']}</span>",
            unsafe_allow_html=True
        )
        col5.markdown(
            f"<div style='line-height:1.8'>"
            f"<span style='color:#888; font-size:11px'>LOW &nbsp;</span>"
            f"<span style='color:#888; font-size:13px'>${row['pred_lower']:,.2f}</span>"
            f"<span style='color:#888'> &nbsp;·&nbsp; </span>"
            f"<span style='color:#888; font-size:11px'>MID &nbsp;</span>"
            f"<span style='font-weight:700; font-size:14px'>${row['pred_median']:,.2f}</span>"
            f"<span style='color:#888'> &nbsp;·&nbsp; </span>"
            f"<span style='color:#888; font-size:11px'>HIGH &nbsp;</span>"
            f"<span style='color:#888; font-size:13px'>${row['pred_upper']:,.2f}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

        if col6.button("View →", key=row['ticker']):
            st.session_state["selected_ticker"] = row["ticker"]
            st.info(f"Go to the **Stock detail** tab to see {row['ticker']} details.")

    st.divider()
    st.caption(
        "Prices from Yahoo Finance · "
        "Tomorrow's range shows 10th / 50th / 90th percentile predictions · "
        "Risk score from Isolation Forest anomaly detection"
    )
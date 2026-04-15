import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

import streamlit as st
from app.utils.data import load_historical_data, load_top_anomalies, MODEL_PERFORMANCE
from app.utils.charts import (
    price_prediction_chart,
    risk_score_chart,
    feature_importance_chart
)


def render(ticker: str):
    st.markdown(
        """
        <script>
            window.parent.document.querySelector('section.main').scrollTo(0, 0);
        </script>
        """,
        unsafe_allow_html=True
    )
    st.title(f"{ticker} — stock detail")

    with st.expander(" How to read this page"):
        st.markdown("""
        **Price vs prediction band chart**
        The green line is the actual historical closing price.
        The blue dashed line is what the model predicted.
        The shaded blue area is the confidence band (10th to 90th percentile).
        When the green line stays inside the shaded area, the prediction was correct.
        Our model achieves this ~74% of the time on average.

        **Risk score history chart**
        Shows how anomalous the stock's behavior was each day.
        Spikes correspond to real market events likeearnings, macro shocks, news.
        - Green zone (0–40): normal behavior
        -  Amber zone (40–75): elevated,  something unusual
        -  Red zone (75–100): high,  significant anomaly

        **Feature importance chart**
        Shows which signals the model relied on most to make its predictions.
        Higher bars = more influential in determining tomorrow's price.

        **Top 5 riskiest days**
        The historical dates where the anomaly model flagged the most unusual behavior.
        These typically correspond to earnings releases, macro events, or sector news.

        **Model performance**
        - MAE = average dollar error on the median prediction
        - Coverage = % of actual prices that landed inside the predicted band
        """)
    # load data
    with st.spinner(f"Loading {ticker} data..."):
        hist      = load_historical_data(ticker)
        anomalies = load_top_anomalies(ticker)
        perf      = MODEL_PERFORMANCE.get(ticker, {})

    # --- prediction card ---
    latest = hist.iloc[-1]
    st.subheader("Latest prediction")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last close",      f"${latest['actual']:,.2f}")
    c1.caption("Most recent historical close")
    c2.metric("Predicted low",   f"${latest['pred_lower']:,.2f}")
    c2.caption("10th percentile — floor estimate")
    c3.metric("Predicted median",f"${latest['pred_median']:,.2f}")
    c3.caption("50th percentile — best single guess")
    c4.metric("Predicted high",  f"${latest['pred_upper']:,.2f}")
    c4.caption("90th percentile — ceiling estimate")

    st.divider()

    # --- charts ---
    st.subheader("Price vs prediction band")
    st.plotly_chart(price_prediction_chart(hist, ticker), use_container_width=True)

    st.subheader("Risk score history")
    st.plotly_chart(risk_score_chart(hist, ticker), use_container_width=True)

    st.subheader("Feature importance")
    st.plotly_chart(feature_importance_chart(ticker), use_container_width=True)

    st.divider()

    # --- top anomaly dates ---
    st.subheader("Top 5 riskiest historical days")
    st.dataframe(
        anomalies.rename(columns={
            "risk_score": "Risk score",
            "close":      "Open price",
            "actual":     "Close price",
            "risk_label": "Risk label"
        }),
        use_container_width=True
    )

    st.divider()

    # --- model performance ---
    st.subheader("Model performance")
    p1, p2 = st.columns(2)
    p1.metric("Val MAE",      f"${perf.get('mae', 0):.2f}")
    p2.metric("Val coverage", f"{perf.get('coverage', 0):.1f}%")
    st.caption("MAE = average dollar error on median prediction. Coverage = % of actual prices inside predicted band.")
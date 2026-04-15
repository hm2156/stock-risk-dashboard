import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st

st.set_page_config(
    page_title="AlphaWatch",
    layout="wide",
    initial_sidebar_state="expanded"
)

from app.views.overview import render as render_overview
from app.views.detail import render as render_detail
from app.views.analytics import render as render_analytics
from app.utils.data import TICKERS

PAGES = [
    "Portfolio overview",
    "Stock detail",
    "Portfolio analytics"
]

# initialize session state defaults
if "current_page" not in st.session_state:
    st.session_state["current_page"] = 0   # index into PAGES list
if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = "AAPL"
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False

# --- sidebar ---
with st.sidebar:
    st.markdown("## AlphaWatch")
    st.caption("Live ML-powered stock risk monitoring")
    st.divider()

    selected = st.radio(
        "Navigate to",
        PAGES,
        index=st.session_state["current_page"]
    )

    # update current page index based on radio selection
    st.session_state["current_page"] = PAGES.index(selected)

    st.divider()

    if selected == "Stock detail":
        ticker = st.selectbox(
            "Select ticker",
            TICKERS,
            index=TICKERS.index(st.session_state["selected_ticker"])
        )
        st.session_state["selected_ticker"] = ticker

    st.divider()
    st.caption("Models: LightGBM quantile regression")
    st.caption("Risk: Isolation Forest anomaly detection")
    st.caption("Data: Yahoo Finance (live)")

# --- render current page ---
page = PAGES[st.session_state["current_page"]]

if page == "Portfolio overview":
    render_overview()
elif page == "Stock detail":
    render_detail(st.session_state["selected_ticker"])
elif page == "Portfolio analytics":
    render_analytics()
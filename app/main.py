import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st

st.set_page_config(
    page_title="AlphaWatch",
    # page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

from app.views.overview  import render as render_overview
from app.views.detail    import render as render_detail
from app.views.analytics import render as render_analytics
from app.utils.data import TICKERS

# initialize session state
if "current_page" not in st.session_state:
    st.session_state["current_page"] = 0
if "selected_ticker" not in st.session_state:
    st.session_state["selected_ticker"] = "AAPL"
if "data_loaded" not in st.session_state:
    st.session_state["data_loaded"] = False

# --- header ---
col1, col2 = st.columns([2, 6])
with col1:
    st.markdown("# AlphaWatch")
    st.caption("ML-powered market risk intelligence")

# --- horizontal tab navigation ---
tab1, tab2, tab3 = st.tabs([
    "Portfolio overview",
    "Stock detail",
    "Portfolio analytics"
])

with tab1:
    render_overview()

with tab2:
    # ticker selector inside the tab
    ticker = st.selectbox(
        "Select ticker",
        TICKERS,
        index=TICKERS.index(st.session_state["selected_ticker"]),
        key="tab_ticker_select"
    )
    st.session_state["selected_ticker"] = ticker
    render_detail(ticker)

with tab3:
    render_analytics()
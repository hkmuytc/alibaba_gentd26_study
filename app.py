import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard.pages import (
    init_session_state,
    render_selected_page,
    render_sidebar,
)
from src.dashboard.theme import apply_dashboard_theme

PAGE_TITLE = "Power Demand Forecasting of Data Center (Alibaba Cluster)"
def main():
    st.set_page_config(
        page_title=PAGE_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_dashboard_theme()

    init_session_state()
    render_selected_page(render_sidebar(PAGE_TITLE))


if __name__ == "__main__":
    main()

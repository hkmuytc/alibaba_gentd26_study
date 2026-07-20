import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PLOTLY_TEMPLATE_NAME = "capstone_soft_blush"


def apply_dashboard_theme() -> None:
    _configure_plotly_defaults()
    _inject_global_styles()


def _configure_plotly_defaults() -> None:
    pio.templates[PLOTLY_TEMPLATE_NAME] = go.layout.Template(
        layout=go.Layout(
            colorway=[
                "#636EFA",
                "#EF553B",
                "#00CC96",
                "#AB63FA",
                "#FFA15A",
                "#19D3F3",
                "#FF6692",
                "#B6E880",
                "#FF97FF",
                "#FECB52",
            ],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#fffdfd",
            font=dict(color="#433537"),
            title=dict(font=dict(color="#433537", size=20)),
            legend=dict(
                bgcolor="rgba(255, 255, 255, 0.9)",
                bordercolor="#e6dcde",
                borderwidth=1,
                font=dict(color="#5d4a4d"),
            ),
            margin=dict(l=24, r=24, t=56, b=24),
            hoverlabel=dict(
                bgcolor="#fffdfd",
                bordercolor="#e6dcde",
                font=dict(color="#433537"),
            ),
            xaxis=dict(
                showline=True,
                linecolor="#ddd4d6",
                gridcolor="#eee7e8",
                zeroline=False,
                title_font=dict(color="#5d4a4d"),
                tickfont=dict(color="#5d4a4d"),
            ),
            yaxis=dict(
                showline=True,
                linecolor="#ddd4d6",
                gridcolor="#eee7e8",
                zeroline=False,
                title_font=dict(color="#5d4a4d"),
                tickfont=dict(color="#5d4a4d"),
            ),
        )
    )

    pio.templates.default = PLOTLY_TEMPLATE_NAME
    px.defaults.template = PLOTLY_TEMPLATE_NAME
    px.defaults.color_discrete_sequence = pio.templates[PLOTLY_TEMPLATE_NAME].layout.colorway


def _inject_global_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --accent: #c86b72;
            --accent-strong: #ae565c;
            --accent-soft: #f7ebec;
            --app-bg-top: #fffdfd;
            --app-bg-bottom: #faf6f6;
            --surface: rgba(255, 255, 255, 0.92);
            --surface-strong: #fffdfd;
            --sidebar-bg: linear-gradient(180deg, #f9f4f4 0%, #f5efef 100%);
            --border: #e7d8da;
            --text: #433537;
            --muted: #756466;
            --shadow: 0 12px 32px rgba(120, 80, 84, 0.08);
        }

        .stApp {
            background:
                radial-gradient(circle at top right, rgba(200, 107, 114, 0.07), transparent 28%),
                linear-gradient(180deg, var(--app-bg-top) 0%, var(--app-bg-bottom) 100%);
            color: var(--text);
        }

        [data-testid='stStatusWidget'],
        #MainMenu,
        [data-testid='stToolbar'],
        footer,
        header {
            display: none !important;
        }

        [data-testid='stHeader'] {
            background: transparent;
        }

        [data-testid='stSidebar'] {
            background: var(--sidebar-bg);
            border-right: 1px solid rgba(174, 86, 92, 0.07);
        }

        [data-testid='stSidebar'] > div:first-child {
            background: transparent;
        }

        [data-testid='stAppViewContainer'] > .main {
            background: transparent;
        }

        [data-testid='block-container'] {
            padding-top: 2rem;
            padding-bottom: 2.5rem;
        }

        h1, h2, h3, h4, h5, h6 {
            color: var(--text);
            letter-spacing: -0.02em;
        }

        p, li, label, span, div {
            color: inherit;
        }

        [data-testid='stMetric'] {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 0.8rem 1rem;
            box-shadow: var(--shadow);
        }

        [data-testid='stMetricLabel'],
        [data-testid='stMetricDelta'] {
            color: var(--muted);
        }

        .stButton > button,
        .stDownloadButton > button,
        button[kind='secondary'] {
            background: rgba(255, 255, 255, 0.95);
            color: var(--text);
            border: 1px solid var(--border);
            border-radius: 999px;
            box-shadow: 0 8px 20px rgba(120, 80, 84, 0.08);
            transition: all 0.18s ease;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover,
        button[kind='secondary']:hover {
            border-color: var(--accent);
            color: var(--accent-strong);
            transform: translateY(-1px);
        }

        button[kind='primary'] {
            background: linear-gradient(135deg, #c86b72 0%, #d89a9f 100%);
            color: white;
            border: 1px solid transparent;
            box-shadow: 0 12px 28px rgba(174, 86, 92, 0.18);
        }

        button[kind='primary']:hover {
            background: linear-gradient(135deg, #ae565c 0%, #c78389 100%);
            color: white;
        }

        .stSelectbox > div[data-baseweb='select'] > div,
        .stMultiSelect > div[data-baseweb='select'] > div,
        .stNumberInput > div > div,
        .stTextInput > div > div,
        .stDateInput > div > div,
        .stTextArea textarea {
            background: rgba(255, 255, 255, 0.92);
            border-radius: 14px;
            border-color: var(--border) !important;
        }

        .stSlider [data-baseweb='slider'] [role='slider'] {
            background-color: var(--accent);
            box-shadow: 0 0 0 6px rgba(200, 107, 114, 0.1);
        }

        .stSlider [data-baseweb='slider'] > div > div {
            background: linear-gradient(90deg, #e8c6c9 0%, #c86b72 100%);
        }

        [data-testid='stSliderThumbValue'],
        [data-testid='stSliderTickBar'] [data-testid='stMarkdownContainer'] {
            background: transparent !important;
            box-shadow: none !important;
            border: none !important;
        }

        [data-testid='stSliderThumbValue'] p,
        [data-testid='stSliderTickBar'] p {
            color: var(--muted) !important;
            font-weight: 500;
        }

        [data-baseweb='tab-list'] {
            gap: 0.4rem;
        }

        button[role='tab'] {
            border-radius: 999px;
            border: 1px solid transparent;
            background: rgba(255, 255, 255, 0.8);
            padding: 0.35rem 0.9rem;
        }

        button[role='tab'][aria-selected='true'] {
            background: var(--accent-soft);
            border-color: var(--border);
            color: var(--accent-strong);
        }

        [data-testid='stExpander'] {
            border: 1px solid var(--border);
            border-radius: 18px;
            background: var(--surface);
            box-shadow: var(--shadow);
        }

        [data-testid='stDataFrame'],
        .stPlotlyChart,
        .stAlert {
            border-radius: 18px;
        }

        .stPlotlyChart {
            background: var(--surface);
            border: 1px solid var(--border);
            padding: 0.4rem 0.6rem;
            box-shadow: var(--shadow);
        }

        .js-plotly-plot .modebar {
            background: rgba(255, 255, 255, 0.92) !important;
            border: 1px solid var(--border);
            border-radius: 999px;
            padding: 0.14rem 0.28rem !important;
            box-shadow: 0 8px 18px rgba(120, 80, 84, 0.08);
            opacity: 1 !important;
            visibility: visible !important;
            right: 0.8rem !important;
            top: 0.35rem !important;
            transform: translateX(-0.4rem);
        }

        .js-plotly-plot .modebar-group {
            margin-left: 0 !important;
            margin-right: 0 !important;
            display: flex;
            align-items: center;
            gap: 0.12rem;
        }

        .js-plotly-plot .modebar-btn {
            display: inline-flex !important;
            align-items: center;
            justify-content: center;
            width: 1.95rem !important;
            min-width: 1.95rem !important;
            height: 1.95rem !important;
            padding: 0.2rem !important;
            margin: 0 !important;
            border-radius: 999px;
            background: transparent !important;
            opacity: 1 !important;
            transition: background-color 0.18s ease;
        }

        .js-plotly-plot .modebar-btn:hover {
            background: var(--accent-soft) !important;
        }

        .js-plotly-plot .modebar-btn svg {
            opacity: 1 !important;
            width: 1rem !important;
            height: 1rem !important;
        }

        .js-plotly-plot .modebar-btn svg path,
        .js-plotly-plot .modebar-btn svg polygon,
        .js-plotly-plot .modebar-btn svg rect,
        .js-plotly-plot .modebar-btn svg circle,
        .js-plotly-plot .modebar-btn svg line {
            fill: var(--muted) !important;
            stroke: var(--muted) !important;
        }

        .js-plotly-plot .modebar--hover,
        .js-plotly-plot:hover .modebar--hover {
            opacity: 1 !important;
            visibility: visible !important;
        }

        .stAlert {
            border: 1px solid var(--border);
        }

        hr {
            border-color: rgba(174, 86, 92, 0.12);
        }

        code {
            color: var(--accent-strong);
            background: rgba(200, 107, 114, 0.08);
            border-radius: 6px;
            padding: 0.12rem 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
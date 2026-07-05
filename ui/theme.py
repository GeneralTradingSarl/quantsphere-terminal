"""Cyber-Quant visual system: palette, CSS injection, Plotly styling."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st


class C:
    BG = "#0E1117"
    PANEL = "#161B26"
    PANEL_2 = "#12151E"
    GRID = "#21262D"
    TEXT = "#E6EDF3"
    MUTED = "#8B949E"
    BLUE = "#00E5FF"      # forecasts / stochastic paths
    GREEN = "#00E676"     # profit / efficiency
    PURPLE = "#D500F7"    # PDE grids
    AMBER = "#FFEA00"     # Kalman / risk boundaries
    RED = "#FF1744"       # losses / VaR breaches
    WHITE = "#FAFBFC"


MONO = "'Cascadia Code', 'JetBrains Mono', 'Consolas', monospace"

_CSS = f"""
<style>
.stApp {{ background: {C.BG}; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: {C.BG}; }}
::-webkit-scrollbar-thumb {{ background: {C.GRID}; border-radius: 6px; }}
::-webkit-scrollbar-thumb:hover {{ background: {C.MUTED}; }}

.qs-header {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.35rem 0.2rem 0.85rem 0.2rem; margin-bottom: 0.4rem;
    border-bottom: 1px solid {C.GRID};
}}
.qs-logo {{
    font-family: {MONO}; font-size: 1.5rem; font-weight: 700; color: {C.WHITE};
    letter-spacing: 0.12em;
}}
.qs-logo .accent {{ color: {C.BLUE}; text-shadow: 0 0 14px {C.BLUE}66; }}
.qs-sub {{
    font-family: {MONO}; font-size: 0.72rem; color: {C.MUTED};
    letter-spacing: 0.42em; margin-left: 0.9rem; text-transform: uppercase;
}}
.qs-badge {{
    font-family: {MONO}; font-size: 0.7rem; padding: 5px 12px;
    border: 1px solid {C.GRID}; border-radius: 6px; letter-spacing: 0.08em;
}}
.qs-badge.native {{ color: {C.GREEN}; border-color: {C.GREEN}44; background: {C.GREEN}0d; }}
.qs-badge.fallback {{ color: {C.AMBER}; border-color: {C.AMBER}44; background: {C.AMBER}0d; }}
.qs-badge.pro {{ color: {C.BLUE}; border-color: {C.BLUE}44; background: {C.BLUE}0d; }}

div[data-testid="stMetric"] {{
    background: linear-gradient(135deg, {C.PANEL} 0%, {C.PANEL_2} 100%);
    border: 1px solid {C.GRID}; border-left: 3px solid {C.BLUE};
    border-radius: 10px; padding: 12px 16px;
}}
div[data-testid="stMetricValue"] {{
    font-family: {MONO}; font-size: 1.28rem; color: {C.TEXT};
}}
div[data-testid="stMetricLabel"] {{
    color: {C.MUTED}; letter-spacing: 0.09em; text-transform: uppercase;
    font-size: 0.68rem;
}}
div[data-testid="stMetricDelta"] {{ font-family: {MONO}; font-size: 0.85rem; }}

.stTabs [data-baseweb="tab-list"] {{
    gap: 4px; background: {C.PANEL_2}; padding: 5px; border-radius: 10px;
    border: 1px solid {C.GRID};
}}
.stTabs [data-baseweb="tab"] {{
    font-family: {MONO}; font-size: 0.78rem; letter-spacing: 0.06em;
    color: {C.MUTED}; border-radius: 7px; padding: 7px 16px; background: transparent;
}}
.stTabs [aria-selected="true"] {{
    color: {C.BLUE} !important; background: {C.BLUE}14 !important;
    text-shadow: 0 0 10px {C.BLUE}55;
}}

section[data-testid="stSidebar"] {{
    background: {C.PANEL_2}; border-right: 1px solid {C.GRID};
}}
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stNumberInput input {{
    font-family: {MONO};
}}

.stButton > button, .stFormSubmitButton > button {{
    font-family: {MONO}; letter-spacing: 0.08em; border-radius: 8px;
    border: 1px solid {C.BLUE}55; background: {C.BLUE}12; color: {C.BLUE};
    transition: all 0.15s ease;
}}
.stButton > button:hover, .stFormSubmitButton > button:hover {{
    border-color: {C.BLUE}; box-shadow: 0 0 16px {C.BLUE}33; color: {C.WHITE};
}}

div[data-testid="stExpander"] {{
    border: 1px solid {C.GRID}; border-radius: 10px; background: {C.PANEL_2};
}}
h1, h2, h3 {{ color: {C.TEXT}; font-family: {MONO}; letter-spacing: 0.04em; }}
.qs-note {{ color: {C.MUTED}; font-size: 0.78rem; font-family: {MONO}; }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def render_header(engine_label: str = "", native: bool = True) -> None:
    st.markdown(
        """
        <div class="qs-header">
          <div>
            <span class="qs-logo">◈ QUANT<span class="accent">SPHERE</span></span>
            <span class="qs-sub">Terminal</span>
          </div>
          <div class="qs-badge pro">INSTITUTIONAL QUANTITATIVE ANALYTICS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=C.BG,
        plot_bgcolor=C.PANEL_2,
        height=height,
        margin=dict(l=48, r=24, t=54 if title else 28, b=40),
        font=dict(family="Cascadia Code, Consolas, monospace", size=12, color=C.TEXT),
        hoverlabel=dict(bgcolor=C.PANEL, bordercolor=C.GRID,
                        font=dict(family="Consolas, monospace", size=12)),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, orientation="h",
                    yanchor="bottom", y=1.02, x=0),
    )
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=14, color=C.MUTED),
                                     x=0, xanchor="left"))
    fig.update_xaxes(gridcolor=C.GRID, zerolinecolor=C.GRID, linecolor=C.GRID)
    fig.update_yaxes(gridcolor=C.GRID, zerolinecolor=C.GRID, linecolor=C.GRID)
    return fig


# Sequential colorscale used for PDE grids and the volatility surface.
PURPLE_SCALE = [
    [0.0, "#0E1117"], [0.25, "#2A1B4E"], [0.5, "#6A1B9A"],
    [0.75, "#AB47BC"], [1.0, "#D500F7"],
]
BLUE_GREEN_SCALE = [
    [0.0, "#0E1117"], [0.35, "#005f73"], [0.7, "#00E5FF"], [1.0, "#00E676"],
]

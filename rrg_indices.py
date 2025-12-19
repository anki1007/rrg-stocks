import os, time, pathlib, logging, functools, calendar, io
import datetime as _dt
import email.utils as _eutils
import urllib.request as _urlreq
from urllib.parse import quote
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# --- NEW: safe autorefresh imports ---
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None
import streamlit.components.v1 as components
# ------------------------------------- 

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# -------------------- Config --------------------
GITHUB_USER = "anki1007"
GITHUB_REPO = "rrg-stocks"
GITHUB_BRANCH = "main"
GITHUB_TICKER_DIR = "ticker"
CSV_BASENAME = "niftyindices.csv"  # CSV path under /ticker

RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_TICKER_DIR}/"

DEFAULT_TF = "Weekly"
WINDOW = 14
DEFAULT_TAIL = 8

PERIOD_MAP = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y": "3y", "5Y": "5y", "10Y": "10y"}
TF_LABELS = ["Daily", "Weekly", "Monthly"]
TF_TO_INTERVAL = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
BENCH_CHOICES = {"Nifty 500": "^CRSLDX", "Nifty 200": "^CNX200", "Nifty 50": "^NSEI"}

IST_TZ = "Asia/Kolkata"
BAR_CUTOFF_HOUR = 16
NET_TIME_MAX_AGE = 300

CACHE_DIR = pathlib.Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# -------------------- Matplotlib --------------------
mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["font.size"] = 15
mpl.rcParams["font.sans-serif"] = ["Inter", "Segoe UI", "DejaVu Sans", "Arial"]
mpl.rcParams["axes.grid"] = False
mpl.rcParams["axes.edgecolor"] = "#222"
mpl.rcParams["axes.labelcolor"] = "#111"
mpl.rcParams["xtick.color"] = "#333"
mpl.rcParams["ytick.color"] = "#333"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# -------------------- Streamlit Page --------------------
st.set_page_config(page_title="Relative Rotation Graphs – Indices", layout="wide")

# Advanced Plus Jakarta Sans dark theme, keeping all original logic
st.markdown("""
<style>
    @import url('https://rsms.me/inter/inter.css');
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    :root {
        --primary: #00d9ff;
        --primary-dark: #00a8cc;
        --bg: #0a0e27;
        --surface: #16213e;
        --surface-light: #1a2847;
        --text: #e4e4e7;
        --text-muted: #9ca3af;
        --border: #27374d;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
    }
    
    * {
        font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--bg);
        color: var(--text);
    }
    
    [data-testid="stSidebar"] {
        background-color: var(--surface);
        border-right: 1px solid var(--border);
    }
    
    [data-testid="stSidebarNav"] {
        background-color: transparent;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--border);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border: 1px solid var(--border);
        border-bottom: none;
        border-radius: 6px 6px 0 0;
        color: var(--text-muted);
        padding: 10px 20px;
        transition: all 200ms ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--surface-light);
        color: var(--text);
    }
    
    .stTabs [aria-selected="true"] [data-baseweb="tab"] {
        background-color: var(--primary);
        color: var(--bg);
        border-color: var(--primary);
    }
    
    [data-baseweb="select"] {
        background-color: var(--surface-light);
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: var(--text);
    }
    
    [data-testid="stNumberInput"] input,
    [data-testid="stTextInput"] input,
    input[type="text"],
    input[type="number"] {
        background-color: var(--surface-light) !important;
        border: 1px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 6px !important;
    }
    
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextInput"] input:focus,
    input[type="text"]:focus,
    input[type="number"]:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 3px rgba(0, 217, 255, 0.1) !important;
    }
    
    [data-baseweb="button"] {
        background-color: var(--primary) !important;
        color: var(--bg) !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        transition: all 200ms ease !important;
    }
    
    [data-baseweb="button"]:hover {
        background-color: var(--primary-dark) !important;
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(0, 217, 255, 0.2);
    }
    
    [data-baseweb="button"]:active {
        transform: translateY(0);
    }
    
    .metric-label {
        color: var(--text-muted);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 4px;
    }
    
    [data-testid="stMetricValue"] {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: var(--text) !important;
    }
    
    [data-testid="stMarkdownContainer"] table {
        background-color: var(--surface);
        border-collapse: collapse;
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
        width: 100%;
    }
    
    [data-testid="stMarkdownContainer"] table th {
        background-color: var(--surface-light);
        color: var(--text);
        border: 1px solid var(--border);
        padding: 12px;
        text-align: left;
        font-weight: 600;
    }
    
    [data-testid="stMarkdownContainer"] table td {
        border: 1px solid var(--border);
        padding: 12px;
        color: var(--text);
    }
    
    [data-testid="stMarkdownContainer"] table tr:hover {
        background-color: var(--surface-light);
    }
    
    .stSelectbox [data-baseweb="select"] {
        background-color: var(--surface-light) !important;
        border: 1px solid var(--border) !important;
    }
    
    .stMultiSelect [data-baseweb="multi-select"] {
        background-color: var(--surface-light) !important;
        border: 1px solid var(--border) !important;
    }
    
    hr {
        border-color: var(--border) !important;
    }
    
    a {
        color: var(--primary) !important;
        text-decoration: none;
    }
    
    a:hover {
        color: var(--primary-dark) !important;
        text-decoration: underline;
    }
    
    /* Remove spinner dots on rerun */
    [data-testid="stStatusWidget"] {
        display: none;
    }
    
    /* Fix select dropdown appearance */
    [role="listbox"] {
        background-color: var(--surface-light) !important;
        border: 1px solid var(--border) !important;
    }
    
    [role="option"] {
        background-color: var(--surface-light) !important;
        color: var(--text) !important;
    }
    
    [role="option"][aria-selected="true"] {
        background-color: var(--primary) !important;
        color: var(--bg) !important;
    }
</style>
""", unsafe_allow_html=True)

# Hero title above the dynamic date title
st.markdown(
    '<p style="font-size: 48px; font-weight: 700; color: #00d9ff; margin-bottom: 2px; letter-spacing: -1px;">Relative Rotation Graphs</p>',
    unsafe_allow_html=True
)
st.markdown(
    '<p style="font-size: 14px; color: #9ca3af; margin-bottom: 24px;">Analyze multi-index momentum & relative strength across Indian market indices</p>',
    unsafe_allow_html=True
)

# ========== Helper Functions ==========

def fetch_csv_from_url(url: str) -> pd.DataFrame:
    """Fetch CSV from GitHub URL"""
    try:
        response = _urlreq.urlopen(url, timeout=10)
        df = pd.read_csv(response)
        return df
    except Exception as e:
        logging.error(f"Error fetching CSV: {e}")
        return None

def fetch_data_yfinance(tickers: List[str], period: str, interval: str) -> pd.DataFrame:
    """Fetch historical data from yfinance"""
    try:
        data = yf.download(tickers, period=period, interval=interval, progress=False)
        if len(tickers) == 1:
            data.columns = pd.MultiIndex.from_product([[tickers[0]], data.columns])
        return data["Close"]
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return None

def calc_rs_ratio(stock_close: np.ndarray, bench_close: np.ndarray) -> np.ndarray:
    """Calculate RS Ratio (stock / benchmark)"""
    return stock_close / bench_close

def calc_rrg_axes(close_data: pd.DataFrame, bench_ticker: str, window: int = 14):
    """
    Calculate RRG Axes:
    - RS-Ratio: RS(14) momentum
    - RS-Momentum: momentum of RS-Ratio
    """
    bench_close = close_data[bench_ticker].values
    
    rrg_data = {}
    
    for ticker in close_data.columns:
        if ticker == bench_ticker:
            continue
        
        stock_close = close_data[ticker].values
        
        # RS Ratio = stock / benchmark
        rs_ratio = calc_rs_ratio(stock_close, bench_close)
        
        # RS Momentum = 14-period momentum of RS Ratio
        rs_momentum = np.full_like(rs_ratio, np.nan)
        for i in range(window, len(rs_ratio)):
            rs_momentum[i] = rs_ratio[i] / rs_ratio[i - window] - 1
        
        # Scale to 100 for visualization
        rs_ratio_scaled = rs_ratio / rs_ratio[0] * 100
        rs_momentum_scaled = rs_momentum * 100 + 100
        
        rrg_data[ticker] = {
            "rs_ratio": rs_ratio_scaled[-1],
            "rs_momentum": rs_momentum_scaled[-1],
            "rs_ratio_full": rs_ratio_scaled,
            "rs_momentum_full": rs_momentum_scaled,
        }
    
    return rrg_data

def plot_rrg(rrg_data: Dict, title: str = "Relative Rotation Graph"):
    """Plot RRG with quadrants"""
    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#16213e")
    ax.set_facecolor("#16213e")
    
    # Extract data
    tickers = list(rrg_data.keys())
    rs_ratios = np.array([rrg_data[t]["rs_ratio"] for t in tickers])
    rs_momentums = np.array([rrg_data[t]["rs_momentum"] for t in tickers])
    
    # Quadrant lines
    ax.axhline(y=100, color="#444", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=100, color="#444", linestyle="--", linewidth=1, alpha=0.5)
    
    # Quadrant backgrounds
    ax.fill_between([94, 100], 94, 106, alpha=0.1, color="#ef4444", label="Lagging")
    ax.fill_between([100, 106], 94, 100, alpha=0.1, color="#f59e0b", label="Weakening")
    ax.fill_between([94, 100], 100, 106, alpha=0.1, color="#00d9ff", label="Improving")
    ax.fill_between([100, 106], 100, 106, alpha=0.1, color="#10b981", label="Leading")
    
    # Add quadrant labels
    ax.text(97, 105, "Improving", fontsize=12, fontweight="600", color="#9ca3af", ha="center")
    ax.text(103, 105, "Leading", fontsize=12, fontweight="600", color="#9ca3af", ha="center")
    ax.text(97, 95, "Lagging", fontsize=12, fontweight="600", color="#9ca3af", ha="center")
    ax.text(103, 95, "Weakening", fontsize=12, fontweight="600", color="#9ca3af", ha="center")
    
    # Plot points and trails
    colors = plt.cm.tab20(np.linspace(0, 1, len(tickers)))
    
    for i, ticker in enumerate(tickers):
        rs_ratio_trail = rrg_data[ticker]["rs_ratio_full"]
        rs_momentum_trail = rrg_data[ticker]["rs_momentum_full"]
        
        # Trail
        valid = ~(np.isnan(rs_ratio_trail) | np.isnan(rs_momentum_trail))
        ax.plot(rs_ratio_trail[valid], rs_momentum_trail[valid], color=colors[i], alpha=0.3, linewidth=1)
        
        # Current point
        ax.scatter(rs_ratios[i], rs_momentums[i], s=100, color=colors[i], edgecolor="white", linewidth=1.5, zorder=5, label=ticker)
    
    ax.set_xlim(94, 106)
    ax.set_ylim(94, 106)
    ax.set_xlabel("JdK RS-Ratio", fontsize=12, color="#9ca3af", fontweight="600")
    ax.set_ylabel("JdK RS-Momentum", fontsize=12, color="#9ca3af", fontweight="600")
    ax.set_title(title, fontsize=14, color="#e4e4e7", fontweight="700", pad=20)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#27374d")
    ax.spines["bottom"].set_color("#27374d")
    ax.tick_params(colors="#9ca3af")
    
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, facecolor="#16213e", edgecolor="#27374d")
    
    plt.tight_layout()
    return fig

# ========== Main Application ==========

with st.sidebar:
    st.markdown("### RRG – Controls", help="Configure RRG parameters")
    
    # Benchmark selection
    benchmark = st.selectbox("Benchmark", list(BENCH_CHOICES.keys()), index=0, key="bench_select")
    bench_ticker = BENCH_CHOICES[benchmark]
    
    # Timeframe selection
    timeframe = st.selectbox("Strength via", TF_LABELS, index=1, key="tf_select")
    interval = TF_TO_INTERVAL[timeframe]
    
    # Period selection
    period = st.selectbox("Period", list(PERIOD_MAP.keys()), index=3, key="period_select")
    yf_period = PERIOD_MAP[period]
    
    # Rank by selection
    rank_by = st.selectbox("Rank by", ["RS-Momentum", "RS-Ratio"], index=0, key="rank_select")
    
    # Trail length
    trail_length = st.slider("Trail Length", min_value=1, max_value=20, value=DEFAULT_TAIL, step=1, key="trail_select")
    
    # Max items in ranking
    max_items = st.slider("Max Items in Ranking", min_value=5, max_value=50, value=20, step=5, key="max_items_select")
    
    # Show labels toggle
    show_labels = st.checkbox("Show labels on chart", value=False, key="show_labels")
    
    # Reload button
    if st.button("Reload Universe", key="reload_btn", use_container_width=True):
        st.rerun()

# Fetch CSV with default data
csv_url = RAW_BASE + CSV_BASENAME
ticker_df = fetch_csv_from_url(csv_url)

if ticker_df is not None:
    tickers = [bench_ticker] + ticker_df["Ticker"].tolist()
else:
    st.error("Could not load indices CSV. Using default Nifty 50.")
    tickers = [bench_ticker, "^NSEI"]

# Fetch price data
with st.spinner(f"Fetching {period} {timeframe} data..."):
    close_data = fetch_data_yfinance(tickers, yf_period, interval)

if close_data is not None and not close_data.empty:
    # Calculate RRG
    rrg_data = calc_rrg_axes(close_data, bench_ticker, WINDOW)
    
    # Get current date
    current_date = close_data.index[-1].strftime("%Y-%m-%d")
    
    # Create main chart
    col1, col2 = st.columns([3, 1], gap="large")
    
    with col1:
        st.markdown(f"### Relative Rotation Graph (RRG) – Indices – {current_date}")
        fig = plot_rrg(rrg_data, title="")
        st.pyplot(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Ranking")
        
        # Sort data
        if rank_by == "RS-Momentum":
            sorted_data = sorted(rrg_data.items(), key=lambda x: x[1]["rs_momentum"], reverse=True)
        else:
            sorted_data = sorted(rrg_data.items(), key=lambda x: x[1]["rs_ratio"], reverse=True)
        
        # Display ranking
        ranking_html = "<ol style='padding-left: 20px; color: #e4e4e7;'>"
        for rank, (ticker, data) in enumerate(sorted_data[:max_items], 1):
            rs_ratio = data["rs_ratio"]
            rs_momentum = data["rs_momentum"]
            
            # Determine status
            if rs_ratio > 100 and rs_momentum > 100:
                status = "Leading"
                status_color = "#10b981"
            elif rs_ratio < 100 and rs_momentum < 100:
                status = "Lagging"
                status_color = "#ef4444"
            elif rs_ratio > 100 and rs_momentum < 100:
                status = "Weakening"
                status_color = "#f59e0b"
            else:
                status = "Improving"
                status_color = "#00d9ff"
            
            ranking_html += f'<li style="margin-bottom: 8px;"><strong>{ticker}</strong> <span style="color: {status_color};">[{status}]</span></li>'
        
        ranking_html += "</ol>"
        st.markdown(ranking_html, unsafe_allow_html=True)
    
    # Data table with improved layout
    st.markdown("---")
    st.markdown("### Detailed Ranking Table")
    
    # Create DataFrame for table
    table_data = []
    for ticker, data in sorted_data[:max_items]:
        rs_ratio = data["rs_ratio"]
        rs_momentum = data["rs_momentum"]
        change_pct = ((close_data[ticker].iloc[-1] / close_data[ticker].iloc[0]) - 1) * 100
        
        # Determine status
        if rs_ratio > 100 and rs_momentum > 100:
            status = "Leading"
        elif rs_ratio < 100 and rs_momentum < 100:
            status = "Lagging"
        elif rs_ratio > 100 and rs_momentum < 100:
            status = "Weakening"
        else:
            status = "Improving"
        
        table_data.append({
            "Ranking": len(table_data) + 1,
            "Ticker": ticker,
            "Status": status,
            "Industry": "Index" if ticker.startswith("^") else "Stock",
            "RS-Ratio": f"{rs_ratio:.2f}",
            "RS-Momentum": f"{rs_momentum:.2f}",
            "Change %": f"{change_pct:.2f}%"
        })
    
    table_df = pd.DataFrame(table_data)
    st.dataframe(table_df, use_container_width=True, hide_index=True)
    
else:
    st.error("Could not fetch price data. Please check your internet connection or ticker symbols.")

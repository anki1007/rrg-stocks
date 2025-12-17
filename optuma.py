# ==========================================================
# OPTUMA-GRADE RRG ENGINE — BLOOMBERG STYLE UI
# ==========================================================

import os, time, pathlib, calendar, functools, io
import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from urllib.parse import quote

# -------------------- STREAMLIT CONFIG --------------------
st.set_page_config(
    page_title="RRG Terminal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- UI THEMES ---------------------------
THEMES = {
    "Bloomberg Dark": """
    <style>
    html, body {
        background:#0b0f14 !important;
        color:#e5e7eb !important;
        font-family: 'IBM Plex Mono','JetBrains Mono', monospace;
        font-size:13px;
    }
    h1,h2,h3 { color:#fbbf24 !important; }
    .stDataFrame { background:#0b0f14; }
    </style>
    """,

    "Bloomberg Light": """
    <style>
    html, body {
        background:#f8fafc !important;
        color:#0f172a !important;
        font-family: 'IBM Plex Mono','JetBrains Mono', monospace;
        font-size:13px;
    }
    h1,h2,h3 { color:#1d4ed8 !important; }
    </style>
    """
}

theme = st.sidebar.selectbox("UI Theme", THEMES.keys())
st.markdown(THEMES[theme], unsafe_allow_html=True)

# -------------------- PARAMETERS --------------------------
WINDOW = 14
DEFAULT_TAIL = 8

BENCHMARKS = {
    "Nifty 50": "^NSEI",
    "Nifty 200": "^CNX200",
    "Nifty 500": "^CRSLDX"
}

# -------------------- HELPERS -----------------------------
def jdk_components(price, bench, win=14):
    df = pd.concat([price, bench], axis=1).dropna()
    rs = 100 * df.iloc[:,0] / df.iloc[:,1]
    m = rs.rolling(win).mean()
    s = rs.rolling(win).std(ddof=0).replace(0, 1e-9)
    rs_ratio = 100 + (rs - m) / s
    roc = rs_ratio.pct_change() * 100
    m2 = roc.rolling(win).mean()
    s2 = roc.rolling(win).std(ddof=0).replace(0, 1e-9)
    rs_mom = 101 + (roc - m2) / s2
    return rs_ratio.dropna(), rs_mom.dropna()

def get_status(x, y):
    if x >= 100 and y >= 100: return "Leading"
    if x < 100 and y >= 100: return "Improving"
    if x >= 100 and y < 100: return "Weakening"
    return "Lagging"

def zscore(s):
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

# -------------------- DATA LOAD ---------------------------
@st.cache_data(ttl=900)
def load_prices(symbols, bench, period="1y"):
    data = yf.download(symbols + [bench], period=period, auto_adjust=True, progress=False)
    return data

# -------------------- SIDEBAR CONTROLS --------------------
st.sidebar.header("RRG Controls")

benchmark_label = st.sidebar.selectbox("Benchmark", BENCHMARKS.keys())
benchmark = BENCHMARKS[benchmark_label]

period = st.sidebar.selectbox("Period", ["6mo","1y","2y","3y"], index=1)
tail = st.sidebar.slider("Trail Length", 3, 20, DEFAULT_TAIL)

x_min, x_max = st.sidebar.slider("RS-Ratio Zoom", 85.0, 115.0, (94.0, 106.0), 0.5)
y_min, y_max = st.sidebar.slider("RS-Momentum Zoom", 85.0, 115.0, (94.0, 106.0), 0.5)

symbols_input = st.sidebar.text_area(
    "Symbols (comma separated, NSE)",
    "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS"
)

symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

# -------------------- PRICE DATA --------------------------
raw = load_prices(symbols, benchmark, period)

bench = raw["Close"][benchmark]
prices = {s: raw["Close"][s].dropna() for s in symbols if s in raw["Close"]}

# -------------------- RRG CALCULATION ---------------------
rr_map, mm_map = {}, {}
for s, px in prices.items():
    rr, mm = jdk_components(px, bench, WINDOW)
    rr_map[s] = rr
    mm_map[s] = mm

idx = bench.index
end_idx = -1
start_idx = -tail

# -------------------- COMPOSITE MOMENTUM SCORE ------------
rows = []
for s in symbols:
    rr = rr_map[s].iloc[end_idx]
    mm = mm_map[s].iloc[end_idx]

    slope = np.polyfit(
        range(tail),
        mm_map[s].iloc[start_idx:end_idx].values,
        1
    )[0]

    rel_perf = (prices[s].iloc[end_idx] / prices[s].iloc[start_idx] - 1) * 100

    rows.append((s, rr, mm, slope, rel_perf))

df = pd.DataFrame(rows, columns=["Symbol","RS-Ratio","RS-Momentum","Slope","RelPerf"])

df["MomentumScore"] = (
    0.35*zscore(df["RS-Ratio"]) +
    0.35*zscore(df["RS-Momentum"]) +
    0.15*zscore(df["Slope"]) +
    0.15*zscore(df["RelPerf"])
)

df["MomentumScore"] = 50 + 10*df["MomentumScore"]
df["Status"] = df.apply(lambda r: get_status(r["RS-Ratio"], r["RS-Momentum"]), axis=1)
df = df.sort_values("MomentumScore", ascending=False).reset_index(drop=True)

# -------------------- LAYOUT ------------------------------
plot_col, table_col = st.columns([3.5, 1.5])

# -------------------- RRG PLOT ----------------------------
with plot_col:
    fig, ax = plt.subplots(figsize=(10,6))
    ax.axhline(100, ls=":", c="gray")
    ax.axvline(100, ls=":", c="gray")

    ax.fill_between([94,100],[94,94],[100,100], color=(1,0,0,0.15))
    ax.fill_between([100,106],[94,94],[100,100], color=(1,1,0,0.15))
    ax.fill_between([100,106],[100,100],[106,106], color=(0,1,0,0.15))
    ax.fill_between([94,100],[100,100],[106,106], color=(0,0,1,0.15))

    for s in symbols:
        rr = rr_map[s].iloc[start_idx:end_idx]
        mm = mm_map[s].iloc[start_idx:end_idx]
        ax.plot(rr, mm, lw=1.2)
        ax.scatter(rr.iloc[-1], mm.iloc[-1], s=80)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("RS-Ratio")
    ax.set_ylabel("RS-Momentum")

    st.pyplot(fig, use_container_width=True)

# -------------------- RIGHT SIDEBAR RANKING ---------------
with table_col:
    st.markdown("### Momentum Ranking")

    for q in ["Leading","Improving","Weakening","Lagging"]:
        with st.expander(q, expanded=(q=="Leading")):
            sub = df[df["Status"]==q].head(10)
            for i,r in sub.iterrows():
                st.markdown(f"**{i+1}. {r['Symbol']}**  \nScore: `{r['MomentumScore']:.1f}`")

# -------------------- FILTERABLE TABLE --------------------
st.markdown("## 📋 RRG Table")

flt_status = st.multiselect("Filter Status", df["Status"].unique(), df["Status"].unique())

table = df[df["Status"].isin(flt_status)]

st.dataframe(
    table[[
        "Symbol","Status","MomentumScore",
        "RS-Ratio","RS-Momentum","Slope","RelPerf"
    ]].rename(columns={
        "MomentumScore":"Momentum Score",
        "RelPerf":"Rel Price %"
    }),
    use_container_width=True,
    height=420
)

st.caption("Optuma-grade Relative Rotation Graph — Bloomberg style terminal UI")

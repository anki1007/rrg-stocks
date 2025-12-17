# ============================================================
# RRG STOCKS — STREAMLIT SAFE VERSION
# FIXED: GitHub API rate-limit issue
# ============================================================

import os, time, pathlib, logging, functools, calendar, io
import datetime as _dt
from urllib.parse import quote
from typing import List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# ============================================================
# CONFIG
# ============================================================

GITHUB_USER   = "anki1007"
GITHUB_REPO   = "rrg-stocks"
GITHUB_BRANCH = "main"
GITHUB_TICKER_DIR = "ticker"

RAW_BASE = (
    f"https://raw.githubusercontent.com/"
    f"{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_TICKER_DIR}/"
)

# 👇 STATIC CSV LIST (NO API CALLS)
CSV_FILES = [
    "nifty50.csv",
    "nifty200.csv",
    "nifty500.csv",
    "niftymidcap150.csv",
    "niftysmallcap250.csv",
    "niftymidsmallcap400.csv",
    "niftytotalmarket.csv",
]

FRIENDLY_NAMES = {
    "nifty50.csv": "Nifty 50",
    "nifty200.csv": "Nifty 200",
    "nifty500.csv": "Nifty 500",
    "niftymidcap150.csv": "Nifty Midcap 150",
    "niftysmallcap250.csv": "Nifty Smallcap 250",
    "niftymidsmallcap400.csv": "Nifty MidSmallcap 400",
    "niftytotalmarket.csv": "Nifty Total Market",
}

BENCHMARKS = {
    "Nifty 50": "^NSEI",
    "Nifty 200": "^CNX200",
    "Nifty 500": "^CRSLDX",
}

PERIOD_MAP = {
    "6M": "6mo",
    "1Y": "1y",
    "2Y": "2y",
    "3Y": "3y",
    "5Y": "5y",
    "10Y": "10y",
}

TF_MAP = {
    "Daily": "1d",
    "Weekly": "1wk",
    "Monthly": "1mo",
}

WINDOW = 14
TAIL_DEFAULT = 8

# ============================================================
# STREAMLIT SETUP
# ============================================================

st.set_page_config(
    page_title="Relative Rotation Graph (RRG)",
    layout="wide"
)

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["font.size"] = 12

# ============================================================
# HELPERS
# ============================================================

def friendly_name(csv):
    return FRIENDLY_NAMES.get(csv, csv.replace(".csv", "").title())

def build_name_maps():
    name_map = {friendly_name(f): f for f in CSV_FILES}
    return name_map, sorted(name_map.keys())

@st.cache_data(ttl=3600)
def load_universe(csv_file):
    url = RAW_BASE + csv_file
    df = pd.read_csv(url)

    cols = {c.lower().replace(" ", ""): c for c in df.columns}
    sym_col = cols.get("symbol") or cols.get("ticker")

    if not sym_col:
        raise ValueError("CSV must contain Symbol column")

    df = df[[sym_col]].dropna()
    df[sym_col] = df[sym_col].astype(str).str.strip()

    return df[sym_col].unique().tolist()

def jdk_rs(price, bench, win=14):
    df = pd.concat([price, bench], axis=1).dropna()
    rs = 100 * df.iloc[:, 0] / df.iloc[:, 1]

    m = rs.rolling(win).mean()
    s = rs.rolling(win).std().replace(0, np.nan)

    rs_ratio = 100 + (rs - m) / s
    rroc = rs_ratio.pct_change() * 100

    m2 = rroc.rolling(win).mean()
    s2 = rroc.rolling(win).std().replace(0, np.nan)

    rs_mom = 101 + (rroc - m2) / s2
    return rs_ratio.dropna(), rs_mom.dropna()

def quadrant(x, y):
    if x >= 100 and y >= 100:
        return "Leading"
    if x < 100 and y >= 100:
        return "Improving"
    if x < 100 and y < 100:
        return "Lagging"
    return "Weakening"

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("RRG Controls")

NAME_MAP, DISPLAY_LIST = build_name_maps()

csv_display = st.sidebar.selectbox("Universe", DISPLAY_LIST)
csv_file = NAME_MAP[csv_display]

bench_display = st.sidebar.selectbox("Benchmark", list(BENCHMARKS))
benchmark = BENCHMARKS[bench_display]

tf_label = st.sidebar.selectbox("Timeframe", list(TF_MAP))
interval = TF_MAP[tf_label]

period_label = st.sidebar.selectbox("Period", list(PERIOD_MAP))
period = PERIOD_MAP[period_label]

tail = st.sidebar.slider("Trail Length", 3, 20, TAIL_DEFAULT)

# ============================================================
# DATA LOAD
# ============================================================

symbols = load_universe(csv_file)

prices = yf.download(
    symbols + [benchmark],
    period=period,
    interval=interval,
    group_by="ticker",
    auto_adjust=True,
    progress=False,
)

bench_close = prices[benchmark]["Close"]

rs_data = {}

for sym in symbols:
    try:
        s = prices[sym]["Close"]
        rr, rm = jdk_rs(s, bench_close, WINDOW)
        if len(rr) > 20:
            rs_data[sym] = (rr, rm)
    except Exception:
        continue

if not rs_data:
    st.error("No valid symbols after calculation.")
    st.stop()

# ============================================================
# PLOT
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

ax.axhline(100, linestyle=":", color="gray")
ax.axvline(100, linestyle=":", color="gray")

ax.set_xlim(94, 106)
ax.set_ylim(94, 106)

ax.set_xlabel("RS Ratio")
ax.set_ylabel("RS Momentum")
ax.set_title(f"RRG — {csv_display} vs {bench_display}")

colors = plt.cm.tab20.colors

for i, (sym, (rr, rm)) in enumerate(rs_data.items()):
    rr_t = rr.iloc[-tail:]
    rm_t = rm.iloc[-tail:]

    ax.plot(rr_t, rm_t, color=colors[i % 20], alpha=0.7)
    ax.scatter(rr_t.iloc[-1], rm_t.iloc[-1], s=70, color=colors[i % 20])
    ax.text(rr_t.iloc[-1] + 0.2, rm_t.iloc[-1], sym, fontsize=9)

st.pyplot(fig, use_container_width=True)

# ============================================================
# TABLE
# ============================================================

rows = []
for sym, (rr, rm) in rs_data.items():
    x, y = rr.iloc[-1], rm.iloc[-1]
    rows.append({
        "Symbol": sym,
        "RS Ratio": round(x, 2),
        "RS Momentum": round(y, 2),
        "Quadrant": quadrant(x, y),
    })

df_out = pd.DataFrame(rows).sort_values("RS Ratio", ascending=False)

st.dataframe(df_out, use_container_width=True)

st.caption("✔ GitHub-API-free | ✔ Streamlit-Cloud-safe | ✔ Production ready")

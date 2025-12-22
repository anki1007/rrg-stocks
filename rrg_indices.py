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
import mplcursors

# --- safe autorefresh ---
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None
import streamlit.components.v1 as components

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# -------------------- Config --------------------
GITHUB_USER = "anki1007"
GITHUB_REPO = "rrg-stocks"
GITHUB_BRANCH = "main"
GITHUB_TICKER_DIR = "ticker"
CSV_BASENAME = "niftyindices.csv"
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_TICKER_DIR}/"

WINDOW = 14
DEFAULT_TAIL = 8
TF_TO_INTERVAL = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
BENCH_CHOICES = {"Nifty 500": "^CRSLDX", "Nifty 200": "^CNX200", "Nifty 50": "^NSEI"}

IST_TZ = "Asia/Kolkata"
BAR_CUTOFF_HOUR = 16
NET_TIME_MAX_AGE = 300

CACHE_DIR = pathlib.Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["font.size"] = 14

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="RRG – Indices", layout="wide")
st.markdown("## Relative Rotation Graphs – Indices")

# -------------------- Helpers --------------------
def _to_yahoo_symbol(s): return s if s.startswith("^") else "^"+s

@st.cache_data(ttl=600)
def load_universe_from_github_csv(basename, cache_bust):
    df = pd.read_csv(RAW_BASE + basename)
    df["Yahoo"] = df["Symbol"].apply(_to_yahoo_symbol)
    meta = {
        r["Yahoo"]: {
            "name": r["Company Name"],
            "industry": r.get("Industry", "-")
        }
        for _, r in df.iterrows()
    }
    return df["Yahoo"].tolist(), meta

def jdk_components(price, bench, win=14):
    df = pd.concat([price, bench], axis=1).dropna()
    rs = 100 * (df.iloc[:,0] / df.iloc[:,1])
    rs_ratio = 100 + (rs - rs.rolling(win).mean()) / rs.rolling(win).std()
    rs_mom = 101 + rs_ratio.pct_change().rolling(win).mean() * 100
    return rs_ratio.dropna(), rs_mom.dropna()

def get_status(x, y):
    if x>=100 and y>=100: return "Leading"
    if x<100 and y>=100: return "Improving"
    if x<100 and y<100: return "Lagging"
    return "Weakening"

# -------------------- Controls --------------------
bench_label = st.sidebar.selectbox("Benchmark", list(BENCH_CHOICES.keys()))
interval_label = st.sidebar.selectbox("Timeframe", list(TF_TO_INTERVAL.keys()), index=1)
interval = TF_TO_INTERVAL[interval_label]

# -------------------- Data --------------------
UNIVERSE, META = load_universe_from_github_csv(
    CSV_BASENAME,
    str(pd.Timestamp.utcnow())
)

bench_symbol = BENCH_CHOICES[bench_label]
raw = yf.download(UNIVERSE + [bench_symbol], period="1y", interval=interval, group_by="ticker", auto_adjust=True)

bench = raw[bench_symbol]["Close"].dropna()

rs_ratio_map, rs_mom_map, tickers = {}, {}, []

for t in UNIVERSE:
    try:
        px = raw[t]["Close"].dropna()
        rr, mm = jdk_components(px, bench)
        rs_ratio_map[t] = rr
        rs_mom_map[t] = mm
        tickers.append(t)
    except:
        pass

end_idx = -1
start_idx = -DEFAULT_TAIL

# -------------------- Plot --------------------
fig, ax = plt.subplots(figsize=(11, 7))
ax.axhline(100, linestyle=":")
ax.axvline(100, linestyle=":")
ax.set_xlim(94,106); ax.set_ylim(94,106)
ax.set_xlabel("RS-Ratio")
ax.set_ylabel("RS-Momentum")

colors = {t: to_hex(plt.cm.tab20(i)) for i,t in enumerate(tickers)}

for t in tickers:
    rr = rs_ratio_map[t].iloc[start_idx:end_idx]
    mm = rs_mom_map[t].iloc[start_idx:end_idx]

    ax.plot(rr, mm, color=colors[t], alpha=0.6)

    last_rr, last_mm = rr.iloc[-1], mm.iloc[-1]

    sc = ax.scatter(last_rr, last_mm, s=90, color=colors[t], edgecolor="black", zorder=5)

    sc._rrg_meta = {
        "name": META[t]["name"],
        "symbol": t,
        "industry": META[t]["industry"],
        "rs_ratio": last_rr,
        "rs_mom": last_mm,
        "status": get_status(last_rr, last_mm),
        "price": raw[t]["Close"].iloc[end_idx],
    }

    ax.annotate(
        META[t]["name"],
        (last_rr, last_mm),
        xytext=(6,6),
        textcoords="offset points",
        fontsize=9
    )

cursor = mplcursors.cursor(ax.collections, hover=True)

@cursor.connect("add")
def on_add(sel):
    m = getattr(sel.artist, "_rrg_meta", None)
    if not m:
        sel.annotation.set_visible(False)
        return
    sel.annotation.set_text(
        f"Name      : {m['name']}\n"
        f"Symbol    : {m['symbol']}\n"
        f"Industry  : {m['industry']}\n"
        f"Status    : {m['status']}\n"
        f"RS-Ratio  : {m['rs_ratio']:.2f}\n"
        f"RS-Mom    : {m['rs_mom']:.2f}\n"
        f"Price     : {m['price']:.2f}"
    )
    sel.annotation.get_bbox_patch().set(fc="#111", alpha=0.9)
    sel.annotation.get_text().set_color("white")

st.pyplot(fig, use_container_width=True)

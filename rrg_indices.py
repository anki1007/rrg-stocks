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

# --- Safe autorefresh ---
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None
import streamlit.components.v1 as components

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# ===================== CONFIG =====================
GITHUB_USER = "anki1007"
GITHUB_REPO = "rrg-stocks"
GITHUB_BRANCH = "main"
GITHUB_TICKER_DIR = "ticker"
CSV_BASENAME = "niftyindices.csv"
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_TICKER_DIR}/"

WINDOW = 14
DEFAULT_TAIL = 8
TF_LABELS = ["Daily", "Weekly", "Monthly"]
TF_TO_INTERVAL = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
BENCH_CHOICES = {
    "Nifty 50": "^NSEI",
    "Nifty 200": "^CNX200",
    "Nifty 500": "^CRSLDX",
}

CACHE_DIR = pathlib.Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["font.size"] = 14
mpl.rcParams["axes.grid"] = False

logging.basicConfig(level=logging.INFO)

# ===================== PAGE =====================
st.set_page_config(
    page_title="Relative Rotation Graphs – Indices",
    layout="wide"
)

st.markdown("## Relative Rotation Graphs – Indices")

# ===================== HELPERS =====================
def _normalize_cols(cols):
    return {c: c.strip().lower().replace(" ", "") for c in cols}

def _to_yahoo_symbol(sym):
    s = str(sym).strip().upper()
    return s if s.startswith("^") else "^" + s

@st.cache_data(ttl=600)
def load_universe_from_github_csv(basename, cache_bust):
    df = pd.read_csv(RAW_BASE + basename)
    cols = _normalize_cols(df.columns)
    sym_col = next(c for c,k in cols.items() if k == "symbol")
    name_col = next((c for c,k in cols.items() if k in ("companyname","name")), sym_col)
    ind_col = next((c for c,k in cols.items() if k in ("industry","sector")), None)

    if ind_col is None:
        df["Industry"] = "-"
        ind_col = "Industry"

    df = df[[sym_col, name_col, ind_col]].copy()
    df.columns = ["Symbol", "Name", "Industry"]
    df["Yahoo"] = df["Symbol"].apply(_to_yahoo_symbol)

    meta = {
        r["Yahoo"]: {
            "name": r["Name"],
            "industry": r["Industry"]
        }
        for _, r in df.iterrows()
    }
    return df["Yahoo"].tolist(), meta

def pick_close(df, symbol):
    if isinstance(df.columns, pd.MultiIndex):
        return df[(symbol, "Close")].dropna()
    return df["Close"].dropna()

def jdk_components(price, bench, win=14):
    df = pd.concat([price, bench], axis=1).dropna()
    rs = 100 * (df.iloc[:,0] / df.iloc[:,1])
    rs_ratio = 100 + (rs - rs.rolling(win).mean()) / rs.rolling(win).std()
    rs_mom = 101 + rs_ratio.pct_change().rolling(win).mean() * 100
    ix = rs_ratio.index.intersection(rs_mom.index)
    return rs_ratio.loc[ix], rs_mom.loc[ix]

def get_status(x, y):
    if x>=100 and y>=100: return "Leading"
    if x<100 and y>=100: return "Improving"
    if x<100 and y<100: return "Lagging"
    return "Weakening"

# ===================== CONTROLS =====================
st.sidebar.header("Controls")

bench_label = st.sidebar.selectbox("Benchmark", list(BENCH_CHOICES.keys()), index=0)
tf_label = st.sidebar.selectbox("Timeframe", TF_LABELS, index=1)
interval = TF_TO_INTERVAL[tf_label]

tail_len = st.sidebar.slider("Trail Length", 3, 20, DEFAULT_TAIL)
show_labels = st.sidebar.toggle("Show Names on Chart", value=True)

# ===================== DATA =====================
UNIVERSE, META = load_universe_from_github_csv(
    CSV_BASENAME,
    cache_bust=str(pd.Timestamp.utcnow())
)

benchmark = BENCH_CHOICES[bench_label]

raw = yf.download(
    UNIVERSE + [benchmark],
    period="1y",
    interval=interval,
    group_by="ticker",
    auto_adjust=True,
    threads=True,
    progress=False
)

bench_px = pick_close(raw, benchmark)

rs_ratio_map, rs_mom_map, tickers = {}, {}, []

for t in UNIVERSE:
    try:
        px = pick_close(raw, t)
        rr, mm = jdk_components(px, bench_px)
        rs_ratio_map[t] = rr
        rs_mom_map[t] = mm
        tickers.append(t)
    except Exception:
        continue

if not tickers:
    st.error("No symbols available after data alignment.")
    st.stop()

# ===================== PLOT =====================
fig, ax = plt.subplots(figsize=(11, 7))

ax.axhline(100, linestyle=":", color="#666")
ax.axvline(100, linestyle=":", color="#666")
ax.set_xlim(94,106)
ax.set_ylim(94,106)

ax.set_xlabel("JdK RS-Ratio")
ax.set_ylabel("JdK RS-Momentum")

colors = {t: to_hex(plt.cm.tab20(i % 20)) for i,t in enumerate(tickers)}

end_idx = -1
start_idx = -tail_len

for t in tickers:
    rr = rs_ratio_map[t].iloc[start_idx:end_idx]
    mm = rs_mom_map[t].iloc[start_idx:end_idx]

    if len(rr) < 2:
        continue

    ax.plot(rr, mm, color=colors[t], alpha=0.6)

    last_rr = rr.iloc[-1]
    last_mm = mm.iloc[-1]

    sc = ax.scatter(
        last_rr,
        last_mm,
        s=90,
        color=colors[t],
        edgecolor="black",
        zorder=5
    )

    sc._rrg_meta = {
        "name": META[t]["name"],
        "symbol": t,
        "industry": META[t]["industry"],
        "rs_ratio": last_rr,
        "rs_mom": last_mm,
        "status": get_status(last_rr, last_mm),
        "price": pick_close(raw, t).iloc[-1]
    }

    if show_labels:
        ax.annotate(
            META[t]["name"],
            (last_rr, last_mm),
            xytext=(6,6),
            textcoords="offset points",
            fontsize=9
        )

# ===================== HOVER =====================
cursor = mplcursors.cursor(ax.collections, hover=True)

@cursor.connect("add")
def on_add(sel):
    meta = getattr(sel.artist, "_rrg_meta", None)
    if not meta:
        sel.annotation.set_visible(False)
        return

    sel.annotation.set_text(
        f"Name      : {meta['name']}\n"
        f"Symbol    : {meta['symbol']}\n"
        f"Industry  : {meta['industry']}\n"
        f"Status    : {meta['status']}\n"
        f"RS-Ratio  : {meta['rs_ratio']:.2f}\n"
        f"RS-Mom    : {meta['rs_mom']:.2f}\n"
        f"Price     : {meta['price']:.2f}"
    )
    sel.annotation.get_bbox_patch().set(fc="#111", alpha=0.9)
    sel.annotation.get_text().set_color("white")

st.pyplot(fig, width="stretch")

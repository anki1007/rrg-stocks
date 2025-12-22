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

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None
import streamlit.components.v1 as components

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# -------------------- CONFIG --------------------
GITHUB_USER = "anki1007"
GITHUB_REPO = "rrg-stocks"
GITHUB_BRANCH = "main"
GITHUB_TICKER_DIR = "ticker"
CSV_BASENAME = "niftyindices.csv"
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_TICKER_DIR}/"

WINDOW = 14
DEFAULT_TAIL = 8
PERIOD_MAP = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y": "3y", "5Y": "5y", "10Y": "10y"}
TF_LABELS = ["Daily", "Weekly", "Monthly"]
TF_TO_INTERVAL = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
BENCH_CHOICES = {"Nifty 500": "^CRSLDX", "Nifty 200": "^CNX200", "Nifty 50": "^NSEI"}

IST_TZ = "Asia/Kolkata"
BAR_CUTOFF_HOUR = 16
CACHE_DIR = pathlib.Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# -------------------- PAGE --------------------
st.set_page_config(page_title="Relative Rotation Graphs – Indices", layout="wide")

# -------------------- HELPERS --------------------
def _normalize_cols(cols):
    return {c: c.strip().lower().replace(" ", "") for c in cols}

def _to_yahoo_symbol(s):
    s = str(s).strip().upper()
    return s if s.startswith("^") else "^" + s

@st.cache_data(ttl=600)
def load_universe_from_github_csv(basename):
    df = pd.read_csv(RAW_BASE + basename)
    cols = _normalize_cols(df.columns)
    sym = next(c for c,k in cols.items() if k == "symbol")
    name = next(c for c,k in cols.items() if k in ("companyname","name"))
    ind = next((c for c,k in cols.items() if k=="industry"), None)
    if not ind:
        df["Industry"] = "-"
        ind = "Industry"

    df = df[[sym,name,ind]].drop_duplicates()
    df.columns = ["Symbol","Name","Industry"]
    df["Yahoo"] = df["Symbol"].apply(_to_yahoo_symbol)

    meta = {
        r["Yahoo"]: {
            "name": r["Name"],
            "industry": r["Industry"]
        } for _,r in df.iterrows()
    }
    return df["Yahoo"].tolist(), meta

def jdk_components(price, bench, win=14):
    df = pd.concat([price, bench], axis=1).dropna()
    rs = 100 * (df.iloc[:,0]/df.iloc[:,1])
    m = rs.rolling(win).mean()
    s = rs.rolling(win).std(ddof=0).replace(0,1e-9)
    rs_ratio = (100 + (rs-m)/s).dropna()
    roc = rs_ratio.pct_change()*100
    m2 = roc.rolling(win).mean()
    s2 = roc.rolling(win).std(ddof=0).replace(0,1e-9)
    rs_mom = (101 + (roc-m2)/s2).dropna()
    ix = rs_ratio.index.intersection(rs_mom.index)
    return rs_ratio[ix], rs_mom[ix]

def get_status(x,y):
    if x<=100 and y<=100: return "Lagging"
    if x>=100 and y>=100: return "Leading"
    if x<=100 and y>=100: return "Improving"
    return "Weakening"

def status_bg(x,y):
    return {
        "Lagging":"#e06a6a",
        "Leading":"#3fa46a",
        "Improving":"#5d86d1",
        "Weakening":"#e2d06b"
    }[get_status(x,y)]

# -------------------- LOAD DATA --------------------
UNIVERSE, META = load_universe_from_github_csv(CSV_BASENAME)

bench_label = st.sidebar.selectbox("Benchmark", list(BENCH_CHOICES))
interval_label = st.sidebar.selectbox("TF", TF_LABELS, index=1)
interval = TF_TO_INTERVAL[interval_label]
period = PERIOD_MAP[st.sidebar.selectbox("Period", list(PERIOD_MAP), index=1)]

raw = yf.download(UNIVERSE + [BENCH_CHOICES[bench_label]],
                  period=period, interval=interval, auto_adjust=True, progress=False)

bench = raw[BENCH_CHOICES[bench_label]]["Close"]

rs_ratio_map, rs_mom_map, prices = {}, {}, {}

for t in UNIVERSE:
    s = raw[t]["Close"].dropna()
    rr, mm = jdk_components(s, bench, WINDOW)
    rs_ratio_map[t] = rr
    rs_mom_map[t] = mm
    prices[t] = s

idx = bench.index
end_idx = len(idx)-1
start_idx = max(end_idx-DEFAULT_TAIL,0)

# -------------------- RRG PLOT --------------------
fig, ax = plt.subplots(figsize=(11,7))
ax.axhline(100, ls=":")
ax.axvline(100, ls=":")
ax.set_xlim(94,106)
ax.set_ylim(94,106)

annot = ax.annotate("", xy=(0,0), xytext=(10,10),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="black", alpha=0.8),
                    color="white")
annot.set_visible(False)

for t in UNIVERSE:
    rr = rs_ratio_map[t].iloc[start_idx:end_idx+1]
    mm = rs_mom_map[t].iloc[start_idx:end_idx+1]
    name = META[t]["name"]

    rr_last, mm_last = rr.iloc[-1], mm.iloc[-1]
    price = prices[t].iloc[end_idx]
    chg = (price/prices[t].iloc[start_idx]-1)*100
    ms = np.hypot(rr_last-100, mm_last-100)

    sc = ax.scatter(rr, mm, s=60)
    sc.meta = dict(name=name, rr=rr_last, mm=mm_last, price=price, chg=chg, ms=ms)

def hover(event):
    for c in ax.collections:
        cont,_ = c.contains(event)
        if cont:
            m=c.meta
            annot.xy=(event.xdata,event.ydata)
            annot.set_text(
                f"{m['name']}\n"
                f"Momentum Score: {m['ms']:.2f}\n"
                f"RS Ratio: {m['rr']:.2f}\n"
                f"RS Momentum: {m['mm']:.2f}\n"
                f"Price: {m['price']:.2f}\n"
                f"Change %: {m['chg']:.2f}"
            )
            annot.set_visible(True)
            fig.canvas.draw_idle()
            return
    annot.set_visible(False)
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("motion_notify_event", hover)
st.pyplot(fig, use_container_width=True)

# -------------------- TABLE CONTROLS --------------------
st.subheader("Table")

flt = st.text_input("Filter (Name / Industry)")
stat = st.multiselect("Status", ["Leading","Lagging","Improving","Weakening"],
                      default=["Leading","Lagging","Improving","Weakening"])
sort = st.selectbox("Sort by", ["RS-Ratio","RS-Momentum","Change %","Price"])

rows=[]
for t in UNIVERSE:
    rr = rs_ratio_map[t].iloc[end_idx]
    mm = rs_mom_map[t].iloc[end_idx]
    status = get_status(rr,mm)
    if status not in stat: continue
    if flt.lower() not in META[t]["name"].lower(): continue

    price = prices[t].iloc[end_idx]
    chg = (price/prices[t].iloc[start_idx]-1)*100

    rows.append({
        "Name": META[t]["name"],
        "Industry": META[t]["industry"],
        "Status": status,
        "RS-Ratio": rr,
        "RS-Momentum": mm,
        "Price": price,
        "Change %": chg
    })

df = pd.DataFrame(rows).sort_values(sort, ascending=False)

with st.expander("Table", expanded=True):
    st.dataframe(df, use_container_width=True)

# ===================== END =====================

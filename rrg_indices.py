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
import mplcursors

# ===================== CONFIG =====================
GITHUB_USER = "anki1007"
GITHUB_REPO = "rrg-stocks"
GITHUB_BRANCH = "main"
CSV_BASENAME = "niftyindices.csv"

RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/"

DEFAULT_TF = "Weekly"
WINDOW = 14
DEFAULT_TAIL = 8

PERIOD_MAP = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y": "3y", "5Y": "5y", "10Y": "10y"}
TF_LABELS = ["Daily", "Weekly", "Monthly"]
TF_TO_INTERVAL = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}

BENCH_CHOICES = {
    "Nifty 500": "^CRSLDX",
    "Nifty 200": "^CNX200",
    "Nifty 50": "^NSEI"
}

IST_TZ = "Asia/Kolkata"
BAR_CUTOFF_HOUR = 16
NET_TIME_MAX_AGE = 300

CACHE_DIR = pathlib.Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ===================== MATPLOTLIB =====================
mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["font.size"] = 15
mpl.rcParams["font.sans-serif"] = ["Inter", "Segoe UI", "DejaVu Sans", "Arial"]
mpl.rcParams["axes.grid"] = False

# ===================== STREAMLIT PAGE =====================
st.set_page_config(page_title="Relative Rotation Graphs – Indices", layout="wide")

# ===================== CSS + DATATABLES =====================
st.markdown("""
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>

<style>
html, body, .stApp { background:#0b0e13; color:#e6eaee; }
.rrg-wrap { max-height: calc(100vh - 260px); overflow:auto; }
.rrg-table { width:100%; border-collapse:collapse; }
.rrg-table th { position:sticky; top:0; background:#121823; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div style="font-size:36px;font-weight:800;">Relative Rotation Graphs – Indices</div>', unsafe_allow_html=True)

# ===================== HELPERS =====================
def _normalize_cols(cols):
    return {c: c.strip().lower().replace(" ", "").replace("_", "") for c in cols}

def _to_yahoo_symbol(raw):
    s = str(raw).strip().upper()
    if s.endswith(".NS") or s.startswith("^"):
        return s
    return "^" + s

@st.cache_data(ttl=600)
def load_universe_from_github_csv(basename, cache_bust):
    df = pd.read_csv(RAW_BASE + basename)
    mapping = _normalize_cols(df.columns)
    sym_col = next(c for c,k in mapping.items() if k in ("symbol","ticker"))
    name_col = next((c for c,k in mapping.items() if k in ("companyname","name")), sym_col)
    ind_col = next((c for c,k in mapping.items() if k in ("industry","sector")), None)
    if not ind_col:
        df["Industry"] = "-"
        ind_col = "Industry"

    sel = df[[sym_col,name_col,ind_col]].copy()
    sel.columns = ["Symbol","Name","Industry"]
    sel["Yahoo"] = sel["Symbol"].apply(_to_yahoo_symbol)

    universe = sel["Yahoo"].tolist()
    meta = {r["Yahoo"]:{
        "name":r["Name"], "industry":r["Industry"]
    } for _,r in sel.iterrows()}
    return universe, meta

def pick_close(df, symbol):
    if isinstance(df.columns, pd.MultiIndex):
        return df[(symbol,"Close")].dropna()
    return df["Close"].dropna()

def jdk_components(price, bench, win=14):
    rs = 100 * (price / bench)
    m = rs.rolling(win).mean()
    s = rs.rolling(win).std().replace(0,np.nan)
    rr = 100 + (rs - m) / s
    mom = 101 + rr.pct_change().rolling(win).mean()
    return rr.dropna(), mom.dropna()

def quadrant(rr, mm):
    if rr>=100 and mm>=100: return "Leading"
    if rr<100 and mm>=100: return "Improving"
    if rr<100 and mm<100: return "Lagging"
    return "Weakening"

# ===================== CONTROLS =====================
st.sidebar.header("Controls")
bench_label = st.sidebar.selectbox("Benchmark", BENCH_CHOICES.keys())
interval_label = st.sidebar.selectbox("Timeframe", TF_LABELS, index=1)
interval = TF_TO_INTERVAL[interval_label]
period = PERIOD_MAP[st.sidebar.selectbox("Period", PERIOD_MAP.keys(), index=1)]
tail_len = st.sidebar.slider("Tail Length", 1, 20, DEFAULT_TAIL)

# ===================== DATA =====================
UNIVERSE, META = load_universe_from_github_csv(CSV_BASENAME, "x")
bench_symbol = BENCH_CHOICES[bench_label]

raw = yf.download(UNIVERSE + [bench_symbol], period=period, interval=interval, auto_adjust=True)
bench = pick_close(raw, bench_symbol)

rs_map, mm_map = {}, {}
for t in UNIVERSE:
    if t == bench_symbol: continue
    px = pick_close(raw, t)
    rr, mm = jdk_components(px, bench)
    if len(rr) > WINDOW:
        rs_map[t], mm_map[t] = rr, mm

idx = bench.index
end_idx = len(idx)-1
start_idx = max(end_idx-tail_len,0)

# ===================== PLOT =====================
fig, ax = plt.subplots(figsize=(10,7))
ax.axhline(100,ls=":",c="gray")
ax.axvline(100,ls=":",c="gray")
ax.set_xlim(94,106); ax.set_ylim(94,106)

scatters = []
for t in rs_map:
    rr = rs_map[t].iloc[start_idx:end_idx+1]
    mm = mm_map[t].iloc[start_idx:end_idx+1]

    ax.plot(rr,mm,alpha=0.6)
    ax.scatter(rr[:-1],mm[:-1],s=22)
    head = ax.scatter(rr.iloc[-1],mm.iloc[-1],s=140,marker=">",label=t)
    scatters.append((head,t))

cursor = mplcursors.cursor([s[0] for s in scatters], hover=True)

@cursor.connect("add")
def on_add(sel):
    sym = scatters[sel.index][1]
    rr = rs_map[sym].iloc[end_idx]
    mm = mm_map[sym].iloc[end_idx]
    px = pick_close(raw, sym).iloc[end_idx]
    chg = (pick_close(raw, sym).iloc[end_idx] /
           pick_close(raw, sym).iloc[start_idx] - 1) * 100
    power = np.hypot(rr-100, mm-100)

    sel.annotation.set_text(
        f"{META[sym]['name']}\n"
        f"RRG Power: {power:.2f}\n"
        f"RS-Ratio: {rr:.2f}\n"
        f"RS-Momentum: {mm:.2f}\n"
        f"Price: {px:.2f}\n"
        f"Change %: {chg:.2f}\n"
        f"Strength: {quadrant(rr,mm)}"
    )

st.pyplot(fig)

# ===================== TABLE =====================
rows=[]
for t in rs_map:
    rr = rs_map[t].iloc[end_idx]
    mm = mm_map[t].iloc[end_idx]
    rows.append({
        "Name": META[t]["name"],
        "Industry": META[t]["industry"],
        "RS-Ratio": round(rr,2),
        "RS-Momentum": round(mm,2),
        "Strength": quadrant(rr,mm)
    })

df = pd.DataFrame(rows)

st.markdown("""
<table id="rrgTable" class="rrg-table">
<thead><tr>
<th>Name</th><th>Industry</th><th>RS-Ratio</th><th>RS-Momentum</th><th>Strength</th>
</tr></thead>
<tbody>
""" +
"".join([f"<tr><td>{r.Name}</td><td>{r.Industry}</td><td>{r['RS-Ratio']}</td><td>{r['RS-Momentum']}</td><td>{r.Strength}</td></tr>"
         for _,r in df.iterrows()]) +
"</tbody></table>", unsafe_allow_html=True)

st.markdown("""
<script>
$(document).ready(function() {
  $('#rrgTable').DataTable({paging:false,info:false});
});
</script>
""", unsafe_allow_html=True)

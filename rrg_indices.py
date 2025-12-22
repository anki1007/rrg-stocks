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

# -------------------- Config --------------------
GITHUB_USER = "anki1007"
GITHUB_REPO = "rrg-stocks"
GITHUB_BRANCH = "main"
CSV_BASENAME = "niftyindices.csv"

RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/"

DEFAULT_TF = "Weekly"
WINDOW = 14
DEFAULT_TAIL = 8

PERIOD_MAP = {
    "6M": "6mo",
    "1Y": "1y",
    "2Y": "2y",
    "3Y": "3y",
    "5Y": "5y",
    "10Y": "10y",
}

TF_LABELS = ["Daily", "Weekly", "Monthly"]
TF_TO_INTERVAL = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}

BENCH_CHOICES = {
    "Nifty 50": "^NSEI",
    "Nifty 200": "^CNX200",
    "Nifty 500": "^CRSLDX",  # unreliable → safely ignored if missing
}

CACHE_DIR = pathlib.Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# -------------------- Matplotlib --------------------
mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["font.size"] = 15
mpl.rcParams["font.sans-serif"] = ["Inter", "Segoe UI", "DejaVu Sans", "Arial"]
mpl.rcParams["axes.grid"] = False

# -------------------- Streamlit --------------------
st.set_page_config(page_title="Relative Rotation Graphs – Indices", layout="wide")

# -------------------- CSS + DataTables --------------------
st.markdown("""
<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>
""", unsafe_allow_html=True)

st.markdown("<h1>Relative Rotation Graphs – Indices</h1>", unsafe_allow_html=True)

# -------------------- Helpers --------------------
def _normalize_cols(cols):
    return {c: c.strip().lower().replace(" ", "").replace("_", "") for c in cols}

def _to_yahoo_symbol(raw):
    s = str(raw).strip().upper()
    if s.endswith(".NS") or s.startswith("^"):
        return s
    return "^" + s

def pick_close(df, symbol: str) -> pd.Series:
    """
    SAFE: returns empty Series if price not found
    """
    if not isinstance(df, pd.DataFrame):
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        if (symbol, "Close") in df.columns:
            return df[(symbol, "Close")].dropna()
        if (symbol, "Adj Close") in df.columns:
            return df[(symbol, "Adj Close")].dropna()
        return pd.Series(dtype=float)

    for col in ("Close", "Adj Close"):
        if col in df.columns:
            return df[col].dropna()

    return pd.Series(dtype=float)

def jdk_components(price, bench, win=14):
    df = pd.concat([price, bench], axis=1).dropna()
    if df.empty:
        return None, None

    rs = 100 * (df.iloc[:, 0] / df.iloc[:, 1])
    m = rs.rolling(win).mean()
    s = rs.rolling(win).std().replace(0, np.nan)

    rr = 100 + (rs - m) / s
    mom = 101 + rr.pct_change().rolling(win).mean()

    return rr.dropna(), mom.dropna()

def quadrant(rr, mm):
    if rr >= 100 and mm >= 100:
        return "Leading"
    if rr < 100 and mm >= 100:
        return "Improving"
    if rr < 100 and mm < 100:
        return "Lagging"
    return "Weakening"

# -------------------- Load Universe --------------------
@st.cache_data(ttl=600)
def load_universe():
    df = pd.read_csv(RAW_BASE + CSV_BASENAME)
    mapping = _normalize_cols(df.columns)

    sym_col = next(c for c, k in mapping.items() if k in ("symbol", "ticker"))
    name_col = next((c for c, k in mapping.items() if k in ("companyname", "name")), sym_col)
    ind_col = next((c for c, k in mapping.items() if k in ("industry", "sector")), None)

    if ind_col is None:
        df["Industry"] = "-"
        ind_col = "Industry"

    df = df[[sym_col, name_col, ind_col]].copy()
    df.columns = ["Symbol", "Name", "Industry"]
    df["Yahoo"] = df["Symbol"].apply(_to_yahoo_symbol)

    universe = df["Yahoo"].tolist()
    meta = {
        r.Yahoo: {"name": r.Name, "industry": r.Industry}
        for _, r in df.iterrows()
    }
    return universe, meta

# -------------------- Controls --------------------
st.sidebar.header("Controls")
bench_label = st.sidebar.selectbox("Benchmark", list(BENCH_CHOICES.keys()))
tf_label = st.sidebar.selectbox("Timeframe", TF_LABELS, index=1)
interval = TF_TO_INTERVAL[tf_label]
period = PERIOD_MAP[st.sidebar.selectbox("Period", list(PERIOD_MAP.keys()), index=1)]
tail_len = st.sidebar.slider("Trail Length", 1, 20, DEFAULT_TAIL)

# -------------------- Data --------------------
UNIVERSE, META = load_universe()
bench_symbol = BENCH_CHOICES[bench_label]

raw = yf.download(
    list(set(UNIVERSE + [bench_symbol])),
    period=period,
    interval=interval,
    auto_adjust=True,
    progress=False,
    threads=True,
)

benchmark = pick_close(raw, bench_symbol)

if benchmark.empty:
    st.error(f"Benchmark {bench_symbol} has no data. Please select another benchmark.")
    st.stop()

idx = benchmark.index
end_idx = len(idx) - 1
start_idx = max(end_idx - tail_len, 0)

rs_map, mm_map = {}, {}

for sym in UNIVERSE:
    if sym == bench_symbol:
        continue

    px = pick_close(raw, sym)
    if px.empty:
        continue

    rr, mm = jdk_components(px, benchmark, WINDOW)
    if rr is None or mm is None or len(rr) < WINDOW:
        continue

    rs_map[sym] = rr
    mm_map[sym] = mm

if not rs_map:
    st.warning("No symbols have sufficient data.")
    st.stop()

# -------------------- Plot --------------------
fig, ax = plt.subplots(figsize=(11, 7))
ax.axhline(100, linestyle=":", color="gray")
ax.axvline(100, linestyle=":", color="gray")
ax.set_xlim(94, 106)
ax.set_ylim(94, 106)
ax.set_title("Relative Rotation Graph")

scatter_refs = []

for sym in rs_map:
    rr = rs_map[sym].iloc[start_idx : end_idx + 1]
    mm = mm_map[sym].iloc[start_idx : end_idx + 1]

    ax.plot(rr, mm, alpha=0.6)
    ax.scatter(rr[:-1], mm[:-1], s=22)
    head = ax.scatter(rr.iloc[-1], mm.iloc[-1], s=140, marker=">")
    scatter_refs.append((head, sym))

cursor = mplcursors.cursor([h[0] for h in scatter_refs], hover=True)

@cursor.connect("add")
def on_add(sel):
    sym = scatter_refs[sel.index][1]
    rr = rs_map[sym].iloc[end_idx]
    mm = mm_map[sym].iloc[end_idx]

    px = pick_close(raw, sym)
    price = px.iloc[end_idx]
    chg = (px.iloc[end_idx] / px.iloc[start_idx] - 1) * 100
    power = np.hypot(rr - 100, mm - 100)

    sel.annotation.set_text(
        f"{META[sym]['name']}\n"
        f"RRG Power: {power:.2f}\n"
        f"RS-Ratio: {rr:.2f}\n"
        f"RS-Momentum: {mm:.2f}\n"
        f"Price: {price:.2f}\n"
        f"Change %: {chg:.2f}\n"
        f"Strength: {quadrant(rr, mm)}"
    )

st.pyplot(fig, use_container_width=True)

# -------------------- Table --------------------
rows = []
for sym in rs_map:
    rr = rs_map[sym].iloc[end_idx]
    mm = mm_map[sym].iloc[end_idx]
    rows.append({
        "Name": META[sym]["name"],
        "Industry": META[sym]["industry"],
        "RS-Ratio": round(rr, 2),
        "RS-Momentum": round(mm, 2),
        "Strength": quadrant(rr, mm),
    })

df = pd.DataFrame(rows)

st.markdown("""
<table id="rrgTable" class="display">
<thead>
<tr>
<th>Name</th><th>Industry</th><th>RS-Ratio</th><th>RS-Momentum</th><th>Strength</th>
</tr>
</thead>
<tbody>
""" +
"".join(
    f"<tr><td>{r.Name}</td><td>{r.Industry}</td><td>{r['RS-Ratio']}</td><td>{r['RS-Momentum']}</td><td>{r.Strength}</td></tr>"
    for _, r in df.iterrows()
) +
"</tbody></table>", unsafe_allow_html=True)

st.markdown("""
<script>
$(document).ready(function() {
    $('#rrgTable').DataTable({ paging:false, info:false });
});
</script>
""", unsafe_allow_html=True)

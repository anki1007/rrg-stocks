# ============================================================
# RRG STOCKS — FINAL CLOUD-SAFE VERSION (YAHOO ONLY)
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

# ================= CONFIG =================

GITHUB_RAW = "https://raw.githubusercontent.com/anki1007/rrg-stocks/main/ticker/"

UNIVERSES = {
    "Nifty 50": "nifty50.csv",
    "Nifty 200": "nifty200.csv",
    "Nifty 500": "nifty500.csv",
    "Nifty Midcap 150": "niftymidcap150.csv",
    "Nifty Smallcap 250": "niftysmallcap250.csv",
    "Nifty Midsmallcap 400": "niftymidsmallcap400.csv",
}

BENCHMARKS = {
    "Nifty 50": "^NSEI",
    "Nifty 200": "^CNX200",
    "Nifty 500": "^CRSLDX",
}

TF_MAP = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
PERIOD_MAP = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y": "3y"}

WINDOW = 14
DEFAULT_TAIL = 8

# ================= PAGE =================

st.set_page_config(page_title="RRG Stocks Terminal", layout="wide")

# ================= CSS =================

st.markdown("""
<style>
.rrg-wrap { max-height:520px; overflow:auto; }
.rrg-table { width:100%; border-collapse:collapse; }
.rrg-table th { position:sticky; top:0; background:#111827; color:#e5e7eb; padding:8px; }
.rrg-table td { padding:6px 8px; }
</style>
""", unsafe_allow_html=True)

# ================= HELPERS =================

def display_symbol(sym):
    return sym.replace(".NS", "")

def tradingview_link(sym):
    return f"https://www.tradingview.com/chart/?symbol=NSE:{display_symbol(sym)}"

def jdk_rrg(price, benchmark):
    rs = 100 * price / benchmark
    rr = 100 + (rs - rs.rolling(WINDOW).mean()) / rs.rolling(WINDOW).std()
    roc = rr.pct_change(fill_method=None) * 100
    mm = 101 + (roc - roc.rolling(WINDOW).mean()) / roc.rolling(WINDOW).std()
    idx = rr.index.intersection(mm.index)
    return rr.loc[idx], mm.loc[idx]

@st.cache_data(ttl=1800)
def download_prices(symbols, benchmark, period, interval):
    data = {}

    bench = yf.download(
        benchmark,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if bench.empty:
        return {}

    data[benchmark] = bench

    for s in symbols:
        try:
            df = yf.download(
                s,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False
            )
            if not df.empty:
                data[s] = df
        except Exception:
            pass

    return data

# ================= SIDEBAR =================

st.sidebar.header("RRG Controls")

universe_name = st.sidebar.selectbox("Universe", list(UNIVERSES.keys()))
benchmark_name = st.sidebar.selectbox("Benchmark", list(BENCHMARKS.keys()))
tf_label = st.sidebar.selectbox("Timeframe", list(TF_MAP.keys()))
period_label = st.sidebar.selectbox("Period", list(PERIOD_MAP.keys()))
tail_len = st.sidebar.slider("Trail Length", 1, 20, DEFAULT_TAIL)

# ================= LOAD SYMBOLS =================

df = pd.read_csv(GITHUB_RAW + UNIVERSES[universe_name])
df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
symbols = df["Symbol"].tolist()

benchmark = BENCHMARKS[benchmark_name]

# ================= DOWNLOAD =================

raw = download_prices(
    symbols,
    benchmark,
    PERIOD_MAP[period_label],
    TF_MAP[tf_label]
)

if benchmark not in raw:
    st.error("Benchmark data unavailable.")
    st.stop()

bench_px = raw[benchmark]["Close"].dropna()

# ================= RRG COMPUTE =================

rs_ratio, rs_mom = {}, {}

for s in symbols:
    if s not in raw:
        continue

    px = raw[s]["Close"].dropna()
    if px.empty:
        continue

    rr, mm = jdk_rrg(px, bench_px)
    if len(rr) < 20:
        continue

    rs_ratio[s] = rr
    rs_mom[s] = mm

# ================= SAFE RANKING (FINAL FIX) =================

rank_metric = {}

for s in rs_ratio:
    try:
        rr_val = float(rs_ratio[s].iloc[-1])
        mm_val = float(rs_mom[s].iloc[-1])
        if np.isnan(rr_val) or np.isnan(mm_val):
            continue
        rank_metric[s] = np.hypot(rr_val - 100.0, mm_val - 100.0)
    except Exception:
        continue

ranked = sorted(rank_metric, key=lambda x: rank_metric[x], reverse=True)
rank_dict = {s: i + 1 for i, s in enumerate(ranked)}

if not ranked:
    st.warning("No valid stocks available.")
    st.stop()

start_idx = max(len(rs_ratio[ranked[0]]) - tail_len, 0)

# ================= PLOT =================

fig, ax = plt.subplots(figsize=(12, 8))

ax.axhline(100, ls=":", c="gray")
ax.axvline(100, ls=":", c="gray")
ax.set_xlim(94, 106)
ax.set_ylim(94, 106)

for s in ranked:
    rr = rs_ratio[s].iloc[start_idx:]
    mm = rs_mom[s].iloc[start_idx:]
    ax.plot(rr, mm, alpha=0.6)
    ax.scatter(rr.iloc[-1], mm.iloc[-1], s=60)

st.pyplot(fig, width="stretch")

# ================= TABLE =================

rows = ""
for s in ranked:
    rows += f"""
    <tr>
      <td>{rank_dict[s]}</td>
      <td><a href="{tradingview_link(s)}" target="_blank">{display_symbol(s)}</a></td>
      <td>{rs_ratio[s].iloc[-1]:.2f}</td>
      <td>{rs_mom[s].iloc[-1]:.2f}</td>
    </tr>
    """

st.markdown(f"""
<div class="rrg-wrap">
<table class="rrg-table">
<thead><tr><th>Rank</th><th>Symbol</th><th>RS</th><th>Momentum</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>
""", unsafe_allow_html=True)

st.caption("RRG Stocks — FINAL, Yahoo-only, Cloud-stable")

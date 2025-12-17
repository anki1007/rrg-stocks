# ============================================================
# RRG STOCKS — FULL TERMINAL DASHBOARD (PRODUCTION READY)
# ============================================================

import time, io
import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
from urllib.parse import quote

# ============================================================
# CONFIG
# ============================================================

GITHUB_RAW = "https://raw.githubusercontent.com/anki1007/rrg-stocks/main/ticker/"

UNIVERSES = {
    "Nifty 50": "nifty50.csv",
    "Nifty 200": "nifty200.csv",
    "Nifty 500": "nifty500.csv",
    "Nifty Midcap 150": "niftymidcap150.csv",
    "Nifty Smallcap 250": "niftysmallcap250.csv",
    "Nifty Midsmallcap 400": "niftymidsmallcap400.csv",
}

BENCH_CHOICES = {
    "Nifty 50": "^NSEI",
    "Nifty 200": "^CNX200",
    "Nifty 500": "^CRSLDX",
}

TF_MAP = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
PERIOD_MAP = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y": "3y", "5Y": "5y", "10Y": "10y"}

WINDOW = 14
DEFAULT_TAIL = 8

# ============================================================
# PAGE
# ============================================================

st.set_page_config(page_title="RRG Stocks Terminal", layout="wide")

# ============================================================
# CSS
# ============================================================

st.markdown("""
<style>
.rrg-wrap { max-height: calc(100vh - 260px); overflow:auto; }
.rrg-table th { position:sticky; top:0; background:#111827; color:#e5e7eb; }
.blink-up { animation: blinkGreen 1.2s ease-out; }
.blink-down { animation: blinkRed 1.2s ease-out; }

@keyframes blinkGreen { from{background:#bbf7d0;} to{background:transparent;} }
@keyframes blinkRed { from{background:#fecaca;} to{background:transparent;} }

.rrg-legend {
  position: fixed; top: 72px; right: 20px;
  background: rgba(15,23,42,0.92); color:#e5e7eb;
  border-radius:10px; padding:12px 14px;
  font-size:13px; z-index:999;
}
.legend-row { display:flex; gap:8px; margin:4px 0; align-items:center; }
.legend-box { width:18px; height:10px; border-radius:3px; }
.legend-green { background: linear-gradient(to right,#166534,#bbf7d0); }
.legend-blue { background: linear-gradient(to right,#1e40af,#dbeafe); }
.legend-yellow { background: linear-gradient(to right,#ca8a04,#fef9c3); }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================

def display_symbol(sym: str) -> str:
    return sym.replace(".NS", "")

def tradingview_link(sym: str) -> str:
    base = display_symbol(sym).replace("-", "_")
    return f"https://www.tradingview.com/chart/?symbol=NSE:{base}"

def rank_gradient(rank: int) -> str:
    if rank <= 30:
        return f"rgb({22 + rank*2},163,74)"
    if rank <= 60:
        return f"rgb(37,{99 + (rank-30)*2},235)"
    if rank <= 90:
        return f"rgb(250,{204 - (rank-60)*2},21)"
    return "transparent"

def jdk_components(price, bench):
    rs = 100 * price / bench
    rr = 100 + (rs - rs.rolling(WINDOW).mean()) / rs.rolling(WINDOW).std()
    roc = rr.pct_change() * 100
    mm = 101 + (roc - roc.rolling(WINDOW).mean()) / roc.rolling(WINDOW).std()
    ix = rr.index.intersection(mm.index)
    return rr.loc[ix], mm.loc[ix]

def export_figure(fig, fmt):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("RRG Controls")

universe_name = st.sidebar.selectbox("Universe", list(UNIVERSES.keys()))
csv_file = UNIVERSES[universe_name]

bench_label = st.sidebar.selectbox("Benchmark", list(BENCH_CHOICES.keys()))
tf_label = st.sidebar.selectbox("Timeframe", list(TF_MAP.keys()))
period_label = st.sidebar.selectbox("Period", list(PERIOD_MAP.keys()))
tail_len = st.sidebar.slider("Trail Length", 1, 20, DEFAULT_TAIL)

live_mode = st.sidebar.toggle("Intraday Auto-Refresh", False)

# ============================================================
# LOAD UNIVERSE
# ============================================================

df = pd.read_csv(GITHUB_RAW + csv_file)
df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
symbols = df["Symbol"].tolist()

# ============================================================
# DATA
# ============================================================

bench = BENCH_CHOICES[bench_label]

raw = yf.download(
    symbols + [bench],
    period=PERIOD_MAP[period_label],
    interval=TF_MAP[tf_label],
    group_by="ticker",
    auto_adjust=True,
    progress=False
)

bench_px = raw[bench]["Close"]

rs_ratio, rs_mom = {}, {}
for s in symbols:
    try:
        rr, mm = jdk_components(raw[s]["Close"], bench_px)
        if len(rr) > 20:
            rs_ratio[s] = rr
            rs_mom[s] = mm
    except Exception:
        continue

# ============================================================
# RANKING
# ============================================================

end_idx = -1
start_idx = max(len(bench_px) + end_idx - tail_len, 0)

ranked = sorted(
    rs_ratio.keys(),
    key=lambda s: np.hypot(
        rs_ratio[s].iloc[end_idx] - 100,
        rs_mom[s].iloc[end_idx] - 100
    ),
    reverse=True
)

rank_dict = {s: i+1 for i, s in enumerate(ranked)}

if "prev_ranks" not in st.session_state:
    st.session_state.prev_ranks = {}

rank_change = {
    s: st.session_state.prev_ranks.get(s, rank_dict[s]) - rank_dict[s]
    for s in ranked
}
st.session_state.prev_ranks = rank_dict.copy()

# ============================================================
# HEATMAP LEGEND
# ============================================================

st.markdown("""
<div class="rrg-legend">
  <b>Rank Heatmap</b>
  <div class="legend-row"><div class="legend-box legend-green"></div>Top 30</div>
  <div class="legend-row"><div class="legend-box legend-blue"></div>31–60</div>
  <div class="legend-row"><div class="legend-box legend-yellow"></div>61–90</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LAYOUT
# ============================================================

plot_col, rank_col = st.columns([4.5, 1.8])

with plot_col:
    fig, ax = plt.subplots(figsize=(10.6, 6.8))

    # Quadrants
    ax.fill_between([94,100],[100,100],[106,106], color="#c7d2fe", alpha=0.6)
    ax.fill_between([100,106],[100,100],[106,106], color="#bbf7d0", alpha=0.6)
    ax.fill_between([94,100],[94,94],[100,100], color="#fecaca", alpha=0.6)
    ax.fill_between([100,106],[94,94],[100,100], color="#fef9c3", alpha=0.6)

    ax.text(95,105,"Improving", weight="bold")
    ax.text(105,105,"Leading", ha="right", weight="bold")
    ax.text(95,95,"Lagging", weight="bold")
    ax.text(105,95,"Weakening", ha="right", weight="bold")

    ax.axhline(100, ls=":", c="gray")
    ax.axvline(100, ls=":", c="gray")
    ax.set_xlim(94,106)
    ax.set_ylim(94,106)
    ax.set_xlabel("RS-Ratio")
    ax.set_ylabel("RS-Momentum")

    for s in ranked:
        rr = rs_ratio[s].iloc[start_idx:end_idx+1]
        mm = rs_mom[s].iloc[start_idx:end_idx+1]
        if len(rr)==0 or len(mm)==0:
            continue
        ax.plot(rr, mm, alpha=0.6)
        ax.scatter(rr.iloc[-1], mm.iloc[-1], s=70)
        if rank_dict[s] <= 15:
            ax.annotate(display_symbol(s),
                        (rr.iloc[-1], mm.iloc[-1]),
                        xytext=(6,6), textcoords="offset points")

    st.pyplot(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇ PNG", export_figure(fig, "png"), "rrg.png")
    with c2:
        st.download_button("⬇ PDF", export_figure(fig, "pdf"), "rrg.pdf")

with rank_col:
    st.markdown("### Ranking")
    for s in ranked[:30]:
        st.markdown(f"**{rank_dict[s]}. {display_symbol(s)}**")

# ============================================================
# TABLE
# ============================================================

rows = []
for s in ranked:
    r = rank_dict[s]
    bg = rank_gradient(r)
    blink = "blink-up" if rank_change[s] > 0 else "blink-down" if rank_change[s] < 0 else ""
    rows.append(f"""
    <tr class="{blink}" style="background:{bg}">
      <td>{r}</td>
      <td><a href="{tradingview_link(s)}" target="_blank">{display_symbol(s)}</a></td>
      <td>{rs_ratio[s].iloc[end_idx]:.2f}</td>
      <td>{rs_mom[s].iloc[end_idx]:.2f}</td>
    </tr>
    """)

table_html = """
<div class="rrg-wrap">
<table class="rrg-table">
<tr><th>Rank</th><th>Symbol</th><th>RS-Ratio</th><th>RS-Momentum</th></tr>
""" + "".join(rows) + "</table></div>"

with st.expander("Table", expanded=True):
    st.markdown(table_html, unsafe_allow_html=True)

st.caption("RRG Stocks Terminal — fully compiled, CSV-authoritative, production ready")

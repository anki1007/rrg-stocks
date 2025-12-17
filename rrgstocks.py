# ============================================================
# RRG STOCKS — FINAL STABLE VERSION (YAHOO FINANCE ONLY)
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

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

BENCHMARKS = {
    "Nifty 50": "^NSEI",
    "Nifty 500": "^CRSLDX",
}

TF_MAP = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
PERIOD_MAP = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y": "3y"}

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
.rrg-wrap { max-height: 520px; overflow:auto; }
.rrg-table { width:100%; border-collapse:collapse; }
.rrg-table th {
  position:sticky; top:0;
  background:#111827; color:#e5e7eb;
  padding:8px; text-align:left;
}
.rrg-table td { padding:6px 8px; }
.table-legend { display:flex; gap:14px; margin-bottom:8px; font-size:12px; }
.legend-box { width:14px; height:8px; border-radius:3px; display:inline-block; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HELPERS
# ============================================================

def display_symbol(sym):
    return sym.replace(".NS", "")

def tradingview_link(sym):
    return f"https://www.tradingview.com/chart/?symbol=NSE:{display_symbol(sym)}"

def rank_color(rank):
    if rank <= 30:
        return "#16a34a"
    if rank <= 60:
        return "#2563eb"
    if rank <= 90:
        return "#facc15"
    return "transparent"

def jdk_rrg(price, benchmark):
    rs = 100 * price / benchmark
    rr = 100 + (rs - rs.rolling(WINDOW).mean()) / rs.rolling(WINDOW).std()
    roc = rr.pct_change(fill_method=None) * 100
    mm = 101 + (roc - roc.rolling(WINDOW).mean()) / roc.rolling(WINDOW).std()
    idx = rr.index.intersection(mm.index)
    return rr.loc[idx], mm.loc[idx]

def export_figure(fig, fmt):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=200, bbox_inches="tight")
    buf.seek(0)
    return buf

@st.cache_data(ttl=1800)
def download_prices(symbols, benchmark, period, interval):
    data = {}
    failed = []

    bench_df = yf.download(
        benchmark,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False
    )

    if bench_df.empty:
        return {}, symbols

    data[benchmark] = bench_df

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
            failed.append(s)

    return data, failed

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("RRG Controls")

universe_name = st.sidebar.selectbox("Universe", list(UNIVERSES.keys())))
benchmark_name = st.sidebar.selectbox("Benchmark", list(BENCHMARKS.keys()))
tf_label = st.sidebar.selectbox("Timeframe", list(TF_MAP.keys()))
period_label = st.sidebar.selectbox("Period", list(PERIOD_MAP.keys()))
tail_len = st.sidebar.slider("Trail Length", 1, 20, DEFAULT_TAIL)

# ============================================================
# LOAD SYMBOLS
# ============================================================

df = pd.read_csv(GITHUB_RAW + UNIVERSES[universe_name])
df["Symbol"] = df["Symbol"].astype(str).str.upper().str.strip()
symbols = df["Symbol"].tolist()

benchmark = BENCHMARKS[benchmark_name]

# ============================================================
# DOWNLOAD DATA
# ============================================================

raw, failed_symbols = download_prices(
    symbols,
    benchmark,
    PERIOD_MAP[period_label],
    TF_MAP[tf_label]
)

if benchmark not in raw:
    st.error("Benchmark data not available from Yahoo.")
    st.stop()

bench_px = raw[benchmark]["Close"].dropna()

# ============================================================
# COMPUTE RRG
# ============================================================

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

# ============================================================
# SAFE RANKING (CRITICAL FIX)
# ============================================================

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

ranked = sorted(rank_metric.keys(), key=lambda s: rank_metric[s], reverse=True)
rank_dict = {s: i + 1 for i, s in enumerate(ranked)}

start_idx = max(len(rs_ratio[ranked[0]]) - tail_len, 0)

# ============================================================
# LAYOUT
# ============================================================

plot_col, rank_col = st.columns([4.5, 1.8])

# ================= CHART =================

with plot_col:
    with st.expander("🔍 Chart Controls", expanded=False):
        x_min, x_max = st.slider("RS-Ratio Range", 90.0, 110.0, (94.0, 106.0), 0.5)
        y_min, y_max = st.slider("RS-Momentum Range", 90.0, 110.0, (94.0, 106.0), 0.5)
        show_labels = st.checkbox("Show Top-15 Labels", True)

    with st.expander("📈 RRG Chart", expanded=True):
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.fill_between([x_min,100],[100,100],[y_max,y_max], color="#c7d2fe", alpha=0.6)
        ax.fill_between([100,x_max],[100,100],[y_max,y_max], color="#bbf7d0", alpha=0.6)
        ax.fill_between([x_min,100],[y_min,y_min],[100,100], color="#fecaca", alpha=0.6)
        ax.fill_between([100,x_max],[y_min,y_min],[100,100], color="#fef9c3", alpha=0.6)

        ax.text(x_min+1,y_max-1,"Improving",weight="bold")
        ax.text(x_max-1,y_max-1,"Leading",ha="right",weight="bold")
        ax.text(x_min+1,y_min+1,"Lagging",weight="bold")
        ax.text(x_max-1,y_min+1,"Weakening",ha="right",weight="bold")

        ax.axhline(100,ls=":",c="gray")
        ax.axvline(100,ls=":",c="gray")

        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
        ax.set_xlabel("RS-Ratio")
        ax.set_ylabel("RS-Momentum")

        for s in ranked:
            rr = rs_ratio[s].iloc[start_idx:]
            mm = rs_mom[s].iloc[start_idx:]
            if rr.empty or mm.empty:
                continue
            ax.plot(rr,mm,alpha=0.6)
            ax.scatter(rr.iloc[-1],mm.iloc[-1],s=60)
            if show_labels and rank_dict[s] <= 15:
                ax.annotate(display_symbol(s),(rr.iloc[-1],mm.iloc[-1]),
                            xytext=(6,6),textcoords="offset points",fontsize=9)

        st.pyplot(fig, width="stretch")

        c1,c2 = st.columns(2)
        with c1:
            st.download_button("⬇ PNG", export_figure(fig,"png"), "rrg.png")
        with c2:
            st.download_button("⬇ PDF", export_figure(fig,"pdf"), "rrg.pdf")

# ================= RANKING =================

with rank_col:
    st.markdown("### Ranking")
    for s in ranked[:30]:
        st.markdown(f"**{rank_dict[s]}. {display_symbol(s)}**")

# ================= TABLE =================

legend = """
<div class="table-legend">
  <div><span class="legend-box" style="background:#16a34a"></span> Top 30</div>
  <div><span class="legend-box" style="background:#2563eb"></span> 31–60</div>
  <div><span class="legend-box" style="background:#facc15"></span> 61–90</div>
</div>
"""

rows = ""
for s in ranked:
    r = rank_dict[s]
    rr_val = float(rs_ratio[s].iloc[-1])
    mm_val = float(rs_mom[s].iloc[-1])
    rows += f"""
    <tr style="background:{rank_color(r)}">
      <td>{r}</td>
      <td><a href="{tradingview_link(s)}" target="_blank">{display_symbol(s)}</a></td>
      <td>{rr_val:.2f}</td>
      <td>{mm_val:.2f}</td>
    </tr>
    """

with st.expander("Table", expanded=True):
    st.markdown(
        legend + f"""
        <div class="rrg-wrap">
        <table class="rrg-table">
        <thead><tr><th>Rank</th><th>Symbol</th><th>RS-Ratio</th><th>RS-Momentum</th></tr></thead>
        <tbody>{rows}</tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True
    )

if failed_symbols:
    st.sidebar.warning(f"{len(failed_symbols)} symbols skipped due to Yahoo limits.")

st.caption("RRG Stocks Terminal — FINAL, Yahoo-only, production stable")

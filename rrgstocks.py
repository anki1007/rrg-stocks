# ============================================================
# RRG STOCKS — FINAL PLOTLY VERSION (YAHOO FINANCE ONLY)
# ============================================================

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go

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
    "Nifty 200": "^CNX200",
    "Nifty 500": "^CRSLDX",
}

TF_MAP = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
PERIOD_MAP = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y": "3y"}

WINDOW = 14
DEFAULT_TAIL = 8

# ============================================================
# PAGE
# ============================================================

st.set_page_config(page_title="RRG Stocks Terminal (Plotly)", layout="wide")

# ============================================================
# CSS (TABLE ONLY)
# ============================================================

st.markdown("""
<style>
.rrg-wrap { max-height:520px; overflow:auto; }
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

def display_symbol(sym: str) -> str:
    return sym.replace(".NS", "")

def tradingview_link(sym: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol=NSE:{display_symbol(sym)}"

def rank_color(rank: int) -> str:
    if rank <= 30:
        return "#16a34a"
    if rank <= 60:
        return "#2563eb"
    if rank <= 90:
        return "#facc15"
    return "transparent"

def jdk_rrg(price: pd.Series, benchmark: pd.Series):
    rs = 100 * price / benchmark
    rr = 100 + (rs - rs.rolling(WINDOW).mean()) / rs.rolling(WINDOW).std()
    roc = rr.pct_change(fill_method=None) * 100
    mm = 101 + (roc - roc.rolling(WINDOW).mean()) / roc.rolling(WINDOW).std()
    idx = rr.index.intersection(mm.index)
    return rr.loc[idx], mm.loc[idx]

@st.cache_data(ttl=1800)
def download_prices(symbols, benchmark, period, interval):
    data = {}

    bench_df = yf.download(
        benchmark,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
        threads=False
    )
    if bench_df.empty:
        return {}

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
            pass

    return data

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("RRG Controls")

universe_name = st.sidebar.selectbox("Universe", list(UNIVERSES.keys()))
benchmark_name = st.sidebar.selectbox(
    "Benchmark",
    list(BENCHMARKS.keys()),
    index=list(BENCHMARKS.keys()).index(universe_name)
    if universe_name in BENCHMARKS else 0
)

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
# DOWNLOAD DATA (YAHOO)
# ============================================================

raw = download_prices(
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
# SAFE RANKING (ONLY PATH)
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

ranked = sorted(rank_metric, key=lambda x: rank_metric[x], reverse=True)
rank_dict = {s: i + 1 for i, s in enumerate(ranked)}

if not ranked:
    st.warning("No valid stocks available.")
    st.stop()

start_idx = max(len(rs_ratio[ranked[0]]) - tail_len, 0)

# ============================================================
# LAYOUT
# ============================================================

plot_col, rank_col = st.columns([4.5, 1.8])

# ============================================================
# PLOTLY RRG CHART (MOUSE ZOOM)
# ============================================================

with plot_col:
    with st.expander("📈 RRG Chart (Interactive)", expanded=True):

        fig = go.Figure()

        # Quadrants
        fig.add_shape(type="rect", x0=94, x1=100, y0=100, y1=106,
                      fillcolor="rgba(199,210,254,0.5)", line_width=0)
        fig.add_shape(type="rect", x0=100, x1=106, y0=100, y1=106,
                      fillcolor="rgba(187,247,208,0.5)", line_width=0)
        fig.add_shape(type="rect", x0=94, x1=100, y0=94, y1=100,
                      fillcolor="rgba(254,202,202,0.5)", line_width=0)
        fig.add_shape(type="rect", x0=100, x1=106, y0=94, y1=100,
                      fillcolor="rgba(254,249,195,0.5)", line_width=0)

        fig.add_hline(y=100, line_dash="dot")
        fig.add_vline(x=100, line_dash="dot")

        for s in ranked:
            rr = rs_ratio[s].iloc[start_idx:]
            mm = rs_mom[s].iloc[start_idx:]
            if rr.empty or mm.empty:
                continue

            fig.add_trace(go.Scatter(
                x=rr,
                y=mm,
                mode="lines",
                line=dict(width=1),
                opacity=0.6,
                showlegend=False,
                hoverinfo="skip"
            ))

            fig.add_trace(go.Scatter(
                x=[rr.iloc[-1]],
                y=[mm.iloc[-1]],
                mode="markers+text" if rank_dict[s] <= 15 else "markers",
                marker=dict(size=10),
                text=[display_symbol(s)] if rank_dict[s] <= 15 else None,
                textposition="top right",
                hovertemplate=(
                    f"<b>{display_symbol(s)}</b><br>"
                    "RS-Ratio: %{x:.2f}<br>"
                    "RS-Momentum: %{y:.2f}<extra></extra>"
                ),
                showlegend=False
            ))

        fig.update_layout(
            height=650,
            xaxis=dict(title="RS-Ratio", range=[94, 106]),
            yaxis=dict(title="RS-Momentum", range=[94, 106],
                       scaleanchor="x", scaleratio=1),
            dragmode="zoom",
            hovermode="closest",
            template="plotly_white",
            margin=dict(l=40, r=40, t=40, b=40)
        )

        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# RIGHT RANKING
# ============================================================

with rank_col:
    st.markdown("### Ranking")
    for s in ranked[:30]:
        st.markdown(f"**{rank_dict[s]}. {display_symbol(s)}**")

    with st.expander("📊 Ranking Table"):
        rows = ""
        for s in ranked[:30]:
            rows += f"""
            <tr>
              <td>{rank_dict[s]}</td>
              <td><a href="{tradingview_link(s)}" target="_blank">{display_symbol(s)}</a></td>
              <td>{rs_ratio[s].iloc[-1]:.2f}</td>
              <td>{rs_mom[s].iloc[-1]:.2f}</td>
            </tr>
            """
        st.markdown(f"""
        <table class="rrg-table">
        <thead><tr><th>Rank</th><th>Symbol</th><th>RS</th><th>Momentum</th></tr></thead>
        <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)

# ============================================================
# MAIN TABLE WITH HEATMAP
# ============================================================

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
    rows += f"""
    <tr style="background:{rank_color(r)}">
      <td>{r}</td>
      <td><a href="{tradingview_link(s)}" target="_blank">{display_symbol(s)}</a></td>
      <td>{rs_ratio[s].iloc[-1]:.2f}</td>
      <td>{rs_mom[s].iloc[-1]:.2f}</td>
    </tr>
    """

with st.expander("Table", expanded=True):
    st.markdown(
        legend + f"""
        <div class="rrg-wrap">
        <table class="rrg-table">
        <thead><tr><th>Rank</th><th>Symbol</th><th>RS</th><th>Momentum</th></tr></thead>
        <tbody>{rows}</tbody>
        </table>
        </div>
        """,
        unsafe_allow_html=True
    )

st.caption("RRG Stocks Terminal — Plotly (Mouse Zoom), Yahoo-only, Production Stable")

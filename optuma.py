# ==========================================================
# RRG – Stocks (FINAL | Phase-2 | Cloud-Safe)
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="RRG – Stocks", layout="wide")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
html, body {
    background:#0b0f14;
    color:#e5e7eb;
    font-family: Inter, Segoe UI, sans-serif;
    font-size:12px;
}
h1 { font-size:26px; font-weight:600; }
</style>
""", unsafe_allow_html=True)

st.title("RRG – Stocks")

# ---------------- CONSTANTS ----------------
WINDOW = 14
TAIL = 8

BENCHMARKS = {
    "NIFTY 50": "^NSEI",
    "NIFTY 200": "^CNX200",
    "NIFTY 500": "^CRSLDX"
}

TIMEFRAMES = {
    "5 min": ("5m", "30d"),
    "15 min": ("15m", "60d"),
    "Daily": ("1d", "2y"),
    "Weekly": ("1wk", "10y"),
    "Monthly": ("1mo", "20y")
}

RRG_COLORS = {
    "Leading": "#22c55e",
    "Improving": "#3b82f6",
    "Weakening": "#facc15",
    "Lagging": "#ef4444"
}

# ---------------- HELPERS ----------------
def quadrant(rr, mm):
    if rr >= 100 and mm >= 100: return "Leading"
    if rr < 100 and mm >= 100: return "Improving"
    if rr >= 100 and mm < 100: return "Weakening"
    return "Lagging"

def jdk_rs(stock, bench):
    df = pd.concat([stock, bench], axis=1).dropna()
    if len(df) < WINDOW * 2:
        return None, None
    rs = 100 * df.iloc[:,0] / df.iloc[:,1]
    rs_ratio = 100 + (rs - rs.rolling(WINDOW).mean()) / rs.rolling(WINDOW).std(ddof=0)
    roc = rs_ratio.pct_change() * 100
    rs_mom = 101 + (roc - roc.rolling(WINDOW).mean()) / roc.rolling(WINDOW).std(ddof=0)
    return rs_ratio.dropna(), rs_mom.dropna()

# ---------------- CSV LOADER (SAFE) ----------------
@st.cache_data(ttl=600)
def list_csv_files():
    base = "https://raw.githubusercontent.com/anki1007/rrg-stocks/main/ticker/"
    idx = pd.read_csv(base + "index.csv")
    return idx["file"].tolist()

@st.cache_data(ttl=600)
def load_universe(csv):
    url = f"https://raw.githubusercontent.com/anki1007/rrg-stocks/main/ticker/{csv}"
    return pd.read_csv(url)

# ---------------- SIDEBAR ----------------
st.sidebar.header("RRG — Controls")

csv_sel = st.sidebar.selectbox(
    "Indices",
    list_csv_files(),
    format_func=lambda x: x.replace(".csv", "").replace("_", " ").title()
)

benchmark_name = st.sidebar.selectbox("Benchmark", BENCHMARKS.keys())
tf_name = st.sidebar.selectbox("Timeframe", TIMEFRAMES.keys())
rank_by = st.sidebar.selectbox(
    "Rank by",
    ["RRG Power (dist)", "RS-Ratio", "RS-Momentum"]
)

show_labels = st.sidebar.toggle("Show labels on graph", True)

st.sidebar.markdown("### Phase-2 Visual Controls")
show_quadrants = st.sidebar.toggle("Show quadrant background", True)
show_centroids = st.sidebar.toggle("Show industry centroids", True)

label_top_n = st.sidebar.slider("Label top N", 0, 30, 12)
x_min, x_max = st.sidebar.slider("RS-Ratio Zoom", 90.0, 110.0, (94.0,106.0), 0.5)
y_min, y_max = st.sidebar.slider("RS-Momentum Zoom", 90.0, 110.0, (94.0,106.0), 0.5)

st.sidebar.markdown("### RRG Play Controls")
play = st.sidebar.toggle("Play", False)
speed = st.sidebar.slider("Speed (ms/frame)", 200, 1500, 600)

# ---------------- DATA LOAD ----------------
interval, yf_period = TIMEFRAMES[tf_name]
universe = load_universe(csv_sel)

symbols = universe["Symbol"].tolist()
names = dict(zip(universe["Symbol"], universe["Company Name"]))
industries = dict(zip(universe["Symbol"], universe["Industry"]))

raw = yf.download(
    symbols + [BENCHMARKS[benchmark_name]],
    interval=interval,
    period=yf_period,
    auto_adjust=True,
    progress=False
)

bench = raw["Close"][BENCHMARKS[benchmark_name]]

rows, trails = [], {}

for s in symbols:
    if s not in raw["Close"]:
        continue

    rr, mm = jdk_rs(raw["Close"][s], bench)
    if rr is None or rr.empty or mm.empty:
        continue

    rr_t = rr.iloc[-TAIL:]
    mm_t = mm.iloc[-TAIL:]

    power = np.sqrt((rr_t.iloc[-1]-100)**2 + (mm_t.iloc[-1]-100)**2)

    sym = s.replace(".NS","")

    rows.append({
        "Symbol": sym,
        "Name": names.get(s,""),
        "Industry": industries.get(s,""),
        "RS-Ratio": rr_t.iloc[-1],
        "RS-Momentum": mm_t.iloc[-1],
        "RRG Power (dist)": power,
        "Status": quadrant(rr_t.iloc[-1], mm_t.iloc[-1])
    })

    trails[sym] = (rr_t, mm_t)

df = pd.DataFrame(rows)
if df.empty:
    st.error("No data available.")
    st.stop()

df["Rank"] = df[rank_by].rank(ascending=False).astype(int)
df = df.sort_values("Rank")

# ---------------- CENTROIDS ----------------
centroids = (
    df.groupby("Industry")[["RS-Ratio","RS-Momentum"]]
      .mean()
      .reset_index()
)

# ---------------- LAYOUT ----------------
main_col, right_col = st.columns([4.5,1.5])

# ---------------- RRG GRAPH ----------------
with main_col:
    fig, ax = plt.subplots(figsize=(11,7))
    ax.axhline(100, ls=":", c="gray")
    ax.axvline(100, ls=":", c="gray")

    if show_quadrants:
        ax.axvspan(100, x_max, 100, y_max, color="#14532d", alpha=0.15)
        ax.axvspan(x_min, 100, 100, y_max, color="#1e3a8a", alpha=0.15)
        ax.axvspan(100, x_max, y_min, 100, color="#78350f", alpha=0.15)
        ax.axvspan(x_min, 100, y_min, 100, color="#7f1d1d", alpha=0.15)

    for _, r in df.iterrows():
        rr_t, mm_t = trails[r["Symbol"]]
        c = RRG_COLORS[r["Status"]]
        ax.plot(rr_t, mm_t, lw=1, color=c)
        ax.scatter(rr_t.iloc[-1], mm_t.iloc[-1], s=70, color=c)

    if show_centroids:
        ax.scatter(
            centroids["RS-Ratio"], centroids["RS-Momentum"],
            marker="D", s=220, color="white",
            edgecolor="black", zorder=5
        )

    if show_labels:
        for s in df.head(label_top_n)["Symbol"]:
            rr_t, mm_t = trails[s]
            ax.text(rr_t.iloc[-1]+0.15, mm_t.iloc[-1]+0.15, s, fontsize=8)

    handles = [plt.Line2D([0],[0], marker='o', color='w',
               markerfacecolor=c, label=k) for k,c in RRG_COLORS.items()]
    ax.legend(handles=handles, loc="upper left")

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("RS-Ratio")
    ax.set_ylabel("RS-Momentum")

    st.pyplot(fig, width="stretch")

# ---------------- RIGHT PANEL ----------------
with right_col:
    with st.expander("Top 30 RS-Momentum", expanded=True):
        top30 = df.sort_values("RS-Momentum", ascending=False).head(30)
        st.dataframe(top30[["Symbol","RS-Momentum","Status"]], height=600)

# ---------------- TABLE ----------------
df["Symbol"] = df["Symbol"].apply(
    lambda s: f"https://www.tradingview.com/chart/?symbol=NSE:{s}"
)

with st.expander("RRG Table", expanded=True):
    st.dataframe(
        df[[
            "Rank","Symbol","Name","Status",
            "Industry","RS-Ratio","RS-Momentum"
        ]],
        column_config={
            "Symbol": st.column_config.LinkColumn(
                "Symbol",
                display_text=lambda url: url.split(":")[-1]
            )
        },
        height=450,
        width="stretch"
    )

# ---------------- PLAY LOOP ----------------
if play:
    time.sleep(speed/1000)
    st.rerun()

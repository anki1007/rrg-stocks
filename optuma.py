# ==========================================================
# OPTUMA-GRADE INTRADAY RRG + SECTOR CENTROIDS
# ==========================================================

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="RRG Terminal", layout="wide")

# ---------------- THEMES ----------------
THEMES = {
    "Bloomberg Dark": """
    <style>
    html, body {
        background:#0b0f14 !important;
        color:#e5e7eb !important;
        font-family: 'IBM Plex Mono','JetBrains Mono', monospace;
        font-size:13px;
    }
    h1,h2,h3 { color:#fbbf24 !important; }
    </style>
    """
}
st.markdown(THEMES["Bloomberg Dark"], unsafe_allow_html=True)

# ---------------- PARAMETERS ----------------
WINDOW = 14

BENCHMARKS = {
    "Nifty 50": "^NSEI"
}

INTERVALS = {
    "Daily": ("1d", "6mo"),
    "15 Min": ("15m", "60d"),
    "5 Min": ("5m", "30d")
}

# ---------------- HELPERS ----------------
def jdk_components(price, bench, win=14):
    df = pd.concat([price, bench], axis=1).dropna()
    rs = 100 * df.iloc[:,0] / df.iloc[:,1]
    m = rs.rolling(win).mean()
    s = rs.rolling(win).std(ddof=0).replace(0, 1e-9)
    rs_ratio = 100 + (rs - m) / s
    roc = rs_ratio.pct_change() * 100
    m2 = roc.rolling(win).mean()
    s2 = roc.rolling(win).std(ddof=0).replace(0, 1e-9)
    rs_mom = 101 + (roc - m2) / s2
    return rs_ratio.dropna(), rs_mom.dropna()

def get_status(x, y):
    if x >= 100 and y >= 100: return "Leading"
    if x < 100 and y >= 100: return "Improving"
    if x >= 100 and y < 100: return "Weakening"
    return "Lagging"

def zscore(s):
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")

interval_label = st.sidebar.selectbox("Timeframe", INTERVALS.keys())
interval, period = INTERVALS[interval_label]

benchmark = BENCHMARKS["Nifty 50"]

symbols_input = st.sidebar.text_area(
    "Stocks (comma separated)",
    "RELIANCE.NS,TCS.NS,INFY.NS,HDFCBANK.NS,ICICIBANK.NS"
)

symbols = [s.strip().upper() for s in symbols_input.split(",") if s.strip()]

show_centroids = st.sidebar.toggle("Show Sector Centroids", value=True)

x_min, x_max = st.sidebar.slider("RS-Ratio Zoom", 90.0, 110.0, (94.0, 106.0), 0.5)
y_min, y_max = st.sidebar.slider("RS-Momentum Zoom", 90.0, 110.0, (94.0, 106.0), 0.5)

# ---------------- DATA LOAD ----------------
@st.cache_data(ttl=300)
def load_data(symbols, bench, interval, period):
    data = yf.download(symbols + [bench],
                       interval=interval,
                       period=period,
                       auto_adjust=True,
                       progress=False)
    return data

raw = load_data(symbols, benchmark, interval, period)

bench = raw["Close"][benchmark]

prices = {s: raw["Close"][s].dropna() for s in symbols if s in raw["Close"]}

# ---------------- SECTOR MAP ----------------
@st.cache_data(ttl=3600)
def fetch_sectors(symbols):
    sectors = {}
    for s in symbols:
        try:
            info = yf.Ticker(s).fast_info
            sectors[s] = info.get("sector", "Other")
        except:
            sectors[s] = "Other"
    return sectors

sector_map = fetch_sectors(symbols)

# ---------------- RRG CALC ----------------
rr_map, mm_map = {}, {}
for s, px in prices.items():
    rr, mm = jdk_components(px, bench, WINDOW)
    rr_map[s] = rr
    mm_map[s] = mm

end_idx = -1
tail = 8

# ---------------- COMPOSITE MOMENTUM ----------------
rows = []
for s in symbols:
    rr = rr_map[s].iloc[end_idx]
    mm = mm_map[s].iloc[end_idx]

    slope = np.polyfit(
        range(tail),
        mm_map[s].iloc[-tail:].values,
        1
    )[0]

    rows.append((s, rr, mm, slope, sector_map[s]))

df = pd.DataFrame(rows, columns=["Symbol","RS-Ratio","RS-Momentum","Slope","Sector"])

df["MomentumScore"] = (
    0.4*zscore(df["RS-Ratio"]) +
    0.4*zscore(df["RS-Momentum"]) +
    0.2*zscore(df["Slope"])
)

df["MomentumScore"] = 50 + 10*df["MomentumScore"]
df["Status"] = df.apply(lambda r: get_status(r["RS-Ratio"], r["RS-Momentum"]), axis=1)

# ---------------- SECTOR CENTROIDS ----------------
centroids = (
    df.groupby("Sector")[["RS-Ratio","RS-Momentum"]]
      .mean()
      .reset_index()
)

# ---------------- LAYOUT ----------------
plot_col, side_col = st.columns([4, 1.6])

# ---------------- RRG PLOT ----------------
with plot_col:
    fig, ax = plt.subplots(figsize=(11,7))
    ax.axhline(100, ls=":", c="gray")
    ax.axvline(100, ls=":", c="gray")

    ax.fill_between([94,100],[94,94],[100,100], color=(1,0,0,0.15))
    ax.fill_between([100,106],[94,94],[100,100], color=(1,1,0,0.15))
    ax.fill_between([100,106],[100,100],[106,106], color=(0,1,0,0.15))
    ax.fill_between([94,100],[100,100],[106,106], color=(0,0,1,0.15))

    # Plot stocks
    for s in symbols:
        rr = rr_map[s].iloc[-tail:]
        mm = mm_map[s].iloc[-tail:]
        ax.plot(rr, mm, lw=1)
        ax.scatter(rr.iloc[-1], mm.iloc[-1], s=70)

    # Sector centroids
    if show_centroids:
        ax.scatter(
            centroids["RS-Ratio"],
            centroids["RS-Momentum"],
            marker="D",
            s=220,
            edgecolor="black",
            linewidth=1.2,
            color="white"
        )
        for _, r in centroids.iterrows():
            ax.text(r["RS-Ratio"]+0.2, r["RS-Momentum"]+0.2,
                    r["Sector"], fontsize=9)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("RS-Ratio")
    ax.set_ylabel("RS-Momentum")

    st.pyplot(fig, use_container_width=True)

# ---------------- RIGHT SIDEBAR RANKING ----------------
with side_col:
    st.markdown("### Momentum Ranking")
    df_sorted = df.sort_values("MomentumScore", ascending=False)

    for q in ["Leading","Improving","Weakening","Lagging"]:
        with st.expander(q, expanded=(q=="Leading")):
            sub = df_sorted[df_sorted["Status"]==q].head(8)
            for i, r in sub.iterrows():
                st.markdown(
                    f"**{r['Symbol']}**  \n"
                    f"Score: `{r['MomentumScore']:.1f}`  \n"
                    f"Sector: {r['Sector']}"
                )

# ---------------- TABLE ----------------
st.markdown("## RRG Table")

st.dataframe(
    df_sorted[[
        "Symbol","Sector","Status",
        "MomentumScore","RS-Ratio","RS-Momentum","Slope"
    ]],
    use_container_width=True,
    height=420
)

st.caption("Intraday Relative Rotation Graph (5m / 15m) with Sector Centroids — Optuma-style")

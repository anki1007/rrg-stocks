# ==========================================================
# OPTUMA-GRADE RRG — INTRADAY + SECTOR CENTROIDS
# BENCHMARK + CSV-DRIVEN SYMBOLS (FINAL)
# ==========================================================

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import requests
from urllib.parse import quote

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="RRG Terminal", layout="wide")

# ---------------- UI THEME ----------------
st.markdown("""
<style>
html, body {
    background:#0b0f14 !important;
    color:#e5e7eb !important;
    font-family: 'IBM Plex Mono','JetBrains Mono', monospace;
    font-size:13px;
}
h1,h2,h3 { color:#fbbf24 !important; }
</style>
""", unsafe_allow_html=True)

# ---------------- BENCHMARKS ----------------
BENCHMARKS = {
    "NIFTY 50": "^NSEI",
    "NIFTY 200": "^CNX200",
    "NIFTY 500": "^CRSLDX"
}

INTERVALS = {
    "Daily": ("1d", "6mo"),
    "15 Min": ("15m", "60d"),
    "5 Min": ("5m", "30d")
}

WINDOW = 14
TAIL = 8

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

# ---------------- CSV SYMBOL LOADER ----------------
@st.cache_data(ttl=600)
def load_symbols_from_github(csv_name):
    url = f"https://raw.githubusercontent.com/anki1007/rrg-stocks/main/ticker/{csv_name}"
    df = pd.read_csv(url)
    col = [c for c in df.columns if "symbol" in c.lower()][0]
    return df[col].dropna().unique().tolist()

@st.cache_data(ttl=600)
def list_csv_files():
    api = "https://api.github.com/repos/anki1007/rrg-stocks/contents/ticker"
    files = requests.get(api, timeout=15).json()
    return sorted([f["name"] for f in files if f["name"].endswith(".csv")])

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")

benchmark_label = st.sidebar.selectbox("Benchmark", BENCHMARKS.keys())
benchmark = BENCHMARKS[benchmark_label]

tf_label = st.sidebar.selectbox("Timeframe", INTERVALS.keys())
interval, period = INTERVALS[tf_label]

csv_files = list_csv_files()
csv_selected = st.sidebar.selectbox("Index Constituents", csv_files)

symbols = load_symbols_from_github(csv_selected)

x_min, x_max = st.sidebar.slider("RS-Ratio Zoom", 90.0, 110.0, (94.0, 106.0), 0.5)
y_min, y_max = st.sidebar.slider("RS-Momentum Zoom", 90.0, 110.0, (94.0, 106.0), 0.5)

show_centroids = st.sidebar.toggle("Show Sector Centroids", True)

# ---------------- DATA ----------------
@st.cache_data(ttl=300)
def load_prices(symbols, bench, interval, period):
    return yf.download(
        symbols + [bench],
        interval=interval,
        period=period,
        auto_adjust=True,
        progress=False
    )

raw = load_prices(symbols, benchmark, interval, period)
bench = raw["Close"][benchmark]
prices = {s: raw["Close"][s].dropna() for s in symbols if s in raw["Close"]}

# ---------------- SECTOR MAP ----------------
@st.cache_data(ttl=3600)
def fetch_sectors(symbols):
    out = {}
    for s in symbols:
        try:
            out[s] = yf.Ticker(s).fast_info.get("sector", "Other")
        except:
            out[s] = "Other"
    return out

sector_map = fetch_sectors(symbols)

# ---------------- RRG ----------------
rows, rr_map, mm_map = [], {}, {}

for s, px in prices.items():
    rr, mm = jdk_components(px, bench, WINDOW)
    rr_map[s], mm_map[s] = rr, mm

    mom_series = mm.iloc[-TAIL:].dropna()
    if len(mom_series) >= 3:
        x = np.arange(len(mom_series))
        slope = np.polyfit(x, mom_series.values, 1)[0]
    else:
        slope = 0.0

    rows.append((s, rr.iloc[-1], mm.iloc[-1], slope, sector_map[s]))

df = pd.DataFrame(rows, columns=[
    "Symbol","RS-Ratio","RS-Momentum","Slope","Sector"
])

df["MomentumScore"] = (
    0.4*zscore(df["RS-Ratio"]) +
    0.4*zscore(df["RS-Momentum"]) +
    0.2*zscore(df["Slope"])
)

df["MomentumScore"] = 50 + 10*df["MomentumScore"]
df["Status"] = df.apply(lambda r: get_status(r["RS-Ratio"], r["RS-Momentum"]), axis=1)
df = df.sort_values("MomentumScore", ascending=False)

# ---------------- PLOT ----------------
fig, ax = plt.subplots(figsize=(11,7))
ax.axhline(100, ls=":", c="gray")
ax.axvline(100, ls=":", c="gray")

for s in df["Symbol"]:
    rr = rr_map[s].iloc[-TAIL:]
    mm = mm_map[s].iloc[-TAIL:]
    ax.plot(rr, mm, lw=1)
    ax.scatter(rr.iloc[-1], mm.iloc[-1], s=60)

if show_centroids:
    cent = df.groupby("Sector")[["RS-Ratio","RS-Momentum"]].mean()
    ax.scatter(cent["RS-Ratio"], cent["RS-Momentum"],
               marker="D", s=220, color="white", edgecolor="black")

ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel("RS-Ratio")
ax.set_ylabel("RS-Momentum")

st.pyplot(fig, use_container_width=True)

# ---------------- TABLE ----------------
st.markdown("## RRG Table")

df["TV"] = df["Symbol"].str.replace(".NS","", regex=False).apply(
    lambda s: f"https://www.tradingview.com/chart/?symbol=NSE:{quote(s)}"
)

display_df = df.copy()
display_df["Symbol"] = display_df["Symbol"].str.replace(".NS","", regex=False)

st.dataframe(
    display_df[[
        "Symbol","Sector","Status",
        "MomentumScore","RS-Ratio","RS-Momentum","Slope"
    ]],
    use_container_width=True,
    height=420,
    column_config={
        "Symbol": st.column_config.LinkColumn(
            "Symbol",
            display_text="Symbol",
            url="TV"
        )
    }
)

st.caption("Optuma-style Intraday RRG using NSE benchmarks and GitHub CSV universe")

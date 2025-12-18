# ==========================================================
# ALPHA MOMENTUM SCREENER – OPTUMA-GRADE RRG (FINAL)
# ==========================================================

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt
import requests
from urllib.parse import quote

# ---------------- CONFIG ----------------
st.set_page_config(layout="wide", page_title="Alpha Momentum Screener")

# ---------------- THEME ----------------
st.markdown("""
<style>
html, body {
    background:#0b0f14;
    color:#e5e7eb;
    font-family: Inter, sans-serif;
}
h1 { color:#4da3ff; font-size:34px; font-weight:700; }
h2 { color:#fbbf24; }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Alpha Momentum Screener</h1>", unsafe_allow_html=True)

# ---------------- BENCHMARKS ----------------
BENCHMARKS = {
    "NIFTY 50": "^NSEI",
    "NIFTY 200": "^CNX200",
    "NIFTY 500": "^CRSLDX"
}

TIMEFRAMES = {
    "5 min close": ("5m", "30d"),
    "15 min close": ("15m", "60d"),
    "30 min close": ("30m", "60d"),
    "1 hr close": ("60m", "90d"),
    "Daily close": ("1d", "5y"),
    "Weekly close": ("1wk", "10y"),
    "Monthly close": ("1mo", "20y")
}

PERIOD_MAP = {
    "6M": 126, "1Y": 252, "2Y": 504,
    "3Y": 756, "5Y": 1260, "10Y": 2520
}

WINDOW = 14
TAIL = 8

# ---------------- HELPERS ----------------
def rrg_calc(px, bench):
    df = pd.concat([px, bench], axis=1).dropna()
    if len(df) < WINDOW * 2:
        return None, None

    rs = 100 * df.iloc[:, 0] / df.iloc[:, 1]
    rs_ratio = 100 + (rs - rs.rolling(WINDOW).mean()) / rs.rolling(WINDOW).std(ddof=0)
    roc = rs_ratio.pct_change() * 100
    rs_mom = 101 + (roc - roc.rolling(WINDOW).mean()) / roc.rolling(WINDOW).std(ddof=0)
    return rs_ratio.dropna(), rs_mom.dropna()

def quadrant(x, y):
    if x >= 100 and y >= 100: return "Leading"
    if x < 100 and y >= 100: return "Improving"
    if x >= 100 and y < 100: return "Weakening"
    return "Lagging"

def tv_link(sym):
    return f"https://www.tradingview.com/chart/?symbol=NSE:{quote(sym)}"

# ---------------- CSV LOADER ----------------
@st.cache_data(ttl=600)
def list_csv():
    url = "https://api.github.com/repos/anki1007/rrg-stocks/contents/ticker"
    return [f["name"] for f in requests.get(url).json() if f["name"].endswith(".csv")]

@st.cache_data(ttl=600)
def load_universe(csv):
    url = f"https://raw.githubusercontent.com/anki1007/rrg-stocks/main/ticker/{csv}"
    df = pd.read_csv(url)
    return df

# ---------------- SIDEBAR ----------------
st.sidebar.header("Controls")

bench_name = st.sidebar.selectbox("Benchmark", BENCHMARKS.keys())
tf_name = st.sidebar.selectbox("Strength vs Timeframe", TIMEFRAMES.keys())
period_name = st.sidebar.selectbox("Period", PERIOD_MAP.keys())
rank_by = st.sidebar.selectbox(
    "Rank by",
    ["RRG Power", "RS-Ratio", "RS-Momentum", "Price % Δ", "Momentum Slope"]
)

csv_files = list_csv()
csv_sel = st.sidebar.selectbox(
    "Indices",
    csv_files,
    format_func=lambda x: x.replace(".csv", "").upper()
)

# ---------------- DATA ----------------
interval, yf_period = TIMEFRAMES[tf_name]
universe = load_universe(csv_sel)

symbols = universe["Symbol"].tolist()
names = dict(zip(universe["Symbol"], universe["Company Name"]))
industries = dict(zip(universe["Symbol"], universe["Industry"]))

raw = yf.download(symbols + [BENCHMARKS[bench_name]],
                  interval=interval, period=yf_period,
                  auto_adjust=True, progress=False)

bench = raw["Close"][BENCHMARKS[bench_name]]

rows, trails = [], {}

for s in symbols:
    if s not in raw["Close"]:
        continue
    rr, mm = rrg_calc(raw["Close"][s], bench)
    if rr is None or mm is None:
        continue

    rr_tail = rr.iloc[-TAIL:]
    mm_tail = mm.iloc[-TAIL:]
    if len(rr_tail) < 3:
        continue

    slope = np.polyfit(range(len(mm_tail)), mm_tail.values, 1)[0]
    power = np.sqrt((rr_tail.iloc[-1]-100)**2 + (mm_tail.iloc[-1]-100)**2)

    rows.append({
        "Symbol": s.replace(".NS",""),
        "Name": names.get(s,""),
        "Industry": industries.get(s,""),
        "RS-Ratio": rr_tail.iloc[-1],
        "RS-Momentum": mm_tail.iloc[-1],
        "Momentum Slope": slope,
        "RRG Power": power,
        "Status": quadrant(rr_tail.iloc[-1], mm_tail.iloc[-1]),
        "TV": tv_link(s.replace(".NS",""))
    })
    trails[s] = (rr_tail, mm_tail)

df = pd.DataFrame(rows)
if df.empty:
    st.error("No data available. Try another timeframe.")
    st.stop()

df["Rank"] = df[rank_by].rank(ascending=False).astype(int)
df = df.sort_values("Rank")

# ---------------- GRAPH ----------------
fig, ax = plt.subplots(figsize=(12,7))
ax.axhline(100, ls=":", c="gray")
ax.axvline(100, ls=":", c="gray")

colors = {
    "Leading":"#22c55e",
    "Improving":"#3b82f6",
    "Weakening":"#facc15",
    "Lagging":"#ef4444"
}

for s, (rr_t, mm_t) in trails.items():
    stt = df.loc[df["Symbol"]==s.replace(".NS",""),"Status"].values
    if len(stt)==0: continue
    ax.plot(rr_t, mm_t, lw=1, color=colors[stt[0]])
    ax.scatter(rr_t.iloc[-1], mm_t.iloc[-1], color=colors[stt[0]], s=60)

ax.set_xlabel("RS-Ratio")
ax.set_ylabel("RS-Momentum")
st.pyplot(fig, width="stretch")

# ---------------- TABLE ----------------
with st.expander("RRG Table", expanded=True):
    st.dataframe(
        df[["Rank","Name","Status","Industry","RS-Ratio","RS-Momentum"]],
        column_config={
            "Name": st.column_config.LinkColumn(
                "Name", display_text="Name", help="Open TradingView", 
                )
        },
        height=420,
        width="stretch"
    )

# ---------------- RIGHT PANEL ----------------
with st.sidebar.expander("Top 30 RS Momentum", expanded=True):
    top30 = df.sort_values("RS-Momentum", ascending=False).head(30)
    st.dataframe(top30[["Symbol","RS-Momentum","Status"]], height=600)

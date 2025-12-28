import os, time, pathlib, logging, functools, calendar, io
import datetime as _dt
import email.utils as _eutils
import urllib.request as _urlreq
from urllib.parse import quote
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import requests
import streamlit as st

# --- Safe autorefresh imports ---
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None
import streamlit.components.v1 as components

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import plotly.graph_objects as go

# -------------------- Defaults --------------------
DEFAULT_TF = "Daily"
DEFAULT_PERIOD = "1Y"

# -------------------- GitHub CSVs -----------------
GITHUB_USER = "anki1007"
GITHUB_REPO = "rrg-stocks"
GITHUB_BRANCH = "main"
GITHUB_TICKER_DIR = "ticker"
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_TICKER_DIR}/"

# -------------------- Config --------------------
PERIOD_MAP = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y": "3y", "5Y": "5y", "10Y": "10y"}
TF_LABELS = ["Daily", "Weekly", "Monthly"]
TF_TO_INTERVAL = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
WINDOW = 14
DEFAULT_TAIL = 8
BENCH_CHOICES = {"Nifty 500": "^CRSLDX", "Nifty 200": "^CNX200", "Nifty 50": "^NSEI"}

IST_TZ = "Asia/Kolkata"
BAR_CUTOFF_HOUR = 16
NET_TIME_MAX_AGE = 300

CACHE_DIR = pathlib.Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# -------------------- Matplotlib --------------------
mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["font.size"] = 15
mpl.rcParams["font.sans-serif"] = ["Inter", "Segoe UI", "DejaVu Sans", "Arial"]
mpl.rcParams["axes.grid"] = False
mpl.rcParams["axes.edgecolor"] = "#222"
mpl.rcParams["axes.labelcolor"] = "#111"
mpl.rcParams["xtick.color"] = "#333"
mpl.rcParams["ytick.color"] = "#333"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# -------------------- Streamlit Page --------------------
st.set_page_config(page_title="Relative Rotation Graph (RRG) – Stocks", layout="wide")

# Advanced Plus Jakarta Sans dark theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&display=swap');

/* Design tokens */
:root {
  --app-font: 'Plus Jakarta Sans', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  --bg: #0b0e13;
  --bg-2: #10141b;
  --border: #1f2732;
  --border-soft: #1a2230;
  --text: #e6eaee;
  --text-dim: #b3bdc7;
  --accent: #7a5cff;
  --accent-2: #2bb0ff;
}

/* Global app */
html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--app-font) !important;
}

/* Main container spacing */
.block-container {
  padding-top: 3.5rem;
}

/* Hero title style */
.hero-title {
  font-weight: 800;
  font-size: clamp(26px, 4.5vw, 40px);
  line-height: 1.05;
  margin: 4px 0 14px 0;
  background: linear-gradient(90deg, var(--accent-2), var(--accent) 60%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  letter-spacing: .2px;
}

/* Sidebar – pro skin */
section[data-testid="stSidebar"] {
  background: var(--bg-2) !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * {
  color: var(--text) !important;
}
section[data-testid="stSidebar"] label {
  font-weight: 700;
  color: var(--text-dim) !important;
}

/* Buttons */
.stButton button {
  background: linear-gradient(180deg, #1b2432, #131922);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 10px;
}
.stButton button:hover {
  filter: brightness(1.06);
}

/* Ranking list typography */
.rrg-rank { font-weight: 700; line-height: 1.25; font-size: 1.05rem; white-space: pre; }
.rrg-rank .row { display: flex; gap: 8px; align-items: baseline; margin: 2px 0; }
.rrg-rank .name { color: #9ecbff; }

/* Scrollable table with sticky header */
.rrg-wrap { max-height: calc(100vh - 260px); overflow: auto; border: 1px solid var(--border-soft); border-radius: 10px; }
.rrg-table { width: 100%; border-collapse: collapse; font-family: var(--app-font); }
.rrg-table th, .rrg-table td { border-bottom: 1px solid var(--border-soft); padding: 8px 10px; font-size: 13px; }
.rrg-table th { position: sticky; top: 0; z-index: 2; text-align: left; background: #121823; color: var(--text-dim); font-weight: 800; letter-spacing: .2px; }
.rrg-name a { color: #9ecbff; text-decoration: none; }
.rrg-name a:hover { text-decoration: underline; }

/* WebKit scrollbars */
.rrg-wrap::-webkit-scrollbar { height: 12px; width: 12px; }
.rrg-wrap::-webkit-scrollbar-thumb { background:#2e3745; border-radius: 8px; }

/* General links */
a { text-decoration: none; color: #9ecbff; }
a:hover { text-decoration: underline; }

/* Headings in main area */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
  color: var(--text) !important;
}

/* Plotly chart container */
.plotly-chart-container {
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# Hero title
st.markdown(
    '<div class="hero-title">Relative Rotation Graphs – Stocks</div>',
    unsafe_allow_html=True,
)

# -------------------- GitHub helpers --------------
@st.cache_data(ttl=600)
def list_csv_files_from_github(user: str, repo: str, branch: str, folder: str) -> List[str]:
    url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder}?ref={branch}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    items = r.json()
    files = [it["name"] for it in items if it.get("type") == "file" and it["name"].lower().endswith(".csv")]
    files.sort()
    return files

_FRIENDLY = {
    "nifty50.csv": "Nifty 50", "nifty200.csv": "Nifty 200", "nifty500.csv": "Nifty 500",
    "niftymidcap150.csv": "Nifty Midcap 150", "niftysmallcap250.csv": "Nifty Smallcap 250",
    "niftymidsmallcap400.csv": "Nifty MidSmallcap 400", "niftytotalmarket.csv": "Nifty Total Market",
}

def friendly_name_from_file(b: str) -> str:
    b2 = b.lower()
    if b2 in _FRIENDLY:
        return _FRIENDLY[b2]
    core = os.path.splitext(b)[0].replace("_", " ").replace("-", " ")
    out = ""
    for ch in core:
        out += (" " + ch) if (ch.isdigit() and out and (out[-1] != " " and not out[-1].isdigit())) else ch
    return out.title()

def build_name_maps_from_github():
    files = list_csv_files_from_github(GITHUB_USER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_TICKER_DIR)
    name_map = {friendly_name_from_file(f): f for f in files}
    return name_map, sorted(name_map.keys())

# -------------------- Universe CSV -----------------
def _normalize_cols(cols: List[str]) -> Dict[str, str]:
    return {c: c.strip().lower().replace(" ", "").replace("_", "") for c in cols}

@st.cache_data(ttl=600)
def load_universe_from_github_csv(basename: str):
    url = RAW_BASE + basename
    df = pd.read_csv(url)
    mapping = _normalize_cols(df.columns.tolist())
    sym_col = next((c for c, k in mapping.items() if k in ("symbol", "ticker", "symbols")), None)
    if sym_col is None:
        raise ValueError("CSV must contain a 'Symbol' column.")
    name_col = next((c for c, k in mapping.items() if k in ("companyname", "name", "company", "companyfullname")), sym_col)
    ind_col = next((c for c, k in mapping.items() if k in ("industry", "sector", "industries")), None)
    if ind_col is None:
        ind_col = "Industry"
        df[ind_col] = "-"
    sel = df[[sym_col, name_col, ind_col]].copy()
    sel.columns = ["Symbol", "Company Name", "Industry"]
    sel["Symbol"] = sel["Symbol"].astype(str).str.strip()
    sel["Company Name"] = sel["Company Name"].astype(str).str.strip()
    sel["Industry"] = sel["Industry"].astype(str).str.strip()
    sel = sel[sel["Symbol"] != ""].drop_duplicates(subset=["Symbol"])
    universe = sel["Symbol"].tolist()
    meta = {
        r["Symbol"]: {"name": r["Company Name"] or r["Symbol"], "industry": r["Industry"] or "-"}
        for _, r in sel.iterrows()
    }
    return universe, meta

# -------------------- Helpers --------------------
def pick_close(df, symbol: str) -> pd.Series:
    if isinstance(df, pd.Series):
        return df.dropna()
    if isinstance(df, pd.DataFrame):
        if isinstance(df.columns, pd.MultiIndex):
            for lvl in ("Close", "Adj Close"):
                if (symbol, lvl) in df.columns:
                    return df[(symbol, lvl)].dropna()
        else:
            for col in ("Close", "Adj Close"):
                if col in df.columns:
                    return df[col].dropna()
    return pd.Series(dtype=float)

def display_symbol(sym: str) -> str:
    return sym[:-3] if sym.upper().endswith(".NS") else sym

def safe_long_name(symbol: str, META: dict) -> str:
    return META.get(symbol, {}).get("name") or symbol

def tv_link_for_symbol(sym: str) -> str:
    return f"https://www.tradingview.com/chart/?symbol={quote('NSE:' + display_symbol(sym).replace('-', '_'), safe='')}"

def format_bar_date(ts: pd.Timestamp, interval: str) -> str:
    ts = pd.Timestamp(ts)
    if interval == "1wk":
        return ts.to_period("W-FRI").end_time.date().isoformat()
    if interval == "1mo":
        return ts.to_period("M").end_time.date().isoformat()
    return ts.date().isoformat()

def jdk_components(price: pd.Series, bench: pd.Series, win=14):
    df = pd.concat([price.rename("p"), bench.rename("b")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    rs = 100 * (df["p"] / df["b"])
    m = rs.rolling(win).mean()
    s = rs.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rs_ratio = (100 + (rs - m) / s).dropna()
    rroc = rs_ratio.pct_change().mul(100)
    m2 = rroc.rolling(win).mean()
    s2 = rroc.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rs_mom = (101 + (rroc - m2) / s2).dropna()
    ix = rs_ratio.index.intersection(rs_mom.index)
    return rs_ratio.loc[ix], rs_mom.loc[ix]

def has_min_coverage(rr, mm, *, min_points=20, lookback_ok=30):
    if rr is None or mm is None:
        return False
    ok = (~rr.isna()) & (~mm.isna())
    if ok.sum() < min_points:
        return False
    tail = ok.iloc[-lookback_ok:] if len(ok) >= lookback_ok else ok
    return bool(tail.any())

def get_status(x, y):
    if x <= 100 and y <= 100:
        return "Lagging"
    if x >= 100 and y >= 100:
        return "Leading"
    if x <= 100 and y >= 100:
        return "Improving"
    if x >= 100 and y <= 100:
        return "Weakening"
    return "Unknown"

def status_bg_color(x, y):
    """Get status badge color for tables - using matching quadrant colors"""
    m = get_status(x, y)
    return {"Lagging": "#dc2626", "Leading": "#15803d", "Improving": "#7c3aed", "Weakening": "#a16207"}.get(m, "#aaaaaa")

# -------------------- IST closed-bar checks --------------------
def _utc_now_from_network(timeout=2.5) -> pd.Timestamp:
    try:
        import ntplib
        c = ntplib.NTPClient()
        for host in ("time.google.com", "time.cloudflare.com", "pool.ntp.org", "asia.pool.ntp.org"):
            try:
                r = c.request(host, version=3, timeout=timeout)
                return pd.Timestamp(r.tx_time, unit="s", tz="UTC")
            except Exception:
                continue
    except Exception:
        pass
    for url in ("https://www.google.com/generate_204", "https://www.cloudflare.com", "https://www.nseindia.com", "https://www.bseindia.com"):
        try:
            req = _urlreq.Request(url, method="HEAD")
            with _urlreq.urlopen(req, timeout=timeout) as resp:
                hdr = resp.headers.get("Date")
                if hdr:
                    dt = _eutils.parsedate_to_datetime(hdr)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=_dt.timezone.utc)
                    return pd.Timestamp(dt).tz_convert("UTC")
        except Exception:
            continue
    return pd.Timestamp.now(tz="UTC")

@st.cache_data(ttl=NET_TIME_MAX_AGE)
def _now_ist_cached():
    return _utc_now_from_network().tz_convert(IST_TZ)

def _to_ist(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(IST_TZ)

def _after_cutoff_ist(now=None):
    now = _now_ist_cached() if now is None else now
    return (now.hour, now.minute) >= (BAR_CUTOFF_HOUR, 0)

def _is_bar_complete_for_timestamp(last_ts, interval, now=None):
    now = _now_ist_cached() if now is None else now
    last_ist = _to_ist(last_ts)
    now_ist = _to_ist(now)
    last_date = last_ist.date()
    today = now_ist.date()
    wd_now = now_ist.weekday()

    if interval == "1d":
        if last_date < today:
            return True
        if last_date == today:
            return _after_cutoff_ist(now_ist)
        return False
    if interval == "1wk":
        days_to_fri = (4 - wd_now) % 7
        this_friday = (now_ist + _dt.timedelta(days=days_to_fri)).date()
        last_friday = this_friday if wd_now >= 4 else (this_friday - _dt.timedelta(days=7))
        if last_date < last_friday:
            return True
        if last_date == last_friday:
            return _after_cutoff_ist(now_ist) if wd_now == 4 else True
        return False
    if interval == "1mo":
        y, m = last_ist.year, last_ist.month
        month_end = _dt.date(y, m, calendar.monthrange(y, m)[1])
        if last_date < month_end:
            return True
        if last_date == month_end:
            if today > month_end:
                return True
            return _after_cutoff_ist(now_ist)
        return False
    return False

# -------------------- Cache / Download --------------------
def _cache_path(symbol, period, interval):
    safe = symbol.replace("^", "").replace(".", "_")
    return CACHE_DIR / f"{safe}_{period}_{interval}.parquet"

def _save_cache(symbol, s, period, interval):
    try:
        s.to_frame("Close").to_parquet(_cache_path(symbol, period, interval))
    except Exception:
        pass

def _load_cache(symbol, period, interval):
    p = _cache_path(symbol, period, interval)
    if p.exists():
        try:
            return pd.read_parquet(p)["Close"].dropna()
        except Exception:
            pass
    return pd.Series(dtype=float)

def retry(n=4, delay=1.5, backoff=2.0):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **k):
            d = delay
            for i in range(n):
                try:
                    return fn(*a, **k)
                except Exception:
                    if i == n - 1:
                        raise
                    time.sleep(d)
                    d *= backoff
        return wrap
    return deco

@st.cache_data(show_spinner=False)
def download_block_with_benchmark(universe, benchmark, period, interval):
    @retry()
    def _dl():
        return yf.download(
            list(universe) + [benchmark],
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )
    raw = _dl()

    def _pick(sym):
        return pick_close(raw, sym).dropna()

    bench = _pick(benchmark)
    if bench is None or bench.empty:
        return bench, {}

    drop_last = not _is_bar_complete_for_timestamp(bench.index[-1], interval, now=_now_ist_cached())

    def _maybe_trim(s):
        return s.iloc[:-1] if (drop_last and len(s) >= 1) else s

    bench = _maybe_trim(bench)
    data = {}
    for t in universe:
        s = _pick(t)
        if not s.empty:
            data[t] = _maybe_trim(s)
        else:
            c = _load_cache(t, period, interval)
            if not c.empty:
                data[t] = _maybe_trim(c)

    if not bench.empty:
        _save_cache(benchmark, bench, period, interval)
    for t, s in data.items():
        _save_cache(t, s, period, interval)
    return bench, data

def symbol_color_map(symbols):
    tab = plt.get_cmap("tab20").colors
    return {s: to_hex(tab[i % len(tab)], keep_alpha=False) for i, s in enumerate(symbols)}

# -------------------- Quadrant Colors (matching indices screenshot) --------------------
QUADRANT_COLORS = {
    "Leading": "#15803d",    # Dark green
    "Improving": "#7c3aed",  # Purple
    "Weakening": "#a16207",  # Brown/amber
    "Lagging": "#dc2626"     # Red
}

QUADRANT_BG_COLORS = {
    "Leading": "rgba(187, 247, 208, 0.6)",
    "Improving": "rgba(233, 213, 255, 0.6)",
    "Weakening": "rgba(254, 249, 195, 0.6)",
    "Lagging": "rgba(254, 202, 202, 0.6)"
}

def status_color(x, y):
    """Get color based on quadrant position"""
    return QUADRANT_COLORS.get(get_status(x, y), "#888888")

# -------------------- Smooth Spline (Catmull-Rom) --------------------
def smooth_spline_curve(x_points, y_points, points_per_segment=8):
    """Create smooth curve using Catmull-Rom spline interpolation"""
    if len(x_points) < 3:
        return np.array(x_points), np.array(y_points)
    
    x_points, y_points = np.array(x_points, dtype=float), np.array(y_points, dtype=float)
    
    def catmull_rom_segment(p0, p1, p2, p3, num_points):
        t = np.linspace(0, 1, num_points, endpoint=False)
        t2, t3 = t * t, t * t * t
        x = 0.5 * ((2*p1[0]) + (-p0[0]+p2[0])*t + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
        y = 0.5 * ((2*p1[1]) + (-p0[1]+p2[1])*t + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
        return x, y
    
    points = np.column_stack([x_points, y_points])
    padded = np.vstack([2*points[0]-points[1], points, 2*points[-1]-points[-2]])
    x_smooth, y_smooth = [], []
    
    for i in range(len(points)-1):
        seg_x, seg_y = catmull_rom_segment(padded[i], padded[i+1], padded[i+2], padded[i+3], points_per_segment)
        x_smooth.extend(seg_x)
        y_smooth.extend(seg_y)
    
    x_smooth.append(x_points[-1])
    y_smooth.append(y_points[-1])
    return np.array(x_smooth), np.array(y_smooth)

# -------------------- Controls --------------------
st.sidebar.header("RRG — Controls")

NAME_MAP, DISPLAY_LIST = build_name_maps_from_github()
if not DISPLAY_LIST:
    st.error("No CSVs found in GitHub /ticker.")
    st.stop()

csv_disp = st.sidebar.selectbox(
    "Indices",
    DISPLAY_LIST,
    index=(DISPLAY_LIST.index("Nifty 200") if "Nifty 200" in DISPLAY_LIST else 0),
)
csv_basename = NAME_MAP[csv_disp]

bench_label = st.sidebar.selectbox(
    "Benchmark",
    list(BENCH_CHOICES.keys()),
    index=list(BENCH_CHOICES.keys()).index("Nifty 500"),
)
interval_label = st.sidebar.selectbox(
    "Strength vs (TF)",
    TF_LABELS,
    index=TF_LABELS.index(DEFAULT_TF),
)
interval = TF_TO_INTERVAL[interval_label]
default_period_for_tf = {"1d": "1Y", "1wk": "3Y", "1mo": "10Y"}[interval]
period_label = st.sidebar.selectbox(
    "Period",
    list(PERIOD_MAP.keys()),
    index=list(PERIOD_MAP.keys()).index(default_period_for_tf),
)
period = PERIOD_MAP[period_label]

rank_modes = ["Momentum Score", "RS-Ratio", "RS-Momentum", "Price %Δ (tail)", "Momentum Slope (tail)"]
rank_mode = st.sidebar.selectbox("Rank by", rank_modes, index=0)
tail_len = st.sidebar.slider("Trail Length", 1, 20, DEFAULT_TAIL, 1)

# Quadrant filtering
st.sidebar.markdown("---")
st.sidebar.markdown("**Quadrant Filters**")
top_per_quadrant = st.sidebar.slider("Top N per quadrant", 5, 50, 25, 5, help="Show top N stocks from each quadrant by Momentum Score")
show_quadrants = {
    "Leading": st.sidebar.checkbox("Leading", value=True),
    "Improving": st.sidebar.checkbox("Improving", value=True),
    "Weakening": st.sidebar.checkbox("Weakening", value=True),
    "Lagging": st.sidebar.checkbox("Lagging", value=True),
}
st.sidebar.markdown("---")

show_labels = st.sidebar.toggle("Show labels on chart", value=True)
label_top_n = st.sidebar.slider("Label top N by distance", 3, 30, 12, 1, disabled=not show_labels)
max_rank_display = st.sidebar.slider("Max items in ranking panel", 10, 50, 30, 1)

diag = st.sidebar.checkbox("Show diagnostics", value=False)

if "playing" not in st.session_state:
    st.session_state.playing = False
st.sidebar.toggle("Play / Pause", value=st.session_state.playing, key="playing")
speed_ms = st.sidebar.slider("Speed (ms/frame)", 1000, 3000, 1500, 50)
looping = st.sidebar.checkbox("Loop", value=True)

# -------------------- Data Build --------------------
UNIVERSE, META = load_universe_from_github_csv(csv_basename)
bench_symbol = BENCH_CHOICES[bench_label]
benchmark_data, tickers_data = download_block_with_benchmark(UNIVERSE, bench_symbol, period, interval)
if benchmark_data is None or benchmark_data.empty:
    st.error("Benchmark returned no data.")
    st.stop()

bench_idx = benchmark_data.index
rs_ratio_map, rs_mom_map, tickers = {}, {}, []
for t, s in tickers_data.items():
    if t == bench_symbol:
        continue
    rr, mm = jdk_components(s, benchmark_data, WINDOW)
    if len(rr) == 0 or len(mm) == 0:
        continue
    rr = rr.reindex(bench_idx)
    mm = mm.reindex(bench_idx)
    if has_min_coverage(rr, mm, min_points=max(WINDOW + 5, 20), lookback_ok=30):
        rs_ratio_map[t] = rr
        rs_mom_map[t] = mm
        tickers.append(t)

if not tickers:
    st.warning("After alignment, no symbols have enough coverage. Try a longer period.")
    st.stop()

# Optional diagnostics
if diag:
    kept = set(rs_ratio_map.keys())
    dropped = [s for s in UNIVERSE if s not in kept and s != bench_symbol]
    st.info(f"CSV universe: {len(UNIVERSE)} | Eligible after coverage: {len(kept)} | Ranked shown: {len(kept)}")
    if dropped:
        st.warning(f"Dropped (no data/insufficient coverage): {len(dropped)}")
        st.write(dropped)

# Initialize visible set BEFORE ranking/perf is computed
if "visible_set" not in st.session_state:
    st.session_state.visible_set = set(tickers)

SYMBOL_COLORS = symbol_color_map(tickers)
idx = bench_idx
idx_len = len(idx)

# -------------------- Quadrant-based filtering --------------------
def get_quadrant(t):
    """Get the quadrant for a ticker at current end_idx"""
    rr = rs_ratio_map[t].iloc[st.session_state.end_idx if "end_idx" in st.session_state else -1]
    mm = rs_mom_map[t].iloc[st.session_state.end_idx if "end_idx" in st.session_state else -1]
    return get_status(rr, mm)

def get_momentum_score(t, idx_pos):
    """Calculate momentum score (distance from center) for a ticker"""
    rr = rs_ratio_map[t].iloc[idx_pos]
    mm = rs_mom_map[t].iloc[idx_pos]
    if pd.isna(rr) or pd.isna(mm):
        return 0.0
    return float(np.hypot(rr - 100.0, mm - 100.0))

def filter_top_per_quadrant(all_tickers, idx_pos, top_n, quadrant_filter):
    """Filter to top N stocks per quadrant by momentum score"""
    # Group tickers by quadrant
    quadrants = {"Leading": [], "Improving": [], "Weakening": [], "Lagging": []}
    
    for t in all_tickers:
        rr = rs_ratio_map[t].iloc[idx_pos]
        mm = rs_mom_map[t].iloc[idx_pos]
        if pd.isna(rr) or pd.isna(mm):
            continue
        status = get_status(rr, mm)
        if status in quadrants:
            score = float(np.hypot(rr - 100.0, mm - 100.0))
            quadrants[status].append((t, score))
    
    # Sort each quadrant by momentum score (descending) and take top N
    filtered = []
    for quadrant, stocks in quadrants.items():
        if quadrant_filter.get(quadrant, True):  # Only include if quadrant is enabled
            sorted_stocks = sorted(stocks, key=lambda x: x[1], reverse=True)[:top_n]
            filtered.extend([t for t, _ in sorted_stocks])
    
    return set(filtered)

# -------------------- Date index + Animation --------------------
if "end_idx" not in st.session_state:
    st.session_state.end_idx = idx_len - 1
st.session_state.end_idx = min(max(st.session_state.end_idx, DEFAULT_TAIL), idx_len - 1)

if st.session_state.playing:
    nxt = st.session_state.end_idx + 1
    if nxt > idx_len - 1:
        nxt = DEFAULT_TAIL if looping else idx_len - 1
        if not looping:
            st.session_state.playing = False
    st.session_state.end_idx = nxt
    # Use st_autorefresh or JS fallback
    if st_autorefresh:
        st_autorefresh(interval=speed_ms, limit=None, key="rrg_auto_refresh")
    else:
        components.html(
            f"<script>setTimeout(function(){{window.parent.location.reload()}},{int(speed_ms)});</script>",
            height=0,
        )

end_idx = st.slider(
    "Date",
    min_value=DEFAULT_TAIL,
    max_value=idx_len - 1,
    step=1,
    key="end_idx",
    format=" ",
)

start_idx = max(end_idx - tail_len, 0)
date_str = format_bar_date(idx[end_idx], interval)

# Apply quadrant-based filtering
st.session_state.visible_set = filter_top_per_quadrant(tickers, end_idx, top_per_quadrant, show_quadrants)

# Show filter summary
active_quadrants = [q for q, enabled in show_quadrants.items() if enabled]
visible_count = len(st.session_state.visible_set)
st.caption(f"Showing top {top_per_quadrant} per quadrant × {len(active_quadrants)} quadrants = **{visible_count} stocks** | Quadrants: {', '.join(active_quadrants)}")

# -------- Title --------
st.markdown(f"### {bench_label} — {period_label} — {interval_label} — {csv_disp} — {date_str}")

# -------------------- Ranking Metric --------------------
def ranking_value(t: str) -> float:
    rr_last = rs_ratio_map[t].iloc[end_idx]
    mm_last = rs_mom_map[t].iloc[end_idx]
    if rank_mode == "Momentum Score":
        return float(np.hypot(rr_last - 100.0, mm_last - 100.0))
    if rank_mode == "RS-Ratio":
        return float(rr_last)
    if rank_mode == "RS-Momentum":
        return float(mm_last)
    if rank_mode == "Price %Δ (tail)":
        px = tickers_data[t].reindex(idx).dropna()
        return float((px.iloc[end_idx] / px.iloc[start_idx] - 1) * 100.0) if len(px.iloc[start_idx : end_idx + 1]) >= 2 else float("-inf")
    if rank_mode == "Momentum Slope (tail)":
        series = rs_mom_map[t].iloc[start_idx : end_idx + 1].dropna()
        if len(series) >= 2:
            x = np.arange(len(series))
            A = np.vstack([x, np.ones(len(x))]).T
            return float(np.linalg.lstsq(A, series.values, rcond=None)[0][0])
        return float("-inf")
    return float("-inf")

# Precompute perf (strongest → weakest)
perf = [(t, ranking_value(t)) for t in tickers if t in st.session_state.visible_set]
perf.sort(key=lambda x: x[1], reverse=True)

# Ranked symbols and rank-map used for BOTH the right panel and the table
ranked_syms = [sym for sym, _ in perf]
rank_dict = {sym: i for i, sym in enumerate(ranked_syms, start=1)}

# -------------------- Plot + Ranking --------------------
plot_col, rank_col = st.columns([4.5, 1.8], gap="medium")

# Helper function for label filtering
def dist_last(t):
    rr_last = rs_ratio_map[t].iloc[end_idx]
    mm_last = rs_mom_map[t].iloc[end_idx]
    return float(np.hypot(rr_last - 100.0, mm_last - 100.0))

allow_labels = (
    {t for t, _ in sorted([(t, dist_last(t)) for t in tickers], key=lambda x: x[1], reverse=True)[:label_top_n]}
    if show_labels
    else set()
)

with plot_col:
    # Build interactive Plotly RRG chart
    fig = go.Figure()

    # Add quadrant backgrounds with new colors matching indices screenshot
    fig.add_shape(type="rect", x0=94, y0=94, x1=100, y1=100, fillcolor=QUADRANT_BG_COLORS["Lagging"], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=100, y0=94, x1=106, y1=100, fillcolor=QUADRANT_BG_COLORS["Weakening"], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=100, y0=100, x1=106, y1=106, fillcolor=QUADRANT_BG_COLORS["Leading"], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=94, y0=100, x1=100, y1=106, fillcolor=QUADRANT_BG_COLORS["Improving"], line_width=0, layer="below")

    # Add center lines (solid, not dotted - matching screenshot)
    fig.add_hline(y=100, line_color="rgba(80,80,80,0.8)", line_width=1.5)
    fig.add_vline(x=100, line_color="rgba(80,80,80,0.8)", line_width=1.5)

    # Add quadrant labels with matching colors
    fig.add_annotation(x=95, y=105.5, text="<b>IMPROVING</b>", showarrow=False, font=dict(size=13, color="#7c3aed"))
    fig.add_annotation(x=105, y=105.5, text="<b>LEADING</b>", showarrow=False, font=dict(size=13, color="#15803d"))
    fig.add_annotation(x=105, y=94.5, text="<b>WEAKENING</b>", showarrow=False, font=dict(size=13, color="#a16207"))
    fig.add_annotation(x=95, y=94.5, text="<b>LAGGING</b>", showarrow=False, font=dict(size=13, color="#dc2626"))

    # Plot each ticker with smooth trail, gradient width/opacity, arrow, and rich hover
    for t in tickers:
        if t not in st.session_state.visible_set:
            continue
        rr = rs_ratio_map[t].iloc[start_idx + 1 : end_idx + 1].dropna()
        mm = rs_mom_map[t].iloc[start_idx + 1 : end_idx + 1].dropna()
        rr, mm = rr.align(mm, join="inner")
        if len(rr) < 2:
            continue

        name = META.get(t, {}).get("name", t)
        industry = META.get(t, {}).get("industry", "-")

        # Get current values for hover
        rr_last = float(rr.values[-1])
        mm_last = float(mm.values[-1])
        status = get_status(rr_last, mm_last)
        
        # Use quadrant-based color (matching indices screenshot)
        color = status_color(rr_last, mm_last)

        # Calculate price and change
        px = tickers_data[t].reindex(idx).dropna()
        price = float(px.iloc[end_idx]) if end_idx < len(px) else np.nan
        chg = ((px.iloc[end_idx] / px.iloc[start_idx] - 1) * 100.0) if (end_idx < len(px) and start_idx < len(px)) else np.nan

        # Calculate Momentum Score (distance from center, higher = stronger)
        rrg_power = float(np.hypot(rr_last - 100.0, mm_last - 100.0))

        # Hover text for head marker
        hover_head = (
            f"<b>{name}</b><br>"
            + f"<b>Status:</b> {status}<br>"
            + f"<b>RS-Ratio:</b> {rr_last:.2f}<br>"
            + f"<b>RS-Momentum:</b> {mm_last:.2f}<br>"
            + f"<b>Momentum Score:</b> {rrg_power:.2f}<br>"
            + f"<b>Price:</b> ₹{price:,.2f}<br>"
            + f"<b>Change %:</b> {chg:+.2f}%<br>"
            + f"<b>Industry:</b> {industry}"
        )

        # Original data points
        x_pts = rr.values.astype(float)
        y_pts = mm.values.astype(float)
        n_original = len(x_pts)

        # Apply Catmull-Rom spline smoothing
        if n_original >= 3:
            x_smooth, y_smooth = smooth_spline_curve(x_pts, y_pts, points_per_segment=8)
        else:
            x_smooth, y_smooth = x_pts, y_pts

        n_smooth = len(x_smooth)

        # Draw smooth trail with gradient width and opacity (thin/transparent at tail, thick/opaque at head)
        if n_smooth >= 2:
            for i in range(n_smooth - 1):
                prog = i / max(1, n_smooth - 2)  # Progress from 0 (tail) to 1 (head)
                line_width = 2.5 + prog * 3  # Width: 2.5 -> 5.5
                opacity = 0.5 + prog * 0.5   # Opacity: 0.5 -> 1.0
                fig.add_trace(
                    go.Scatter(
                        x=[x_smooth[i], x_smooth[i+1]],
                        y=[y_smooth[i], y_smooth[i+1]],
                        mode='lines',
                        line=dict(color=color, width=line_width),
                        opacity=opacity,
                        hoverinfo='skip',
                        showlegend=False,
                    )
                )

        # Trail marker points (on original data points, not smoothed) - gradient size
        trail_sizes = [5 + (i / max(1, n_original - 1)) * 5 for i in range(n_original)]
        if n_original > 1:
            fig.add_trace(
                go.Scatter(
                    x=x_pts[:-1],
                    y=y_pts[:-1],
                    mode='markers',
                    marker=dict(
                        size=trail_sizes[:-1],
                        color=color,
                        opacity=0.8,
                        line=dict(color='white', width=1)
                    ),
                    hoverinfo='skip',
                    showlegend=False,
                )
            )

        # Head marker (larger, with white border) - matching screenshot
        fig.add_trace(
            go.Scatter(
                x=[rr_last],
                y=[mm_last],
                mode='markers',
                marker=dict(
                    size=14,
                    color=color,
                    line=dict(color='white', width=2.5)
                ),
                text=[hover_head],
                hoverinfo='text',
                hoverlabel=dict(bgcolor="#1a1f2e", bordercolor=color, font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="white")),
                showlegend=False,
            )
        )

        # Add arrow head showing direction (matching screenshot style)
        if n_original >= 2:
            dx = x_pts[-1] - x_pts[-2]
            dy = y_pts[-1] - y_pts[-2]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0.01:
                fig.add_annotation(
                    x=x_pts[-1],
                    y=y_pts[-1],
                    ax=x_pts[-1] - dx/length * 0.35,
                    ay=y_pts[-1] - dy/length * 0.35,
                    xref='x',
                    yref='y',
                    axref='x',
                    ayref='y',
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.8,
                    arrowwidth=3,
                    arrowcolor=color,
                )

        # Add label for top N by distance
        if show_labels and t in allow_labels:
            fig.add_annotation(
                x=rr_last,
                y=mm_last,
                text=f"<b>{name}</b>",
                showarrow=True,
                arrowhead=0,
                arrowwidth=1.5,
                arrowcolor=color,
                ax=30,
                ay=-25,
                font=dict(size=11, color=color),
                bgcolor='rgba(0,0,0,0)',
                borderwidth=0,
            )

    # Update layout - matching dark theme from screenshot
    fig.update_layout(
        title=dict(text=f"<b>Relative Rotation Graph</b> | {date_str}", font=dict(size=18, color='#e6eaee'), x=0.5),
        xaxis=dict(title="<b>JdK RS-Ratio</b>", range=[94, 106], showgrid=True, gridcolor='rgba(150,150,150,0.2)', tickfont=dict(color='#b3bdc7')),
        yaxis=dict(title="<b>JdK RS-Momentum</b>", range=[94, 106], showgrid=True, gridcolor='rgba(150,150,150,0.2)', tickfont=dict(color='#b3bdc7')),
        plot_bgcolor='#fafafa',
        paper_bgcolor='#0b0e13',
        margin=dict(l=60, r=30, t=60, b=60),
        hoverlabel=dict(align="left"),
        height=620,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"], "displaylogo": False})

with rank_col:
    st.markdown("### Ranking by Quadrant")
    
    # Group by quadrant with momentum scores
    quadrant_stocks = {"Leading": [], "Improving": [], "Weakening": [], "Lagging": []}
    for sym in ranked_syms:
        rr = float(rs_ratio_map[sym].iloc[end_idx])
        mm = float(rs_mom_map[sym].iloc[end_idx])
        status = get_status(rr, mm)
        mom_score = float(np.hypot(rr - 100.0, mm - 100.0))
        if status in quadrant_stocks:
            quadrant_stocks[status].append((sym, mom_score, rr, mm))
    
    # Sort each quadrant by momentum score
    for quadrant in quadrant_stocks:
        quadrant_stocks[quadrant].sort(key=lambda x: x[1], reverse=True)
    
    # Quadrant colors and icons - matching chart colors
    q_config = {
        "Leading": {"color": "#15803d", "bg": "rgba(187, 247, 208, 0.3)", "icon": "🟢"},
        "Improving": {"color": "#7c3aed", "bg": "rgba(233, 213, 255, 0.3)", "icon": "🟣"},
        "Weakening": {"color": "#a16207", "bg": "rgba(254, 249, 195, 0.3)", "icon": "🟡"},
        "Lagging": {"color": "#dc2626", "bg": "rgba(254, 202, 202, 0.3)", "icon": "🔴"},
    }
    
    # Build collapsible HTML sections
    ranking_html = """
    <style>
    .quadrant-section {
        margin-bottom: 8px;
        border-radius: 8px;
        overflow: hidden;
    }
    .quadrant-header {
        padding: 10px 12px;
        cursor: pointer;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-weight: 800;
        font-size: 14px;
        user-select: none;
        transition: filter 0.2s;
    }
    .quadrant-header:hover {
        filter: brightness(1.1);
    }
    .quadrant-header .arrow {
        transition: transform 0.2s;
        font-size: 12px;
    }
    .quadrant-header.collapsed .arrow {
        transform: rotate(-90deg);
    }
    .quadrant-content {
        max-height: 2000px;
        overflow: hidden;
        transition: max-height 0.3s ease-out;
    }
    .quadrant-content.collapsed {
        max-height: 0;
    }
    .stock-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 6px 12px;
        border-bottom: 1px solid rgba(255,255,255,0.05);
        font-size: 13px;
    }
    .stock-row:hover {
        background: rgba(255,255,255,0.05);
    }
    .stock-name {
        flex: 1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin-right: 8px;
    }
    .stock-rank {
        color: #8b949e;
        font-weight: 700;
        min-width: 28px;
        margin-right: 8px;
    }
    .mom-score {
        font-weight: 700;
        font-size: 12px;
        padding: 2px 6px;
        border-radius: 4px;
        background: rgba(255,255,255,0.1);
    }
    </style>
    """
    
    for quadrant in ["Leading", "Improving", "Weakening", "Lagging"]:
        if not show_quadrants.get(quadrant, True):
            continue
        
        stocks = quadrant_stocks[quadrant]
        if not stocks:
            continue
        
        cfg = q_config[quadrant]
        count = len(stocks)
        
        ranking_html += f"""
        <div class="quadrant-section">
            <div class="quadrant-header" style="background:{cfg['bg']}; color:{cfg['color']}" onclick="this.classList.toggle('collapsed'); this.nextElementSibling.classList.toggle('collapsed');">
                <span>{cfg['icon']} {quadrant} ({count})</span>
                <span class="arrow">▼</span>
            </div>
            <div class="quadrant-content" style="background: rgba(0,0,0,0.2);">
        """
        
        for sym, mom_score, rr, mm in stocks:
            name = META.get(sym, {}).get("name", sym)
            # Truncate long names
            display_name = name[:30] + "..." if len(name) > 30 else name
            ranking_html += f"""
                <div class="stock-row">
                    <span class="stock-rank">{rank_dict.get(sym, '-')}</span>
                    <span class="stock-name" style="color:{cfg['color']}" title="{name}">{display_name}</span>
                    <span class="mom-score" style="color:{cfg['color']}">{mom_score:.1f}</span>
                </div>
            """
        
        ranking_html += """
            </div>
        </div>
        """
    
    if ranking_html:
        components.html(ranking_html, height=500, scrolling=True)
    else:
        st.write("—")

# -------------------- Interactive Table --------------------
def make_interactive_table(rows):
    """Generate a fully self-contained interactive HTML table with sorting and filtering"""
    table_id = "rrg_table_" + str(abs(hash(str(len(rows)))) % 10000)

    headers = ["Ranking", "Name", "Status", "Industry", "RS-Ratio", "RS-Momentum", "Momentum Score", "Price", "Change %"]
    header_keys = ["rank", "name", "status", "industry", "rs_ratio", "rs_mom", "rrg_power", "price", "chg"]

    # Build header with sort icons
    th_cells = []
    for i, h in enumerate(headers):
        sort_class = "sort-asc" if i == 0 else ""
        th_cells.append(f'<th class="{sort_class}" data-col="{i}" data-key="{header_keys[i]}"><span>{h}</span><span class="sort-icon"></span></th>')
    th = "<tr>" + "".join(th_cells) + "</tr>"

    # Build rows with data attributes for filtering/sorting
    tr_list = []
    for r in rows:
        rr_val = r["rs_ratio"] if not pd.isna(r["rs_ratio"]) else 0
        mm_val = r["rs_mom"] if not pd.isna(r["rs_mom"]) else 0
        price_val = r["price"] if not pd.isna(r["price"]) else 0
        chg_val = r["chg"] if not pd.isna(r["chg"]) else 0

        # Calculate Momentum Score (distance from center)
        rrg_power_val = float(np.hypot(rr_val - 100.0, mm_val - 100.0))

        rr_txt = "-" if pd.isna(r["rs_ratio"]) else f"{r['rs_ratio']:.2f}"
        mm_txt = "-" if pd.isna(r["rs_mom"]) else f"{r['rs_mom']:.2f}"
        rrg_power_txt = f"{rrg_power_val:.2f}"
        price_txt = "-" if pd.isna(r["price"]) else f"₹{r['price']:,.2f}"
        chg_txt = "-" if pd.isna(r["chg"]) else f"{r['chg']:+.2f}%"

        # Color for change % column
        chg_color = "#4ade80" if r.get("chg", 0) and r.get("chg", 0) > 0 else "#f87171" if r.get("chg", 0) and r.get("chg", 0) < 0 else "#9ca3af"

        # Status badge color
        status_bg = r["bg"]
        status_fg = r["fg"]

        # Escape single quotes in name for data attribute
        safe_name = r["name"].replace("'", "&#39;").lower()
        safe_industry = r["industry"].replace("'", "&#39;").lower()

        tr_list.append(
            f"<tr class='rrg-row' "
            + f"data-rank='{r['rank']}' data-name='{safe_name}' data-status='{r['status'].lower()}' "
            + f"data-industry='{safe_industry}' data-rs_ratio='{rr_val}' data-rs_mom='{mm_val}' "
            + f"data-rrg_power='{rrg_power_val}' data-price='{price_val}' data-chg='{chg_val}'>"
            + f"<td class='rank-cell'>{r['rank']}</td>"
            + f"<td class='rrg-name'><a href='{r['tv']}' target='_blank'>{r['name']}</a></td>"
            + f"<td><span class='status-badge' style='background:{status_bg}; color:{status_fg}'>{r['status']}</span></td>"
            + f"<td class='industry-cell'>{r['industry']}</td>"
            + f"<td>{rr_txt}</td>"
            + f"<td>{mm_txt}</td>"
            + f"<td class='power-cell'>{rrg_power_txt}</td>"
            + f"<td>{price_txt}</td>"
            + f"<td class='chg-cell' style='color:{chg_color}'>{chg_txt}</td>"
            + "</tr>"
        )

    # Get unique values for dropdowns
    statuses = sorted(set(r["status"] for r in rows))
    industries = sorted(set(r["industry"] for r in rows if r["industry"] != "-"))

    status_options = '<option value="">All Statuses</option>' + "".join(f'<option value="{s.lower()}">{s}</option>' for s in statuses)
    industry_options = '<option value="">All Industries</option>' + "".join(f'<option value="{ind.lower()}">{ind}</option>' for ind in industries)

    # Calculate height based on rows
    table_height = min(650, 90 + len(rows) * 44)

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');
            
            * {{
                box-sizing: border-box;
                margin: 0;
                padding: 0;
            }}
            
            body {{
                font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
                background: #10141b;
                color: #e6eaee;
                padding: 12px;
            }}
            
            .filter-row {{
                display: flex;
                gap: 10px;
                margin-bottom: 12px;
                flex-wrap: wrap;
                align-items: center;
            }}
            
            .filter-row input, .filter-row select {{
                background: #0b0e13;
                border: 1px solid #1f2732;
                color: #e6eaee;
                padding: 8px 12px;
                border-radius: 8px;
                font-family: inherit;
                font-size: 13px;
            }}
            
            .filter-row input:focus, .filter-row select:focus {{
                outline: none;
                border-color: #7a5cff;
            }}
            
            .filter-row input::placeholder {{
                color: #b3bdc7;
            }}
            
            .filter-badge {{
                background: #7a5cff;
                color: white;
                padding: 5px 12px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: 700;
            }}
            
            .table-wrap {{
                max-height: {table_height - 60}px;
                overflow: auto;
                border: 1px solid #1f2732;
                border-radius: 10px;
            }}
            
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            
            th, td {{
                border-bottom: 1px solid #1a2230;
                padding: 10px 12px;
                font-size: 13px;
                text-align: left;
            }}
            
            th {{
                position: sticky;
                top: 0;
                z-index: 2;
                background: #121823;
                color: #b3bdc7;
                font-weight: 800;
                cursor: pointer;
                user-select: none;
                transition: background 0.2s;
                white-space: nowrap;
            }}
            
            th:hover {{
                background: #1a2233;
            }}
            
            .sort-icon {{
                margin-left: 6px;
                opacity: 0.5;
                font-size: 10px;
            }}
            
            th.sort-asc .sort-icon::after {{ content: '▲'; opacity: 1; }}
            th.sort-desc .sort-icon::after {{ content: '▼'; opacity: 1; }}
            th:not(.sort-asc):not(.sort-desc) .sort-icon::after {{ content: '⇅'; }}
            
            /* Row styling - neutral background */
            .rrg-row {{
                background: #0d1117;
                transition: background 0.15s;
            }}
            
            .rrg-row:hover {{
                background: #161b22;
            }}
            
            .rrg-row:nth-child(even) {{
                background: #0f1419;
            }}
            
            .rrg-row:nth-child(even):hover {{
                background: #161b22;
            }}
            
            /* Name column link */
            .rrg-name a {{
                color: #58a6ff;
                text-decoration: none;
                font-weight: 600;
            }}
            
            .rrg-name a:hover {{
                text-decoration: underline;
                color: #79b8ff;
            }}
            
            /* Rank column */
            .rank-cell {{
                font-weight: 700;
                color: #8b949e;
                text-align: center;
                width: 70px;
            }}
            
            /* Status badge */
            .status-badge {{
                display: inline-block;
                padding: 4px 10px;
                border-radius: 6px;
                font-size: 12px;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.3px;
            }}
            
            /* Industry column */
            .industry-cell {{
                color: #8b949e;
                font-size: 12px;
            }}
            
            /* Change % column */
            .chg-cell {{
                font-weight: 700;
                text-align: right;
            }}
            
            /* Momentum Score column */
            .power-cell {{
                font-weight: 600;
                color: #a78bfa;
            }}
            
            tr.hidden-row {{
                display: none;
            }}
            
            /* Scrollbar */
            .table-wrap::-webkit-scrollbar {{
                height: 10px;
                width: 10px;
            }}
            .table-wrap::-webkit-scrollbar-thumb {{
                background: #2e3745;
                border-radius: 8px;
            }}
            .table-wrap::-webkit-scrollbar-track {{
                background: #10141b;
            }}
        </style>
    </head>
    <body>
        <div class="filter-row">
            <input type="text" id="{table_id}_search" placeholder="🔍 Search by name..." style="min-width: 180px;">
            <select id="{table_id}_status">{status_options}</select>
            <select id="{table_id}_industry">{industry_options}</select>
            <span class="filter-badge" id="{table_id}_count">{len(rows)} / {len(rows)}</span>
        </div>
        <div class="table-wrap">
            <table id="{table_id}">
                <thead>{th}</thead>
                <tbody>{''.join(tr_list)}</tbody>
            </table>
        </div>
        
        <script>
        (function() {{
            const table = document.getElementById('{table_id}');
            const headers = table.querySelectorAll('th');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            const searchInput = document.getElementById('{table_id}_search');
            const statusFilter = document.getElementById('{table_id}_status');
            const industryFilter = document.getElementById('{table_id}_industry');
            const countBadge = document.getElementById('{table_id}_count');
            
            let currentSort = {{ col: 0, asc: true }};
            
            function updateCount() {{
                const visible = rows.filter(r => !r.classList.contains('hidden-row')).length;
                countBadge.textContent = visible + ' / ' + rows.length;
            }}
            
            function sortTable(colIndex, key) {{
                const isNumeric = ['rank', 'rs_ratio', 'rs_mom', 'rrg_power', 'price', 'chg'].includes(key);
                const asc = currentSort.col === colIndex ? !currentSort.asc : (colIndex === 0);
                currentSort = {{ col: colIndex, asc: asc }};
                
                headers.forEach((h, i) => {{
                    h.classList.remove('sort-asc', 'sort-desc');
                    if (i === colIndex) {{
                        h.classList.add(asc ? 'sort-asc' : 'sort-desc');
                    }}
                }});
                
                rows.sort((a, b) => {{
                    let aVal = a.dataset[key] || '';
                    let bVal = b.dataset[key] || '';
                    
                    if (isNumeric) {{
                        aVal = parseFloat(aVal) || 0;
                        bVal = parseFloat(bVal) || 0;
                        return asc ? aVal - bVal : bVal - aVal;
                    }} else {{
                        return asc ? aVal.localeCompare(bVal) : bVal.localeCompare(aVal);
                    }}
                }});
                
                rows.forEach(row => tbody.appendChild(row));
            }}
            
            function filterTable() {{
                const searchTerm = searchInput.value.toLowerCase();
                const statusTerm = statusFilter.value;
                const industryTerm = industryFilter.value;
                
                rows.forEach(row => {{
                    const name = row.dataset.name || '';
                    const status = row.dataset.status || '';
                    const industry = row.dataset.industry || '';
                    
                    const matchesSearch = !searchTerm || name.includes(searchTerm);
                    const matchesStatus = !statusTerm || status === statusTerm;
                    const matchesIndustry = !industryTerm || industry === industryTerm;
                    
                    if (matchesSearch && matchesStatus && matchesIndustry) {{
                        row.classList.remove('hidden-row');
                    }} else {{
                        row.classList.add('hidden-row');
                    }}
                }});
                
                updateCount();
            }}
            
            headers.forEach((header, index) => {{
                header.addEventListener('click', () => {{
                    sortTable(index, header.dataset.key);
                }});
            }});
            
            searchInput.addEventListener('input', filterTable);
            statusFilter.addEventListener('change', filterTable);
            industryFilter.addEventListener('change', filterTable);
            
            updateCount();
        }})();
        </script>
    </body>
    </html>
    """

    return html_content, table_height


# Build table rows for ALL stocks (not just filtered visible_set)
# Sort all tickers by momentum score for table display
all_stocks_with_scores = []
for t in tickers:
    rr = float(rs_ratio_map[t].iloc[end_idx])
    mm = float(rs_mom_map[t].iloc[end_idx])
    mom_score = float(np.hypot(rr - 100.0, mm - 100.0))
    all_stocks_with_scores.append((t, mom_score))

all_stocks_with_scores.sort(key=lambda x: x[1], reverse=True)
all_ranked_syms = [t for t, _ in all_stocks_with_scores]
all_rank_dict = {sym: i for i, sym in enumerate(all_ranked_syms, start=1)}

rows = []
for t in all_ranked_syms:
    rr = float(rs_ratio_map[t].iloc[end_idx])
    mm = float(rs_mom_map[t].iloc[end_idx])
    status = get_status(rr, mm)
    bg = status_bg_color(rr, mm)
    fg = "#ffffff" if bg in ("#e06a6a", "#3fa46a", "#5d86d1") else "#000000"
    px = tickers_data[t].reindex(idx).dropna()
    price = float(px.iloc[end_idx]) if end_idx < len(px) else np.nan
    chg = ((px.iloc[end_idx] / px.iloc[start_idx] - 1) * 100.0) if (end_idx < len(px) and start_idx < len(px)) else np.nan
    rows.append({
        "rank": all_rank_dict.get(t, ""),
        "name": META.get(t, {}).get("name", t),
        "status": status,
        "industry": META.get(t, {}).get("industry", "-"),
        "rs_ratio": rr,
        "rs_mom": mm,
        "price": price,
        "chg": chg,
        "bg": bg,
        "fg": fg,
        "tv": tv_link_for_symbol(t),
    })

with st.expander(f"📊 Full Table — All {len(rows)} Stocks", expanded=True):
    table_html, table_height = make_interactive_table(rows)
    components.html(table_html, height=table_height, scrolling=False)

# -------------------- Downloads --------------------
def export_ranks_csv(all_stocks_sorted):
    out = []
    for t, mom_score in all_stocks_sorted:
        rr = float(rs_ratio_map[t].iloc[end_idx])
        mm = float(rs_mom_map[t].iloc[end_idx])
        out.append((
            all_rank_dict[t],
            t,
            META.get(t, {}).get("name", t),
            META.get(t, {}).get("industry", "-"),
            mom_score,
            rr,
            mm,
            get_status(rr, mm),
        ))
    df = pd.DataFrame(out, columns=["ranking", "symbol", "name", "industry", "momentum_score", "rs_ratio", "rs_momentum", "status"])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


def export_table_csv(rows_):
    df = pd.DataFrame([{
        "ranking": r["rank"],
        "name": r["name"],
        "industry": r["industry"],
        "status": r["status"],
        "rs_ratio": r["rs_ratio"],
        "rs_momentum": r["rs_mom"],
        "price": r["price"],
        "pct_change_tail": r["chg"],
    } for r in rows_])
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode()


c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "Download Ranks CSV",
        data=export_ranks_csv(all_stocks_with_scores),
        file_name=f"ranks_{date_str}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with c2:
    st.download_button(
        "Download Table CSV",
        data=export_table_csv(rows),
        file_name=f"table_{date_str}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption("Names open TradingView. Use Play/Pause to watch rotation; Speed controls frame interval; Loop wraps frames.")



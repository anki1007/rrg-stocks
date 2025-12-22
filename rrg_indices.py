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

# --- NEW: safe autorefresh imports ---
try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None
import streamlit.components.v1 as components
# -------------------------------------

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# -------------------- Config --------------------
GITHUB_USER = "anki1007"
GITHUB_REPO = "rrg-stocks"
GITHUB_BRANCH = "main"
GITHUB_TICKER_DIR = "ticker"
CSV_BASENAME = "niftyindices.csv"  # CSV path under /ticker
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_TICKER_DIR}/"

DEFAULT_TF = "Weekly"
WINDOW = 14
DEFAULT_TAIL = 8
PERIOD_MAP = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y": "3y", "5Y": "5y", "10Y": "10y"}
TF_LABELS = ["Daily", "Weekly", "Monthly"]
TF_TO_INTERVAL = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
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
st.set_page_config(page_title="Relative Rotation Graphs – Indices", layout="wide")

# Advanced Plus Jakarta Sans dark theme, keeping all original logic
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

/* General links (TradingView, etc.) */
a { text-decoration: none; color: #9ecbff; }
a:hover { text-decoration: underline; }

/* Headings in main area */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
  color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# Hero title above the dynamic date title
st.markdown(
    '<div class="hero-title">Relative Rotation Graphs – Indices</div>',
    unsafe_allow_html=True,
)

# -------------------- Helpers --------------------
def _normalize_cols(cols: List[str]) -> Dict[str, str]:
    return {c: c.strip().lower().replace(" ", "").replace("_", "") for c in cols}

def _to_yahoo_symbol(raw_sym: str) -> str:
    s = str(raw_sym).strip().upper()
    if s.endswith(".NS") or s.startswith("^"):
        return s
    return "^" + s  # map index codes like CNXIT -> ^CNXIT

# ---- CSV loaders (GitHub with cache-bust, or uploaded override) ----
@st.cache_data(ttl=600)
def load_universe_from_github_csv(basename: str, cache_bust: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    url = RAW_BASE + basename
    df = pd.read_csv(url)
    mapping = _normalize_cols(df.columns.tolist())
    sym_col = next((c for c,k in mapping.items() if k in ("symbol","ticker","symbols")), None)
    if sym_col is None:
        raise ValueError("CSV must contain 'Symbol' column.")
    name_col = next((c for c,k in mapping.items() if k in ("companyname","name","company","companyfullname")), sym_col)
    ind_col  = next((c for c,k in mapping.items() if k in ("industry","sector","industries")), None)
    if ind_col is None:
        ind_col = "Industry"
        df[ind_col] = "-"

    sel = df[[sym_col, name_col, ind_col]].copy()
    sel.columns = ["Symbol","Company Name","Industry"]
    sel = sel[sel["Symbol"].astype(str).str.strip() != ""].drop_duplicates(subset=["Symbol"])

    sel["Yahoo"] = sel["Symbol"].apply(_to_yahoo_symbol)
    universe = sel["Yahoo"].tolist()
    meta = {
        r["Yahoo"]: {
            "name": (r["Company Name"] or r["Yahoo"]),
            "industry": (r["Industry"] or "-"),
            "raw_symbol": r["Symbol"],
            "is_equity": r["Yahoo"].endswith(".NS"),
        }
        for _, r in sel.iterrows()
    }
    # cache_bust is unused in code but varies the cache key
    _ = cache_bust
    return universe, meta

def load_universe_from_uploaded_csv(file) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    df = pd.read_csv(file)
    mapping = _normalize_cols(df.columns.tolist())
    sym_col = next((c for c,k in mapping.items() if k in ("symbol","ticker","symbols")), None)
    if sym_col is None:
        raise ValueError("Uploaded CSV must contain 'Symbol' column.")
    name_col = next((c for c,k in mapping.items() if k in ("companyname","name","company","companyfullname")), sym_col)
    ind_col  = next((c for c,k in mapping.items() if k in ("industry","sector","industries")), None)
    if ind_col is None:
        ind_col = "Industry"
        df[ind_col] = "-"

    sel = df[[sym_col, name_col, ind_col]].copy()
    sel.columns = ["Symbol","Company Name","Industry"]
    sel = sel[sel["Symbol"].astype(str).str.strip() != ""].drop_duplicates(subset=["Symbol"])
    sel["Yahoo"] = sel["Symbol"].apply(_to_yahoo_symbol)

    universe = sel["Yahoo"].tolist()
    meta = {
        r["Yahoo"]: {
            "name": (r["Company Name"] or r["Yahoo"]),
            "industry": (r["Industry"] or "-"),
            "raw_symbol": r["Symbol"],
            "is_equity": r["Yahoo"].endswith(".NS"),
        }
        for _, r in sel.iterrows()
    }
    return universe, meta

def pick_close(df, symbol: str) -> pd.Series:
    if isinstance(df, pd.Series): return df.dropna()
    if isinstance(df, pd.DataFrame):
        if isinstance(df.columns, pd.MultiIndex):
            for lvl in ("Close","Adj Close"):
                if (symbol, lvl) in df.columns: return df[(symbol, lvl)].dropna()
        else:
            for col in ("Close","Adj Close"):
                if col in df.columns: return df[col].dropna()
    return pd.Series(dtype=float)

def display_symbol(sym: str) -> str:
    return sym[:-3] if sym.upper().endswith(".NS") else sym.lstrip("^")

def tv_link_for_symbol(yahoo_sym: str) -> str:
    if yahoo_sym.endswith(".NS"):
        return f"https://www.tradingview.com/chart/?symbol={quote('NSE:'+display_symbol(yahoo_sym).replace('-','_'), safe='')}"
    return f"https://www.tradingview.com/chart/?symbol={quote(display_symbol(yahoo_sym), safe='')}"

def format_bar_date(ts: pd.Timestamp, interval: str) -> str:
    ts = pd.Timestamp(ts)
    if interval=="1wk": return ts.to_period("W-FRI").end_time.date().isoformat()
    if interval=="1mo": return ts.to_period("M").end_time.date().isoformat()
    return ts.date().isoformat()

def jdk_components(price: pd.Series, bench: pd.Series, win=14):
    df = pd.concat([price.rename("p"), bench.rename("b")], axis=1).dropna()
    if df.empty: return pd.Series(dtype=float), pd.Series(dtype=float)
    rs = 100 * (df["p"]/df["b"])
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
    if rr is None or mm is None: return False
    ok = (~rr.isna()) & (~mm.isna())
    if ok.sum() < min_points: return False
    tail = ok.iloc[-lookback_ok:] if len(ok) >= lookback_ok else ok
    return bool(tail.any())

def get_status(x, y):
    if x<=100 and y<=100: return "Lagging"
    if x>=100 and y>=100: return "Leading"
    if x<=100 and y>=100: return "Improving"
    if x>=100 and y<=100: return "Weakening"
    return "Unknown"

def status_bg_color(x, y):
    m = get_status(x, y)
    return {"Lagging":"#e06a6a","Leading":"#3fa46a","Improving":"#5d86d1","Weakening":"#e2d06b"}.get(m,"#aaaaaa")

# -------------------- IST closed-bar checks --------------------
def _utc_now_from_network(timeout=2.5) -> pd.Timestamp:
    try:
        import ntplib
        c=ntplib.NTPClient()
        for host in ("time.google.com","time.cloudflare.com","pool.ntp.org","asia.pool.ntp.org"):
            try:
                r=c.request(host, version=3, timeout=timeout)
                return pd.Timestamp(r.tx_time, unit="s", tz="UTC")
            except Exception:
                continue
    except Exception:
        pass
    for url in ("https://www.google.com/generate_204","https://www.cloudflare.com","https://www.nseindia.com","https://www.bseindia.com"):
        try:
            req=_urlreq.Request(url, method="HEAD")
            with _urlreq.urlopen(req, timeout=timeout) as resp:
                hdr=resp.headers.get("Date")
                if hdr:
                    dt=_eutils.parsedate_to_datetime(hdr)
                    if dt.tzinfo is None: dt=dt.replace(tzinfo=_dt.timezone.utc)
                    return pd.Timestamp(dt).tz_convert("UTC")
        except Exception:
            continue
    return pd.Timestamp.now(tz="UTC")

@st.cache_data(ttl=NET_TIME_MAX_AGE)
def _now_ist_cached():
    return _utc_now_from_network().tz_convert(IST_TZ)

def _to_ist(ts):
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None: ts = ts.tz_localize("UTC")
    return ts.tz_convert(IST_TZ)

def _after_cutoff_ist(now=None):
    now=_now_ist_cached() if now is None else now
    return (now.hour, now.minute) >= (BAR_CUTOFF_HOUR, 0)

def _is_bar_complete_for_timestamp(last_ts, interval, now=None):
    now=_now_ist_cached() if now is None else now
    last_ist=_to_ist(last_ts); now_ist=_to_ist(now)
    last_date=last_ist.date(); today=now_ist.date(); wd_now=now_ist.weekday()

    if interval=="1d":
        if last_date < today: return True
        if last_date == today: return _after_cutoff_ist(now_ist)
        return False
    if interval=="1wk":
        days_to_fri=(4-wd_now)%7
        this_friday=(now_ist+_dt.timedelta(days=days_to_fri)).date()
        last_friday=this_friday if wd_now>=4 else (this_friday - _dt.timedelta(days=7))
        if last_date < last_friday: return True
        if last_date == last_friday: return _after_cutoff_ist(now_ist) if wd_now==4 else True
        return False
    if interval=="1mo":
        y,m=last_ist.year, last_ist.month
        month_end=_dt.date(y,m,calendar.monthrange(y,m)[1])
        if last_date < month_end: return True
        if last_date == month_end:
            if today > month_end: return True
            return _after_cutoff_ist(now_ist)
        return False
    return False

# -------------------- Cache / Download --------------------
def _cache_path(symbol, period, interval):
    safe = symbol.replace("^","").replace(".","_")
    return CACHE_DIR / f"{safe}_{period}_{interval}.parquet"

def _save_cache(symbol, s, period, interval):
    try:
        s.to_frame("Close").to_parquet(_cache_path(symbol,period,interval))
    except Exception:
        pass

def _load_cache(symbol, period, interval):
    p=_cache_path(symbol,period,interval)
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
            d=delay
            for i in range(n):
                try:
                    return fn(*a, **k)
                except Exception:
                    if i==n-1: raise
                    time.sleep(d); d*=backoff
        return wrap
    return deco

@st.cache_data(show_spinner=False)
def download_block_with_benchmark(universe, benchmark, period, interval):
    @retry()
    def _dl():
        return yf.download(list(universe)+[benchmark], period=period, interval=interval,
                           group_by="ticker", auto_adjust=True, progress=False, threads=True)
    raw=_dl()
    def _pick(sym): return pick_close(raw, sym).dropna()

    bench=_pick(benchmark)
    if bench is None or bench.empty:
        return bench, {}

    drop_last = not _is_bar_complete_for_timestamp(bench.index[-1], interval, now=_now_ist_cached())
    def _maybe_trim(s): return s.iloc[:-1] if (drop_last and len(s)>=1) else s

    bench=_maybe_trim(bench)
    data={}
    for t in universe:
        s=_pick(t)
        if not s.empty:
            data[t]=_maybe_trim(s)
        else:
            c=_load_cache(t,period,interval)
            if not c.empty:
                data[t]=_maybe_trim(c)

    if not bench.empty:
        _save_cache(benchmark, bench, period, interval)
    for t,s in data.items():
        _save_cache(t, s, period, interval)
    return bench, data

def symbol_color_map(symbols):
    tab = plt.get_cmap("tab20").colors
    return {s: to_hex(tab[i % len(tab)], keep_alpha=False) for i, s in enumerate(symbols)}

# -------------------- Controls --------------------
st.sidebar.header("RRG — Controls")

# Data source controls
uploaded = st.sidebar.file_uploader("Upload indices CSV (optional override)", type=["csv"])
if st.sidebar.button("Reload universe"):
    st.cache_data.clear()
    st.rerun()

bench_label = st.sidebar.selectbox("Benchmark", list(BENCH_CHOICES.keys()), index=0)
interval_label = st.sidebar.selectbox("Strength vs (TF)", TF_LABELS, index=TF_LABELS.index(DEFAULT_TF))
interval = TF_TO_INTERVAL[interval_label]
default_period_for_tf = {"1d": "1Y", "1wk": "1Y", "1mo": "10Y"}[interval]
period_label = st.sidebar.selectbox("Period", list(PERIOD_MAP.keys()), index=list(PERIOD_MAP.keys()).index(default_period_for_tf))
period = PERIOD_MAP[period_label]

rank_modes = ["RRG Power (dist)", "RS-Ratio", "RS-Momentum", "Price %Δ (tail)", "Momentum Slope (tail)"]
rank_mode = st.sidebar.selectbox("Rank by", rank_modes, index=0)
tail_len = st.sidebar.slider("Trail Length", 1, 20, DEFAULT_TAIL, 1)
show_labels = st.sidebar.toggle("Show labels on chart", value=False)
label_top_n = st.sidebar.slider("Label top N by distance", 3, 30, 12, 1, disabled=not show_labels)
max_rank_display = st.sidebar.slider("Max items in ranking panel", 10, 200, 50, 1)

diag = st.sidebar.checkbox("Show diagnostics", value=False)

if "playing" not in st.session_state:
    st.session_state.playing = False
st.sidebar.toggle("Play / Pause", value=st.session_state.playing, key="playing")
speed_ms = st.sidebar.slider("Speed (ms/frame)", 1000, 3000, 1500, 50)
looping = st.sidebar.checkbox("Loop", value=True)

# -------------------- Data Build --------------------
if uploaded is not None:
    UNIVERSE, META = load_universe_from_uploaded_csv(uploaded)
else:
    # cache_bust varies the cache key so updated CSVs are fetched within ttl window
    UNIVERSE, META = load_universe_from_github_csv(CSV_BASENAME, cache_bust=str(pd.Timestamp.utcnow().floor("1min")))

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
    if has_min_coverage(rr, mm, min_points=max(WINDOW+5,20), lookback_ok=30):
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
    # --- FIX: use st_autorefresh or JS fallback ---
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
    format=" "
)

start_idx = max(end_idx - tail_len, 0)
date_str = format_bar_date(idx[end_idx], interval)

# -------- Title --------
st.markdown(f"###{date_str}")

# -------------------- Ranking Metric (1 = strongest) --------------------
def ranking_value(t: str) -> float:
    rr_last = rs_ratio_map[t].iloc[end_idx]
    mm_last = rs_mom_map[t].iloc[end_idx]
    if rank_mode == "RRG Power (dist)":
        return float(np.hypot(rr_last - 100.0, mm_last - 100.0))
    if rank_mode == "RS-Ratio":
        return float(rr_last)
    if rank_mode == "RS-Momentum":
        return float(mm_last)
    if rank_mode == "Price %Δ (tail)":
        px = tickers_data[t].reindex(idx).dropna()
        return float((px.iloc[end_idx] / px.iloc[start_idx] - 1) * 100.0) if len(px.iloc[start_idx:end_idx+1]) >= 2 else float("-inf")
    if rank_mode == "Momentum Slope (tail)":
        series = rs_mom_map[t].iloc[start_idx:end_idx+1].dropna()
        if len(series) >= 2:
            x = np.arange(len(series)); A = np.vstack([x, np.ones(len(x))]).T
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

with plot_col:
    fig, ax = plt.subplots(1, 1, figsize=(10.6, 6.8))
    ax.set_facecolor("#F5F5DC")        # inner plot background
    fig.patch.set_facecolor("#F5F5DC") # outer figure background
    ax.set_title("Relative Rotation Graph (RRG)", fontsize=15, pad=10)
    ax.set_xlabel("JdK RS-Ratio", fontsize=14)
    ax.set_ylabel("JdK RS-Momentum", fontsize=14)
    ax.axhline(100, color="#777", linestyle=":", linewidth=1.1)
    ax.axvline(100, color="#777", linestyle=":", linewidth=1.1)
    ax.fill_between([94, 100], [94, 94], [100, 100], color=(1.0, 0.0, 0.0, 0.20))
    ax.fill_between([100, 106], [94, 94], [100, 100], color=(1.0, 1.0, 0.0, 0.20))
    ax.fill_between([100, 106], [100, 100], [106, 106], color=(0.0, 1.0, 0.0, 0.20))
    ax.fill_between([94, 100], [100, 100], [106, 106], color=(0.0, 0.0, 1.0, 0.20))
    ax.text(95, 105, "Improving", fontsize=13, weight="bold")
    ax.text(104, 105, "Leading", fontsize=13, weight="bold", ha="right")
    ax.text(104, 95, "Weakening", fontsize=13, weight="bold", ha="right")
    ax.text(95, 95, "Lagging", fontsize=13, weight="bold")
    ax.set_xlim(94, 106); ax.set_ylim(94, 106)

    def dist_last(t):
        rr_last = rs_ratio_map[t].iloc[end_idx]
        mm_last = rs_mom_map[t].iloc[end_idx]
        return float(np.hypot(rr_last - 100.0, mm_last - 100.0))

    allow_labels = {t for t, _ in sorted([(t, dist_last(t)) for t in tickers],
                                         key=lambda x: x[1], reverse=True)[:label_top_n]} if show_labels else set()

    for t in tickers:
        if t not in st.session_state.visible_set:
            continue
        rr = rs_ratio_map[t].iloc[start_idx + 1 : end_idx + 1].dropna()
        mm = rs_mom_map[t].iloc[start_idx + 1 : end_idx + 1].dropna()
        rr, mm = rr.align(mm, join="inner")
        if len(rr) == 0 or len(mm) == 0:
            continue
        ax.plot(rr.values, mm.values, linewidth=1.2, alpha=0.7, color=SYMBOL_COLORS[t])
        sizes = [22] * (len(rr) - 1) + [76]
        ax.scatter(rr.values, mm.values, s=sizes, linewidths=0.6,
                   facecolor=SYMBOL_COLORS[t], edgecolor="#333")
        if show_labels and t in allow_labels:
            ax.annotate(t, (rr.values[-1], mm.values[-1]), fontsize=11,
                        color=SYMBOL_COLORS[t], xytext=(6, 6), textcoords="offset points")

    st.pyplot(fig, use_container_width=True)

with rank_col:
    st.markdown("### Ranking")
    if ranked_syms:
        rows_html = []
        for sym in ranked_syms[:max_rank_display]:
            rr = float(rs_ratio_map[sym].iloc[end_idx])
            mm = float(rs_mom_map[sym].iloc[end_idx])
            stat = get_status(rr, mm)
            color = SYMBOL_COLORS.get(sym, "#333")
            name = META.get(sym, {}).get("name", sym)
            rows_html.append(
                f'<div class="row" style="color:{color}"><span>{rank_dict[sym]}.</span>'
                f'<span class="name">{name}</span><span>[{stat}]</span></div>'
            )
        st.markdown(f'<div class="rrg-rank">{"".join(rows_html)}</div>', unsafe_allow_html=True)
    else:
        st.write("—")

# -------------------- Table --------------------
def make_table_html(rows):
    headers = ["Ranking", "Name", "Status", "Industry", "RS-Ratio", "RS-Momentum", "Price", "Change %"]
    th = "<tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr>"
    tr = []
    for r in rows:
        rr_txt  = "-" if pd.isna(r["rs_ratio"]) else f"{r['rs_ratio']:.2f}"
        mm_txt  = "-" if pd.isna(r["rs_mom"])  else f"{r['rs_mom']:.2f}"
        price_txt = "-" if pd.isna(r["price"]) else f"{r['price']:.2f}"
        chg_txt   = "-" if pd.isna(r["chg"])   else f"{r['chg']:.2f}"
        tr.append(
            "<tr class='rrg-row' style='background:%s; color:%s'>" % (r["bg"], r["fg"]) +
            f"<td>{r['rank']}</td>" +
            f"<td class='rrg-name'><a href='{r['tv']}' target='_blank'>{r['name']}</a></td>" +
            f"<td>{r['status']}</td>" +
            f"<td>{r['industry']}</td>" +
            f"<td>{rr_txt}</td>" +
            f"<td>{mm_txt}</td>" +
            f"<td>{price_txt}</td>" +
            f"<td>{chg_txt}</td>" +
            "</tr>"
        )
    return f"<div class='rrg-wrap'><table class='rrg-table'>{th}{''.join(tr)}</table></div>"

# Build table rows IN RANK ORDER so it matches the right panel
rows = []
for t in ranked_syms:
    rr = float(rs_ratio_map[t].iloc[end_idx])
    mm = float(rs_mom_map[t].iloc[end_idx])
    status = get_status(rr, mm)
    bg = status_bg_color(rr, mm)
    fg = "#ffffff" if bg in ("#e06a6a", "#3fa46a", "#5d86d1") else "#000000"
    px = tickers_data[t].reindex(idx).dropna()
    price = float(px.iloc[end_idx]) if end_idx < len(px) else np.nan
    chg = ((px.iloc[end_idx] / px.iloc[start_idx] - 1) * 100.0) if (end_idx < len(px) and start_idx < len(px)) else np.nan
    rows.append({
        "rank": rank_dict.get(t, ""),
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

with st.expander("Table", expanded=True):
    st.markdown(make_table_html(rows), unsafe_allow_html=True)

# -------------------- Downloads --------------------
def export_ranks_csv(perf_sorted):
    out=[]
    for t,_m in perf_sorted:
        rr=float(rs_ratio_map[t].iloc[end_idx]); mm=float(rs_mom_map[t].iloc[end_idx])
        out.append((rank_dict[t], t, META.get(t,{}).get("name",t), META.get(t,{}).get("industry","-"),
                    _m, rr, mm, get_status(rr, mm)))
    df=pd.DataFrame(out, columns=["ranking","symbol","name","industry","rank_metric","rs_ratio","rs_momentum","status"])
    buf=io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode()

def export_table_csv(rows_):
    df=pd.DataFrame([{
        "ranking": r["rank"],
        "name": r["name"],
        "industry": r["industry"],
        "status": r["status"],
        "rs_ratio": r["rs_ratio"],
        "rs_momentum": r["rs_mom"],
        "price": r["price"],
        "pct_change_tail": r["chg"],
    } for r in rows_])
    buf=io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode()

c1, c2 = st.columns(2)
with c1:
    st.download_button("Download Ranks CSV", data=export_ranks_csv(perf),
                       file_name=f"ranks_{date_str}.csv", mime="text/csv", use_container_width=True)
with c2:
    st.download_button("Download Table CSV", data=export_table_csv(rows),
                       file_name=f"table_{date_str}.csv", mime="text/csv", use_container_width=True)

st.caption("Names open TradingView. Use Play/Pause to watch rotation; Speed controls frame interval; Loop wraps frames.")

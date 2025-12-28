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

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:
    st_autorefresh = None
import streamlit.components.v1 as components

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import plotly.graph_objects as go

# -------------------- Config --------------------
GITHUB_USER = "anki1007"
GITHUB_REPO = "rrg-stocks"
GITHUB_BRANCH = "main"
CSV_BASENAME = "niftyindices.csv"
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/"

DEFAULT_TF = "Weekly"
WINDOW = 14
DEFAULT_TAIL = 8
PERIOD_MAP = {"3M": "3mo", "6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y": "3y", "5Y": "5y", "10Y": "10y"}
TF_LABELS = ["60 Min", "240 Min","Daily", "Weekly", "Monthly"]
TF_TO_INTERVAL = {"60 Min": "60m","240 Min": "240m", "Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
BENCH_CHOICES = {"Nifty 500": "^CRSLDX", "Nifty 200": "^CNX200", "Nifty 50": "^NSEI"}

IST_TZ = "Asia/Kolkata"
BAR_CUTOFF_HOUR = 16
NET_TIME_MAX_AGE = 300

CACHE_DIR = pathlib.Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["font.size"] = 15
mpl.rcParams["font.sans-serif"] = ["Inter", "Segoe UI", "DejaVu Sans", "Arial"]
mpl.rcParams["axes.grid"] = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

st.set_page_config(page_title="Relative Rotation Graphs – Indices", layout="wide")

# -------------------- CSS Styling --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
  --app-font: 'Plus Jakarta Sans', system-ui, sans-serif;
  --bg: #0b0e13;
  --bg-2: #10141b;
  --bg-3: #161b24;
  --border: #1f2732;
  --text: #e6eaee;
  --text-dim: #8b949e;
  --accent: #7a5cff;
}

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--app-font) !important;
}

header[data-testid="stHeader"] { background: var(--bg) !important; border-bottom: 1px solid var(--border) !important; }
[data-testid="stToolbar"] { background: var(--bg) !important; }
.main .block-container { background: var(--bg) !important; padding-top: 1rem; max-width: 100%; }

.hero-title {
  font-weight: 800; font-size: clamp(24px, 4vw, 36px); margin: 0 0 8px 0;
  background: linear-gradient(90deg, #2bb0ff, #7a5cff 60%);
  -webkit-background-clip: text; background-clip: text; color: transparent;
}

section[data-testid="stSidebar"] { background: var(--bg-2) !important; border-right: 1px solid var(--border); }
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] label { font-weight: 600; color: var(--text-dim) !important; font-size: 13px; }

.stButton button {
  background: linear-gradient(180deg, #1b2432, #131922);
  color: var(--text); border: 1px solid var(--border); border-radius: 8px; font-weight: 600; font-size: 13px;
}
.stButton button:hover { filter: brightness(1.1); }

div[data-testid="stExpander"] { background: var(--bg-2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
div[data-testid="stExpander"] > details { background: var(--bg-2) !important; }
div[data-testid="stExpander"] summary { background: var(--bg-2) !important; color: var(--text) !important; }
div[data-testid="stExpander"] summary span { color: var(--text) !important; }

.timeline-container { background: var(--bg-2); border: 1px solid var(--border); border-radius: 10px; padding: 12px 16px; margin-bottom: 12px; }
.timeline-header { display: flex; justify-content: space-between; align-items: center; }
.timeline-date { font-size: 18px; font-weight: 700; color: var(--text); }
.timeline-range { font-size: 13px; color: var(--text-dim); }

[data-baseweb="popover"] { background: var(--bg-2) !important; border: 1px solid var(--border) !important; }
[data-baseweb="popover"] li { background: var(--bg-2) !important; color: var(--text) !important; }
[data-baseweb="popover"] li:hover { background: #1a2233 !important; }

.stSlider [data-baseweb="slider"] [role="slider"] { background: var(--accent) !important; }
.stDownloadButton button { background: var(--bg-2) !important; border: 1px solid var(--border) !important; color: var(--text) !important; }
.stCaption, small { color: var(--text-dim) !important; }
.stCheckbox label span { color: var(--text) !important; }
div[data-testid="stCheckbox"] { margin: 1px 0 !important; }

a { text-decoration: none; color: #58a6ff; }
a:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# -------------------- Helpers --------------------
def _normalize_cols(cols: List[str]) -> Dict[str, str]:
    return {c: c.strip().lower().replace(" ", "").replace("_", "") for c in cols}

def _to_yahoo_symbol(raw_sym: str) -> str:
    s = str(raw_sym).strip().upper()
    if s.endswith(".NS") or s.startswith("^"): return s
    return "^" + s

@st.cache_data(ttl=600)
def load_universe_from_github_csv(basename: str, cache_bust: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    url = RAW_BASE + basename
    df = pd.read_csv(url)
    mapping = _normalize_cols(df.columns.tolist())
    sym_col = next((c for c,k in mapping.items() if k in ("symbol","ticker","symbols")), None)
    if sym_col is None: raise ValueError("CSV must contain 'Symbol' column.")
    name_col = next((c for c,k in mapping.items() if k in ("companyname","name","company","companyfullname")), sym_col)
    ind_col = next((c for c,k in mapping.items() if k in ("industry","sector","industries")), None)
    if ind_col is None: ind_col = "Industry"; df[ind_col] = "-"
    sel = df[[sym_col, name_col, ind_col]].copy()
    sel.columns = ["Symbol","Company Name","Industry"]
    sel = sel[sel["Symbol"].astype(str).str.strip() != ""].drop_duplicates(subset=["Symbol"])
    sel["Yahoo"] = sel["Symbol"].apply(_to_yahoo_symbol)
    universe = sel["Yahoo"].tolist()
    meta = {r["Yahoo"]: {"name": (r["Company Name"] or r["Yahoo"]), "industry": (r["Industry"] or "-"), "raw_symbol": r["Symbol"]} for _, r in sel.iterrows()}
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
    if interval=="60m":
        if ts.tzinfo is not None: ts_ist = ts.tz_convert(IST_TZ)
        else: ts_ist = ts.tz_localize("UTC").tz_convert(IST_TZ)
        bar_end = ts_ist + pd.Timedelta(hours=1)
        return bar_end.strftime("%Y-%m-%d %H:%M")
    if interval=="1wk": return ts.to_period("W-FRI").end_time.date().isoformat()
    if interval=="1mo": return ts.to_period("M").end_time.date().isoformat()
    return ts.date().isoformat()

def filter_nse_market_hours(df_or_series, interval: str):
    if interval != "60m" or df_or_series is None or (hasattr(df_or_series, 'empty') and df_or_series.empty):
        return df_or_series
    idx = df_or_series.index
    if idx.tz is None: idx_ist = idx.tz_localize("UTC").tz_convert(IST_TZ)
    else: idx_ist = idx.tz_convert(IST_TZ)
    valid_mask = pd.Series([(9 <= ts.hour <= 14) for ts in idx_ist], index=df_or_series.index)
    return df_or_series[valid_mask]

def jdk_components(price: pd.Series, bench: pd.Series, win=14):
    df = pd.concat([price.rename("p"), bench.rename("b")], axis=1).dropna()
    if df.empty: return pd.Series(dtype=float), pd.Series(dtype=float)
    rs = 100 * (df["p"]/df["b"])
    m = rs.rolling(win).mean(); s = rs.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rs_ratio = (100 + (rs - m) / s).dropna()
    rroc = rs_ratio.pct_change().mul(100)
    m2 = rroc.rolling(win).mean(); s2 = rroc.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rs_mom = (101 + (rroc - m2) / s2).dropna()
    ix = rs_ratio.index.intersection(rs_mom.index)
    return rs_ratio.loc[ix], rs_mom.loc[ix]

def has_min_coverage(rr, mm, *, min_points=20, lookback_ok=30):
    if rr is None or mm is None: return False
    ok = (~rr.isna()) & (~mm.isna())
    if ok.sum() < min_points: return False
    return bool((ok.iloc[-lookback_ok:] if len(ok) >= lookback_ok else ok).any())

def get_status(x, y):
    if x<=100 and y<=100: return "Lagging"
    if x>=100 and y>=100: return "Leading"
    if x<=100 and y>=100: return "Improving"
    if x>=100 and y<=100: return "Weakening"
    return "Unknown"

# Quadrant colors - DARKER for visibility
QUADRANT_COLORS = {"Leading": "#15803d", "Improving": "#7c3aed", "Weakening": "#a16207", "Lagging": "#dc2626"}
QUADRANT_BG_COLORS = {"Leading": "rgba(187, 247, 208, 0.6)", "Improving": "rgba(233, 213, 255, 0.6)", "Weakening": "rgba(254, 249, 195, 0.6)", "Lagging": "rgba(254, 202, 202, 0.6)"}

def status_color(x, y): return QUADRANT_COLORS.get(get_status(x, y), "#888888")

# -------------------- Smooth Spline (NumPy only) --------------------
def smooth_spline_curve(x_points, y_points, points_per_segment=8):
    if len(x_points) < 3: return np.array(x_points), np.array(y_points)
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
        x_smooth.extend(seg_x); y_smooth.extend(seg_y)
    x_smooth.append(x_points[-1]); y_smooth.append(y_points[-1])
    return np.array(x_smooth), np.array(y_smooth)

# -------------------- Time helpers --------------------
def _utc_now_from_network(timeout=2.5) -> pd.Timestamp:
    try:
        import ntplib
        c=ntplib.NTPClient()
        for host in ("time.google.com","time.cloudflare.com","pool.ntp.org"):
            try: r=c.request(host, version=3, timeout=timeout); return pd.Timestamp(r.tx_time, unit="s", tz="UTC")
            except: continue
    except: pass
    for url in ("https://www.google.com/generate_204","https://www.cloudflare.com"):
        try:
            req=_urlreq.Request(url, method="HEAD")
            with _urlreq.urlopen(req, timeout=timeout) as resp:
                hdr=resp.headers.get("Date")
                if hdr:
                    dt=_eutils.parsedate_to_datetime(hdr)
                    if dt.tzinfo is None: dt=dt.replace(tzinfo=_dt.timezone.utc)
                    return pd.Timestamp(dt).tz_convert("UTC")
        except: continue
    return pd.Timestamp.now(tz="UTC")

@st.cache_data(ttl=NET_TIME_MAX_AGE)
def _now_ist_cached(): return _utc_now_from_network().tz_convert(IST_TZ)

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
    if interval=="60m":
        if last_date < today: return True
        return (now_ist.hour*60+now_ist.minute) >= ((last_ist.hour+1)*60+last_ist.minute)
    if interval=="1d":
        if last_date < today: return True
        return _after_cutoff_ist(now_ist) if last_date == today else False
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
        if last_date == month_end: return True if today > month_end else _after_cutoff_ist(now_ist)
        return False
    return False

# -------------------- Download --------------------
def _cache_path(symbol, period, interval):
    return CACHE_DIR / f"{symbol.replace('^','').replace('.','_')}_{period}_{interval}.parquet"

def _save_cache(symbol, s, period, interval):
    try: s.to_frame("Close").to_parquet(_cache_path(symbol,period,interval))
    except: pass

def _load_cache(symbol, period, interval):
    p=_cache_path(symbol,period,interval)
    if p.exists():
        try: return pd.read_parquet(p)["Close"].dropna()
        except: pass
    return pd.Series(dtype=float)

def retry(n=4, delay=1.5, backoff=2.0):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **k):
            d=delay
            for i in range(n):
                try: return fn(*a, **k)
                except:
                    if i==n-1: raise
                    time.sleep(d); d*=backoff
        return wrap
    return deco

@st.cache_data(show_spinner=False, ttl=300)
def download_block_with_benchmark(universe, benchmark, period, interval):
    @retry()
    def _dl(): return yf.download(list(universe)+[benchmark], period=period, interval=interval, group_by="ticker", auto_adjust=True, progress=False, threads=True)
    raw=_dl()
    def _pick(sym): return pick_close(raw, sym).dropna()
    bench=_pick(benchmark)
    if bench is None or bench.empty: return bench, {}
    bench = filter_nse_market_hours(bench, interval)
    if bench is None or bench.empty: return bench, {}
    now_ist = _now_ist_cached()
    if interval == "60m":
        drop_last = now_ist.weekday() < 5 and now_ist.hour >= 9 and (now_ist.hour < 15 or (now_ist.hour == 15 and now_ist.minute < 30))
    else:
        drop_last = not _is_bar_complete_for_timestamp(bench.index[-1], interval, now=now_ist)
    def _maybe_trim(s): return s.iloc[:-1] if (drop_last and len(s)>=1) else s
    bench=_maybe_trim(bench)
    data={}
    for t in universe:
        s=_pick(t)
        if not s.empty:
            s = filter_nse_market_hours(s, interval)
            if not s.empty: data[t]=_maybe_trim(s)
        else:
            c=_load_cache(t,period,interval)
            if not c.empty:
                c = filter_nse_market_hours(c, interval)
                if not c.empty: data[t]=_maybe_trim(c)
    if not bench.empty: _save_cache(benchmark, bench, period, interval)
    for t,s in data.items(): _save_cache(t, s, period, interval)
    return bench, data

def symbol_color_map(symbols):
    tab = plt.get_cmap("tab20").colors
    return {s: to_hex(tab[i % len(tab)], keep_alpha=False) for i, s in enumerate(symbols)}

# -------------------- Page --------------------
st.markdown('<div class="hero-title">Relative Rotation Graphs – Indices</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Controls")
bench_label = st.sidebar.selectbox("Benchmark", list(BENCH_CHOICES.keys()), index=0)
interval_label = st.sidebar.selectbox("Timeframe", TF_LABELS, index=TF_LABELS.index(DEFAULT_TF))
interval = TF_TO_INTERVAL[interval_label]

if interval == "60m":
    period_label = st.sidebar.selectbox("Date Range", ["1M", "3M", "6M"], index=1)
    period = {"1M": "1mo", "3M": "3mo", "6M": "6mo"}[period_label]
else:
    default_p = {"1d": "1Y", "1wk": "1Y", "1mo": "10Y"}[interval]
    period_label = st.sidebar.selectbox("Date Range", list(PERIOD_MAP.keys()), index=list(PERIOD_MAP.keys()).index(default_p))
    period = PERIOD_MAP[period_label]

tail_len = st.sidebar.slider("Tail Length", 1, 20, DEFAULT_TAIL, 1)

# Show Labels - default to True (checked)
if "show_labels" not in st.session_state:
    st.session_state.show_labels = True
show_labels = st.sidebar.checkbox("Show Labels", value=st.session_state.show_labels, key="show_labels")

st.sidebar.markdown("---")
st.sidebar.markdown("**Animation**")
if "playing" not in st.session_state: st.session_state.playing = False
st.sidebar.toggle("▶ Play", value=st.session_state.playing, key="playing")
speed_ms = st.sidebar.slider("Speed (ms)", 500, 3000, 1200, 100)
looping = st.sidebar.checkbox("Loop", value=True)

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear(); st.rerun()

# Load data
UNIVERSE, META = load_universe_from_github_csv(CSV_BASENAME, str(pd.Timestamp.utcnow().floor("1min")))
bench_symbol = BENCH_CHOICES[bench_label]
benchmark_data, tickers_data = download_block_with_benchmark(UNIVERSE, bench_symbol, period, interval)
if benchmark_data is None or benchmark_data.empty: st.error("Benchmark returned no data."); st.stop()

bench_idx = benchmark_data.index
rs_ratio_map, rs_mom_map, tickers = {}, {}, []
for t, s in tickers_data.items():
    if t == bench_symbol: continue
    rr, mm = jdk_components(s, benchmark_data, WINDOW)
    if len(rr) == 0 or len(mm) == 0: continue
    rr, mm = rr.reindex(bench_idx), mm.reindex(bench_idx)
    if has_min_coverage(rr, mm, min_points=max(WINDOW+5,20), lookback_ok=30):
        rs_ratio_map[t], rs_mom_map[t] = rr, mm
        tickers.append(t)

if not tickers: st.warning("No symbols have enough coverage."); st.stop()

SYMBOL_COLORS = symbol_color_map(tickers)
idx = bench_idx
idx_len = len(idx)

# Session state - initialize checkbox keys for each ticker
for t in tickers:
    cb_key = f"cb_{t}"
    if cb_key not in st.session_state:
        st.session_state[cb_key] = True  # Default all visible

if "end_idx" not in st.session_state: st.session_state.end_idx = idx_len - 1
st.session_state.end_idx = min(max(st.session_state.end_idx, DEFAULT_TAIL), idx_len - 1)
if "view_mode" not in st.session_state: st.session_state.view_mode = "Fit"

# Animation
if st.session_state.playing:
    nxt = st.session_state.end_idx + 1
    if nxt > idx_len - 1:
        nxt = DEFAULT_TAIL if looping else idx_len - 1
        if not looping: st.session_state.playing = False
    st.session_state.end_idx = nxt
    if st_autorefresh: st_autorefresh(interval=speed_ms, limit=None, key="rrg_auto_refresh")
    else: components.html(f"<script>setTimeout(function(){{window.parent.location.reload()}},{int(speed_ms)});</script>", height=0)

end_idx = st.session_state.end_idx
start_idx = max(end_idx - tail_len, 0)
date_str = format_bar_date(idx[end_idx], interval)
start_date_str = format_bar_date(idx[DEFAULT_TAIL], interval)
end_date_full = format_bar_date(idx[-1], interval)

# Group tickers by quadrant
def get_ticker_quadrant(t):
    rr = float(rs_ratio_map[t].iloc[end_idx]) if not pd.isna(rs_ratio_map[t].iloc[end_idx]) else 100
    mm = float(rs_mom_map[t].iloc[end_idx]) if not pd.isna(rs_mom_map[t].iloc[end_idx]) else 100
    return get_status(rr, mm)

quadrant_tickers = {"Leading": [], "Improving": [], "Weakening": [], "Lagging": []}
for t in tickers:
    q = get_ticker_quadrant(t)
    if q in quadrant_tickers: quadrant_tickers[q].append(t)

def momentum_score(t):
    rr = float(rs_ratio_map[t].iloc[end_idx]) if not pd.isna(rs_ratio_map[t].iloc[end_idx]) else 100
    mm = float(rs_mom_map[t].iloc[end_idx]) if not pd.isna(rs_mom_map[t].iloc[end_idx]) else 100
    return np.hypot(rr - 100, mm - 100)

for q in quadrant_tickers: quadrant_tickers[q].sort(key=momentum_score, reverse=True)

# -------------------- Layout: Chart + Right Panel --------------------
main_col, right_col = st.columns([3.5, 1])

with main_col:
    # Timeline
    st.markdown(f'<div class="timeline-container"><div class="timeline-header"><span class="timeline-date">{date_str}</span><span class="timeline-range">{start_date_str} to {end_date_full}</span></div></div>', unsafe_allow_html=True)
    
    # Sparkline
    spark_fig = go.Figure()
    spark_fig.add_trace(go.Scatter(x=list(range(len(benchmark_data))), y=benchmark_data.values, mode='lines', line=dict(color='#3b82f6', width=1.5), fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)', hoverinfo='skip'))
    spark_fig.add_vline(x=end_idx, line_color='#ef4444', line_width=2)
    spark_fig.update_layout(height=50, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis=dict(showticklabels=False, showgrid=False, zeroline=False), yaxis=dict(showticklabels=False, showgrid=False, zeroline=False), showlegend=False)
    st.plotly_chart(spark_fig, use_container_width=True, config={'displayModeBar': False})
    
    # Slider
    end_idx = st.slider("Date", min_value=DEFAULT_TAIL, max_value=idx_len-1, step=1, key="end_idx", format=" ", label_visibility="collapsed")
    
    # Controls row with All/None buttons
    ctrl_cols = st.columns([1,1,1,1,1,1,1,1,2])
    
    with ctrl_cols[0]:
        if st.button("All", use_container_width=True, key="sel_all"):
            for t in tickers: 
                st.session_state[f"cb_{t}"] = True
            st.rerun()
    with ctrl_cols[1]:
        if st.button("None", use_container_width=True, key="clr_all"):
            for t in tickers: 
                st.session_state[f"cb_{t}"] = False
            st.rerun()
    
    def go_prev():
        if st.session_state.end_idx > DEFAULT_TAIL: st.session_state.end_idx -= 1
    def go_next():
        if st.session_state.end_idx < idx_len - 1: st.session_state.end_idx += 1
    def go_latest(): st.session_state.end_idx = idx_len - 1
    with ctrl_cols[2]: st.button("◀ Prev", use_container_width=True, on_click=go_prev)
    with ctrl_cols[3]: st.button("Next ▶", use_container_width=True, on_click=go_next)
    with ctrl_cols[4]: st.button("Latest", use_container_width=True, on_click=go_latest)
    with ctrl_cols[5]:
        if st.button("Fit", use_container_width=True): st.session_state.view_mode = "Fit"
    with ctrl_cols[6]:
        if st.button("Center", use_container_width=True): st.session_state.view_mode = "Center"
    with ctrl_cols[7]:
        if st.button("Max", use_container_width=True): st.session_state.view_mode = "Max"
    
    start_idx = max(end_idx - tail_len, 0)
    
    # Chart range
    all_rr, all_mm = [], []
    for t in tickers:
        # Read visibility from checkbox key
        is_visible = st.session_state.get(f"cb_{t}", True)
        if is_visible:
            rr = rs_ratio_map[t].iloc[start_idx:end_idx+1].dropna()
            mm = rs_mom_map[t].iloc[start_idx:end_idx+1].dropna()
            all_rr.extend(rr.values); all_mm.extend(mm.values)
    
    if st.session_state.view_mode == "Fit" and all_rr and all_mm:
        x_min, x_max = min(all_rr)-1, max(all_rr)+1
        y_min, y_max = min(all_mm)-1, max(all_mm)+1
        x_min, x_max = min(x_min, 99), max(x_max, 101)
        y_min, y_max = min(y_min, 99), max(y_max, 101)
    elif st.session_state.view_mode == "Center": x_min, x_max, y_min, y_max = 97, 103, 97, 103
    else: x_min, x_max, y_min, y_max = 94, 106, 94, 106
    
    # Build chart
    fig = go.Figure()
    fig.add_shape(type="rect", x0=x_min, y0=y_min, x1=100, y1=100, fillcolor=QUADRANT_BG_COLORS["Lagging"], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=100, y0=y_min, x1=x_max, y1=100, fillcolor=QUADRANT_BG_COLORS["Weakening"], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=100, y0=100, x1=x_max, y1=y_max, fillcolor=QUADRANT_BG_COLORS["Leading"], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=x_min, y0=100, x1=100, y1=y_max, fillcolor=QUADRANT_BG_COLORS["Improving"], line_width=0, layer="below")
    fig.add_hline(y=100, line_color="rgba(80,80,80,0.8)", line_width=1.5)
    fig.add_vline(x=100, line_color="rgba(80,80,80,0.8)", line_width=1.5)
    
    lox, loy = (x_max-x_min)*0.15, (y_max-y_min)*0.08
    fig.add_annotation(x=x_min+lox, y=y_max-loy, text="<b>IMPROVING</b>", showarrow=False, font=dict(size=13, color="#7c3aed"))
    fig.add_annotation(x=x_max-lox, y=y_max-loy, text="<b>LEADING</b>", showarrow=False, font=dict(size=13, color="#15803d"))
    fig.add_annotation(x=x_max-lox, y=y_min+loy, text="<b>WEAKENING</b>", showarrow=False, font=dict(size=13, color="#a16207"))
    fig.add_annotation(x=x_min+lox, y=y_min+loy, text="<b>LAGGING</b>", showarrow=False, font=dict(size=13, color="#dc2626"))
    
    for t in tickers:
        if not st.session_state.get(f"cb_{t}", True): continue
        rr = rs_ratio_map[t].iloc[start_idx+1:end_idx+1].dropna()
        mm = rs_mom_map[t].iloc[start_idx+1:end_idx+1].dropna()
        rr, mm = rr.align(mm, join="inner")
        if len(rr) < 2: continue
        
        name = META.get(t, {}).get("name", t)
        rr_last, mm_last = float(rr.values[-1]), float(mm.values[-1])
        color = status_color(rr_last, mm_last)
        
        px = tickers_data[t].reindex(idx).dropna()
        price = float(px.iloc[end_idx]) if end_idx < len(px) else np.nan
        chg = ((px.iloc[end_idx]/px.iloc[start_idx]-1)*100) if (end_idx < len(px) and start_idx < len(px)) else np.nan
        
        hover = f"<b>{name}</b><br>Status: {get_status(rr_last, mm_last)}<br>RS-Ratio: {rr_last:.2f}<br>RS-Mom: {mm_last:.2f}<br>Price: ₹{price:,.2f}<br>Chg: {chg:+.2f}%"
        
        x_pts, y_pts = rr.values.astype(float), mm.values.astype(float)
        n_original = len(x_pts)
        
        if n_original >= 3: x_smooth, y_smooth = smooth_spline_curve(x_pts, y_pts, points_per_segment=8)
        else: x_smooth, y_smooth = x_pts, y_pts
        
        n_smooth = len(x_smooth)
        if n_smooth >= 2:
            for i in range(n_smooth - 1):
                prog = i / max(1, n_smooth - 2)
                fig.add_trace(go.Scatter(x=[x_smooth[i], x_smooth[i+1]], y=[y_smooth[i], y_smooth[i+1]], mode='lines', line=dict(color=color, width=2.5+prog*3), opacity=0.5+prog*0.5, hoverinfo='skip', showlegend=False))
        
        trail_sizes = [5 + (i/max(1, n_original-1))*5 for i in range(n_original)]
        if n_original > 1:
            fig.add_trace(go.Scatter(x=x_pts[:-1], y=y_pts[:-1], mode='markers', marker=dict(size=trail_sizes[:-1], color=color, opacity=0.8, line=dict(color='white', width=1)), hoverinfo='skip', showlegend=False))
        
        fig.add_trace(go.Scatter(x=[rr_last], y=[mm_last], mode='markers', marker=dict(size=14, color=color, line=dict(color='white', width=2.5)), text=[hover], hoverinfo='text', showlegend=False))
        
        if n_original >= 2:
            dx, dy = x_pts[-1]-x_pts[-2], y_pts[-1]-y_pts[-2]
            length = np.sqrt(dx**2+dy**2)
            if length > 0.01:
                fig.add_annotation(x=x_pts[-1], y=y_pts[-1], ax=x_pts[-1]-dx/length*0.35, ay=y_pts[-1]-dy/length*0.35, xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, arrowsize=1.8, arrowwidth=3, arrowcolor=color)
        
        if show_labels:
            fig.add_annotation(x=rr_last, y=mm_last, text=f"<b>{name}</b>", showarrow=True, arrowhead=0, arrowwidth=1.5, arrowcolor=color, ax=30, ay=-25, font=dict(size=11, color=color), bgcolor='rgba(0,0,0,0)', borderwidth=0)
    
    fig.update_layout(
        title=dict(text=f"<b>Relative Rotation Graph</b> | {date_str}", font=dict(size=18, color='#e6eaee'), x=0.5),
        xaxis=dict(title="<b>JdK RS-Ratio</b>", range=[x_min, x_max], showgrid=True, gridcolor='rgba(150,150,150,0.2)', tickfont=dict(color='#b3bdc7')),
        yaxis=dict(title="<b>JdK RS-Momentum</b>", range=[y_min, y_max], showgrid=True, gridcolor='rgba(150,150,150,0.2)', tickfont=dict(color='#b3bdc7')),
        plot_bgcolor='#fafafa', paper_bgcolor='#0b0e13', margin=dict(l=60, r=30, t=60, b=60), height=620
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})

# Right panel - Quadrant checkboxes
with right_col:
    st.markdown("**Symbols**")
    
    q_icons = {"Leading": "🟢", "Improving": "🟣", "Weakening": "🟡", "Lagging": "🔴"}
    for quadrant in ["Leading", "Improving", "Weakening", "Lagging"]:
        q_tickers = quadrant_tickers[quadrant]
        with st.expander(f"{q_icons[quadrant]} {quadrant} ({len(q_tickers)})", expanded=(quadrant in ["Leading", "Improving"])):
            for t in q_tickers:
                name = META.get(t, {}).get("name", t)
                st.checkbox(
                    f"{name[:20]}{'...' if len(name)>20 else ''}", 
                    key=f"cb_{t}"
                )

# -------------------- Interactive Table --------------------
def make_interactive_table(rows):
    table_id = "rrg_tbl_" + str(abs(hash(str(len(rows)))) % 10000)
    headers = ["Rank", "Name", "Status", "Industry", "RS-Ratio", "RS-Mom", "Score", "Price", "Chg %"]
    header_keys = ["rank", "name", "status", "industry", "rs_ratio", "rs_mom", "score", "price", "chg"]
    
    th = "<tr>" + "".join(f'<th data-col="{i}" data-key="{header_keys[i]}" class="{"sort-asc" if i==0 else ""}"><span>{h}</span><span class="sort-icon"></span></th>' for i, h in enumerate(headers)) + "</tr>"
    
    tr_list = []
    for r in rows:
        rr_val = r["rs_ratio"] if not pd.isna(r["rs_ratio"]) else 0
        mm_val = r["rs_mom"] if not pd.isna(r["rs_mom"]) else 0
        score_val = r["score"] if not pd.isna(r["score"]) else 0
        price_val = r["price"] if not pd.isna(r["price"]) else 0
        chg_val = r["chg"] if not pd.isna(r["chg"]) else 0
        
        rr_txt = f"{r['rs_ratio']:.2f}" if not pd.isna(r['rs_ratio']) else "-"
        mm_txt = f"{r['rs_mom']:.2f}" if not pd.isna(r['rs_mom']) else "-"
        score_txt = f"{r['score']:.2f}" if not pd.isna(r['score']) else "-"
        price_txt = f"₹{r['price']:,.2f}" if not pd.isna(r['price']) else "-"
        chg_txt = f"{r['chg']:+.2f}%" if not pd.isna(r['chg']) else "-"
        chg_color = "#4ade80" if chg_val > 0 else "#f87171" if chg_val < 0 else "#9ca3af"
        
        safe_name = r['name'].replace("'", "&#39;").lower()
        safe_industry = r['industry'].replace("'", "&#39;").lower()
        
        tr_list.append(f"<tr class='rrg-row' data-rank='{r['rank']}' data-name='{safe_name}' data-status='{r['status'].lower()}' data-industry='{safe_industry}' data-rs_ratio='{rr_val}' data-rs_mom='{mm_val}' data-score='{score_val}' data-price='{price_val}' data-chg='{chg_val}'><td style='text-align:center;color:#8b949e;font-weight:700;'>{r['rank']}</td><td class='rrg-name'><a href='{r['tv']}' target='_blank'>{r['name']}</a></td><td><span style='background:{r['bg']};color:#fff;padding:3px 8px;border-radius:4px;font-size:11px;font-weight:700;'>{r['status']}</span></td><td style='color:#8b949e;font-size:12px;'>{r['industry']}</td><td>{rr_txt}</td><td>{mm_txt}</td><td style='color:#a78bfa;font-weight:600;'>{score_txt}</td><td>{price_txt}</td><td style='color:{chg_color};font-weight:600;'>{chg_txt}</td></tr>")
    
    statuses = sorted(set(r['status'] for r in rows))
    industries = sorted(set(r['industry'] for r in rows if r['industry'] != '-'))
    status_opts = '<option value="">All Statuses</option>' + ''.join(f'<option value="{s.lower()}">{s}</option>' for s in statuses)
    industry_opts = '<option value="">All Industries</option>' + ''.join(f'<option value="{i.lower()}">{i}</option>' for i in industries)
    
    return f"""<!DOCTYPE html><html><head><style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&display=swap');
*{{box-sizing:border-box;margin:0;padding:0}}body{{font-family:'Plus Jakarta Sans',sans-serif;background:#10141b;color:#e6eaee;padding:12px}}
.filter-row{{display:flex;gap:10px;margin-bottom:12px;flex-wrap:wrap;align-items:center}}
.filter-row input,.filter-row select{{background:#0b0e13;border:1px solid #1f2732;color:#e6eaee;padding:8px 12px;border-radius:8px;font-family:inherit;font-size:13px}}
.filter-row input:focus,.filter-row select:focus{{outline:none;border-color:#7a5cff}}.filter-row input::placeholder{{color:#8b949e}}
.filter-badge{{background:#7a5cff;color:white;padding:5px 12px;border-radius:20px;font-size:12px;font-weight:700}}
.table-wrap{{max-height:450px;overflow:auto;border:1px solid #1f2732;border-radius:10px}}table{{width:100%;border-collapse:collapse}}
th,td{{border-bottom:1px solid #1a2230;padding:10px 12px;font-size:13px;text-align:left}}
th{{position:sticky;top:0;z-index:2;background:#121823;color:#8b949e;font-weight:700;cursor:pointer;user-select:none;white-space:nowrap}}
th:hover{{background:#1a2233}}.sort-icon{{margin-left:6px;opacity:0.5;font-size:10px}}
th.sort-asc .sort-icon::after{{content:'▲';opacity:1}}th.sort-desc .sort-icon::after{{content:'▼';opacity:1}}th:not(.sort-asc):not(.sort-desc) .sort-icon::after{{content:'⇅'}}
.rrg-row{{background:#0d1117;transition:background 0.15s}}.rrg-row:hover{{background:#161b22}}.rrg-row:nth-child(even){{background:#0f1419}}
.rrg-name a{{color:#58a6ff;text-decoration:none;font-weight:600}}.rrg-name a:hover{{text-decoration:underline}}tr.hidden-row{{display:none}}
.table-wrap::-webkit-scrollbar{{height:10px;width:10px}}.table-wrap::-webkit-scrollbar-thumb{{background:#2e3745;border-radius:8px}}
</style></head><body>
<div class="filter-row"><input type="text" id="{table_id}_search" placeholder="🔍 Search..." style="min-width:180px"><select id="{table_id}_status">{status_opts}</select><select id="{table_id}_industry">{industry_opts}</select><span class="filter-badge" id="{table_id}_count">{len(rows)} / {len(rows)}</span></div>
<div class="table-wrap"><table id="{table_id}"><thead>{th}</thead><tbody>{''.join(tr_list)}</tbody></table></div>
<script>(function(){{const table=document.getElementById('{table_id}'),headers=table.querySelectorAll('th'),tbody=table.querySelector('tbody'),rows=Array.from(tbody.querySelectorAll('tr')),searchInput=document.getElementById('{table_id}_search'),statusFilter=document.getElementById('{table_id}_status'),industryFilter=document.getElementById('{table_id}_industry'),countBadge=document.getElementById('{table_id}_count');let currentSort={{col:0,asc:true}};function updateCount(){{countBadge.textContent=rows.filter(r=>!r.classList.contains('hidden-row')).length+' / '+rows.length}}function sortTable(colIndex,key){{const isNumeric=['rank','rs_ratio','rs_mom','score','price','chg'].includes(key),asc=currentSort.col===colIndex?!currentSort.asc:(colIndex===0);currentSort={{col:colIndex,asc}};headers.forEach((h,i)=>{{h.classList.remove('sort-asc','sort-desc');if(i===colIndex)h.classList.add(asc?'sort-asc':'sort-desc')}});rows.sort((a,b)=>{{let aVal=a.dataset[key]||'',bVal=b.dataset[key]||'';if(isNumeric){{aVal=parseFloat(aVal)||0;bVal=parseFloat(bVal)||0;return asc?aVal-bVal:bVal-aVal}}return asc?aVal.localeCompare(bVal):bVal.localeCompare(aVal)}});rows.forEach(row=>tbody.appendChild(row))}}function filterTable(){{const searchTerm=searchInput.value.toLowerCase(),statusTerm=statusFilter.value,industryTerm=industryFilter.value;rows.forEach(row=>{{const name=row.dataset.name||'',status=row.dataset.status||'',industry=row.dataset.industry||'';if((!searchTerm||name.includes(searchTerm))&&(!statusTerm||status===statusTerm)&&(!industryTerm||industry===industryTerm))row.classList.remove('hidden-row');else row.classList.add('hidden-row')}});updateCount()}}headers.forEach((h,i)=>h.addEventListener('click',()=>sortTable(i,h.dataset.key)));searchInput.addEventListener('input',filterTable);statusFilter.addEventListener('change',filterTable);industryFilter.addEventListener('change',filterTable);updateCount()}})();</script>
</body></html>""", min(550, 120 + len(rows) * 40)

# Build rows
def ranking_value(t):
    rr_last = rs_ratio_map[t].iloc[end_idx]
    mm_last = rs_mom_map[t].iloc[end_idx]
    return float(np.hypot(rr_last - 100.0, mm_last - 100.0))

perf = [(t, ranking_value(t)) for t in tickers]
perf.sort(key=lambda x: x[1], reverse=True)

rows = []
for rank, (t, _) in enumerate(perf, 1):
    rr = float(rs_ratio_map[t].iloc[end_idx]) if not pd.isna(rs_ratio_map[t].iloc[end_idx]) else np.nan
    mm = float(rs_mom_map[t].iloc[end_idx]) if not pd.isna(rs_mom_map[t].iloc[end_idx]) else np.nan
    status = get_status(rr, mm)
    px = tickers_data[t].reindex(idx).dropna()
    price = float(px.iloc[end_idx]) if end_idx < len(px) else np.nan
    chg = ((px.iloc[end_idx]/px.iloc[start_idx]-1)*100) if (end_idx < len(px) and start_idx < len(px)) else np.nan
    score = np.hypot(rr-100, mm-100) if not (pd.isna(rr) or pd.isna(mm)) else np.nan
    rows.append({"rank": rank, "name": META.get(t, {}).get("name", t), "status": status, "industry": META.get(t, {}).get("industry", "-"), "rs_ratio": rr, "rs_mom": mm, "score": score, "price": price, "chg": chg, "bg": status_color(rr, mm), "tv": tv_link_for_symbol(t)})

with st.expander("📊 Full Rankings Table", expanded=True):
    table_html, table_height = make_interactive_table(rows)
    components.html(table_html, height=table_height, scrolling=False)

# Download
def export_csv(rows_):
    df = pd.DataFrame([{"rank": r["rank"], "name": r["name"], "industry": r["industry"], "status": r["status"], "rs_ratio": r["rs_ratio"], "rs_mom": r["rs_mom"], "score": r["score"], "price": r["price"], "chg": r["chg"]} for r in rows_])
    buf = io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode()

st.download_button("📥 Download CSV", data=export_csv(rows), file_name=f"rrg_{date_str}.csv", mime="text/csv", use_container_width=True)
st.caption("Click names to open TradingView. Checkboxes control chart visibility - table shows all.")

import os, json, time, pathlib, logging, functools, calendar, io
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

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.patches import FancyArrowPatch

# -------------------- Defaults --------------------
DEFAULT_TF = "Daily"
DEFAULT_PERIOD = "1Y"

# -------------------- GitHub CSVs -----------------
GITHUB_USER = "anki1007"
GITHUB_REPO = "rrg-stocks"
GITHUB_BRANCH = "main"
GITHUB_TICKER_DIR = "ticker"
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_TICKER_DIR}/"

# -------------------- Matplotlib ------------------
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.edgecolor'] = '#222'
mpl.rcParams['axes.labelcolor'] = '#111'
mpl.rcParams['xtick.color'] = '#333'
mpl.rcParams['ytick.color'] = '#333'
mpl.rcParams['font.size'] = 13
mpl.rcParams['font.sans-serif'] = ['Inter','Segoe UI','DejaVu Sans','Arial']

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# -------------------- Streamlit page --------------
st.set_page_config(page_title="Relative Rotation Graph (RRG)", layout="wide")
st.markdown("""
<style>
:root { color-scheme: light !important; }
[data-testid="stAppViewContainer"] { background: #ffffff !important; }
[data-testid="stHeader"] { background: #ffffff00; }
.block-container { padding-top: 1.0rem; }

/* Global text bump for readability */
html, body, [data-testid="stSidebar"], [data-testid="stMarkdownContainer"] { font-size: 16px; }

/* Fix low-contrast headings/labels */
h1, h2, h3, h4, h5, h6, strong, b { color:#0f172a !important; }
[data-testid="stMarkdownContainer"] h3 { font-weight: 800; }
[data-testid="stSlider"] label, label span, .st-cq { color:#0f172a !important; }

/* Ranking list */
.rrg-rank { font-weight: 700; line-height: 1.25; font-size: 1.05rem; white-space: pre; }
.rrg-rank .row { display: flex; gap: 8px; align-items: baseline; margin: 2px 0; }
.rrg-rank .name { color: #0b57d0; }

/* Status badge styling */
.status-badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 13px;
    text-align: center;
    min-width: 80px;
}
.status-leading { background: #22c55e; color: white; }
.status-improving { background: #3b82f6; color: white; }
.status-weakening { background: #eab308; color: #1a1a1a; }
.status-lagging { background: #ef4444; color: white; }

/* Scrollable table wrapper with sticky header */
.rrg-wrap {
  max-height: calc(100vh - 260px);
  overflow: auto;
  border: 1px solid #e5e5e5; border-radius: 6px;
}
.rrg-table { width: 100%; border-collapse: collapse; font-family: 'Segoe UI', -apple-system, Arial, sans-serif; }
.rrg-table th, .rrg-table td { border-bottom: 1px solid #ececec; padding: 10px 10px; font-size: 15px; }
.rrg-table th {
  position: sticky; top: 0; z-index: 2;
  text-align: left; background: #eef2f7; color: #0f172a; font-weight: 800; letter-spacing: .2px;
}
.rrg-row { transition: background .12s ease; background: #ffffff; }
.rrg-row:hover { background: #f8fafc; }
.rrg-name a { color: #0b57d0; text-decoration: underline; }

/* Make scrollbars visible on WebKit (Chrome/Edge) */
.rrg-wrap::-webkit-scrollbar { height: 12px; width: 12px; }
.rrg-wrap::-webkit-scrollbar-thumb { background:#c7ccd6; border-radius: 8px; }

/* Positive/negative change colors */
.positive-change { color: #16a34a; font-weight: 600; }
.negative-change { color: #dc2626; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

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
    "nifty50.csv":"Nifty 50","nifty200.csv":"Nifty 200","nifty500.csv":"Nifty 500",
    "niftymidcap150.csv":"Nifty Midcap 150","niftysmallcap250.csv":"Nifty Smallcap 250",
    "niftymidsmallcap400.csv":"Nifty MidSmallcap 400","niftytotalmarket.csv":"Nifty Total Market",
}
def friendly_name_from_file(b: str) -> str:
    b2=b.lower()
    if b2 in _FRIENDLY: return _FRIENDLY[b2]
    core=os.path.splitext(b)[0].replace("_"," ").replace("-"," ")
    out=""
    for ch in core:
        out += (" "+ch) if (ch.isdigit() and out and (out[-1]!=" " and not out[-1].isdigit())) else ch
    return out.title()

def build_name_maps_from_github():
    files = list_csv_files_from_github(GITHUB_USER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_TICKER_DIR)
    name_map = {friendly_name_from_file(f): f for f in files}
    return name_map, sorted(name_map.keys())

# -------------------- Universe CSV -----------------
def _normalize_cols(cols: List[str]) -> Dict[str, str]:
    return {c: c.strip().lower().replace(" ","").replace("_","") for c in cols}

@st.cache_data(ttl=600)
def load_universe_from_github_csv(basename: str):
    url = RAW_BASE + basename
    df = pd.read_csv(url)
    mapping = _normalize_cols(df.columns.tolist())
    sym_col = next((c for c,k in mapping.items() if k in ("symbol","ticker","symbols")), None)
    if sym_col is None: raise ValueError("CSV must contain a 'Symbol' column.")
    name_col = next((c for c,k in mapping.items() if k in ("companyname","name","company","companyfullname")), sym_col)
    ind_col  = next((c for c,k in mapping.items() if k in ("industry","sector","industries")), None)
    if ind_col is None: ind_col = "Industry"; df[ind_col] = "-"
    sel = df[[sym_col, name_col, ind_col]].copy()
    sel.columns = ["Symbol","Company Name","Industry"]
    sel["Symbol"]=sel["Symbol"].astype(str).str.strip()
    sel["Company Name"]=sel["Company Name"].astype(str).str.strip()
    sel["Industry"]=sel["Industry"].astype(str).str.strip()
    sel = sel[sel["Symbol"]!=""].drop_duplicates(subset=["Symbol"])
    universe = sel["Symbol"].tolist()
    meta = {r["Symbol"]:{ "name":r["Company Name"] or r["Symbol"], "industry":r["Industry"] or "-" } for _,r in sel.iterrows()}
    return universe, meta

# -------------------- Config & utils ----------------
PERIOD_MAP = {"6M":"6mo","1Y":"1y","2Y":"2y","3Y":"3y","5Y":"5y","10Y":"10y"}
TF_LABELS = ["Daily","Weekly","Monthly"]
TF_TO_INTERVAL = {"Daily":"1d","Weekly":"1wk","Monthly":"1mo"}
WINDOW = 14
DEFAULT_TAIL = 8
BENCH_CHOICES = {"Nifty 500":"^CRSLDX","Nifty 200":"^CNX200","Nifty 50":"^NSEI"}
TOP_N_PER_QUADRANT = 25  # Maximum stocks to display per quadrant

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
    return sym[:-3] if sym.upper().endswith(".NS") else sym

def safe_long_name(symbol: str, META: dict) -> str:
    return META.get(symbol,{}).get("name") or symbol

def format_bar_date(ts: pd.Timestamp, interval: str) -> str:
    ts = pd.Timestamp(ts)
    if interval=="1wk": return ts.to_period("W-FRI").end_time.date().isoformat()
    if interval=="1mo": return ts.to_period("M").end_time.date().isoformat()
    return ts.date().isoformat()

def jdk_components(price: pd.Series, bench: pd.Series, win=14):
    df=pd.concat([price.rename("p"), bench.rename("b")], axis=1).dropna()
    if df.empty: return pd.Series(dtype=float), pd.Series(dtype=float)
    rs = 100 * (df["p"]/df["b"])
    m = rs.rolling(win).mean()
    s = rs.rolling(win).std(ddof=0).replace(0,np.nan).fillna(1e-9)
    rs_ratio = (100 + (rs-m)/s).dropna()
    rroc = rs_ratio.pct_change().mul(100)
    m2 = rroc.rolling(win).mean()
    s2 = rroc.rolling(win).std(ddof=0).replace(0,np.nan).fillna(1e-9)
    rs_mom = (101 + (rroc-m2)/s2).dropna()
    ix = rs_ratio.index.intersection(rs_mom.index)
    return rs_ratio.loc[ix], rs_mom.loc[ix]

def has_min_coverage(rr, mm, *, min_points=20, lookback_ok=30):
    if rr is None or mm is None: return False
    ok=(~rr.isna()) & (~mm.isna())
    if ok.sum()<min_points: return False
    tail = ok.iloc[-lookback_ok:] if len(ok)>=lookback_ok else ok
    return bool(tail.any())

def get_status(x, y):
    if x<=100 and y<=100: return "Lagging"
    if x>=100 and y>=100: return "Leading"
    if x<=100 and y>=100: return "Improving"
    if x>=100 and y<=100: return "Weakening"
    return "Unknown"

def status_bg_color(x,y):
    m=get_status(x,y)
    return {"Lagging":"#ef4444","Leading":"#22c55e","Improving":"#3b82f6","Weakening":"#eab308"}.get(m,"#aaaaaa")

def compute_momentum_score(rr: float, mm: float) -> float:
    """Compute momentum score as distance from center weighted by quadrant"""
    if np.isnan(rr) or np.isnan(mm):
        return 0.0
    # Distance from center (100, 100)
    dist = np.hypot(rr - 100.0, mm - 100.0)
    # Weight by quadrant (positive for Leading/Improving, negative for Lagging/Weakening)
    if rr >= 100 and mm >= 100:  # Leading
        return dist
    elif rr < 100 and mm >= 100:  # Improving
        return dist * 0.8
    elif rr >= 100 and mm < 100:  # Weakening
        return -dist * 0.6
    else:  # Lagging
        return -dist
    
def compute_rrg_power(rr: float, mm: float) -> float:
    """Compute RRG Power (distance from center)"""
    if np.isnan(rr) or np.isnan(mm):
        return 0.0
    return float(np.hypot(rr - 100.0, mm - 100.0))

# -------------------- Closed-bar enforcement --------
IST_TZ="Asia/Kolkata"; BAR_CUTOFF_HOUR=10; NET_TIME_MAX_AGE=300
def _utc_now_from_network(timeout=2.5) -> pd.Timestamp:
    try:
        import ntplib
        c=ntplib.NTPClient()
        for host in ("time.google.com","time.cloudflare.com","pool.ntp.org","asia.pool.ntp.org"):
            try:
                r=c.request(host, version=3, timeout=timeout)
                return pd.Timestamp(r.tx_time, unit="s", tz="UTC")
            except Exception: continue
    except Exception: pass
    for url in ("https://www.google.com/generate_204","https://www.cloudflare.com","https://www.nseindia.com","https://www.bseindia.com"):
        try:
            req=_urlreq.Request(url, method="HEAD")
            with _urlreq.urlopen(req, timeout=timeout) as resp:
                date_hdr=resp.headers.get("Date")
                if date_hdr:
                    dt=_eutils.parsedate_to_datetime(date_hdr)
                    if dt.tzinfo is None: dt=dt.replace(tzinfo=_dt.timezone.utc)
                    return pd.Timestamp(dt).tz_convert("UTC")
        except Exception: continue
    return pd.Timestamp.now(tz="UTC")

@st.cache_data(ttl=NET_TIME_MAX_AGE)
def _now_ist_cached():
    return _utc_now_from_network().tz_convert(IST_TZ)

def _to_ist(ts):
    ts=pd.Timestamp(ts)
    if ts.tzinfo is None: ts=ts.tz_localize("UTC")
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

# -------------------- Cache / Download --------------
CACHE_DIR = pathlib.Path("cache"); CACHE_DIR.mkdir(exist_ok=True)
def _cache_path(symbol, period, interval):
    safe=symbol.replace("^","").replace(".","_")
    return CACHE_DIR/f"{safe}_{period}_{interval}.parquet"
def _save_cache(symbol,s,period,interval):
    try: s.to_frame("Close").to_parquet(_cache_path(symbol,period,interval))
    except Exception: pass
def _load_cache(symbol,period,interval):
    p=_cache_path(symbol,period,interval)
    if p.exists():
        try: return pd.read_parquet(p)["Close"].dropna()
        except Exception: pass
    return pd.Series(dtype=float)

def retry(n=4, delay=1.5, backoff=2.0):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **k):
            d=delay
            for i in range(n):
                try: return fn(*a, **k)
                except Exception as e:
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
    if bench is None or bench.empty: return bench, {}
    drop_last = not _is_bar_complete_for_timestamp(bench.index[-1], interval, now=_now_ist_cached())
    def _maybe_trim(s): return s.iloc[:-1] if (drop_last and len(s)>=1) else s
    bench=_maybe_trim(bench)
    data={}
    for t in universe:
        s=_pick(t)
        if not s.empty: data[t]=_maybe_trim(s)
        else:
            c=_load_cache(t,period,interval)
            if not c.empty: data[t]=_maybe_trim(c)
    if not bench.empty: _save_cache(benchmark, bench, period, interval)
    for t,s in data.items(): _save_cache(t, s, period, interval)
    return bench, data

def symbol_color_map(symbols):
    tab=plt.get_cmap('tab20').colors
    return {s: to_hex(tab[i%len(tab)], keep_alpha=False) for i,s in enumerate(symbols)}

# -------------------- Controls (left) ----------------
st.sidebar.header("RRG — Controls")

NAME_MAP, DISPLAY_LIST = build_name_maps_from_github()
if not DISPLAY_LIST:
    st.error("No CSVs found in GitHub /ticker.")
    st.stop()

csv_disp = st.sidebar.selectbox("Indices", DISPLAY_LIST, index=(DISPLAY_LIST.index("Nifty 200") if "Nifty 200" in DISPLAY_LIST else 0))
csv_basename = NAME_MAP[csv_disp]

bench_label = st.sidebar.selectbox("Benchmark", list(BENCH_CHOICES.keys()), index=list(BENCH_CHOICES.keys()).index("Nifty 500"))
interval_label = st.sidebar.selectbox("Strength vs (TF)", TF_LABELS, index=TF_LABELS.index(DEFAULT_TF))
interval = TF_TO_INTERVAL[interval_label]
default_period_for_tf = {"1d":"1Y","1wk":"3Y","1mo":"10Y"}[interval]
period_label = st.sidebar.selectbox("Period", list(PERIOD_MAP.keys()),
                                    index=list(PERIOD_MAP.keys()).index(default_period_for_tf))
period = PERIOD_MAP[period_label]

rank_modes = ["RRG Power (dist)","RS-Ratio","RS-Momentum","Price %Δ (tail)","Momentum Slope (tail)"]
rank_mode = st.sidebar.selectbox("Rank by", rank_modes, index=0)
tail_len = st.sidebar.slider("Trail Length", 1, 20, DEFAULT_TAIL, 1)

# Optional: label toggle (default OFF to avoid clutter)
show_labels = st.sidebar.toggle("Show labels on chart", value=False)
label_top_n = st.sidebar.slider("Label top N by distance", 3, 30, 12, 1, disabled=not show_labels)

# Top N per quadrant control
top_n_per_quadrant = st.sidebar.slider("Max stocks per quadrant", 5, 50, TOP_N_PER_QUADRANT, 5)

# ---------- Playback controls ----------
if "playing" not in st.session_state: st.session_state.playing = False
play_toggle = st.sidebar.toggle("Play / Pause", value=st.session_state.playing, key="playing")
speed_ms = st.sidebar.slider("Speed (ms/frame)", 1000, 3000, 1500, 50)
looping = st.sidebar.checkbox("Loop", value=True)

# -------------------- Data build ---------------------
UNIVERSE, META = load_universe_from_github_csv(csv_basename)
bench_symbol = BENCH_CHOICES[bench_label]
benchmark_data, tickers_data = download_block_with_benchmark(UNIVERSE, bench_symbol, period, interval)
if benchmark_data is None or benchmark_data.empty:
    st.error("Benchmark returned no data.")
    st.stop()

bench_idx = benchmark_data.index
rs_ratio_map, rs_mom_map, kept = {}, {}, []
for t,s in tickers_data.items():
    if t==bench_symbol: continue
    rr, mm = jdk_components(s, benchmark_data, WINDOW)
    if len(rr)==0 or len(mm)==0: continue
    rr=rr.reindex(bench_idx); mm=mm.reindex(bench_idx)
    if has_min_coverage(rr, mm, min_points=max(WINDOW+5,20), lookback_ok=30):
        rs_ratio_map[t]=rr; rs_mom_map[t]=mm; kept.append(t)

if not kept:
    st.warning("After alignment, no symbols have enough coverage. Try a longer period.")
    st.stop()

tickers = kept
SYMBOL_COLORS = symbol_color_map(tickers)
idx = bench_idx; idx_len = len(idx)

# -------------------- Date index + animation ----------
if "end_idx" not in st.session_state:
    st.session_state.end_idx = idx_len - 1
st.session_state.end_idx = min(max(st.session_state.end_idx, DEFAULT_TAIL), idx_len - 1)

if st.session_state.playing:
    nxt = st.session_state.end_idx + 1
    if nxt > idx_len - 1:
        if looping:
            nxt = DEFAULT_TAIL
        else:
            nxt = idx_len - 1
            st.session_state.playing = False
    st.session_state.end_idx = nxt

    # advance to next frame after a delay, then rerun
    time.sleep(speed_ms / 1000.0)
    try:
        st.rerun()               # Streamlit ≥ 1.30
    except AttributeError:
        st.experimental_rerun()  # Older Streamlit fallback

end_idx = st.slider("Date", min_value=DEFAULT_TAIL, max_value=idx_len-1,
                    value=st.session_state.end_idx, step=1, key="end_idx",
                    format=" ", help="RRG date position (closed bars only).")

start_idx = max(end_idx - tail_len, 0)
date_str = format_bar_date(idx[end_idx], interval)

# -------------------- Title -------------------------
st.markdown(f"**Relative Rotation Graph (RRG) — {bench_label} — {period_label} — {interval_label} — {csv_disp} — {date_str}**")

# -------------------- Categorize stocks by quadrant and select top N ----------
def categorize_by_quadrant(tickers, rs_ratio_map, rs_mom_map, end_idx, top_n=25):
    """Categorize stocks by quadrant and return top N per quadrant based on distance from center"""
    quadrants = {
        "Leading": [],
        "Improving": [],
        "Weakening": [],
        "Lagging": []
    }
    
    for t in tickers:
        rr = rs_ratio_map[t].iloc[end_idx]
        mm = rs_mom_map[t].iloc[end_idx]
        if np.isnan(rr) or np.isnan(mm):
            continue
        
        status = get_status(rr, mm)
        dist = np.hypot(rr - 100.0, mm - 100.0)
        quadrants[status].append((t, dist, rr, mm))
    
    # Sort each quadrant by distance (farthest from center first) and take top N
    selected = set()
    for status in quadrants:
        quadrants[status].sort(key=lambda x: x[1], reverse=True)
        for t, _, _, _ in quadrants[status][:top_n]:
            selected.add(t)
    
    return selected, quadrants

# Get top N stocks per quadrant
selected_tickers, quadrant_data = categorize_by_quadrant(tickers, rs_ratio_map, rs_mom_map, end_idx, top_n_per_quadrant)

if "visible_set" not in st.session_state:
    st.session_state.visible_set = set(tickers)

# Filter visible set to only include selected tickers
display_tickers = st.session_state.visible_set.intersection(selected_tickers)

# -------------------- Build stock data for tooltips ----------
stock_info = {}
for t in tickers:
    rr = float(rs_ratio_map[t].iloc[end_idx])
    mm = float(rs_mom_map[t].iloc[end_idx])
    px = tickers_data[t].reindex(idx).dropna()
    price = float(px.iloc[end_idx]) if end_idx < len(px) else np.nan
    chg = ((px.iloc[end_idx]/px.iloc[start_idx]-1)*100.0) if (end_idx < len(px) and start_idx < len(px)) else np.nan
    
    stock_info[t] = {
        "name": safe_long_name(t, META),
        "status": get_status(rr, mm),
        "momentum_score": compute_momentum_score(rr, mm),
        "rs_ratio": rr,
        "rs_momentum": mm,
        "price": price,
        "change_pct": chg,
        "rrg_power": compute_rrg_power(rr, mm)
    }

# -------------------- Layout: Plot + Ranking ----------
plot_col, rank_col = st.columns([4.5, 1.8], gap="medium")

with plot_col:
    fig, ax = plt.subplots(1, 1, figsize=(10.6, 6.8))
    ax.set_title("Relative Rotation Graph (RRG)", fontsize=15, pad=10)
    ax.set_xlabel("JdK RS-Ratio", fontsize=14); ax.set_ylabel("JdK RS-Momentum", fontsize=14)
    ax.axhline(y=100, color="#777", linestyle=":", linewidth=1.1)
    ax.axvline(x=100, color="#777", linestyle=":", linewidth=1.1)
    ax.fill_between([94,100],[94,94],[100,100], color=(1.0,0.0,0.0,0.20))
    ax.fill_between([100,106],[94,94],[100,100], color=(1.0,1.0,0.0,0.20))
    ax.fill_between([100,106],[100,100],[106,106], color=(0.0,1.0,0.0,0.20))
    ax.fill_between([94,100],[100,100],[106,106], color=(0.0,0.0,1.0,0.20))
    ax.text(95,105,"Improving", fontsize=13, color="#111", weight="bold")
    ax.text(104,105,"Leading",   fontsize=13, color="#111", weight="bold", ha="right")
    ax.text(104,95,"Weakening",  fontsize=13, color="#111", weight="bold", ha="right")
    ax.text(95,95,"Lagging",     fontsize=13, color="#111", weight="bold")
    ax.set_xlim(94,106); ax.set_ylim(94,106)

    def dist_last(t):
        rr_last=rs_ratio_map[t].iloc[end_idx]; mm_last=rs_mom_map[t].iloc[end_idx]
        return float(np.hypot(rr_last-100.0, mm_last-100.0))

    label_allow_set = set()
    if show_labels:
        # Only consider display_tickers for labels
        label_allow_set = set([t for t,_ in sorted([(t, dist_last(t)) for t in display_tickers], key=lambda x:x[1], reverse=True)[:label_top_n]])

    # Store annotation data for interactive tooltips
    scatter_data = []
    
    for t in tickers:
        if t not in display_tickers: continue
        rr=rs_ratio_map[t].iloc[start_idx+1:end_idx+1].dropna()
        mm=rs_mom_map[t].iloc[start_idx+1:end_idx+1].dropna()
        rr,mm=rr.align(mm, join="inner")
        if len(rr)==0 or len(mm)==0: continue
        
        # Plot the trail
        ax.plot(rr.values, mm.values, linewidth=1.2, alpha=0.7, color=SYMBOL_COLORS[t])
        
        # Plot points with sizes (larger for the last point)
        sizes=[22]*(len(rr)-1)+[76]
        ax.scatter(rr.values, mm.values, s=sizes, linewidths=0.6,
                   facecolor=SYMBOL_COLORS[t], edgecolor="#333333")
        
        # Add arrowhead at the end to show direction
        if len(rr) >= 2:
            # Draw arrow from second-to-last to last point
            x_start, y_start = rr.values[-2], mm.values[-2]
            x_end, y_end = rr.values[-1], mm.values[-1]
            
            # Calculate arrow direction
            dx = x_end - x_start
            dy = y_end - y_start
            
            # Only draw arrow if there's meaningful movement
            if abs(dx) > 0.01 or abs(dy) > 0.01:
                ax.annotate('', xy=(x_end, y_end), xytext=(x_start, y_start),
                           arrowprops=dict(arrowstyle='->', color=SYMBOL_COLORS[t], 
                                          lw=1.8, mutation_scale=12))
        
        if show_labels and t in label_allow_set:
            rr_last, mm_last = rr.values[-1], mm.values[-1]
            ax.annotate(f"{t}", (rr_last, mm_last), fontsize=11, color=SYMBOL_COLORS[t],
                        xytext=(6,6), textcoords="offset points")
        
        # Store data for potential tooltip display
        scatter_data.append({
            "symbol": t,
            "name": stock_info[t]["name"],
            "x": rr.values[-1],
            "y": mm.values[-1]
        })
    
    st.pyplot(fig, use_container_width=True)
    
    # Display count of stocks shown per quadrant
    quad_counts = {q: len([t for t in display_tickers if get_status(rs_ratio_map[t].iloc[end_idx], rs_mom_map[t].iloc[end_idx]) == q]) for q in ["Leading", "Improving", "Weakening", "Lagging"]}
    st.caption(f"Showing top {top_n_per_quadrant} per quadrant: 🟢 Leading: {quad_counts['Leading']} | 🔵 Improving: {quad_counts['Improving']} | 🟡 Weakening: {quad_counts['Weakening']} | 🔴 Lagging: {quad_counts['Lagging']}")

with rank_col:
    st.markdown("### Ranking")
    def compute_rank_metric(t: str) -> float:
        rr_last=rs_ratio_map[t].iloc[end_idx]; mm_last=rs_mom_map[t].iloc[end_idx]
        if np.isnan(rr_last) or np.isnan(mm_last): return float("-inf")
        if rank_mode=="RRG Power (dist)": return float(np.hypot(rr_last-100.0, mm_last-100.0))
        if rank_mode=="RS-Ratio": return float(rr_last)
        if rank_mode=="RS-Momentum": return float(mm_last)
        if rank_mode=="Price %Δ (tail)":
            px=tickers_data[t].reindex(idx).dropna()
            return float((px.iloc[end_idx]/px.iloc[start_idx]-1)*100.0) if len(px.iloc[start_idx:end_idx+1])>=2 else float("-inf")
        if rank_mode=="Momentum Slope (tail)":
            series=rs_mom_map[t].iloc[start_idx:end_idx+1].dropna()
            if len(series)>=2:
                x=np.arange(len(series)); A=np.vstack([x, np.ones(len(x))]).T
                return float(np.linalg.lstsq(A, series.values, rcond=None)[0][0])
            return float("-inf")
        return float("-inf")

    perf=[]
    for t in tickers:
        if t not in display_tickers: continue
        perf.append((t, compute_rank_metric(t)))
    perf.sort(key=lambda x:x[1], reverse=True)

    if not perf:
        st.write("—")
    else:
        rows_html=[]
        for i,(sym,_m) in enumerate(perf[:22], start=1):
            rr=float(rs_ratio_map[sym].iloc[end_idx]); mm=float(rs_mom_map[sym].iloc[end_idx])
            stat=get_status(rr, mm)
            color=SYMBOL_COLORS.get(sym, "#333")
            name=safe_long_name(sym, META)
            rows_html.append(
                f'<div class="row" style="color:{color}"><span>{i}.</span>'
                f'<span class="name">{name}</span>'
                f'<span>[{stat}]</span></div>'
            )
        st.markdown(f'<div class="rrg-rank">{"".join(rows_html)}</div>', unsafe_allow_html=True)

# -------------------- Interactive Tooltip Section ----------
st.markdown("### 📊 Stock Details (Hover Info)")
st.markdown("*Click on a stock in the table below to see detailed information*")

# -------------------- Table under the plot with sorting and filtering -----------
def get_status_badge(status):
    """Return HTML for status badge with appropriate color"""
    status_class = f"status-{status.lower()}"
    return f'<span class="status-badge {status_class}">{status}</span>'

def format_change(val):
    """Format change percentage with color"""
    if pd.isna(val):
        return "-"
    cls = "positive-change" if val >= 0 else "negative-change"
    sign = "+" if val >= 0 else ""
    return f'<span class="{cls}">{sign}{val:.2f}%</span>'

# Build dataframe for the table
table_data = []
for t in tickers:
    if t not in st.session_state.visible_set: continue
    info = stock_info[t]
    tv_link = f'https://www.tradingview.com/chart/?symbol={quote("NSE:"+display_symbol(t).replace("-","_"), safe="")}'
    
    table_data.append({
        "Symbol": display_symbol(t),
        "Name": info["name"],
        "Status": info["status"],
        "Industry": META.get(t,{}).get("industry","-"),
        "Price": info["price"],
        "Change %": info["change_pct"],
        "Momentum Score": info["momentum_score"],
        "RS-Ratio": info["rs_ratio"],
        "RS-Momentum": info["rs_momentum"],
        "RRG Power": info["rrg_power"],
        "TV Link": tv_link,
        "_raw_symbol": t
    })

df_table = pd.DataFrame(table_data)

# Collapsible + interactive table with filtering and sorting
with st.expander("📋 Detailed Table (Sortable & Filterable)", expanded=True):
    # Add filters
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        status_filter = st.multiselect(
            "Filter by Status",
            options=["Leading", "Improving", "Weakening", "Lagging"],
            default=[]
        )
    
    with filter_col2:
        industry_options = sorted(df_table["Industry"].unique().tolist())
        industry_filter = st.multiselect(
            "Filter by Industry",
            options=industry_options,
            default=[]
        )
    
    with filter_col3:
        search_term = st.text_input("🔍 Search by Name/Symbol", "")
    
    # Apply filters
    filtered_df = df_table.copy()
    
    if status_filter:
        filtered_df = filtered_df[filtered_df["Status"].isin(status_filter)]
    
    if industry_filter:
        filtered_df = filtered_df[filtered_df["Industry"].isin(industry_filter)]
    
    if search_term:
        search_lower = search_term.lower()
        filtered_df = filtered_df[
            filtered_df["Name"].str.lower().str.contains(search_lower, na=False) |
            filtered_df["Symbol"].str.lower().str.contains(search_lower, na=False)
        ]
    
    # Display count
    st.markdown(f"**Showing {len(filtered_df)} of {len(df_table)} stocks**")
    
    # Create display dataframe with formatted columns
    display_df = filtered_df[["Symbol", "Name", "Status", "Industry", "Price", "Change %", 
                              "Momentum Score", "RS-Ratio", "RS-Momentum", "RRG Power"]].copy()
    
    # Format numeric columns
    display_df["Price"] = display_df["Price"].apply(lambda x: f"₹{x:.2f}" if not pd.isna(x) else "-")
    display_df["Change %"] = display_df["Change %"].apply(lambda x: f"{x:+.2f}%" if not pd.isna(x) else "-")
    display_df["Momentum Score"] = display_df["Momentum Score"].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "-")
    display_df["RS-Ratio"] = display_df["RS-Ratio"].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "-")
    display_df["RS-Momentum"] = display_df["RS-Momentum"].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "-")
    display_df["RRG Power"] = display_df["RRG Power"].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "-")
    
    # Use Streamlit's native dataframe with sorting
    st.dataframe(
        display_df,
        use_container_width=True,
        height=400,
        column_config={
            "Symbol": st.column_config.TextColumn("Symbol", width="small"),
            "Name": st.column_config.TextColumn("Name", width="medium"),
            "Status": st.column_config.TextColumn("Status", width="small"),
            "Industry": st.column_config.TextColumn("Industry", width="medium"),
            "Price": st.column_config.TextColumn("Price", width="small"),
            "Change %": st.column_config.TextColumn("Change %", width="small"),
            "Momentum Score": st.column_config.TextColumn("Mom Score", width="small"),
            "RS-Ratio": st.column_config.TextColumn("RS-Ratio", width="small"),
            "RS-Momentum": st.column_config.TextColumn("RS-Mom", width="small"),
            "RRG Power": st.column_config.TextColumn("RRG Power", width="small"),
        },
        hide_index=True
    )
    
    # Alternative: HTML table with colored status badges
    st.markdown("---")
    st.markdown("**Detailed View with Status Colors:**")
    
    def make_enhanced_table_html(df):
        headers = ["#", "Name", "Status", "Industry", "Price", "Change %", "Mom Score", "RS-Ratio", "RS-Mom", "RRG Power"]
        th = "<tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr>"
        tr = []
        
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            status = row["Status"]
            status_class = f"status-{status.lower()}"
            
            # Format change with color
            chg_val = filtered_df.iloc[i-1]["Change %"] if i <= len(filtered_df) else np.nan
            if pd.isna(chg_val):
                chg_html = "-"
            else:
                chg_color = "#16a34a" if chg_val >= 0 else "#dc2626"
                chg_sign = "+" if chg_val >= 0 else ""
                chg_html = f'<span style="color:{chg_color}; font-weight:600">{chg_sign}{chg_val:.2f}%</span>'
            
            # Get original symbol for TV link
            orig_sym = filtered_df.iloc[i-1]["_raw_symbol"] if i <= len(filtered_df) else ""
            tv_link = f'https://www.tradingview.com/chart/?symbol={quote("NSE:"+display_symbol(orig_sym).replace("-","_"), safe="")}'
            
            tr.append(
                f'<tr class="rrg-row">'
                f'<td>{i}</td>'
                f'<td class="rrg-name"><a href="{tv_link}" target="_blank">{row["Name"]}</a></td>'
                f'<td><span class="status-badge {status_class}">{status}</span></td>'
                f'<td>{row["Industry"]}</td>'
                f'<td>{row["Price"]}</td>'
                f'<td>{chg_html}</td>'
                f'<td>{row["Momentum Score"]}</td>'
                f'<td>{row["RS-Ratio"]}</td>'
                f'<td>{row["RS-Momentum"]}</td>'
                f'<td>{row["RRG Power"]}</td>'
                f'</tr>'
            )
        
        return f'<div class="rrg-wrap"><table class="rrg-table">{th}{"".join(tr)}</table></div>'
    
    st.markdown(make_enhanced_table_html(display_df), unsafe_allow_html=True)

# -------------------- Downloads ----------------------
def export_ranks_csv(perf_sorted):
    out=[]
    for t,_m in perf_sorted:
        rr=float(rs_ratio_map[t].iloc[end_idx]); mm=float(rs_mom_map[t].iloc[end_idx])
        info = stock_info.get(t, {})
        out.append((t, META.get(t,{}).get("name",t), META.get(t,{}).get("industry","-"),
                    _m, rr, mm, get_status(rr, mm), 
                    info.get("momentum_score", 0), info.get("rrg_power", 0)))
    df=pd.DataFrame(out, columns=["symbol","name","industry","rank_metric","rs_ratio","rs_momentum","status","momentum_score","rrg_power"])
    buf=io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode()

def export_table_csv(table_data):
    df=pd.DataFrame([{
        "symbol": r["Symbol"],
        "name": r["Name"], 
        "industry": r["Industry"], 
        "status": r["Status"],
        "price": r["Price"], 
        "pct_change_tail": r["Change %"],
        "momentum_score": r["Momentum Score"],
        "rs_ratio": r["RS-Ratio"],
        "rs_momentum": r["RS-Momentum"],
        "rrg_power": r["RRG Power"]
    } for r in table_data])
    buf=io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode()

# Recalculate perf for all visible tickers
perf=[]
for t in tickers:
    if t not in st.session_state.visible_set: continue
    rr_last=rs_ratio_map[t].iloc[end_idx]; mm_last=rs_mom_map[t].iloc[end_idx]
    if rank_mode=="RRG Power (dist)":
        metric=float(np.hypot(rr_last-100.0, mm_last-100.0))
    elif rank_mode=="RS-Ratio":
        metric=float(rr_last)
    elif rank_mode=="RS-Momentum":
        metric=float(mm_last)
    elif rank_mode=="Price %Δ (tail)":
        px=tickers_data[t].reindex(idx).dropna()
        metric=float((px.iloc[end_idx]/px.iloc[start_idx]-1)*100.0) if len(px.iloc[start_idx:end_idx+1])>=2 else float("-inf")
    else:
        series=rs_mom_map[t].iloc[start_idx:end_idx+1].dropna()
        metric=float(np.linalg.lstsq(np.vstack([np.arange(len(series)), np.ones(len(series))]).T, series.values, rcond=None)[0][0]) if len(series)>=2 else float("-inf")
    perf.append((t, metric))
perf.sort(key=lambda x:x[1], reverse=True)

dl1, dl2 = st.columns(2)
with dl1:
    st.download_button("📥 Download Ranks CSV", data=export_ranks_csv(perf),
                       file_name=f"ranks_{date_str}.csv", mime="text/csv", use_container_width=True)
with dl2:
    st.download_button("📥 Download Table CSV", data=export_table_csv(table_data),
                       file_name=f"table_{date_str}.csv", mime="text/csv", use_container_width=True)

st.caption("Names open TradingView. Use Play/Pause to watch rotation; Speed adjusts frame interval, and Loop wraps frames.")

# -------------------- Legend/Help Section ----------------------
with st.expander("ℹ️ Understanding RRG Metrics"):
    st.markdown("""
    **Quadrant Interpretation:**
    - 🟢 **Leading** (Top-Right): Strong relative strength and positive momentum - best performers
    - 🔵 **Improving** (Top-Left): Weak relative strength but gaining momentum - potential turnaround candidates
    - 🟡 **Weakening** (Bottom-Right): Strong relative strength but losing momentum - watch for rotation
    - 🔴 **Lagging** (Bottom-Left): Weak relative strength and negative momentum - underperformers
    
    **Metrics Explained:**
    - **RS-Ratio**: Relative strength compared to benchmark (100 = neutral)
    - **RS-Momentum**: Rate of change in relative strength (100 = neutral)
    - **Momentum Score**: Combined score factoring in quadrant position and distance from center
    - **RRG Power**: Distance from center (100, 100) - indicates strength of trend
    
    **Arrow Direction**: Shows where the stock is heading based on recent movement
    """)

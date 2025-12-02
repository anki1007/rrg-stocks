import os, json, time, pathlib, logging, functools, calendar, webbrowser, io
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

# ================== Defaults & Flags ==================
DEFAULT_TF = "Daily"          # Strength vs (TF) default
DEFAULT_PERIOD = "1Y"         # Period default

# ================== GitHub CSV Folder =================
GITHUB_USER = "anki1007"
GITHUB_REPO = "rrg-stocks"
GITHUB_BRANCH = "main"
GITHUB_TICKER_DIR = "ticker"

# (We’ll fetch CSVs via raw.githubusercontent.com)
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_TICKER_DIR}/"

# ================== Appearance ========================
mpl.rcParams['figure.dpi'] = 110
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.edgecolor'] = '#222'
mpl.rcParams['axes.labelcolor'] = '#111'
mpl.rcParams['xtick.color'] = '#333'
mpl.rcParams['ytick.color'] = '#333'
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.sans-serif'] = ['Inter', 'Segoe UI', 'DejaVu Sans', 'Arial']

# ================== Logging ===========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ================== Helpers: GitHub directory listing ==
@st.cache_data(ttl=600)
def list_csv_files_from_github(user: str, repo: str, branch: str, folder: str) -> List[str]:
    url = f"https://api.github.com/repos/{user}/{repo}/contents/{folder}?ref={branch}"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    items = r.json()
    files = [it["name"] for it in items if it.get("type") == "file" and it["name"].lower().endswith(".csv")]
    files.sort()
    return files

# ---- Friendly display names for the Indices dropdown ----
_FRIENDLY_OVERRIDES = {
    "nifty50.csv": "Nifty 50",
    "nifty200.csv": "Nifty 200",
    "nifty500.csv": "Nifty 500",
    "niftymidcap150.csv": "Nifty Midcap 150",
    "niftysmallcap250.csv": "Nifty Smallcap 250",
    "niftymidsmallcap400.csv": "Nifty MidSmallcap 400",
    "niftytotalmarket.csv": "Nifty Total Market",
}
def friendly_name_from_file(basename: str) -> str:
    b = basename.lower()
    if b in _FRIENDLY_OVERRIDES:
        return _FRIENDLY_OVERRIDES[b]
    core = os.path.splitext(basename)[0].replace("_", " ").replace("-", " ")
    pretty = ""
    for ch in core:
        if ch.isdigit() and (not pretty or (pretty[-1] != " " and not pretty[-1].isdigit())):
            pretty += " " + ch
        else:
            pretty += ch
    return pretty.title()

def build_name_maps_from_github() -> Tuple[Dict[str,str], List[str]]:
    files = list_csv_files_from_github(GITHUB_USER, GITHUB_REPO, GITHUB_BRANCH, GITHUB_TICKER_DIR)
    if not files: return {}, []
    name_map = {friendly_name_from_file(f): f for f in files}
    display_list = sorted(name_map.keys())
    return name_map, display_list

# ================== Universe from CSV =================
def _normalize_cols(cols: List[str]) -> Dict[str, str]:
    norm = {}
    for c in cols:
        key = c.strip().lower().replace(" ", "").replace("_", "")
        norm[c] = key
    return norm

@st.cache_data(ttl=600)
def load_universe_from_github_csv(basename: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    url = RAW_BASE + basename
    df = pd.read_csv(url)
    if df.empty:
        raise ValueError(f"CSV is empty: {basename}")

    mapping = _normalize_cols(list(df.columns))
    sym_col = next((c for c,k in mapping.items() if k in ("symbol","ticker","symbols")), None)
    if sym_col is None:
        raise ValueError("CSV must contain a 'Symbol' column.")
    name_col = next((c for c,k in mapping.items() if k in ("companyname","name","company","companyfullname")), None)
    if name_col is None:
        name_col = sym_col
    ind_col = next((c for c,k in mapping.items() if k in ("industry","sector","industries")), None)
    if ind_col is None:
        ind_col = "Industry"; df[ind_col] = "-"

    sel = df[[sym_col, name_col, ind_col]].copy()
    sel.columns = ["Symbol", "Company Name", "Industry"]
    sel["Symbol"] = sel["Symbol"].astype(str).str.strip()
    sel["Company Name"] = sel["Company Name"].astype(str).str.strip()
    sel["Industry"] = sel["Industry"].astype(str).str.strip()
    sel = sel[sel["Symbol"].notna() & (sel["Symbol"] != "")]
    sel = sel.drop_duplicates(subset=["Symbol"])

    universe = sel["Symbol"].tolist()
    meta = {row["Symbol"]: {"name": row["Company Name"] or row["Symbol"],
                            "industry": row["Industry"] or "-"}
            for _, row in sel.iterrows()}
    return universe, meta

# ================== Config ============================
PERIOD_MAP = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y":"3y", "5Y":"5y", "10Y":"10y"}
TF_LABELS = ["Daily", "Weekly", "Monthly"]
TF_TO_INTERVAL = {"Daily":"1d", "Weekly":"1wk", "Monthly":"1mo"}
TF_DEFAULT_PERIOD = {"1d":"1y", "1wk":"3y", "1mo":"10y"}

WINDOW = 14
DEFAULT_TAIL = 8

BENCH_CHOICES = {"Nifty 500":"^CRSLDX", "Nifty 200":"^CNX200", "Nifty 50":"^NSEI"}
DEFAULT_BENCH_LABEL = "Nifty 500"

# ================== Utilities =========================
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

def safe_long_name(symbol: str, META: dict) -> str:
    return META.get(symbol, {}).get("name") or symbol

def display_symbol(sym: str) -> str:
    return sym[:-3] if sym.upper().endswith(".NS") else sym

def format_bar_date(ts: pd.Timestamp, interval: str) -> str:
    ts = pd.Timestamp(ts)
    if interval == "1wk":
        return ts.to_period("W-FRI").end_time.date().isoformat()
    if interval == "1mo":
        return ts.to_period("M").end_time.date().isoformat()
    return ts.date().isoformat()

def jdk_components(price: pd.Series, bench: pd.Series, win: int = 14):
    df = pd.concat([price.rename("p"), bench.rename("b")], axis=1).dropna()
    if df.empty: return pd.Series(dtype=float), pd.Series(dtype=float)
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

def has_min_coverage(rr: pd.Series, mm: pd.Series, *, min_points: int = 20, lookback_ok: int = 30) -> bool:
    if rr is None or mm is None: return False
    ok = (~rr.isna()) & (~mm.isna())
    if ok.sum() < min_points: return False
    tail = ok.iloc[-lookback_ok:] if len(ok) >= lookback_ok else ok
    return bool(tail.any())

def get_status(x: float, y: float) -> str:
    if x <= 100 and y <= 100: return "Lagging"
    if x >= 100 and y >= 100: return "Leading"
    if x <= 100 and y >= 100: return "Improving"
    if x >= 100 and y <= 100: return "Weakening"
    return "Unknown"

def status_bg_color(x: float, y: float) -> str:
    m = get_status(x,y)
    return {"Lagging":"#e06a6a","Leading":"#3fa46a","Improving":"#5d86d1","Weakening":"#e2d06b"}.get(m,"#aaaaaa")

# ================== Closed-bar helpers =================
IST_TZ = "Asia/Kolkata"
BAR_CUTOFF_HOUR = 18
NET_TIME_MAX_AGE = 300
_NET_NOW_CACHE = {"ts": None, "mono": 0.0}

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
    for url in ("https://www.google.com/generate_204",
                "https://www.cloudflare.com",
                "https://www.nseindia.com",
                "https://www.bseindia.com"):
        try:
            req = _urlreq.Request(url, method="HEAD")
            with _urlreq.urlopen(req, timeout=timeout) as resp:
                date_hdr = resp.headers.get("Date")
                if date_hdr:
                    dt = _eutils.parsedate_to_datetime(date_hdr)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=_dt.timezone.utc)
                    return pd.Timestamp(dt).tz_convert("UTC")
        except Exception:
            continue
    raise RuntimeError("Network time unavailable")

@st.cache_data(ttl=NET_TIME_MAX_AGE)
def _now_ist_cached() -> pd.Timestamp:
    try:
        utc_now = _utc_now_from_network()
        return utc_now.tz_convert(IST_TZ)
    except Exception:
        return pd.Timestamp.now(tz=IST_TZ)

def _to_ist(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(IST_TZ)

def _after_cutoff_ist(now=None) -> bool:
    if now is None:
        now = _now_ist_cached()
    return (now.hour, now.minute) >= (BAR_CUTOFF_HOUR, 0)

def _is_bar_complete_for_timestamp(last_ts: pd.Timestamp, interval: str, now=None) -> bool:
    if now is None:
        now = _now_ist_cached()
    last_ist = _to_ist(last_ts)
    now_ist  = _to_ist(now)
    last_date = last_ist.date()
    today     = now_ist.date()
    wd_now    = now_ist.weekday()

    if interval == "1d":
        if last_date < today: return True
        if last_date == today: return _after_cutoff_ist(now_ist)
        return False

    if interval == "1wk":
        days_to_fri = (4 - wd_now) % 7
        this_friday = (now_ist + _dt.timedelta(days=days_to_fri)).date()
        last_friday = this_friday if wd_now >= 4 else (this_friday - _dt.timedelta(days=7))
        if last_date < last_friday: return True
        if last_date == last_friday:
            if wd_now < 4: return True
            if wd_now == 4: return _after_cutoff_ist(now_ist)
            return True
        return False

    if interval == "1mo":
        y, m = last_ist.year, last_ist.month
        month_end = _dt.date(y, m, calendar.monthrange(y, m)[1])
        if last_date < month_end: return True
        if last_date == month_end:
            if today > month_end: return True
            return _after_cutoff_ist(now_ist)
        return False

    return False

# ================== Cache + Retry ======================
CACHE_DIR = pathlib.Path("cache"); CACHE_DIR.mkdir(exist_ok=True)
def _cache_path(symbol: str, period: str, interval: str) -> pathlib.Path:
    safe = symbol.replace("^","").replace(".","_")
    return CACHE_DIR / f"{safe}_{period}_{interval}.parquet"
def _save_cache(symbol: str, s: pd.Series, period: str, interval: str):
    try: s.to_frame("Close").to_parquet(_cache_path(symbol, period, interval))
    except Exception as e: logging.warning(f"Cache save failed for {symbol}: {e}")
def _load_cache(symbol: str, period: str, interval: str) -> pd.Series:
    p=_cache_path(symbol, period, interval)
    if p.exists():
        try: return pd.read_parquet(p)["Close"].dropna()
        except Exception as e: logging.warning(f"Cache read failed {symbol}: {e}")
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
                    logging.warning(f"{fn.__name__} failed {i+1}/{n}: {e} → retry in {d:.1f}s"); time.sleep(d); d*=backoff
        return wrap
    return deco

# ================== Download ===========================
@st.cache_data(show_spinner=False)
def download_block_with_benchmark(universe: List[str], benchmark: str, period: str, interval: str):
    @retry()
    def _dl():
        raw = yf.download(list(universe)+[benchmark], period=period, interval=interval,
                          group_by="ticker", auto_adjust=True, progress=False, threads=True)
        return raw
    raw = _dl()

    def _pick(sym):
        return pick_close(raw, sym).dropna()

    bench = _pick(benchmark)
    if bench is None or bench.empty:
        return bench, {}

    keep_last = _is_bar_complete_for_timestamp(bench.index[-1], interval, now=_now_ist_cached())
    drop_last = not keep_last

    def _maybe_trim(s: pd.Series):
        if drop_last and len(s) >= 1:
            return s.iloc[:-1]
        return s

    bench = _maybe_trim(bench)
    data: Dict[str,pd.Series] = {}
    for t in universe:
        s = _pick(t)
        if not s.empty:
            data[t] = _maybe_trim(s)
        else:
            c = _load_cache(t, period, interval)
            if not c.empty:
                data[t] = _maybe_trim(c)

    # persist cache
    if not bench.empty: _save_cache(benchmark, bench, period, interval)
    for t, s in data.items(): _save_cache(t, s, period, interval)
    return bench, data

# ================== Colors =============================
def symbol_color_map(symbols: List[str]) -> Dict[str, str]:
    tab = plt.get_cmap('tab20').colors
    return {s: to_hex(tab[i % len(tab)], keep_alpha=False) for i,s in enumerate(symbols)}

# ================== Streamlit UI =======================
st.set_page_config(page_title="Relative Rotation Graph (RRG)", layout="wide")

st.title("Relative Rotation Graph (RRG)")

NAME_MAP, DISPLAY_LIST = build_name_maps_from_github()
if not DISPLAY_LIST:
    st.error("Could not list CSVs from GitHub /ticker. Check repo access.")
    st.stop()

# Sidebar controls (functional, minimal)
with st.sidebar:
    st.header("Controls")
    csv_disp = st.selectbox("Indices", DISPLAY_LIST, index=(DISPLAY_LIST.index("Nifty 200") if "Nifty 200" in DISPLAY_LIST else 0))
    csv_basename = NAME_MAP[csv_disp]

    bench_label = st.selectbox("Benchmark", list(BENCH_CHOICES.keys()), index=list(BENCH_CHOICES.keys()).index(DEFAULT_BENCH_LABEL))
    interval_label = st.selectbox("Strength vs (TF)", TF_LABELS, index=TF_LABELS.index(DEFAULT_TF))
    interval = TF_TO_INTERVAL[interval_label]

    # Period defaults per TF
    default_period_for_tf = {"1d":"1Y", "1wk":"3Y", "1mo":"10Y"}[interval]
    period_label = st.selectbox("Period", list(PERIOD_MAP.keys()),
                                index=list(PERIOD_MAP.keys()).index(default_period_for_tf))
    period = PERIOD_MAP[period_label]

    rank_modes = ["RRG Power (dist)", "RS-Ratio", "RS-Momentum", "Price %Δ (tail)", "Momentum Slope (tail)"]
    rank_mode = st.selectbox("Rank by", rank_modes, index=0)

    tail_len = st.slider("Trail Length", 1, 20, DEFAULT_TAIL, 1)

# Load universe
UNIVERSE, META = load_universe_from_github_csv(csv_basename)

# Download data (with closed-bar enforcement)
bench_symbol = BENCH_CHOICES[bench_label]
benchmark_data, tickers_data = download_block_with_benchmark(UNIVERSE, bench_symbol, period, interval)
if benchmark_data is None or benchmark_data.empty:
    st.error("Benchmark returned no data.")
    st.stop()

# Build RS-Ratio & RS-Momentum
bench_idx = benchmark_data.index
rs_ratio_map, rs_mom_map = {}, {}
kept = []
for t, s in tickers_data.items():
    if t == bench_symbol: continue
    rr, mm = jdk_components(s, benchmark_data, WINDOW)
    if len(rr)==0 or len(mm)==0: continue
    rr = rr.reindex(bench_idx)
    mm = mm.reindex(bench_idx)
    if has_min_coverage(rr, mm, min_points=max(WINDOW+5,20), lookback_ok=30):
        rs_ratio_map[t]=rr; rs_mom_map[t]=mm; kept.append(t)

if not kept:
    st.warning("After alignment, no symbols have enough coverage. Try a longer period.")
    st.stop()

tickers = kept
SYMBOL_COLORS = symbol_color_map(tickers)
idx = bench_idx
idx_len = len(idx)

# Date slider
end_idx = st.slider("Date position", min_value=DEFAULT_TAIL, max_value=idx_len-1, value=idx_len-1, step=1,
                    format=" ", help="Move to change the RRG date (closed bars only).")
start_idx = max(end_idx - tail_len, 0)
date_str = format_bar_date(idx[end_idx], interval)
st.caption(f"Date: **{date_str}**  |  TF: **{interval_label}**  |  Period: **{period_label}**  |  Universe: **{csv_disp}**  |  Benchmark: **{bench_label}**")

# Compute visibility set (default: all)
if "visible_set" not in st.session_state:
    st.session_state.visible_set = set(tickers)
with st.expander("Show/Hide Symbols", expanded=False):
    cols = st.columns(4)
    vis_changes = []
    for i, t in enumerate(tickers):
        with cols[i % 4]:
            chk = st.checkbox(display_symbol(t), value=(t in st.session_state.visible_set), key=f"vis_{t}")
            if chk and t not in st.session_state.visible_set:
                vis_changes.append(("add", t))
            elif (not chk) and t in st.session_state.visible_set:
                vis_changes.append(("remove", t))
    for op, t in vis_changes:
        if op == "add": st.session_state.visible_set.add(t)
        else: st.session_state.visible_set.discard(t)

visible = [t for t in tickers if t in st.session_state.visible_set]

# --------- Plot RRG ----------
def plot_rrg():
    fig, ax = plt.subplots(1, 1, figsize=(8.8, 6.2))
    ax.set_title("Relative Rotation Graph (RRG)", fontsize=13, pad=10)
    ax.set_xlabel("JdK RS-Ratio"); ax.set_ylabel("JdK RS-Momentum")
    ax.axhline(y=100, color="#777", linestyle=":", linewidth=1.0)
    ax.axvline(x=100, color="#777", linestyle=":", linewidth=1.0)
    ax.fill_between([94,100],[94,94],[100,100], color=(1.0,0.0,0.0,0.25))
    ax.fill_between([100,106],[94,94],[100,100], color=(1.0,1.0,0.0,0.25))
    ax.fill_between([100,106],[100,100],[106,106], color=(0.0,1.0,0.0,0.25))
    ax.fill_between([94,100],[100,100],[106,106], color=(0.0,0.0,1.0,0.25))
    ax.text(95,105,"Improving", fontsize=11, color="#111", weight="bold")
    ax.text(104,105,"Leading",   fontsize=11, color="#111", weight="bold", ha="right")
    ax.text(104,95,"Weakening",  fontsize=11, color="#111", weight="bold", ha="right")
    ax.text(95,95,"Lagging",     fontsize=11, color="#111", weight="bold")
    ax.set_xlim(94,106); ax.set_ylim(94,106)

    for t in visible:
        rr = rs_ratio_map[t].iloc[start_idx+1 : end_idx+1].dropna()
        mm = rs_mom_map[t].iloc[start_idx+1 : end_idx+1].dropna()
        rr, mm = rr.align(mm, join="inner")
        if len(rr)==0 or len(mm)==0: 
            continue
        ax.plot(rr.values, mm.values, linewidth=1.1, alpha=0.6, color=SYMBOL_COLORS[t])
        sizes = [18] * (len(rr) - 1) + [70]
        ax.scatter(rr.values, mm.values, s=sizes, linewidths=0.6,
                   facecolor=SYMBOL_COLORS[t], edgecolor="#333333")
        rr_last = rr.values[-1]; mm_last = mm.values[-1]
        s_txt = get_status(rr_last, mm_last)
        ax.annotate(f"{t}  [{s_txt}]", (rr_last, mm_last), fontsize=9, color=SYMBOL_COLORS[t])
    return fig

fig = plot_rrg()
st.pyplot(fig, use_container_width=True)

# --------- Ranking panel + table ----------
def compute_rank_metric(t: str) -> float:
    rr_last = rs_ratio_map[t].iloc[end_idx]; mm_last = rs_mom_map[t].iloc[end_idx]
    if np.isnan(rr_last) or np.isnan(mm_last): return float("-inf")
    if rank_mode == "RRG Power (dist)": return float(np.hypot(rr_last-100.0, mm_last-100.0))
    if rank_mode == "RS-Ratio":         return float(rr_last)
    if rank_mode == "RS-Momentum":      return float(mm_last)
    if rank_mode == "Price %Δ (tail)":
        px = tickers_data[t].reindex(idx).dropna()
        if len(px.iloc[start_idx:end_idx+1]) >= 2:
            return float((px.iloc[end_idx] / px.iloc[start_idx] - 1) * 100.0)
        return float("-inf")
    if rank_mode == "Momentum Slope (tail)":
        series = rs_mom_map[t].iloc[start_idx:end_idx+1].dropna()
        if len(series) >= 2:
            x = np.arange(len(series)); A = np.vstack([x, np.ones(len(x))]).T
            slope = np.linalg.lstsq(A, series.values, rcond=None)[0][0]
            return float(slope)
        return float("-inf")
    return float("-inf")

performances=[]
for t in visible:
    try:
        metric=compute_rank_metric(t)
        performances.append((t, metric))
    except Exception:
        pass
performances.sort(key=lambda x:x[1], reverse=True)

left_col, right_col = st.columns([1,2])

with left_col:
    st.subheader("Ranking")
    rank_lines = []
    for rank,(symbol,metric) in enumerate(performances[:22], start=1):
        rr_last=float(rs_ratio_map[symbol].iloc[end_idx]); mm_last=float(rs_mom_map[symbol].iloc[end_idx])
        if np.isnan(rr_last) or np.isnan(mm_last): continue
        stat=get_status(rr_last, mm_last)
        rank_lines.append(f"{rank}. {display_symbol(symbol)} [{stat}]")
    st.text("\n".join(rank_lines) if rank_lines else "—")

with right_col:
    st.subheader("Table")
    rows=[]
    for t in visible:
        rr=float(rs_ratio_map[t].iloc[end_idx]); mm=float(rs_mom_map[t].iloc[end_idx])
        if np.isnan(rr) or np.isnan(mm): continue
        px=tickers_data[t].reindex(idx).dropna()
        price=float(px.iloc[end_idx]) if end_idx < len(px) else np.nan
        chg=((px.iloc[end_idx]/px.iloc[start_idx]-1)*100.0) if (end_idx < len(px) and start_idx < len(px)) else np.nan
        rows.append({
            "#": (dict(performances).get(t, np.nan)),
            "Name": safe_long_name(t, META),
            "Status": get_status(rr, mm),
            "Industry": META.get(t,{}).get("industry","-"),
            "Price": price,
            "Change %": chg,
            "Symbol": t,
            "TradingView": f"https://www.tradingview.com/chart/?symbol={quote('NSE:' + display_symbol(t).replace('-', '_'), safe='')}",
        })
    df = pd.DataFrame(rows)
    # Show a compact table
    st.dataframe(df[["#", "Name", "Status", "Industry", "Price", "Change %", "Symbol", "TradingView"]],
                 hide_index=True, use_container_width=True)

# --------- Exports ----------
def export_ranks_csv() -> bytes:
    rows=[]
    for t,_m in performances:
        rr=float(rs_ratio_map[t].iloc[end_idx]); mm=float(rs_mom_map[t].iloc[end_idx])
        rows.append((t, META.get(t,{}).get("name",t), META.get(t,{}).get("industry","-"),
                     dict(performances).get(t, np.nan), rr, mm, get_status(rr, mm)))
    df=pd.DataFrame(rows, columns=["symbol","name","industry","rank_metric","rs_ratio","rs_momentum","status"])
    buff=io.StringIO(); df.to_csv(buff, index=False); return buff.getvalue().encode()

def export_table_csv() -> bytes:
    rows=[]
    for t in visible:
        rr=float(rs_ratio_map[t].iloc[end_idx]); mm=float(rs_mom_map[t].iloc[end_idx])
        px=tickers_data[t].reindex(idx).dropna()
        price=float(px.iloc[end_idx]) if end_idx < len(px) else np.nan
        chg=((px.iloc[end_idx]/px.iloc[start_idx]-1)*100.0) if (end_idx < len(px) and start_idx < len(px)) else np.nan
        rows.append((t, META.get(t,{}).get("name",t), META.get(t,{}).get("industry","-"),
                     get_status(rr, mm), rr, mm, price, chg))
    df=pd.DataFrame(rows, columns=["symbol","name","industry","status","rs_ratio","rs_momentum","price","pct_change_tail"])
    buff=io.StringIO(); df.to_csv(buff, index=False); return buff.getvalue().encode()

col_a, col_b = st.columns(2)
with col_a:
    st.download_button(
        "Download Ranks CSV",
        data=export_ranks_csv(),
        file_name=f"ranks_{date_str}.csv",
        mime="text/csv",
        use_container_width=True,
    )
with col_b:
    st.download_button(
        "Download Table CSV",
        data=export_table_csv(),
        file_name=f"table_{date_str}.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.caption("Tip: Use the expander above to hide symbols you don’t want on the plot/ranking.")

# rrgstocks.py — Streamlit Relative Rotation Graph (RRG) app
# --------------------------------------------------------------------------------------
# Run locally:  streamlit run rrgstocks.py
# Streamlit Cloud: commit this file + a /ticker folder with your CSV universes.
# --------------------------------------------------------------------------------------

import os, json, time, pathlib, logging, functools, calendar
import datetime as _dt
import email.utils as _eutils
import urllib.request as _urlreq
from urllib.parse import quote
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib as mpl
mpl.use("Agg")  # headless backend for Streamlit
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

import streamlit as st

# ================== Defaults & Flags ==================
DEFAULT_TF = "Daily"        # Strength vs (TF) default
DEFAULT_PERIOD = "1Y"       # Period default

# ================== Where CSVs live ===================
# On Streamlit Cloud, keep universes inside the repo at ./ticker/<files>.csv
CSV_DIR = os.environ.get("RRG_CSV_DIR", "./ticker")
DEFAULT_CSV_BASENAME = "nifty200.csv"         # default selection
DEFAULT_CSV_PATH = os.path.join(CSV_DIR, DEFAULT_CSV_BASENAME)

# ================== Appearance ========================
mpl.rcParams.update({
    'figure.dpi': 110,
    'axes.grid': False,
    'axes.edgecolor': '#222',
    'axes.labelcolor': '#111',
    'xtick.color': '#333',
    'ytick.color': '#333',
    'font.size': 10,
    'font.sans-serif': ['Inter', 'Segoe UI', 'DejaVu Sans', 'Arial'],
})

# ================== Logging ===========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

# ================== Helpers for CSV universes =========
def _normalize_cols(cols: List[str]) -> Dict[str, str]:
    norm = {}
    for c in cols:
        key = c.strip().lower().replace(" ", "").replace("_", "")
        norm[c] = key
    return norm

def load_universe_from_csv(csv_path: str) -> Tuple[List[str], Dict[str, Dict[str, str]]]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"CSV is empty: {csv_path}")

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

def list_csv_files(folder: str) -> List[str]:
    try:
        files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
        files.sort()
        return files
    except Exception:
        return []

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

def build_name_maps(folder: str):
    files = list_csv_files(folder)
    if not files:
        return {}, []
    name_map = {friendly_name_from_file(f): f for f in files}
    display_list = sorted(name_map.keys())
    return name_map, display_list

# ================== Time/Bar completeness (IST) =======
IST_TZ = "Asia/Kolkata"
BAR_CUTOFF_HOUR = 18
NET_TIME_MAX_AGE = 300
_NET_NOW_CACHE = {"ts": None, "mono": 0.0}

def _utc_now_from_network(timeout=2.5) -> pd.Timestamp:
    # Use HEAD Date header (no extra deps)
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
    # Fallback to system clock
    return pd.Timestamp.utcnow().tz_localize("UTC")

def _now_ist() -> pd.Timestamp:
    try:
        m = time.monotonic()
        if (_NET_NOW_CACHE["ts"] is None) or (m - _NET_NOW_CACHE["mono"] > NET_TIME_MAX_AGE):
            utc_now = _utc_now_from_network()
            _NET_NOW_CACHE["ts"] = utc_now.tz_convert(IST_TZ)
            _NET_NOW_CACHE["mono"] = m
        return _NET_NOW_CACHE["ts"]
    except Exception:
        return pd.Timestamp.now(tz=IST_TZ)

def _to_ist(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert(IST_TZ)

def _after_cutoff_ist(now=None) -> bool:
    if now is None:
        now = _now_ist()
    return (now.hour, now.minute) >= (BAR_CUTOFF_HOUR, 0)

def _is_bar_complete_for_timestamp(last_ts: pd.Timestamp, interval: str, now=None) -> bool:
    if now is None:
        now = _now_ist()
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

# ================== Config mappings ====================
PERIOD_MAP = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y":"3y", "5Y":"5y", "10Y":"10y"}
TF_LABELS = ["Daily", "Weekly", "Monthly"]
TF_TO_INTERVAL = {"Daily":"1d", "Weekly":"1wk", "Monthly":"1mo"}
TF_DEFAULT_PERIOD = {"1d":"1y", "1wk":"3y", "1mo":"10y"}

BENCH_CHOICES = {"Nifty 500":"^CRSLDX", "Nifty 200":"^CNX200", "Nifty 50":"^NSEI"}
DEFAULT_BENCH_LABEL = "Nifty 500"

# ================== Math (JdK proxies) =================
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

# ================== Caching ============================
@st.cache_data(show_spinner=False)
def download_block_with_benchmark(tickers: List[str], benchmark: str, period: str, interval: str):
    raw = yf.download(list(tickers)+[benchmark], period=period, interval=interval,
                      group_by="ticker", auto_adjust=True, progress=False, threads=True)
    def _pick(sym):
        if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
            for lvl in ("Close","Adj Close"):
                if (sym, lvl) in raw.columns: return raw[(sym, lvl)].dropna()
        else:
            for col in ("Close","Adj Close"):
                if col in raw.columns: return raw[col].dropna()
        return pd.Series(dtype=float)

    bench = _pick(benchmark)
    if bench is None or bench.empty:
        return bench, {}

    keep_last = _is_bar_complete_for_timestamp(bench.index[-1], interval, now=_now_ist())
    drop_last = not keep_last
    def _trim(s: pd.Series):
        if drop_last and len(s) >= 1: return s.iloc[:-1]
        return s

    bench = _trim(bench)
    data: Dict[str,pd.Series] = {}
    for t in tickers:
        s = _pick(t)
        if not s.empty:
            data[t] = _trim(s)
    return bench, data

def symbol_color_map(symbols: List[str]) -> Dict[str, str]:
    tab = plt.get_cmap('tab20').colors
    return {s: to_hex(tab[i % len(tab)], keep_alpha=False) for i,s in enumerate(symbols)}

# ================== UI (Streamlit) =====================
st.set_page_config(page_title="Relative Rotation Graph (RRG)", layout="wide")
st.title("Relative Rotation Graph (RRG)")

# Sidebar — universe selection
st.sidebar.header("Universe / Benchmark")
NAME_MAP, DISPLAY_LIST = build_name_maps(CSV_DIR)
uploaded_csv = None
if not DISPLAY_LIST:
    st.sidebar.warning("No CSVs found in ./ticker. Upload one to proceed.")
    uploaded_csv = st.sidebar.file_uploader("Upload universe CSV", type=["csv"])

if uploaded_csv is not None:
    # Treat the uploaded file as a one-off universe
    df_up = pd.read_csv(uploaded_csv)
    tmp_dir = pathlib.Path(st.session_state.get("_tmp_dir", "./.tmp"))
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / "uploaded.csv"
    df_up.to_csv(tmp_path, index=False)

    NAME_MAP = { "Uploaded file": str(tmp_path.name) }
    DISPLAY_LIST = ["Uploaded file"]
    CSV_PATH = str(tmp_path)
else:
    if not DISPLAY_LIST:
        st.stop()
    default_disp = friendly_name_from_file(DEFAULT_CSV_BASENAME) if DEFAULT_CSV_BASENAME in NAME_MAP.values() else DISPLAY_LIST[0]
    csv_disp = st.sidebar.selectbox("Indices", DISPLAY_LIST, index=(DISPLAY_LIST.index(default_disp) if default_disp in DISPLAY_LIST else 0))
    CSV_PATH = os.path.join(CSV_DIR, NAME_MAP.get(csv_disp, DEFAULT_CSV_BASENAME))

# Benchmark / TF / Period
bench_label = st.sidebar.selectbox("Benchmark", list(BENCH_CHOICES.keys()), index=list(BENCH_CHOICES.keys()).index(DEFAULT_BENCH_LABEL))
tf_label = st.sidebar.selectbox("Strength vs (TF)", TF_LABELS, index=TF_LABELS.index(DEFAULT_TF))
interval = TF_TO_INTERVAL[tf_label]
# Suggested default period per TF
suggested = TF_DEFAULT_PERIOD.get(interval, PERIOD_MAP[DEFAULT_PERIOD])
period_label = st.sidebar.selectbox("Period", list(PERIOD_MAP.keys()), index=list(PERIOD_MAP.values()).index(suggested))
period = PERIOD_MAP[period_label]

# Tail length
tail_len = st.sidebar.slider("Trail Length", min_value=1, max_value=20, value=8, step=1)

# Rank mode
RANK_MODES = ["RRG Power (dist)", "RS-Ratio", "RS-Momentum", "Price %Δ (tail)", "Momentum Slope (tail)"]
rank_mode = st.sidebar.selectbox("Rank by", RANK_MODES, index=0)

# Load universe
try:
    UNIVERSE, META = load_universe_from_csv(CSV_PATH)
except Exception as e:
    st.error(f"Could not load universe CSV.\n\n{e}")
    st.stop()

# Fetch data (with fallback across benchmarks if needed)
bench_order = [bench_label] + [l for l in BENCH_CHOICES.keys() if l != bench_label]
last_err = None
for lbl in bench_order:
    try:
        bench_symbol = BENCH_CHOICES[lbl]
        bench_series, px_map = download_block_with_benchmark(UNIVERSE, bench_symbol, period, interval)
        if bench_series is None or bench_series.empty:
            raise RuntimeError("Empty benchmark")
        benchmark_used_label = lbl
        benchmark_used = bench_symbol
        tickers_data = px_map
        break
    except Exception as e:
        last_err = e
else:
    st.error(f"No benchmark returned data. Last error: {last_err}")
    st.stop()

# Prepare RRG inputs
rs_ratio_map, rs_mom_map = {}, {}
kept = []
bench_idx = bench_series.index

for t, s in tickers_data.items():
    if t == benchmark_used:
        continue
    rr, mm = jdk_components(s, bench_series, win=14)
    if len(rr)==0 or len(mm)==0:
        continue
    rr = rr.reindex(bench_idx)
    mm = mm.reindex(bench_idx)
    if has_min_coverage(rr, mm, min_points=20, lookback_ok=30):
        rs_ratio_map[t]=rr; rs_mom_map[t]=mm; kept.append(t)

if not kept:
    st.error("After alignment, no symbols have enough coverage. Try a longer period.")
    st.stop()

tickers = kept
SYMBOL_COLORS = symbol_color_map(tickers)
idx = bench_idx
idx_len = len(idx)

# Date slider
def _format_bar_date(ts: pd.Timestamp, interval: str) -> str:
    ts = pd.Timestamp(ts)
    if interval == "1wk":
        return ts.to_period("W-FRI").end_time.date().isoformat()
    if interval == "1mo":
        return ts.to_period("M").end_time.date().isoformat()
    return ts.date().isoformat()

st.subheader(f"{friendly_name_from_file(os.path.basename(CSV_PATH))} — {benchmark_used_label} — {period_label} — {tf_label}")

if idx_len < 2:
    st.warning("Not enough bars to plot.")
    st.stop()

end_pos = st.slider(
    "Date index",
    min_value=min(8, idx_len-1),
    max_value=idx_len-1,
    value=idx_len-1,
    step=1,
    format="%d",
)

start_pos = max(end_pos - tail_len, 0)
current_date_str = _format_bar_date(idx[end_pos], interval)
st.caption(f"Date: **{current_date_str}**")

# Build plot
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
def init_plot_axes():
    ax.clear()
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

init_plot_axes()

visible = st.multiselect(
    "Visible symbols",
    options=[t for t in tickers],
    default=[t for t in tickers][:50],  # keep initial list reasonable
)

for t in tickers:
    if t not in visible:
        continue
    rr = rs_ratio_map[t].iloc[start_pos+1 : end_pos+1].dropna()
    mm = rs_mom_map[t].iloc[start_pos+1 : end_pos+1].dropna()
    rr, mm = rr.align(mm, join="inner")
    if len(rr)==0 or len(mm)==0:
        continue
    col = SYMBOL_COLORS[t]
    ax.plot(rr.values, mm.values, linewidth=1.1, alpha=0.7, color=col)
    ax.scatter(rr.values[:-1], mm.values[:-1], s=18, linewidths=0.6, color=col, edgecolors="#333333")
    ax.scatter(rr.values[-1],  mm.values[-1],  s=70, linewidths=0.8, color=col, edgecolors="#333333")
    ax.annotate(t, (rr.values[-1], mm.values[-1]), fontsize=9, color=col)

st.pyplot(fig, use_container_width=True)

# Ranking pane
def compute_rank_metric(t: str, start_pos: int, end_pos: int) -> float:
    rr_last = rs_ratio_map[t].iloc[end_pos]; mm_last = rs_mom_map[t].iloc[end_pos]
    if np.isnan(rr_last) or np.isnan(mm_last): return float("-inf")
    if rank_mode == "RRG Power (dist)": return float(np.hypot(rr_last-100.0, mm_last-100.0))
    if rank_mode == "RS-Ratio":         return float(rr_last)
    if rank_mode == "RS-Momentum":      return float(mm_last)
    if rank_mode == "Price %Δ (tail)":
        px = tickers_data[t].reindex(idx).dropna()
        if len(px.iloc[start_pos:end_pos+1]) >= 2:
            return float((px.iloc[end_pos] / px.iloc[start_pos] - 1) * 100.0)
        return float("-inf")
    if rank_mode == "Momentum Slope (tail)":
        series = rs_mom_map[t].iloc[start_pos:end_pos+1].dropna()
        if len(series) >= 2:
            x = np.arange(len(series)); A = np.vstack([x, np.ones(len(x))]).T
            slope = np.linalg.lstsq(A, series.values, rcond=None)[0][0]
            return float(slope)
        return float("-inf")
    return float("-inf")

rows=[]
for t in tickers:
    if t not in visible:
        continue
    rr_last = float(rs_ratio_map[t].iloc[end_pos]); mm_last = float(rs_mom_map[t].iloc[end_pos])
    if np.isnan(rr_last) or np.isnan(mm_last):
        continue
    metric = compute_rank_metric(t, start_pos, end_pos)
    px = tickers_data[t].reindex(idx).dropna()
    price = float(px.iloc[end_pos]) if end_pos < len(px) else np.nan
    chg = ((px.iloc[end_pos]/px.iloc[start_pos]-1)*100.0) if (end_pos < len(px) and start_pos < len(px)) else np.nan
    rows.append({
        "symbol": t,
        "name": META.get(t,{}).get("name", t),
        "industry": META.get(t,{}).get("industry","-"),
        "status": get_status(rr_last, mm_last),
        "rs_ratio": rr_last,
        "rs_momentum": mm_last,
        "price": price,
        "pct_change_tail": chg,
        "rank_metric": metric,
    })

df_rank = pd.DataFrame(rows)
if not df_rank.empty:
    df_rank = df_rank.sort_values("rank_metric", ascending=False).reset_index(drop=True)
    st.subheader("Ranking")
    st.dataframe(df_rank, use_container_width=True)

    # Downloads
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download ranks CSV",
            data=df_rank.to_csv(index=False),
            file_name=f"ranks_{current_date_str}.csv",
            mime="text/csv",
        )
    with c2:
        minimal_table = df_rank[["symbol","name","industry","status","rs_ratio","rs_momentum","price","pct_change_tail"]]
        st.download_button(
            "Download table CSV",
            data=minimal_table.to_csv(index=False),
            file_name=f"table_{current_date_str}.csv",
            mime="text/csv",
        )
else:
    st.info("No visible symbols to rank. Use the selector above to choose some.")

# Quick TradingView links (optional)
with st.expander("Open in TradingView"):
    def tv_symbol(symbol: str) -> str:
        s = symbol[:-3] if symbol.upper().endswith(".NS") else symbol
        s = s.replace("-", "_")
        return f"NSE:{s}"
    links = [f"- [{t}]({'https://www.tradingview.com/chart/?symbol=' + quote(tv_symbol(t), safe='')})" for t in visible[:100]]
    if links:
        st.markdown("\n".join(links))
    else:
        st.write("Select some symbols first.")

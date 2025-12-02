import os, time, pathlib, logging, functools, calendar, json
import datetime as _dt
import email.utils as _eutils
import urllib.request as _urlreq
from urllib.parse import quote
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

# ---- Matplotlib headless by default (works in Streamlit & servers) ----
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# ---- Streamlit is always safe to import on cloud; desktop mode skips it ----
import streamlit as st

# ================== Defaults & Flags ==================
DEFAULT_TF = "Daily"       # Strength vs (TF)
DEFAULT_PERIOD = "1Y"      # Period for downloads
WINDOW = 14                # JdK rolling window
DEFAULT_TAIL = 8           # Trail length

# ================== CSV Universes =====================
CSV_DIR = os.environ.get("RRG_CSV_DIR", "./ticker")
DEFAULT_CSV_BASENAME = "nifty200.csv"

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

# ================== Friendly names ====================
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
    if b in _FRIENDLY_OVERRIDES: return _FRIENDLY_OVERRIDES[b]
    core = os.path.splitext(basename)[0].replace("_", " ").replace("-", " ")
    pretty = ""
    for ch in core:
        if ch.isdigit() and (not pretty or (pretty[-1] != " " and not pretty[-1].isdigit())):
            pretty += " " + ch
        else:
            pretty += ch
    return pretty.title()

def list_csv_files(folder: str) -> List[str]:
    try:
        files = [f for f in os.listdir(folder) if f.lower().endswith(".csv")]
        files.sort(); return files
    except Exception:
        return []

def _normalize_cols(cols: List[str]) -> Dict[str,str]:
    out={}
    for c in cols:
        out[c]=c.strip().lower().replace(" ","").replace("_","")
    return out

def load_universe_from_csv(csv_path: str) -> Tuple[List[str], Dict[str, Dict[str,str]]]:
    if not os.path.exists(csv_path): raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if df.empty: raise ValueError(f"CSV is empty: {csv_path}")
    m = _normalize_cols(df.columns.tolist())
    sym_col = next((c for c,k in m.items() if k in ("symbol","ticker","symbols")), None)
    if sym_col is None: raise ValueError("CSV must contain a 'Symbol' column.")
    name_col = next((c for c,k in m.items() if k in ("companyname","name","company","companyfullname")), None) or sym_col
    ind_col = next((c for c,k in m.items() if k in ("industry","sector","industries")), None)
    if ind_col is None:
        ind_col = "Industry"; df[ind_col]="-"
    sel = df[[sym_col, name_col, ind_col]].copy()
    sel.columns = ["Symbol","Company Name","Industry"]
    for c in sel.columns: sel[c]=sel[c].astype(str).str.strip()
    sel = sel[sel["Symbol"].notna() & (sel["Symbol"]!="")].drop_duplicates("Symbol")
    universe = sel["Symbol"].tolist()
    meta = {r["Symbol"]:{ "name": r["Company Name"] or r["Symbol"], "industry": r["Industry"] or "-" }
            for _,r in sel.iterrows()}
    return universe, meta

def build_name_maps(folder: str):
    files = list_csv_files(folder)
    if not files: return {}, []
    name_map = {friendly_name_from_file(f): f for f in files}
    return name_map, sorted(name_map.keys())

# ================== Time & bar completeness (IST) =====
IST_TZ = "Asia/Kolkata"
BAR_CUTOFF_HOUR = 18
NET_TIME_MAX_AGE = 300
_NET_NOW_CACHE = {"ts": None, "mono": 0.0}

def _utc_now_from_network(timeout=2.5) -> pd.Timestamp:
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
                    if dt.tzinfo is None: dt = dt.replace(tzinfo=_dt.timezone.utc)
                    return pd.Timestamp(dt).tz_convert("UTC")
        except Exception:
            continue
    return pd.Timestamp.utcnow().tz_localize("UTC")

def _now_ist() -> pd.Timestamp:
    try:
        m = time.monotonic()
        if (_NET_NOW_CACHE["ts"] is None) or (m - _NET_NOW_CACHE["mono"] > NET_TIME_MAX_AGE):
            _NET_NOW_CACHE["ts"] = _utc_now_from_network().tz_convert(IST_TZ)
            _NET_NOW_CACHE["mono"] = m
        return _NET_NOW_CACHE["ts"]
    except Exception:
        return pd.Timestamp.now(tz=IST_TZ)

def _to_ist(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None: ts = ts.tz_localize("UTC")
    return ts.tz_convert(IST_TZ)

def _after_cutoff_ist(now=None) -> bool:
    if now is None: now=_now_ist()
    return (now.hour, now.minute) >= (BAR_CUTOFF_HOUR, 0)

def _is_bar_complete_for_timestamp(last_ts: pd.Timestamp, interval: str, now=None) -> bool:
    if now is None: now=_now_ist()
    last_ist=_to_ist(last_ts); now_ist=_to_ist(now)
    last_date=last_ist.date(); today=now_ist.date(); wd_now=now_ist.weekday()
    if interval=="1d":
        if last_date<today: return True
        if last_date==today: return _after_cutoff_ist(now_ist)
        return False
    if interval=="1wk":
        days_to_fri=(4-wd_now)%7
        this_friday=(now_ist+_dt.timedelta(days=days_to_fri)).date()
        last_friday=this_friday if wd_now>=4 else (this_friday-_dt.timedelta(days=7))
        if last_date<last_friday: return True
        if last_date==last_friday:
            if wd_now<4: return True
            if wd_now==4: return _after_cutoff_ist(now_ist)
            return True
        return False
    if interval=="1mo":
        y,m = last_ist.year, last_ist.month
        month_end=_dt.date(y,m,calendar.monthrange(y,m)[1])
        if last_date<month_end: return True
        if last_date==month_end:
            if today>month_end: return True
            return _after_cutoff_ist(now_ist)
        return False
    return False

# ================== Config maps =======================
PERIOD_MAP = {"6M":"6mo","1Y":"1y","2Y":"2y","3Y":"3y","5Y":"5y","10Y":"10y"}
TF_LABELS = ["Daily","Weekly","Monthly"]
TF_TO_INTERVAL = {"Daily":"1d","Weekly":"1wk","Monthly":"1mo"}
TF_DEFAULT_PERIOD = {"1d":"1y","1wk":"3y","1mo":"10y"}

BENCH_CHOICES = {"Nifty 500":"^CRSLDX", "Nifty 200":"^CNX200", "Nifty 50":"^NSEI"}
DEFAULT_BENCH_LABEL = "Nifty 500"

# ================== JdK proxy math ====================
def jdk_components(price: pd.Series, bench: pd.Series, win: int = WINDOW):
    df = pd.concat([price.rename("p"), bench.rename("b")], axis=1).dropna()
    if df.empty: return pd.Series(dtype=float), pd.Series(dtype=float)
    rs = 100 * (df["p"]/df["b"])
    m = rs.rolling(win).mean()
    s = rs.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rs_ratio = (100 + (rs - m)/s).dropna()
    rroc = rs_ratio.pct_change().mul(100)
    m2 = rroc.rolling(win).mean()
    s2 = rroc.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rs_mom = (101 + (rroc - m2)/s2).dropna()
    ix = rs_ratio.index.intersection(rs_mom.index)
    return rs_ratio.loc[ix], rs_mom.loc[ix]

def has_min_coverage(rr: pd.Series, mm: pd.Series, *, min_points=20, lookback_ok=30) -> bool:
    if rr is None or mm is None: return False
    ok=(~rr.isna()) & (~mm.isna())
    if ok.sum()<min_points: return False
    tail = ok.iloc[-lookback_ok:] if len(ok)>=lookback_ok else ok
    return bool(tail.any())

def get_status(x: float, y: float) -> str:
    if x <= 100 and y <= 100: return "Lagging"
    if x >= 100 and y >= 100: return "Leading"
    if x <= 100 and y >= 100: return "Improving"
    if x >= 100 and y <= 100: return "Weakening"
    return "Unknown"

def status_bg_color(x: float, y: float) -> str:
    return {"Lagging":"#e06a6a","Leading":"#3fa46a","Improving":"#5d86d1","Weakening":"#e2d06b"}.get(get_status(x,y),"#aaaaaa")

# ================== Data download (cached) =============
@st.cache_data(show_spinner=False)
def download_block_with_benchmark(tickers: List[str], benchmark: str, period: str, interval: str):
    raw = yf.download(list(tickers)+[benchmark], period=period, interval=interval,
                      group_by="ticker", auto_adjust=True, progress=False, threads=True)
    def _pick(sym):
        if isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
            for lvl in ("Close","Adj Close"):
                if (sym, lvl) in raw.columns: return raw[(sym,lvl)].dropna()
        else:
            for col in ("Close","Adj Close"):
                if col in raw.columns: return raw[col].dropna()
        return pd.Series(dtype=float)
    bench = _pick(benchmark)
    if bench is None or bench.empty: return bench, {}
    drop_last = not _is_bar_complete_for_timestamp(bench.index[-1], interval, now=_now_ist())
    def _trim(s: pd.Series):
        if drop_last and len(s)>=1: return s.iloc[:-1]
        return s
    bench = _trim(bench)
    data={}
    for t in tickers:
        s=_pick(t)
        if not s.empty: data[t]=_trim(s)
    return bench, data

def symbol_color_map(symbols: List[str]) -> Dict[str,str]:
    tab = plt.get_cmap('tab20').colors
    return {s: to_hex(tab[i % len(tab)], keep_alpha=False) for i,s in enumerate(symbols)}

# ================== Shared render helpers ==============
def init_rrg_axes(ax):
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

def format_bar_date(ts: pd.Timestamp, interval: str) -> str:
    ts = pd.Timestamp(ts)
    if interval=="1wk": return ts.to_period("W-FRI").end_time.date().isoformat()
    if interval=="1mo": return ts.to_period("M").end_time.date().isoformat()
    return ts.date().isoformat()

def compute_rank_metric(rank_mode: str, t: str, rs_ratio_map, rs_mom_map, tickers_data, idx, start_pos, end_pos) -> float:
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

# ================== STREAMLIT FRONTEND =================
def run_streamlit():
    st.set_page_config(page_title="Relative Rotation Graph (RRG)", layout="wide")
    st.title("Relative Rotation Graph (RRG)")

    # Sidebar: universes & controls (mirrors your original control panel)
    st.sidebar.header("RRG — Controls")
    NAME_MAP, DISPLAY_LIST = build_name_maps(CSV_DIR)
    uploaded_csv=None
    if not DISPLAY_LIST:
        st.sidebar.warning("No CSVs found in ./ticker. Upload one to proceed.")
        uploaded_csv = st.sidebar.file_uploader("Upload universe CSV", type=["csv"])

    if uploaded_csv is not None:
        df_up = pd.read_csv(uploaded_csv)
        tmp_dir = pathlib.Path("./.tmp"); tmp_dir.mkdir(exist_ok=True)
        tmp_path = tmp_dir / "uploaded.csv"; df_up.to_csv(tmp_path, index=False)
        NAME_MAP = {"Uploaded file": tmp_path.name}; DISPLAY_LIST = ["Uploaded file"]
        CSV_PATH = str(tmp_path)
    else:
        if not DISPLAY_LIST: st.stop()
        default_disp = friendly_name_from_file(DEFAULT_CSV_BASENAME) if DEFAULT_CSV_BASENAME in NAME_MAP.values() else DISPLAY_LIST[0]
        csv_disp = st.sidebar.selectbox("Indices", DISPLAY_LIST, index=(DISPLAY_LIST.index(default_disp) if default_disp in DISPLAY_LIST else 0))
        CSV_PATH = os.path.join(CSV_DIR, NAME_MAP.get(csv_disp, DEFAULT_CSV_BASENAME))

    bench_label = st.sidebar.selectbox("Benchmark", list(BENCH_CHOICES.keys()),
                                       index=list(BENCH_CHOICES.keys()).index(DEFAULT_BENCH_LABEL))
    tf_label = st.sidebar.selectbox("Strength vs (TF)", TF_LABELS, index=TF_LABELS.index(DEFAULT_TF))
    interval = TF_TO_INTERVAL[tf_label]
    suggested = TF_DEFAULT_PERIOD.get(interval, PERIOD_MAP[DEFAULT_PERIOD])
    period_label = st.sidebar.selectbox("Period", list(PERIOD_MAP.keys()),
                                        index=list(PERIOD_MAP.values()).index(suggested))
    period = PERIOD_MAP[period_label]
    tail_len = st.sidebar.slider("Trail Length", 1, 20, DEFAULT_TAIL, 1)
    RANK_MODES = ["RRG Power (dist)","RS-Ratio","RS-Momentum","Price %Δ (tail)","Momentum Slope (tail)"]
    rank_mode = st.sidebar.selectbox("Rank by", RANK_MODES, index=0)

    try:
        UNIVERSE, META = load_universe_from_csv(CSV_PATH)
    except Exception as e:
        st.error(f"Could not load universe CSV.\n\n{e}"); st.stop()

    # Download with fallback benchmarks (like your build_rrg_with_benchmark order)
    bench_order = [bench_label] + [l for l in BENCH_CHOICES.keys() if l != bench_label]
    last_err=None
    for lbl in bench_order:
        try:
            bench_symbol = BENCH_CHOICES[lbl]
            bench_series, px_map = download_block_with_benchmark(UNIVERSE, bench_symbol, period, interval)
            if bench_series is None or bench_series.empty: raise RuntimeError("Empty benchmark")
            benchmark_used_label = lbl; benchmark_used = bench_symbol
            tickers_data = px_map; break
        except Exception as e:
            last_err=e
    else:
        st.error(f"No benchmark returned data. Last error: {last_err}"); st.stop()

    # Build RRG series
    rs_ratio_map, rs_mom_map = {}, {}
    kept=[]; bench_idx = bench_series.index
    for t, s in tickers_data.items():
        if t==benchmark_used: continue
        rr, mm = jdk_components(s, bench_series, WINDOW)
        if len(rr)==0 or len(mm)==0: continue
        rr = rr.reindex(bench_idx); mm = mm.reindex(bench_idx)
        if has_min_coverage(rr,mm,min_points=max(WINDOW+5,20), lookback_ok=30):
            rs_ratio_map[t]=rr; rs_mom_map[t]=mm; kept.append(t)
    if not kept:
        st.error("After alignment, no symbols have enough coverage. Try a longer period."); st.stop()

    tickers = kept
    SYMBOL_COLORS = symbol_color_map(tickers)
    idx = bench_idx; idx_len = len(idx)
    st.subheader(f"{friendly_name_from_file(os.path.basename(CSV_PATH))} — {benchmark_used_label} — {period_label} — {tf_label}")
    if idx_len<2: st.warning("Not enough bars to plot."); st.stop()

    end_pos = st.slider("Date index", min_value=min(DEFAULT_TAIL, idx_len-1),
                        max_value=idx_len-1, value=idx_len-1, step=1, format="%d")
    start_pos = max(end_pos - tail_len, 0)
    current_date_str = format_bar_date(idx[end_pos], interval)
    st.caption(f"Date: **{current_date_str}**")

    # Visible list (keeps your “Visible” column concept)
    visible = st.multiselect("Visible symbols", options=tickers, default=tickers[:60])

    # Plot
    fig, ax = plt.subplots(1,1, figsize=(8,6))
    init_rrg_axes(ax)
    for t in tickers:
        if t not in visible: continue
        rr = rs_ratio_map[t].iloc[start_pos+1:end_pos+1].dropna()
        mm = rs_mom_map[t].iloc[start_pos+1:end_pos+1].dropna()
        rr, mm = rr.align(mm, join="inner")
        if len(rr)==0 or len(mm)==0: continue
        col = SYMBOL_COLORS[t]
        ax.plot(rr.values, mm.values, linewidth=1.1, alpha=0.7, color=col)
        ax.scatter(rr.values[:-1], mm.values[:-1], s=18, linewidths=0.6, color=col, edgecolors="#333333")
        ax.scatter(rr.values[-1],  mm.values[-1],  s=70, linewidths=0.8, color=col, edgecolors="#333333")
        ax.annotate(t, (rr.values[-1], mm.values[-1]), fontsize=9, color=col)
    st.pyplot(fig, use_container_width=True)

    # Ranking table (same metrics as your UI)
    rows=[]
    for t in tickers:
        if t not in visible: continue
        rr_last=float(rs_ratio_map[t].iloc[end_pos]); mm_last=float(rs_mom_map[t].iloc[end_pos])
        if np.isnan(rr_last) or np.isnan(mm_last): continue
        metric = compute_rank_metric(rank_mode, t, rs_ratio_map, rs_mom_map, tickers_data, idx, start_pos, end_pos)
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
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("rank_metric", ascending=False).reset_index(drop=True)
        st.subheader("Ranking")
        st.dataframe(df, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download ranks CSV", df.to_csv(index=False), file_name=f"ranks_{current_date_str}.csv", mime="text/csv")
        with c2:
            minimal = df[["symbol","name","industry","status","rs_ratio","rs_momentum","price","pct_change_tail"]]
            st.download_button("Download table CSV", minimal.to_csv(index=False), file_name=f"table_{current_date_str}.csv", mime="text/csv")
    else:
        st.info("No visible symbols to rank.")

    # TradingView links (clickable, like your name-click)
    with st.expander("Open in TradingView"):
        def tv_symbol(symbol: str) -> str:
            s = symbol[:-3] if symbol.upper().endswith(".NS") else symbol
            return f"NSE:{s.replace('-', '_')}"
        links = [f"- [{t}]({'https://www.tradingview.com/chart/?symbol=' + quote(tv_symbol(t), safe='')})" for t in visible[:120]]
        st.markdown("\n".join(links) if links else "Select some symbols first.")

# ================== OPTIONAL DESKTOP/TK FRONTEND =======
def run_desktop_tk():
    """Local-only desktop UI. Tk is imported lazily so cloud deployments aren’t affected."""
    import tkinter as tk
    from tkinter import ttk, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    # Minimal re-implementation of your original window & ranking panel
    root = tk.Tk()
    root.title("Relative Rotation Graph (RRG)")
    root.geometry("1500x900"); root.resizable(True, True)
    style = ttk.Style(root)
    try: style.theme_use("clam")
    except Exception: pass

    paned = ttk.Panedwindow(root, orient="horizontal")
    left = ttk.Frame(paned, width=360); right = ttk.Frame(paned)
    paned.add(left, weight=0); paned.add(right, weight=1); paned.pack(fill="both", expand=True)

    left_top = ttk.Frame(left, padding=8); left_top.pack(side="top", fill="x")
    ttk.Label(left_top, text="RRG — Controls", font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0,8))

    # CSV picker
    NAME_MAP, DISPLAY_LIST = build_name_maps(CSV_DIR)
    csv_var = tk.StringVar(value=friendly_name_from_file(DEFAULT_CSV_BASENAME if DEFAULT_CSV_BASENAME in NAME_MAP.values() else (DISPLAY_LIST[0] if DISPLAY_LIST else "")))
    ttk.Label(left_top, text="Indices", font=("Segoe UI",10,"bold")).pack(anchor="w")
    indices_menu = ttk.Combobox(left_top, textvariable=csv_var, values=DISPLAY_LIST, state="readonly", width=24); indices_menu.pack(anchor="w", pady=(0,8))

    bench_var = tk.StringVar(value=DEFAULT_BENCH_LABEL)
    ttk.Label(left_top, text="Benchmark", font=("Segoe UI",10,"bold")).pack(anchor="w")
    bench_menu = ttk.Combobox(left_top, textvariable=bench_var, values=list(BENCH_CHOICES.keys()), state="readonly", width=18); bench_menu.pack(anchor="w", pady=(0,8))

    tf_var = tk.StringVar(value=DEFAULT_TF)
    ttk.Label(left_top, text="Strength vs (TF)", font=("Segoe UI",10,"bold")).pack(anchor="w")
    tf_menu = ttk.Combobox(left_top, textvariable=tf_var, values=TF_LABELS, state="readonly", width=12); tf_menu.pack(anchor="w", pady=(0,8))

    period_var = tk.StringVar(value=DEFAULT_PERIOD)
    ttk.Label(left_top, text="Period", font=("Segoe UI",10,"bold")).pack(anchor="w")
    period_menu = ttk.Combobox(left_top, textvariable=period_var, values=list(PERIOD_MAP.keys()), state="readonly", width=8); period_menu.pack(anchor="w", pady=(0,8))

    ttk.Label(left_top, text="Trail Length", font=("Segoe UI",10,"bold")).pack(anchor="w", pady=(10,0))
    tail_scale = ttk.Scale(left_top, from_=1, to=20, value=DEFAULT_TAIL, orient="horizontal"); tail_scale.pack(fill="x", pady=(0,8))

    # Right plot
    right_top = ttk.Frame(right); right_top.pack(side="top", fill="both", expand=True)
    fig, ax = plt.subplots(1,1); init_rrg_axes(ax)
    canvas = FigureCanvasTkAgg(fig, master=right_top); canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    # Ranking table area
    right_bottom = ttk.Frame(right); right_bottom.pack(side="bottom", fill="x")
    date_slider_frame = ttk.Frame(right_bottom, padding=(8,2)); date_slider_frame.pack(side="bottom", fill="x")
    date_label = ttk.Label(date_slider_frame, text="Date: -"); date_label.pack(side="right")
    date_scale = ttk.Scale(date_slider_frame, from_=0, to=1, value=1, orient="horizontal"); date_scale.pack(side="left", fill="x", expand=True, padx=(0,8))

    table_container = tk.Frame(master=right_bottom); table_container.pack(side="bottom", fill="both", expand=1)
    table_canvas = tk.Canvas(table_container, highlightthickness=0)
    vsb = ttk.Scrollbar(table_container, orient="vertical", command=table_canvas.yview)
    table_canvas.configure(yscrollcommand=vsb.set); vsb.pack(side="right", fill="y"); table_canvas.pack(side="left", fill="both", expand=1)
    table = tk.Frame(table_canvas); table_window = table_canvas.create_window((0,0), window=table, anchor="nw")
    def _on_frame_config(_e=None): table_canvas.configure(scrollregion=table_canvas.bbox("all"))
    def _on_canvas_config(e): table_canvas.itemconfig(table_window, width=e.width)
    table.bind("<Configure>", _on_frame_config); table_canvas.bind("<Configure>", _on_canvas_config)

    HEADERS=["#"," ","Name","Status","Industry","Price","Change %","Visible"]
    for j,h in enumerate(HEADERS):
        tk.Label(table, text=h, relief=tk.RIDGE, font=("Segoe UI",12,"bold"),
                 anchor=("w" if h in ("Name","Industry") else "center")).grid(row=0, column=j, sticky="nsew", padx=1, pady=1)
    for c in range(len(HEADERS)):
        table.grid_columnconfigure(c, weight=(3 if c in (2,4) else 1), uniform="cols")

    # Data state
    UNIVERSE=[]; META={}
    tickers=[]; tickers_data={}; rs_ratio_map={}; rs_mom_map={}
    idx=pd.DatetimeIndex([]); idx_len=0
    SYMBOL_COLORS={}; tickers_to_show=[]

    def rebuild_all():
        nonlocal UNIVERSE,META,tickers,tickers_data,rs_ratio_map,rs_mom_map,idx,idx_len,SYMBOL_COLORS,tickers_to_show
        # inputs
        disp = csv_var.get(); NAME_MAP,_disp = build_name_maps(CSV_DIR); csv_file = NAME_MAP.get(disp)
        csv_path = os.path.join(CSV_DIR, csv_file) if csv_file else os.path.join(CSV_DIR, DEFAULT_CSV_BASENAME)
        UNIVERSE,META = load_universe_from_csv(csv_path)
        bench_label = bench_var.get(); interval = TF_TO_INTERVAL[tf_var.get()]
        period = PERIOD_MAP[period_var.get()]
        # download (fallback)
        for lbl in [bench_label] + [l for l in BENCH_CHOICES if l!=bench_label]:
            try:
                bench_symbol = BENCH_CHOICES[lbl]
                bench_series, px_map = download_block_with_benchmark(UNIVERSE, bench_symbol, period, interval)
                if bench_series is None or bench_series.empty: raise RuntimeError("Empty benchmark")
                benchmark_used_label = lbl; benchmark_used=bench_symbol
                tickers_data = px_map; break
            except Exception: continue
        else:
            messagebox.showwarning("Data","Could not load benchmark/index data."); return

        # build series
        kept=[]; rs_ratio_map={}; rs_mom_map={}
        bench_idx = bench_series.index
        for t,s in tickers_data.items():
            if t==benchmark_used: continue
            rr,mm = jdk_components(s, bench_series, WINDOW)
            if len(rr)==0 or len(mm)==0: continue
            rr=rr.reindex(bench_idx); mm=mm.reindex(bench_idx)
            if has_min_coverage(rr,mm,min_points=max(WINDOW+5,20), lookback_ok=30):
                rs_ratio_map[t]=rr; rs_mom_map[t]=mm; kept.append(t)
        if not kept:
            messagebox.showwarning("Data","No symbols with enough coverage."); return

        tickers=kept; SYMBOL_COLORS.update(symbol_color_map(tickers))
        idx=bench_idx; idx_len=len(idx)
        date_scale.configure(from_=DEFAULT_TAIL, to=idx_len-1); date_scale.set(idx_len-1)
        date_label.config(text=f"Date: {format_bar_date(idx[idx_len-1], interval)}")
        tickers_to_show[:] = tickers[:]

        # Plot & rows
        for w in table.grid_slaves():
            if int(w.grid_info().get("row",0))==0: continue
            w.destroy()
        init_rrg_axes(ax)
        for t in tickers:
            tk.Label(table, text="", relief=tk.RIDGE).grid(row=len(table.grid_slaves())+1, column=0)  # placeholder
        canvas.draw()

    def refresh_plot(*_):
        if idx_len==0: return
        interval = TF_TO_INTERVAL[tf_var.get()]
        end_pos=int(round(date_scale.get())); start_pos=max(end_pos - int(round(tail_scale.get())), 0)
        date_label.config(text=f"Date: {format_bar_date(idx[end_pos], interval)}")
        init_rrg_axes(ax)
        for t in tickers:
            if t not in tickers_to_show: continue
            rr=rs_ratio_map[t].iloc[start_pos+1:end_pos+1].dropna()
            mm=rs_mom_map[t].iloc[start_pos+1:end_pos+1].dropna()
            rr,mm = rr.align(mm, join="inner")
            if len(rr)==0 or len(mm)==0: continue
            col = SYMBOL_COLORS.get(t,"#444")
            ax.plot(rr.values, mm.values, linewidth=1.1, alpha=0.7, color=col)
            ax.scatter(rr.values[:-1], mm.values[:-1], s=18, linewidths=0.6, color=col, edgecolors="#333333")
            ax.scatter(rr.values[-1],  mm.values[-1],  s=70, linewidths=0.8, color=col, edgecolors="#333333")
            ax.annotate(t, (rr.values[-1], mm.values[-1]), fontsize=9, color=col)
        canvas.draw()

    indices_menu.bind("<<ComboboxSelected>>", lambda e: (rebuild_all(), refresh_plot()))
    bench_menu.bind("<<ComboboxSelected>>", lambda e: (rebuild_all(), refresh_plot()))
    tf_menu.bind("<<ComboboxSelected>>", lambda e: (rebuild_all(), refresh_plot()))
    period_menu.bind("<<ComboboxSelected>>", lambda e: (rebuild_all(), refresh_plot()))
    date_scale.configure(command=lambda v: refresh_plot())

    rebuild_all(); refresh_plot()
    root.mainloop()

# ================== ENTRY =============================
if os.environ.get("DESKTOP","0") == "1":
    # Optional local desktop run (requires system Tk)
    run_desktop_tk()
else:
    # Default is Streamlit UI (works on Streamlit Cloud)
    run_streamlit()

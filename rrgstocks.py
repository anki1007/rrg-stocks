import os, json, time, pathlib, logging, functools, calendar, webbrowser
import datetime as _dt
import email.utils as _eutils
import urllib.request as _urlreq
from urllib.parse import quote
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors

import tkinter as tk
from tkinter import ttk, messagebox

# ================== Defaults & Flags ==================
DEFAULT_TF = "Daily"          # Strength vs (TF) default
DEFAULT_PERIOD = "1Y"         # Period default
APPLY_SAVED_STATE = False     # Set True if you want to restore last session

# ================== User CSV Folder ===================
CSV_DIR  = r"D:\RRG\ticker"                          # folder to scan
DEFAULT_CSV_BASENAME = "nifty200.csv"                # default selection
CSV_PATH = os.path.join(CSV_DIR, DEFAULT_CSV_BASENAME)

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
    handlers=[logging.FileHandler("rrg.log"), logging.StreamHandler()]
)

# ================== Universe from CSV =================
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
    except Exception as e:
        logging.warning(f"[CSV list] {e}")
        return []

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

def build_name_maps(folder: str):
    files = list_csv_files(folder)
    if not files:
        return {}, []
    name_map = {friendly_name_from_file(f): f for f in files}
    display_list = sorted(name_map.keys())
    return name_map, display_list

def set_universe_from_path(csv_path: str):
    global UNIVERSE, META, CSV_PATH
    UNIVERSE, META = load_universe_from_csv(csv_path)
    CSV_PATH = csv_path
    logging.info(f"Loaded {len(UNIVERSE)} symbols from CSV: {os.path.basename(csv_path)}")

# first load
set_universe_from_path(CSV_PATH)

# ================== Config ============================
PERIOD_MAP = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y":"3y", "5Y":"5y", "10Y":"10y"}
period_label = DEFAULT_PERIOD
period = PERIOD_MAP[period_label]

TF_LABELS = ["Daily", "Weekly", "Monthly"]
TF_TO_INTERVAL = {"Daily":"1d", "Weekly":"1wk", "Monthly":"1mo"}
TF_DEFAULT_PERIOD = {"1d":"1y", "1wk":"3y", "1mo":"10y"}

interval = TF_TO_INTERVAL[DEFAULT_TF]   # default Daily
WINDOW = 14
DEFAULT_TAIL = 8

BENCH_CHOICES = {"Nifty 500":"^CRSLDX", "Nifty 200":"^CNX200", "Nifty 50":"^NSEI"}
DEFAULT_BENCH_LABEL = "Nifty 500"

# ================== Globals ===========================
is_playing = False
tickers: List[str] = []
tickers_data: Dict[str, pd.Series] = {}
benchmark_data: pd.Series = pd.Series(dtype=float)
benchmark_used_label: str = DEFAULT_BENCH_LABEL
benchmark_used: str = BENCH_CHOICES[DEFAULT_BENCH_LABEL]

rs_ratio_map: Dict[str, pd.Series] = {}
rs_mom_map: Dict[str, pd.Series] = {}

idx: pd.DatetimeIndex = pd.DatetimeIndex([])
idx_len: int = 0

SYMBOL_COLORS: Dict[str, str] = {}

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

def safe_long_name(symbol: str) -> str:
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

def table_text_color(bg: str) -> str:
    return "white" if bg in ("#e06a6a","#3fa46a","#5d86d1") else "black"

# ================== TradingView helpers ================
def tv_symbol(symbol: str) -> str:
    s = display_symbol(symbol).replace("-", "_")
    return f"NSE:{s}"

def open_tradingview(symbol: str):
    try:
        url = "https://www.tradingview.com/chart/?symbol=" + quote(tv_symbol(symbol), safe="")
        webbrowser.open_new_tab(url)
    except Exception as e:
        logging.warning(f"Could not open TradingView for {symbol}: {e}")
        try:
            messagebox.showwarning("TradingView", f"Couldn't open TradingView for {symbol}.\n\n{e}")
        except Exception:
            pass

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

def _now_ist(use_network=True) -> pd.Timestamp:
    if use_network:
        try:
            m = time.monotonic()
            if (_NET_NOW_CACHE["ts"] is None) or (m - _NET_NOW_CACHE["mono"] > NET_TIME_MAX_AGE):
                utc_now = _utc_now_from_network()
                _NET_NOW_CACHE["ts"] = utc_now.tz_convert(IST_TZ)
                _NET_NOW_CACHE["mono"] = m
            return _NET_NOW_CACHE["ts"]
        except Exception as e:
            logging.warning(f"[Time] Network time failed; using system clock. {e}")
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

# ================== Cache + Retry ======================
CACHE_DIR = pathlib.Path("cache"); CACHE_DIR.mkdir(exist_ok=True)
def _cache_path(symbol: str) -> pathlib.Path:
    safe = symbol.replace("^","").replace(".","_")
    return CACHE_DIR / f"{safe}_{period}_{interval}.parquet"
def _save_cache(symbol: str, s: pd.Series):
    try: s.to_frame("Close").to_parquet(_cache_path(symbol))
    except Exception as e: logging.warning(f"Cache save failed for {symbol}: {e}")
def _load_cache(symbol: str) -> pd.Series:
    p=_cache_path(symbol)
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
@retry()
def download_block_with_benchmark(tickers: List[str], benchmark: str, period: str, interval: str, exclude_partial=True):
    raw = yf.download(list(tickers)+[benchmark], period=period, interval=interval,
                      group_by="ticker", auto_adjust=True, progress=False, threads=True)

    def _pick(sym):
        return pick_close(raw, sym).dropna()

    bench = _pick(benchmark)
    if bench is None or bench.empty:
        return bench, {}

    keep_last = _is_bar_complete_for_timestamp(bench.index[-1], interval, now=_now_ist())
    drop_last = not keep_last

    def _maybe_trim(s: pd.Series):
        if drop_last and len(s) >= 1:
            return s.iloc[:-1]
        return s

    bench = _maybe_trim(bench)
    data: Dict[str,pd.Series] = {}
    for t in tickers:
        s = _pick(t)
        if not s.empty:
            data[t] = _maybe_trim(s)
        else:
            c = _load_cache(t)
            if not c.empty:
                data[t] = _maybe_trim(c)

    if not bench.empty: _save_cache(benchmark, bench)
    for t, s in data.items(): _save_cache(t, s)
    return bench, data

# ================== Colors =============================
def symbol_color_map(symbols: List[str]) -> Dict[str, str]:
    tab = plt.get_cmap('tab20').colors
    return {s: to_hex(tab[i % len(tab)], keep_alpha=False) for i,s in enumerate(symbols)}

# ================== Build ==============================
def build_rrg_with_benchmark(label_order: List[str]):
    global tickers, tickers_data, benchmark_data, benchmark_used, benchmark_used_label
    global rs_ratio_map, rs_mom_map, idx, idx_len, SYMBOL_COLORS

    last_err=None
    for lbl in label_order:
        try:
            b = BENCH_CHOICES[lbl]
            bench, data = download_block_with_benchmark(UNIVERSE, b, period, interval, exclude_partial=True)
            if bench is None or bench.empty: raise RuntimeError("Empty benchmark")
            benchmark_used_label = lbl; benchmark_used = b
            benchmark_data = bench; tickers_data = data
            logging.info(f"Using benchmark: {lbl} ({b}), interval={interval}, period={period}")
            break
        except Exception as e:
            last_err=e; logging.warning(f"Benchmark {lbl} failed: {e}")
    else:
        raise RuntimeError(f"No benchmark returned data. Last error: {last_err}")

    rs_ratio_map, rs_mom_map = {}, {}
    kept=[]
    bench_idx = benchmark_data.index

    for t, s in tickers_data.items():
        if t == benchmark_used: continue
        rr, mm = jdk_components(s, benchmark_data, WINDOW)
        if len(rr)==0 or len(mm)==0: continue
        rr = rr.reindex(bench_idx)
        mm = mm.reindex(bench_idx)
        if has_min_coverage(rr, mm, min_points=max(WINDOW+5,20), lookback_ok=30):
            rs_ratio_map[t]=rr; rs_mom_map[t]=mm; kept.append(t)

    if not kept:
        raise RuntimeError("After alignment, no symbols have enough coverage. Try longer period.")

    tickers = kept
    SYMBOL_COLORS = symbol_color_map(tickers)
    idx = bench_idx
    idx_len = len(idx)

# ================== UI ================================
root = tk.Tk()
root.title("Relative Rotation Graph (RRG)")
root.geometry("1500x900")
root.resizable(True, True)

style = ttk.Style(root)
try: style.theme_use('clam')
except Exception: pass

paned = ttk.Panedwindow(root, orient="horizontal")
left = ttk.Frame(paned, width=360); right = ttk.Frame(paned)
paned.add(left, weight=0); paned.add(right, weight=1); paned.pack(fill="both", expand=True)

left_top = ttk.Frame(left, padding=8); left_top.pack(side="top", fill="x")
ttk.Label(left_top, text="RRG — Controls", font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

# ---- Indices dropdown with friendly names ----
NAME_MAP, DISPLAY_LIST = build_name_maps(CSV_DIR)       # display → filename
if DEFAULT_CSV_BASENAME not in NAME_MAP.values() and os.path.exists(CSV_PATH):
    disp = friendly_name_from_file(DEFAULT_CSV_BASENAME)
    NAME_MAP[disp] = DEFAULT_CSV_BASENAME
    DISPLAY_LIST = sorted(set(DISPLAY_LIST + [disp]))

current_disp = friendly_name_from_file(os.path.basename(CSV_PATH))
csv_disp_var = tk.StringVar(value=current_disp)
ttk.Label(left_top, text="Indices", font=("Segoe UI", 10, "bold")).pack(anchor="w")
indices_menu = ttk.Combobox(left_top, textvariable=csv_disp_var, values=DISPLAY_LIST, state="readonly", width=24)
indices_menu.pack(anchor="w", pady=(0,8))

def on_indices_change(_e=None):
    disp = csv_disp_var.get()
    basename = NAME_MAP.get(disp)
    if not basename:
        return
    path = os.path.join(CSV_DIR, basename)
    logging.info(f"[CSV switch] {disp} -> {basename}")
    try:
        set_universe_from_path(path)
        initial_build()
        root.title(f"Relative Rotation Graph (RRG) — {bench_var.get()} — {period_var.get()} — {tf_var.get()} — {disp}")
    except Exception as e:
        logging.warning(f"[CSV switch failed] {e}")
        try:
            messagebox.showwarning("Indices", f"Could not load {disp}\n\n{e}")
        except Exception:
            pass
        csv_disp_var.set(friendly_name_from_file(os.path.basename(CSV_PATH)))

indices_menu.bind("<<ComboboxSelected>>", on_indices_change)

bench_var = tk.StringVar(value=DEFAULT_BENCH_LABEL)
ttk.Label(left_top, text="Benchmark", font=("Segoe UI", 10, "bold")).pack(anchor="w")
bench_menu = ttk.Combobox(left_top, textvariable=bench_var, values=list(BENCH_CHOICES.keys()), state="readonly", width=18)
bench_menu.pack(anchor="w", pady=(0,8))

# Strength timeframe dropdown (Default = Daily)
tf_var = tk.StringVar(value=DEFAULT_TF)
ttk.Label(left_top, text="Strength vs (TF)", font=("Segoe UI", 10, "bold")).pack(anchor="w")
tf_menu = ttk.Combobox(left_top, textvariable=tf_var, values=TF_LABELS, state="readonly", width=12)
tf_menu.pack(anchor="w", pady=(0,8))

PERIOD_CHOICES = list(PERIOD_MAP.keys())
period_var = tk.StringVar(value=DEFAULT_PERIOD)
ttk.Label(left_top, text="Period", font=("Segoe UI", 10, "bold")).pack(anchor="w")
period_menu = ttk.Combobox(left_top, textvariable=period_var, values=PERIOD_CHOICES, state="readonly", width=8)
period_menu.pack(anchor="w", pady=(0,8))

RANK_MODES = ["RRG Power (dist)", "RS-Ratio", "RS-Momentum", "Price %Δ (tail)", "Momentum Slope (tail)"]
rank_mode_var = tk.StringVar(value=RANK_MODES[0])
ttk.Label(left_top, text="Rank by", font=("Segoe UI", 10, "bold")).pack(anchor="w")
rank_menu = ttk.Combobox(left_top, textvariable=rank_mode_var, values=RANK_MODES, state="readonly", width=22)
rank_menu.pack(anchor="w", pady=(0,8))

ttk.Label(left_top, text="Trail Length", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10,0))
tail_scale = ttk.Scale(left_top, from_=1, to=20, value=DEFAULT_TAIL, orient="horizontal")
tail_scale.pack(fill="x", pady=(0,8))

right_top = ttk.Frame(right); right_top.pack(side="top", fill="both", expand=True)
right_bottom = ttk.Frame(right); right_bottom.pack(side="bottom", fill="x")

fig, ax = plt.subplots(1, 1)
canvas = FigureCanvasTkAgg(fig, master=right_top); canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

ax_rank = fig.add_axes([0.84, 0.12, 0.14, 0.76]); ax_rank.set_title("Ranking", fontsize=11, pad=6); ax_rank.axis("off")
rank_text_objects = []

# -------- Scrollable grid table --------
HEADERS = ["#", " ", "Name", "Status", "Industry", "Price", "Change %", "Visible"]

table_container = tk.Frame(master=right_bottom)
table_container.pack(side="bottom", fill="both", expand=1)

table_canvas = tk.Canvas(table_container, highlightthickness=0)
vsb = ttk.Scrollbar(table_container, orient="vertical", command=table_canvas.yview)
table_canvas.configure(yscrollcommand=vsb.set)
vsb.pack(side="right", fill="y")
table_canvas.pack(side="left", fill="both", expand=1)

table = tk.Frame(table_canvas)
table_window = table_canvas.create_window((0, 0), window=table, anchor="nw")

def _on_frame_config(_event=None):
    table_canvas.configure(scrollregion=table_canvas.bbox("all"))
def _on_canvas_config(event):
    table_canvas.itemconfig(table_window, width=event.width)
table.bind("<Configure>", _on_frame_config)
table_canvas.bind("<Configure>", _on_canvas_config)

def _bind_mousewheel(widget):
    def _on_mousewheel(e):
        delta = -1 if e.delta > 0 else 1
        table_canvas.yview_scroll(delta, "units")
        return "break"
    def _on_linux_scroll(e):
        table_canvas.yview_scroll(-1 if e.num == 4 else 1, "units")
        return "break"
    widget.bind_all("<MouseWheel>", _on_mousewheel)
    widget.bind_all("<Button-4>", _on_linux_scroll)
    widget.bind_all("<Button-5>", _on_linux_scroll)
_bind_mousewheel(table_canvas)

for j, h in enumerate(HEADERS):
    tk.Label(
        table,
        text=h,
        relief=tk.RIDGE,
        font=("Segoe UI", 12, "bold"),
        anchor=("w" if h in ("Name", "Industry") else "center")
    ).grid(row=0, column=j, sticky="nsew", padx=1, pady=1)

for c in range(len(HEADERS)):
    table.grid_columnconfigure(c, weight=(3 if c in (2, 4) else 1), uniform="cols")

date_slider_frame = ttk.Frame(right_bottom, padding=(8,2)); date_slider_frame.pack(side="bottom", fill="x")
date_label = ttk.Label(date_slider_frame, text="Date: -"); date_label.pack(side="right")
date_scale = ttk.Scale(date_slider_frame, from_=0, to=1, value=1, orient="horizontal"); date_scale.pack(side="left", fill="x", expand=True, padx=(0,8))

# ================== Plot Axes ==========================
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

# ================== UI Data Structures =================
meta = {}
tickers_to_show: List[str] = []
row_widgets: Dict[str,dict] = {}
row_order: List[str] = []
scatter_plots: Dict[str, any] = {}
line_plots: Dict[str, any] = {}
annotations: Dict[str, any] = {}

def rebuild_row(symbol: str, row: int):
    name = meta[symbol]["name"]; industry = META.get(symbol,{}).get("industry","-")
    if idx_len == 0: price = chg = np.nan; status = "-"
    else:
        end_pos = int(round(date_scale.get())); start_pos = max(end_pos - int(round(tail_scale.get())), 0)
        last_ts = idx[end_pos]; start_ts = idx[start_pos]
        px_series = tickers_data[symbol].reindex(idx).dropna()
        price = float(px_series.loc[last_ts]) if last_ts in px_series.index else np.nan
        chg = (px_series.loc[last_ts] / px_series.loc[start_ts] - 1) * 100 if (start_ts in px_series.index and last_ts in px_series.index) else np.nan
        rr_val = rs_ratio_map[symbol].iloc[-1]; mm_val = rs_mom_map[symbol].iloc[-1]
        status = get_status(rr_val, mm_val)

    rr_val = rs_ratio_map[symbol].iloc[-1]; mm_val = rs_mom_map[symbol].iloc[-1]
    bg = status_bg_color(rr_val, mm_val); fg = table_text_color(bg)

    lbl_rank = tk.Label(table, text="", relief=tk.RIDGE)
    lbl_rank.grid(row=row, column=0, sticky="nsew", padx=1, pady=1)

    tk.Label(table, text="", bg=SYMBOL_COLORS[symbol], relief=tk.RIDGE).grid(row=row, column=1, sticky="nsew", padx=1, pady=1)

    name_fg_link = "#1a73e8"
    name_lbl = tk.Label(
        table, text=name, relief=tk.RIDGE, bg=bg, fg=name_fg_link,
        font=("Segoe UI", 11, "underline"), anchor="w", cursor="hand2"
    )
    name_lbl.grid(row=row, column=2, sticky="nsew", padx=1, pady=1)
    name_lbl.bind("<Button-1>", lambda _e, sym=symbol: open_tradingview(sym))

    lbl_stat = tk.Label(table, text=status, relief=tk.RIDGE, bg=bg, fg=fg, font=("Segoe UI", 11))
    lbl_stat.grid(row=row, column=3, sticky="nsew", padx=1, pady=1)

    tk.Label(table, text=industry, relief=tk.RIDGE, bg=bg, fg=fg, font=("Segoe UI", 11), anchor="w").grid(row=row, column=4, sticky="nsew", padx=1, pady=1)

    lbl_px  = tk.Label(table, text=("-" if pd.isna(price) else f"{price:.2f}"), relief=tk.RIDGE, bg=bg, fg=fg, font=("Segoe UI", 11))
    lbl_px.grid(row=row, column=5, sticky="nsew", padx=1, pady=1)
    lbl_chg = tk.Label(table, text=("-" if pd.isna(chg) else f"{chg:.2f}"), relief=tk.RIDGE, bg=bg, fg=fg, font=("Segoe UI", 11))
    lbl_chg.grid(row=row, column=6, sticky="nsew", padx=1, pady=1)

    var_vis = tk.BooleanVar(value=True)
    chk = ttk.Checkbutton(table, variable=var_vis)
    chk.grid(row=row, column=7, sticky="nsew", padx=1, pady=1); chk.state(["selected"])
    def on_checkbox_click(_):
        if var_vis.get():
            if symbol not in tickers_to_show: tickers_to_show.append(symbol)
        else:
            if symbol in tickers_to_show: tickers_to_show.remove(symbol)
    chk.bind("<ButtonRelease-1>", on_checkbox_click)

    row_widgets[symbol] = {
        "rank": lbl_rank, "status": lbl_stat, "price": lbl_px, "chg": lbl_chg,
        "vis_var": var_vis, "name": name_lbl,
    }

def swap_symbol(row: int, old: str, new: str):
    global tickers, tickers_data, rs_ratio_map, rs_mom_map, SYMBOL_COLORS, meta, idx
    try:
        raw = yf.download(new, period=period, interval=interval, auto_adjust=True, progress=False)
        s = pick_close(raw, new).dropna()
        if not _is_bar_complete_for_timestamp(s.index[-1], interval, now=_now_ist()) and len(s) >= 1:
            s = s.iloc[:-1]
        if s.empty: raise ValueError(f"No data for {new}")
        rr, mm = jdk_components(s, benchmark_data, WINDOW)
        if rr.empty or mm.empty: raise ValueError(f"Insufficient overlap for {new}")
        rr = rr.reindex(idx); mm = mm.reindex(idx)
        if not has_min_coverage(rr, mm, min_points=max(WINDOW+5,20), lookback_ok=30):
            raise ValueError(f"{new} doesn't have enough history on the benchmark timeline")

        tickers_data[new] = s; meta[new] = {"symbol": new, "name": safe_long_name(new)}
        rs_ratio_map[new] = rr; rs_mom_map[new] = mm
        SYMBOL_COLORS[new] = SYMBOL_COLORS.get(old, to_hex(plt.get_cmap('tab20')(len(SYMBOL_COLORS) % 20)))

        for col in range(len(HEADERS)):
            w = table.grid_slaves(row=row, column=col)
            if w: w[0].destroy()
        row_widgets.pop(old, None)

        if old in tickers: tickers[tickers.index(old)] = new
        else: tickers.append(new)
        if old in tickers_to_show:
            tickers_to_show.remove(old); tickers_to_show.append(new)
        row_order[row-1] = new

        scatter_plots.pop(old, None); line_plots.pop(old, None); annotations.pop(old, None)
        scatter_plots[new] = ax.scatter([], [], s=10, linewidths=0.6, color=SYMBOL_COLORS[new])
        line_plots[new]    = ax.plot([], [], linewidth=1.1, alpha=0.6, color=SYMBOL_COLORS[new])[0]
        annotations[new]   = ax.annotate(new, (0, 0), fontsize=8, color=SYMBOL_COLORS[new])
        rebuild_row(new, row)
    except Exception as ex:
        logging.warning(f"[Swap failed] {ex}")

def current_window():
    if idx_len == 0: return 0, 0
    end_pos = int(round(date_scale.get()))
    T = int(round(tail_scale.get()))
    start_pos = max(end_pos - T, 0)
    return start_pos, end_pos

def compute_rank_metric(t: str, start_pos: int, end_pos: int) -> float:
    rr_last = rs_ratio_map[t].iloc[end_pos]; mm_last = rs_mom_map[t].iloc[end_pos]
    if np.isnan(rr_last) or np.isnan(mm_last): return float("-inf")
    mode = rank_mode_var.get()
    if mode == "RRG Power (dist)": return float(np.hypot(rr_last-100.0, mm_last-100.0))
    if mode == "RS-Ratio":         return float(rr_last)
    if mode == "RS-Momentum":      return float(mm_last)
    if mode == "Price %Δ (tail)":
        px = tickers_data[t].reindex(idx).dropna()
        if len(px.iloc[start_pos:end_pos+1]) >= 2:
            return float((px.iloc[end_pos] / px.iloc[start_pos] - 1) * 100.0)
        return float("-inf")
    if mode == "Momentum Slope (tail)":
        series = rs_mom_map[t].iloc[start_pos:end_pos+1].dropna()
        if len(series) >= 2:
            x = np.arange(len(series)); A = np.vstack([x, np.ones(len(x))]).T
            slope = np.linalg.lstsq(A, series.values, rcond=None)[0][0]
            return float(slope)
        return float("-inf")
    return float("-inf")

RANK_FONT_SIZE = 11
RANK_FONT_WEIGHT = "bold"

def animate(_):
    global rank_text_objects
    if idx_len == 0:
        canvas.draw_idle(); return []

    if is_playing:
        nxt = min(int(round(date_scale.get())) + 1, idx_len - 1)
        date_scale.set(nxt)

    start_pos, end_pos = current_window()
    date_label.config(text=f"Date: {format_bar_date(idx[end_pos], interval)}")

    for t in row_order:
        if t not in rs_ratio_map or t not in rs_mom_map: continue
        visible = t in tickers_to_show
        rr = rs_ratio_map[t].iloc[start_pos+1 : end_pos+1].dropna()
        mm = rs_mom_map[t].iloc[start_pos+1 : end_pos+1].dropna()
        rr, mm = rr.align(mm, join="inner")

        if not visible or len(rr)==0 or len(mm)==0:
            scatter_plots[t].set_offsets(np.empty((0, 2)))
            line_plots[t].set_data([], [])
            annotations[t].set_text("")
        else:
            col = SYMBOL_COLORS[t]
            pts = np.column_stack([rr.values, mm.values])
            sizes = [18] * (len(rr) - 1) + [70]
            scatter_plots[t].set_offsets(pts)
            scatter_plots[t].set_sizes(sizes)
            scatter_plots[t].set_facecolor(col)
            scatter_plots[t].set_edgecolor('#333333')
            line_plots[t].set_data(rr.values, mm.values)
            line_plots[t].set_color(col)
            rr_last = rr.values[-1]; mm_last = mm.values[-1]
            s_txt = get_status(rr_last, mm_last)
            annotations[t].set_text(f"{t}  [{s_txt}]")
            annotations[t].set_position((rr_last, mm_last))
            annotations[t].set_fontsize(9)

        rw = row_widgets.get(t)
        if rw:
            px = tickers_data[t].reindex(idx).dropna()
            price = float(px.iloc[end_pos]) if end_pos < len(px) else np.nan
            chg = ((px.iloc[end_pos]/px.iloc[start_pos]-1)*100.0) if (end_pos < len(px) and start_pos < len(px)) else np.nan
            rr_last_full = float(rs_ratio_map[t].iloc[end_pos]); mm_last_full = float(rs_mom_map[t].iloc[end_pos])
            bg = status_bg_color(rr_last_full, mm_last_full); fg = table_text_color(bg)
            for w in ("status","price","chg"):
                try: row_widgets[t][w].config(bg=bg, fg=fg)
                except Exception: pass
            try:
                if "name" in row_widgets[t]:
                    row_widgets[t]["name"].config(bg=bg)
                row_widgets[t]["status"].config(text=get_status(rr_last_full, mm_last_full))
                row_widgets[t]["price"].config(text=("-" if pd.isna(price) else f"{price:.2f}"))
                row_widgets[t]["chg"].config(text=("-" if pd.isna(chg) else f"{chg:.2f}"))
            except Exception: pass

    performances=[]
    for t in tickers:
        if t not in tickers_to_show: continue
        try:
            metric=compute_rank_metric(t, start_pos, end_pos)
            performances.append((t, metric))
        except Exception: pass
    performances.sort(key=lambda x:x[1], reverse=True)

    for txt in rank_text_objects:
        try: txt.remove()
        except Exception: pass
    rank_text_objects=[]
    max_rows=22; spacing=0.042
    for rank,(symbol,metric) in enumerate(performances[:max_rows], start=1):
        rr_last=float(rs_ratio_map[symbol].iloc[end_pos]); mm_last=float(rs_mom_map[symbol].iloc[end_pos])
        if np.isnan(rr_last) or np.isnan(mm_last): continue
        stat=get_status(rr_last, mm_last)
        y=1 - rank*spacing
        txt=ax_rank.text(
            0.0, y, f"{rank}. {display_symbol(symbol)} [{stat}]",
            fontsize=RANK_FONT_SIZE, fontweight=RANK_FONT_WEIGHT,
            color=SYMBOL_COLORS.get(symbol, '#333'),
            transform=ax_rank.transAxes
        )
        rank_text_objects.append(txt)

    ranks_dict={sym:r for r,(sym,_) in enumerate(performances, start=1)}
    for sym in row_order:
        rw=row_widgets.get(sym)
        if rw and "rank" in rw:
            rk=ranks_dict.get(sym, "")
            try: rw["rank"].config(text=str(rk))
            except Exception: pass

    canvas.draw_idle()
    return list(scatter_plots.values()) + list(line_plots.values()) + list(annotations.values()) + rank_text_objects

anim = animation.FuncAnimation(fig, animate, interval=150, blit=False, cache_frame_data=False)

hover_cursor = mplcursors.cursor(ax, hover=True)
@hover_cursor.connect("add")
def _on_add(sel):
    if idx_len == 0: return
    x, y = sel.target[0], sel.target[1]
    end_pos = int(round(date_scale.get()))
    nearest, dmin = None, 1e12
    for t in tickers_to_show:
        try:
            rr=float(rs_ratio_map[t].iloc[end_pos]); mm=float(rs_mom_map[t].iloc[end_pos])
            if np.isnan(rr) or np.isnan(mm): continue
            dx = rr - x; dy = mm - y; d = dx*dx + dy*dy
            if d < dmin: dmin, nearest = d, t
        except Exception: pass
    if nearest:
        rr=float(rs_ratio_map[nearest].iloc[end_pos]); mm=float(rs_mom_map[nearest].iloc[end_pos])
        status=get_status(rr, mm); strength=float(np.hypot(rr-100.0, mm-100.0))
        sel.annotation.set(text=f"{display_symbol(nearest)}  [{status}]\nRSR: {rr:.2f}  RSM: {mm:.2f}\nStrength: {strength:.2f}",
                           bbox=dict(boxstyle="round", fc="#ffffff", ec=SYMBOL_COLORS.get(nearest, "#333")),
                           color="#111")

# ================== Export Buttons =====================
def export_ranks_to_csv():
    if idx_len == 0: messagebox.showwarning("Export", "No data to export."); return
    end_pos=int(round(date_scale.get())); start_pos=max(end_pos - int(round(tail_scale.get())), 0)
    rows=[]
    for t in tickers:
        if t not in tickers_to_show: continue
        rr=float(rs_ratio_map[t].iloc[end_pos]); mm=float(rs_mom_map[t].iloc[end_pos])
        if np.isnan(rr) or np.isnan(mm): continue
        metric=compute_rank_metric(t, start_pos, end_pos)
        rows.append((t, META.get(t,{}).get("name",t), META.get(t,{}).get("industry","-"), metric, rr, mm, get_status(rr, mm)))
    rows.sort(key=lambda x:x[3], reverse=True)
    df=pd.DataFrame(rows, columns=["symbol","name","industry","rank_metric","rs_ratio","rs_momentum","status"])
    out=f"ranks_{format_bar_date(idx[end_pos], interval)}.csv"
    df.to_csv(out, index=False)
    logging.info(f"[Saved] {out}"); messagebox.showinfo("Export", f"Saved {out}")

def export_table_to_csv():
    if idx_len == 0: messagebox.showwarning("Export", "No data to export."); return
    end_pos=int(round(date_scale.get())); start_pos=max(end_pos - int(round(tail_scale.get())), 0)
    rows=[]
    for t in tickers:
        if t not in tickers_to_show: continue
        rr=float(rs_ratio_map[t].iloc[end_pos]); mm=float(rs_mom_map[t].iloc[end_pos])
        if np.isnan(rr) or np.isnan(mm): continue
        px=tickers_data[t].reindex(idx).dropna()
        price=float(px.iloc[end_pos]) if end_pos < len(px) else np.nan
        chg=((px.iloc[end_pos]/px.iloc[start_pos]-1)*100.0) if (end_pos < len(px) and start_pos < len(px)) else np.nan
        rows.append((t, META.get(t,{}).get("name",t), META.get(t,{}).get("industry","-"),
                     get_status(rr, mm), rr, mm, price, chg))
    df=pd.DataFrame(rows, columns=["symbol","name","industry","status","rs_ratio","rs_momentum","price","pct_change_tail"])
    out=f"table_{format_bar_date(idx[end_pos], interval)}.csv"
    df.to_csv(out, index=False)
    logging.info(f"[Saved] {out}"); messagebox.showinfo("Export", f"Saved {out}")

# ================== Control Hooks ======================
def toggle_play():
    global is_playing; is_playing = not is_playing

def on_date_scale(val):
    if idx_len == 0: return
    v = int(round(float(val))); v = max(0, min(v, idx_len - 1))
    date_label.config(text=f"Date: {format_bar_date(idx[v], interval)}")

def on_benchmark_change(_e=None):
    try: initial_build()
    except Exception as e:
        logging.warning(f"[Benchmark switch] {e}")
        try: messagebox.showwarning("Benchmark", f"Could not load benchmark: {bench_var.get()}\n\n{e}")
        except Exception: pass

def on_period_change(_e=None):
    global period_label, period
    new_label = period_var.get(); new_period = PERIOD_MAP.get(new_label, period)
    if new_period == period: return
    period_label = new_label; period = new_period
    logging.info(f"[Period] Switching to {new_label} ({new_period})")
    try: initial_build()
    except Exception as e:
        logging.warning(f"[Period switch] {e}")
        try: messagebox.showwarning("Period", f"Could not reload for period {period_var.get()}.\n\n{e}")
        except Exception: pass

def on_tf_change(_e=None):
    global interval, period
    new_tf = tf_var.get()
    interval = TF_TO_INTERVAL.get(new_tf, "1d")
    suggested = TF_DEFAULT_PERIOD.get(interval, period)
    if period not in PERIOD_MAP.values():
        period = suggested
        period_var.set([k for k,v in PERIOD_MAP.items() if v == suggested][0])
    try:
        initial_build()
    except Exception as e:
        logging.warning(f"[TF switch] {e}")
        try: messagebox.showwarning("Timeframe", f"Could not reload for timeframe {new_tf}.\n\n{e}")
        except Exception: pass

play_btn = ttk.Button(left_top, text="Play / Pause", command=toggle_play); play_btn.pack(anchor="w", pady=(6,2))
export_ranks_btn = ttk.Button(left_top, text="Export Ranks CSV", command=export_ranks_to_csv); export_ranks_btn.pack(anchor="w", pady=(2,2))
export_table_btn = ttk.Button(left_top, text="Export Table CSV", command=export_table_to_csv); export_table_btn.pack(anchor="w", pady=(2,6))
bench_menu.bind("<<ComboboxSelected>>", on_benchmark_change)
period_menu.bind("<<ComboboxSelected>>", on_period_change)
tf_menu.bind("<<ComboboxSelected>>", on_tf_change)
date_scale.configure(command=on_date_scale)

# ================== Build & State ======================
def create_table_and_plots():
    for w in table.grid_slaves():
        if int(w.grid_info().get("row", 0)) == 0: continue
        w.destroy()
    scatter_plots.clear(); line_plots.clear(); annotations.clear()
    init_plot_axes()
    for t in tickers:
        scatter_plots[t] = ax.scatter([], [], s=10, linewidths=0.6, color=SYMBOL_COLORS[t])
        line_plots[t]    = ax.plot([], [], linewidth=1.1, alpha=0.6, color=SYMBOL_COLORS[t])[0]
        annotations[t]   = ax.annotate(t, (0, 0), fontsize=8, color=SYMBOL_COLORS[t])
    row_widgets.clear(); row_order[:] = []
    for i, t in enumerate(tickers):
        row_order.append(t); rebuild_row(t, i + 1)

def initial_build():
    global idx, idx_len, meta, tickers_to_show
    sel_label = bench_var.get()
    order = [sel_label] + [l for l in BENCH_CHOICES.keys() if l != sel_label]
    try:
        build_rrg_with_benchmark(order)
    except Exception as e:
        logging.warning(f"[Initial build] {e}")
        idx = pd.DatetimeIndex([]); idx_len = 0
        try: messagebox.showwarning("Data", f"Could not load benchmark/index data.\n\n{e}")
        except Exception: pass
        init_plot_axes(); canvas.draw(); return

    date_scale.configure(from_=DEFAULT_TAIL, to=idx_len - 1)
    date_scale.set(idx_len - 1)
    date_label.config(text=f"Date: {format_bar_date(idx[idx_len - 1], interval)}")
    meta = {t: {"symbol": t, "name": safe_long_name(t)} for t in tickers}
    tickers_to_show[:] = tickers[:]
    create_table_and_plots()
    disp = friendly_name_from_file(os.path.basename(CSV_PATH))
    root.title(f"Relative Rotation Graph (RRG) — {sel_label} — {period_var.get()} — {tf_var.get()} — {disp}")

def save_state_and_quit():
    try:
        state = {
            "benchmark": bench_var.get(),
            "period": period_var.get(),
            "tf": tf_var.get(),
            "visible": [t for t in tickers if row_widgets.get(t, {}).get("vis_var", tk.BooleanVar(value=True)).get()],
            "tail": int(round(tail_scale.get())),
            "date_index": int(round(date_scale.get())) if idx_len else 0,
            "rank_mode": rank_mode_var.get(),
            "csv_file": os.path.basename(CSV_PATH),
        }
        with open("rrg_state.json", "w") as f: json.dump(state, f)
    except Exception as e:
        logging.warning(f"[State save] {e}")
    root.destroy()

def load_state():
    global period_label, period, interval
    if not APPLY_SAVED_STATE:
        return
    if not os.path.exists("rrg_state.json"):
        return
    try:
        s = json.load(open("rrg_state.json", "r"))
        csv_file = s.get("csv_file")
        if csv_file:
            path = os.path.join(CSV_DIR, csv_file)
            if os.path.exists(path):
                set_universe_from_path(path)
                csv_disp = friendly_name_from_file(csv_file)
                NAME_MAP[csv_disp] = csv_file
                csv_disp_var.set(csv_disp)
        bmk = s.get("benchmark")
        if bmk in BENCH_CHOICES: bench_var.set(bmk)
        if "period" in s and s["period"] in PERIOD_MAP:
            period_label = s["period"]; period = PERIOD_MAP[period_label]; period_var.set(period_label)
        if "tf" in s and s["tf"] in TF_LABELS:
            tf_var.set(s["tf"]); interval = TF_TO_INTERVAL[tf_var.get()]
        if "rank_mode" in s and s["rank_mode"] in ["RRG Power (dist)","RS-Ratio","RS-Momentum","Price %Δ (tail)","Momentum Slope (tail)"]:
            rank_mode_var.set(s["rank_mode"])
    except Exception as e:
        logging.warning(f"[State load] {e}")

root.protocol("WM_DELETE_WINDOW", save_state_and_quit)

# ================== Start =============================
load_state()          # will do nothing if APPLY_SAVED_STATE=False
initial_build()
canvas.draw()
root.mainloop()

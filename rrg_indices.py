import os, json, time, yaml, pathlib, logging, functools, webbrowser
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors

import tkinter as tk
from tkinter import ttk, messagebox

# ---------------- Appearance defaults ----------------
mpl.rcParams['figure.dpi'] = 110
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.edgecolor'] = '#222'
mpl.rcParams['axes.labelcolor'] = '#111'
mpl.rcParams['xtick.color'] = '#333'
mpl.rcParams['ytick.color'] = '#333'
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.sans-serif'] = ['Inter', 'Segoe UI', 'DejaVu Sans', 'Arial']

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.FileHandler("rrg.log"), logging.StreamHandler()]
)

# ---------------- Config (YAML) ----------------
CFG_PATH = "rrg_config.yaml"
REQUIRED_TICKERS = [
    "^CNXAUTO","^CNXFMCG","^CNXIT","^NSEBANK","^CNXPHARMA","^CNXENERGY",
    "^CNXREALTY","^CNXINFRA","^CNXPSUBANK","^NSMIDCP","^NSEMDCP50",
    "^CNXCONSUM","^CNXCMDT","^CNXMEDIA","^CNXSERVICE",
    "^CNXMETAL","^CNXPSE","NIFTY_CPSE.NS","NIFTY_FIN_SERVICE.NS","NIFTYMIDCAP150.NS"
]

if not os.path.exists(CFG_PATH):
    DEFAULT_CFG = {
        "ui_default_period": "1Y",
        "interval": "1wk",
        "window": 14,
        "default_tail": 8,
        "default_benchmark": "^CRSLDX",
        "benchmark_choices": ["^CRSLDX", "^CNX200", "^NSEI", "NIFTYMIDCAP150.NS"],
        "tickers": REQUIRED_TICKERS,
        "cache_dir": "cache"
    }
    with open(CFG_PATH, "w") as f:
        yaml.safe_dump(DEFAULT_CFG, f)
    print("Created default rrg_config.yaml")

CFG = yaml.safe_load(open(CFG_PATH, "r"))

# --- Benchmark display <-> symbol mapping ---
BENCHMARK_LABEL_TO_SYMBOL = {
    "NIFTY 50": "^NSEI",
    "NIFTY 200": "^CNX200",
    "NIFTY 500": "^CRSLDX",
    "NIFTY Midcap 150": "NIFTYMIDCAP150.NS",
}
BENCHMARK_SYMBOL_TO_LABEL = {v: k for k, v in BENCHMARK_LABEL_TO_SYMBOL.items()}
BENCHMARK_SYMBOLS = list(BENCHMARK_LABEL_TO_SYMBOL.values())

DEFAULT_BENCHMARK = CFG.get("default_benchmark", "^CRSLDX")
if DEFAULT_BENCHMARK not in BENCHMARK_SYMBOLS:
    DEFAULT_BENCHMARK = "^CRSLDX"
DEFAULT_BENCHMARK_LABEL = BENCHMARK_SYMBOL_TO_LABEL.get(DEFAULT_BENCHMARK, "NIFTY 500")

# Period choices
PERIOD_MAP = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y": "3y"}
period_label = CFG.get("ui_default_period", "1Y")
period = PERIOD_MAP.get(period_label, "1y")

# Timeframe choices (display -> resample mode)
TIMEFRAME_CHOICES = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
_cfg_int = CFG.get("interval", "1wk")
_default_tf_label = next((k for k, v in TIMEFRAME_CHOICES.items() if v == _cfg_int), "Weekly")
interval = TIMEFRAME_CHOICES[_default_tf_label]  # '1d'/'1wk'/'1mo'

WINDOW = int(CFG.get("window", 14))
DEFAULT_TAIL = int(CFG.get("default_tail", 8))

# Merge config tickers with REQUIRED_TICKERS (preserve order, dedupe)
_cfg_tickers = CFG.get("tickers", [])
UNIVERSE = list(dict.fromkeys(_cfg_tickers + REQUIRED_TICKERS))

CACHE_DIR = pathlib.Path(CFG.get("cache_dir", "cache"))
CACHE_DIR.mkdir(exist_ok=True)

STATE_FILE = "rrg_state.json"
NAMES_FILE = CACHE_DIR / "names.json"

# ---------------- Theme ----------------
THEME_CHOICES = ["System", "Light", "Dark"]
current_theme = "System"

def _theme_palette(mode: str):
    if mode not in ("Light", "Dark"):
        mode = "Light"
    if mode == "Light":
        return {
            "bg": "#f5f6f8",
            "panel": "#ffffff",
            "text": "#111111",
            "muted": "#333333",
            "axes": "#ffffff",
            "axes_edge": "#222222",
            "grid": "#777777",
            "figure": "#f5f6f8",
            "rank_text": "#222222",
            "quad_alpha": 0.22,
            "quad_edges": "#777777",
            "scatter_edge": "#333333",
        }
    else:
        return {
            "bg": "#0f1115",
            "panel": "#151821",
            "text": "#e8eaed",
            "muted": "#c2c3c7",
            "axes": "#121420",
            "axes_edge": "#565b66",
            "grid": "#6b7280",
            "figure": "#0f1115",
            "rank_text": "#e8eaed",
            "quad_alpha": 0.20,
            "quad_edges": "#6b7280",
            "scatter_edge": "#8b8f99",
        }

# ---------------- Globals ----------------
is_playing = False

tickers: List[str] = []
tickers_data: Dict[str, pd.Series] = {}
benchmark_data: pd.Series = pd.Series(dtype=float)
benchmark_used: str = ""

rs_ratio_map: Dict[str, pd.Series] = {}
rs_mom_map: Dict[str, pd.Series] = {}

idx: pd.DatetimeIndex = pd.DatetimeIndex([])
idx_len: int = 0

SYMBOL_COLORS: Dict[str, str] = {}

# Hover helpers
hover_cursor = None
artist_to_symbol: Dict[plt.Collection, str] = {}

# cache for fast price lookups during animation
px_cache: Dict[str, pd.Series] = {}

# ---------------- Utilities ----------------
IST = "Asia/Kolkata"

def set_index_to_1700(dt_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx_local = pd.to_datetime(dt_index)
    try:
        if idx_local.tz is not None:
            idx_local = idx_local.tz_convert(IST).tz_localize(None)
    except Exception:
        pass
    return pd.DatetimeIndex(idx_local.normalize() + pd.Timedelta(hours=17))

def to_1700_ist_series(s: pd.Series) -> pd.Series:
    if s.empty: return s
    s = s.copy()
    s.index = set_index_to_1700(s.index)
    return s

def resample_series_to_mode(s: pd.Series, mode: str) -> pd.Series:
    if s.empty:
        return s
    daily = to_1700_ist_series(s)
    if mode == "1d":
        return daily
    idx_dates = pd.DatetimeIndex(daily.index.date)
    daily = pd.Series(daily.values, index=idx_dates, name=daily.name)
    if mode == "1wk":
        w = daily.resample("W-FRI").last().dropna()
        w.index = pd.DatetimeIndex(w.index) + pd.Timedelta(hours=17)
        return w
    if mode == "1mo":
        m = daily.resample("ME").last().dropna()
        m.index = pd.DatetimeIndex(m.index) + pd.Timedelta(hours=17)
        return m
    return daily

def _now_ist():
    return pd.Timestamp.now(IST)

def _is_last_bar_complete(index: pd.DatetimeIndex, mode: str) -> bool:
    if len(index) == 0: return False
    last = pd.Timestamp(index[-1]).tz_localize(IST)
    now = _now_ist()
    if mode == "1d":
        return now >= last
    return now >= last + pd.Timedelta(days=1)

def safe_long_name(symbol: str) -> str:
    try:
        info = yf.Ticker(symbol).info or {}
        return info.get("longName") or info.get("shortName") or symbol
    except Exception:
        return symbol

def safe_long_name_cached(sym: str) -> str:
    try:
        if NAMES_FILE.exists():
            cache = json.load(open(NAMES_FILE, "r"))
        else:
            cache = {}
        if sym not in cache:
            cache[sym] = safe_long_name(sym)
            json.dump(cache, open(NAMES_FILE, "w"))
        return cache.get(sym, sym)
    except Exception:
        return sym

def clean_symbol(sym: str) -> str:
    return sym.replace("^", "").replace(".NS", "")

def display_name(sym: str) -> str:
    nm = meta.get(sym, {}).get("name") if isinstance(meta, dict) else None
    return (nm or "").strip() or clean_symbol(sym)

def get_status(x: float, y: float) -> str:
    if x <= 100 and y <= 100: return "Lagging"
    if x >= 100 and y >= 100: return "Leading"
    if x <= 100 and y >= 100: return "Improving"
    if x >= 100 and y <= 100: return "Weakening"
    return "Unknown"

def status_bg_color(x: float, y: float) -> str:
    m = get_status(x, y)
    if m == "Lagging":   return "#e06a6a"
    if m == "Leading":   return "#3fa46a"
    if m == "Improving": return "#5d86d1"
    if m == "Weakening": return "#e2d06b"
    return "#aaaaaa"

def table_text_color(bg: str) -> str:
    return "white" if bg in ("#e06a6a","#3fa46a","#5d86d1") else "black"

def jdk_components(price: pd.Series, bench: pd.Series, win: int = 14) -> Tuple[pd.Series, pd.Series]:
    df = pd.concat([price.rename("p"), bench.rename("b")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    rs = 100 * (df["p"] / df["b"])
    m = rs.rolling(win).mean()
    s = rs.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rs_ratio = (100 + (rs - m) / s).dropna()
    rsr_roc = rs_ratio.pct_change().mul(100)
    m2 = rsr_roc.rolling(win).mean()
    s2 = rsr_roc.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rs_mom = (101 + (rsr_roc - m2) / s2).dropna()
    ix = rs_ratio.index.intersection(rs_mom.index)
    return rs_ratio.loc[ix], rs_mom.loc[ix]

# ---------------- Cache + Retry ----------------
def cache_path(symbol: str) -> pathlib.Path:
    safe = symbol.replace("^","").replace(".","_")
    return CACHE_DIR / f"{safe}_{period}_{interval}.parquet"

def save_series_cache(symbol: str, s: pd.Series):
    try:
        s.to_frame("Close").to_parquet(cache_path(symbol))
    except Exception as e:
        logging.warning(f"Cache save failed for {symbol}: {e}")

def load_series_cache(symbol: str) -> pd.Series:
    p = cache_path(symbol)
    if p.exists():
        try:
            df = pd.read_parquet(p)
            return df["Close"].dropna()
        except Exception as e:
            logging.warning(f"Cache read failed for {symbol}: {e}")
    return pd.Series(dtype=float)

def retry(n=4, delay=1.5, backoff=2.0):
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            d = delay
            for i in range(n):
                try:
                    return fn(*a, **kw)
                except Exception as e:
                    if i == n - 1:
                        raise
                    logging.warning(f"{fn.__name__} failed ({i+1}/{n}): {e}. Retry in {d:.1f}s")
                    time.sleep(d); d *= backoff
        return wrapper
    return deco

# ---------------- Data download & resample ----------------
def pick_close(raw, symbol: str) -> pd.Series:
    if isinstance(raw, pd.Series):
        return raw.dropna()
    if isinstance(raw, pd.DataFrame):
        if isinstance(raw.columns, pd.MultiIndex):
            for lvl in ("Close", "Adj Close"):
                if (symbol, lvl) in raw.columns:
                    return raw[(symbol, lvl)].dropna()
        else:
            for col in ("Close", "Adj Close"):
                if col in raw.columns:
                    return raw[col].dropna()
    return pd.Series(dtype=float)

@retry()
def download_block_daily_then_resample(tickers: List[str], benchmark: str, period: str, mode: str, exclude_partial=True):
    symbols = list(dict.fromkeys(list(tickers) + [benchmark]))
    raw = yf.download(symbols, period=period, interval="1d",
                      group_by="ticker", auto_adjust=True, progress=False, threads=True)

    def _pipe(sym):
        s = pick_close(raw, sym)
        if s.empty:
            cached = load_series_cache(sym)
            return cached if not cached.empty else pd.Series(dtype=float)
        s = resample_series_to_mode(s, mode).dropna()
        if exclude_partial and len(s) >= 1:
            if not _is_last_bar_complete(s.index, mode):
                s = s.iloc[:-1]
        return s

    bench = _pipe(benchmark)
    data: Dict[str, pd.Series] = {}
    for t in tickers:
        ss = _pipe(t)
        if not ss.empty:
            data[t] = ss

    if not bench.empty:
        save_series_cache(benchmark, bench)
    for t, s in data.items():
        save_series_cache(t, s)

    return bench, data

# ---------------- Colors ----------------
tab20 = plt.get_cmap('tab20').colors
def symbol_color_map(symbols: List[str]) -> Dict[str, str]:
    return {s: to_hex(tab20[i % len(tab20)], keep_alpha=False) for i, s in enumerate(symbols)}

# ---------------- Build data ----------------
def required_overlap_by_mode(mode: str) -> int:
    if mode == "1mo":
        return max(WINDOW // 2 + 6, 12)
    return max(WINDOW + 5, 20)

def build_rrg_with_benchmark_order(bench_order: List[str]):
    global tickers, tickers_data, benchmark_data, benchmark_used
    global rs_ratio_map, rs_mom_map, idx, idx_len, SYMBOL_COLORS, px_cache

    last_err = None
    tickers_data.clear()
    benchmark_used = None

    for b in bench_order:
        try:
            bench, data = download_block_daily_then_resample(
                UNIVERSE, b, period, interval, exclude_partial=True
            )
            if bench is None or bench.empty:
                raise RuntimeError(f"No data for {b}")
            benchmark_data = bench
            tickers_data = data
            benchmark_used = b
            logging.info(f"Using benchmark: {b} @ timeframe {interval} (resampled from daily)")
            break
        except Exception as e:
            last_err = e
            logging.warning(f"Benchmark {b} failed: {e}")
            continue

    if benchmark_used is None:
        raise RuntimeError(f"No benchmark returned data. Last error: {last_err}")

    rs_ratio_map, rs_mom_map = {}, {}
    for t, s in tickers_data.items():
        if t == benchmark_used: continue
        rr, mm = jdk_components(s, benchmark_data, WINDOW)
        if len(rr) and len(mm) and not rr.isna().all() and not mm.isna().all():
            rs_ratio_map[t] = rr
            rs_mom_map[t] = mm

    if not rs_ratio_map:
        raise RuntimeError("No valid tickers after RS calc.")

    common = None
    for t in rs_ratio_map:
        c = rs_ratio_map[t].index.intersection(rs_mom_map[t].index)
        common = c if common is None else common.intersection(c)

    min_overlap = required_overlap_by_mode(interval)
    if common is None or len(common) < min_overlap:
        raise RuntimeError("Not enough overlapping data to draw RRG.")

    for t in list(rs_ratio_map.keys()):
        rs_ratio_map[t] = rs_ratio_map[t].reindex(common)
        rs_mom_map[t]   = rs_mom_map[t].reindex(common)

    tickers = list(rs_ratio_map.keys())
    SYMBOL_COLORS = symbol_color_map(tickers)
    idx = common
    idx_len = len(idx)

    px_cache = {t: tickers_data[t].reindex(idx).dropna() for t in tickers if t in tickers_data}

# ---------------- TradingView Symbol Mapping ----------------
TRADINGVIEW_MAP = {
    "^NSEI": "NIFTY",
    "^NSEBANK": "BANKNIFTY",
    "^CNXIT": "CNXIT",
    "^CNXFMCG": "CNXFMCG",
    "^CNXPHARMA": "CNXPHARMA",
    "^CNXENERGY": "CNXENERGY",
    "^CNXAUTO": "CNXAUTO",
    "^CNXREALTY": "CNXREALTY",
    "^CNXINFRA": "CNXINFRA",
    "^CNXPSE": "CNXPSE",
    "^CNXPSUBANK": "CNXPSUBANK",
    "^CNXCMDT": "CNXCOMMODITIES",
    "^CNXCONSUM": "CNXCONSUMPTION",
    "^CNXMEDIA": "CNXMEDIA",
    "^CNXSERVICE": "CNXSERVICE",
    "^CNXMETAL": "CNXMETAL",
    "^NSEMDCP50": "NIFTYMIDCAP50",
    "^NSMIDCP": "NIFTYJR",
    # .NS index variants
    "NIFTYMIDCAP150.NS": "NIFTYMIDCAP150",
    "NIFTY_CPSE.NS": "CPSE",
    "NIFTY_FIN_SERVICE.NS": "CNXFINANCE",
}

INDEX_PREFIXES = ("CNX", "NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "MIDCAP")

def tv_symbol_and_is_index(sym: str):
    if sym in TRADINGVIEW_MAP:
        return TRADINGVIEW_MAP[sym], True
    if sym.startswith("^"):
        return sym.replace("^", ""), True
    if sym.endswith(".NS"):
        base = sym[:-3]
        if base in ("NIFTYMIDCAP150","CPSE","CNXFINANCE"):
            return base, True
        return base, False
    if sym.startswith(INDEX_PREFIXES):
        return sym, True
    return sym.replace("^",""), sym.startswith(INDEX_PREFIXES)

def tv_build_link(sym: str):
    tv_sym, is_index = tv_symbol_and_is_index(sym)
    if is_index:
        return tv_sym, f"https://www.tradingview.com/chart/?symbol={tv_sym}"
    else:
        return f"NSE:{tv_sym}", f"https://www.tradingview.com/chart/?symbol=NSE:{tv_sym}"

# ---------------- Simple Tooltip ----------------
class Tooltip:
    def __init__(self, widget, text="", delay_ms=250):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._tip = None
        self._after = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)

    def _schedule(self, _event=None):
        self._cancel()
        self._after = self.widget.after(self.delay_ms, self._show)

    def _show(self):
        if self._tip or not self.text:
            return
        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.attributes("-topmost", True)
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip.wm_geometry(f"+{x}+{y}")
        pal = _theme_palette(current_theme)
        bg = pal.get("panel", "#ffffe0")
        fg = pal.get("text", "#111")
        frm = tk.Frame(self._tip, bg=bg, bd=1, relief="solid")
        frm.pack()
        tk.Label(frm, text=self.text, bg=bg, fg=fg, padx=6, pady=3, justify="left").pack()

    def _hide(self, _event=None):
        self._cancel()
        if self._tip:
            try: self._tip.destroy()
            except Exception: pass
            self._tip = None

    def _cancel(self):
        if self._after:
            try: self.widget.after_cancel(self._after)
            except Exception: pass
            self._after = None

    def set_text(self, text):
        self.text = text

# ---------------- UI ----------------
root = tk.Tk()
root.title("RRG Indicator — Stallions")
root.geometry("1500x900")
root.resizable(True, True)

style = ttk.Style(root)
try: style.theme_use('clam')
except Exception: pass

paned = ttk.Panedwindow(root, orient="horizontal")
left = ttk.Frame(paned, width=340)
right = ttk.Frame(paned)
paned.add(left, weight=0)
paned.add(right, weight=1)
paned.pack(fill="both", expand=True)

left_top = ttk.Frame(left, padding=8)
left_top.pack(side="top", fill="x")

ttk.Label(left_top, text="RRG — Controls", font=("Segoe UI", 13, "bold")).pack(anchor="w", pady=(0, 8))

# Benchmark dropdown (friendly labels)
bench_var = tk.StringVar(value=DEFAULT_BENCHMARK_LABEL)
ttk.Label(left_top, text="Benchmark", font=("Segoe UI", 10, "bold")).pack(anchor="w")
bench_menu = ttk.Combobox(
    left_top,
    textvariable=bench_var,
    values=list(BENCHMARK_LABEL_TO_SYMBOL.keys()),
    state="readonly",
    width=18
)
bench_menu.pack(anchor="w", pady=(0,8))

# Timeframe dropdown
ttk.Label(left_top, text="Timeframe", font=("Segoe UI", 10, "bold")).pack(anchor="w")
timeframe_var = tk.StringVar(value=_default_tf_label)
tf_menu = ttk.Combobox(left_top, textvariable=timeframe_var,
                       values=list(TIMEFRAME_CHOICES.keys()),
                       state="readonly", width=18)
tf_menu.pack(anchor="w", pady=(0,8))

PERIOD_CHOICES = list(PERIOD_MAP.keys())
period_var = tk.StringVar(value=period_label if period_label in PERIOD_CHOICES else "1Y")
ttk.Label(left_top, text="Period", font=("Segoe UI", 10, "bold")).pack(anchor="w")
period_menu = ttk.Combobox(left_top, textvariable=period_var, values=PERIOD_CHOICES, state="readonly", width=8)
period_menu.pack(anchor="w", pady=(0,8))

RANK_MODES = ["RRG Power (dist)", "RS-Ratio", "RS-Momentum", "Price %Δ (tail)", "Momentum Slope (tail)"]
rank_mode_var = tk.StringVar(value=RANK_MODES[0])
ttk.Label(left_top, text="Rank by", font=("Segoe UI", 10, "bold")).pack(anchor="w")
rank_menu = ttk.Combobox(left_top, textvariable=rank_mode_var, values=RANK_MODES, state="readonly", width=22)
rank_menu.pack(anchor="w", pady=(0,8))

# Theme switcher
ttk.Label(left_top, text="Theme", font=("Segoe UI", 10, "bold")).pack(anchor="w")
theme_var = tk.StringVar(value=current_theme)
theme_menu = ttk.Combobox(left_top, textvariable=theme_var, values=["Light","Dark"], state="readonly", width=18)
theme_menu.pack(anchor="w", pady=(0,8))

ttk.Label(left_top, text="Trail Length", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(10,0))
tail_scale = ttk.Scale(left_top, from_=1, to=20, value=DEFAULT_TAIL, orient="horizontal")
tail_scale.pack(fill="x", pady=(0,8))

right_top = ttk.Frame(right)
right_top.pack(side="top", fill="both", expand=True)
right_bottom = ttk.Frame(right)
right_bottom.pack(side="bottom", fill="both", expand=False)

fig, ax = plt.subplots(1, 1)
canvas = FigureCanvasTkAgg(fig, master=right_top)
canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

ax_rank = fig.add_axes([0.84, 0.12, 0.14, 0.76])
ax_rank.set_title("Ranking", fontsize=11, pad=6)
ax_rank.axis("off")
rank_text_objects = []

# --- Date slider FIRST (top) ---
date_slider_frame = ttk.Frame(right_bottom, padding=(8,2))
date_slider_frame.pack(side="top", fill="x")
date_label = ttk.Label(date_slider_frame, text="Date: -")
date_label.pack(side="right")
date_scale = ttk.Scale(date_slider_frame, from_=0, to=1, value=1, orient="horizontal")
date_scale.pack(side="left", fill="x", expand=True, padx=(0,8))

# -------- Scrollable Table AFTER slider --------
_table_wrap = ttk.Frame(master=right_bottom)
_row_frame = ttk.Frame(_table_wrap)
_table_wrap.pack(side="top", fill=tk.BOTH, expand=1)

_table_canvas = tk.Canvas(_table_wrap, highlightthickness=0)
_table_vsb = ttk.Scrollbar(_table_wrap, orient="vertical", command=_table_canvas.yview)
_table_canvas.configure(yscrollcommand=_table_vsb.set)
_table_canvas.pack(side="left", fill=tk.BOTH, expand=1)
_table_vsb.pack(side="right", fill="y")

table = tk.Frame(master=_table_canvas)
table_id = _table_canvas.create_window((0,0), window=table, anchor="nw")

def _on_table_configure(event=None):
    _table_canvas.configure(scrollregion=_table_canvas.bbox("all"))
def _on_table_width(event):
    _table_canvas.itemconfigure(table_id, width=event.width)

table.bind("<Configure>", _on_table_configure)
_table_canvas.bind("<Configure>", _on_table_width)

HEADERS = ["SL No.", " ", "Name", "Status", "Price", "Change %", "Visible"]
for j, h in enumerate(HEADERS):
    lbl = tk.Label(table, text=h, relief=tk.RIDGE, font=("Segoe UI", 12, "bold"))
    lbl.grid(row=0, column=j, sticky="nsew", padx=1, pady=1)

for c in range(len(HEADERS)):
    if c in (0, 1, 6):
        table.grid_columnconfigure(c, weight=1, uniform="cols")
    else:
        table.grid_columnconfigure(c, weight=3, uniform="cols")

# ---------------- Theme application ----------------
def apply_theme(mode: str):
    global current_theme
    current_theme = mode
    pal = _theme_palette("Light" if mode=="Light" else "Dark")

    mpl.rcParams['axes.edgecolor'] = pal["axes_edge"]
    mpl.rcParams['axes.labelcolor'] = pal["text"]
    mpl.rcParams['xtick.color'] = pal["muted"]
    mpl.rcParams['ytick.color'] = pal["muted"]
    mpl.rcParams['text.color'] = pal["text"]
    mpl.rcParams['figure.facecolor'] = pal["figure"]
    mpl.rcParams['axes.facecolor'] = pal["axes"]

    fig.set_facecolor(pal["figure"])
    ax.set_facecolor(pal["axes"])

    try:
        root.configure(bg=pal["bg"])
        left.configure(style="Side.TFrame")
        right.configure(style="Side.TFrame")
        style.configure("Side.TFrame", background=pal["bg"])
        style.configure("TLabel", background=pal["bg"], foreground=pal["text"])
    except Exception:
        pass

    init_plot_axes()
    canvas.draw_idle()

def rrg_limits(pad=1.5):
    if not rs_ratio_map or not rs_mom_map:
        return (96-pad, 104+pad), (96-pad, 104+pad)
    xs = np.concatenate([v.values for v in rs_ratio_map.values()]) if rs_ratio_map else np.array([100.])
    ys = np.concatenate([v.values for v in rs_mom_map.values()])   if rs_mom_map else np.array([100.])
    x0, x1 = np.nanmin(xs), np.nanmax(xs); y0, y1 = np.nanmin(ys), np.nanmax(ys)
    x0, x1 = min(x0, 96), max(x1, 104); y0, y1 = min(y0, 96), max(y1, 104)
    return (x0-pad, x1+pad), (y0-pad, y1+pad)

def init_plot_axes():
    pal = _theme_palette(current_theme)
    ax.clear()
    ax.set_facecolor(pal["axes"])
    ax.set_title("RRG Indicator", fontsize=13, pad=10, color=pal["text"])
    ax.set_xlabel("JdK RS-Ratio", color=pal["text"])
    ax.set_ylabel("JdK RS-Momentum", color=pal["text"])

    ax.axhline(y=100, color=pal["quad_edges"], linestyle=":", linewidth=1.0)
    ax.axvline(x=100, color=pal["quad_edges"], linestyle=":", linewidth=1.0)

    a = _theme_palette(current_theme)["quad_alpha"] * 0.9
    ax.fill_between([92, 100], [92, 92],  [100, 100], color=(1.0, 0.0, 0.0, a))  # Lagging
    ax.fill_between([100, 108],[92, 92],  [100, 100], color=(1.0, 1.0, 0.0, a))  # Weakening
    ax.fill_between([100, 108],[100, 100],[108, 108], color=(0.0, 1.0, 0.0, a))  # Leading
    ax.fill_between([92, 100], [100, 100],[108, 108], color=(0.0, 0.0, 1.0, a))  # Improving

    ax.text(95, 105, "Improving", fontsize=11, color=pal["text"], weight="bold")
    ax.text(104, 105, "Leading",   fontsize=11, color=pal["text"], weight="bold", ha="right")
    ax.text(104, 95,  "Weakening", fontsize=11, color=pal["text"], weight="bold", ha="right")
    ax.text(95, 95,   "Lagging",   fontsize=11, color=pal["text"], weight="bold")

    (xl, xr), (yl, yr) = rrg_limits()
    ax.set_xlim(xl, xr); ax.set_ylim(yl, yr)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.35)

    ax_rank.set_facecolor(pal["axes"])
    ax_rank.set_title("Ranking", fontsize=11, pad=6, color=pal["text"])

meta = {}
tickers_to_show: List[str] = []
row_widgets = {}
row_order: List[str] = []

scatter_plots: Dict[str, plt.Collection] = {}
line_plots = {}
annotations = {}

# -------- Hover reattachment helper --------
def _refresh_hover_cursor():
    global hover_cursor
    try:
        artists = list(scatter_plots.values())
        if not artists:
            return
        hover_cursor = mplcursors.cursor(artists, hover=True)

        @_attach(hover_cursor, "add")
        def _on_add(sel):
            if idx_len == 0:
                return
            sc = sel.artist
            sym = artist_to_symbol.get(sc)
            if not sym:
                return
            point_idx = int(sel.index) if hasattr(sel, "index") and sel.index is not None else None
            if point_idx is None:
                return
            start_pos, end_pos = current_window()
            abs_idx = start_pos + 1 + point_idx
            if abs_idx < 0 or abs_idx >= idx_len:
                return

            rr_val = float(rs_ratio_map[sym].iloc[abs_idx])
            mm_val = float(rs_mom_map[sym].iloc[abs_idx])

            px = px_cache.get(sym, pd.Series(dtype=float))
            if start_pos < len(px) and abs_idx < len(px):
                pct = (px.iloc[abs_idx] / px.iloc[start_pos] - 1.0) * 100.0
            else:
                pct = float("nan")

            status = get_status(rr_val, mm_val)
            name = display_name(sym)
            pct_txt = "-" if np.isnan(pct) else f"{pct:+.2f}%"
            dstr = str(idx[abs_idx].date())
            bm  = BENCHMARK_SYMBOL_TO_LABEL.get(benchmark_used, benchmark_used)

            sel.annotation.set(
                text=(f"{name}\n[{status}]  {dstr}\n"
                      f"RS-Ratio: {rr_val:.2f}  RS-Mom: {mm_val:.2f}\n"
                      f"Price Δ (tail): {pct_txt}  vs {bm}"),
                bbox=dict(boxstyle="round", fc="#ffffff", ec=SYMBOL_COLORS.get(sym, "#333")),
                color="#111"
            )
    except Exception:
        pass

def _attach(cursor_obj, event_name):
    def deco(fn):
        cursor_obj.connect(event_name, fn)
        return fn
    return deco

def rebuild_row(symbol: str, row: int):
    name = meta[symbol]["name"]
    if idx_len == 0:
        price = np.nan; chg = np.nan; status = "-"
    else:
        end_pos = int(round(date_scale.get()))
        start_pos = max(end_pos - int(round(tail_scale.get())), 0)
        px_series = px_cache.get(symbol, pd.Series(dtype=float))
        price = float(px_series.iloc[end_pos]) if end_pos < len(px_series) else np.nan
        chg = (px_series.iloc[end_pos] / px_series.iloc[start_pos] - 1) * 100 if (start_pos < len(px_series) and end_pos < len(px_series)) else np.nan
        rr_val = rs_ratio_map[symbol].iloc[-1]; mm_val = rs_mom_map[symbol].iloc[-1]
        status = get_status(rr_val, mm_val)

    rr_val = rs_ratio_map[symbol].iloc[-1]; mm_val = rs_mom_map[symbol].iloc[-1]
    bg = status_bg_color(rr_val, mm_val); fg = table_text_color(bg)

    lbl_rank = tk.Label(table, text="", relief=tk.RIDGE)
    lbl_rank.grid(row=row, column=0, sticky="nsew", padx=1, pady=1)

    swatch = tk.Label(table, text="", bg=SYMBOL_COLORS[symbol], relief=tk.RIDGE)
    swatch.grid(row=row, column=1, sticky="nsew", padx=1, pady=1)

    # Name hyperlink (symbol hidden)
    def open_tv(event, sym=symbol):
        tv_disp, url = tv_build_link(sym)
        webbrowser.open(url)

    tv_disp, _url_preview = tv_build_link(symbol)

    lbl_name = tk.Label(
        table,
        text=display_name(symbol),
        fg="blue",
        cursor="hand2",
        relief=tk.RIDGE,
        font=("Segoe UI", 11, "underline"),
        bg=bg
    )
    lbl_name.grid(row=row, column=2, sticky="nsew", padx=1, pady=1)
    lbl_name.bind("<Button-1>", open_tv)
    Tooltip(lbl_name, text=f"Open in TradingView: {tv_disp}", delay_ms=250)

    lbl_stat = tk.Label(table, text=status, relief=tk.RIDGE, bg=bg, fg=fg, font=("Segoe UI", 11))
    lbl_stat.grid(row=row, column=3, sticky="nsew", padx=1, pady=1)

    lbl_px = tk.Label(table, text=("-" if pd.isna(price) else f"{price:.2f}"), relief=tk.RIDGE, bg=bg, fg=fg, font=("Segoe UI", 11))
    lbl_px.grid(row=row, column=4, sticky="nsew", padx=1, pady=1)

    lbl_chg = tk.Label(table, text=("-" if pd.isna(chg) else f"{chg:.2f}"), relief=tk.RIDGE, bg=bg, fg=fg, font=("Segoe UI", 11))
    lbl_chg.grid(row=row, column=5, sticky="nsew", padx=1, pady=1)

    var_vis = tk.BooleanVar(value=True)
    chk = ttk.Checkbutton(table, variable=var_vis)
    chk.grid(row=row, column=6, sticky="nsew", padx=1, pady=1); chk.state(["selected"])

    def on_checkbox_click(_):
        if var_vis.get():
            if symbol not in tickers_to_show: tickers_to_show.append(symbol)
        else:
            if symbol in tickers_to_show: tickers_to_show.remove(symbol)
    chk.bind("<ButtonRelease-1>", on_checkbox_click)

    row_widgets[symbol] = {
        "rank": lbl_rank, "status": lbl_stat, "price": lbl_px, "chg": lbl_chg,
        "vis_var": var_vis
    }

def swap_symbol(row: int, old: str, new: str):
    global tickers, tickers_data, rs_ratio_map, rs_mom_map, SYMBOL_COLORS, meta, idx, px_cache
    try:
        raw = yf.download(new, period=period, interval="1d", auto_adjust=True, progress=False)
        s = pick_close(raw, new)
        if s.empty: raise ValueError(f"No data for {new}")
        s = resample_series_to_mode(s, interval).dropna()
        if len(s) >= 1 and not _is_last_bar_complete(s.index, interval):
            s = s.iloc[:-1]
        if s.empty: raise ValueError(f"No resampled data for {new}")
        rr, mm = jdk_components(s, benchmark_data, WINDOW)
        if rr.empty or mm.empty: raise ValueError(f"Insufficient overlap for {new}")
        common = rr.index.intersection(mm.index).intersection(idx)
        rr = rr.reindex(common); mm = mm.reindex(common)
        if rr.isna().all() or mm.isna().all(): raise ValueError(f"No common index for {new}")

        tickers_data[new] = s; meta[new] = {"symbol": new, "name": safe_long_name_cached(new)}
        rs_ratio_map[new] = rr; rs_mom_map[new] = mm
        px_cache[new] = s.reindex(idx).dropna()
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

        sc = ax.scatter([], [], s=10, linewidths=0.6, color=SYMBOL_COLORS[new])
        ln = ax.plot([], [], linewidth=1.1, alpha=0.6, color=SYMBOL_COLORS[new])[0]
        an = ax.annotate(new, (0, 0), fontsize=8, color=SYMBOL_COLORS[new])

        scatter_plots[new] = sc
        line_plots[new]    = ln
        annotations[new]   = an
        artist_to_symbol[sc] = new

        _refresh_hover_cursor()
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
    mode = rank_mode_var.get()
    if mode == "RRG Power (dist)":
        wq = 1.2 if (rr_last>=100 and mm_last>=100) else (0.9 if (rr_last>=100 and mm_last<100) else 1.0)
        return float(np.hypot(rr_last-100.0, mm_last-100.0) * wq)
    if mode == "RS-Ratio":         return float(rr_last)
    if mode == "RS-Momentum":      return float(mm_last)
    if mode == "Price %Δ (tail)":
        px = px_cache.get(t, pd.Series(dtype=float))
        if len(px.iloc[start_pos:end_pos+1]) >= 2:
            return float((px.iloc[end_pos] / px.iloc[start_pos] - 1) * 100.0)
        return float("-inf")
    if mode == "Momentum Slope (tail)":
        series = rs_mom_map[t].iloc[start_pos:end_pos+1]
        if len(series) >= 2:
            x = np.arange(len(series)); A = np.vstack([x, np.ones(len(x))]).T
            slope = np.linalg.lstsq(A, series.values, rcond=None)[0][0]
            return float(slope)
        return float("-inf")
    return float("-inf")

def jitter_labels(min_dx=0.4, min_dy=0.4, step=0.25):
    texts = list(annotations.values())
    for i in range(len(texts)):
        xi, yi = texts[i].get_position()
        for j in range(i+1, len(texts)):
            xj, yj = texts[j].get_position()
            if abs(xi-xj) < min_dx and abs(yi-yj) < min_dy:
                texts[j].set_position((xj + step, yj - step))

def animate(_):
    global rank_text_objects, is_playing
    if idx_len == 0:
        canvas.draw_idle()
        return []

    if is_playing:
        nxt = min(int(round(date_scale.get())) + 1, idx_len - 1)
        date_scale.set(nxt)

    start_pos, end_pos = current_window()
    date_label.config(text=f"Date: {str(idx[end_pos].date())}")

    for t in row_order:
        if t not in rs_ratio_map or t not in rs_mom_map: continue
        visible = t in tickers_to_show
        rr = rs_ratio_map[t].iloc[start_pos+1 : end_pos+1]
        mm = rs_mom_map[t].iloc[start_pos+1 : end_pos+1]

        if not visible or len(rr)==0 or len(mm)==0:
            scatter_plots[t].set_offsets(np.empty((0, 2)))
            line_plots[t].set_data([], [])
            annotations[t].set_text("")
        else:
            col = SYMBOL_COLORS[t]
            pts = np.column_stack([rr.values, mm.values])
            n = len(rr)
            sizes  = np.linspace(12, 70, n)
            alphas = np.linspace(0.40, 0.95, n)
            scatter_plots[t].set_offsets(pts)
            scatter_plots[t].set_sizes(sizes)
            scatter_plots[t].set_facecolor([mpl.colors.to_rgba(col, a) for a in alphas])
            scatter_plots[t].set_edgecolor(_theme_palette(current_theme)["scatter_edge"])
            line_plots[t].set_data(rr.values, mm.values)
            line_plots[t].set_color(col)
            s_txt = get_status(rr.values[-1], mm.values[-1])
            annotations[t].set_text(f"{t}  [{s_txt}]")
            annotations[t].set_position((rr.values[-1], mm.values[-1]))
            annotations[t].set_fontsize(9)

        rw = row_widgets.get(t)
        if rw:
            px = px_cache.get(t, pd.Series(dtype=float))
            price = float(px.iloc[end_pos]) if end_pos < len(px) else np.nan
            chg = ((px.iloc[end_pos]/px.iloc[start_pos]-1)*100.0) if (end_pos < len(px) and start_pos < len(px)) else np.nan
            rr_last = float(rs_ratio_map[t].iloc[end_pos]); mm_last = float(rs_mom_map[t].iloc[end_pos])
            bg = status_bg_color(rr_last, mm_last); fg = table_text_color(bg)
            try:
                rw["status"].config(text=get_status(rr_last, mm_last), bg=bg, fg=fg)
                rw["price"].config(text=("-" if pd.isna(price) else f"{price:.2f}"), bg=bg, fg=fg)
                rw["chg"].config(text=("-" if pd.isna(chg) else f"{chg:.2f}"), bg=bg, fg=fg)
            except Exception:
                pass

    # stop auto-play at end
    if is_playing and int(round(date_scale.get())) >= idx_len - 1:
        is_playing = False

    performances = []
    for t in tickers:
        if t not in tickers_to_show: continue
        try:
            metric = compute_rank_metric(t, start_pos, end_pos)
            performances.append((t, metric))
        except Exception:
            pass
    performances.sort(key=lambda x: x[1], reverse=True)

    for txt in rank_text_objects:
        try: txt.remove()
        except Exception: pass
    rank_text_objects = []
    max_rows = 22; spacing = 0.042

    # Build rank map for SL No. column in table
    rank_map = {}
    for r_i, (symb, _metric) in enumerate(performances, start=1):
        rank_map[symb] = r_i

    # Update SL No. in table rows
    for symb, widgets in row_widgets.items():
        try:
            sl = rank_map.get(symb, "")
            widgets["rank"].config(text=str(sl))
        except Exception:
            pass

    # Draw ranking panel text
    for rank, (symbol, metric) in enumerate(performances[:max_rows], start=1):
        rr_last = float(rs_ratio_map[symbol].iloc[end_pos])
        mm_last = float(rs_mom_map[symbol].iloc[end_pos])
        stat = get_status(rr_last, mm_last)
        name = display_name(symbol)
        y = 1 - rank * spacing
        txt = ax_rank.text(
            0.0, y,
            f"{rank}. {name} [{stat}]",
            fontsize=10,
            color=_theme_palette(current_theme)["rank_text"],
            transform=ax_rank.transAxes,
            picker=True
        )
        rank_text_objects.append(txt)

    (xl, xr), (yl, yr) = rrg_limits()
    ax.set_xlim(xl, xr); ax.set_ylim(yl, yr)
    jitter_labels()

    canvas.draw_idle()
    return list(scatter_plots.values()) + list(line_plots.values()) + list(annotations.values()) + rank_text_objects

anim = animation.FuncAnimation(fig, animate, interval=170, blit=False, cache_frame_data=False)

def export_ranks_to_csv():
    if idx_len == 0:
        messagebox.showwarning("Export", "No data to export."); return
    end_pos = int(round(date_scale.get()))
    start_pos = max(end_pos - int(round(tail_scale.get())), 0)
    rows = []
    for t in tickers:
        if t not in tickers_to_show: continue
        rr = float(rs_ratio_map[t].iloc[end_pos]); mm = float(rs_mom_map[t].iloc[end_pos])
        metric = compute_rank_metric(t, start_pos, end_pos)
        rows.append((t, metric, rr, mm, get_status(rr, mm)))
    rows.sort(key=lambda x: x[1], reverse=True)
    df = pd.DataFrame(rows, columns=["symbol","rank_metric","rs_ratio","rs_momentum","status"])
    out = f"ranks_{idx[end_pos].date()}.csv"
    df.to_csv(out, index=False)
    logging.info(f"[Saved] {out}")
    messagebox.showinfo("Export", f"Saved {out}")

def export_table_to_csv():
    if idx_len == 0:
        messagebox.showwarning("Export", "No data to export."); return
    end_pos = int(round(date_scale.get()))
    start_pos = max(end_pos - int(round(tail_scale.get())), 0)
    rows = []
    for t in tickers:
        if t not in tickers_to_show: continue
        rr = float(rs_ratio_map[t].iloc[end_pos]); mm = float(rs_mom_map[t].iloc[end_pos])
        px = px_cache.get(t, pd.Series(dtype=float))
        price = float(px.iloc[end_pos]) if end_pos < len(px) else np.nan
        chg = ((px.iloc[end_pos]/px.iloc[start_pos]-1)*100.0) if (end_pos < len(px) and start_pos < len(px)) else np.nan
        rows.append((t, display_name(t), get_status(rr, mm), rr, mm, price, chg))
    df = pd.DataFrame(rows, columns=["symbol","name","status","rs_ratio","rs_momentum","price","pct_change_tail"])
    out = f"table_{idx[end_pos].date()}.csv"
    df.to_csv(out, index=False)
    logging.info(f"[Saved] {out}")
    messagebox.showinfo("Export", f"Saved {out}")

def export_chart_png():
    if idx_len == 0:
        messagebox.showwarning("Export", "No chart to export."); return
    stamp = str(idx[int(round(date_scale.get()))].date())
    bml   = BENCHMARK_SYMBOL_TO_LABEL.get(benchmark_used, benchmark_used).replace(" ", "")
    out   = f"rrg_{bml}_{stamp}.png"
    fig.savefig(out, bbox_inches="tight", dpi=180)
    logging.info(f"[Saved] {out}")
    messagebox.showinfo("Export", f"Saved {out}")

def toggle_play():
    global is_playing
    is_playing = not is_playing

def on_date_scale(val):
    if idx_len == 0: return
    v = int(round(float(val)))
    v = max(0, min(v, idx_len - 1))
    if float(date_scale.get()) != v:
        date_scale.set(v)
    date_label.config(text=f"Date: {str(idx[v].date())}")

def on_benchmark_change(_e=None):
    try:
        initial_build()
    except Exception as e:
        logging.warning(f"[Benchmark switch] {e}")
        try: messagebox.showwarning("Benchmark", f"Could not load benchmark: {bench_var.get()}\n\n{e}")
        except Exception: pass

def on_period_change(_e=None):
    global period_label, period
    try:
        new_label = period_var.get()
        new_period = PERIOD_MAP.get(new_label, "1y")
        if new_period == period: return
        period_label = new_label; period = new_period
        logging.info(f"[Period] Switching to {new_label} ({new_period})")
        initial_build()
    except Exception as e:
        logging.warning(f"[Period switch] {e}")
        try: messagebox.showwarning("Period", f"Could not reload for period {period_var.get()}.\n\n{e}")
        except Exception: pass

def on_timeframe_change(_e=None):
    global interval
    try:
        sel = timeframe_var.get()
        new_mode = TIMEFRAME_CHOICES.get(sel, "1wk")
        if new_mode == interval: return
        interval = new_mode
        logging.info(f"[Timeframe] Switching to {sel} (mode={interval})")
        initial_build()
    except Exception as e:
        logging.warning(f"[Timeframe switch] {e}")
        try: messagebox.showwarning("Timeframe", f"Could not reload for timeframe {timeframe_var.get()}.\n\n{e}")
        except Exception: pass

def on_theme_change(_e=None):
    apply_theme(theme_var.get())

play_btn = ttk.Button(left_top, text="Play / Pause", command=toggle_play); play_btn.pack(anchor="w", pady=(6,2))
export_ranks_btn = ttk.Button(left_top, text="Export Ranks CSV", command=export_ranks_to_csv); export_ranks_btn.pack(anchor="w", pady=(2,2))
export_table_btn = ttk.Button(left_top, text="Export Table CSV", command=export_table_to_csv); export_table_btn.pack(anchor="w", pady=(2,2))
export_png_btn = ttk.Button(left_top, text="Export Chart PNG", command=export_chart_png); export_png_btn.pack(anchor="w", pady=(2,6))
bench_menu.bind("<<ComboboxSelected>>", on_benchmark_change)
period_menu.bind("<<ComboboxSelected>>", on_period_change)
tf_menu.bind("<<ComboboxSelected>>", on_timeframe_change)
theme_menu.bind("<<ComboboxSelected>>", on_theme_change)
date_scale.configure(command=on_date_scale)

def create_table_and_plots():
    for w in table.grid_slaves():
        if int(w.grid_info().get("row", 0)) == 0:
            continue
        w.destroy()

    scatter_plots.clear(); line_plots.clear(); annotations.clear()
    artist_to_symbol.clear()
    init_plot_axes()

    for t in tickers:
        sc = ax.scatter([], [], s=10, linewidths=0.6, color=SYMBOL_COLORS[t])
        ln = ax.plot([], [], linewidth=1.1, alpha=0.6, color=SYMBOL_COLORS[t])[0]
        an = ax.annotate(t, (0, 0), fontsize=8, color=SYMBOL_COLORS[t])

        scatter_plots[t] = sc
        line_plots[t]    = ln
        annotations[t]   = an
        artist_to_symbol[sc] = t

    _refresh_hover_cursor()

    row_widgets.clear(); row_order[:] = []
    for i, t in enumerate(tickers):
        row_order.append(t)
        rebuild_row(t, i + 1)

def _on_pick_rank(event):
    artist = event.artist
    if artist not in rank_text_objects:
        return
    label = artist.get_text()
    try:
        name_part = label.split(". ",1)[1].rsplit(" [",1)[0]
    except Exception:
        return
    sym_key = next((k for k in tickers if display_name(k) == name_part), None)
    if not sym_key:
        return
    if sym_key in tickers_to_show:
        tickers_to_show.remove(sym_key)
    else:
        tickers_to_show.append(sym_key)
    canvas.draw_idle()

canvas.mpl_connect("pick_event", _on_pick_rank)

def initial_build():
    global idx, idx_len, meta, tickers_to_show, px_cache
    sel_label = bench_var.get()
    sel_symbol = BENCHMARK_LABEL_TO_SYMBOL.get(sel_label, "^CRSLDX")

    bench_order = [sel_symbol] + [s for s in BENCHMARK_SYMBOLS if s != sel_symbol]
    try:
        build_rrg_with_benchmark_order(bench_order)
    except Exception as e:
        logging.warning(f"[Initial build] {e}")
        idx = pd.DatetimeIndex([]); idx_len = 0
        try: messagebox.showwarning("Data", f"Could not load benchmark/index data.\n\n{e}")
        except Exception: pass
        init_plot_axes(); canvas.draw(); return

    date_scale.configure(from_=DEFAULT_TAIL, to=idx_len - 1)
    date_scale.set(idx_len - 1)
    date_label.config(text=f"Date: {str(idx[idx_len - 1].date())}")

    meta = {t: {"symbol": t, "name": safe_long_name_cached(t)} for t in tickers}
    tickers_to_show[:] = tickers[:]

    create_table_and_plots()
    root.title(f"RRG Indicator — {BENCHMARK_SYMBOL_TO_LABEL.get(benchmark_used, benchmark_used)} — {period_var.get()} — {timeframe_var.get()} ")

def save_state_and_quit():
    try:
        sel_label = bench_var.get()
        sel_symbol = BENCHMARK_LABEL_TO_SYMBOL.get(sel_label, "^CRSLDX")
        state = {
            "benchmark": sel_symbol,
            "period": period_var.get(),
            "timeframe": timeframe_var.get(),
            "theme": current_theme,
            "visible": [t for t in tickers if row_widgets.get(t, {}).get("vis_var", tk.BooleanVar(value=True)).get()],
            "tail": int(round(tail_scale.get())),
            "date_index": int(round(date_scale.get())) if idx_len else 0,
            "rank_mode": rank_mode_var.get(),
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)
    except Exception as e:
        logging.warning(f"[State save] {e}")
    root.destroy()

def load_state():
    global period_label, period, interval, current_theme
    if not os.path.exists(STATE_FILE): return
    try:
        s = json.load(open(STATE_FILE, "r"))
        bmk_sym = s.get("benchmark")
        if bmk_sym in BENCHMARK_SYMBOLS:
            bench_var.set(BENCHMARK_SYMBOL_TO_LABEL.get(bmk_sym, "NIFTY 500"))
        if "period" in s and s["period"] in PERIOD_MAP:
            period_label = s["period"]; period = PERIOD_MAP[period_label]; period_var.set(period_label)
        if "rank_mode" in s and s["rank_mode"] in RANK_MODES:
            rank_mode_var.set(s["rank_mode"])
        if "timeframe" in s and s["timeframe"] in TIMEFRAME_CHOICES:
            timeframe_var.set(s["timeframe"])
            interval = TIMEFRAME_CHOICES[s["timeframe"]]
        if "theme" in s and s["theme"] in ["Light","Dark"]:
            theme_var.set(s["theme"])
            current_theme = s["theme"]
    except Exception as e:
        logging.warning(f"[State load] {e}")

root.protocol("WM_DELETE_WINDOW", save_state_and_quit)

# ---------------- Start ----------------
load_state()
apply_theme("Light")
initial_build()
canvas.draw()
root.mainloop()

# app.py — RRG Indicator (Streamlit port of your Tk UI)
# - Uses your GitHub /ticker folder for CSV universes
# - JdK RS-Ratio/Momentum logic preserved
# - TradingView links in Name column
# - Status-colored table rows + color swatch column
# - Scrollable table inside expander
# - Export ranks/table PNG

import os, io, json, time, yaml, pathlib, logging, functools
from typing import Dict, List, Tuple

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# ---------------- Appearance (match your defaults) ----------------
mpl.rcParams['figure.dpi'] = 110
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.edgecolor'] = '#222'
mpl.rcParams['axes.labelcolor'] = '#111'
mpl.rcParams['xtick.color'] = '#333'
mpl.rcParams['ytick.color'] = '#333'
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.sans-serif'] = ['Inter', 'Segoe UI', 'DejaVu Sans', 'Arial']

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

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
        "cache_dir": "cache",
        "github_ticker_base": "https://raw.githubusercontent.com/anki1007/rrg-stocks/main/ticker/",
        "ticker_csvs": [
            "nifty200.csv","nifty500.csv","niftymidcap150.csv",
            "niftymidsmallcap400.csv","niftysmallcap250.csv","niftytotalmarket.csv"
        ]
    }
    with open(CFG_PATH, "w") as f:
        yaml.safe_dump(DEFAULT_CFG, f)

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

# Period & timeframe
PERIOD_MAP = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y": "3y"}
period_label_cfg = CFG.get("ui_default_period", "1Y")
TIMEFRAME_CHOICES = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
interval_cfg = CFG.get("interval", "1wk")
_default_tf_label = next((k for k, v in TIMEFRAME_CHOICES.items() if v == interval_cfg), "Weekly")

WINDOW = int(CFG.get("window", 14))
DEFAULT_TAIL = int(CFG.get("default_tail", 8))

# Universe baseline
_cfg_tickers = CFG.get("tickers", [])
UNIVERSE = list(dict.fromkeys(_cfg_tickers + REQUIRED_TICKERS))

CACHE_DIR = pathlib.Path(CFG.get("cache_dir", "cache"))
CACHE_DIR.mkdir(exist_ok=True)
NAMES_FILE = CACHE_DIR / "names.json"

# ---------------- Theme palettes ----------------
pal_light = {
    "bg": "#f5f6f8","panel": "#ffffff","text": "#111111","muted": "#333333",
    "axes": "#ffffff","axes_edge": "#222222","grid": "#777777","figure": "#f5f6f8",
    "rank_text": "#222222","quad_alpha": 0.22,"quad_edges": "#777777","scatter_edge": "#333333",
}
pal_dark = {
    "bg": "#0f1115","panel": "#151821","text": "#e8eaed","muted": "#c2c3c7",
    "axes": "#121420","axes_edge": "#565b66","grid": "#6b7280","figure": "#0f1115",
    "rank_text": "#e8eaed","quad_alpha": 0.20,"quad_edges": "#6b7280","scatter_edge": "#8b8f99",
}

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
    if s.empty: return s
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
    if mode == "1d": return now >= last
    return now >= last + pd.Timedelta(days=1)

def safe_long_name(symbol: str) -> str:
    try:
        info = yf.Ticker(symbol).info or {}
        return info.get("longName") or info.get("shortName") or symbol
    except Exception:
        return symbol

def safe_long_name_cached(sym: str) -> str:
    try:
        cache = json.load(open(NAMES_FILE, "r")) if NAMES_FILE.exists() else {}
        if sym not in cache:
            cache[sym] = safe_long_name(sym)
            json.dump(cache, open(NAMES_FILE, "w"))
        return cache.get(sym, sym)
    except Exception:
        return sym

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
            return pd.Series(dtype=float)
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
    return bench, data

# ---------------- CSV Universe from GitHub /ticker ----------------
def load_github_csv_universe() -> List[str]:
    base = CFG.get("github_ticker_base","")
    files = CFG.get("ticker_csvs",[])
    extra = []
    for fname in files:
        try:
            url = base + fname
            df = pd.read_csv(url)
            cols = [c for c in df.columns if str(c).lower() in ("symbol","ticker","tickers","symbols")]
            if cols:
                vals = df[cols[0]].astype(str).str.strip().tolist()
                for v in vals:
                    v = v.strip()
                    if v.endswith(".NS") or v.startswith("^"):
                        extra.append(v)
                    else:
                        extra.append(v + ".NS")
        except Exception:
            continue
    return list(dict.fromkeys(extra))

# ---------------- Build data ----------------
def build_rrg(benchmark: str, universe: List[str], period: str, mode: str, window: int):
    bench, data = download_block_daily_then_resample(universe, benchmark, period, mode, True)
    if bench is None or bench.empty:
        raise RuntimeError("No benchmark data")
    rs_ratio_map, rs_mom_map = {}, {}
    for t, s in data.items():
        if t == benchmark: continue
        rr, mm = jdk_components(s, bench, window)
        if len(rr) and len(mm) and not rr.isna().all() and not mm.isna().all():
            ix = rr.index.intersection(mm.index)
            rs_ratio_map[t] = rr.loc[ix]
            rs_mom_map[t]   = mm.loc[ix]
    if not rs_ratio_map:
        raise RuntimeError("No valid tickers after RS calc.")
    common = None
    for t in rs_ratio_map:
        c = rs_ratio_map[t].index.intersection(rs_mom_map[t].index)
        common = c if common is None else common.intersection(c)
    min_overlap = max(window + 5, 20)
    if common is None or len(common) < min_overlap:
        raise RuntimeError("Not enough overlapping data to draw RRG.")
    for t in list(rs_ratio_map.keys()):
        rs_ratio_map[t] = rs_ratio_map[t].reindex(common)
        rs_mom_map[t]   = rs_mom_map[t].reindex(common)
    idx = common
    px_cache = {t: data[t].reindex(idx).dropna() for t in rs_ratio_map if t in data}
    return bench, rs_ratio_map, rs_mom_map, px_cache, idx

def symbol_color_map(symbols: List[str]) -> Dict[str, str]:
    tab20 = plt.get_cmap('tab20').colors
    return {s: to_hex(tab20[i % len(tab20)], keep_alpha=False) for i, s in enumerate(symbols)}

def display_name(sym: str) -> str:
    return safe_long_name_cached(sym)

# ---------------- Ranking metric (same modes) ----------------
def compute_rank_metric(t: str, start_pos: int, end_pos: int,
                        rs_ratio_map=None, rs_mom_map=None, px_cache=None, mode="RRG Power (dist)"):
    rr_last = float(rs_ratio_map[t].iloc[end_pos]); mm_last = float(rs_mom_map[t].iloc[end_pos])
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

# ---------------- Render table (scrollable, links, colors) ----------------
def render_rrg_table(
    tickers, rs_ratio_map, rs_mom_map, px_cache, idx, start_pos, sel_pos, COLORS,
    tv_build_link, get_status, status_bg_color, rk_mode
):
    import numpy as np
    import streamlit as st

    if len(idx) == 0:
        st.info("No data to display.")
        return

    def fmt_num(x, n=2):
        if x is None or (isinstance(x,float) and np.isnan(x)):
            return "-"
        return f"{x:,.{n}f}"

    # Build rows
    rows = []
    for t in tickers:
        try:
            rr = float(rs_ratio_map[t].iloc[sel_pos]); mm = float(rs_mom_map[t].iloc[sel_pos])
            px = px_cache.get(t, pd.Series(dtype=float))
            price = float(px.iloc[sel_pos]) if sel_pos < len(px) else np.nan
            chg = ((px.iloc[sel_pos]/px.iloc[start_pos]-1)*100.0) if (sel_pos < len(px) and start_pos < len(px)) else np.nan
            status = get_status(rr, mm)
            rows.append((t, display_name(t), status, price, chg))
        except Exception:
            pass

    # Ranking → SL No.
    all_perf = []
    for t in tickers:
        try:
            all_perf.append((t, compute_rank_metric(
                t, start_pos, sel_pos,
                rs_ratio_map=rs_ratio_map, rs_mom_map=rs_mom_map, px_cache=px_cache, mode=rk_mode
            )))
        except Exception:
            pass
    all_perf.sort(key=lambda x: x[1], reverse=True)
    rank_map = {sym:i for i,(sym,_m) in enumerate(all_perf, start=1)}

    # CSS + HTML
    table_css = """
    <style>
    .rrg-wrap { max-height: 520px; overflow-y: auto; border: 1px solid rgba(128,128,128,0.25); border-radius: 6px; }
    .rrg-table { width:100%; border-collapse:collapse; }
    .rrg-table th, .rrg-table td {
      padding:10px 12px; border-bottom:1px solid rgba(128,128,128,0.22);
      font-family: Segoe UI, Inter, Arial, system-ui; font-size:15px; line-height: 1.2;
    }
    .rrg-table th { position: sticky; top: 0; background: rgba(200,200,200,0.10); text-align:left; z-index: 1; }
    .rrg-color { width:14px; height:14px; border-radius:3px; display:inline-block; border:1px solid rgba(0,0,0,0.25); margin-right:4px; vertical-align: -2px; }
    .rrg-name { font-weight: 600; text-decoration: underline; }
    </style>
    """

    html = [table_css, f"<p><strong>Date:</strong> {str(idx[sel_pos].date())}</p>", """
    <div class="rrg-wrap">
    <table class="rrg-table">
    <thead><tr>
    <th style="width:72px;">SL No.</th>
    <th style="width:26px;"></th>
    <th>Name</th>
    <th style="width:130px;">Status</th>
    <th style="width:140px;">Price</th>
    <th style="width:140px;">Change %</th>
    </tr></thead>
    <tbody>
    """]

    for sym, name, status, price, chg in rows:
        sl = rank_map.get(sym, "")
        _, url = tv_build_link(sym)
        rr_last = float(rs_ratio_map[sym].iloc[sel_pos]); mm_last = float(rs_mom_map[sym].iloc[sel_pos])
        bg = status_bg_color(rr_last, mm_last)
        fg = "white" if bg in ("#e06a6a","#3fa46a","#5d86d1") else "black"
        strip = COLORS.get(sym, "#888")
        html.append(
            f'<tr style="background:{bg}; color:{fg};">'
            f'<td>{sl}</td>'
            f'<td><span class="rrg-color" style="background:{strip}"></span></td>'
            f'<td><a class="rrg-name" href="{url}" target="_blank" rel="noopener noreferrer" style="color:{fg};">{name}</a></td>'
            f'<td>{status}</td>'
            f'<td>{fmt_num(price,2)}</td>'
            f'<td>{fmt_num(chg,2)}</td>'
            f'</tr>'
        )

    html.append("</tbody></table></div>")

    with st.expander("Table", expanded=True):
        st.markdown("".join(html), unsafe_allow_html=True)

# ========================== STREAMLIT APP ==========================
st.set_page_config(page_title="RRG Indicator — Stallions", layout="wide")
st.title("RRG Indicator")

with st.sidebar:
    st.markdown("**RRG — Controls**")
    bench_label = st.selectbox("Benchmark", list(BENCHMARK_LABEL_TO_SYMBOL.keys()),
                               index=list(BENCHMARK_LABEL_TO_SYMBOL.keys()).index(DEFAULT_BENCHMARK_LABEL))
    timeframe_label = st.selectbox("Timeframe", list(TIMEFRAME_CHOICES.keys()),
                                   index=list(TIMEFRAME_CHOICES.keys()).index(_default_tf_label))
    period_label = st.selectbox("Period", list(PERIOD_MAP.keys()),
                                index=list(PERIOD_MAP.keys()).index(period_label_cfg))
    rank_modes = ["RRG Power (dist)", "RS-Ratio", "RS-Momentum", "Price %Δ (tail)", "Momentum Slope (tail)"]
    rank_mode = st.selectbox("Rank by", rank_modes, index=0)
    theme = st.selectbox("Theme", ["Light","Dark","System"], index=0)
    tail = st.slider("Trail Length", 1, 20, DEFAULT_TAIL, 1)

PAL = pal_dark if theme=="Dark" else pal_light
mpl.rcParams['axes.edgecolor'] = PAL["axes_edge"]
mpl.rcParams['axes.labelcolor'] = PAL["text"]
mpl.rcParams['xtick.color'] = PAL["muted"]
mpl.rcParams['ytick.color'] = PAL["muted"]
mpl.rcParams['text.color'] = PAL["text"]
mpl.rcParams['figure.facecolor'] = PAL["figure"]
mpl.rcParams['axes.facecolor'] = PAL["axes"]

# Universe (required + GitHub CSVs)
universe = list(dict.fromkeys(UNIVERSE + load_github_csv_universe()))
benchmark = BENCHMARK_LABEL_TO_SYMBOL[bench_label]
interval = TIMEFRAME_CHOICES[timeframe_label]
period = PERIOD_MAP[period_label]

# Data build
try:
    bench, rs_ratio_map, rs_mom_map, px_cache, idx = build_rrg(benchmark, universe, period, interval, WINDOW)
    tickers = list(rs_ratio_map.keys())
    COLORS = symbol_color_map(tickers)
except Exception as e:
    st.error(f"Data build failed: {e}")
    st.stop()

# Date slider (top)
end_idx = len(idx) - 1
sel_pos = st.slider("Date", 0, end_idx, end_idx, 1)
start_pos = max(sel_pos - tail, 0)

# Visible selection (mirrors checkbox column in desktop)
if "visible" not in st.session_state:
    st.session_state.visible = tickers.copy()
visible = st.multiselect("Visible series", options=tickers, default=st.session_state.visible, key="visible")

# ---- Layout: chart + ranking
c1, c2 = st.columns([4, 1])

# Chart
with c1:
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_facecolor(PAL["axes"])
    ax.set_title("RRG Indicator", fontsize=13, pad=10, color=PAL["text"])
    ax.set_xlabel("JdK RS-Ratio"); ax.set_ylabel("JdK RS-Momentum")
    ax.axhline(y=100, color=PAL["quad_edges"], linestyle=":", linewidth=1.0)
    ax.axvline(x=100, color=PAL["quad_edges"], linestyle=":", linewidth=1.0)

    a = PAL["quad_alpha"] * 0.9
    ax.fill_between([92, 100], [92, 92],  [100, 100], color=(1.0, 0.0, 0.0, a))  # Lagging
    ax.fill_between([100, 108],[92, 92],  [100, 100], color=(1.0, 1.0, 0.0, a))  # Weakening
    ax.fill_between([100, 108],[100, 100],[108, 108], color=(0.0, 1.0, 0.0, a))  # Leading
    ax.fill_between([92, 100], [100, 100],[108, 108], color=(0.0, 0.0, 1.0, a))  # Improving

    ax.text(95, 105, "Improving", fontsize=11, color=PAL["text"], weight="bold")
    ax.text(104, 105, "Leading",   fontsize=11, color=PAL["text"], weight="bold", ha="right")
    ax.text(104, 95,  "Weakening", fontsize=11, color=PAL["text"], weight="bold", ha="right")
    ax.text(95, 95,   "Lagging",   fontsize=11, color=PAL["text"], weight="bold")

    xs_all, ys_all = [], []
    for t in tickers:
        if t not in visible:
            continue
        rr = rs_ratio_map[t].iloc[start_pos+1:sel_pos+1]
        mm = rs_mom_map[t].iloc[start_pos+1:sel_pos+1]
        if len(rr)==0 or len(mm)==0: continue
        n = len(rr)
        sizes  = np.linspace(12, 70, n)
        alphas = np.linspace(0.40, 0.95, n)
        ax.scatter(rr.values, mm.values, s=sizes,
                   facecolors=[(*mpl.colors.to_rgb(COLORS[t]), a_) for a_ in alphas],
                   edgecolors=PAL["scatter_edge"], linewidths=0.6)
        ax.plot(rr.values, mm.values, linewidth=1.1, alpha=0.6, color=COLORS[t])
        ax.annotate(f"{t}  [{get_status(rr.values[-1], mm.values[-1])}]",
                    (rr.values[-1], mm.values[-1]), fontsize=9, color=COLORS[t])
        xs_all.extend(rr.values); ys_all.extend(mm.values)

    if xs_all and ys_all:
        x0, x1 = min(xs_all + [96]), max(xs_all + [104])
        y0, y1 = min(ys_all + [96]), max(ys_all + [104])
        pad = 1.5
        ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y0 - pad, y1 + pad)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.35)
    st.pyplot(fig, clear_figure=True)

# Ranking
with c2:
    st.markdown("### Ranking")
    perf = []
    for t in tickers:
        if t not in visible:
            continue
        try:
            perf.append((t, compute_rank_metric(
                t, start_pos, sel_pos,
                rs_ratio_map=rs_ratio_map, rs_mom_map=rs_mom_map, px_cache=px_cache, mode=rank_mode
            )))
        except Exception:
            pass
    perf.sort(key=lambda x: x[1], reverse=True)
    for i, (sym, _m) in enumerate(perf[:22], start=1):
        rr_last = float(rs_ratio_map[sym].iloc[sel_pos]); mm_last = float(rs_mom_map[sym].iloc[sel_pos])
        st.write(f"{i}. {display_name(sym)} [{get_status(rr_last, mm_last)}]")

# ---- Table (scrollable + expander + colored + hyperlink)
render_rrg_table(
    tickers=tickers,
    rs_ratio_map=rs_ratio_map,
    rs_mom_map=rs_mom_map,
    px_cache=px_cache,
    idx=idx,
    start_pos=start_pos,
    sel_pos=sel_pos,
    COLORS=COLORS,
    tv_build_link=tv_build_link,
    get_status=get_status,
    status_bg_color=status_bg_color,
    rk_mode=rank_mode,
)

# ---- Export buttons
colA, colB, colC = st.columns(3)
with colA:
    if st.button("Export Ranks CSV"):
        df = pd.DataFrame(
            [(t,
              compute_rank_metric(t, start_pos, sel_pos,
                                  rs_ratio_map=rs_ratio_map, rs_mom_map=rs_mom_map, px_cache=px_cache, mode=rank_mode),
              float(rs_ratio_map[t].iloc[sel_pos]),
              float(rs_mom_map[t].iloc[sel_pos]),
              get_status(float(rs_ratio_map[t].iloc[sel_pos]), float(rs_mom_map[t].iloc[sel_pos])))
             for t in tickers if t in visible],
            columns=["symbol","rank_metric","rs_ratio","rs_momentum","status"]
        ).sort_values("rank_metric", ascending=False)
        st.download_button("Download ranks.csv", df.to_csv(index=False),
                           file_name=f"ranks_{idx[sel_pos].date()}.csv")

with colB:
    if st.button("Export Table CSV"):
        rows = []
        for t in tickers:
            if t not in visible: continue
            rr = float(rs_ratio_map[t].iloc[sel_pos]); mm = float(rs_mom_map[t].iloc[sel_pos])
            px = px_cache.get(t, pd.Series(dtype=float))
            price = float(px.iloc[sel_pos]) if sel_pos < len(px) else np.nan
            chg = ((px.iloc[sel_pos]/px.iloc[start_pos]-1)*100.0) if (sel_pos < len(px) and start_pos < len(px)) else np.nan
            rows.append({
                "symbol": t,
                "name": display_name(t),
                "status": get_status(rr, mm),
                "rs_ratio": rr,
                "rs_momentum": mm,
                "price": price,
                "pct_change_tail": chg
            })
        df = pd.DataFrame(rows)
        st.download_button("Download table.csv", df.to_csv(index=False),
                           file_name=f"table_{idx[sel_pos].date()}.csv")

with colC:
    if st.button("Export Chart PNG"):
        buf = io.BytesIO()
        fig = plt.gcf()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=180)
        st.download_button(
            "Download rrg.png",
            buf.getvalue(),
            file_name=f"rrg_{BENCHMARK_SYMBOL_TO_LABEL.get(benchmark,benchmark).replace(' ','')}_{idx[sel_pos].date()}.png",
            mime="image/png"
        )

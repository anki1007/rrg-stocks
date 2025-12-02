import os, io, json, time, yaml, pathlib, logging, functools
from typing import Dict, List, Tuple

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# ------------ Appearance (keep same defaults) ------------
mpl.rcParams['figure.dpi'] = 110
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.edgecolor'] = '#222'
mpl.rcParams['axes.labelcolor'] = '#111'
mpl.rcParams['xtick.color'] = '#333'
mpl.rcParams['ytick.color'] = '#333'
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.sans-serif'] = ['Inter', 'Segoe UI', 'DejaVu Sans', 'Arial']

# ------------ Logging ------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ------------ Config (YAML) ------------
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
        # point to your GitHub ticker folder (raw)
        "github_ticker_base": "https://raw.githubusercontent.com/anki1007/rrg-stocks/main/ticker/",
        "ticker_csvs": [
            "nifty200.csv","nifty500.csv","niftymidcap150.csv","niftymidsmallcap400.csv",
            "niftysmallcap250.csv","niftytotalmarket.csv"
        ]
    }
    with open(CFG_PATH, "w") as f:
        yaml.safe_dump(DEFAULT_CFG, f)
CFG = yaml.safe_load(open(CFG_PATH, "r"))

# ------------ Constants / mappings ------------
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

PERIOD_MAP = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y": "3y"}
period_label = CFG.get("ui_default_period", "1Y")
period = PERIOD_MAP.get(period_label, "1y")

TIMEFRAME_CHOICES = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
interval = CFG.get("interval", "1wk")
_default_tf_label = next((k for k, v in TIMEFRAME_CHOICES.items() if v == interval), "Weekly")

WINDOW = int(CFG.get("window", 14))
DEFAULT_TAIL = int(CFG.get("default_tail", 8))

# Merge config tickers with required
_cfg_tickers = CFG.get("tickers", [])
UNIVERSE = list(dict.fromkeys(_cfg_tickers + REQUIRED_TICKERS))

CACHE_DIR = pathlib.Path(CFG.get("cache_dir", "cache"))
CACHE_DIR.mkdir(exist_ok=True)

NAMES_FILE = CACHE_DIR / "names.json"

IST = "Asia/Kolkata"

# ------------ Helpers ------------
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
    s = s.copy(); s.index = set_index_to_1700(s.index)
    return s

def resample_series_to_mode(s: pd.Series, mode: str) -> pd.Series:
    if s.empty: return s
    daily = to_1700_ist_series(s)
    if mode == "1d": return daily
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

def _now_ist(): return pd.Timestamp.now(IST)

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
    if df.empty: return pd.Series(dtype=float), pd.Series(dtype=float)
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
                    if i == n - 1: raise
                    logging.warning(f"{fn.__name__} failed ({i+1}/{n}): {e}. Retry in {d:.1f}s")
                    time.sleep(d); d *= backoff
        return wrapper
    return deco

def pick_close(raw, symbol: str) -> pd.Series:
    if isinstance(raw, pd.Series): return raw.dropna()
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
        if s.empty: return pd.Series(dtype=float)
        s = resample_series_to_mode(s, mode).dropna()
        if exclude_partial and len(s) >= 1 and not _is_last_bar_complete(s.index, mode):
            s = s.iloc[:-1]
        return s

    bench = _pipe(benchmark)
    data: Dict[str, pd.Series] = {t: _pipe(t) for t in tickers}
    return bench, {t: s for t, s in data.items() if not s.empty}

# ------------ Load extra tickers from GitHub CSVs ------------
def load_github_csv_universe() -> List[str]:
    base = CFG.get("github_ticker_base","")
    files = CFG.get("ticker_csvs",[])
    extra = []
    for fname in files:
        try:
            url = base + fname
            df = pd.read_csv(url)
            # try common column names
            cols = [c for c in df.columns if str(c).lower() in ("symbol","ticker","tickers","symbols")]
            if cols:
                vals = df[cols[0]].astype(str).str.strip().tolist()
                extra.extend([v if v.endswith(".NS") or v.startswith("^") else (v+".NS") for v in vals])
        except Exception:
            continue
    return list(dict.fromkeys(extra))

# ------------ Build RRG ------------
def build_rrg(benchmark: str, universe: List[str], period: str, mode: str, window: int):
    bench, data = download_block_daily_then_resample(universe, benchmark, period, mode, True)
    if bench is None or bench.empty: raise RuntimeError("No benchmark data")
    rs_ratio_map, rs_mom_map = {}, {}
    for t, s in data.items():
        if t == benchmark: continue
        rr, mm = jdk_components(s, bench, window)
        if len(rr) and len(mm):
            ix = rr.index.intersection(mm.index)
            rs_ratio_map[t] = rr.loc[ix]
            rs_mom_map[t]   = mm.loc[ix]
    if not rs_ratio_map: raise RuntimeError("No valid tickers after RS calc.")
    common = None
    for t in rs_ratio_map:
        c = rs_ratio_map[t].index.intersection(rs_mom_map[t].index)
        common = c if common is None else common.intersection(c)
    if common is None or len(common) < max(window+5, 20):
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

# ------------ Streamlit UI ------------
st.set_page_config(page_title="RRG Indicator — Stallions", layout="wide")
st.title("RRG Indicator")

with st.sidebar:
    st.markdown("**RRG — Controls**")
    bench_label = st.selectbox("Benchmark", list(BENCHMARK_LABEL_TO_SYMBOL.keys()),
                               index=list(BENCHMARK_LABEL_TO_SYMBOL.keys()).index(DEFAULT_BENCHMARK_LABEL))
    timeframe_label = st.selectbox("Timeframe", list(TIMEFRAME_CHOICES.keys()),
                                   index=list(TIMEFRAME_CHOICES.keys()).index(_default_tf_label))
    period_label = st.selectbox("Period", list(PERIOD_MAP.keys()),
                                index=list(PERIOD_MAP.keys()).index(CFG.get("ui_default_period","1Y")))
    rank_modes = ["RRG Power (dist)", "RS-Ratio", "RS-Momentum", "Price %Δ (tail)", "Momentum Slope (tail)"]
    rank_mode = st.selectbox("Rank by", rank_modes, index=0)
    theme = st.selectbox("Theme", ["Light","Dark"], index=0)
    tail = st.slider("Trail Length", 1, 20, DEFAULT_TAIL, 1)

pal_light = {
    "bg": "#f5f6f8","panel": "#ffffff","text": "#111111","muted": "#333333",
    "axes": "#ffffff","axes_edge": "#222222","grid": "#777777","figure": "#f5f6f8",
    "rank_text": "#222222","quad_alpha": 0.22,"quad_edges": "#777777","scatter_edge": "#333333",
}
pal_dark = {
    "bg": "#0f1115","panel": "#151821","text": "#e8eaed","muted": "#c2c3c7","axes": "#121420",
    "axes_edge": "#565b66","grid": "#6b7280","figure": "#0f1115","rank_text": "#e8eaed",
    "quad_alpha": 0.20,"quad_edges": "#6b7280","scatter_edge": "#8b8f99",
}
PAL = pal_light if theme=="Light" else pal_dark

mpl.rcParams['axes.edgecolor'] = PAL["axes_edge"]
mpl.rcParams['axes.labelcolor'] = PAL["text"]
mpl.rcParams['xtick.color'] = PAL["muted"]
mpl.rcParams['ytick.color'] = PAL["muted"]
mpl.rcParams['text.color'] = PAL["text"]
mpl.rcParams['figure.facecolor'] = PAL["figure"]
mpl.rcParams['axes.facecolor'] = PAL["axes"]

# Build universe (required + GitHub CSVs)
universe = list(dict.fromkeys(UNIVERSE + load_github_csv_universe()))
benchmark = BENCHMARK_LABEL_TO_SYMBOL[bench_label]
interval = TIMEFRAME_CHOICES[timeframe_label]
period = PERIOD_MAP[period_label]

# Build RRG data
try:
    bench, rs_ratio_map, rs_mom_map, px_cache, idx = build_rrg(benchmark, universe, period, interval, WINDOW)
    tickers = list(rs_ratio_map.keys())
    colors = symbol_color_map(tickers)
except Exception as e:
    st.error(f"Data build failed: {e}")
    st.stop()

# Date slider
end_idx = len(idx)-1
sel_pos = st.slider("Date", 0, end_idx, end_idx, 1, help="Choose end date for the tail")
start_pos = max(sel_pos - tail, 0)

# Layout: chart + ranking (two columns)
c1, c2 = st.columns([4,1])

# --- Chart ---
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

    # draw trails
    xs_all, ys_all = [], []
    for t in tickers:
        rr = rs_ratio_map[t].iloc[start_pos+1:sel_pos+1]
        mm = rs_mom_map[t].iloc[start_pos+1:sel_pos+1]
        if len(rr)==0 or len(mm)==0: continue
        pts = np.column_stack([rr.values, mm.values])
        n = len(rr)
        sizes  = np.linspace(12, 70, n)
        alphas = np.linspace(0.40, 0.95, n)
        ax.scatter(rr.values, mm.values, s=sizes,
                   facecolors=[(*mpl.colors.to_rgb(colors[t]), a) for a in alphas],
                   edgecolors=PAL["scatter_edge"], linewidths=0.6)
        ax.plot(rr.values, mm.values, linewidth=1.1, alpha=0.6, color=colors[t])
        ax.annotate(f"{t}  [{get_status(rr.values[-1], mm.values[-1])}]",
                    (rr.values[-1], mm.values[-1]), fontsize=9, color=colors[t])

        xs_all.extend(rr.values); ys_all.extend(mm.values)

    # auto-limits similar to desktop
    if xs_all and ys_all:
        x0, x1 = min(xs_all + [96]), max(xs_all + [104])
        y0, y1 = min(ys_all + [96]), max(ys_all + [104])
        pad = 1.5
        ax.set_xlim(x0 - pad, x1 + pad); ax.set_ylim(y0 - pad, y1 + pad)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.35)
    st.pyplot(fig, clear_figure=True)

# --- Ranking panel ---
def compute_rank_metric(t: str, start_pos: int, end_pos: int) -> float:
    rr_last = float(rs_ratio_map[t].iloc[end_pos]); mm_last = float(rs_mom_map[t].iloc[end_pos])
    if rank_mode == "RRG Power (dist)":
        wq = 1.2 if (rr_last>=100 and mm_last>=100) else (0.9 if (rr_last>=100 and mm_last<100) else 1.0)
        return float(np.hypot(rr_last-100.0, mm_last-100.0) * wq)
    if rank_mode == "RS-Ratio":         return float(rr_last)
    if rank_mode == "RS-Momentum":      return float(mm_last)
    if rank_mode == "Price %Δ (tail)":
        px = px_cache.get(t, pd.Series(dtype=float))
        if len(px.iloc[start_pos:end_pos+1]) >= 2:
            return float((px.iloc[end_pos] / px.iloc[start_pos] - 1) * 100.0)
        return float("-inf")
    if rank_mode == "Momentum Slope (tail)":
        series = rs_mom_map[t].iloc[start_pos:end_pos+1]
        if len(series) >= 2:
            x = np.arange(len(series)); A = np.vstack([x, np.ones(len(x))]).T
            slope = np.linalg.lstsq(A, series.values, rcond=None)[0][0]
            return float(slope)
        return float("-inf")
    return float("-inf")

with c2:
    st.markdown("### Ranking")
    perf = []
    for t in tickers:
        try:
            perf.append((t, compute_rank_metric(t, start_pos, sel_pos)))
        except Exception:
            pass
    perf.sort(key=lambda x: x[1], reverse=True)
    for i, (sym, _m) in enumerate(perf[:22], start=1):
        rr_last = float(rs_ratio_map[sym].iloc[sel_pos]); mm_last = float(rs_mom_map[sym].iloc[sel_pos])
        st.write(f"{i}. {display_name(sym)} [{get_status(rr_last, mm_last)}]")

# --- Table below ---
rows = []
for t in tickers:
    try:
        rr = float(rs_ratio_map[t].iloc[sel_pos]); mm = float(rs_mom_map[t].iloc[sel_pos])
        px = px_cache.get(t, pd.Series(dtype=float))
        price = float(px.iloc[sel_pos]) if sel_pos < len(px) else np.nan
        chg = ((px.iloc[sel_pos]/px.iloc[start_pos]-1)*100.0) if (sel_pos < len(px) and start_pos < len(px)) else np.nan
        rows.append({
            "SL No.": "", "Name": display_name(t), "Status": get_status(rr, mm),
            "Price": None if np.isnan(price) else round(price,2),
            "Change %": None if np.isnan(chg) else round(chg,2)
        })
    except Exception:
        pass

# Fill SL No. by current ranking
rank_map = {sym:i for i,(sym,_m) in enumerate(perf, start=1)}
for r in rows:
    sym = next((k for k in tickers if display_name(k)==r["Name"]), None)
    r["SL No."] = rank_map.get(sym,"")

st.markdown(f"**Date:** {str(idx[sel_pos].date())}")
st.dataframe(pd.DataFrame(rows), use_container_width=True)

# --- Export buttons (downloads) ---
colA, colB, colC = st.columns(3)
with colA:
    if st.button("Export Ranks CSV"):
        df = pd.DataFrame(
            [(t, compute_rank_metric(t, start_pos, sel_pos),
              float(rs_ratio_map[t].iloc[sel_pos]), float(rs_mom_map[t].iloc[sel_pos]),
              get_status(float(rs_ratio_map[t].iloc[sel_pos]), float(rs_mom_map[t].iloc[sel_pos])))
             for t in tickers],
            columns=["symbol","rank_metric","rs_ratio","rs_momentum","status"]
        ).sort_values("rank_metric", ascending=False)
        st.download_button("Download ranks.csv", df.to_csv(index=False), file_name=f"ranks_{idx[sel_pos].date()}.csv")

with colB:
    if st.button("Export Table CSV"):
        df = pd.DataFrame(rows)
        st.download_button("Download table.csv", df.to_csv(index=False), file_name=f"table_{idx[sel_pos].date()}.csv")

with colC:
    if st.button("Export Chart PNG"):
        fig2 = plt.gcf()
        buf = io.BytesIO()
        fig2.savefig(buf, format="png", bbox_inches="tight", dpi=180)
        st.download_button("Download rrg.png", buf.getvalue(), file_name=f"rrg_{BENCHMARK_SYMBOL_TO_LABEL.get(benchmark,benchmark).replace(' ','')}_{idx[sel_pos].date()}.png", mime="image/png")

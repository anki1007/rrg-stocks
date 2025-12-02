import os, time, pathlib, logging, functools, calendar, io
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

# ---------- Config ----------
GITHUB_USER = "anki1007"
GITHUB_REPO = "rrg-stocks"
GITHUB_BRANCH = "main"
GITHUB_TICKER_DIR = "ticker"
CSV_BASENAME = "niftyindices.csv"  # <— your CSV
RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_TICKER_DIR}/"

DEFAULT_TF = "Weekly"
WINDOW = 14
DEFAULT_TAIL = 8
PERIOD_MAP = {"6M": "6mo", "1Y": "1y", "2Y": "2y", "3Y": "3y", "5Y": "5y", "10Y": "10y"}
TF_LABELS = ["Daily", "Weekly", "Monthly"]
TF_TO_INTERVAL = {"Daily": "1d", "Weekly": "1wk", "Monthly": "1mo"}
BENCH_CHOICES = {"Nifty 500": "^CRSLDX", "Nifty 200": "^CNX200", "Nifty 50": "^NSEI"}

IST_TZ = "Asia/Kolkata"
BAR_CUTOFF_HOUR = 18
NET_TIME_MAX_AGE = 300

CACHE_DIR = pathlib.Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# ---------- Matplotlib ----------
mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["font.size"] = 13
mpl.rcParams["font.sans-serif"] = ["Inter", "Segoe UI", "DejaVu Sans", "Arial"]
mpl.rcParams["axes.grid"] = False
mpl.rcParams["axes.edgecolor"] = "#222"
mpl.rcParams["axes.labelcolor"] = "#111"
mpl.rcParams["xtick.color"] = "#333"
mpl.rcParams["ytick.color"] = "#333"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ---------- Streamlit page ----------
st.set_page_config(page_title="Relative Rotation Graph (RRG)", layout="wide")
st.markdown(
    """
<style>
:root { color-scheme: light !important; }
[data-testid="stAppViewContainer"] { background: #ffffff !important; }
[data-testid="stHeader"] { background: #ffffff00; }
.block-container { padding-top: 1.0rem; }
html, body, [data-testid="stSidebar"], [data-testid="stMarkdownContainer"] { font-size: 16px; }
h1, h2, h3, h4, h5, h6, strong, b { color:#0f172a !important; }
[data-testid="stMarkdownContainer"] h3 { font-weight: 800; }
[data-testid="stSlider"] label, label span { color:#0f172a !important; }
.rrg-rank { font-weight: 700; line-height: 1.25; font-size: 1.05rem; white-space: pre; }
.rrg-rank .row { display: flex; gap: 8px; align-items: baseline; margin: 2px 0; }
.rrg-rank .name { color: #0b57d0; }
.rrg-wrap { max-height: calc(100vh - 260px); overflow: auto; border: 1px solid #e5e5e5; border-radius: 6px; }
.rrg-table { width: 100%; border-collapse: collapse; font-family: 'Segoe UI', -apple-system, Arial, sans-serif; }
.rrg-table th, .rrg-table td { border-bottom: 1px solid #ececec; padding: 10px 10px; font-size: 15px; }
.rrg-table th { position: sticky; top: 0; z-index: 2; text-align: left; background: #eef2f7; color: #0f172a; font-weight: 800; letter-spacing: .2px; }
.rrg-name a { color: #0b57d0; text-decoration: underline; }
.rrg-wrap::-webkit-scrollbar { height: 12px; width: 12px; }
.rrg-wrap::-webkit-scrollbar-thumb { background:#c7ccd6; border-radius: 8px; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
def _normalize_cols(cols: List[str]) -> Dict[str, str]:
    return {c: c.strip().lower().replace(" ", "").replace("_", "") for c in cols}

def _to_yahoo_symbol(raw_sym: str) -> str:
    s = str(raw_sym).strip().upper()
    if s.endswith(".NS") or s.startswith("^"):
        return s
    return "^" + s  # treat as index

@st.cache_data(ttl=600)
def load_universe_from_github_csv(basename: str):
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

def tv_link_for_symbol(yahoo_sym: str, META: dict) -> str:
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

# ---------- IST closed-bar checks ----------
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

# ---------- Cache / download ----------
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

# ---------- Controls ----------
st.sidebar.header("RRG — Controls")
bench_label = st.sidebar.selectbox("Benchmark", list(BENCH_CHOICES.keys()), index=0)
interval_label = st.sidebar.selectbox("Strength vs (TF)", TF_LABELS, index=TF_LABELS.index(DEFAULT_TF))
interval = TF_TO_INTERVAL[interval_label]
default_period_for_tf = {"1d": "1Y", "1wk": "1Y", "1mo": "5Y"}[interval]
period_label = st.sidebar.selectbox("Period", list(PERIOD_MAP.keys()), index=list(PERIOD_MAP.keys()).index(default_period_for_tf))
period = PERIOD_MAP[period_label]
rank_modes = ["RRG Power (dist)", "RS-Ratio", "RS-Momentum", "Price %Δ (tail)", "Momentum Slope (tail)"]
rank_mode = st.sidebar.selectbox("Rank by", rank_modes, index=0)
tail_len = st.sidebar.slider("Trail Length", 1, 20, DEFAULT_TAIL, 1)
show_labels = st.sidebar.toggle("Show labels on chart", value=False)
label_top_n = st.sidebar.slider("Label top N by distance", 3, 30, 12, 1, disabled=not show_labels)
if "playing" not in st.session_state:
    st.session_state.playing = False
st.sidebar.toggle("Play / Pause", value=st.session_state.playing, key="playing")
speed_ms = st.sidebar.slider("Speed (ms/frame)", 150, 1500, 300, 50)
looping = st.sidebar.checkbox("Loop", value=True)

# ---------- Data build ----------
UNIVERSE, META = load_universe_from_github_csv(CSV_BASENAME)
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

SYMBOL_COLORS = symbol_color_map(tickers)
idx = bench_idx
idx_len = len(idx)

# ---------- Date index + animation ----------
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
    st.autorefresh(interval=speed_ms, key="rrg_auto_refresh")

end_idx = st.slider("Date", min_value=DEFAULT_TAIL, max_value=idx_len - 1,
                    value=st.session_state.end_idx, step=1, key="end_idx", format=" ")
start_idx = max(end_idx - tail_len, 0)
date_str = format_bar_date(idx[end_idx], interval)

st.markdown(f"**Relative Rotation Graph (RRG) — {bench_label} — {period_label} — {interval_label} — {CSV_BASENAME} — {date_str}**")

# ---------- Plot + Ranking ----------
plot_col, rank_col = st.columns([4.5, 1.8], gap="medium")

with plot_col:
    fig, ax = plt.subplots(1, 1, figsize=(10.6, 6.8))
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
    ax.set_xlim(94, 106)
    ax.set_ylim(94, 106)

    if "visible_set" not in st.session_state:
        st.session_state.visible_set = set(tickers)

    def dist_last(t):
        rr_last = rs_ratio_map[t].iloc[end_idx]
        mm_last = rs_mom_map[t].iloc[end_idx]
        return float(np.hypot(rr_last - 100.0, mm_last - 100.0))

    allow_labels = set()
    if show_labels:
        allow_labels = {t for t, _ in sorted([(t, dist_last(t)) for t in tickers],
                                             key=lambda x: x[1], reverse=True)[:label_top_n]}

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

    def compute_rank_metric(t: str) -> float:
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

    perf = []
    for t in tickers:
        if t not in st.session_state.visible_set:
            continue
        perf.append((t, compute_rank_metric(t)))
    perf.sort(key=lambda x: x[1], reverse=True)

    if perf:
        rows_html = []
        for i, (sym, _) in enumerate(perf[:22], start=1):
            rr = float(rs_ratio_map[sym].iloc[end_idx])
            mm = float(rs_mom_map[sym].iloc[end_idx])
            stat = get_status(rr, mm)
            color = SYMBOL_COLORS.get(sym, "#333")
            name = META.get(sym, {}).get("name", sym)
            rows_html.append(
                f'<div class="row" style="color:{color}"><span>{i}.</span>'
                f'<span class="name">{name}</span><span>[{stat}]</span></div>'
            )
        st.markdown(f'<div class="rrg-rank">{"".join(rows_html)}</div>', unsafe_allow_html=True)
    else:
        st.write("—")

# ---------- Table ----------
def make_table_html(rows):
    headers = ["#", "Name", "Status", "Industry", "Price", "Change %"]
    th = "<tr>" + "".join([f"<th>{h}</th>" for h in headers]) + "</tr>"
    tr = []
    for r in rows:
        price_txt = "-" if pd.isna(r["price"]) else f"{r['price']:.2f}"
        chg_txt   = "-" if pd.isna(r["chg"])   else f"{r['chg']:.2f}"
        tr.append(
            "<tr class='rrg-row' style='background:%s; color:%s'>" % (r["bg"], r["fg"]) +
            f"<td>{r['rank']}</td>" +
            f"<td class='rrg-name'><a href='{r['tv']}' target='_blank'>{r['name']}</a></td>" +
            f"<td>{r['status']}</td>" +
            f"<td>{r['industry']}</td>" +
            f"<td>{price_txt}</td>" +
            f"<td>{chg_txt}</td>" +
            "</tr>"
        )
    return f"<div class='rrg-wrap'><table class='rrg-table'>{th}{''.join(tr)}</table></div>"

# rank by RS-Ratio for the SL.No column (stable)
rank_dict = {
    sym: i
    for i, (sym, _) in enumerate(
        sorted([(t, rs_ratio_map[t].iloc[end_idx]) for t in tickers if t in st.session_state.visible_set],
               key=lambda x: x[1], reverse=True),
        start=1,
    )
}

rows = []
for t in tickers:
    if t not in st.session_state.visible_set:
        continue
    rr = float(rs_ratio_map[t].iloc[end_idx])
    mm = float(rs_mom_map[t].iloc[end_idx])
    status = get_status(rr, mm)
    bg = status_bg_color(rr, mm)
    fg = "#ffffff" if bg in ("#e06a6a", "#3fa46a", "#5d86d1") else "#000000"
    px = tickers_data[t].reindex(idx).dropna()
    price = float(px.iloc[end_idx]) if end_idx < len(px) else np.nan
    chg = ((px.iloc[end_idx] / px.iloc[start_idx] - 1) * 100.0) if (end_idx < len(px) and start_idx < len(px)) else np.nan
    tv = tv_link_for_symbol(t, META)

    # >>> FIXED: use `chg` (not `ch`) <<<
    rows.append({
        "rank": rank_dict.get(t, ""),
        "name": META.get(t, {}).get("name", t),
        "status": status,
        "industry": META.get(t, {}).get("industry", "-"),
        "price": price,
        "chg": chg,          # <-- correct key/value
        "bg": bg,
        "fg": fg,
        "tv": tv,
    })

with st.expander("Table", expanded=True):
    st.markdown(make_table_html(rows), unsafe_allow_html=True)

# ---------- Downloads ----------
def export_ranks_csv(perf_sorted):
    out=[]
    for t,_m in perf_sorted:
        rr=float(rs_ratio_map[t].iloc[end_idx]); mm=float(rs_mom_map[t].iloc[end_idx])
        out.append((t, META.get(t,{}).get("name",t), META.get(t,{}).get("industry","-"),
                    _m, rr, mm, get_status(rr, mm)))
    df=pd.DataFrame(out, columns=["symbol","name","industry","rank_metric","rs_ratio","rs_momentum","status"])
    buf=io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode()

def export_table_csv(rows_):
    df=pd.DataFrame([{
        "name": r["name"], "industry": r["industry"], "status": r["status"],
        "price": r["price"], "pct_change_tail": r["chg"]
    } for r in rows_])
    buf=io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode()

# recompute perf for download using chosen rank mode
perf_dl=[]
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
    perf_dl.append((t, metric))
perf_dl.sort(key=lambda x:x[1], reverse=True)

c1, c2 = st.columns(2)
with c1:
    st.download_button("Download Ranks CSV", data=export_ranks_csv(perf_dl),
                       file_name=f"ranks_{date_str}.csv", mime="text/csv", use_container_width=True)
with c2:
    st.download_button("Download Table CSV", data=export_table_csv(rows),
                       file_name=f"table_{date_str}.csv", mime="text/csv", use_container_width=True)

st.caption("Names open TradingView. Use Play/Pause to watch rotation; Speed controls frame interval; Loop wraps frames.")

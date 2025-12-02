# app.py — Streamlit RRG with Play/Pause + white UI + name links + scrollable table
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
mpl.rcParams['figure.dpi'] = 110
mpl.rcParams['axes.grid'] = False
mpl.rcParams['axes.edgecolor'] = '#222'
mpl.rcParams['axes.labelcolor'] = '#111'
mpl.rcParams['xtick.color'] = '#333'
mpl.rcParams['ytick.color'] = '#333'
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.sans-serif'] = ['Segoe UI','Inter','DejaVu Sans','Arial']

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# -------------------- Streamlit page --------------
st.set_page_config(page_title="Relative Rotation Graph (RRG)", layout="wide")
st.markdown("""
<style>
:root { color-scheme: light !important; }
[data-testid="stAppViewContainer"] { background: #ffffff !important; }
[data-testid="stHeader"] { background: #ffffff00; }
.block-container { padding-top: 1.0rem; }

/* ranking */
.rrg-rank { font-weight: 700; line-height: 1.2; font-size: 0.98rem; white-space: pre; }

/* table */
.rrg-wrap { max-height: 520px; overflow-y: auto; border: 1px solid #e5e5e5; border-radius: 8px; }
.rrg-table { width: 100%; border-collapse: separate; border-spacing: 0; font-family: 'Segoe UI', -apple-system, Arial, sans-serif; }
.rrg-table th, .rrg-table td { border-bottom: 1px solid #ececec; padding: 10px 12px; font-size: 14px; }
.rrg-table th { position: sticky; top: 0; z-index: 2; text-align: left; background: #eaeaea; font-weight: 700; font-size: 15px;}
.rrg-row { transition: background .15s ease; }
.rrg-name a { color: #1a73e8; text-decoration: underline; }
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
    out=""; 
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
    meta = {r["Symbol"]:{"name":r["Company Name"] or r["Symbol"], "industry":r["Industry"] or "-"} for _,r in sel.iterrows()}
    return universe, meta

# -------------------- Config & utils ----------------
PERIOD_MAP = {"6M":"6mo","1Y":"1y","2Y":"2y","3Y":"3y","5Y":"5y","10Y":"10y"}
TF_LABELS = ["Daily","Weekly","Monthly"]
TF_TO_INTERVAL = {"Daily":"1d","Weekly":"1wk","Monthly":"1mo"}
WINDOW = 14
DEFAULT_TAIL = 8
BENCH_CHOICES = {"Nifty 500":"^CRSLDX","Nifty 200":"^CNX200","Nifty 50":"^NSEI"}

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
    return {"Lagging":"#e06a6a","Leading":"#3fa46a","Improving":"#5d86d1","Weakening":"#e2d06b"}.get(m,"#aaaaaa")

# -------------------- Closed-bar enforcement --------
IST_TZ="Asia/Kolkata"; BAR_CUTOFF_HOUR=18; NET_TIME_MAX_AGE=300
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

# ---------- Playback controls ----------
if "playing" not in st.session_state: st.session_state.playing = False
st.session_state.playing = st.sidebar.toggle("Play / Pause", value=st.session_state.playing, key="playing")
speed_ms = st.sidebar.slider("Speed (ms/frame)", 150, 1500, 300, 50)
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
        if looping: nxt = DEFAULT_TAIL
        else:
            nxt = idx_len - 1
            st.session_state.playing = False
    st.session_state.end_idx = nxt
    st.autorefresh(interval=speed_ms, key="rrg_auto_refresh")

end_idx = st.slider("Date", min_value=DEFAULT_TAIL, max_value=idx_len-1,
                    value=st.session_state.end_idx, step=1, key="end_idx",
                    format=" ", help="RRG date position (closed bars only).")

start_idx = max(end_idx - tail_len, 0)
date_str = format_bar_date(idx[end_idx], interval)

# -------------------- Title -------------------------
st.markdown(f"**Relative Rotation Graph (RRG) — {bench_label} — {period_label} — {interval_label} — {csv_disp} — {date_str}**")

# -------------------- Layout: Plot + Ranking ----------
plot_col, rank_col = st.columns([4.5, 1.8], gap="medium")

with plot_col:
    fig, ax = plt.subplots(1, 1, figsize=(9.8, 6.4))
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

    if "visible_set" not in st.session_state:
        st.session_state.visible_set = set(tickers)

    # plot trails with NAME labels (not symbols)
    for t in tickers:
        if t not in st.session_state.visible_set: continue
        rr=rs_ratio_map[t].iloc[start_idx+1:end_idx+1].dropna()
        mm=rs_mom_map[t].iloc[start_idx+1:end_idx+1].dropna()
        rr,mm=rr.align(mm, join="inner")
        if len(rr)==0 or len(mm)==0: continue
        ax.plot(rr.values, mm.values, linewidth=1.1, alpha=0.6, color=SYMBOL_COLORS[t])
        sizes=[18]*(len(rr)-1)+[70]
        ax.scatter(rr.values, mm.values, s=sizes, linewidths=0.6,
                   facecolor=SYMBOL_COLORS[t], edgecolor="#333333")
        rr_last, mm_last = rr.values[-1], mm.values[-1]
        label_name = safe_long_name(t, META)  # <-- use company name
        ax.annotate(f"{label_name}  [{get_status(rr_last, mm_last)}]", (rr_last, mm_last),
                    fontsize=9, color=SYMBOL_COLORS[t])
    st.pyplot(fig, use_container_width=True)

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
        if t not in st.session_state.visible_set: continue
        perf.append((t, compute_rank_metric(t)))
    perf.sort(key=lambda x:x[1], reverse=True)

    if not perf:
        st.write("—")
    else:
        lines=[]
        for i,(sym,_m) in enumerate(perf[:22], start=1):
            rr=float(rs_ratio_map[sym].iloc[end_idx]); mm=float(rs_mom_map[sym].iloc[end_idx])
            stat=get_status(rr, mm)
            color=SYMBOL_COLORS.get(sym, "#333")
            name = safe_long_name(sym, META)
            tv = f'https://www.tradingview.com/chart/?symbol={quote("NSE:"+display_symbol(sym).replace("-","_"), safe="")}'
            lines.append(f'<div style="color:{color}">{i}. <a href="{tv}" target="_blank" style="color:{color};text-decoration:underline">{name}</a> [{stat}]</div>')
        st.markdown(f'<div class="rrg-rank">{"".join(lines)}</div>', unsafe_allow_html=True)

# -------------------- Table under the plot -----------
def make_table_html(rows):
    # header row (sticky)
    th = "<tr>" + "".join([f"<th>{h}</th>" for h in ["#", "Name", "Status", "Industry", "Price", "Change %"]]) + "</tr>"
    tr = []
    for r in rows:
        bg = r["bg"]; fg = r["fg"]
        tr.append(
            f'<tr class="rrg-row" style="background:{bg}; color:{fg}">'
            f'<td>{r["rank"]}</td>'
            f'<td class="rrg-name"><a href="{r["tv"]}" target="_blank">{r["name"]}</a></td>'
            f'<td>{r["status"]}</td>'
            f'<td>{r["industry"]}</td>'
            f'<td>{("-" if pd.isna(r["price"]) else f"{r["price"]:.2f}")}</td>'
            f'<td>{("-" if pd.isna(r["chg"]) else f"{r["chg"]:.2f}")}</td>'
            f'</tr>'
        )
    return f'<div class="rrg-wrap"><table class="rrg-table">{th}{"".join(tr)}</table></div>'

# Simple rank (by RS-Ratio) so the first column isn’t empty
rank_dict = {sym:i for i,(sym,_m) in enumerate(sorted(
    [(t, rs_ratio_map[t].iloc[end_idx]) for t in tickers if t in st.session_state.visible_set],
    key=lambda x:x[1], reverse=True), start=1)}

rows=[]
for t in tickers:
    if t not in st.session_state.visible_set: continue
    rr=float(rs_ratio_map[t].iloc[end_idx]); mm=float(rs_mom_map[t].iloc[end_idx])
    status=get_status(rr, mm)
    bg = status_bg_color(rr, mm)
    fg = "#ffffff" if bg in ("#e06a6a","#3fa46a","#5d86d1") else "#000000"
    px=tickers_data[t].reindex(idx).dropna()
    price=float(px.iloc[end_idx]) if end_idx < len(px) else np.nan
    chg=((px.iloc[end_idx]/px.iloc[start_idx]-1)*100.0) if (end_idx < len(px) and start_idx < len(px)) else np.nan
    tv=f'https://www.tradingview.com/chart/?symbol={quote("NSE:"+display_symbol(t).replace("-","_"), safe="")}'
    rows.append({
        "rank": rank_dict.get(t, ""), "name": safe_long_name(t, META), "status": status,
        "industry": META.get(t,{}).get("industry","-"), "price": price, "chg": chg,
        "bg": bg, "fg": fg, "tv": tv
    })

st.markdown("### Table")
with st.expander("Show / Hide Table", expanded=True):
    st.markdown(make_table_html(rows), unsafe_allow_html=True)

# -------------------- Downloads ----------------------
def export_ranks_csv(perf_sorted):
    out=[]
    for t,_m in perf_sorted:
        rr=float(rs_ratio_map[t].iloc[end_idx]); mm=float(rs_mom_map[t].iloc[end_idx])
        out.append((t, META.get(t,{}).get("name",t), META.get(t,{}).get("industry","-"),
                    _m, rr, mm, get_status(rr, mm)))
    df=pd.DataFrame(out, columns=["symbol","name","industry","rank_metric","rs_ratio","rs_momentum","status"])
    buf=io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode()

def export_table_csv(rows):
    df=pd.DataFrame([{
        "name": r["name"], "industry": r["industry"], "status": r["status"],
        "price": r["price"], "pct_change_tail": r["chg"]
    } for r in rows])
    buf=io.StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode()

# Ranking (same metric as side panel) for CSV
perf=[]
def metric_for_csv(t):
    rr_last=rs_ratio_map[t].iloc[end_idx]; mm_last=rs_mom_map[t].iloc[end_idx]
    if rank_mode=="RRG Power (dist)": return float(np.hypot(rr_last-100.0, mm_last-100.0))
    if rank_mode=="RS-Ratio": return float(rr_last)
    if rank_mode=="RS-Momentum": return float(mm_last)
    if rank_mode=="Price %Δ (tail)":
        px=tickers_data[t].reindex(idx).dropna()
        return float((px.iloc[end_idx]/px.iloc[start_idx]-1)*100.0) if len(px.iloc[start_idx:end_idx+1])>=2 else float("-inf")
    series=rs_mom_map[t].iloc[start_idx:end_idx+1].dropna()
    return float(np.linalg.lstsq(np.vstack([np.arange(len(series)), np.ones(len(series))]).T, series.values, rcond=None)[0][0]) if len(series)>=2 else float("-inf")

for t in tickers:
    if t not in st.session_state.visible_set: continue
    perf.append((t, metric_for_csv(t)))
perf.sort(key=lambda x:x[1], reverse=True)

dl1, dl2 = st.columns(2)
with dl1:
    st.download_button("Download Ranks CSV", data=export_ranks_csv(perf),
                       file_name=f"ranks_{date_str}.csv", mime="text/csv", use_container_width=True)
with dl2:
    st.download_button("Download Table CSV", data=export_table_csv(rows),
                       file_name=f"table_{date_str}.csv", mime="text/csv", use_container_width=True)

st.caption("Names are clickable (TradingView). RRG labels use company names. Use Play/Pause to watch rotation; the table is scrollable and collapsible.")

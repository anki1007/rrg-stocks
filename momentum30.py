# streamlit_app.py — Pure Streamlit Momentum Screener (web-ready)

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# ---------------- Page + CSS ----------------
st.set_page_config(page_title="Momentum Screener", layout="wide")

st.markdown("""
<style>
html, body, [class^="css"] { font-family: "Segoe UI", system-ui, -apple-system, Arial, sans-serif; }
.block-container { padding-top: 8px; padding-bottom: 8px; }
section[data-testid="stSidebar"] .block-container { padding: 16px 12px; }

/* Table shell */
.table-wrap { max-height: 74vh; overflow: auto; border: 1px solid #cbd5e1; border-radius: 10px; background:#ffffff; }
.table { width: 100%; border-collapse: separate; border-spacing: 0; }
.table thead th {
  position: sticky; top: 0;
  background: #e6edf5; color:#0f172a; font-weight: 800;
  border-bottom: 2px solid #cbd5e1; padding: 10px; white-space: nowrap;
}
.table tbody td {
  padding: 8px 10px; border-bottom: 1px solid #e5e7eb; white-space: nowrap; color: #0f172a;
  background-clip: padding-box;
}

/* Row color bands */
.row-green  td { background: #2aa86e1a; }   /* soft green */
.row-yellow td { background: #ffd84d33; }   /* soft yellow */
.row-blue   td { background: #3b82f633; }   /* soft blue */
.row-red    td { background: #ef444433; }   /* soft red */

/* Serial chip */
.serial {
  background:#1f2937; color:#fff; font-weight:700; border-radius:8px;
  padding: 2px 8px; display:inline-block; min-width: 28px; text-align:center;
}

a.name-link { color: inherit; text-decoration: none; font-weight: 600; }
a.name-link:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# ---------------- Config ----------------
# Six benchmark options you requested; resolver will pick the first symbol that works else fall back to ^NSEI.
BENCHMARKS: Dict[str, List[str]] = {
    "Nifty 50": ["^NSEI"],
    "Nifty 200": ["^CNX200", "^NSE200", "^NSEI"],
    "Nifty 500": ["^CRSLDX", "^CNX500", "^NSE500", "^NSEI"],
    "Nifty Midcap 150": ["^NIFTYMIDCAP150.NS", "^NSEI"],
    "Nifty Mid Smallcap 400": ["^NIFTYMIDSML400.NS", "^NIFTYMIDSMALLCAP400.NS", "^NSEI"],
    "Nifty Total Market": ["^NIFTYTOTALMKT.NS", "^NIFTYTOTMKT", "^NSEI"],
}
DEFAULT_PERIODS = {"1Y": "1y", "2Y": "2y", "3Y": "3y"}
RS_LOOKBACK_DAYS = 252
JDK_WINDOW = 21

# ---------------- CSV discovery (repo /ticker) ----------------
def discover_universe_csvs() -> Dict[str, Path]:
    """Build Indices Universe from ./ticker/*.csv (exclude niftyindices.csv)."""
    root = Path(__file__).resolve().parent
    tdir = root / "ticker"
    out: Dict[str, Path] = {}
    if not tdir.exists():
        return out

    pretty = {
        "nifty50.csv":             "Nifty 50",
        "nifty200.csv":            "Nifty 200",
        "nifty500.csv":            "Nifty 500",
        "niftymidcap150.csv":      "Nifty Midcap 150",
        "niftymidsmallcap400.csv": "Nifty Mid Smallcap 400",
        "niftysmallcap250.csv":    "Nifty Smallcap 250",
        "niftytotalmarket.csv":    "Nifty Total Market",
    }

    for p in sorted(tdir.glob("*.csv")):
        fname = p.name.lower()
        if fname == "niftyindices.csv":
            continue
        label = pretty.get(fname, p.stem.replace("_", " ").title())
        out[label] = p

    return dict(sorted(out.items(), key=lambda kv: kv[0].lower()))

# ---------------- Helpers ----------------
def tv_symbol_from_yf(sym: str) -> str:
    s = sym.strip().upper()
    return "NSE:" + s[:-3] if s.endswith(".NS") else "NSE:" + s

def tradingview_url(sym: str) -> str:
    return f"https://in.tradingview.com/chart/?symbol={tv_symbol_from_yf(sym)}"

def _pick_close(df, symbol: str) -> pd.Series:
    if isinstance(df, pd.Series): return df.dropna()
    if isinstance(df, pd.DataFrame):
        if isinstance(df.columns, pd.MultiIndex):
            for lvl in ("Adj Close", "Close"):
                if (symbol, lvl) in df.columns: return df[(symbol, lvl)].dropna()
        else:
            for col in ("Adj Close", "Close"):
                if col in df.columns: return df[col].dropna()
    return pd.Series(dtype=float)

def jdk_components(price: pd.Series, bench: pd.Series, win: int = JDK_WINDOW):
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

def perf_quadrant(x, y) -> str:
    if x >= 100 and y >= 100: return "Leading"
    if x < 100 and y >= 100:  return "Improving"
    if x < 100 and y < 100:   return "Lagging"
    return "Weakening"

def analyze_momentum(adj: pd.Series) -> dict | None:
    if adj is None or adj.empty or len(adj) < 252: return None
    ema100 = adj.ewm(span=100, adjust=False).mean()
    try:
        one_year = (adj.iloc[-1] / adj.iloc[-252] - 1.0) * 100.0
    except Exception:
        return None
    high_52w = adj.iloc[-252:].max()
    within_20 = adj.iloc[-1] >= high_52w * 0.8
    if len(adj) < 126: return None
    six = adj.iloc[-126:]
    up_days = (six.pct_change() > 0).sum() / len(six) * 100.0
    if (adj.iloc[-1] >= ema100.iloc[-1] and one_year >= 6.5 and within_20 and up_days > 45.0):
        try:
            r6 = (adj.iloc[-1]/adj.iloc[-126]-1.0)*100.0
            r3 = (adj.iloc[-1]/adj.iloc[-63]-1.0)*100.0
            r1 = (adj.iloc[-1]/adj.iloc[-21]-1.0)*100.0
        except Exception:
            return None
        return {"Return_6M": r6, "Return_3M": r3, "Return_1M": r1}
    return None

@st.cache_data(show_spinner=False)
def yf_download_cached(tickers: List[str], period: str, interval: str = "1d"):
    return yf.download(tickers, period=period, interval=interval, auto_adjust=True,
                       group_by="ticker", progress=False, threads=True)

def resolve_benchmark_symbol(label: str) -> str:
    """Pick the first working Yahoo symbol from the list; else fall back to ^NSEI."""
    for sym in BENCHMARKS.get(label, ["^NSEI"]):
        try:
            probe = yf.download(sym, period="3mo", interval="1d", progress=False, auto_adjust=True)
            s = _pick_close(probe, sym).dropna()
            if not s.empty:
                return sym
        except Exception:
            pass
    return "^NSEI"

def resample_weekly(series: pd.Series) -> pd.Series:
    return series.resample("W-FRI").last().dropna()

# ---------------- Core build ----------------
def build_table(universe: pd.DataFrame, bench_symbol: str, period_key: str, timeframe: str) -> pd.DataFrame:
    period = DEFAULT_PERIODS.get(period_key, "1y")
    tickers = universe["Symbol"].tolist()
    raw = yf_download_cached(tickers + [bench_symbol], period=period, interval="1d")

    bench = _pick_close(raw, bench_symbol).dropna()
    if timeframe == "Weekly":
        bench = resample_weekly(bench)
    if bench.empty:
        raise RuntimeError(f"Benchmark {bench_symbol} series empty")

    cutoff = bench.index.max() - pd.Timedelta(days=RS_LOOKBACK_DAYS + 5)
    bench_rs = bench.loc[bench.index >= cutoff].copy()

    rows = []
    for _, rec in universe.iterrows():
        sym, name, ind = rec.Symbol, rec.Name, rec.Industry
        s = _pick_close(raw, sym).dropna()
        if s.empty:
            continue
        if timeframe == "Weekly":
            s = resample_weekly(s)

        mom = analyze_momentum(s)
        if mom is None:
            continue

        s_rs = s.loc[s.index >= cutoff].copy()
        rr, mm = jdk_components(s_rs, bench_rs)
        if rr.empty or mm.empty:
            continue
        ix = rr.index.intersection(mm.index)
        rr_last = float(rr.loc[ix].iloc[-1])
        mm_last = float(mm.loc[ix].iloc[-1])

        rows.append({
            "#": None, "Name": name, "Industry": ind,
            "RS-Ratio": rr_last, "RS-Momentum": mm_last,
            "Status": perf_quadrant(rr_last, mm_last),
            "Return_6M": mom["Return_6M"], "Return_3M": mom["Return_3M"], "Return_1M": mom["Return_1M"],
            "Symbol": sym
        })

    if not rows:
        raise RuntimeError("No tickers passed the filters.")

    df = pd.DataFrame(rows)

    # Rank stack and sort
    df["Rank_6M"] = df["Return_6M"].rank(ascending=False, method="min")
    df["Rank_3M"] = df["Return_3M"].rank(ascending=False, method="min")
    df["Rank_1M"] = df["Return_1M"].rank(ascending=False, method="min")
    df["Final Rank"] = df["Rank_6M"] + df["Rank_3M"] + df["Rank_1M"]

    # Round for display
    for col in ("Return_6M","Return_3M","Return_1M"):
        df[col] = df[col].round(1)
    df["RS-Ratio"] = df["RS-Ratio"].round(2)
    df["RS-Momentum"] = df["RS-Momentum"].round(2)

    df = df.sort_values("Final Rank", ascending=True).reset_index(drop=True)

    # Serial numbers & band classes
    df["#"] = np.arange(1, len(df)+1)
    def band_class(n):
        if n <= 30: return "row-green"
        if n <= 60: return "row-yellow"
        if n <= 90: return "row-blue"
        return "row-red"
    df["__rowclass"] = df["#"].apply(band_class)

    show_cols = ["#", "Name", "Status", "Industry", "RS-Ratio", "RS-Momentum", "Return_6M", "Return_3M", "Return_1M"]
    return df[show_cols + ["Symbol", "__rowclass"]]

# ---------------- HTML render ----------------
def render_table(df: pd.DataFrame):
    headers = ["#", "Name", "Status", "Industry", "RS-Ratio", "RS-Momentum", "Return 6M", "Return 3M", "Return 1M"]
    body = []
    for _, r in df.iterrows():
        rowcls = r["__rowclass"]
        name = r["Name"]; sym = r["Symbol"]; url = tradingview_url(sym)
        cells = [
            f'<td><span class="serial">{int(r["#"])}</span></td>',
            f'<td><a class="name-link" href="{url}" target="_blank">{name}</a></td>',
            f'<td style="text-align:center;">{r["Status"]}</td>',
            f'<td>{r["Industry"]}</td>',
            f'<td style="text-align:center;">{r["RS-Ratio"]:.2f}</td>',
            f'<td style="text-align:center;">{r["RS-Momentum"]:.2f}</td>',
            f'<td style="text-align:center;">{r["Return_6M"]:.1f}</td>',
            f'<td style="text-align:center;">{r["Return_3M"]:.1f}</td>',
            f'<td style="text-align:center;">{r["Return_1M"]:.1f}</td>',
        ]
        body.append(f'<tr class="{rowcls}">{"".join(cells)}</tr>')

    thead = "".join(f"<th>{h}</th>" for h in headers)
    html = f"""
    <div class="table-wrap">
      <table class="table">
        <thead><tr>{thead}</tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ---------------- Sidebar (ONLY requested controls) ----------------
csv_map = discover_universe_csvs()
if not csv_map:
    st.sidebar.error("No CSVs found under ./ticker/")
    st.stop()

with st.sidebar:
    st.markdown("### Momentum Screener Controls")

    # 1) Indices Universe (from ./ticker, excluding niftyindices.csv)
    indices = st.selectbox("Indices Universe", list(csv_map.keys()),
                           index=(list(csv_map.keys()).index("Nifty 200")
                                  if "Nifty 200" in csv_map else 0))

    # 2) Benchmark (six options)
    benchmark_key = st.selectbox("Benchmark", list(BENCHMARKS.keys()),
                                 index=(list(BENCHMARKS.keys()).index("Nifty 500")
                                        if "Nifty 500" in BENCHMARKS else 0))

    # 3) Load / Refresh
    load_click = st.button("Load / Refresh")

    # 4) Timeframe
    timeframe = st.selectbox("Timeframe", ["Daily", "Weekly"], index=0)

    # 5) Period
    period_key = st.selectbox("Period", list(DEFAULT_PERIODS.keys()), index=0)

    # 6) Export CSV (filled after load)
    export_csv_btn = st.empty()

info = st.empty()
table_slot = st.empty()

# ---------------- Build & render ----------------
if load_click or "last_df" not in st.session_state:
    try:
        bench_symbol = resolve_benchmark_symbol(benchmark_key)
        csv_path = csv_map[indices]
        dfu = pd.read_csv(csv_path)

        cols = {c.strip().lower(): c for c in dfu.columns}
        for req in ("symbol", "company name", "industry"):
            if req not in cols:
                raise ValueError(f"{csv_path.name} must include columns: Symbol, Company Name, Industry")

        dfu = dfu[[cols["symbol"], cols["company name"], cols["industry"]]].copy()
        dfu.columns = ["Symbol", "Name", "Industry"]

        table_df = build_table(dfu, bench_symbol, period_key, timeframe)
        st.session_state["last_df"] = table_df.copy()
        st.session_state["meta"] = {
            "indices": indices, "csv": csv_path.name,
            "bench_key": benchmark_key, "bench_symbol": bench_symbol,
            "timeframe": timeframe, "period": period_key
        }
        info.write(
            f"Rows: **{len(table_df)}** • CSV: `{csv_path.name}` • Benchmark: `{benchmark_key}` ({bench_symbol}) • {timeframe} • {period_key}"
        )
    except Exception as e:
        info.error(f"Error: {e}")
        st.stop()

if "last_df" in st.session_state:
    table_df = st.session_state["last_df"]
    render_table(table_df)

    # Export CSV (sidebar)
    export_df = table_df.drop(columns=["Symbol", "__rowclass"]).copy()
    export_csv_btn.download_button(
        "Export CSV",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{st.session_state['meta']['indices'].replace(' ','_').lower()}_{st.session_state['meta']['timeframe'].lower()}_{st.session_state['meta']['period'].lower()}_momentum.csv",
        mime="text/csv"
    )


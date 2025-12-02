import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# =============== EXACT UI FEEL ===============
st.set_page_config(page_title="Nifty Total Market Momentum", layout="wide")
# No title/caption—keep the canvas clean like the Tkinter window.
# Add minimal CSS to mimic fonts, sticky header, and compact spacing.
st.markdown("""
<style>
/* overall font like Tkinter example */
html, body, [class^="css"]  {
  font-family: "Segoe UI", system-ui, -apple-system, Arial, sans-serif;
  font-size: 14px;
}
/* tighten default block spacing */
section.main > div { padding-top: 8px; }
.block-container { padding-top: 8px; padding-bottom: 8px; }
/* toolbar buttons */
div.stButton>button {
  padding: 6px 14px;
  border-radius: 6px;
}
.stSelectbox, .stTextInput { font-size: 14px; }
.table-wrap {
  max-height: 78vh;
  overflow: auto;
  border: 1px solid #cbd5e1;
  border-radius: 8px;
}
.table-wrap table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
}
.table-wrap th {
  position: sticky; top: 0;
  background: #ececec;
  padding: 8px 10px;
  font-weight: 700;
  border-bottom: 2px solid #cbd5e1;
  white-space: nowrap;
}
.table-wrap td {
  padding: 6px 10px;
  border-bottom: 1px solid #e5e7eb;
  white-space: nowrap;
}
a.name-link { text-decoration: none; }
a.name-link:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# ================== CONFIG ==================
# Benchmarks with alias candidates (first working symbol wins)
BENCHMARKS: Dict[str, list] = {
    "NIFTY 50": ["^NSEI"],
    "Nifty 200": ["^CNX200", "^NSE200", "^NSEI"],
    "Nifty 500": ["^CRSLDX", "^CNX500", "^NSE500", "^NSEI"],
    "Nifty Midcap 150": ["^NIFTYMIDCAP150.NS", "^NSEMDCP150", "^NSEI"],
    "Nifty Smallcap 250": ["^NIFTYSMLCAP250.NS", "^NSESMLCAP250", "^NSEI"],
}

MOMENTUM_YEARS = 2
RS_LOOKBACK_DAYS = 252
JDK_WINDOW = 21

# ================== CSV DISCOVERY ==================
def discover_universe_csvs() -> Dict[str, Path]:
    root = Path(__file__).resolve().parent
    ticker_dir = root / "ticker"
    files = {}
    if ticker_dir.exists():
        for p in sorted(ticker_dir.glob("*.csv")):
            canonical = {
                "nifty50": "Nifty 50",
                "nifty200": "Nifty 200",
                "nifty500": "Nifty 500",
                "niftyindices": "Nifty Indices",
                "niftymidcap150": "Nifty Midcap 150",
                "niftysmallcap250": "Nifty Smallcap 250",
                "niftymidsmallcap400": "Nifty Mid+Small 400",
                "niftytotalmarket": "Nifty Total Market",
            }
            key = canonical.get(p.stem.lower(), p.stem.replace("_", " ").title())
            files[key] = p
    return files

# ================== HELPERS ==================
def tv_symbol_from_yf(symbol: str) -> str:
    s = symbol.strip().upper()
    return "NSE:" + s[:-3] if s.endswith(".NS") else "NSE:" + s

def tradingview_chart_url(symbol: str) -> str:
    return f"https://in.tradingview.com/chart/?symbol={tv_symbol_from_yf(symbol)}"

def _pick_close(df, symbol: str) -> pd.Series:
    if isinstance(df, pd.Series):
        return df.dropna()
    if isinstance(df, pd.DataFrame):
        if isinstance(df.columns, pd.MultiIndex):
            for lvl in ("Adj Close", "Close"):
                if (symbol, lvl) in df.columns:
                    return df[(symbol, lvl)].dropna()
        else:
            for col in ("Adj Close", "Close"):
                if col in df.columns:
                    return df[col].dropna()
    return pd.Series(dtype=float)

def jdk_components(price: pd.Series, bench: pd.Series, win: int = JDK_WINDOW):
    df = pd.concat([price.rename("p"), bench.rename("b")], axis=1).dropna()
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
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

def perf_quadrant(x: float, y: float) -> str:
    if x >= 100 and y >= 100: return "Leading"
    if x < 100 and y >= 100:  return "Improving"
    if x < 100 and y < 100:   return "Lagging"
    return "Weakening"

def analyze_momentum(adj: pd.Series) -> dict | None:
    if adj is None or adj.empty or len(adj) < 252:
        return None
    ema100 = adj.ewm(span=100, adjust=False).mean()
    try:
        one_year_return = (adj.iloc[-1] / adj.iloc[-252] - 1.0) * 100.0
    except Exception:
        return None
    high_52w = adj.iloc[-252:].max()
    within_20pct_high = adj.iloc[-1] >= high_52w * 0.8
    if len(adj) < 126:
        return None
    six_month = adj.iloc[-126:]
    up_days_pct = (six_month.pct_change() > 0).sum() / len(six_month) * 100.0
    if (adj.iloc[-1] >= ema100.iloc[-1] and one_year_return >= 6.5 and
        within_20pct_high and up_days_pct > 45.0):
        try:
            r6 = (adj.iloc[-1] / adj.iloc[-126] - 1.0) * 100.0
            r3 = (adj.iloc[-1] / adj.iloc[-63]  - 1.0) * 100.0
            r1 = (adj.iloc[-1] / adj.iloc[-21]  - 1.0) * 100.0
        except Exception:
            return None
        return {"Return_6M": r6, "Return_3M": r3, "Return_1M": r1}
    return None

@st.cache_data(show_spinner=False)
def yf_download_cached(tickers: List[str], period: str = "2y", interval: str = "1d"):
    return yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )

def resolve_benchmark_symbol(preferred_key: str) -> str:
    for sym in BENCHMARKS.get(preferred_key, []):
        try:
            probe = yf.download(sym, period="3mo", interval="1d", progress=False, auto_adjust=True)
            s = _pick_close(probe, sym).dropna()
            if not s.empty:
                return sym
        except Exception:
            pass
    return "^NSEI"

def build_table_dataframe(benchmark_symbol: str, universe_df: pd.DataFrame) -> pd.DataFrame:
    tickers = universe_df["Symbol"].tolist()
    raw = yf_download_cached(tickers + [benchmark_symbol], period=f"{MOMENTUM_YEARS}y", interval="1d")

    bench = _pick_close(raw, benchmark_symbol).dropna()
    if bench.empty:
        raise RuntimeError(f"Benchmark {benchmark_symbol} series empty")

    cutoff = bench.index.max() - pd.Timedelta(days=RS_LOOKBACK_DAYS + 5)
    bench_rs = bench.loc[bench.index >= cutoff].copy()

    rows = []
    for _, rec in universe_df.iterrows():
        sym = rec.Symbol
        name = rec.Name
        industry = rec.Industry

        s = _pick_close(raw, sym).dropna()
        if s.empty:
            continue

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
        status = perf_quadrant(rr_last, mm_last)

        rows.append({
            "Name": name,
            "Industry": industry,
            "Return_6M": mom["Return_6M"],
            "Return_3M": mom["Return_3M"],
            "Return_1M": mom["Return_1M"],
            "RS-Ratio": rr_last,
            "RS-Momentum": mm_last,
            "Performance": status,
            "Final_Rank": None,
            "Position": None,
            "Symbol": sym,
        })

    if not rows:
        raise RuntimeError("No tickers passed the filters.")

    df = pd.DataFrame(rows)
    for col in ("Return_6M", "Return_3M", "Return_1M"):
        df[col] = df[col].round(1)
    df["RS-Ratio"] = df["RS-Ratio"].round(2)
    df["RS-Momentum"] = df["RS-Momentum"].round(2)

    df["Rank_6M"] = df["Return_6M"].rank(ascending=False, method="min")
    df["Rank_3M"] = df["Return_3M"].rank(ascending=False, method="min")
    df["Rank_1M"] = df["Return_1M"].rank(ascending=False, method="min")
    df["Final_Rank"] = df["Rank_6M"] + df["Rank_3M"] + df["Rank_1M"]
    df = df.sort_values("Final_Rank").reset_index(drop=True)
    df["Position"] = np.arange(1, len(df) + 1)
    df.insert(0, "S.No", np.arange(1, len(df) + 1))

    order = [
        "S.No", "Name", "Industry",
        "Return_6M", "Rank_6M",
        "Return_3M", "Rank_3M",
        "Return_1M", "Rank_1M",
        "RS-Ratio", "RS-Momentum", "Performance",
        "Final_Rank", "Position", "Symbol"
    ]
    return df[order]

def bg_for_serial(sno: int) -> str:
    if sno <= 30: return "#dff5df"   # light green
    if sno <= 60: return "#fff6b3"   # light yellow
    if sno <= 90: return "#dfe9ff"   # light blue
    return "#f7d6d6"                 # light red

def df_to_html_table(df: pd.DataFrame) -> str:
    headers = [
        "S.No", "Name", "Industry",
        "Return_6M", "Rank_6M",
        "Return_3M", "Rank_3M",
        "Return_1M", "Rank_1M",
        "RS-Ratio", "RS-Momentum", "Performance",
        "Final_Rank", "Position"
    ]
    rows_html = []
    for _, r in df.iterrows():
        sno = int(r["S.No"])
        bg = bg_for_serial(sno)
        name = str(r["Name"])
        sym = str(r["Symbol"]).strip()
        url = tradingview_chart_url(sym) if sym else "#"
        vals = [
            str(sno),
            f'<a class="name-link" href="{url}" target="_blank">{name}</a>',
            str(r["Industry"]),
            f'{r["Return_6M"]:.1f}',
            f'{r["Rank_6M"]:.0f}',
            f'{r["Return_3M"]:.1f}',
            f'{r["Rank_3M"]:.0f}',
            f'{r["Return_1M"]:.1f}',
            f'{r["Rank_1M"]:.0f}',
            f'{r["RS-Ratio"]:.2f}',
            f'{r["RS-Momentum"]:.2f}',
            str(r["Performance"]),
            f'{r["Final_Rank"]:.0f}',
            f'{r["Position"]:.0f}',
        ]
        tds = "".join(
            f'<td style="text-align:{ "left" if i in (1,2) else "center"};">{v}</td>'
            for i, v in enumerate(vals)
        )
        rows_html.append(f'<tr style="background:{bg};">{tds}</tr>')

    ths = "".join(
        f'<th style="text-align:{ "left" if h in ("Name","Industry") else "center"};">{h}</th>'
        for h in headers
    )
    return f"""
    <div class="table-wrap">
      <table>
        <thead><tr>{ths}</tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
      </table>
    </div>
    """

# ================== UI (Toolbar + Table) ==================
csv_map = discover_universe_csvs()
if not csv_map:
    st.error("No CSVs found under ./ticker/*.csv")
    st.stop()

toolbar_cols = st.columns([1.1, 1.1, 0.8, 0.8, 6])  # spacing like Tkinter top bar
with toolbar_cols[0]:
    bench_key = st.selectbox("Benchmark:", list(BENCHMARKS.keys()), index=list(BENCHMARKS.keys()).index("Nifty 500") if "Nifty 500" in BENCHMARKS else 0)
with toolbar_cols[1]:
    options = list(csv_map.keys())
    default_idx = options.index("Nifty 200") if "Nifty 200" in options else 0
    uni_key = st.selectbox("Universe:", options, index=default_idx)
with toolbar_cols[2]:
    load_click = st.button("Load / Refresh")
with toolbar_cols[3]:
    export_placeholder = st.empty()

info_placeholder = st.empty()
table_placeholder = st.empty()

if load_click or "last_df" not in st.session_state:
    try:
        bench_symbol = resolve_benchmark_symbol(bench_key)
        uni_df = pd.read_csv(csv_map[uni_key])
        # normalize columns
        cols = {c.strip().lower(): c for c in uni_df.columns}
        uni_df = uni_df[[cols["symbol"], cols["company name"], cols["industry"]]].copy()
        uni_df.columns = ["Symbol", "Name", "Industry"]
        df = build_table_dataframe(bench_symbol, uni_df)
        st.session_state["last_df"] = df.copy()
        st.session_state["meta"] = {"bench_key": bench_key, "bench_symbol": bench_symbol, "uni_key": uni_key}
        info_placeholder.write(f"Rows: **{len(df)}**")
    except Exception as e:
        info_placeholder.error(f"Error: {e}")
        st.stop()

if "last_df" in st.session_state:
    df = st.session_state["last_df"]
    table_placeholder.markdown(df_to_html_table(df), unsafe_allow_html=True)

    # Export button (CSV) like original
    export_df = df.drop(columns=["Symbol"]).copy()
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    export_placeholder.download_button(
        "Export CSV",
        data=csv_bytes,
        file_name=f"{st.session_state['meta']['uni_key'].replace(' ','_').lower()}_momentum.csv",
        mime="text/csv",
    )

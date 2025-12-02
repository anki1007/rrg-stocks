import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st

# ================== CONFIG ==================
# Benchmarks are stable; keep explicit mapping
BENCHMARKS: Dict[str, str] = {
    "NIFTY 50": "^NSEI",
    "Nifty 200": "^CNX200",
    "Nifty 500": "^CRSLDX",
    "Nifty Midcap 150": "^NIFTYMIDCAP150.NS",
    "Nifty Smallcap 250": "^NIFTYSMLCAP250.NS",
}

MOMENTUM_YEARS = 2
RS_LOOKBACK_DAYS = 252
JDK_WINDOW = 21

# ================== CSV DISCOVERY ==================
def discover_universe_csvs() -> Dict[str, Path]:
    """Find all *.csv under repo-relative 'ticker' folder and build a nice label -> path map."""
    root = Path(__file__).resolve().parent
    ticker_dir = root / "ticker"
    files = {}
    if ticker_dir.exists():
        for p in sorted(ticker_dir.glob("*.csv")):
            # Pretty label from filename
            label = p.stem.replace("_", " ").title()
            # Preserve original Nifty names for your set
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
            key = canonical.get(p.stem.lower(), label)
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

def load_universe_from_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in df.columns}
    need = ["symbol", "company name", "industry"]
    for n in need:
        if n not in cols:
            raise ValueError(f"{path.name} must include columns: {need}. Missing: {n}")
    df = df[[cols["symbol"], cols["company name"], cols["industry"]]].copy()
    df.columns = ["Symbol", "Name", "Industry"]
    df = df.dropna(subset=["Symbol"])
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Industry"] = df["Industry"].astype(str).str.strip()
    df = df[df["Symbol"] != ""].drop_duplicates(subset=["Symbol"])
    return df

# ================== FETCH ==================
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

def resample_weekly_last(series: pd.Series) -> pd.Series:
    return series.resample("W-FRI").last().dropna()

def build_table_dataframe(benchmark: str, universe_df: pd.DataFrame, price_tf: str) -> pd.DataFrame:
    tickers = universe_df["Symbol"].tolist()
    raw = yf_download_cached(tickers + [benchmark], period=f"{MOMENTUM_YEARS}y", interval="1d")

    bench = _pick_close(raw, benchmark).dropna()
    if bench.empty:
        raise RuntimeError(f"Benchmark {benchmark} series empty")

    if price_tf == "Weekly":
        bench = resample_weekly_last(bench)

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
        if price_tf == "Weekly":
            s = resample_weekly_last(s)

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

# ================== RENDER ==================
def bg_for_serial(sno: int) -> str:
    if sno <= 30: return "#dff5df"
    if sno <= 60: return "#fff6b3"
    if sno <= 90: return "#dfe9ff"
    return "#f7d6d6"

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
            f'<a href="{url}" target="_blank">{name}</a>',
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
            f'<td style="padding:6px 10px; text-align:{ "left" if i in (1,2) else "center"};">{v}</td>'
            for i, v in enumerate(vals)
        )
        rows_html.append(f'<tr style="background:{bg}; border-bottom:1px solid #e5e7eb;">{tds}</tr>')

    ths = "".join(
        f'<th style="position:sticky; top:0; background:#ececec; padding:8px 10px; '
        f'font-weight:700; text-align:{ "left" if h in ("Name","Industry") else "center"}; '
        f'border-bottom:2px solid #cbd5e1;">{h}</th>'
        for h in headers
    )
    return f"""
    <div style="max-height:70vh; overflow:auto; border:1px solid #33415533; border-radius:12px;">
      <table style="width:100%; border-collapse:separate; border-spacing:0; font-family:Segoe UI,system-ui,-apple-system,Arial; font-size:14px;">
        <thead><tr>{ths}</tr></thead>
        <tbody>{''.join(rows_html)}</tbody>
      </table>
    </div>
    """

# ================== APP ==================
st.set_page_config(page_title="Relative Rotation Graph (RRG) — Momentum Screen", layout="wide")
st.title("Relative Rotation Graph (RRG) — Momentum Screen")
st.caption("CSV path fixed: the app reads CSVs from the repo’s **ticker/** folder. Hyperlink is on the **Name** column.")

with st.sidebar:
    st.subheader("⚙️ Settings")
    csv_map = discover_universe_csvs()
    if not csv_map:
        st.error("No CSVs found under ./ticker/*.csv. Please add files like ticker/nifty200.csv.")
        st.stop()

    # Prefer Nifty Total Market if present, else first
    options = list(csv_map.keys())
    default_idx = options.index("Nifty Total Market") if "Nifty Total Market" in options else 0
    uni_key = st.selectbox("Universe CSV", options, index=default_idx)

    bench_key = st.selectbox("Benchmark", list(BENCHMARKS.keys()),
                             index=(list(BENCHMARKS.keys()).index("Nifty 500")
                                    if "Nifty 500" in BENCHMARKS else 0))
    price_tf = st.radio("Price timeframe", ["Daily", "Weekly"], horizontal=True)
    st.markdown("---")
    run_btn = st.button("🔄 Load / Refresh", use_container_width=True)

info = st.empty()
table_slot = st.empty()
download_slot = st.empty()

if run_btn or "last_df" not in st.session_state:
    try:
        info.info(f"Loading universe: {uni_key} …")
        uni = load_universe_from_csv(csv_map[uni_key])
        bench = BENCHMARKS[bench_key]
        df = build_table_dataframe(bench, uni, price_tf)
        st.session_state["last_df"] = df.copy()
        st.session_state["last_meta"] = {"universe": uni_key, "benchmark": bench_key, "price_tf": price_tf}
        info.success(f"Loaded {len(df)} rows • Universe: {uni_key} • Benchmark: {bench_key} • {price_tf}")
    except Exception as e:
        info.error(f"Error: {e}")
        st.stop()

if "last_df" in st.session_state:
    df = st.session_state["last_df"]
    with st.expander(f"📊 Momentum Table — {len(df)} rows (collapse/expand)", expanded=True):
        st.markdown(df_to_html_table(df), unsafe_allow_html=True)

    export_df = df.drop(columns=["Symbol"]).copy()
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    download_slot.download_button(
        "⬇️ Export CSV",
        data=csv_bytes,
        file_name=f"{st.session_state['last_meta']['universe'].replace(' ','_').lower()}_momentum.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown(
        """
        <div style="display:flex; gap:14px; align-items:center; font-size:13px; opacity:.85;">
          <span>Row color bands: </span>
          <span style="background:#dff5df; padding:4px 10px; border-radius:6px;">Top 1–30</span>
          <span style="background:#fff6b3; padding:4px 10px; border-radius:6px;">31–60</span>
          <span style="background:#dfe9ff; padding:4px 10px; border-radius:6px;">61–90</span>
          <span style="background:#f7d6d6; padding:4px 10px; border-radius:6px;">91+</span>
        </div>
        """,
        unsafe_allow_html=True
    )

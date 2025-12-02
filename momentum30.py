import os
import io
import webbrowser
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
from datetime import timedelta

# ================== BECNHMARK CONFIG ==================
BENCHMARKS = {
    "NIFTY 50": "^NSEI",
    "Nifty 200": "^CNX200",
    "Nifty 500": "^CRSLDX",
    "Nifty Midcap 150": "^NIFTYMIDCAP150.NS",
    "Nifty Smallcap 250": "^NIFTYSMLCAP250.NS",
}

# CSVs hosted on GitHub (Indices Universe)
GITHUB_BASE = "https://raw.githubusercontent.com/anki1007/rrg-stocks/main/ticker/"
CSV_FILES = {
    "Nifty 200":           GITHUB_BASE + "nifty200.csv",
    "Nifty 500":           GITHUB_BASE + "nifty500.csv",
    "Nifty Midcap 150":    GITHUB_BASE + "niftymidcap150.csv",
    "Nifty Mid Small 400": GITHUB_BASE + "niftymidsmallcap400.csv",
    "Nifty Smallcap 250":  GITHUB_BASE + "niftysmallcap250.csv",
    "Nifty Total Market":  GITHUB_BASE + "niftytotalmarket.csv",
}

# Core logic parameters (kept)
RS_LOOKBACK_DAYS = 252    # ~1y window for RS/JdK
JDK_WINDOW = 21           # JdK standard window

# ================== HELPERS (same logic) ==================
def tv_symbol_from_yf(symbol: str) -> str:
    s = symbol.strip().upper()
    return "NSE:" + s[:-3] if s.endswith(".NS") else "NSE:" + s

def tradingview_chart_url(symbol: str) -> str:
    return f"https://in.tradingview.com/chart/?symbol={tv_symbol_from_yf(symbol)}"

def _pick_close(df, symbol: str) -> pd.Series:
    """Safely pick Close/Adj Close for ticker from yfinance output."""
    if isinstance(df, pd.Series):
        return df.dropna()

    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)

    if isinstance(df.columns, pd.MultiIndex):
        for lvl in ("Close", "Adj Close"):
            col = (symbol, lvl)
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").dropna()
        return pd.Series(dtype=float)
    else:
        for col in ("Close", "Adj Close"):
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce").dropna()
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
    """Filter + compute 6M/3M/1M returns (unchanged)."""
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
def load_universe_from_csv(url: str) -> pd.DataFrame:
    df = pd.read_csv(url)
    cols = {c.strip().lower(): c for c in df.columns}
    need = ["symbol", "company name", "industry"]
    for n in need:
        if n not in cols:
            raise ValueError(f"CSV must include columns: {need}. Missing: {n}")
    df = df[[cols["symbol"], cols["company name"], cols["industry"]]].copy()
    df.columns = ["Symbol", "Name", "Industry"]
    df = df.dropna(subset=["Symbol"])
    df["Symbol"] = df["Symbol"].astype(str).str.strip()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["Industry"] = df["Industry"].astype(str).str.strip()
    df = df[df["Symbol"] != ""].drop_duplicates(subset=["Symbol"])
    return df

@st.cache_data(show_spinner=True)
def fetch_prices(tickers: list[str], benchmark: str, period: str, interval: str):
    # Use yfinance to grab all tickers + benchmark together
    raw = yf.download(
        tickers + [benchmark],
        period=period,            # e.g., "2y"
        interval=interval,        # e.g., "1d"
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
    )
    return raw

def row_bg_for_serial(sno: int) -> str:
    # Do not change color criteria
    if sno <= 30: return "#dff5df"  # light green
    if sno <= 60: return "#fff6b3"  # light yellow
    if sno <= 90: return "#dfe9ff"  # light blue
    return "#f7d6d6"                # light red

def build_table_dataframe(raw, benchmark: str, universe_df: pd.DataFrame) -> pd.DataFrame:
    bench = _pick_close(raw, benchmark).dropna()
    if bench.empty:
        raise RuntimeError(f"Benchmark {benchmark} series empty")

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
            "Symbol": sym,  # kept for link; hidden in display/export if needed
            "Chart": tradingview_chart_url(sym),
        })

    if not rows:
        raise RuntimeError("No tickers passed the filters.")

    df = pd.DataFrame(rows)

    # rounding (unchanged)
    for col in ("Return_6M", "Return_3M", "Return_1M"):
        df[col] = pd.to_numeric(df[col], errors="coerce").round(1)
    df["RS-Ratio"] = pd.to_numeric(df["RS-Ratio"], errors="coerce").round(2)
    df["RS-Momentum"] = pd.to_numeric(df["RS-Momentum"], errors="coerce").round(2)

    # ranks and position (unchanged)
    df["Rank_6M"] = df["Return_6M"].rank(ascending=False, method="min")
    df["Rank_3M"] = df["Return_3M"].rank(ascending=False, method="min")
    df["Rank_1M"] = df["Return_1M"].rank(ascending=False, method="min")
    df["Final_Rank"] = df["Rank_6M"] + df["Rank_3M"] + df["Rank_1M"]
    df = df.sort_values("Final_Rank").reset_index(drop=True)
    df["Position"] = np.arange(1, len(df) + 1)
    df.insert(0, "S.No", np.arange(1, len(df) + 1))

    # Final order (keep Symbol only for link, we can hide from styled output if you prefer)
    order = ["S.No", "Name", "Industry",
             "Return_6M", "Rank_6M",
             "Return_3M", "Rank_3M",
             "Return_1M", "Rank_1M",
             "RS-Ratio", "RS-Momentum", "Performance",
             "Final_Rank", "Position", "Chart", "Symbol"]
    return df[order]

def style_rows(df: pd.DataFrame):
    """Apply row background based on S.No (keep exact color criteria)."""
    def _row_style(_row):
        sno = int(_row["S.No"])
        bg = row_bg_for_serial(sno)
        return [f"background-color: {bg}"] * len(df.columns)

    styler = df.style.apply(lambda r: _row_style(r), axis=1)
    # Left-align Name & Industry like your Tk app
    styler = styler.set_properties(subset=["Name", "Industry"], **{"text-align": "left"})
    # Center-align numeric-ish columns to match original
    num_cols = [c for c in df.columns if c not in ("Name", "Industry", "Chart", "Symbol")]
    styler = styler.set_properties(subset=num_cols, **{"text-align": "center"})
    return styler

# ================== STREAMLIT UI ==================
st.set_page_config(page_title="Momentum Screener", layout="wide")
st.title("Momentum Screener")

# ---- Sidebar controls (left dashboard) ----
st.sidebar.header("Controls")

indices_universe = st.sidebar.selectbox("a) Indices Universe", list(CSV_FILES.keys()), index=4)
benchmark_key = st.sidebar.selectbox("b) Benchmark", list(BENCHMARKS.keys()), index=1)

# Timeframe (yfinance interval). Core logic uses daily; offering choices but 1d is recommended.
timeframe = st.sidebar.selectbox("c) Timeframe", ["1d", "1wk", "1mo"], index=0)

# Period for yfinance download
period = st.sidebar.selectbox(
    "d) Period",
    ["1y", "2y", "3y", "5y"],
    index=1
)

col_btn1, col_btn2 = st.sidebar.columns(2)
do_load = col_btn1.button("e) Load / Refresh", use_container_width=True)
# export button will appear below table when data is ready

# ---- Status / Info ----
st.caption(
    "Tip: Results depend on the CSV universe and available price history. "
    "If few rows appear, try a longer **Period** (e.g., 2y+) and **Timeframe** = 1d."
)

# ---- Run build on click or when first landing (auto-run once) ----
if "ran_once" not in st.session_state:
    st.session_state.ran_once = True
    do_load = True

if do_load:
    try:
        uni_url = CSV_FILES[indices_universe]
        universe_df = load_universe_from_csv(uni_url)

        benchmark = BENCHMARKS[benchmark_key]
        tickers = universe_df["Symbol"].tolist()

        with st.spinner("Fetching prices…"):
            raw = fetch_prices(tickers, benchmark, period=period, interval=timeframe)

        df = build_table_dataframe(raw, benchmark, universe_df)

        # Build a display dataframe that hides Symbol but keeps Chart link
        display_cols = [
            "S.No", "Name", "Industry",
            "Return_6M", "Rank_6M",
            "Return_3M", "Rank_3M",
            "Return_1M", "Rank_1M",
            "RS-Ratio", "RS-Momentum", "Performance",
            "Final_Rank", "Position", "Chart"
        ]
        display_df = df[display_cols].copy()

        st.subheader("Screened Momentum Table")
        # Make Chart clickable via column_config LinkColumn
        st.dataframe(
            style_rows(display_df),
            use_container_width=True,
            column_config={
                "Chart": st.column_config.LinkColumn("Chart", help="Open in TradingView"),
            },
            height=600,
        )

        # Summary bar
        st.info(
            f"Rows: **{len(df)}**  |  Universe: **{indices_universe}**  |  "
            f"Benchmark: **{benchmark_key}**  |  Timeframe: **{timeframe}**  |  Period: **{period}**"
        )

        # ---- Export CSV (f) ----
        csv_bytes = df.drop(columns=["Symbol"]).to_csv(index=False).encode("utf-8")
        st.download_button(
            label="f) Export CSV",
            data=csv_bytes,
            file_name=f"{indices_universe.replace(' ', '').lower()}_momentum.csv",
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as e:
        st.error(str(e))

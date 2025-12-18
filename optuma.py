import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import requests
from datetime import datetime

# ============================================================================
# CONFIG & SETUP
# ============================================================================
st.set_page_config(
    layout="wide",
    page_title="RRG Dashboard - Multi-Index Analysis",
    initial_sidebar_state="expanded",
)

# Minimal theme tweaks (can be removed if you use .streamlit/config.toml)
st.markdown(
    """
    <style>
    .stMetric-value { font-size: 20px !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Correct Yahoo Finance benchmark symbols
BENCHMARKS = {
    "NIFTY 50": "NSEI",
    "NIFTY 200": "CNX200",
    "NIFTY 500": "CRSLDX",
}

# Timeframes for yfinance
TIMEFRAMES = {
    "5 min close": ("5m", "60d"),
    "15 min close": ("15m", "60d"),
    "30 min close": ("30m", "60d"),
    "1 hr close": ("60m", "90d"),
    "Daily": ("1d", "5y"),
    "Weekly": ("1wk", "10y"),
    "Monthly": ("1mo", "20y"),
}

# Period mapping to bars
PERIOD_MAP = {
    "6M": 126,
    "1Y": 252,
    "2Y": 504,
    "3Y": 756,
    "5Y": 1260,
    "10Y": 2520,
}

# RRG Config
WINDOW = 14
TAIL = 8  # length of historical trail

# Quadrant colors
QUADRANT_COLORS = {
    "Improving": "#3b82f6",
    "Leading": "#22c55e",
    "Weakening": "#facc15",
    "Lagging": "#ef4444",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=600)
def list_csv_from_github():
    """Fetch CSV filenames from GitHub repository."""
    url = "https://api.github.com/repos/anki1007/rrg-stocks/contents/ticker"
    try:
        response = requests.get(url)
        files = [
            f["name"].replace(".csv", "").upper()
            for f in response.json()
            if f["name"].endswith(".csv")
        ]
        return sorted(files)
    except Exception as e:
        st.error(f"Error fetching CSV list: {e}")
        return []


@st.cache_data(ttl=600)
def load_universe(csv_name: str) -> pd.DataFrame:
    """Load stock universe from GitHub CSV."""
    url = f"https://github.com/anki1007/rrg-stocks/main/ticker/{csv_name.lower()}.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading {csv_name} universe: {e}")
        return pd.DataFrame()


def rrg_calc(px: pd.Series, bench: pd.Series):
    """Calculate RRG metrics: RS-Ratio and RS-Momentum."""
    df = pd.concat([px, bench], axis=1).dropna()
    if len(df) < WINDOW + 2:
        return None, None

    rs = 100 * (df.iloc[:, 0] / df.iloc[:, 1])
    rs_ratio = 100 * (rs - rs.rolling(WINDOW).mean()) / rs.rolling(WINDOW).std(ddof=0)

    roc = rs_ratio.pct_change() * 100
    rs_momentum = 101 * (roc - roc.rolling(WINDOW).mean()) / roc.rolling(WINDOW).std(
        ddof=0
    )

    return rs_ratio.dropna(), rs_momentum.dropna()


def quadrant(x: float, y: float) -> str:
    """Determine RRG quadrant based on RS-Ratio and RS-Momentum."""
    if x > 100 and y > 100:
        return "Leading"
    if x < 100 and y > 100:
        return "Improving"
    if x < 100 and y < 100:
        return "Weakening"
    return "Lagging"


def get_tv_link(sym: str) -> str:
    """Generate TradingView chart link."""
    return f"https://www.tradingview.com/chart/?symbol=NSE:{sym.replace('.NS', '')}"


def format_symbol(sym: str) -> str:
    """Remove .NS suffix from symbol."""
    return sym.replace(".NS", "")


def calculate_price_change(current_price: float, historical_price: float) -> float:
    """Calculate percentage price change."""
    if historical_price == 0:
        return 0.0
    return (current_price - historical_price) / historical_price * 100.0


def highlight_status(val: str) -> str:
    """Styler for Status column."""
    if val == "Leading":
        return "background-color: #22c55e; color: white"
    if val == "Improving":
        return "background-color: #3b82f6; color: white"
    if val == "Weakening":
        return "background-color: #facc15; color: black"
    if val == "Lagging":
        return "background-color: #ef4444; color: white"
    return ""


# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================
st.sidebar.markdown("### ⚙️ Controls")

csv_files = list_csv_from_github()
csv_selected = st.sidebar.selectbox(
    "📊 Indices", csv_files, help="Select stock universe from available CSVs"
)

bench_name = st.sidebar.selectbox(
    "🎯 Benchmark",
    list(BENCHMARKS.keys()),
    help="Select benchmark index for relative strength",
)

tf_name = st.sidebar.selectbox(
    "⏱️ Strength vs Timeframe",
    list(TIMEFRAMES.keys()),
    help="Select timeframe for RRG analysis",
)

period_name = st.sidebar.selectbox(
    "📅 Period", list(PERIOD_MAP.keys()), help="Select analysis period"
)

rank_by = st.sidebar.selectbox(
    "🏆 Rank by",
    ["RRG Power", "RS-Ratio", "RS-Momentum", "Price % Change", "Momentum Slope"],
    help="Select ranking metric",
)

st.sidebar.markdown("---")

# ============================================================================
# DATA LOADING & CALCULATION
# ============================================================================
try:
    interval, yf_period = TIMEFRAMES[tf_name]

    universe = load_universe(csv_selected)
    if universe.empty:
        st.error("Failed to load universe data")
        st.stop()

    symbols = universe["Symbol"].tolist()
    names_dict = dict(zip(universe["Symbol"], universe["Company Name"]))
    industries_dict = dict(zip(universe["Symbol"], universe["Industry"]))

    # Download data for all stocks + benchmark
    raw = yf.download(
        symbols + [BENCHMARKS[bench_name]],
        interval=interval,
        period=yf_period,
        auto_adjust=True,
        progress=False,
        threads=True,
    )

    bench = raw["Close"][BENCHMARKS[bench_name]]

    rows = []
    trails = {}  # symbol -> (rr_tail, mm_tail)

    for s in symbols:
        if s not in raw["Close"].columns:
            continue

        px = raw["Close"][s].dropna()
        if px.empty:
            continue

        rr, mm = rrg_calc(px, bench)
        if rr is None or mm is None:
            continue

        rr_tail = rr.iloc[-TAIL:]
        mm_tail = mm.iloc[-TAIL:]
        if len(rr_tail) < 3:
            continue

        slope = np.polyfit(range(len(mm_tail)), mm_tail.values, 1)[0]
        power = np.sqrt((rr_tail.iloc[-1] - 100) ** 2 + (mm_tail.iloc[-1] - 100) ** 2)

        idx_hist = max(0, len(px) - PERIOD_MAP[period_name])
        historical_price = px.iloc[idx_hist]
        current_price = px.iloc[-1]
        price_change = calculate_price_change(current_price, historical_price)

        status = quadrant(rr_tail.iloc[-1], mm_tail.iloc[-1])
        tv_link = get_tv_link(s)

        rows.append(
            {
                "Symbol": s,
                "Name": names_dict.get(s, s),
                "Industry": industries_dict.get(s, "N/A"),
                "Price": round(current_price, 2),
                "Change %": round(price_change, 2),
                "RS-Ratio": round(rr_tail.iloc[-1], 2),
                "RS-Momentum": round(mm_tail.iloc[-1], 2),
                "Momentum Slope": round(slope, 2),
                "RRG Power": round(power, 2),
                "Status": status,
                "TV Link": tv_link,
            }
        )

        trails[s] = (rr_tail, mm_tail)

    df = pd.DataFrame(rows)
    if df.empty:
        st.error("No data available. Try another combination.")
        st.stop()

    rank_col_map = {
        "RRG Power": "RRG Power",
        "RS-Ratio": "RS-Ratio",
        "RS-Momentum": "RS-Momentum",
        "Price % Change": "Change %",
        "Momentum Slope": "Momentum Slope",
    }
    rank_column = rank_col_map[rank_by]
    df["Rank"] = df[rank_column].rank(ascending=False, method="min").astype(int)
    df = df.sort_values("Rank")
    df["Sl No."] = range(1, len(df) + 1)

except Exception as e:
    st.error(f"Error in data processing: {str(e)}")
    st.stop()

# ============================================================================
# LAYOUT
# ============================================================================
col_left, col_main, col_right = st.columns([1, 3, 1])

# --------------------------------------------------------------------
# LEFT: Legend & stats
# --------------------------------------------------------------------
with col_left:
    st.markdown("### 📍 Legend")
    for status, color in QUADRANT_COLORS.items():
        st.markdown(
            f"""
            <div style="display:flex; align-items:center; margin:6px 0;">
              <div style="width:16px; height:16px; border-radius:50%;
                          background:{color}; margin-right:8px;"></div>
              <span style="font-size:13px;">{status}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### 📊 Stats")
    s1, s2 = st.columns(2)
    with s1:
        st.metric("Total Stocks", len(df))
        st.metric("Leading", int((df["Status"] == "Leading").sum()))
    with s2:
        st.metric("Improving", int((df["Status"] == "Improving").sum()))
        st.metric("Weakening", int((df["Status"] == "Weakening").sum()))
        st.metric("Lagging", int((df["Status"] == "Lagging").sum()))

# --------------------------------------------------------------------
# CENTER: RRG with trails & dark legend
# --------------------------------------------------------------------
with col_main:
    st.markdown("### Relative Rotation Graph")

    fig = go.Figure()

    x_min = df["RS-Ratio"].min() - 10
    x_max = df["RS-Ratio"].max() + 10
    y_min = df["RS-Momentum"].min() - 10
    y_max = df["RS-Momentum"].max() + 10

    # Quadrants
    fig.add_shape(
        type="rect",
        x0=100,
        y0=100,
        x1=x_max,
        y1=y_max,
        fillcolor="#22c55e",
        opacity=0.05,
        line_width=0,
        name="Leading",
    )
    fig.add_shape(
        type="rect",
        x0=x_min,
        y0=100,
        x1=100,
        y1=y_max,
        fillcolor="#3b82f6",
        opacity=0.05,
        line_width=0,
        name="Improving",
    )
    fig.add_shape(
        type="rect",
        x0=x_min,
        y0=y_min,
        x1=100,
        y1=100,
        fillcolor="#ef4444",
        opacity=0.05,
        line_width=0,
        name="Lagging",
    )
    fig.add_shape(
        type="rect",
        x0=100,
        y0=y_min,
        x1=x_max,
        y1=100,
        fillcolor="#facc15",
        opacity=0.05,
        line_width=0,
        name="Weakening",
    )

    # Axes cross
    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.6)
    fig.add_vline(x=100, line_dash="dash", line_color="gray", opacity=0.6)

    # Trails + markers
    for status in ["Leading", "Improving", "Weakening", "Lagging"]:
        df_status = df[df["Status"] == status]

        # Tails
        for _, row in df_status.iterrows():
            sym = row["Symbol"]
            rr_tail, mm_tail = trails.get(sym, (None, None))
            if rr_tail is None or mm_tail is None:
                continue
            fig.add_trace(
                go.Scatter(
                    x=rr_tail,
                    y=mm_tail,
                    mode="lines",
                    line=dict(color=QUADRANT_COLORS[status], width=1),
                    opacity=0.4,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Last-point markers
        fig.add_trace(
            go.Scatter(
                x=df_status["RS-Ratio"],
                y=df_status["RS-Momentum"],
                mode="markers+text",
                name=status,
                marker=dict(
                    size=10,
                    color=QUADRANT_COLORS[status],
                    line=dict(width=1, color="#111827"),
                ),
                text=[format_symbol(s) for s in df_status["Symbol"]],
                textposition="top center",
                textfont=dict(size=10, color="#111827"),
                customdata=np.stack(
                    [df_status["Symbol"], df_status["Name"]], axis=-1
                ),
                hovertemplate="<b>%{customdata[0]}</b><br>"
                "Company: %{customdata[1]}<br>"
                "RS-Ratio: %{x:.2f}<br>"
                "RS-Momentum: %{y:.2f}<br>"
                f"Status: {status}"
                "<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"{csv_selected} | {tf_name} | {period_name} | Benchmark: {bench_name}",
        title_font=dict(color="#111827", size=18),
        xaxis=dict(
            title="RS-Ratio (X-axis)",
            title_font=dict(color="#111827", size=12),
            tickfont=dict(color="#111827", size=11),
            zeroline=False,
        ),
        yaxis=dict(
            title="RS-Momentum (Y-axis)",
            title_font=dict(color="#111827", size=12),
            tickfont=dict(color="#111827", size=11),
            zeroline=False,
        ),
        legend=dict(
            title="Quadrants",
            title_font=dict(color="#111827", size=12),
            font=dict(color="#111827", size=11),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(15,23,42,0.25)",
            borderwidth=1,
            x=0.02,
            y=0.98,
        ),
        hovermode="closest",
        height=500,
        template="plotly_white",
        plot_bgcolor="#f9fafb",
    )

    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    with st.expander("Results Table (click to expand/collapse)", expanded=True):
        display_df = df[
            [
                "Sl No.",
                "Symbol",
                "Name",
                "Status",
                "Industry",
                "Price",
                "Change %",
                "RS-Ratio",
                "RS-Momentum",
                "Momentum Slope",
                "RRG Power",
                "Rank",
            ]
        ].copy()
        display_df["Symbol"] = display_df["Symbol"].apply(format_symbol)
        styled_df = display_df.style.applymap(highlight_status, subset=["Status"])
        st.dataframe(styled_df, use_container_width=True, height=400)

        csv_data = display_df.to_csv(index=False)
        st.download_button(
            label="Download Results CSV",
            data=csv_data,
            file_name=f"rrg_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# --------------------------------------------------------------------
# RIGHT: Top‑30 RRG Power by quadrant
# --------------------------------------------------------------------
with col_right:
    st.markdown("### 🏅 Top 30 RRG Power")

    df_top30 = df.nlargest(30, "RRG Power").copy()
    df_top30["Symbol_fmt"] = df_top30["Symbol"].apply(format_symbol)

    sections = [
        ("Leading", "🟢"),
        ("Improving", "🔵"),
        ("Weakening", "🟡"),
        ("Lagging", "🔴"),
    ]

    for status, icon in sections:
        subset = df_top30[df_top30["Status"] == status]
        if subset.empty:
            continue

        with st.expander(f"{icon} {status}", expanded=(status == "Leading")):
            for _, row in subset.iterrows():
                tv_link = row["TV Link"]
                color = QUADRANT_COLORS[status]
                st.markdown(
                    f"""
                    <div style="padding:6px; margin-bottom:4px;
                                background:rgba(15,23,42,0.35);
                                border-left:3px solid {color};
                                border-radius:4px;">
                      <small><b>#{int(row['Rank'])}</b></small><br>
                      <b style="color:{color};">{row['Symbol_fmt']}</b><br>
                      <small>{row['Industry'][:26]}</small><br>
                      <small style="color:{color};">
                        Power: {row['RRG Power']:.2f}
                      </small><br>
                      <small>
                        <a href="{tv_link}" target="_blank"
                           style="color:#60a5fa;text-decoration:none;">
                          Chart ↗
                        </a>
                      </small>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align:center; font-size:12px; color:#9ca3af;">
          <b>RRG Dashboard v2.0</b><br>
          Last Updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} IST<br>
          Data Source: Yahoo Finance | Benchmark: {bench_name}
        </div>
        """,
        unsafe_allow_html=True,
    )

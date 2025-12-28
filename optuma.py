import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import io
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    layout="wide",
    page_title="RRG Dashboard - Multi-Index Analysis",
    initial_sidebar_state="expanded",
)

# ============================================================================
# GLOBAL CSS (DARK THEME)
# ============================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;700;800&display=swap');

/* Design tokens */
:root {
  --app-font: 'Plus Jakarta Sans', system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
  --bg: #0b0e13;
  --bg-2: #10141b;
  --border: #1f2732;
  --border-soft: #1a2230;
  --text: #e6eaee;
  --text-dim: #b3bdc7;
  --accent: #7a5cff;
  --accent-2: #2bb0ff;
}

/* Global app */
html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: var(--app-font) !important;
}

/* Main container spacing */
.block-container {
  padding-top: 2.5rem;
  max-width: 100%;
  padding-left: 1.25rem;
  padding-right: 1.25rem;
}

/* Hero title style */
.hero-title {
  font-weight: 800;
  font-size: clamp(26px, 4.5vw, 40px);
  line-height: 1.05;
  margin: 4px 0 14px 0;
  background: linear-gradient(90deg, var(--accent-2), var(--accent) 60%);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  letter-spacing: .2px;
}

/* Sidebar – pro skin */
section[data-testid="stSidebar"] {
  background: var(--bg-2) !important;
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * {
  color: var(--text) !important;
}
section[data-testid="stSidebar"] label {
  font-weight: 700;
  color: var(--text-dim) !important;
}

/* Buttons (used also by sidebar buttons) */
.stButton button {
  background: linear-gradient(180deg, #1b2432, #131922);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 10px;
}
.stButton button:hover {
  filter: brightness(1.06);
}

/* Headings in main area */
h1, h2, h3, h4, h5, h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
.stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
  color: var(--text) !important;
}

/* Metrics */
[data-testid="stMetric"] {
  background: var(--bg-2);
  padding: 12px;
  border-radius: 8px;
  border: 1px solid var(--border);
}

/* Animation Play/Pause buttons (Plotly updatemenus look) */
.modebar-btn {
  font-family: var(--app-font);
}
</style>
""",
    unsafe_allow_html=True,
)

# Hero title
st.markdown(
    '<div class="hero-title">Relative Rotation Graphs – Dashboard</div>',
    unsafe_allow_html=True,
)

# ============================================================================
# CONFIGURATION
# ============================================================================
BENCHMARKS = {
    "NIFTY 50": "^NSEI",
    "NIFTY 200": "^CNX200",
    "NIFTY 500": "^CRSLDX",
}

TIMEFRAMES = {
    "5 min": ("5m", "60d"),
    "15 min": ("15m", "60d"),
    "30 min": ("30m", "60d"),
    "1 hr": ("60m", "90d"),
    "4 hr": ("240m", "120d"),
    "Daily": ("1d", "5y"),
    "Weekly": ("1wk", "10y"),
    "Monthly": ("1mo", "20y"),
}

PERIOD_MAP = {
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "2Y": 504,
    "3Y": 756,
    "5Y": 1260,
    "10Y": 2520,
}

# Quadrant colors
QUADRANT_COLORS = {
    "Leading": "#15803d",    # Dark green
    "Improving": "#7c3aed",  # Purple
    "Weakening": "#a16207",  # Amber
    "Lagging": "#dc2626",    # Red
}

QUADRANT_BG_COLORS = {
    "Leading": "rgba(187, 247, 208, 0.6)",
    "Improving": "rgba(233, 213, 255, 0.6)",
    "Weakening": "rgba(254, 249, 195, 0.6)",
    "Lagging": "rgba(254, 202, 202, 0.6)",
}

QUADRANT_LABEL_COLORS = {
    "Leading": "#0d4a1f",
    "Improving": "#5b21b6",
    "Weakening": "#713f12",
    "Lagging": "#991b1b",
}

WINDOW = 14
TAIL_LENGTH = 8

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
@st.cache_data(ttl=600)
def list_csv_from_github():
    """Fetch CSV filenames from GitHub repository (ticker folder)."""
    url = "https://api.github.com/repos/anki1007/rrg-stocks/contents/ticker"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            files = [
                f["name"].replace(".csv", "").upper()
                for f in data
                if isinstance(f, dict) and f.get("name", "").endswith(".csv")
            ]
            return sorted(files) if files else []
        return []
    except Exception:
        return []  # fail-safe


@st.cache_data(ttl=600)
def load_universe(csv_name):
    """Load stock universe from GitHub CSV."""
    url = f"https://raw.githubusercontent.com/anki1007/rrg-stocks/main/ticker/{csv_name.lower()}.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception:
        return pd.DataFrame()


def get_adjusted_close_date(timeframe):
    """Reference date for weekly / monthly timeframe."""
    today = datetime.now()
    if timeframe == "Weekly":
        days_since_friday = (today.weekday() - 4) % 7
        if days_since_friday == 0 and today.weekday() == 4:
            last_friday = today
        else:
            last_friday = today - timedelta(days=(days_since_friday or 7))
        return last_friday.strftime("%Y-%m-%d")
    elif timeframe == "Monthly":
        first_day_this_month = today.replace(day=1)
        last_day_prev_month = first_day_this_month - timedelta(days=1)
        return last_day_prev_month.strftime("%Y-%m-%d")
    return today.strftime("%Y-%m-%d")


def calculate_jdk_rrg(ticker_series, benchmark_series, window=WINDOW):
    """Compute JdK RS-Ratio, RS-Momentum, distance, heading, velocity."""
    aligned = pd.DataFrame({"ticker": ticker_series, "benchmark": benchmark_series}).dropna()
    if len(aligned) < window + 2:
        return None, None, None, None, None

    rs = 100 * (aligned["ticker"] / aligned["benchmark"])
    rs_mean = rs.rolling(window=window).mean()
    rs_std = rs.rolling(window=window).std(ddof=0)
    rs_ratio = 100 + (rs - rs_mean) / rs_std

    rsr_roc = 100 * (rs_ratio / rs_ratio.shift(1) - 1)
    rsm_mean = rsr_roc.rolling(window=window).mean()
    rsm_std = rsr_roc.rolling(window=window).std(ddof=0)
    rs_mom = 101 + (rsr_roc - rsm_mean) / rsm_std

    distance = np.sqrt((rs_ratio - 100) ** 2 + (rs_mom - 100) ** 2)
    heading = np.arctan2(rs_mom - 100, rs_ratio - 100) * 180 / np.pi
    heading = (heading + 360) % 360
    velocity = distance.diff().abs()

    n = min(len(rs_ratio), len(rs_mom), len(distance), len(heading), len(velocity))
    return (
        rs_ratio.iloc[-n:].reset_index(drop=True),
        rs_mom.iloc[-n:].reset_index(drop=True),
        distance.iloc[-n:].reset_index(drop=True),
        heading.iloc[-n:].reset_index(drop=True),
        velocity.iloc[-n:].reset_index(drop=True),
    )


def quadrant(x, y):
    if x > 100 and y > 100:
        return "Leading"
    if x < 100 and y > 100:
        return "Improving"
    if x < 100 and y < 100:
        return "Lagging"
    return "Weakening"


def get_heading_direction(h):
    if 22.5 <= h < 67.5:
        return "↗ NE"
    if 67.5 <= h < 112.5:
        return "↑ N"
    if 112.5 <= h < 157.5:
        return "↖ NW"
    if 157.5 <= h < 202.5:
        return "← W"
    if 202.5 <= h < 247.5:
        return "↙ SW"
    if 247.5 <= h < 292.5:
        return "↓ S"
    if 292.5 <= h < 337.5:
        return "↘ SE"
    return "→ E"


def get_tv_link(sym):
    clean = sym.replace(".NS", "")
    return f"https://www.tradingview.com/chart/?symbol=NSE:{clean}"


def format_symbol(sym):
    return sym.replace(".NS", "")


def calculate_price_change(current_price, historical_price):
    if historical_price == 0:
        return 0
    return (current_price - historical_price) / historical_price * 100.0


def select_graph_stocks(df, min_stocks=60):
    """Quadrant-balanced selection for plotting."""
    graph_idx = []
    for status in ["Leading", "Improving", "Weakening", "Lagging"]:
        df_q = df[df["Status"] == status].copy()
        if len(df_q) == 0:
            continue
        if len(df_q) < 10:
            graph_idx.extend(df_q.index.tolist())
        else:
            if status in ["Leading", "Improving"]:
                top = df_q.nlargest(10, "RRG Power")
            else:
                top = df_q.nsmallest(10, "RRG Power")
            graph_idx.extend(top.index.tolist())

    if len(graph_idx) < min_stocks:
        remaining = df.index.difference(graph_idx)
        need = min_stocks - len(graph_idx)
        extra = df.loc[remaining].nlargest(need, "RRG Power")
        graph_idx.extend(extra.index.tolist())

    return df.loc[graph_idx]


def smooth_spline_curve(x_points, y_points, points_per_segment=8):
    """Catmull-Rom spline for smooth trails."""
    if len(x_points) < 3:
        return np.array(x_points), np.array(y_points)

    x_points = np.array(x_points, dtype=float)
    y_points = np.array(y_points, dtype=float)

    def seg(p0, p1, p2, p3, n_pts):
        t = np.linspace(0, 1, n_pts, endpoint=False)
        t2 = t * t
        t3 = t2 * t
        x = 0.5 * (
            (2 * p1[0])
            + (-p0[0] + p2[0]) * t
            + (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2
            + (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3
        )
        y = 0.5 * (
            (2 * p1[1])
            + (-p0[1] + p2[1]) * t
            + (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2
            + (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3
        )
        return x, y

    pts = np.column_stack([x_points, y_points])
    padded = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
    xs, ys = [], []
    for i in range(len(pts) - 1):
        sx, sy = seg(padded[i], padded[i + 1], padded[i + 2], padded[i + 3], points_per_segment)
        xs.extend(sx)
        ys.extend(sy)
    xs.append(x_points[-1])
    ys.append(y_points[-1])
    return np.array(xs), np.array(ys)


# ============================================================================
# SESSION STATE INIT
# ============================================================================
if "load_clicked" not in st.session_state:
    st.session_state.load_clicked = False
if "df_cache" not in st.session_state:
    st.session_state.df_cache = None
if "rs_history_cache" not in st.session_state:
    st.session_state.rs_history_cache = {}

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.markdown("### ⚙️ Controls")

csv_files = list_csv_from_github()
if not csv_files:
    st.sidebar.warning("⚠️ Unable to fetch indices from GitHub. Check connection.")
    csv_files = ["NIFTY200"]

default_csv_index = 0
for i, name in enumerate(csv_files):
    if "NIFTY200" in name.upper():
        default_csv_index = i
        break

csv_selected = st.sidebar.selectbox("Indices", csv_files, index=default_csv_index)

bench_list = list(BENCHMARKS.keys())
bench_name = st.sidebar.selectbox("Benchmark", bench_list, index=2)

tf_name = st.sidebar.selectbox("Strength vs Timeframe", list(TIMEFRAMES.keys()))

TF_DEFAULT_PERIOD = {
    "5 min": "3M",
    "15 min": "3M",
    "30 min": "3M",
    "1 hr": "3M",
    "4 hr": "3M",
    "Daily": "6M",
    "Weekly": "1Y",
    "Monthly": "1Y",
}

prev_tf = st.session_state.get("_tf_prev")
if (prev_tf is None) or (prev_tf != tf_name):
    st.session_state["period_select"] = TF_DEFAULT_PERIOD.get(tf_name, "6M")
    st.session_state["_tf_prev"] = tf_name

period_name = st.sidebar.selectbox("Period", list(PERIOD_MAP.keys()), key="period_select")

rank_by = st.sidebar.selectbox(
    "Rank by",
    ["RRG Power", "RS-Ratio", "RS-Momentum", "Distance", "Price % Change"],
    index=0,
)

st.sidebar.markdown("---")
top_n = st.sidebar.slider("Show Top N", min_value=5, max_value=100, value=50)
st.sidebar.markdown("---")
trail_length = st.sidebar.slider("Trail Length", min_value=1, max_value=14, value=5)
show_labels = st.sidebar.checkbox("Show Labels on Chart", value=True)
label_top_n = st.sidebar.slider(
    "Label Top N (by distance)", min_value=3, max_value=50, value=15, disabled=not show_labels
)
st.sidebar.markdown("---")
export_csv = st.sidebar.checkbox("Export CSV", value=True)
st.sidebar.markdown("---")

if st.sidebar.button("📥 Load Data", use_container_width=True):
    st.session_state.load_clicked = True

if st.sidebar.button("🔄 Clear", use_container_width=True):
    st.session_state.load_clicked = False
    st.session_state.df_cache = None
    st.session_state.rs_history_cache = {}
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 Legend")
st.sidebar.markdown("🟢 **Leading**: Strong RS, ↑ Momentum")
st.sidebar.markdown("🟣 **Improving**: Weak RS, ↑ Momentum")
st.sidebar.markdown("🟡 **Weakening**: Strong RS, ↓ Momentum")
st.sidebar.markdown("🔴 **Lagging**: Weak RS, ↓ Momentum")
st.sidebar.markdown(f"**Benchmark: {bench_name}**")
st.sidebar.markdown(f"**Window: {WINDOW} periods**")

if tf_name in ["Weekly", "Monthly"]:
    adj_date = get_adjusted_close_date(tf_name)
    st.sidebar.markdown(f"**Adj Close Date: {adj_date}**")

# ============================================================================
# DATA LOADING
# ============================================================================
if st.session_state.load_clicked:
    try:
        interval, yf_period = TIMEFRAMES[tf_name]
        uni = load_universe(csv_selected)
        if uni.empty:
            st.error("❌ Failed to load universe data. Check CSV.")
            st.stop()

        symbols = uni["Symbol"].tolist()
        names_map = dict(zip(uni["Symbol"], uni["Company Name"]))
        industry_map = dict(zip(uni["Symbol"], uni["Industry"]))

        with st.spinner(f"📥 Downloading {len(symbols)} symbols ({tf_name}) ..."):
            raw = yf.download(
                symbols + [BENCHMARKS[bench_name]],
                interval=interval,
                period=yf_period,
                auto_adjust=True,
                progress=False,
                threads=True,
            )

        if BENCHMARKS[bench_name] not in raw["Close"].columns:
            st.error(f"❌ Benchmark {bench_name} data unavailable.")
            st.stop()

        bench = raw["Close"][BENCHMARKS[bench_name]]
        rows = []
        rs_history = {}
        ok_count = fail_count = 0

        for s in symbols:
            if s not in raw["Close"].columns:
                fail_count += 1
                continue

            try:
                rs_ratio, rs_mom, dist, head, vel = calculate_jdk_rrg(raw["Close"][s], bench)
                if rs_ratio is None or len(rs_ratio) < 3:
                    fail_count += 1
                    continue

                tail_len = min(14, len(rs_ratio))
                rs_history[format_symbol(s)] = {
                    "rs_ratio": rs_ratio.iloc[-tail_len:].tolist(),
                    "rs_momentum": rs_mom.iloc[-tail_len:].tolist(),
                }

                rsr = rs_ratio.iloc[-1]
                rsm = rs_mom.iloc[-1]
                d_now = dist.iloc[-1]
                h_now = head.iloc[-1]
                v_now = 0 if pd.isna(vel.iloc[-1]) else vel.iloc[-1]

                power = np.sqrt((rsr - 100) ** 2 + (rsm - 100) ** 2)
                series = raw["Close"][s]
                cur_price = series.iloc[-1]
                idx_hist = max(0, len(series) - PERIOD_MAP[period_name])
                hist_price = series.iloc[idx_hist]
                chg = calculate_price_change(cur_price, hist_price)
                status = quadrant(rsr, rsm)
                direction = get_heading_direction(h_now)

                rows.append(
                    {
                        "Symbol": format_symbol(s),
                        "Name": names_map.get(s, s),
                        "Industry": industry_map.get(s, "N/A"),
                        "Price": round(cur_price, 2),
                        "Change %": round(chg, 2),
                        "RS-Ratio": round(rsr, 2),
                        "RS-Momentum": round(rsm, 2),
                        "RRG Power": round(power, 2),
                        "Distance": round(d_now, 2),
                        "Heading": round(h_now, 1),
                        "Direction": direction,
                        "Velocity": round(v_now, 3),
                        "Status": status,
                        "TV Link": get_tv_link(s),
                    }
                )
                ok_count += 1
            except Exception:
                fail_count += 1

        if ok_count == 0:
            st.error("No data available after processing.")
            st.stop()

        st.success(f"✅ Loaded {ok_count} symbols | ⚠️ Skipped {fail_count}")

        df = pd.DataFrame(rows)
        rank_col_map = {
            "RRG Power": "RRG Power",
            "RS-Ratio": "RS-Ratio",
            "RS-Momentum": "RS-Momentum",
            "Distance": "Distance",
            "Price % Change": "Change %",
        }
        rank_col = rank_col_map[rank_by]
        df["Rank"] = df[rank_col].rank(ascending=False, method="min").astype(int)
        df = df.sort_values("Rank")
        df["Sl No."] = range(1, len(df) + 1)

        st.session_state.df_cache = df
        st.session_state.rs_history_cache = rs_history

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.stop()

# ============================================================================
# DISPLAY
# ============================================================================
if st.session_state.df_cache is not None:
    df = st.session_state.df_cache
    rs_history = st.session_state.rs_history_cache

    col_left, col_main, col_right = st.columns([1.25, 5.4, 1.35], gap="medium")

    # LEFT
    with col_left:
        st.markdown("### 📍 Legend")
        status_counts = df["Status"].value_counts()
        icon_map = {"Leading": "🟢", "Improving": "🟣", "Weakening": "🟡", "Lagging": "🔴"}
        for s in ["Leading", "Improving", "Weakening", "Lagging"]:
            st.markdown(f"{icon_map[s]} {s}: {status_counts.get(s, 0)}")

        st.markdown("---")
        st.markdown("### 📊 Stats")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Total", len(df))
            st.metric("Lead", len(df[df["Status"] == "Leading"]))
        with col_s2:
            st.metric("Impr", len(df[df["Status"] == "Improving"]))
            st.metric("Weak", len(df[df["Status"] == "Weakening"]))
        st.metric("Lag", len(df[df["Status"] == "Lagging"]))

    # MAIN
    with col_main:
        tab1, tab2 = st.tabs(["📊 Static RRG", "🎬 Rotation Animation"])

        # ---------------- STATIC RRG ----------------
        with tab1:
            st.markdown("## Relative Rotation Graph")
            st.markdown(
                f"**{csv_selected} | {tf_name} | {period_name} | Benchmark: {bench_name}**"
            )

            df_graph = select_graph_stocks(df, min_stocks=40)

            if show_labels:
                label_syms = set(
                    df_graph.nlargest(label_top_n, "Distance")["Symbol"].tolist()
                )
            else:
                label_syms = set()

            fig = go.Figure()
            x_min = df["RS-Ratio"].min() - 2
            x_max = df["RS-Ratio"].max() + 2
            y_min = df["RS-Momentum"].min() - 2
            y_max = df["RS-Momentum"].max() + 2
            x_range = max(abs(100 - x_min), abs(x_max - 100))
            y_range = max(abs(100 - y_min), abs(y_max - 100))

            # Quadrant backgrounds
            fig.add_shape(
                type="rect",
                x0=100,
                y0=100,
                x1=100 + x_range + 2,
                y1=100 + y_range + 2,
                fillcolor=QUADRANT_BG_COLORS["Leading"],
                line_width=0,
                layer="below",
            )
            fig.add_shape(
                type="rect",
                x0=100 - x_range - 2,
                y0=100,
                x1=100,
                y1=100 + y_range + 2,
                fillcolor=QUADRANT_BG_COLORS["Improving"],
                line_width=0,
                layer="below",
            )
            fig.add_shape(
                type="rect",
                x0=100 - x_range - 2,
                y0=100 - y_range - 2,
                x1=100,
                y1=100,
                fillcolor=QUADRANT_BG_COLORS["Lagging"],
                line_width=0,
                layer="below",
            )
            fig.add_shape(
                type="rect",
                x0=100,
                y0=100 - y_range - 2,
                x1=100 + x_range + 2,
                y1=100,
                fillcolor=QUADRANT_BG_COLORS["Weakening"],
                line_width=0,
                layer="below",
            )

            fig.add_hline(y=100, line_color="rgba(80,80,80,0.8)", line_width=1.5)
            fig.add_vline(x=100, line_color="rgba(80,80,80,0.8)", line_width=1.5)

            lx = x_range * 0.6
            ly = y_range * 0.7
            fig.add_annotation(
                x=100 + lx,
                y=100 + ly,
                text="<b>LEADING</b>",
                showarrow=False,
                font=dict(size=14, color=QUADRANT_COLORS["Leading"]),
            )
            fig.add_annotation(
                x=100 - lx,
                y=100 + ly,
                text="<b>IMPROVING</b>",
                showarrow=False,
                font=dict(size=14, color=QUADRANT_COLORS["Improving"]),
            )
            fig.add_annotation(
                x=100 - lx,
                y=100 - ly,
                text="<b>LAGGING</b>",
                showarrow=False,
                font=dict(size=14, color=QUADRANT_COLORS["Lagging"]),
            )
            fig.add_annotation(
                x=100 + lx,
                y=100 - ly,
                text="<b>WEAKENING</b>",
                showarrow=False,
                font=dict(size=14, color=QUADRANT_COLORS["Weakening"]),
            )

            for _, row in df_graph.iterrows():
                sym = row["Symbol"]
                status = row["Status"]
                color = QUADRANT_COLORS[status]

                if sym in rs_history:
                    tail = rs_history[sym]
                    rsr_tail = tail["rs_ratio"][-trail_length:]
                    rsm_tail = tail["rs_momentum"][-trail_length:]

                    x_pts = np.array(rsr_tail, dtype=float)
                    y_pts = np.array(rsm_tail, dtype=float)

                    if len(x_pts) >= 2:
                        if len(x_pts) >= 3:
                            xs, ys = smooth_spline_curve(x_pts, y_pts, points_per_segment=8)
                        else:
                            xs, ys = x_pts, y_pts

                        for i in range(len(xs) - 1):
                            prog = i / max(1, len(xs) - 2)
                            lw = 2.5 + prog * 3
                            op = 0.4 + prog * 0.6
                            fig.add_trace(
                                go.Scatter(
                                    x=[xs[i], xs[i + 1]],
                                    y=[ys[i], ys[i + 1]],
                                    mode="lines",
                                    line=dict(color=color, width=lw),
                                    opacity=op,
                                    hoverinfo="skip",
                                    showlegend=False,
                                )
                            )

                        # markers on original points except last
                        if len(x_pts) > 1:
                            sizes = [
                                5 + (i / max(1, len(x_pts) - 1)) * 5
                                for i in range(len(x_pts))
                            ]
                            fig.add_trace(
                                go.Scatter(
                                    x=x_pts[:-1],
                                    y=y_pts[:-1],
                                    mode="markers",
                                    marker=dict(
                                        size=sizes[:-1],
                                        color=color,
                                        opacity=0.7,
                                        line=dict(color="white", width=1),
                                    ),
                                    hoverinfo="skip",
                                    showlegend=False,
                                )
                            )

                        # arrow head
                        dx = x_pts[-1] - x_pts[-2]
                        dy = y_pts[-1] - y_pts[-2]
                        length = np.sqrt(dx ** 2 + dy ** 2)
                        if length > 0.01:
                            fig.add_annotation(
                                x=x_pts[-1],
                                y=y_pts[-1],
                                ax=x_pts[-1] - dx / length * 0.4,
                                ay=y_pts[-1] - dy / length * 0.4,
                                xref="x",
                                yref="y",
                                axref="x",
                                ayref="y",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1.8,
                                arrowwidth=3,
                                arrowcolor=color,
                            )

                hover_text = (
                    f"<b>{row['Symbol']}</b> - {row['Name']}<br>"
                    f"<b>Status:</b> {row['Status']}<br>"
                    f"<b>RS-Ratio:</b> {row['RS-Ratio']:.2f}<br>"
                    f"<b>RS-Momentum:</b> {row['RS-Momentum']:.2f}<br>"
                    f"<b>Momentum Score:</b> {row['RRG Power']:.2f}<br>"
                    f"<b>Price:</b> ₹{row['Price']:,.2f}<br>"
                    f"<b>Change %:</b> {row['Change %']:+.2f}%<br>"
                    f"<b>Industry:</b> {row['Industry']}<br>"
                    f"<b>Direction:</b> {row['Direction']}"
                )

            # head markers
                fig.add_trace(
                    go.Scatter(
                        x=[row["RS-Ratio"]],
                        y=[row["RS-Momentum"]],
                        mode="markers",
                        marker=dict(
                            size=14,
                            color=color,
                            line=dict(color="white", width=2.5),
                        ),
                        text=[hover_text],
                        hoverinfo="text",
                        hoverlabel=dict(
                            bgcolor="#1a1f2e",
                            bordercolor=color,
                            font=dict(
                                family="Plus Jakarta Sans, sans-serif",
                                size=12,
                                color="white",
                            ),
                        ),
                        showlegend=False,
                    )
                )

                if show_labels and sym in label_syms:
                    fig.add_annotation(
                        x=row["RS-Ratio"],
                        y=row["RS-Momentum"],
                        text=f"<b>{sym}</b>",
                        showarrow=True,
                        arrowhead=0,
                        arrowwidth=1.5,
                        arrowcolor=color,
                        ax=25,
                        ay=-20,
                        font=dict(size=10, color=color),
                        bgcolor="rgba(0,0,0,0)",
                        borderwidth=0,
                    )

            fig.update_layout(
                height=620,
                title=dict(
                    text=f"<b>Relative Rotation Graph</b> | {datetime.now().strftime('%Y-%m-%d')}",
                    font=dict(size=18, color="#e6eaee"),
                    x=0.5,
                ),
                xaxis=dict(
                    title=dict(text="<b>JdK RS-Ratio</b>", font=dict(color="#e6eaee")),
                    range=[100 - x_range - 1, 100 + x_range + 1],
                    showgrid=True,
                    gridcolor="rgba(150,150,150,0.2)",
                    tickfont=dict(color="#b3bdc7"),
                ),
                yaxis=dict(
                    title=dict(text="<b>JdK RS-Momentum</b>", font=dict(color="#e6eaee")),
                    range=[100 - y_range - 1, 100 + y_range + 1],
                    showgrid=True,
                    gridcolor="rgba(150,150,150,0.2)",
                    tickfont=dict(color="#b3bdc7"),
                ),
                plot_bgcolor="#fafafa",
                paper_bgcolor="#0b0e13",
                font=dict(
                    color="#e6eaee",
                    size=12,
                    family="Plus Jakarta Sans, sans-serif",
                ),
                hovermode="closest",
                showlegend=False,
                margin=dict(l=60, r=30, t=80, b=60),
            )

            st.plotly_chart(fig, width="stretch", config={"displayModeBar": True, "displaylogo": False})

            st.markdown("---")

            # ------------- INTERACTIVE TABLE (HTML + JS TEMPLATE) -------------
            with st.expander("📊 **Detailed Analysis** (Click to expand/collapse)", expanded=True):
                table_rows = ""
                for _, r in df.iterrows():
                    status = r["Status"]
                    status_color = QUADRANT_COLORS.get(status, "#808080")
                    chg_color = "#4ade80" if r["Change %"] > 0 else "#f87171" if r["Change %"] < 0 else "#9ca3af"

                    table_rows += f"""
                    <tr>
                        <td>{int(r['Sl No.'])}</td>
                        <td class="symbol-cell"><a href="{r['TV Link']}" target="_blank">{r['Symbol']}</a></td>
                        <td class="name-cell">{r['Name'][:25]}{'...' if len(r['Name']) > 25 else ''}</td>
                        <td class="industry-cell">{r['Industry'][:20]}{'...' if len(r['Industry']) > 20 else ''}</td>
                        <td>₹{r['Price']:,.2f}</td>
                        <td style="color: {chg_color}; font-weight: 600;">{r['Change %']:+.2f}%</td>
                        <td><span class="status-badge" style="background:{status_color};">{status}</span></td>
                        <td>{r['RS-Ratio']:.2f}</td>
                        <td>{r['RS-Momentum']:.2f}</td>
                        <td class="power-cell">{r['RRG Power']:.2f}</td>
                        <td>{r['Distance']:.2f}</td>
                        <td style="color: #fbbf24;">{r['Direction']}</td>
                    </tr>
                    """

                industry_options = " ".join(
                    [
                        f'<option value="{ind}">{ind}</option>'
                        for ind in sorted(df["Industry"].unique())
                    ]
                )
                total_rows = len(df)

                # Template as plain string (no f-string), placeholders replaced manually
                table_template = """
                <style>
                    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

                    * {
                        box-sizing: border-box;
                    }

                    html, body {
                        width: 100%;
                        margin: 0;
                        padding: 0;
                        background: #0b0e13;
                    }

                    .table-container {
                        font-family: 'Plus Jakarta Sans', system-ui, sans-serif;
                        background: #10141b;
                        border-radius: 10px;
                        overflow: hidden;
                        border: 1px solid #1f2732;
                        width: 100%;
                        display: block;
                    }

                    .search-container {
                        padding: 12px;
                        background: #0b0e13;
                        border-bottom: 1px solid #1f2732;
                        display: flex;
                        gap: 10px;
                        align-items: center;
                        flex-wrap: wrap;
                        width: 100%;
                    }

                    .search-box {
                        padding: 8px 12px;
                        background: #1a2230;
                        border: 1px solid #2e3745;
                        border-radius: 8px;
                        color: #e6eaee;
                        font-size: 13px;
                        outline: none;
                        min-width: 200px;
                        font-family: inherit;
                    }

                    .search-box:focus {
                        border-color: #7a5cff;
                    }

                    .filter-select {
                        padding: 8px 12px;
                        background: #1a2230;
                        border: 1px solid #2e3745;
                        border-radius: 8px;
                        color: #e6eaee;
                        font-size: 13px;
                        outline: none;
                        font-family: inherit;
                    }

                    .filter-badge {
                        background: #7a5cff;
                        color: white;
                        padding: 5px 12px;
                        border-radius: 20px;
                        font-size: 12px;
                        font-weight: 700;
                    }

                    .table-wrapper {
                        max-height: 550px;
                        overflow: auto;
                        width: 100%;
                        overflow-x: auto;
                    }

                    .rrg-table {
                        width: 100%;
                        min-width: 1100px;
                        border-collapse: collapse;
                        font-size: 13px;
                    }

                    .rrg-table th {
                        position: sticky;
                        top: 0;
                        z-index: 10;
                        background: #121823;
                        color: #b3bdc7;
                        padding: 12px 10px;
                        text-align: left;
                        font-weight: 800;
                        border-bottom: 1px solid #1f2732;
                        cursor: pointer;
                        user-select: none;
                        white-space: nowrap;
                    }

                    .rrg-table th:hover {
                        background: #1a2233;
                    }

                    .sort-icon {
                        margin-left: 6px;
                        opacity: 0.5;
                        font-size: 10px;
                    }

                    .rrg-table td {
                        padding: 10px;
                        border-bottom: 1px solid #1a2230;
                        color: #e6eaee;
                    }

                    .rrg-table tbody tr {
                        background: #0d1117;
                        transition: background 0.15s;
                    }

                    .rrg-table tbody tr:nth-child(even) {
                        background: #0f1419;
                    }

                    .rrg-table tbody tr:hover {
                        background: #161b22;
                    }

                    .symbol-cell a {
                        color: #58a6ff;
                        text-decoration: none;
                        font-weight: 700;
                    }

                    .symbol-cell a:hover {
                        text-decoration: underline;
                    }

                    .name-cell {
                        color: #9ca3af;
                        font-size: 12px;
                    }

                    .industry-cell {
                        color: #8b949e;
                        font-size: 12px;
                    }

                    .status-badge {
                        display: inline-block;
                        padding: 4px 10px;
                        border-radius: 6px;
                        font-size: 11px;
                        font-weight: 700;
                        color: white;
                        text-transform: uppercase;
                    }

                    .power-cell {
                        font-weight: 600;
                        color: #a78bfa;
                    }

                    tr.hidden {
                        display: none;
                    }

                    .table-wrapper::-webkit-scrollbar {
                        height: 10px;
                        width: 10px;
                    }

                    .table-wrapper::-webkit-scrollbar-thumb {
                        background: #2e3745;
                        border-radius: 8px;
                    }

                    .table-wrapper::-webkit-scrollbar-track {
                        background: #10141b;
                    }
                </style>

                <div class="table-container">
                    <div class="search-container">
                        <input type="text" id="searchBox" class="search-box"
                               placeholder="🔍 Search symbol or name..." onkeyup="filterTable()">
                        <select id="statusFilter" class="filter-select" onchange="filterTable()">
                            <option value="">All Status</option>
                            <option value="Leading">🟢 Leading</option>
                            <option value="Improving">🟣 Improving</option>
                            <option value="Weakening">🟡 Weakening</option>
                            <option value="Lagging">🔴 Lagging</option>
                        </select>
                        <select id="industryFilter" class="filter-select" onchange="filterTable()">
                            <option value="">All Industries</option>
                            __INDUSTRY_OPTIONS__
                        </select>
                        <span class="filter-badge" id="countBadge">__COUNT__ / __COUNT__</span>
                    </div>

                    <div class="table-wrapper">
                        <table class="rrg-table" id="dataTable">
                            <thead>
                                <tr>
                                    <th onclick="sortTable(0)">#<span class="sort-icon">⇅</span></th>
                                    <th onclick="sortTable(1)">Symbol<span class="sort-icon">⇅</span></th>
                                    <th onclick="sortTable(2)">Name<span class="sort-icon">⇅</span></th>
                                    <th onclick="sortTable(3)">Industry<span class="sort-icon">⇅</span></th>
                                    <th onclick="sortTable(4)">Price<span class="sort-icon">⇅</span></th>
                                    <th onclick="sortTable(5)">Change<span class="sort-icon">⇅</span></th>
                                    <th onclick="sortTable(6)">Status<span class="sort-icon">⇅</span></th>
                                    <th onclick="sortTable(7)">RS-Ratio<span class="sort-icon">⇅</span></th>
                                    <th onclick="sortTable(8)">RS-Mom<span class="sort-icon">⇅</span></th>
                                    <th onclick="sortTable(9)">Power<span class="sort-icon">⇅</span></th>
                                    <th onclick="sortTable(10)">Distance<span class="sort-icon">⇅</span></th>
                                    <th onclick="sortTable(11)">Direction<span class="sort-icon">⇅</span></th>
                                </tr>
                            </thead>
                            <tbody id="tableBody">
                                __ROWS__
                            </tbody>
                        </table>
                    </div>
                </div>

                <script>
                    let sortDirection = {};
                    const totalRows = __COUNT__;

                    function sortTable(columnIndex) {
                        const tbody = document.getElementById("tableBody");
                        const rows = Array.from(tbody.querySelectorAll("tr"));

                        sortDirection[columnIndex] = !sortDirection[columnIndex];
                        const ascending = sortDirection[columnIndex];

                        rows.sort((a, b) => {
                            let aValue = a.cells[columnIndex].textContent.trim();
                            let bValue = b.cells[columnIndex].textContent.trim();

                            aValue = aValue.replace(/[₹,%]/g, '');
                            bValue = bValue.replace(/[₹,%]/g, '');

                            const aNum = parseFloat(aValue);
                            const bNum = parseFloat(bValue);

                            if (!isNaN(aNum) && !isNaN(bNum)) {
                                return ascending ? aNum - bNum : bNum - aNum;
                            }

                            return ascending ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
                        });

                        tbody.innerHTML = '';
                        rows.forEach(row => tbody.appendChild(row));
                    }

                    function filterTable() {
                        const searchBox = document.getElementById("searchBox").value.toLowerCase();
                        const statusFilter = document.getElementById("statusFilter").value;
                        const industryFilter = document.getElementById("industryFilter").value;
                        const tbody = document.getElementById("tableBody");
                        const rows = tbody.getElementsByTagName("tr");
                        let visibleCount = 0;

                        for (let i = 0; i < rows.length; i++) {
                            const row = rows[i];
                            const symbol = row.cells[1].textContent.toLowerCase();
                            const name = row.cells[2].textContent.toLowerCase();
                            const industry = row.cells[3].textContent;
                            const status = row.cells[6].textContent.trim();

                            const matchesSearch = symbol.includes(searchBox) || name.includes(searchBox);
                            const matchesStatus = !statusFilter || status === statusFilter;
                            const matchesIndustry = !industryFilter || industry === industryFilter;

                            if (matchesSearch && matchesStatus && matchesIndustry) {
                                row.classList.remove("hidden");
                                visibleCount++;
                            } else {
                                row.classList.add("hidden");
                            }
                        }

                        document.getElementById("countBadge").textContent =
                            visibleCount + " / " + totalRows;
                    }
                </script>
                """

                html_table = (
                    table_template.replace("__INDUSTRY_OPTIONS__", industry_options)
                    .replace("__COUNT__", str(total_rows))
                    .replace("__ROWS__", table_rows)
                )

                st.components.v1.html(html_table, height=650, scrolling=False)

        # ---------------- ANIMATION TAB ----------------
        with tab2:
            st.markdown("### 🎬 Stock Rotation Animation")
            st.info(f"Analyze rotation patterns over {trail_length} periods")

            # Prepare animation history
            anim_hist = {}
            for sym in df_graph["Symbol"]:
                if sym in rs_history:
                    tail = rs_history[sym]
                    max_len = min(trail_length, len(tail["rs_ratio"]))
                    anim_hist[sym] = {
                        "rs_ratio": tail["rs_ratio"][-max_len:],
                        "rs_momentum": tail["rs_momentum"][-max_len:],
                    }

            if anim_hist:
                max_frames = max(
                    len(anim_hist[s]["rs_ratio"])
                    for s in anim_hist
                    if anim_hist[s]["rs_ratio"]
                )

                x_range_a = x_range
                y_range_a = y_range

                fig_a = go.Figure()

                # Quadrant backgrounds
                fig_a.add_shape(
                    type="rect",
                    x0=100,
                    y0=100,
                    x1=100 + x_range_a + 2,
                    y1=100 + y_range_a + 2,
                    fillcolor=QUADRANT_BG_COLORS["Leading"],
                    line_width=0,
                    layer="below",
                )
                fig_a.add_shape(
                    type="rect",
                    x0=100 - x_range_a - 2,
                    y0=100,
                    x1=100,
                    y1=100 + y_range_a + 2,
                    fillcolor=QUADRANT_BG_COLORS["Improving"],
                    line_width=0,
                    layer="below",
                )
                fig_a.add_shape(
                    type="rect",
                    x0=100 - x_range_a - 2,
                    y0=100 - y_range_a - 2,
                    x1=100,
                    y1=100,
                    fillcolor=QUADRANT_BG_COLORS["Lagging"],
                    line_width=0,
                    layer="below",
                )
                fig_a.add_shape(
                    type="rect",
                    x0=100,
                    y0=100 - y_range_a - 2,
                    x1=100 + x_range_a + 2,
                    y1=100,
                    fillcolor=QUADRANT_BG_COLORS["Weakening"],
                    line_width=0,
                    layer="below",
                )

                lx = x_range_a * 0.6
                ly = y_range_a * 0.7
                fig_a.add_annotation(
                    x=100 + lx,
                    y=100 + ly,
                    text="<b>LEADING</b>",
                    showarrow=False,
                    font=dict(size=14, color=QUADRANT_COLORS["Leading"]),
                )
                fig_a.add_annotation(
                    x=100 - lx,
                    y=100 + ly,
                    text="<b>IMPROVING</b>",
                    showarrow=False,
                    font=dict(size=14, color=QUADRANT_COLORS["Improving"]),
                )
                fig_a.add_annotation(
                    x=100 - lx,
                    y=100 - ly,
                    text="<b>LAGGING</b>",
                    showarrow=False,
                    font=dict(size=14, color=QUADRANT_COLORS["Lagging"]),
                )
                fig_a.add_annotation(
                    x=100 + lx,
                    y=100 - ly,
                    text="<b>WEAKENING</b>",
                    showarrow=False,
                    font=dict(size=14, color=QUADRANT_COLORS["Weakening"]),
                )

                fig_a.add_hline(y=100, line_color="rgba(80,80,80,0.8)", line_width=1.5)
                fig_a.add_vline(x=100, line_color="rgba(80,80,80,0.8)", line_width=1.5)

                STATUSES = ["Leading", "Improving", "Weakening", "Lagging"]

                def build_tail_xy(status, frame_idx):
                    xs_all, ys_all = [], []
                    syms = df_graph[df_graph["Status"] == status]["Symbol"].tolist()
                    for sym in syms:
                        if sym not in anim_hist:
                            continue
                        xs_full = anim_hist[sym]["rs_ratio"][:frame_idx]
                        ys_full = anim_hist[sym]["rs_momentum"][:frame_idx]
                        xs = xs_full[-trail_length:]
                        ys = ys_full[-trail_length:]
                        if len(xs) < 2:
                            continue
                        if len(xs) >= 3:
                            sx, sy = smooth_spline_curve(xs, ys, points_per_segment=8)
                        else:
                            sx, sy = np.array(xs, dtype=float), np.array(ys, dtype=float)
                        xs_all.extend(list(sx) + [None])
                        ys_all.extend(list(sy) + [None])
                    return xs_all, ys_all

                def build_heads(status, frame_idx):
                    pts = []
                    syms = df_graph[df_graph["Status"] == status]["Symbol"].tolist()
                    for sym in syms:
                        if sym not in anim_hist:
                            continue
                        hist = anim_hist[sym]
                        if frame_idx <= len(hist["rs_ratio"]):
                            pts.append(
                                {
                                    "symbol": sym,
                                    "x": hist["rs_ratio"][frame_idx - 1],
                                    "y": hist["rs_momentum"][frame_idx - 1],
                                }
                            )
                    return pts

                # initial traces
                init_traces = []
                for status in STATUSES:
                    tx, ty = build_tail_xy(status, 1)
                    init_traces.append(
                        go.Scatter(
                            x=tx,
                            y=ty,
                            mode="lines",
                            line=dict(color=QUADRANT_COLORS[status], width=3),
                            opacity=0.85,
                            hoverinfo="skip",
                            showlegend=False,
                        )
                    )
                    heads = build_heads(status, 1)
                    init_traces.append(
                        go.Scatter(
                            x=[p["x"] for p in heads],
                            y=[p["y"] for p in heads],
                            mode="markers+text",
                            name=status,
                            text=[f"<b>{p['symbol']}</b>" for p in heads],
                            textfont=dict(
                                color=QUADRANT_LABEL_COLORS[status],
                                size=12,
                                family="Plus Jakarta Sans, sans-serif",
                            ),
                            textposition="top center",
                            marker=dict(
                                size=14,
                                color=QUADRANT_COLORS[status],
                                line=dict(color="white", width=2.5),
                                opacity=0.95,
                            ),
                            hovertemplate="<b>%{text}</b><br>RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<extra></extra>",
                        )
                    )

                fig_a.data = init_traces

                frames = []
                for frame_i in range(1, max_frames + 1):
                    fr_traces = []
                    for status in STATUSES:
                        tx, ty = build_tail_xy(status, frame_i)
                        fr_traces.append(
                            go.Scatter(
                                x=tx,
                                y=ty,
                                mode="lines",
                                line=dict(color=QUADRANT_COLORS[status], width=3),
                                opacity=0.85,
                                hoverinfo="skip",
                                showlegend=False,
                            )
                        )
                        heads = build_heads(status, frame_i)
                        fr_traces.append(
                            go.Scatter(
                                x=[p["x"] for p in heads],
                                y=[p["y"] for p in heads],
                                mode="markers+text",
                                name=status,
                                text=[f"<b>{p['symbol']}</b>" for p in heads],
                                textfont=dict(
                                    color=QUADRANT_LABEL_COLORS[status],
                                    size=12,
                                    family="Plus Jakarta Sans, sans-serif",
                                ),
                                textposition="top center",
                                marker=dict(
                                    size=14,
                                    color=QUADRANT_COLORS[status],
                                    line=dict(color="white", width=2.5),
                                    opacity=0.95,
                                ),
                                hovertemplate="<b>%{text}</b><br>RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<extra></extra>",
                            )
                        )
                    frames.append(go.Frame(data=fr_traces, name=str(frame_i - 1)))

                fig_a.frames = frames

                fig_a.update_layout(
                    updatemenus=[
                        {
                            "type": "buttons",
                            "direction": "right",
                            "showactive": False,
                            "bgcolor": "#131922",
                            "bordercolor": "#1f2732",
                            "borderwidth": 1,
                            "font": {"color": "#e6eaee", "size": 13},
                            "pad": {"r": 8, "t": 0, "b": 0, "l": 0},
                            "buttons": [
                                {
                                    "label": "▶ Play",
                                    "method": "animate",
                                    "args": [
                                        None,
                                        {
                                            "frame": {"duration": 800, "redraw": True},
                                            "fromcurrent": True,
                                            "mode": "immediate",
                                            "transition": {
                                                "duration": 300,
                                                "easing": "cubic-in-out",
                                            },
                                        },
                                    ],
                                },
                                {
                                    "label": "⏸ Pause",
                                    "method": "animate",
                                    "args": [
                                        [None],
                                        {
                                            "frame": {"duration": 0, "redraw": False},
                                            "mode": "immediate",
                                            "transition": {"duration": 0},
                                        },
                                    ],
                                },
                            ],
                            "x": 0.02,
                            "y": 1.14,
                            "xanchor": "left",
                            "yanchor": "top",
                        }
                    ],
                    sliders=[
                        {
                            "active": 0,
                            "steps": [
                                {
                                    "args": [
                                        [f.name],
                                        {
                                            "frame": {"duration": 0, "redraw": True},
                                            "mode": "immediate",
                                            "transition": {"duration": 0},
                                        },
                                    ],
                                    "label": f"Period {i+1}/{len(frames)}",
                                    "method": "animate",
                                }
                                for i, f in enumerate(frames)
                            ],
                            "x": 0.1,
                            "len": 0.85,
                            "xanchor": "left",
                            "y": 0,
                            "yanchor": "top",
                            "pad": {"b": 10, "t": 50},
                            "currentvalue": {
                                "visible": True,
                                "prefix": "Period: ",
                                "xanchor": "right",
                            },
                            "transition": {"duration": 300, "easing": "cubic-in-out"},
                        }
                    ],
                )

                fig_a.update_layout(
                    height=700,
                    plot_bgcolor="#fafafa",
                    paper_bgcolor="#0b0e13",
                    font=dict(
                        color="#e6eaee",
                        size=12,
                        family="Plus Jakarta Sans, sans-serif",
                    ),
                    xaxis=dict(
                        title=dict(
                            text="<b>JdK RS-Ratio</b>",
                            font=dict(color="#e6eaee"),
                        ),
                        gridcolor="rgba(150,150,150,0.2)",
                        range=[100 - x_range_a - 1, 100 + x_range_a + 1],
                        zeroline=False,
                        tickfont=dict(color="#b3bdc7"),
                    ),
                    yaxis=dict(
                        title=dict(
                            text="<b>JdK RS-Momentum</b>",
                            font=dict(color="#e6eaee"),
                        ),
                        gridcolor="rgba(150,150,150,0.2)",
                        range=[100 - y_range_a - 1, 100 + y_range_a + 1],
                        zeroline=False,
                        tickfont=dict(color="#b3bdc7"),
                    ),
                    legend=dict(
                        x=1.02,
                        y=1,
                        bgcolor="rgba(30, 30, 30, 0.8)",
                        bordercolor="rgba(100, 100, 100, 0.3)",
                        borderwidth=1,
                    ),
                    hovermode="closest",
                    title=dict(
                        text=f"<b>Animated RRG</b> | {csv_selected} | {tf_name} | {bench_name}",
                        font=dict(size=16, color="#e6eaee"),
                        x=0.5,
                        xanchor="center",
                    ),
                )

                st.plotly_chart(fig_a, width="stretch", config={"displayModeBar": True})
                st.success(f"✅ Animation: {len(frames)} periods | Trail: {trail_length}")

            else:
                st.warning("No animation data available. Load data first.")

    # RIGHT SIDEBAR: Top 30 per quadrant
    with col_right:
        st.markdown("### 🚀 Top 30 Per Quadrant")
        icons = {"Leading": "🟢", "Improving": "🟣", "Weakening": "🟡", "Lagging": "🔴"}

        for status in ["Leading", "Improving", "Weakening", "Lagging"]:
            df_all = df[df["Status"] == status].sort_values("RRG Power", ascending=False)
            df_top = df_all.head(30)
            if df_top.empty:
                continue
            c = QUADRANT_COLORS.get(status, "#808080")
            ic = icons[status]
            with st.expander(
                f"{ic} **{status}** (Top {len(df_top)} of {len(df_all)})",
                expanded=(status == "Leading"),
            ):
                for _, r in df_top.iterrows():
                    st.markdown(
                        f"""
                        <div style="padding:6px; margin-bottom:4px; background:rgba(200,200,200,0.05);
                                    border-left:3px solid {c}; border-radius:4px;">
                            <small><b><a href="{r['TV Link']}" target="_blank"
                                style="color:#58a6ff; text-decoration:none;">#{int(r['Sl No.'])}</a></b></small>
                            <br><b style="color:{c}; font-size:12px;">{r['Symbol']}</b>
                            <br><small style="font-size:10px; color:#9ca3af;">{r['Industry'][:18]}</small>
                            <br><small style="color:{c}; font-size:10px;">⚡ {r['RRG Power']:.2f}</small>
                            <small style="color:#6b7280; font-size:10px;"> | 📏 {r['Distance']:.2f}</small>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        st.markdown("---")

    # FOOTER + CSV
    st.markdown("---")
    st.markdown(
        f"""
        <div style="text-align:center; color:#6b7280; font-size:10px;">
            <b>RRG Analysis Dashboard</b><br>
            Data: Yahoo Finance | JdK RRG logic inspired by Optuma scripting guide.<br>
            Displaying {len(df_graph)} stocks on graph | Total {len(df)} stocks analyzed<br>
            <i>For educational purposes only. Not financial advice.</i>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if export_csv:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button(
            "📥 Download Complete Data",
            data=buf.getvalue(),
            file_name=f"RRG_Complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

else:
    st.info("⬅️ Select indices and click **Load Data** to start analysis.")

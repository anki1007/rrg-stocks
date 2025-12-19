import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import io
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    layout="wide",
    page_title="RRG Dashboard - Multi-Index Analysis with Animation",
    initial_sidebar_state="expanded"
)

# Dark theme with enhanced styling
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 28px; }
    [data-testid="stMetricLabel"] { font-size: 14px; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

BENCHMARKS = {
    "NIFTY 50": "^NSEI",
    "NIFTY 200": "^CNX200",
    "NIFTY 500": "^CRSLDX"
}

TIMEFRAMES = {
    "5 min close": ("5m", "60d"),
    "15 min close": ("15m", "60d"),
    "30 min close": ("30m", "60d"),
    "1 hr close": ("60m", "90d"),
    "Daily": ("1d", "5y"),
    "Weekly": ("1wk", "10y"),
    "Monthly": ("1mo", "20y")
}

PERIOD_MAP = {
    "6M": 126,
    "1Y": 252,
    "2Y": 504,
    "3Y": 756,
    "5Y": 1260,
    "10Y": 2520
}

QUADRANT_COLORS = {
    "Leading": "#22c55e",
    "Improving": "#3b82f6",
    "Weakening": "#fbbf24",
    "Lagging": "#ef4444"
}

WINDOW = 12
TAIL_LENGTH = 5

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=600)
def list_csv_from_github():
    """Fetch CSV filenames from GitHub repository"""
    url = "https://api.github.com/repos/anki1007/rrg-stocks/contents/ticker"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            files = [f['name'].replace('.csv', '').upper() for f in data
                    if isinstance(f, dict) and f.get('name', '').endswith('.csv')]
            return sorted(files) if files else []
        else:
            return []
    except Exception:
        return []

@st.cache_data(ttl=600)
def load_universe(csv_name):
    """Load stock universe from GitHub CSV"""
    url = f"https://raw.githubusercontent.com/anki1007/rrg-stocks/main/ticker/{csv_name.lower()}.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception:
        return pd.DataFrame()

def get_adjusted_close_date(timeframe):
    """Get the appropriate date for adjusted close based on timeframe."""
    today = datetime.now()
    if timeframe == "Weekly":
        days_since_friday = (today.weekday() - 4) % 7
        if days_since_friday == 0 and today.weekday() == 4:
            last_friday = today
        else:
            last_friday = today - timedelta(days=(days_since_friday or 7))
        return last_friday.strftime('%Y-%m-%d')
    elif timeframe == "Monthly":
        first_day_this_month = today.replace(day=1)
        last_day_prev_month = first_day_this_month - timedelta(days=1)
        return last_day_prev_month.strftime('%Y-%m-%d')
    else:
        return today.strftime('%Y-%m-%d')

def calculate_jdk_rrg(ticker_series, benchmark_series, window=WINDOW):
    """Calculate complete JdK RRG metrics using proper z-score method"""
    aligned_data = pd.DataFrame({
        'ticker': ticker_series,
        'benchmark': benchmark_series
    }).dropna()

    if len(aligned_data) < window + 2:
        return None, None, None, None, None

    rs = 100 * (aligned_data['ticker'] / aligned_data['benchmark'])
    rs_mean = rs.rolling(window=window).mean()
    rs_std = rs.rolling(window=window).std(ddof=0)
    rs_ratio = (100 + (rs - rs_mean) / rs_std)

    rsr_roc = 100 * ((rs_ratio / rs_ratio.shift(1)) - 1)
    rsm_mean = rsr_roc.rolling(window=window).mean()
    rsm_std = rsr_roc.rolling(window=window).std(ddof=0)
    rs_momentum = (101 + ((rsr_roc - rsm_mean) / rsm_std))

    distance = np.sqrt((rs_ratio - 100) ** 2 + (rs_momentum - 100) ** 2)
    heading = np.arctan2(rs_momentum - 100, rs_ratio - 100) * 180 / np.pi
    heading = (heading + 360) % 360
    velocity = distance.diff().abs()

    min_len = min(len(rs_ratio), len(rs_momentum), len(distance), len(heading), len(velocity))
    return (rs_ratio.iloc[-min_len:].reset_index(drop=True),
            rs_momentum.iloc[-min_len:].reset_index(drop=True),
            distance.iloc[-min_len:].reset_index(drop=True),
            heading.iloc[-min_len:].reset_index(drop=True),
            velocity.iloc[-min_len:].reset_index(drop=True))

def quadrant(x, y):
    """Determine RRG quadrant"""
    if x > 100 and y > 100:
        return "Leading"
    elif x < 100 and y > 100:
        return "Improving"
    elif x < 100 and y < 100:
        return "Weakening"
    else:
        return "Lagging"

def get_heading_direction(heading):
    """Convert heading degrees to compass direction"""
    if 22.5 <= heading < 67.5:
        return "↗ NE"
    elif 67.5 <= heading < 112.5:
        return "↑ N"
    elif 112.5 <= heading < 157.5:
        return "↖ NW"
    elif 157.5 <= heading < 202.5:
        return "← W"
    elif 202.5 <= heading < 247.5:
        return "↙ SW"
    elif 247.5 <= heading < 292.5:
        return "↓ S"
    elif 292.5 <= heading < 337.5:
        return "↘ SE"
    else:
        return "→ E"

def get_tv_link(sym):
    """Generate TradingView link"""
    clean_sym = sym.replace('.NS', '')
    return f"https://www.tradingview.com/chart/?symbol=NSE:{clean_sym}"

def format_symbol(sym):
    """Remove .NS suffix"""
    return sym.replace('.NS', '')

def calculate_price_change(current_price, historical_price):
    """Calculate percentage change"""
    if historical_price == 0:
        return 0
    return ((current_price - historical_price) / historical_price) * 100

def select_graph_stocks(df, min_stocks=40):
    """Select stocks for graph display with quadrant balancing"""
    graph_stocks = []
    for status in ["Leading", "Improving", "Weakening", "Lagging"]:
        df_quad = df[df['Status'] == status].copy()
        if len(df_quad) == 0:
            continue
        elif len(df_quad) < 10:
            graph_stocks.extend(df_quad.index.tolist())
        else:
            if status in ["Leading", "Improving"]:
                top_10 = df_quad.nlargest(10, 'RRG Power')
            else:
                top_10 = df_quad.nsmallest(10, 'RRG Power')
            graph_stocks.extend(top_10.index.tolist())

    if len(graph_stocks) < min_stocks:
        remaining_indices = df.index.difference(graph_stocks)
        additional_needed = min_stocks - len(graph_stocks)
        additional_stocks = df.loc[remaining_indices].nlargest(additional_needed, 'RRG Power')
        graph_stocks.extend(additional_stocks.index.tolist())

    return df.loc[graph_stocks]

# ============================================================================
# SESSION STATE
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
if csv_files:
    for i, csv in enumerate(csv_files):
        if 'NIFTY200' in csv.upper() or 'NIFTY 200' in csv.upper():
            default_csv_index = i
            break

csv_selected = st.sidebar.selectbox("Indices", csv_files, index=default_csv_index, key="csv_select")
bench_list = list(BENCHMARKS.keys())
default_bench_index = 2
bench_name = st.sidebar.selectbox("Benchmark", bench_list, index=default_bench_index, key="bench_select")
tf_name = st.sidebar.selectbox("Strength vs Timeframe", list(TIMEFRAMES.keys()), key="tf_select")
period_name = st.sidebar.selectbox("Period", list(PERIOD_MAP.keys()), index=0, key="period_select")

rank_by = st.sidebar.selectbox(
    "Rank by",
    ["RRG Power", "RS-Ratio", "RS-Momentum", "Distance", "Price % Change"],
    index=0,
    key="rank_select"
)

st.sidebar.markdown("---")
top_n = st.sidebar.slider("Show Top N", min_value=5, max_value=50, value=15)
st.sidebar.markdown("---")
export_csv = st.sidebar.checkbox("Export CSV", value=True)
st.sidebar.markdown("---")

if st.sidebar.button("📥 Load Data", use_container_width=True, key="load_btn", type="primary"):
    st.session_state.load_clicked = True

if st.sidebar.button("🔄 Clear", use_container_width=True, key="clear_btn"):
    st.session_state.load_clicked = False
    st.session_state.df_cache = None
    st.session_state.rs_history_cache = {}
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📍 Legend")
st.sidebar.markdown("🟢 **Leading**: Strong RS, ↑ Momentum")
st.sidebar.markdown("🔵 **Improving**: Weak RS, ↑ Momentum")
st.sidebar.markdown("🟡 **Weakening**: Weak RS, ↓ Momentum")
st.sidebar.markdown("🔴 **Lagging**: Strong RS, ↓ Momentum")
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
        universe = load_universe(csv_selected)

        if universe.empty:
            st.error("❌ Failed to load universe data. Check CSV name.")
            st.stop()

        symbols = universe['Symbol'].tolist()
        names_dict = dict(zip(universe['Symbol'], universe['Company Name']))
        industries_dict = dict(zip(universe['Symbol'], universe['Industry']))

        with st.spinner(f"📥 Downloading {len(symbols)} symbols from {tf_name}..."):
            raw = yf.download(
                symbols + [BENCHMARKS[bench_name]],
                interval=interval,
                period=yf_period,
                auto_adjust=True,
                progress=False,
                threads=True
            )

        if BENCHMARKS[bench_name] not in raw['Close'].columns:
            st.error(f"❌ Benchmark {bench_name} data unavailable.")
            st.stop()

        bench = raw['Close'][BENCHMARKS[bench_name]]

        rows = []
        rs_history = {}
        success_count = 0
        failed_count = 0

        for s in symbols:
            if s not in raw['Close'].columns:
                failed_count += 1
                continue

            try:
                rs_ratio, rs_momentum, distance, heading, velocity = calculate_jdk_rrg(
                    raw['Close'][s], bench, window=WINDOW
                )

                if rs_ratio is None or len(rs_ratio) < 3:
                    failed_count += 1
                    continue

                tail_length = min(TAIL_LENGTH, len(rs_ratio))
                rs_history[format_symbol(s)] = {
                    'rs_ratio': rs_ratio.iloc[-tail_length:].tolist(),
                    'rs_momentum': rs_momentum.iloc[-tail_length:].tolist()
                }

                rsr_current = rs_ratio.iloc[-1]
                rsm_current = rs_momentum.iloc[-1]
                dist_current = distance.iloc[-1]
                head_current = heading.iloc[-1]
                vel_current = velocity.iloc[-1] if not pd.isna(velocity.iloc[-1]) else 0

                power = np.sqrt((rsr_current - 100) ** 2 + (rsm_current - 100) ** 2)
                current_price = raw['Close'][s].iloc[-1]
                historical_price = raw['Close'][s].iloc[max(0, len(raw['Close'][s]) - PERIOD_MAP[period_name])]
                price_change = calculate_price_change(current_price, historical_price)

                status = quadrant(rsr_current, rsm_current)
                direction = get_heading_direction(head_current)

                rows.append({
                    'Symbol': format_symbol(s),
                    'Name': names_dict.get(s, s),
                    'Industry': industries_dict.get(s, 'N/A'),
                    'Price': round(current_price, 2),
                    'Change %': round(price_change, 2),
                    'RS-Ratio': round(rsr_current, 2),
                    'RS-Momentum': round(rsm_current, 2),
                    'RRG Power': round(power, 2),
                    'Distance': round(dist_current, 2),
                    'Heading': round(head_current, 1),
                    'Direction': direction,
                    'Velocity': round(vel_current, 3),
                    'Status': status,
                    'TV Link': get_tv_link(s)
                })

                success_count += 1

            except Exception:
                failed_count += 1
                continue

        if success_count > 0:
            st.success(f"✅ Loaded {success_count} symbols | ⚠️ Skipped {failed_count}")
        else:
            st.error("No data available.")
            st.stop()

        df = pd.DataFrame(rows)

        if df.empty:
            st.error("No data after processing.")
            st.stop()

        rank_col_map = {
            "RRG Power": "RRG Power",
            "RS-Ratio": "RS-Ratio",
            "RS-Momentum": "RS-Momentum",
            "Distance": "Distance",
            "Price % Change": "Change %"
        }

        rank_column = rank_col_map[rank_by]
        df['Rank'] = df[rank_column].rank(ascending=False, method='min').astype(int)
        df = df.sort_values('Rank')
        df['Sl No.'] = range(1, len(df) + 1)

        st.session_state.df_cache = df
        st.session_state.rs_history_cache = rs_history

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

if st.session_state.df_cache is not None:
    df = st.session_state.df_cache
    rs_history = st.session_state.rs_history_cache

    col_left, col_main, col_right = st.columns([1, 3, 1], gap="medium")

    # ========================================================================
    # LEFT SIDEBAR
    # ========================================================================

    with col_left:
        st.markdown("### 📍 Legend")
        status_counts = df['Status'].value_counts()
        status_colors_map = {"Leading": "🟢", "Improving": "🔵", "Weakening": "🟡", "Lagging": "🔴"}
        
        for status in ["Leading", "Improving", "Weakening", "Lagging"]:
            count = status_counts.get(status, 0)
            st.markdown(f"{status_colors_map[status]} {status}: {count}")

        st.markdown("---")
        st.markdown("### 📊 Stats")
        col_stat1, col_stat2 = st.columns(2)

        with col_stat1:
            st.metric("Total", len(df))
            st.metric("Leading", len(df[df['Status'] == 'Leading']))

        with col_stat2:
            st.metric("Improving", len(df[df['Status'] == 'Improving']))
            st.metric("Weakening", len(df[df['Status'] == 'Weakening']))
            st.metric("Lagging", len(df[df['Status'] == 'Lagging']))

    # ========================================================================
    # MAIN CONTENT - RRG GRAPH
    # ========================================================================

    with col_main:
        st.markdown("## Relative Rotation Graph")
        st.markdown(f"**{csv_selected} | {tf_name} | {period_name} | Benchmark: {bench_name}**")

        df_graph = select_graph_stocks(df, min_stocks=40)

        fig_rrg = go.Figure()

        x_min = df['RS-Ratio'].min() - 5
        x_max = df['RS-Ratio'].max() + 5
        y_min = df['RS-Momentum'].min() - 5
        y_max = df['RS-Momentum'].max() + 5

        quadrant_size = max(abs(100 - x_min), abs(x_max - 100), abs(100 - y_min), abs(y_max - 100))

        # Quadrant backgrounds
        fig_rrg.add_shape(type="rect", x0=100, y0=100, x1=100+quadrant_size, y1=100+quadrant_size,
            fillcolor="rgba(34, 197, 94, 0.12)", line=dict(color="rgba(34, 197, 94, 0.4)", width=2), layer="below")
        fig_rrg.add_shape(type="rect", x0=100-quadrant_size, y0=100, x1=100, y1=100+quadrant_size,
            fillcolor="rgba(59, 130, 246, 0.12)", line=dict(color="rgba(59, 130, 246, 0.4)", width=2), layer="below")
        fig_rrg.add_shape(type="rect", x0=100-quadrant_size, y0=100-quadrant_size, x1=100, y1=100,
            fillcolor="rgba(251, 191, 36, 0.12)", line=dict(color="rgba(251, 191, 36, 0.4)", width=2), layer="below")
        fig_rrg.add_shape(type="rect", x0=100, y0=100-quadrant_size, x1=100+quadrant_size, y1=100,
            fillcolor="rgba(239, 68, 68, 0.12)", line=dict(color="rgba(239, 68, 68, 0.4)", width=2), layer="below")

        # Quadrant labels
        fig_rrg.add_annotation(
            x=100+quadrant_size*0.5, y=100+quadrant_size*0.5,
            text="Leading",
            showarrow=False,
            font=dict(size=18, color="#0d4a1f", family="Arial")
        )

        fig_rrg.add_annotation(
            x=100-quadrant_size*0.5, y=100+quadrant_size*0.5,
            text="Improving",
            showarrow=False,
            font=dict(size=18, color="#1e3a8a", family="Arial")
        )

        fig_rrg.add_annotation(
            x=100-quadrant_size*0.5, y=100-quadrant_size*0.5,
            text="Weakening",
            showarrow=False,
            font=dict(size=18, color="#713f12", family="Arial")
        )

        fig_rrg.add_annotation(
            x=100+quadrant_size*0.5, y=100-quadrant_size*0.5,
            text="Lagging",
            showarrow=False,
            font=dict(size=18, color="#7f1d1d", family="Arial")
        )

        # Center lines
        fig_rrg.add_hline(y=100, line_dash="dash", line_color="rgba(100, 100, 100, 0.6)", line_width=2, layer="below")
        fig_rrg.add_vline(x=100, line_dash="dash", line_color="rgba(100, 100, 100, 0.6)", line_width=2, layer="below")

        # Add data points with TAILS
        for status in ["Leading", "Improving", "Weakening", "Lagging"]:
            df_status = df_graph[df_graph['Status'] == status]
            if not df_status.empty:
                hover_text = []
                for _, row in df_status.iterrows():
                    if row['Symbol'] in rs_history:
                        tail_data = rs_history[row['Symbol']]
                        rs_ratio_tail = tail_data['rs_ratio']
                        rs_momentum_tail = tail_data['rs_momentum']

                        if len(rs_ratio_tail) > 1:
                            fig_rrg.add_trace(go.Scatter(
                                x=rs_ratio_tail,
                                y=rs_momentum_tail,
                                mode='lines',
                                line=dict(color=QUADRANT_COLORS[status], width=2.5, dash='dot'),
                                showlegend=False,
                                hoverinfo='skip'
                            ))

                        if len(rs_ratio_tail) >= 2:
                            x_tail = rs_ratio_tail[-2]
                            y_tail = rs_momentum_tail[-2]
                            x_head = rs_ratio_tail[-1]
                            y_head = rs_momentum_tail[-1]

                            fig_rrg.add_annotation(
                                x=x_head,
                                y=y_head,
                                ax=x_tail,
                                ay=y_tail,
                                xref='x',
                                yref='y',
                                axref='x',
                                ayref='y',
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1.2,
                                arrowwidth=2.5,
                                arrowcolor=QUADRANT_COLORS[status]
                            )

                    hover_info = (
                        f"<b>{row['Name']}</b>\n"
                        f"Symbol: {row['Symbol']}\n"
                        f"Industry: {row['Industry']}\n"
                        f"Price: ₹{row['Price']:.2f} | {row['Change %']:+.2f}%\n\n"
                        f"<b>JdK Metrics:</b>\n"
                        f"RS-Ratio: {row['RS-Ratio']:.2f} | RS-Momentum: {row['RS-Momentum']:.2f}\n"
                        f"Distance: {row['Distance']:.2f} | Velocity: {row['Velocity']:.3f}\n"
                        f"Heading: {row['Heading']:.1f}° {row['Direction']}\n"
                        f"Status: <b>{row['Status']}</b>"
                    )
                    hover_text.append(hover_info)

                # Add markers
                fig_rrg.add_trace(go.Scatter(
                    x=df_status['RS-Ratio'],
                    y=df_status['RS-Momentum'],
                    mode='markers+text',
                    name=status,
                    text=df_status['Symbol'],
                    textposition="top center",
                    customdata=hover_text,
                    marker=dict(
                        size=16,
                        color=QUADRANT_COLORS[status],
                        line=dict(color='white', width=2.5),
                        opacity=0.95
                    ),
                    hovertemplate='%{customdata}<extra></extra>'
                ))

        fig_rrg.update_layout(
            height=700,
            plot_bgcolor='rgba(18, 18, 18, 0.95)',
            paper_bgcolor='rgba(10, 10, 10, 0.98)',
            font=dict(color='#e0e0e0', size=12, family='Inter, sans-serif'),
            xaxis=dict(title="RS-Ratio →", gridcolor='rgba(50, 50, 50, 0.5)',
                      range=[x_min, x_max], zeroline=False),
            yaxis=dict(title="RS-Momentum →", gridcolor='rgba(50, 50, 50, 0.5)',
                      range=[y_min, y_max], zeroline=False),
            legend=dict(x=1.02, y=1, bgcolor='rgba(30, 30, 30, 0.8)',
                       bordercolor='rgba(100, 100, 100, 0.3)', borderwidth=1),
            hovermode='closest',
            title=dict(text=f"RRG: {csv_selected} | {tf_name} | {bench_name}",
                      font=dict(size=16, color='#ffffff'), x=0.5, xanchor='center')
        )

        st.plotly_chart(fig_rrg, use_container_width=True, config={'displayModeBar': True})

        # ====================================================================
        # 🎬 ANIMATED RRG GRAPH (Toggle Option)
        # ====================================================================

        st.markdown("---")
        show_animation = st.checkbox("🎬 Enable Animation (Rotation Check)", value=False)

        if show_animation and rs_history:
            st.markdown("### 🎬 Animated Relative Rotation")
            st.info(f"Animation shows stock rotation through quadrants over {TAIL_LENGTH} periods")

            # Create animation frames
            frames = []
            frame_labels = []

            # Get maximum history length
            valid_symbols = [sym for sym in rs_history if sym in df_graph['Symbol'].values]
            if valid_symbols:
                max_history = max([len(rs_history[sym]['rs_ratio']) for sym in valid_symbols])

                for frame_idx in range(1, max_history + 1):
                    frame_data = []

                    for status in ["Leading", "Improving", "Weakening", "Lagging"]:
                        df_status = df_graph[df_graph['Status'] == status]

                        for _, row in df_status.iterrows():
                            if row['Symbol'] in rs_history:
                                tail_data = rs_history[row['Symbol']]

                                if frame_idx <= len(tail_data['rs_ratio']):
                                    x_val = tail_data['rs_ratio'][frame_idx - 1]
                                    y_val = tail_data['rs_momentum'][frame_idx - 1]

                                    frame_data.append({
                                        'symbol': row['Symbol'],
                                        'name': row['Name'],
                                        'x': x_val,
                                        'y': y_val,
                                        'status': status,
                                        'color': QUADRANT_COLORS[status],
                                        'price': row['Price'],
                                        'change': row['Change %'],
                                        'industry': row['Industry']
                                    })

                    frames.append(frame_data)
                    frame_labels.append(f"Period {frame_idx}/{max_history}")

                # Create animated figure
                fig_animated = go.Figure()

                # Add quadrant backgrounds
                fig_animated.add_shape(type="rect", x0=100, y0=100, x1=100+quadrant_size, y1=100+quadrant_size,
                    fillcolor="rgba(34, 197, 94, 0.12)", line=dict(color="rgba(34, 197, 94, 0.4)", width=2), layer="below")
                fig_animated.add_shape(type="rect", x0=100-quadrant_size, y0=100, x1=100, y1=100+quadrant_size,
                    fillcolor="rgba(59, 130, 246, 0.12)", line=dict(color="rgba(59, 130, 246, 0.4)", width=2), layer="below")
                fig_animated.add_shape(type="rect", x0=100-quadrant_size, y0=100-quadrant_size, x1=100, y1=100,
                    fillcolor="rgba(251, 191, 36, 0.12)", line=dict(color="rgba(251, 191, 36, 0.4)", width=2), layer="below")
                fig_animated.add_shape(type="rect", x0=100, y0=100-quadrant_size, x1=100+quadrant_size, y1=100,
                    fillcolor="rgba(239, 68, 68, 0.12)", line=dict(color="rgba(239, 68, 68, 0.4)", width=2), layer="below")

                # Add quadrant labels
                fig_animated.add_annotation(x=100+quadrant_size*0.5, y=100+quadrant_size*0.5, text="Leading",
                    showarrow=False, font=dict(size=18, color="#0d4a1f", family="Arial"))
                fig_animated.add_annotation(x=100-quadrant_size*0.5, y=100+quadrant_size*0.5, text="Improving",
                    showarrow=False, font=dict(size=18, color="#1e3a8a", family="Arial"))
                fig_animated.add_annotation(x=100-quadrant_size*0.5, y=100-quadrant_size*0.5, text="Weakening",
                    showarrow=False, font=dict(size=18, color="#713f12", family="Arial"))
                fig_animated.add_annotation(x=100+quadrant_size*0.5, y=100-quadrant_size*0.5, text="Lagging",
                    showarrow=False, font=dict(size=18, color="#7f1d1d", family="Arial"))

                # Center lines
                fig_animated.add_hline(y=100, line_dash="dash", line_color="rgba(100, 100, 100, 0.6)", line_width=2)
                fig_animated.add_vline(x=100, line_dash="dash", line_color="rgba(100, 100, 100, 0.6)", line_width=2)

                # Add initial frame data
                if frames:
                    initial_frame = frames[0]
                    for status in ["Leading", "Improving", "Weakening", "Lagging"]:
                        status_data = [d for d in initial_frame if d['status'] == status]

                        if status_data:
                            fig_animated.add_trace(go.Scatter(
                                x=[d['x'] for d in status_data],
                                y=[d['y'] for d in status_data],
                                mode='markers+text',
                                name=status,
                                text=[d['symbol'] for d in status_data],
                                textposition="top center",
                                marker=dict(size=16, color=QUADRANT_COLORS[status],
                                           line=dict(color='white', width=2.5), opacity=0.95),
                                hovertemplate='<b>%{text}</b><br>RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<extra></extra>'
                            ))

                    # Create animation frames
                    plotly_frames = []
                    for i, frame_data in enumerate(frames):
                        frame_traces = []

                        for status in ["Leading", "Improving", "Weakening", "Lagging"]:
                            status_data = [d for d in frame_data if d['status'] == status]

                            if status_data:
                                frame_traces.append(go.Scatter(
                                    x=[d['x'] for d in status_data],
                                    y=[d['y'] for d in status_data],
                                    mode='markers+text',
                                    name=status,
                                    text=[d['symbol'] for d in status_data],
                                    textposition="top center",
                                    marker=dict(size=16, color=QUADRANT_COLORS[status],
                                               line=dict(color='white', width=2.5), opacity=0.95),
                                    hovertemplate='<b>%{text}</b><br>RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<extra></extra>'
                                ))

                        plotly_frames.append(go.Frame(data=frame_traces, name=str(i)))

                    fig_animated.frames = plotly_frames

                    # Add play/pause buttons and slider
                    fig_animated.update_layout(
                        updatemenus=[{
                            'type': 'buttons',
                            'showactive': False,
                            'buttons': [
                                {'label': '▶ Play', 'method': 'animate',
                                 'args': [None, {'frame': {'duration': 800, 'redraw': True},
                                                'fromcurrent': True, 'mode': 'immediate',
                                                'transition': {'duration': 300, 'easing': 'cubic-in-out'}}]},
                                {'label': '⏸ Pause', 'method': 'animate',
                                 'args': [[None], {'frame': {'duration': 0, 'redraw': False},
                                                  'mode': 'immediate', 'transition': {'duration': 0}}]}
                            ],
                            'x': 0.1, 'y': 1.15, 'xanchor': 'left', 'yanchor': 'top'
                        }],
                        sliders=[{
                            'active': 0,
                            'steps': [{'args': [[f.name], {'frame': {'duration': 0, 'redraw': True},
                                                           'mode': 'immediate', 'transition': {'duration': 0}}],
                                      'label': frame_labels[i], 'method': 'animate'}
                                     for i, f in enumerate(plotly_frames)],
                            'x': 0.1, 'len': 0.85, 'xanchor': 'left', 'y': 0,
                            'yanchor': 'top', 'pad': {'b': 10, 't': 50},
                            'currentvalue': {'visible': True, 'prefix': 'Period: ', 'xanchor': 'right'},
                            'transition': {'duration': 300, 'easing': 'cubic-in-out'}
                        }]
                    )

                    fig_animated.update_layout(
                        height=700,
                        plot_bgcolor='rgba(18, 18, 18, 0.95)',
                        paper_bgcolor='rgba(10, 10, 10, 0.98)',
                        font=dict(color='#e0e0e0', size=12, family='Inter, sans-serif'),
                        xaxis=dict(title="RS-Ratio →", gridcolor='rgba(50, 50, 50, 0.5)',
                                   range=[x_min, x_max], zeroline=False),
                        yaxis=dict(title="RS-Momentum →", gridcolor='rgba(50, 50, 50, 0.5)',
                                   range=[y_min, y_max], zeroline=False),
                        legend=dict(x=1.02, y=1, bgcolor='rgba(30, 30, 30, 0.8)',
                                   bordercolor='rgba(100, 100, 100, 0.3)', borderwidth=1),
                        hovermode='closest',
                        title=dict(text=f"Animated RRG: {csv_selected} | {tf_name} | {bench_name}",
                                  font=dict(size=16, color='#ffffff'), x=0.5, xanchor='center')
                    )

                    st.plotly_chart(fig_animated, use_container_width=True, config={'displayModeBar': True})
                    st.success(f"✅ Animation ready: {len(frames)} periods tracked")

    # ========================================================================
    # RIGHT SIDEBAR - TABLE
    # ========================================================================

    with col_right:
        st.markdown("### 📈 Top Stocks")

        df_display = df.head(top_n)[['Sl No.', 'Symbol', 'Status', 'Price', 'Change %', 'RS-Ratio', 'RS-Momentum', 'Distance']].copy()

        status_emoji = {'Leading': '🟢', 'Improving': '🔵', 'Weakening': '🟡', 'Lagging': '🔴'}
        df_display['Status'] = df_display['Status'].map(status_emoji) + ' ' + df_display['Status']

        st.dataframe(df_display, use_container_width=True, hide_index=True, height=600)

        if export_csv:
            st.markdown("---")
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"RRG_{csv_selected}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

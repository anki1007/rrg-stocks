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
    page_title="RRG Dashboard - Multi-Index Analysis",
    initial_sidebar_state="expanded"
)

# Dark theme with enhanced styling
st.markdown("""
<style>
    body { background-color: #111827; }
    .main { background-color: #111827; }
    [data-testid="stSidebar"] { background-color: #1f2937; }
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
        
        # Quadrant labels - DARK AND BOLD
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
                        f"<b>{row['Name']}</b><br>"
                        f"Symbol: {row['Symbol']}<br>"
                        f"Industry: {row['Industry']}<br>"
                        f"Price: ₹{row['Price']:.2f} | {row['Change %']:+.2f}%<br>"
                        f"<b>JdK Metrics:</b><br>"
                        f"RS-Ratio: {row['RS-Ratio']:.2f} | RS-Momentum: {row['RS-Momentum']:.2f}<br>"
                        f"Distance: {row['Distance']:.2f} | Velocity: {row['Velocity']:.3f}<br>"
                        f"Heading: {row['Heading']:.1f}° {row['Direction']}<br>"
                        f"Status: <b>{row['Status']}</b>"
                    )
                    hover_text.append(hover_info)
                
                # Add markers - LARGE AND VISIBLE
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
                    hovertemplate='%{customdata}<extra></extra>',
                    textfont=dict(
                        color='#000000',
                        size=11,
                        family='Arial'
                    )
                ))
        
        # Enhanced layout
        fig_rrg.update_layout(
            height=550,
            xaxis_title="RS-Ratio (X-axis)",
            yaxis_title="RS-Momentum (Y-axis)",
            plot_bgcolor="rgba(255, 255, 255, 0.95)",
            paper_bgcolor="rgba(255, 255, 255, 0.95)",
            font=dict(color="#000000", size=13, family="Arial"),
            hovermode="closest",
            xaxis=dict(
                gridcolor="rgba(200, 200, 200, 0.5)", 
                zeroline=False,
                title_font=dict(size=14, color="#000000", family="Arial"),
                tickfont=dict(size=12, color="#000000")
            ),
            yaxis=dict(
                gridcolor="rgba(200, 200, 200, 0.5)", 
                zeroline=False,
                title_font=dict(size=14, color="#000000", family="Arial"),
                tickfont=dict(size=12, color="#000000")
            ),
            showlegend=False,
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        st.plotly_chart(fig_rrg, use_container_width=True)
        
        st.markdown("---")
        
        # COLLAPSIBLE TABLE WITH MAXIMUM VISIBILITY
        with st.expander("📊 **Detailed Analysis** (Click to expand/collapse)", expanded=True):
            table_rows = ""
            for _, row in df.iterrows():
                status_class = f"status-{row['Status'].lower()}"
                table_rows += f"""
                <tr>
                    <td style="color: #ffffff !important;">{int(row['Sl No.'])}</td>
                    <td style="color: #60a5fa !important; text-align: left;"><a href="{row['TV Link']}" target="_blank" style="color: #60a5fa !important; text-decoration: none; font-weight: bold;">{row['Symbol']}</a></td>
                    <td style="color: #e5e7eb !important; text-align: left;">{row['Industry']}</td>
                    <td style="color: #ffffff !important;">₹{row['Price']:.2f}</td>
                    <td style="color: #ffffff !important;">{row['Change %']:+.2f}%</td>
                    <td style="color: #ffffff !important;">{row['RRG Power']:.2f}</td>
                    <td><span class="{status_class}">{row['Status']}</span></td>
                    <td style="color: #ffffff !important;">{row['RS-Ratio']:.2f}</td>
                    <td style="color: #ffffff !important;">{row['RS-Momentum']:.2f}</td>
                    <td style="color: #ffffff !important;">{row['Distance']:.2f}</td>
                    <td style="color: #fbbf24 !important;">{row['Direction']}</td>
                    <td style="color: #ffffff !important;">{row['Velocity']:.3f}</td>
                </tr>
                """
            
            html_table = f"""
            <style>
                .rrg-table-wrapper {{
                    max-height: 600px;
                    overflow-y: auto;
                    overflow-x: auto;
                    border: 2px solid #4b5563;
                    border-radius: 8px;
                    background-color: #1f2937;
                }}
                .rrg-table-custom {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 14px;
                    font-family: 'Segoe UI', Arial, sans-serif;
                }}
                .rrg-table-custom thead {{
                    position: sticky;
                    top: 0;
                    z-index: 100;
                    background-color: #111827;
                }}
                .rrg-table-custom th {{
                    background-color: #111827 !important;
                    color: #ffffff !important;
                    padding: 14px 10px !important;
                    text-align: center !important;
                    font-weight: bold !important;
                    border: 1px solid #4b5563 !important;
                    font-size: 13px !important;
                }}
                .rrg-table-custom td {{
                    padding: 10px !important;
                    text-align: center !important;
                    border: 1px solid #4b5563 !important;
                    color: #ffffff !important;
                    background-color: #1f2937 !important;
                    font-size: 13px !important;
                }}
                .rrg-table-custom tbody tr:nth-child(even) td {{
                    background-color: #374151 !important;
                }}
                .rrg-table-custom tbody tr:hover td {{
                    background-color: #4b5563 !important;
                }}
                .status-leading {{
                    background: linear-gradient(135deg, rgba(34, 197, 94, 0.7), rgba(34, 197, 94, 0.5)) !important;
                    color: #4ade80 !important;
                    font-weight: bold !important;
                    padding: 6px 14px !important;
                    border-radius: 6px !important;
                    display: inline-block !important;
                }}
                .status-improving {{
                    background: linear-gradient(135deg, rgba(59, 130, 246, 0.7), rgba(59, 130, 246, 0.5)) !important;
                    color: #60a5fa !important;
                    font-weight: bold !important;
                    padding: 6px 14px !important;
                    border-radius: 6px !important;
                    display: inline-block !important;
                }}
                .status-weakening {{
                    background: linear-gradient(135deg, rgba(251, 191, 36, 0.7), rgba(251, 191, 36, 0.5)) !important;
                    color: #fbbf24 !important;
                    font-weight: bold !important;
                    padding: 6px 14px !important;
                    border-radius: 6px !important;
                    display: inline-block !important;
                }}
                .status-lagging {{
                    background: linear-gradient(135deg, rgba(239, 68, 68, 0.7), rgba(239, 68, 68, 0.5)) !important;
                    color: #f87171 !important;
                    font-weight: bold !important;
                    padding: 6px 14px !important;
                    border-radius: 6px !important;
                    display: inline-block !important;
                }}
            </style>
            <div class="rrg-table-wrapper">
                <table class="rrg-table-custom">
                    <thead>
                        <tr>
                            <th>Sl No.</th>
                            <th>Symbol</th>
                            <th>Industry</th>
                            <th>Price</th>
                            <th>Change %</th>
                            <th>Strength</th>
                            <th>Status</th>
                            <th>RS-Ratio</th>
                            <th>RS-Momentum</th>
                            <th>Distance</th>
                            <th>Direction</th>
                            <th>Velocity</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
            """
            
            st.components.v1.html(html_table, height=620, scrolling=True)
    
    # ========================================================================
    # RIGHT SIDEBAR - TOP 30 PER QUADRANT
    # ========================================================================
    with col_right:
        st.markdown("### 🚀 Top 30 Per Quadrant")
        
        for status in ["Leading", "Improving", "Weakening", "Lagging"]:
            df_status_all = df[df['Status'] == status].sort_values('RRG Power', ascending=False)
            df_status_top30 = df_status_all.head(30)
            
            if not df_status_top30.empty:
                status_color = QUADRANT_COLORS.get(status, "#808080")
                status_icon = {"Leading": "🟢", "Improving": "🔵", "Weakening": "🟡", "Lagging": "🔴"}[status]
                
                total_in_quadrant = len(df_status_all)
                showing = len(df_status_top30)
                
                with st.expander(f"{status_icon} **{status}** (Top {showing} of {total_in_quadrant})", expanded=(status == "Leading")):
                    for idx, (_, row) in enumerate(df_status_top30.iterrows(), 1):
                        tv_link = row['TV Link']
                        
                        st.markdown(f"""
                        <div style="padding: 6px; margin-bottom: 4px; background: rgba(200,200,200,0.05); 
                                    border-left: 3px solid {status_color}; border-radius: 4px;">
                            <small><b><a href="{tv_link}" target="_blank" 
                                style="color: #60a5fa; text-decoration: none;">#{int(row['Sl No.'])}</a></b></small>
                            <br><b style="color: {status_color}; font-size: 12px;">{row['Symbol']}</b>
                            <br><small style="font-size: 10px;">{row['Industry'][:18]}</small>
                            <br><small style="color: {status_color}; font-size: 10px;">⚡ {row['RRG Power']:.2f}</small>
                            <small style="color: #9ca3af; font-size: 10px;"> | 📏 {row['Distance']:.2f}</small>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # ========================================================================
    # FOOTER
    # ========================================================================
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #888; font-size: 10px;">
        <b>RRG Analysis Dashboard</b><br>
        Data: Yahoo Finance | Charts: TradingView<br>
        Displaying {len(df_graph)} stocks on graph | Total {len(df)} stocks analyzed<br>
        Reference: <a href="https://www.optuma.com/blog/scripting-for-rrgs" target="_blank" 
                      style="color: #0066cc;">Optuma RRG Scripting Guide</a><br>
        <i>Disclaimer: For educational purposes only. Not financial advice.</i>
    </div>
    """, unsafe_allow_html=True)
    
    if export_csv:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Download Complete Data",
            data=csv_data,
            file_name=f"RRG_Complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

else:
    st.info("⬅️ Select indices and click **Load Data** to start analysis")


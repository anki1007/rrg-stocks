import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import io
import base64
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

# Dark theme
st.markdown("""
<style>
    body { background-color: #111827; }
    .main { background-color: #111827; }
    [data-testid="stSidebar"] { background-color: #1f2937; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# CONFIGURATION - FIXED BENCHMARKS
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

WINDOW = 12  # Standard RRG window

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
        
        # Handle both list and error responses
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
    """
    Get the appropriate date for adjusted close based on timeframe.
    - Weekly: Last Friday's close
    - Monthly: Last day of month's close
    - Daily: Today's close
    """
    today = datetime.now()
    
    if timeframe == "Weekly":
        # Get last Friday (4 = Friday in weekday())
        days_since_friday = (today.weekday() - 4) % 7
        if days_since_friday == 0 and today.weekday() == 4:
            last_friday = today
        else:
            last_friday = today - timedelta(days=(days_since_friday or 7))
        return last_friday.strftime('%Y-%m-%d')
    
    elif timeframe == "Monthly":
        # Get last day of previous month
        first_day_this_month = today.replace(day=1)
        last_day_prev_month = first_day_this_month - timedelta(days=1)
        return last_day_prev_month.strftime('%Y-%m-%d')
    
    else:  # Daily and intraday
        return today.strftime('%Y-%m-%d')

def calculate_jdk_rrg(ticker_series, benchmark_series, window=WINDOW):
    """
    Calculate complete JdK RRG metrics using proper z-score method
    
    Reference: https://www.optuma.com/blog/scripting-for-rrgs
    
    Returns:
        rs_ratio: RS-Ratio (X-axis) - JDKRS().Ratio
        rs_momentum: RS-Momentum (Y-axis) - JDKRS().Momentum
        distance: Distance from center (100,100) - JDKRS().Distance
        heading: Direction in degrees (0-360) - JDKRS().Heading
        velocity: Vector movement speed - JDKRS().Velocity
    """
    # Ensure same length and aligned index
    aligned_data = pd.DataFrame({
        'ticker': ticker_series,
        'benchmark': benchmark_series
    }).dropna()
    
    if len(aligned_data) < window + 2:
        return None, None, None, None, None
    
    # 1. Calculate Relative Strength (RS) = ticker / benchmark
    rs = 100 * (aligned_data['ticker'] / aligned_data['benchmark'])
    
    # 2. Calculate JdK RS-Ratio (z-score of RS)
    # RS-Ratio is the X-axis value
    rs_mean = rs.rolling(window=window).mean()
    rs_std = rs.rolling(window=window).std(ddof=0)
    rs_ratio = (100 + (rs - rs_mean) / rs_std)
    
    # 3. Calculate Rate of Change of RS-Ratio
    rsr_roc = 100 * ((rs_ratio / rs_ratio.shift(1)) - 1)
    
    # 4. Calculate JdK RS-Momentum (z-score of RS-Ratio ROC)
    # RS-Momentum is the Y-axis value
    rsm_mean = rsr_roc.rolling(window=window).mean()
    rsm_std = rsr_roc.rolling(window=window).std(ddof=0)
    rs_momentum = (101 + ((rsr_roc - rsm_mean) / rsm_std))
    
    # 5. Calculate JDKRS().Distance - Distance from center (100,100)
    distance = np.sqrt((rs_ratio - 100) ** 2 + (rs_momentum - 100) ** 2)
    
    # 6. Calculate JDKRS().Heading - Direction in degrees (0-360)
    heading = np.arctan2(rs_momentum - 100, rs_ratio - 100) * 180 / np.pi
    heading = (heading + 360) % 360
    
    # 7. Calculate JDKRS().Velocity - Vector movement speed
    velocity = distance.diff().abs()
    
    # Align all series
    min_len = min(len(rs_ratio), len(rs_momentum), len(distance), len(heading), len(velocity))
    
    rs_ratio_clean = rs_ratio.iloc[-min_len:].reset_index(drop=True)
    rs_momentum_clean = rs_momentum.iloc[-min_len:].reset_index(drop=True)
    distance_clean = distance.iloc[-min_len:].reset_index(drop=True)
    heading_clean = heading.iloc[-min_len:].reset_index(drop=True)
    velocity_clean = velocity.iloc[-min_len:].reset_index(drop=True)
    
    return rs_ratio_clean, rs_momentum_clean, distance_clean, heading_clean, velocity_clean

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
    """Generate TradingView link (without .NS suffix)"""
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

# ============================================================================
# SESSION STATE & INITIALIZATION
# ============================================================================

if "load_clicked" not in st.session_state:
    st.session_state.load_clicked = False
if "df_cache" not in st.session_state:
    st.session_state.df_cache = None
if "df_top_cache" not in st.session_state:
    st.session_state.df_top_cache = None

# ============================================================================
# SIDEBAR - CONTROLS WITH LOAD BUTTON
# ============================================================================

st.sidebar.markdown("### ⚙️ Controls")

# Get CSV files list
csv_files = list_csv_from_github()

if not csv_files:
    st.sidebar.warning("⚠️ Unable to fetch indices from GitHub. Check connection.")
    csv_files = ["NIFTY200"]  # Fallback

# Set default to NIFTY200
default_csv_index = 0
if csv_files:
    for i, csv in enumerate(csv_files):
        if 'NIFTY200' in csv.upper() or 'NIFTY 200' in csv.upper():
            default_csv_index = i
            break

csv_selected = st.sidebar.selectbox(
    "Indices",
    csv_files,
    index=default_csv_index,
    key="csv_select"
)

# Set default benchmark to NIFTY 500
bench_list = list(BENCHMARKS.keys())
default_bench_index = 2  # NIFTY 500 is at index 2
bench_name = st.sidebar.selectbox(
    "Benchmark",
    bench_list,
    index=default_bench_index,
    key="bench_select"
)

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

# ============================================================================
# LOAD BUTTON - TRIGGER DATA FETCH
# ============================================================================

st.sidebar.markdown("---")

col_load, col_reset = st.sidebar.columns(2)

with col_load:
    if st.button("📥 Load Data", use_container_width=True, key="load_btn"):
        st.session_state.load_clicked = True

with col_reset:
    if st.button("🔄 Clear", use_container_width=True, key="clear_btn"):
        st.session_state.load_clicked = False
        st.session_state.df_cache = None
        st.session_state.df_top_cache = None
        st.rerun()

st.sidebar.markdown("---")

st.sidebar.markdown("### 📍 Legend")
st.sidebar.markdown("🟢 **Leading**: Strong RS, ↑ Momentum")
st.sidebar.markdown("🔵 **Improving**: Weak RS, ↑ Momentum")
st.sidebar.markdown("🟡 **Weakening**: Weak RS, ↓ Momentum")
st.sidebar.markdown("🔴 **Lagging**: Strong RS, ↓ Momentum")

st.sidebar.markdown(f"**Benchmark: {BENCHMARKS[bench_name]}**")
st.sidebar.markdown(f"**Window: {WINDOW} periods**")

if tf_name in ["Weekly", "Monthly"]:
    adj_date = get_adjusted_close_date(tf_name)
    st.sidebar.markdown(f"**Adj Close Date: {adj_date}**")

# ============================================================================
# DATA LOADING - ONLY WHEN LOAD BUTTON CLICKED
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
        
        # Check benchmark data availability
        if BENCHMARKS[bench_name] not in raw['Close'].columns:
            st.error(f"❌ Benchmark {bench_name} data unavailable.")
            st.stop()
        
        bench = raw['Close'][BENCHMARKS[bench_name]]
        
        rows = []
        success_count = 0
        failed_count = 0
        
        for s in symbols:
            if s not in raw['Close'].columns:
                failed_count += 1
                continue
            
            try:
                # Use complete JdK RRG formula
                rs_ratio, rs_momentum, distance, heading, velocity = calculate_jdk_rrg(
                    raw['Close'][s], bench, window=WINDOW
                )
                
                if rs_ratio is None or len(rs_ratio) < 3:
                    failed_count += 1
                    continue
                
                # Get latest values
                rsr_current = rs_ratio.iloc[-1]
                rsm_current = rs_momentum.iloc[-1]
                dist_current = distance.iloc[-1]
                head_current = heading.iloc[-1]
                vel_current = velocity.iloc[-1] if not pd.isna(velocity.iloc[-1]) else 0
                
                # Calculate RRG Power (distance from center 100,100)
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
        
        df_top = df.head(top_n).copy()
        
        # Cache results
        st.session_state.df_cache = df
        st.session_state.df_top_cache = df_top
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

# ============================================================================
# DISPLAY RESULTS (only if data loaded)
# ============================================================================

if st.session_state.df_cache is not None:
    df = st.session_state.df_cache
    df_top = st.session_state.df_top_cache
    
    # ========================================================================
    # MAIN LAYOUT - THREE COLUMNS
    # ========================================================================
    
    col_left, col_main, col_right = st.columns([1, 3, 1], gap="medium")
    
    # ========================================================================
    # LEFT SIDEBAR - LEGEND & STATS
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
    # MAIN CONTENT - RRG CHART & DETAILED TABLE
    # ========================================================================
    
    with col_main:
        st.markdown("## Relative Rotation Graph")
        st.markdown(f"**{csv_selected} | {tf_name} | {period_name} | Benchmark: {bench_name}**")
        
        # Create RRG chart with equal quadrants
        fig_rrg = go.Figure()
        
        x_min = df['RS-Ratio'].min() - 5
        x_max = df['RS-Ratio'].max() + 5
        y_min = df['RS-Momentum'].min() - 5
        y_max = df['RS-Momentum'].max() + 5
        
        quadrant_size = max(abs(100 - x_min), abs(x_max - 100), abs(100 - y_min), abs(y_max - 100))
        
        # Quadrant backgrounds
        fig_rrg.add_shape(type="rect", x0=100, y0=100, x1=100+quadrant_size, y1=100+quadrant_size,
                          fillcolor="rgba(34, 197, 94, 0.1)", line=dict(color="rgba(34, 197, 94, 0.3)", width=2), layer="below")
        fig_rrg.add_shape(type="rect", x0=100-quadrant_size, y0=100, x1=100, y1=100+quadrant_size,
                          fillcolor="rgba(59, 130, 246, 0.1)", line=dict(color="rgba(59, 130, 246, 0.3)", width=2), layer="below")
        fig_rrg.add_shape(type="rect", x0=100-quadrant_size, y0=100-quadrant_size, x1=100, y1=100,
                          fillcolor="rgba(251, 191, 36, 0.1)", line=dict(color="rgba(251, 191, 36, 0.3)", width=2), layer="below")
        fig_rrg.add_shape(type="rect", x0=100, y0=100-quadrant_size, x1=100+quadrant_size, y1=100,
                          fillcolor="rgba(239, 68, 68, 0.1)", line=dict(color="rgba(239, 68, 68, 0.3)", width=2), layer="below")
        
        # Quadrant labels
        fig_rrg.add_annotation(x=100+quadrant_size*0.5, y=100+quadrant_size*0.5, text="Leading", 
                              showarrow=False, font=dict(size=12, color="rgba(34, 197, 94, 0.5)"))
        fig_rrg.add_annotation(x=100-quadrant_size*0.5, y=100+quadrant_size*0.5, text="Improving", 
                              showarrow=False, font=dict(size=12, color="rgba(59, 130, 246, 0.5)"))
        fig_rrg.add_annotation(x=100-quadrant_size*0.5, y=100-quadrant_size*0.5, text="Weakening", 
                              showarrow=False, font=dict(size=12, color="rgba(251, 191, 36, 0.5)"))
        fig_rrg.add_annotation(x=100+quadrant_size*0.5, y=100-quadrant_size*0.5, text="Lagging", 
                              showarrow=False, font=dict(size=12, color="rgba(239, 68, 68, 0.5)"))
        
        # Center lines
        fig_rrg.add_hline(y=100, line_dash="dash", line_color="rgba(150, 150, 150, 0.5)", layer="below")
        fig_rrg.add_vline(x=100, line_dash="dash", line_color="rgba(150, 150, 150, 0.5)", layer="below")
        
        # Add data points by status with detailed hover
        for status in ["Leading", "Improving", "Weakening", "Lagging"]:
            df_status = df_top[df_top['Status'] == status]
            if not df_status.empty:
                hover_text = []
                for _, row in df_status.iterrows():
                    tv_link = row['TV Link']
                    hover_info = (
                        f"<b><a href='{tv_link}' target='_blank' style='color: #0066cc;'>{row['Symbol']}</a></b><br>"
                        f"<b>{row['Name']}</b><br>"
                        f"Industry: {row['Industry']}<br>"
                        f"Price: ₹{row['Price']:.2f} | {row['Change %']:+.2f}%<br>"
                        f"<b>JdK Metrics:</b><br>"
                        f"RS-Ratio: {row['RS-Ratio']:.2f} | RS-Momentum: {row['RS-Momentum']:.2f}<br>"
                        f"Distance: {row['Distance']:.2f} | Velocity: {row['Velocity']:.3f}<br>"
                        f"Heading: {row['Heading']:.1f}° {row['Direction']}<br>"
                        f"Status: <b>{row['Status']}</b>"
                    )
                    hover_text.append(hover_info)
                
                fig_rrg.add_trace(go.Scatter(
                    x=df_status['RS-Ratio'],
                    y=df_status['RS-Momentum'],
                    mode='markers+text',
                    name=status,
                    text=df_status['Symbol'],
                    textposition="top center",
                    customdata=hover_text,
                    marker=dict(
                        size=14,
                        color=QUADRANT_COLORS[status],
                        line=dict(color='white', width=2),
                        opacity=0.9
                    ),
                    hovertemplate='%{customdata}<extra></extra>'
                ))
        
        fig_rrg.update_layout(
            height=500,
            xaxis_title="RS-Ratio (X-axis)",
            yaxis_title="RS-Momentum (Y-axis)",
            plot_bgcolor='rgba(240, 240, 245, 0.9)',
            paper_bgcolor='rgba(240, 240, 245, 0.9)',
            font=dict(color='#000000', size=11),
            hovermode='closest',
            xaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)', zeroline=False),
            yaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)', zeroline=False),
            showlegend=False,
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        st.plotly_chart(fig_rrg, use_container_width=True)
        
        st.markdown("---")
        st.markdown("## Detailed Analysis")
        
        df_display = df_top[[
            'Sl No.', 'Symbol', 'Industry', 'Price', 'Change %', 'RRG Power', 'Status', 'RS-Ratio', 'RS-Momentum', 'Distance', 'Direction', 'Velocity'
        ]].copy()
        
        display_data = []
        for _, row in df_display.iterrows():
            tv_link = df_top[df_top['Symbol'] == row['Symbol']]['TV Link'].values[0]
            display_data.append({
                'Sl No.': int(row['Sl No.']),
                'Symbol': f"[{row['Symbol']}]({tv_link})",
                'Industry': row['Industry'],
                'Price': f"₹{row['Price']:.2f}",
                'Change %': f"{row['Change %']:+.2f}%",
                'Strength': f"{row['RRG Power']:.2f}",
                'Status': row['Status'],
                'RS-Ratio': f"{row['RS-Ratio']:.2f}",
                'RS-Momentum': f"{row['RS-Momentum']:.2f}",
                'Distance': f"{row['Distance']:.2f}",
                'Direction': row['Direction'],
                'Velocity': f"{row['Velocity']:.3f}"
            })
        
        st.dataframe(pd.DataFrame(display_data), use_container_width=True, height=400)
    
    # ========================================================================
    # RIGHT SIDEBAR - TOP 30 RRG POWER
    # ========================================================================
    
    with col_right:
        st.markdown("### 🚀 Top 30 RRG Power")
        
        df_ranked = df.nlargest(30, 'RRG Power')[['Sl No.', 'Symbol', 'Industry', 'RRG Power', 'Distance', 'Status']]
        
        for idx, (_, row) in enumerate(df_ranked.iterrows(), 1):
            status_color = QUADRANT_COLORS.get(row['Status'], '#808080')
            tv_link = df[df['Symbol'] == row['Symbol']]['TV Link'].values[0]
            
            st.markdown(
                f"""
                <div style='padding: 8px; margin-bottom: 6px; background: rgba(200,200,200,0.1); border-left: 3px solid {status_color}; border-radius: 4px;'>
                    <small><b><a href="{tv_link}" target="_blank" style='color: #0066cc; text-decoration: none;'>#{int(row["Sl No."])}</a></b></small>
                    <br><b style='color: {status_color};'>{row['Symbol']}</b>
                    <br><small>{row['Industry'][:18]}</small>
                    <br><small style='color: {status_color};'>💪 Power: {row['RRG Power']:.2f}</small>
                    <br><small style='color: {status_color};'>📏 Dist: {row['Distance']:.2f}</small>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Export CSV
        if export_csv:
            df_export = df_top[[
                'Sl No.', 'Symbol', 'Industry', 'Price', 'Change %', 
                'RRG Power', 'RS-Ratio', 'RS-Momentum', 'Distance', 'Heading', 'Direction', 'Velocity', 'Status'
            ]].copy()
            
            csv_buffer = io.StringIO()
            df_export.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            b64_csv = base64.b64encode(csv_data.encode()).decode()
            
            st.markdown(
                f'<a href="data:file/csv;base64,{b64_csv}" download="RRG_Analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv" style="display: inline-block; padding: 10px 12px; background: #22c55e; color: #000; border-radius: 6px; text-decoration: none; font-weight: 600; width: 100%; text-align: center; font-size: 12px;">📥 Download CSV</a>',
                unsafe_allow_html=True
            )
else:
    st.info("👈 Select indices and click **📥 Load Data** to start analysis")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 10px;'>
    💡 RRG Analysis Dashboard • Data: Yahoo Finance • Charts: TradingView<br>
    📊 JdK Metrics: RS-Ratio (X) | RS-Momentum (Y) | Distance | Heading (°) | Velocity<br>
    🗓️ Weekly: Last Friday Close | Monthly: Last Day of Month Close | Daily: Day Close<br>
    Reference: <a href='https://www.optuma.com/blog/scripting-for-rrgs' target='_blank' style='color: #0066cc;'>Optuma RRG Scripting Guide</a><br>
    <i>Disclaimer: For educational purposes only. Not financial advice.</i>
</div>
""", unsafe_allow_html=True)

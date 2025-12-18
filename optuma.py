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

# Dark theme with status color styling
st.markdown("""
<style>
    body { background-color: #111827; }
    .main { background-color: #111827; }
    [data-testid="stSidebar"] { background-color: #1f2937; }
    
    /* Status color backgrounds */
    .status-leading {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.3), rgba(34, 197, 94, 0.15));
        padding: 4px 8px;
        border-radius: 4px;
        color: #22c55e;
        font-weight: bold;
    }
    .status-improving {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.3), rgba(59, 130, 246, 0.15));
        padding: 4px 8px;
        border-radius: 4px;
        color: #3b82f6;
        font-weight: bold;
    }
    .status-weakening {
        background: linear-gradient(135deg, rgba(251, 191, 36, 0.3), rgba(251, 191, 36, 0.15));
        padding: 4px 8px;
        border-radius: 4px;
        color: #fbbf24;
        font-weight: bold;
    }
    .status-lagging {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.3), rgba(239, 68, 68, 0.15));
        padding: 4px 8px;
        border-radius: 4px;
        color: #ef4444;
        font-weight: bold;
    }
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

def get_status_html(status):
    """Return HTML styled status badge"""
    status_class = f"status-{status.lower()}"
    return f'<span class="{status_class}">{status}</span>'

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
    csv_files = ["NIFTY200"]

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

bench_list = list(BENCHMARKS.keys())
default_bench_index = 2
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
# LOAD & CLEAR BUTTONS - HORIZONTAL LAYOUT
# ============================================================================
st.sidebar.markdown("---")

# Horizontal Load Button (full width)
if st.sidebar.button("📥 Load Data", use_container_width=True, key="load_btn", type="primary"):
    st.session_state.load_clicked = True

# Horizontal Clear Button (full width below Load)
if st.sidebar.button("🔄 Clear", use_container_width=True, key="clear_btn"):
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
st.sidebar.markdown(f"**Benchmark: {bench_name}**")  # Display benchmark name
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
                rs_ratio, rs_momentum, distance, heading, velocity = calculate_jdk_rrg(
                    raw['Close'][s], bench, window=WINDOW
                )
                
                if rs_ratio is None or len(rs_ratio) < 3:
                    failed_count += 1
                    continue
                
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
        df_top = df.head(top_n).copy()
        
        st.session_state.df_cache = df
        st.session_state.df_top_cache = df_top
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

# ============================================================================
# DISPLAY RESULTS
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
        
        # Create RRG chart with dark labels for better visualization
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
        
        # Quadrant labels with DARK colors for visualization
        fig_rrg.add_annotation(x=100+quadrant_size*0.5, y=100+quadrant_size*0.5, text="Leading",
                              showarrow=False, font=dict(size=14, color="#166534", family="Arial Black"))
        fig_rrg.add_annotation(x=100-quadrant_size*0.5, y=100+quadrant_size*0.5, text="Improving",
                              showarrow=False, font=dict(size=14, color="#1e3a8a", family="Arial Black"))
        fig_rrg.add_annotation(x=100-quadrant_size*0.5, y=100-quadrant_size*0.5, text="Weakening",
                              showarrow=False, font=dict(size=14, color="#92400e", family="Arial Black"))
        fig_rrg.add_annotation(x=100+quadrant_size*0.5, y=100-quadrant_size*0.5, text="Lagging",
                              showarrow=False, font=dict(size=14, color="#7f1d1d", family="Arial Black"))
        
        # Center lines
        fig_rrg.add_hline(y=100, line_dash="dash", line_color="rgba(150, 150, 150, 0.5)", layer="below")
        fig_rrg.add_vline(x=100, line_dash="dash", line_color="rgba(150, 150, 150, 0.5)", layer="below")
        
        # Add data points
        for status in ["Leading", "Improving", "Weakening", "Lagging"]:
            df_status = df_top[df_top['Status'] == status]
            if not df_status.empty:
                hover_text = []
                for _, row in df_status.iterrows():
                    hover_info = (
                        f"<b>{row['Symbol']}</b><br>"
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
                    hovertemplate='%{customdata}<extra></extra>',
                    textfont=dict(color='#000000', size=10, family='Arial Black')  # Dark text for symbols
                ))
        
        fig_rrg.update_layout(
            height=500,
            xaxis_title="RS-Ratio (X-axis)",
            yaxis_title="RS-Momentum (Y-axis)",
            plot_bgcolor="rgba(240, 240, 245, 0.9)",
            paper_bgcolor="rgba(240, 240, 245, 0.9)",
            font=dict(color="#000000", size=11),
            hovermode="closest",
            xaxis=dict(gridcolor="rgba(200, 200, 200, 0.3)", zeroline=False),
            yaxis=dict(gridcolor="rgba(200, 200, 200, 0.3)", zeroline=False),
            showlegend=False,
            margin=dict(l=80, r=80, t=100, b=80)
        )
        
        st.plotly_chart(fig_rrg, use_container_width=True)
        
        st.markdown("---")
        st.markdown("## Detailed Analysis")
        
        # Create table with clickable symbols and colored status
        display_data = []
        for _, row in df_top.iterrows():
            # Create clickable TradingView link for symbol
            symbol_link = f'<a href="{row["TV Link"]}" target="_blank" style="color: #0066cc; text-decoration: none; font-weight: bold;">{row["Symbol"]}</a>'
            status_html = get_status_html(row['Status'])
            
            display_data.append({
                'Sl No.': int(row['Sl No.']),
                'Symbol': symbol_link,
                'Industry': row['Industry'],
                'Price': f"₹{row['Price']:.2f}",
                'Change %': f"{row['Change %']:+.2f}%",
                'Strength': f"{row['RRG Power']:.2f}",
                'Status': status_html,
                'RS-Ratio': f"{row['RS-Ratio']:.2f}",
                'RS-Momentum': f"{row['RS-Momentum']:.2f}",
                'Distance': f"{row['Distance']:.2f}",
                'Direction': row['Direction'],
                'Velocity': f"{row['Velocity']:.3f}"
            })
        
        # Display as HTML table for proper rendering
        table_html = pd.DataFrame(display_data).to_html(escape=False, index=False)
        st.markdown(table_html, unsafe_allow_html=True)
    
    # ========================================================================
    # RIGHT SIDEBAR - TOP 30 RRG POWER (COLLAPSIBLE)
    # ========================================================================
    with col_right:
        st.markdown("### 🚀 Top 30 RRG Power")
        
        df_ranked = df.nlargest(30, 'RRG Power')[['Sl No.', 'Symbol', 'Industry', 'RRG Power', 'Distance', 'Status']]
        
        # Group by Status for collapsible sections
        for status in ["Leading", "Improving", "Weakening", "Lagging"]:
            df_status_group = df_ranked[df_ranked['Status'] == status]
            
            if not df_status_group.empty:
                status_color = QUADRANT_COLORS.get(status, "#808080")
                status_icon = {"Leading": "🟢", "Improving": "🔵", "Weakening": "🟡", "Lagging": "🔴"}[status]
                
                # Collapsible expander for each quadrant
                with st.expander(f"{status_icon} **{status}** ({len(df_status_group)})", expanded=(status == "Leading")):
                    for idx, (_, row) in enumerate(df_status_group.iterrows(), 1):
                        tv_link = df[df['Symbol'] == row['Symbol']]['TV Link'].values[0]
                        
                        st.markdown(f"""
                        <div style="padding: 8px; margin-bottom: 6px; background: rgba(200,200,200,0.1); 
                                    border-left: 3px solid {status_color}; border-radius: 4px;">
                            <small><b><a href="{tv_link}" target="_blank" 
                                style="color: #0066cc; text-decoration: none;">#{int(row['Sl No.'])}</a></b></small>
                            <br><b style="color: {status_color};">{row['Symbol']}</b>
                            <br><small>{row['Industry'][:18]}</small>
                            <br><small style="color: {status_color};">⚡ Power: {row['RRG Power']:.2f}</small>
                            <br><small style="color: {status_color};">📏 Dist: {row['Distance']:.2f}</small>
                        </div>
                        """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # ========================================================================
    # FOOTER & EXPORT
    # ========================================================================
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align: center; color: #888; font-size: 10px;">
        <b>RRG Analysis Dashboard</b><br>
        Data: Yahoo Finance | Charts: TradingView<br>
        JdK Metrics: RS-Ratio (X) • RS-Momentum (Y) • Distance • Heading • Velocity<br>
        Weekly: Last Friday Close | Monthly: Last Day of Month Close | Daily: Day Close<br>
        Reference: <a href="https://www.optuma.com/blog/scripting-for-rrgs" target="_blank" 
                      style="color: #0066cc;">Optuma RRG Scripting Guide</a><br>
        <i>Disclaimer: For educational purposes only. Not financial advice.</i>
    </div>
    """, unsafe_allow_html=True)
    
    # Export CSV
    if export_csv:
        df_export = df_top[['Sl No.', 'Symbol', 'Industry', 'Price', 'Change %', 'RRG Power', 
                            'RS-Ratio', 'RS-Momentum', 'Distance', 'Heading', 'Direction', 
                            'Velocity', 'Status']].copy()
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        b64_csv = base64.b64encode(csv_data.encode()).decode()
        
        st.markdown(f"""
        <a href="data:file/csv;base64,{b64_csv}" 
           download="RRG_Analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv" 
           style="display: inline-block; padding: 10px 12px; background: #22c55e; color: #000; 
                  border-radius: 6px; text-decoration: none; font-weight: 600; width: 100%; 
                  text-align: center; font-size: 12px;">
           📥 Download CSV
        </a>
        """, unsafe_allow_html=True)

else:
    st.info("⬅️ Select indices and click **Load Data** to start analysis")

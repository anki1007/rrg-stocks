import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import requests
from urllib.parse import quote
from datetime import datetime, timedelta
import json

# ============================================================================
# CONFIG & SETUP
# ============================================================================

st.set_page_config(
    layout="wide",
    page_title="RRG Dashboard - Multi-Index Analysis",
    initial_sidebar_state="expanded"
)

# Theme Configuration
st.markdown("""
    <style>
        :root {
            --improving-color: #3b82f6;
            --leading-color: #22c55e;
            --weakening-color: #facc15;
            --lagging-color: #ef4444;
        }
        [data-testid="stMetricValue"] {
            font-size: 20px;
        }
        .stSelectbox > div > div {
            font-size: 14px;
        }
    </style>
""", unsafe_allow_html=True)

# Define Benchmarks
BENCHMARKS = {
    "NIFTY 50": "^NSEI",
    "NIFTY 200": "^CNX200",
    "NIFTY 500": "^CRSLDX"  # Adjusted for available data
}

# Define Timeframes with intervals and yfinance periods
TIMEFRAMES = {
    "5 min close": ("5m", "60d"),
    "15 min close": ("15m", "60d"),
    "30 min close": ("30m", "60d"),
    "1 hr close": ("60m", "90d"),
    "Daily": ("1d", "5y"),
    "Weekly": ("1wk", "10y"),
    "Monthly": ("1mo", "20y")
}

# Period mapping to number of periods
PERIOD_MAP = {
    "6M": 126,
    "1Y": 252,
    "2Y": 504,
    "3Y": 756,
    "5Y": 1260,
    "10Y": 2520
}

# RRG Configuration
WINDOW = 14
TAIL = 8

# Color mapping for quadrants
QUADRANT_COLORS = {
    "Improving": "#3b82f6",
    "Leading": "#22c55e",
    "Weakening": "#facc15",
    "Lagging": "#ef4444"
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=600)
def list_csv_from_github():
    """Fetch CSV filenames from GitHub repository"""
    url = "https://api.github.com/repos/anki1007/rrg-stocks/contents/ticker"
    try:
        response = requests.get(url)
        files = [f['name'].replace('.csv', '').upper() for f in response.json() 
                if f['name'].endswith('.csv')]
        return sorted(files)
    except Exception as e:
        st.error(f"Error fetching CSV list: {e}")
        return []

@st.cache_data(ttl=600)
def load_universe(csv_name):
    """Load stock universe from GitHub CSV"""
    url = f"https://raw.githubusercontent.com/anki1007/rrg-stocks/main/ticker/{csv_name.lower()}.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading {csv_name} universe: {e}")
        return pd.DataFrame()

def rrg_calc(px, bench):
    """Calculate RRG metrics: RS-Ratio and RS-Momentum"""
    df = pd.concat([px, bench], axis=1).dropna()
    
    if len(df) < WINDOW + 2:
        return None, None
    
    # RS-Ratio: (Stock / Benchmark) - normalized
    rs = 100 * (df.iloc[:, 0] / df.iloc[:, 1])
    rs_ratio = 100 * (rs - rs.rolling(WINDOW).mean()) / rs.rolling(WINDOW).std(ddof=0)
    
    # RS-Momentum: Rate of change of RS-Ratio
    roc = rs_ratio.pct_change() * 100
    rs_momentum = 101 * (roc - roc.rolling(WINDOW).mean()) / roc.rolling(WINDOW).std(ddof=0)
    
    return rs_ratio.dropna(), rs_momentum.dropna()

def quadrant(x, y):
    """Determine RRG quadrant based on RS-Ratio and RS-Momentum"""
    if x > 100 and y > 100:
        return "Leading"
    elif x < 100 and y > 100:
        return "Improving"
    elif x < 100 and y < 100:
        return "Weakening"
    else:
        return "Lagging"

def get_tv_link(sym):
    """Generate TradingView chart link"""
    return f"https://www.tradingview.com/chart/?symbol=NSE:{sym.replace('.NS', '')}"

def format_symbol(sym):
    """Remove .NS suffix from symbol"""
    return sym.replace('.NS', '')

def calculate_price_change(current_price, historical_price):
    """Calculate percentage change"""
    if historical_price == 0:
        return 0
    return ((current_price - historical_price) / historical_price) * 100

# ============================================================================
# SIDEBAR CONTROLS - REORDERED SEQUENCE
# ============================================================================

st.sidebar.markdown("### ⚙️ Controls")

# 1. INDICES (Dropdown)
csv_files = list_csv_from_github()
csv_selected = st.sidebar.selectbox(
    "📊 Indices",
    csv_files,
    help="Select stock universe from available CSVs"
)

# 2. BENCHMARK (Dropdown)
bench_name = st.sidebar.selectbox(
    "🎯 Benchmark",
    list(BENCHMARKS.keys()),
    help="Select benchmark index for relative strength"
)

# 3. STRENGTH vs TIMEFRAME (Dropdown)
tf_name = st.sidebar.selectbox(
    "⏱️ Strength vs Timeframe",
    list(TIMEFRAMES.keys()),
    help="Select timeframe for RRG analysis"
)

# 4. PERIOD (Dropdown)
period_name = st.sidebar.selectbox(
    "📅 Period",
    list(PERIOD_MAP.keys()),
    help="Select analysis period"
)

# 5. RANK BY (Dropdown)
rank_by = st.sidebar.selectbox(
    "🏆 Rank by",
    ["RRG Power", "RS-Ratio", "RS-Momentum", "Price % Change", "Momentum Slope"],
    help="Select ranking metric"
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
    
    symbols = universe['Symbol'].tolist()
    names_dict = dict(zip(universe['Symbol'], universe['Company Name']))
    industries_dict = dict(zip(universe['Symbol'], universe['Industry']))
    
    # Download data for benchmark and all stocks
    raw = yf.download(
        symbols + [BENCHMARKS[bench_name]],
        interval=interval,
        period=yf_period,
        auto_adjust=True,
        progress=False,
        threads=True
    )
    
    bench = raw['Close'][BENCHMARKS[bench_name]]
    
    rows = []
    trails = {}
    
    for s in symbols:
        if s not in raw['Close'].columns:
            continue
        
        rr, mm = rrg_calc(raw['Close'][s], bench)
        
        if rr is None or mm is None:
            continue
        
        rr_tail = rr.iloc[-TAIL:]
        mm_tail = mm.iloc[-TAIL:]
        
        if len(rr_tail) < 3:
            continue
        
        # Calculate momentum slope using polyfit
        slope = np.polyfit(range(len(mm_tail)), mm_tail.values, 1)[0]
        
        # Calculate RRG Power: combined strength metric
        power = np.sqrt((rr_tail.iloc[-1] - 100) ** 2 + (mm_tail.iloc[-1] - 100) ** 2)
        
        # Calculate price change
        current_price = raw['Close'][s].iloc[-1]
        historical_price = raw['Close'][s].iloc[max(0, len(raw['Close'][s]) - PERIOD_MAP[period_name])]
        price_change = calculate_price_change(current_price, historical_price)
        
        status = quadrant(rr_tail.iloc[-1], mm_tail.iloc[-1])
        tv_link = get_tv_link(s)
        
        rows.append({
            'Symbol': s,
            'Name': names_dict.get(s, s),
            'Industry': industries_dict.get(s, 'N/A'),
            'Price': round(current_price, 2),
            'Change %': round(price_change, 2),
            'RS-Ratio': round(rr_tail.iloc[-1], 2),
            'RS-Momentum': round(mm_tail.iloc[-1], 2),
            'Momentum Slope': round(slope, 2),
            'RRG Power': round(power, 2),
            'Status': status,
            'TV Link': tv_link
        })
        
        trails[s] = (rr_tail, mm_tail)
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        st.error("No data available. Try another combination.")
        st.stop()
    
    # Rank by selected metric
    rank_col_map = {
        "RRG Power": "RRG Power",
        "RS-Ratio": "RS-Ratio",
        "RS-Momentum": "RS-Momentum",
        "Price % Change": "Change %",
        "Momentum Slope": "Momentum Slope"
    }
    
    rank_column = rank_col_map[rank_by]
    df['Rank'] = df[rank_column].rank(ascending=False, method='min').astype(int)
    df = df.sort_values('Rank')
    df['Sl No.'] = range(1, len(df) + 1)
    
except Exception as e:
    st.error(f"Error in data processing: {str(e)}")
    st.stop()

# ============================================================================
# MAIN LAYOUT - THREE COLUMNS
# ============================================================================

col_left, col_main, col_right = st.columns([1, 3, 1])

# ============================================================================
# LEFT SIDEBAR - LEGEND
# ============================================================================

with col_left:
    st.markdown("### 📍 Legend")
    
    for status, color in QUADRANT_COLORS.items():
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; margin: 8px 0;">
                <div style="width: 16px; height: 16px; background-color: {color}; border-radius: 50%; margin-right: 8px;"></div>
                <span>{status}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    st.markdown("### 📈 Stats")
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Total Stocks", len(df))
        st.metric("Leading", len(df[df['Status'] == 'Leading']))
    with col_stat2:
        st.metric("Improving", len(df[df['Status'] == 'Improving']))
        st.metric("Weakening", len(df[df['Status'] == 'Weakening']))
    
    st.metric("Lagging", len(df[df['Status'] == 'Lagging']))

# ============================================================================
# MAIN AREA - RRG GRAPH & TABLE
# ============================================================================

with col_main:
    # RRG SCATTER PLOT WITH PLOTLY
    st.markdown("### 📊 Relative Rotation Graph")
    
    fig = go.Figure()
    
    # Add quadrant background rectangles
    fig.add_shape(type="rect", x0=100, y0=100, x1=df['RS-Ratio'].max() + 10, y1=df['RS-Momentum'].max() + 10,
                  fillcolor="#22c55e", opacity=0.05, line_width=0, name="Leading")
    fig.add_shape(type="rect", x0=df['RS-Ratio'].min() - 10, y0=100, x1=100, y1=df['RS-Momentum'].max() + 10,
                  fillcolor="#3b82f6", opacity=0.05, line_width=0, name="Improving")
    fig.add_shape(type="rect", x0=df['RS-Ratio'].min() - 10, y0=df['RS-Momentum'].min() - 10, x1=100, y1=100,
                  fillcolor="#ef4444", opacity=0.05, line_width=0, name="Lagging")
    fig.add_shape(type="rect", x0=100, y0=df['RS-Momentum'].min() - 10, x1=df['RS-Ratio'].max() + 10, y1=100,
                  fillcolor="#facc15", opacity=0.05, line_width=0, name="Weakening")
    
    # Add center lines
    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=100, line_dash="dash", line_color="gray", opacity=0.5)
    
    # Add scatter points for each quadrant
    for status in ["Leading", "Improving", "Weakening", "Lagging"]:
        df_status = df[df['Status'] == status]
        
        fig.add_trace(go.Scatter(
            x=df_status['RS-Ratio'],
            y=df_status['RS-Momentum'],
            mode='markers+text',
            name=status,
            marker=dict(
                size=8,
                color=QUADRANT_COLORS[status],
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            text=[format_symbol(s) for s in df_status['Symbol']],
            textposition="top center",
            textfont=dict(size=9, color="black"),
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         'Company: %{customdata[1]}<br>' +
                         'RS-Ratio: %{x:.2f}<br>' +
                         'RS-Momentum: %{y:.2f}<br>' +
                         'Status: ' + status +
                         '<extra></extra>',
            customdata=df_status[['Name', 'Industry']].values
        ))
    
    fig.update_layout(
        title=f"{bench_name} | {tf_name} | {period_name}",
        xaxis_title="RS-Ratio",
        yaxis_title="RS-Momentum",
        hovermode='closest',
        height=500,
        template="plotly_white",
        showlegend=True,
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
        xaxis=dict(zeroline=False),
        yaxis=dict(zeroline=False)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # DATA TABLE
    st.markdown("### 📋 Detailed Analysis")
    
    with st.expander("Results Table (Click to expand/collapse)", expanded=True):
        # Create color-coded dataframe for display
        display_df = df[['Sl No.', 'Symbol', 'Name', 'Status', 'Industry', 'Price', 'Change %', 
                         'RS-Ratio', 'RS-Momentum', 'Momentum Slope', 'RRG Power', 'Rank']].copy()
        
        # Format display symbols without .NS
        display_df['Symbol'] = display_df['Symbol'].apply(format_symbol)
        
        # Create styled dataframe
        def highlight_status(val):
            if val == 'Leading':
                return 'background-color: #22c55e; color: white'
            elif val == 'Improving':
                return 'background-color: #3b82f6; color: white'
            elif val == 'Weakening':
                return 'background-color: #facc15; color: black'
            elif val == 'Lagging':
                return 'background-color: #ef4444; color: white'
            return ''
        
        # Display with sorting capability
        st.dataframe(
            display_df.style.applymap(highlight_status, subset=['Status']),
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results (CSV)",
            data=csv,
            file_name=f"rrg_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

# ============================================================================
# RIGHT SIDEBAR - TOP 30 RS MOMENTUM
# ============================================================================

with col_right:
    st.markdown("### 🚀 Top 30 RS-Momentum")
    
    with st.expander("Top Performers", expanded=True):
        top30 = df.nlargest(30, 'RS-Momentum')[['Sl No.', 'Symbol', 'Name', 'RS-Momentum', 'Status']].copy()
        top30['Symbol'] = top30['Symbol'].apply(format_symbol)
        
        # Color code status
        top30_styled = top30.style.applymap(
            lambda val: 'background-color: #22c55e; color: white' if val == 'Leading' else
                       'background-color: #3b82f6; color: white' if val == 'Improving' else
                       'background-color: #facc15; color: black' if val == 'Weakening' else
                       'background-color: #ef4444; color: white' if val == 'Lagging' else '',
            subset=['Status']
        )
        
        st.dataframe(top30_styled, use_container_width=True, height=600)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; font-size: 12px; color: gray;">
    <b>RRG Dashboard v2.0</b> | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')} |
    Data Source: Yahoo Finance | Benchmarks: {bench_name}
    </div>
    """,
    unsafe_allow_html=True
)

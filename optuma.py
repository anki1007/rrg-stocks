"""
RRG Dashboard - Streamlit Multi-Index Analysis
Minimal, Clean UI matching reference design
"""

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import requests
from datetime import datetime
import json
import io
import base64

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    layout="wide",
    page_title="RRG Dashboard - Multi-Index Analysis",
    initial_sidebar_state="expanded"
)

# Minimal theme
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

WINDOW = 14
TAIL = 8

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
    """Calculate RRG metrics"""
    df = pd.concat([px, bench], axis=1).dropna()
    if len(df) < WINDOW + 2:
        return None, None
    
    rs = 100 * (df.iloc[:, 0] / df.iloc[:, 1])
    rs_ratio = 100 * (rs - rs.rolling(WINDOW).mean()) / rs.rolling(WINDOW).std(ddof=0)
    roc = rs_ratio.pct_change() * 100
    rs_momentum = 101 * (roc - roc.rolling(WINDOW).mean()) / roc.rolling(WINDOW).std(ddof=0)
    
    return rs_ratio.dropna(), rs_momentum.dropna()

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
# SIDEBAR - CONTROLS
# ============================================================================

st.sidebar.markdown("### ⚙️ Controls")

csv_files = list_csv_from_github()
csv_selected = st.sidebar.selectbox("Indices", csv_files)

bench_name = st.sidebar.selectbox("Benchmark", list(BENCHMARKS.keys()))

tf_name = st.sidebar.selectbox("Strength vs Timeframe", list(TIMEFRAMES.keys()))

period_name = st.sidebar.selectbox("Period", list(PERIOD_MAP.keys()))

rank_by = st.sidebar.selectbox("Rank by", ["RRG Power", "RS-Ratio", "RS-Momentum", "Price % Change", "Momentum Slope"])

st.sidebar.markdown("---")

top_n = st.sidebar.slider("Show Top N", min_value=5, max_value=50, value=15)

st.sidebar.markdown("---")

# Export
export_csv = st.sidebar.checkbox("Export CSV", value=True)
play_animation = st.sidebar.checkbox("Play Animation", value=False)
animation_speed = st.sidebar.slider("Animation Speed (ms)", 100, 2000, 500, 100)

st.sidebar.markdown("---")

st.sidebar.markdown("### 📍 Legend")
st.sidebar.markdown("🟢 **Leading**: Strong RS, ↑ Momentum")
st.sidebar.markdown("🔵 **Improving**: Weak RS, ↑ Momentum")
st.sidebar.markdown("🟡 **Weakening**: Weak RS, ↓ Momentum")
st.sidebar.markdown("🔴 **Lagging**: Strong RS, ↓ Momentum")

# ============================================================================
# DATA LOADING
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
    
    with st.spinner(f"Downloading data for {len(symbols)} symbols..."):
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
        
        slope = np.polyfit(range(len(mm_tail)), mm_tail.values, 1)[0]
        power = np.sqrt((rr_tail.iloc[-1] - 100) ** 2 + (mm_tail.iloc[-1] - 100) ** 2)
        
        current_price = raw['Close'][s].iloc[-1]
        historical_price = raw['Close'][s].iloc[max(0, len(raw['Close'][s]) - PERIOD_MAP[period_name])]
        price_change = calculate_price_change(current_price, historical_price)
        
        status = quadrant(rr_tail.iloc[-1], mm_tail.iloc[-1])
        
        rows.append({
            'Symbol': format_symbol(s),
            'Name': names_dict.get(s, s),
            'Industry': industries_dict.get(s, 'N/A'),
            'Price': round(current_price, 2),
            'Change %': round(price_change, 2),
            'RS-Ratio': round(rr_tail.iloc[-1], 2),
            'RS-Momentum': round(mm_tail.iloc[-1], 2),
            'Momentum Slope': round(slope, 2),
            'RRG Power': round(power, 2),
            'Status': status,
            'TV Link': get_tv_link(s),
            'RR_Current': rr_tail.iloc[-1],
            'MM_Current': mm_tail.iloc[-1]
        })
        
        trails[format_symbol(s)] = (rr_tail, mm_tail)
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        st.error("No data available. Try another combination.")
        st.stop()
    
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
    
    df_top = df.head(top_n).copy()
    
except Exception as e:
    st.error(f"Error in data processing: {str(e)}")
    st.stop()

# ============================================================================
# MAIN LAYOUT - THREE COLUMNS (MATCHING REFERENCE UI)
# ============================================================================

col_left, col_main, col_right = st.columns([1, 3, 1], gap="medium")

# ============================================================================
# LEFT SIDEBAR - LEGEND & STATS
# ============================================================================

with col_left:
    st.markdown("### Legend")
    
    status_counts = df['Status'].value_counts()
    status_colors_map = {"Leading": "🟢", "Improving": "🔵", "Weakening": "🟡", "Lagging": "🔴"}
    
    for status in ["Leading", "Improving", "Weakening", "Lagging"]:
        count = status_counts.get(status, 0)
        st.markdown(f"{status_colors_map[status]} {status}: {count}")
    
    st.markdown("---")
    st.markdown("### Stats")
    
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.metric("Total Stocks", len(df))
        st.metric("Leading", len(df[df['Status'] == 'Leading']))
    with col_stat2:
        st.metric("Improving", len(df[df['Status'] == 'Improving']))
        st.metric("Weakening", len(df[df['Status'] == 'Weakening']))
    
    st.metric("Lagging", len(df[df['Status'] == 'Lagging']))

# ============================================================================
# MAIN CONTENT - RRG CHART & TABLE
# ============================================================================

with col_main:
    st.markdown("## Relative Rotation Graph")
    st.markdown(f"**{csv_selected} | 5 min close | 6M**")
    
    # Create RRG chart with LIGHT background for visibility
    fig_rrg = go.Figure()
    
    # Add quadrant backgrounds with visible colors
    fig_rrg.add_shape(type="rect", x0=100, y0=100, x1=df['RS-Ratio'].max() + 5, y1=df['RS-Momentum'].max() + 5,
                      fillcolor="rgba(34, 197, 94, 0.1)", line=dict(color="rgba(34, 197, 94, 0.3)", width=2), layer="below")
    fig_rrg.add_shape(type="rect", x0=df['RS-Ratio'].min() - 5, y0=100, x1=100, y1=df['RS-Momentum'].max() + 5,
                      fillcolor="rgba(59, 130, 246, 0.1)", line=dict(color="rgba(59, 130, 246, 0.3)", width=2), layer="below")
    fig_rrg.add_shape(type="rect", x0=df['RS-Ratio'].min() - 5, y0=df['RS-Momentum'].min() - 5, x1=100, y1=100,
                      fillcolor="rgba(251, 191, 36, 0.1)", line=dict(color="rgba(251, 191, 36, 0.3)", width=2), layer="below")
    fig_rrg.add_shape(type="rect", x0=100, y0=df['RS-Momentum'].min() - 5, x1=df['RS-Ratio'].max() + 5, y1=100,
                      fillcolor="rgba(239, 68, 68, 0.1)", line=dict(color="rgba(239, 68, 68, 0.3)", width=2), layer="below")
    
    # Add center lines
    fig_rrg.add_hline(y=100, line_dash="dash", line_color="rgba(150, 150, 150, 0.5)", layer="below")
    fig_rrg.add_vline(x=100, line_dash="dash", line_color="rgba(150, 150, 150, 0.5)", layer="below")
    
    # Add data points by status
    for status in ["Leading", "Improving", "Weakening", "Lagging"]:
        df_status = df_top[df_top['Status'] == status]
        if not df_status.empty:
            fig_rrg.add_trace(go.Scatter(
                x=df_status['RS-Ratio'],
                y=df_status['RS-Momentum'],
                mode='markers+text',
                name=status,
                text=df_status['Symbol'],
                textposition="top center",
                marker=dict(
                    size=12,
                    color=QUADRANT_COLORS[status],
                    line=dict(color='white', width=1.5),
                    opacity=0.9
                ),
                hovertemplate='<b>%{text}</b><br>RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<extra></extra>'
            ))
    
    fig_rrg.update_layout(
        height=450,
        xaxis_title="RS-Ratio",
        yaxis_title="RS-Momentum",
        plot_bgcolor='rgba(240, 240, 245, 0.9)',
        paper_bgcolor='rgba(240, 240, 245, 0.9)',
        font=dict(color='#000000', size=11),
        hovermode='closest',
        xaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)', zeroline=False),
        yaxis=dict(gridcolor='rgba(200, 200, 200, 0.3)', zeroline=False),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255, 255, 255, 0.8)', bordercolor='#999', borderwidth=1)
    )
    
    st.plotly_chart(fig_rrg, use_container_width=True)
    
    st.markdown("---")
    st.markdown("## Detailed Analysis")
    
    with st.expander("Results Table (Click to expand/collapse)", expanded=True):
        df_display = df_top[[
            'Sl No.', 'Symbol', 'Name', 'Status', 'Industry', 'Price', 'Change %', 'RS-Ratio', 'RS-Momentum'
        ]].copy()
        
        # Display table with proper formatting
        display_data = []
        for idx, row in df_display.iterrows():
            tv_link = df_top.iloc[idx]['TV Link']
            display_data.append({
                'Sl No.': int(row['Sl No.']),
                'Symbol': f"[{row['Symbol']}]({tv_link})",
                'Name': row['Name'],
                'Status': row['Status'],
                'Industry': row['Industry'],
                'Price': f"₹{row['Price']:.2f}",
                'Change %': f"{row['Change %']:+.2f}%",
                'RS-Ratio': f"{row['RS-Ratio']:.2f}",
                'RS-Momentum': f"{row['RS-Momentum']:.2f}"
            })
        
        st.dataframe(pd.DataFrame(display_data), use_container_width=True)

# ============================================================================
# RIGHT SIDEBAR - TOP 30 RS-MOMENTUM
# ============================================================================

with col_right:
    st.markdown("### Top 30 RS-Momentum")
    
    df_ranked = df.nlargest(30, 'RS-Momentum')[['Sl No.', 'Symbol', 'Name', 'RS-Momentum', 'Status']]
    
    # Create styled display
    for idx, (_, row) in enumerate(df_ranked.iterrows(), 1):
        status_color = QUADRANT_COLORS.get(row['Status'], '#808080')
        tv_link = df[df['Symbol'] == row['Symbol']]['TV Link'].values[0]
        
        st.markdown(
            f"""
            <div style='padding: 8px; margin-bottom: 8px; background: rgba(200,200,200,0.1); border-left: 3px solid {status_color}; border-radius: 4px;'>
                <b><a href="{tv_link}" target="_blank" style='color: #0066cc; text-decoration: none;'>{int(row["Sl No."])}</a></b> 
                <span style='color: {status_color}; font-weight: bold;'>{row['Symbol']}</span>
                <br><small>{row['Name']}</small>
                <br><span style='color: {status_color};'>{row['RS-Momentum']:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    
    # Export CSV
    if export_csv:
        df_export = df_top[[
            'Sl No.', 'Symbol', 'Name', 'Industry', 'Price', 'Change %', 
            'RS-Ratio', 'RS-Momentum', 'Momentum Slope', 'RRG Power', 'Status'
        ]].copy()
        
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        b64_csv = base64.b64encode(csv_data.encode()).decode()
        
        st.markdown(
            f'<a href="data:file/csv;base64,{b64_csv}" download="RRG_Analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv" style="display: inline-block; padding: 10px 16px; background: #22c55e; color: #000; border-radius: 6px; text-decoration: none; font-weight: 600; width: 100%; text-align: center;">📥 Download CSV</a>',
            unsafe_allow_html=True
        )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 11px;'>
    💡 RRG Analysis Dashboard • Data: Yahoo Finance • Charts: TradingView<br>
    <i>Disclaimer: For educational purposes only. Not financial advice.</i>
</div>
""", unsafe_allow_html=True)


"""
RRG Dashboard - Advanced Streamlit Multi-Index Analysis
Includes: Live RRG Charts, Sector Rotation Animation, CSV/Chart Export, TradingView Integration
"""

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
import io
from PIL import Image
import base64

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    layout="wide",
    page_title="RRG Dashboard - Advanced Analysis",
    initial_sidebar_state="expanded",
    menu_items={"About": "RRG Dashboard with Sector Rotation & Export Features"}
)

# Dark theme CSS
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: rgba(0, 212, 255, 0.05); border-radius: 8px; padding: 12px; }
    .stMetric [data-testid="metric-container"] { background-color: transparent; }
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    .dataframe { font-size: 12px !important; }
    .leading { color: #22c55e; }
    .improving { color: #3b82f6; }
    .weakening { color: #fbbf24; }
    .lagging { color: #ef4444; }
    .export-btn { display: inline-block; padding: 8px 16px; background: #00d4ff; color: #000; border-radius: 6px; text-decoration: none; font-weight: 600; margin-right: 8px; }
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

SECTOR_COLORS = {
    "IT": "#00d4ff",
    "Finance": "#22c55e",
    "Energy": "#fbbf24",
    "Healthcare": "#ef4444",
    "Auto": "#3b82f6",
    "Pharma": "#ec4899",
    "FMCG": "#8b5cf6",
    "Metals": "#f59e0b",
    "Realty": "#06b6d4",
    "Utilities": "#10b981",
    "Telecom": "#6366f1",
    "PSU": "#f87171",
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
        st.error(f"❌ Error fetching CSV list: {e}")
        return []

@st.cache_data(ttl=600)
def load_universe(csv_name):
    """Load stock universe from GitHub CSV"""
    url = f"https://raw.githubusercontent.com/anki1007/rrg-stocks/main/ticker/{csv_name.lower()}.csv"
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"❌ Error loading {csv_name} universe: {e}")
        return pd.DataFrame()

def rrg_calc(px, bench):
    """Calculate RRG metrics: RS-Ratio and RS-Momentum"""
    df = pd.concat([px, bench], axis=1).dropna()
    if len(df) < WINDOW + 2:
        return None, None
    
    rs = 100 * (df.iloc[:, 0] / df.iloc[:, 1])
    rs_ratio = 100 * (rs - rs.rolling(WINDOW).mean()) / rs.rolling(WINDOW).std(ddof=0)
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
    clean_sym = sym.replace('.NS', '')
    return f"https://www.tradingview.com/chart/?symbol=NSE:{clean_sym}"

def calculate_price_change(current_price, historical_price):
    """Calculate percentage change"""
    if historical_price == 0:
        return 0
    return ((current_price - historical_price) / historical_price) * 100

def get_download_link(df, filename, file_type="csv"):
    """Generate download link for dataframe"""
    if file_type == "csv":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        return f'<a href="data:file/csv;base64,{b64}" download="{filename}.csv" class="export-btn">📥 {filename}.csv</a>'

# ============================================================================
# SIDEBAR - CONTROLS
# ============================================================================

st.sidebar.markdown("### ⚙️ RRG CONTROLS")

csv_files = list_csv_from_github()
csv_selected = st.sidebar.selectbox("📊 Indices", csv_files, help="Select stock universe from available CSVs")

bench_name = st.sidebar.selectbox("🎯 Benchmark", list(BENCHMARKS.keys()), help="Select benchmark index for relative strength")

tf_name = st.sidebar.selectbox("⏱️ Strength vs Timeframe", list(TIMEFRAMES.keys()), help="Select timeframe for RRG analysis")

period_name = st.sidebar.selectbox("📅 Period", list(PERIOD_MAP.keys()), help="Select analysis period")

rank_by = st.sidebar.selectbox(
    "🏆 Rank by",
    ["RRG Power", "RS-Ratio", "RS-Momentum", "Price % Change", "Momentum Slope"],
    help="Select ranking metric"
)

st.sidebar.markdown("---")

top_n = st.sidebar.slider("🔝 Show Top N", min_value=5, max_value=50, value=15, help="Number of top stocks to display")

# Filter by sector
st.sidebar.markdown("### 🏢 SECTOR FILTER")
all_sectors = st.sidebar.checkbox("All Sectors", value=True)
selected_sectors = []
if not all_sectors:
    selected_sectors = st.sidebar.multiselect("Select Sectors", list(SECTOR_COLORS.keys()))

st.sidebar.markdown("---")

# Export section
st.sidebar.markdown("### 📤 EXPORT OPTIONS")
export_csv = st.sidebar.checkbox("📊 Export CSV", value=True)
export_chart = st.sidebar.checkbox("📈 Export Chart as PNG", value=False)

st.sidebar.markdown("---")

# Animation controls
st.sidebar.markdown("### 🎬 SECTOR ROTATION ANIMATION")
animation_speed = st.sidebar.slider("Animation Speed (ms)", min_value=100, max_value=2000, value=500, step=100)
play_animation = st.sidebar.checkbox("▶️ Play Rotation Animation", value=False)

st.sidebar.markdown("---")

# Legend
st.sidebar.markdown("### 📍 LEGEND")
legend_cols = st.sidebar.columns(2)
with legend_cols[0]:
    st.markdown("🟢 **Leading**: Strong RS, ↑ Momentum")
    st.markdown("🔵 **Improving**: Weak RS, ↑ Momentum")
with legend_cols[1]:
    st.markdown("🟡 **Weakening**: Weak RS, ↓ Momentum")
    st.markdown("🔴 **Lagging**: Strong RS, ↓ Momentum")

# ============================================================================
# DATA LOADING & CALCULATION
# ============================================================================

try:
    interval, yf_period = TIMEFRAMES[tf_name]
    universe = load_universe(csv_selected)
    
    if universe.empty:
        st.error("❌ Failed to load universe data")
        st.stop()
    
    symbols = universe['Symbol'].tolist()
    names_dict = dict(zip(universe['Symbol'], universe['Company Name']))
    industries_dict = dict(zip(universe['Symbol'], universe['Industry']))
    
    with st.spinner(f"📥 Downloading data for {len(symbols)} symbols..."):
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
            'TV Link': tv_link,
            'RR_Current': rr_tail.iloc[-1],
            'MM_Current': mm_tail.iloc[-1]
        })
        
        trails[s] = (rr_tail, mm_tail)
    
    df = pd.DataFrame(rows)
    
    if df.empty:
        st.error("❌ No data available. Try another combination.")
        st.stop()
    
    # Apply sector filter
    if not all_sectors and selected_sectors:
        df = df[df['Industry'].isin(selected_sectors)]
    
    # Ranking
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
    
    # Prepare CSV export data
    df_export = df_top[[
        'Sl No.', 'Symbol', 'Name', 'Industry', 'Price', 'Change %', 
        'RS-Ratio', 'RS-Momentum', 'Momentum Slope', 'RRG Power', 'Status'
    ]].copy()
    
except Exception as e:
    st.error(f"❌ Error in data processing: {str(e)}")
    st.stop()

# ============================================================================
# MAIN LAYOUT - THREE COLUMNS
# ============================================================================

col_left, col_main, col_right = st.columns([1, 3, 1], gap="small")

# ============================================================================
# LEFT SIDEBAR - DISTRIBUTION & EXPORT
# ============================================================================

with col_left:
    st.markdown("### 📊 DISTRIBUTION")
    
    status_counts = df['Status'].value_counts()
    status_colors = [QUADRANT_COLORS.get(status, '#808080') for status in status_counts.index]
    
    fig_dist = go.Figure(data=[
        go.Bar(
            y=status_counts.index,
            x=status_counts.values,
            orientation='h',
            marker=dict(color=status_colors),
            text=status_counts.values,
            textposition='auto'
        )
    ])
    
    fig_dist.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        xaxis_title="Count",
        yaxis_title="Status",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0', size=11),
        xaxis=dict(gridcolor='rgba(0,212,255,0.1)'),
    )
    
    st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("### 📈 TOP METRICS")
    
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        avg_rs = df['RS-Ratio'].mean()
        st.metric("Avg RS-Ratio", f"{avg_rs:.2f}", delta="vs 100", delta_color="off")
    
    with col_m2:
        avg_mm = df['RS-Momentum'].mean()
        st.metric("Avg RS-Mom", f"{avg_mm:.2f}", delta="vs 100", delta_color="off")
    
    best = df.iloc[0]
    st.markdown(f"### 🌟 TOP PICK")
    st.markdown(f"**{best['Symbol']}** ({best['Industry']})")
    st.markdown(f"₹{best['Price']} | {best['Change %']:+.2f}%")
    st.markdown(f"[📈 TradingView Chart]({best['TV Link']})", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📤 DOWNLOAD DATA")
    
    # CSV Export
    if export_csv:
        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        b64_csv = base64.b64encode(csv_data.encode()).decode()
        st.markdown(
            f'<a href="data:file/csv;base64,{b64_csv}" download="RRG_Analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv" style="display: inline-block; padding: 8px 16px; background: #22c55e; color: #000; border-radius: 6px; text-decoration: none; font-weight: 600;">📥 Download CSV</a>',
            unsafe_allow_html=True
        )
        st.caption("✅ CSV ready for download")

# ============================================================================
# MAIN CONTENT - RRG CHART & TABLE
# ============================================================================

with col_main:
    st.markdown("## 🔄 RELATIVE ROTATION GRAPH")
    
    # Create base RRG figure
    fig_rrg = go.Figure()
    
    # Add quadrant backgrounds
    fig_rrg.add_shape(type="rect", x0=100, y0=100, x1=df['RS-Ratio'].max() + 5, y1=df['RS-Momentum'].max() + 5,
                      fillcolor="rgba(34, 197, 94, 0.05)", line=dict(color="rgba(34, 197, 94, 0.2)", width=1), layer="below")
    fig_rrg.add_shape(type="rect", x0=df['RS-Ratio'].min() - 5, y0=100, x1=100, y1=df['RS-Momentum'].max() + 5,
                      fillcolor="rgba(59, 130, 246, 0.05)", line=dict(color="rgba(59, 130, 246, 0.2)", width=1), layer="below")
    fig_rrg.add_shape(type="rect", x0=df['RS-Ratio'].min() - 5, y0=df['RS-Momentum'].min() - 5, x1=100, y1=100,
                      fillcolor="rgba(251, 191, 36, 0.05)", line=dict(color="rgba(251, 191, 36, 0.2)", width=1), layer="below")
    fig_rrg.add_shape(type="rect", x0=100, y0=df['RS-Momentum'].min() - 5, x1=df['RS-Ratio'].max() + 5, y1=100,
                      fillcolor="rgba(239, 68, 68, 0.05)", line=dict(color="rgba(239, 68, 68, 0.2)", width=1), layer="below")
    
    # Add center lines
    fig_rrg.add_hline(y=100, line_dash="dash", line_color="rgba(0, 212, 255, 0.3)", layer="below")
    fig_rrg.add_vline(x=100, line_dash="dash", line_color="rgba(0, 212, 255, 0.3)", layer="below")
    
    # Add labels for quadrants
    fig_rrg.add_annotation(text="LEADING", x=df['RS-Ratio'].max()-2, y=df['RS-Momentum'].max()-2, 
                          font=dict(size=12, color="rgba(34, 197, 94, 0.3)"), showarrow=False)
    fig_rrg.add_annotation(text="IMPROVING", x=df['RS-Ratio'].min()+2, y=df['RS-Momentum'].max()-2, 
                          font=dict(size=12, color="rgba(59, 130, 246, 0.3)"), showarrow=False)
    fig_rrg.add_annotation(text="WEAKENING", x=df['RS-Ratio'].min()+2, y=df['RS-Momentum'].min()+2, 
                          font=dict(size=12, color="rgba(251, 191, 36, 0.3)"), showarrow=False)
    fig_rrg.add_annotation(text="LAGGING", x=df['RS-Ratio'].max()-2, y=df['RS-Momentum'].min()+2, 
                          font=dict(size=12, color="rgba(239, 68, 68, 0.3)"), showarrow=False)
    
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
                    size=10,
                    color=QUADRANT_COLORS[status],
                    line=dict(color='white', width=1),
                    opacity=0.8
                ),
                hovertemplate='<b>%{text}</b><br>RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<extra></extra>'
            ))
    
    fig_rrg.update_layout(
        height=450,
        title_x=0.5,
        xaxis_title="RS-Ratio (Relative Strength)",
        yaxis_title="RS-Momentum (Rate of Change)",
        plot_bgcolor='rgba(0,0,0,0.2)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0', size=11),
        hovermode='closest',
        xaxis=dict(gridcolor='rgba(0,212,255,0.1)', zeroline=False),
        yaxis=dict(gridcolor='rgba(0,212,255,0.1)', zeroline=False),
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.3)', bordercolor='rgba(0,212,255,0.2)', borderwidth=1)
    )
    
    chart_placeholder = st.plotly_chart(fig_rrg, use_container_width=True, key="main_rrg_chart")
    
    # Sector Rotation Animation
    if play_animation:
        st.markdown("### 🎬 SECTOR ROTATION ANIMATION")
        animation_placeholder = st.empty()
        
        sectors_present = df_top['Industry'].unique()
        animation_frames = len(sectors_present) + 1
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for frame in range(animation_frames):
            progress_bar.progress(frame / animation_frames)
            
            if frame == 0:
                status_text.info("📊 Starting sector rotation analysis...")
            else:
                sector_filter = sectors_present[frame - 1]
                status_text.info(f"🔄 Now viewing: **{sector_filter}** sector")
                
                df_sector = df_top[df_top['Industry'] == sector_filter]
                
                fig_sector = go.Figure()
                
                fig_sector.add_shape(type="rect", x0=100, y0=100, x1=df['RS-Ratio'].max() + 5, y1=df['RS-Momentum'].max() + 5,
                                    fillcolor="rgba(34, 197, 94, 0.05)", line=dict(color="rgba(34, 197, 94, 0.2)", width=1), layer="below")
                fig_sector.add_shape(type="rect", x0=df['RS-Ratio'].min() - 5, y0=100, x1=100, y1=df['RS-Momentum'].max() + 5,
                                    fillcolor="rgba(59, 130, 246, 0.05)", line=dict(color="rgba(59, 130, 246, 0.2)", width=1), layer="below")
                fig_sector.add_shape(type="rect", x0=df['RS-Ratio'].min() - 5, y0=df['RS-Momentum'].min() - 5, x1=100, y1=100,
                                    fillcolor="rgba(251, 191, 36, 0.05)", line=dict(color="rgba(251, 191, 36, 0.2)", width=1), layer="below")
                fig_sector.add_shape(type="rect", x0=100, y0=df['RS-Momentum'].min() - 5, x1=df['RS-Ratio'].max() + 5, y1=100,
                                    fillcolor="rgba(239, 68, 68, 0.05)", line=dict(color="rgba(239, 68, 68, 0.2)", width=1), layer="below")
                
                fig_sector.add_hline(y=100, line_dash="dash", line_color="rgba(0, 212, 255, 0.3)", layer="below")
                fig_sector.add_vline(x=100, line_dash="dash", line_color="rgba(0, 212, 255, 0.3)", layer="below")
                
                sector_color = SECTOR_COLORS.get(sector_filter, "#00d4ff")
                
                fig_sector.add_trace(go.Scatter(
                    x=df_sector['RS-Ratio'],
                    y=df_sector['RS-Momentum'],
                    mode='markers+text',
                    text=df_sector['Symbol'],
                    textposition="top center",
                    marker=dict(
                        size=14,
                        color=sector_color,
                        line=dict(color='white', width=2),
                        opacity=0.9
                    ),
                    hovertemplate='<b>%{text}</b><br>RS-Ratio: %{x:.2f}<br>RS-Momentum: %{y:.2f}<extra></extra>'
                ))
                
                fig_sector.update_layout(
                    height=450,
                    title=f"<b>{sector_filter} Sector Rotation</b>",
                    title_x=0.5,
                    xaxis_title="RS-Ratio",
                    yaxis_title="RS-Momentum",
                    plot_bgcolor='rgba(0,0,0,0.2)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#e0e0e0', size=11),
                    xaxis=dict(gridcolor='rgba(0,212,255,0.1)', zeroline=False),
                    yaxis=dict(gridcolor='rgba(0,212,255,0.1)', zeroline=False),
                    showlegend=False
                )
                
                animation_placeholder.plotly_chart(fig_sector, use_container_width=True, config={'displayModeBar': False})
            
            import time
            time.sleep(animation_speed / 1000)
        
        status_text.success("✅ Sector rotation animation complete!")
    
    st.markdown("---")
    st.markdown("## 📋 SCREENING TABLE")
    
    df_display = df_top[[
        'Sl No.', 'Symbol', 'Industry', 'Price', 'Change %', 
        'RS-Ratio', 'RS-Momentum', 'Status'
    ]].copy()
    
    df_display['Price'] = df_display['Price'].apply(lambda x: f"₹{x:,.2f}")
    df_display['Change %'] = df_display['Change %'].apply(lambda x: f"{x:+.2f}%")
    df_display['RS-Ratio'] = df_display['RS-Ratio'].apply(lambda x: f"{x:.2f}")
    df_display['RS-Momentum'] = df_display['RS-Momentum'].apply(lambda x: f"{x:.2f}")
    
    html_table = "<table style='width:100%; border-collapse: collapse; font-size: 12px;'>"
    html_table += "<thead style='background: rgba(0,212,255,0.1); border-bottom: 2px solid rgba(0,212,255,0.3);'>"
    html_table += "<tr style='color: #00d4ff; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;'>"
    
    for col in df_display.columns:
        html_table += f"<th style='padding: 10px; text-align: left;'>{col}</th>"
    
    html_table += "</tr></thead><tbody>"
    
    for idx, row in df_display.iterrows():
        status_color = QUADRANT_COLORS.get(df_top.iloc[idx]['Status'], '#808080')
        tv_link = df_top.iloc[idx]['TV Link']
        
        html_table += "<tr style='border-bottom: 1px solid rgba(0,212,255,0.1);'>"
        html_table += f"<td style='padding: 10px;'>{row['Sl No.']}</td>"
        html_table += f"<td style='padding: 10px;'><a href='{tv_link}' target='_blank' style='color: #00d4ff; text-decoration: none; font-weight: 600;'>{row['Symbol']}</a></td>"
        html_table += f"<td style='padding: 10px;'>{row['Industry']}</td>"
        html_table += f"<td style='padding: 10px;'>{row['Price']}</td>"
        html_table += f"<td style='padding: 10px; color: {'#22c55e' if '+' in row['Change %'] else '#ef4444'};'>{row['Change %']}</td>"
        html_table += f"<td style='padding: 10px;'>{row['RS-Ratio']}</td>"
        html_table += f"<td style='padding: 10px;'>{row['RS-Momentum']}</td>"
        html_table += f"<td style='padding: 10px;'><span style='background: {status_color}33; color: {status_color}; padding: 4px 8px; border-radius: 4px; font-weight: 600; font-size: 10px;'>{df_top.iloc[idx]['Status']}</span></td>"
        html_table += "</tr>"
    
    html_table += "</tbody></table>"
    
    st.markdown(html_table, unsafe_allow_html=True)

# ============================================================================
# RIGHT SIDEBAR - MOMENTUM TRENDS & SECTOR ANALYSIS
# ============================================================================

with col_right:
    st.markdown("### 📉 MOMENTUM TRENDS")
    
    top_5 = df_top.head(5)
    fig_trend = go.Figure()
    
    for idx, row in top_5.iterrows():
        symbol = row['Symbol']
        if symbol in trails:
            rr_tail, mm_tail = trails[symbol]
            fig_trend.add_trace(go.Scatter(
                x=list(range(len(mm_tail))),
                y=mm_tail.values,
                mode='lines+markers',
                name=symbol,
                line=dict(color=QUADRANT_COLORS.get(row['Status'], '#808080'), width=2),
                marker=dict(size=6)
            ))
    
    fig_trend.update_layout(
        height=350,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title="Periods",
        yaxis_title="RS-Momentum",
        plot_bgcolor='rgba(0,0,0,0.2)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0', size=10),
        hovermode='x unified',
        xaxis=dict(gridcolor='rgba(0,212,255,0.1)'),
        yaxis=dict(gridcolor='rgba(0,212,255,0.1)'),
        legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0.3)', bordercolor='rgba(0,212,255,0.2)', borderwidth=1, font=dict(size=9))
    )
    
    fig_trend.add_hline(y=100, line_dash="dash", line_color="rgba(0,212,255,0.3)")
    
    st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})
    
    st.markdown("---")
    st.markdown("### 🎯 QUADRANT ANALYSIS")
    
    for status in ["Leading", "Improving", "Weakening", "Lagging"]:
        count = len(df[df['Status'] == status])
        color = QUADRANT_COLORS[status]
        st.markdown(
            f"<div style='background: {color}22; border-left: 3px solid {color}; padding: 8px; margin-bottom: 8px; border-radius: 4px;'>"
            f"<span style='color: {color}; font-weight: 600;'>{status}</span>: {count} stocks"
            f"</div>",
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    st.markdown("### 🏢 SECTOR BREAKDOWN")
    
    sector_counts = df_top['Industry'].value_counts()
    for sector, count in sector_counts.items():
        sector_color = SECTOR_COLORS.get(sector, '#00d4ff')
        st.markdown(
            f"<div style='background: {sector_color}11; border-left: 2px solid {sector_color}; padding: 6px; margin-bottom: 6px; border-radius: 4px; font-size: 12px;'>"
            f"<span style='color: {sector_color}; font-weight: 500;'>{sector}</span>: {count}"
            f"</div>",
            unsafe_allow_html=True
        )
    
    st.markdown("---")
    st.markdown("### ⏱️ DATA INFO")
    st.markdown(f"**Index:** {csv_selected}")
    st.markdown(f"**Benchmark:** {bench_name}")
    st.markdown(f"**Timeframe:** {tf_name}")
    st.markdown(f"**Period:** {period_name}")
    st.markdown(f"**Symbols Analyzed:** {len(df)}")
    st.markdown(f"**Rank By:** {rank_by}")
    st.markdown(f"**Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #808080; font-size: 11px;'>
    💡 <b>Advanced RRG Analysis Dashboard</b><br>
    Features: Live Charts • Sector Rotation • CSV Export • TradingView Integration<br>
    Data: Yahoo Finance | Dashboard: Streamlit | Charts: Plotly<br>
    <i>Disclaimer: For educational purposes only. Not financial advice.</i>
</div>
""", unsafe_allow_html=True)

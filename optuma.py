import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import requests
from datetime import datetime, timedelta
import time

# ============================================================================
# CONFIG & SETUP
# ============================================================================

st.set_page_config(
    layout="wide",
    page_title="RRG Dashboard - Multi-Index Analysis",
    initial_sidebar_state="expanded"
)

# ============================================================================
# STYLING - WHITE BACKGROUND & LIGHT THEME FOR QUADRANT VISIBILITY
# ============================================================================

st.markdown("""
    <style>
        :root {
            --leading-color: #22c55e;
            --improving-color: #3b82f6;
            --weakening-color: #facc15;
            --lagging-color: #ef4444;
        }
        
        /* Light theme for quadrant colors to be visible */
        [data-testid="stAppViewContainer"] {
            background-color: #ffffff !important;
        }
        [data-testid="stHeader"] {
            background-color: #ffffff00 !important;
        }
        
        /* Table styling for better readability */
        .rrg-wrap {
            max-height: calc(100vh - 260px);
            overflow: auto;
            border: 1px solid #e5e5e5;
            border-radius: 6px;
        }
        
        .rrg-table {
            width: 100%;
            border-collapse: collapse;
            font-family: Segoe UI, -apple-system, Arial, sans-serif;
        }
        
        .rrg-table th, .rrg-table td {
            border-bottom: 1px solid #ececec;
            padding: 10px;
            font-size: 14px;
            text-align: left;
        }
        
        .rrg-table th {
            position: sticky;
            top: 0;
            z-index: 2;
            background: #eef2f7;
            color: #0f172a;
            font-weight: 800;
            letter-spacing: 0.2px;
        }
        
        .rrg-row {
            transition: background 0.12s ease;
        }
        
        .rrg-name a {
            color: #0b57d0;
            text-decoration: underline;
        }
        
        /* Top 30 ranking panel */
        .rrg-rank {
            font-weight: 700;
            line-height: 1.25;
            font-size: 1.05rem;
            white-space: pre;
        }
        
        .rrg-rank .row {
            display: flex;
            gap: 8px;
            align-items: baseline;
            margin: 2px 0;
        }
        
        .rrg-rank .name {
            color: #0b57d0;
        }
        
    </style>
""", unsafe_allow_html=True)

# Define Benchmarks
BENCHMARKS = {
    "NIFTY 50": "^NSEI",
    "NIFTY 200": "^CNXIT",
    "NIFTY 500": "^CNXIT"
}

# Define Timeframes
TIMEFRAMES = {
    "5 min close": ("5m", "60d"),
    "15 min close": ("15m", "60d"),
    "30 min close": ("30m", "60d"),
    "1 hr close": ("60m", "90d"),
    "Daily": ("1d", "5y"),
    "Weekly": ("1wk", "10y"),
    "Monthly": ("1mo", "20y")
}

# Period mapping
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
DEFAULTTAIL = 8

# Color mapping - matches your script exactly
QUADRANT_COLORS = {
    "Leading": "#3fa46a",
    "Improving": "#5d86d1",
    "Weakening": "#e2d06b",
    "Lagging": "#e06a6a"
}

# ============================================================================
# EXACT FORMULA FROM YOUR SCRIPT - jdkcomponents
# ============================================================================

def jdkcomponents(price, bench, win=14):
    """
    JDK RRG Components - Exact formula from your reference script
    RS-Ratio = 100 * (rs - mean) / std
    RS-Momentum = 101 * (rroc - mean2) / std2
    where rs = 100 * price/bench
    and rroc = rate of change of rs-ratio
    """
    df = pd.concat([price.rename('p'), bench.rename('b')], axis=1).dropna()
    
    if df.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    # RS = 100 * (price / benchmark)
    rs = 100 * (df['p'] / df['b'])
    
    # RS-Ratio = 100 * (rs - mean) / std
    m = rs.rolling(win).mean()
    s = rs.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rsratio = 100 * (rs - m) / s
    
    # Rate of change of RS-Ratio
    rroc = rsratio.pct_change().mul(100)
    
    # RS-Momentum = 101 * (rroc - mean2) / std2
    m2 = rroc.rolling(win).mean()
    s2 = rroc.rolling(win).std(ddof=0).replace(0, np.nan).fillna(1e-9)
    rsmom = 101 * (rroc - m2) / s2
    
    # Align indices
    ix = rsratio.index.intersection(rsmom.index)
    
    return rsratio.loc[ix], rsmom.loc[ix]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_data(ttl=600)
def list_csv_from_github():
    """Fetch CSV filenames from GitHub"""
    url = "https://api.github.com/repos/anki1007/rrg-stocks/contents/ticker"
    try:
        response = requests.get(url, timeout=15)
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

def getStatus(x, y):
    """Determine quadrant based on RS-Ratio and RS-Momentum"""
    if x > 100 and y > 100:
        return "Lagging"
    elif x > 100 and y < 100:
        return "Leading"
    elif x < 100 and y > 100:
        return "Improving"
    elif x < 100 and y < 100:
        return "Weakening"
    return "Unknown"

def display_symbol(sym):
    """Remove .NS suffix"""
    return sym[:-3] if sym.upper().endswith('.NS') else sym

def format_bar_date(ts, interval):
    """Format date based on interval"""
    ts = pd.Timestamp(ts)
    if interval == "1wk":
        return ts.to_period('W-FRI').end_time.date().isoformat()
    elif interval == "1mo":
        return ts.to_period('M').end_time.date().isoformat()
    return ts.date().isoformat()

# ============================================================================
# SIDEBAR CONTROLS
# ============================================================================

st.sidebar.markdown("### ⚙️ RRG Controls")

# Load CSV files
csv_files = list_csv_from_github()
csv_selected = st.sidebar.selectbox(
    "📊 Indices",
    csv_files,
    help="Select stock universe"
)

# Benchmark
bench_name = st.sidebar.selectbox(
    "🎯 Benchmark",
    list(BENCHMARKS.keys()),
    help="Select benchmark index"
)

# Timeframe
tf_name = st.sidebar.selectbox(
    "⏱️ Strength vs Timeframe",
    list(TIMEFRAMES.keys()),
    help="Select timeframe"
)

# Period
period_name = st.sidebar.selectbox(
    "📅 Period",
    list(PERIOD_MAP.keys()),
    help="Select analysis period"
)

# Rank by
rank_by = st.sidebar.selectbox(
    "🏆 Rank by",
    ["RRG Power", "RS-Ratio", "RS-Momentum"],
    help="Select ranking metric"
)

# Tail length
tail_len = st.sidebar.slider("Trail Length", 1, 20, DEFAULTTAIL, 1)

st.sidebar.markdown("---")

# PLAYBACK CONTROLS
st.sidebar.markdown("### ▶️ Playback Controls")

if 'playing' not in st.session_state:
    st.session_state.playing = False

play_toggle = st.sidebar.toggle(
    "▶️ Play/Pause",
    value=st.session_state.playing,
    key="playing",
    help="Animate through historical rotations"
)

speed_ms = st.sidebar.slider(
    "Speed (ms/frame)",
    150, 1500, 300, 50,
    help="Lower = faster animation"
)

looping = st.sidebar.checkbox(
    "Loop",
    value=True,
    help="Restart from beginning when done"
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
    
    # Download data
    raw = yf.download(
        symbols + [BENCHMARKS[bench_name]],
        interval=interval,
        period=yf_period,
        auto_adjust=True,
        progress=False,
        threads=True
    )
    
    # Extract close prices
    def pick_close(df, symbol):
        if isinstance(df, pd.Series):
            return df.dropna()
        if isinstance(df, pd.DataFrame):
            if isinstance(df.columns, pd.MultiIndex):
                for lvl in ['Close', 'Adj Close']:
                    if (symbol, lvl) in df.columns:
                        return df[symbol, lvl].dropna()
            else:
                for col in ['Close', 'Adj Close']:
                    if col in df.columns:
                        return df[col].dropna()
        return pd.Series(dtype=float)
    
    bench = pick_close(raw, BENCHMARKS[bench_name])
    
    # Calculate RRG for each stock
    rsratiomap = {}
    rsmommap = {}
    kept = []
    
    for t in symbols:
        if t == BENCHMARKS[bench_name]:
            continue
        
        s = pick_close(raw, t)
        
        if s.empty:
            continue
        
        rr, mm = jdkcomponents(s, bench, WINDOW)
        
        if len(rr) == 0 or len(mm) == 0:
            continue
        
        # Reindex to benchmark index
        rr = rr.reindex(bench.index)
        mm = mm.reindex(bench.index)
        
        # Check minimum coverage
        ok = ~(rr.isna() | mm.isna())
        if ok.sum() < max(WINDOW + 5, 20):
            continue
        
        rsratiomap[t] = rr
        rsmommap[t] = mm
        kept.append(t)
    
    tickers = kept
    
    if not tickers:
        st.error("No stocks have sufficient data coverage")
        st.stop()
    
    benchidx = bench.index
    idxlen = len(benchidx)
    
    # ANIMATION STATE
    if 'endidx' not in st.session_state:
        st.session_state.endidx = idxlen - 1
    
    st.session_state.endidx = min(max(st.session_state.endidx, DEFAULTTAIL), idxlen - 1)
    
    # Playback logic
    if st.session_state.playing:
        nxt = st.session_state.endidx + 1
        if nxt >= idxlen - 1:
            if looping:
                nxt = DEFAULTTAIL
            else:
                nxt = idxlen - 1
                st.session_state.playing = False
        st.session_state.endidx = nxt
        time.sleep(speed_ms / 1000.0)
        st.rerun()
    
    # Slider for manual control
    endidx = st.slider(
        "Date Position",
        min_value=DEFAULTTAIL,
        max_value=idxlen - 1,
        value=st.session_state.endidx,
        step=1,
        key="endidx",
        help="RRG position (closed bars only)"
    )
    
    st.session_state.endidx = endidx
    startidx = max(endidx - tail_len, 0)
    datestr = format_bar_date(benchidx[endidx], interval)
    
except Exception as e:
    st.error(f"Error: {str(e)}")
    st.stop()

# ============================================================================
# MAIN LAYOUT
# ============================================================================

st.markdown(f"### 📊 Relative Rotation Graph | {bench_name} | {tf_name} | {period_name} | {datestr}")

# Three column layout
col_left, col_main, col_right = st.columns([0.8, 3.5, 1.2], gap="medium")

# ============================================================================
# LEFT SIDEBAR - LEGEND
# ============================================================================

with col_left:
    st.markdown("**Legend**")
    for status, color in QUADRANT_COLORS.items():
        st.markdown(
            f'<div style="display: flex; align-items: center; margin: 6px 0;"><div style="width: 12px; height: 12px; background-color: {color}; border-radius: 50%; margin-right: 8px;"></div><span style="font-size:12px;">{status}</span></div>',
            unsafe_allow_html=True
        )

# ============================================================================
# MAIN AREA - RRG GRAPH
# ============================================================================

with col_main:
    # Create Plotly figure with WHITE background
    fig = go.Figure()
    
    # Add quadrant backgrounds with WHITE chart area
    fig.add_shape(type="rect", x0=94, y0=100, x1=106, y1=106,
                  fillcolor="#22c55e", opacity=0.08, line_width=0)
    fig.add_shape(type="rect", x0=100, y0=100, x1=106, y1=106,
                  fillcolor="#22c55e", opacity=0.08, line_width=0)
    fig.add_shape(type="rect", x0=100, y0=94, x1=106, y1=100,
                  fillcolor="#facc15", opacity=0.08, line_width=0)
    fig.add_shape(type="rect", x0=94, y0=100, x1=100, y1=106,
                  fillcolor="#3b82f6", opacity=0.08, line_width=0)
    fig.add_shape(type="rect", x0=94, y0=94, x1=100, y1=100,
                  fillcolor="#ef4444", opacity=0.08, line_width=0)
    
    # Center lines
    fig.add_hline(y=100, line_dash="dash", line_color="gray", opacity=0.4, line_width=1)
    fig.add_vline(x=100, line_dash="dash", line_color="gray", opacity=0.4, line_width=1)
    
    # Plot lines and points for each stock
    for t in tickers:
        if t not in rsratiomap or t not in rsmommap:
            continue
        
        rr = rsratiomap[t].iloc[startidx+1:endidx+1].dropna()
        mm = rsmommap[t].iloc[startidx+1:endidx+1].dropna()
        
        rr, mm = rr.align(mm, join='inner')
        
        if len(rr) == 0 or len(mm) == 0:
            continue
        
        # Plot line trail
        fig.add_trace(go.Scatter(
            x=rr.values, y=mm.values,
            mode='lines',
            line=dict(width=1.2, color=QUADRANT_COLORS.get(getStatus(rr.iloc[-1], mm.iloc[-1]), '#666666'), opacity=0.6),
            name=display_symbol(t),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Current position point
        size = 22 + (len(rr) - 1) * 0.6
        fig.add_trace(go.Scatter(
            x=[rr.iloc[-1]], y=[mm.iloc[-1]],
            mode='markers+text',
            marker=dict(
                size=8,
                color=QUADRANT_COLORS.get(getStatus(rr.iloc[-1], mm.iloc[-1]), '#666666'),
                line=dict(width=0.6, color='#333333')
            ),
            text=[display_symbol(t)],
            textposition='top center',
            textfont=dict(size=9, color='black'),
            hovertemplate=f'<b>{display_symbol(t)}</b><br>RS-Ratio: %{{x:.2f}}<br>RS-Momentum: %{{y:.2f}}<extra></extra>',
            showlegend=False
        ))
    
    # Update layout - WHITE background
    fig.update_layout(
        title=None,
        xaxis_title="JdK RS-Ratio",
        yaxis_title="JdK RS-Momentum",
        xaxis=dict(range=[94, 106], zeroline=False),
        yaxis=dict(range=[94, 106], zeroline=False),
        hovermode='closest',
        height=500,
        template='plotly',
        plot_bgcolor='rgba(255,255,255,1)',
        paper_bgcolor='rgba(255,255,255,1)',
        font=dict(color='#000000'),
        showlegend=False,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# RIGHT SIDEBAR - TOP 30 RS-MOMENTUM (3 COLUMNS ONLY)
# ============================================================================

with col_right:
    st.markdown("**🚀 Top Performers**")
    
    # Calculate rankings for all stocks
    perf = []
    for t in tickers:
        if t not in rsratiomap or t not in rsmommap:
            continue
        
        rr_last = float(rsratiomap[t].iloc[endidx]) if endidx < len(rsratiomap[t]) else np.nan
        mm_last = float(rsmommap[t].iloc[endidx]) if endidx < len(rsmommap[t]) else np.nan
        
        if np.isnan(rr_last) or np.isnan(mm_last):
            continue
        
        # Rank by selected metric
        if rank_by == "RRG Power":
            metric = float(np.hypot(rr_last - 100, mm_last - 100))
        elif rank_by == "RS-Ratio":
            metric = float(rr_last)
        else:  # RS-Momentum
            metric = float(mm_last)
        
        perf.append((t, metric, rr_last, mm_last))
    
    # Sort by metric (descending)
    perf.sort(key=lambda x: x[1], reverse=True)
    
    # Generate HTML for top 30
    if perf:
        html_rows = []
        for i, (sym, metric, rr, mm) in enumerate(perf[:30], start=1):
            status = getStatus(rr, mm)
            color_map = {"Leading": "#3fa46a", "Improving": "#5d86d1", "Weakening": "#e2d06b", "Lagging": "#e06a6a"}
            bg_color = color_map.get(status, "#aaaaaa")
            fg_color = "#ffffff" if bg_color in ["#3fa46a", "#5d86d1", "#e06a6a"] else "#000000"
            
            tv_link = f'https://www.tradingview.com/chart/?symbol=NSE:{display_symbol(sym).replace("-", "")}'
            
            html_rows.append(
                f'''<div class="row" style="background:{bg_color}; color:{fg_color}; padding:4px 6px; border-radius:3px; margin:2px 0;">
                    <span style="font-weight:bold; width:20px;">{i}</span>
                    <span class="name"><a href="{tv_link}" target="_blank" style="color:{fg_color}; text-decoration:underline;">{display_symbol(sym)}</a></span>
                    <span style="margin-left:auto;">{status}</span>
                </div>'''
            )
        
        st.markdown(f'<div class="rrg-rank">{"".join(html_rows)}</div>', unsafe_allow_html=True)

# ============================================================================
# FULL WIDTH TABLE BELOW GRAPH
# ============================================================================

st.markdown("### 📋 Detailed Analysis")

# Build table
table_rows = []
for t in tickers:
    if t not in rsratiomap or t not in rsmommap:
        continue
    
    rr_last = float(rsratiomap[t].iloc[endidx]) if endidx < len(rsratiomap[t]) else np.nan
    mm_last = float(rsmommap[t].iloc[endidx]) if endidx < len(rsmommap[t]) else np.nan
    
    if np.isnan(rr_last) or np.isnan(mm_last):
        continue
    
    status = getStatus(rr_last, mm_last)
    bg_color_map = {"Leading": "#3fa46a", "Improving": "#5d86d1", "Weakening": "#e2d06b", "Lagging": "#e06a6a"}
    bg = bg_color_map.get(status, "#aaaaaa")
    fg = "#ffffff" if bg in ["#3fa46a", "#5d86d1", "#e06a6a"] else "#000000"
    
    tv_link = f'https://www.tradingview.com/chart/?symbol=NSE:{display_symbol(t).replace("-", "")}'
    
    # Calculate price change
    px = yf.download(t, interval=interval, period=yf_period, auto_adjust=True, progress=False)
    if isinstance(px, pd.DataFrame):
        px = px['Close']
    
    px = px.dropna()
    price = float(px.iloc[endidx]) if endidx < len(px) else np.nan
    chg = ((px.iloc[endidx] - px.iloc[startidx]) / px.iloc[startidx] * 100) if startidx < len(px) and endidx < len(px) else np.nan
    
    table_rows.append({
        'Rank': len(table_rows) + 1,
        'Symbol': display_symbol(t),
        'Status': status,
        'Name': names_dict.get(t, t),
        'Industry': industries_dict.get(t, '-'),
        'Price': f"{price:.2f}" if not np.isnan(price) else "-",
        'Change%': f"{chg:.2f}%" if not np.isnan(chg) else "-",
        'RS-Ratio': f"{rr_last:.2f}",
        'RS-Momentum': f"{mm_last:.2f}",
        'bg': bg,
        'fg': fg,
        'tv_link': tv_link
    })

# Generate HTML table
if table_rows:
    html_table = '''<table class="rrg-table">
        <thead><tr>
            <th>Rank</th>
            <th>Symbol</th>
            <th>Status</th>
            <th>Name</th>
            <th>Industry</th>
            <th>Price</th>
            <th>Change %</th>
            <th>RS-Ratio</th>
            <th>RS-Momentum</th>
        </tr></thead>
        <tbody>'''
    
    for row in table_rows:
        html_table += f'''<tr class="rrg-row" style="background:{row['bg']}; color:{row['fg']};">
            <td>{row['Rank']}</td>
            <td class="rrg-name"><a href="{row['tv_link']}" target="_blank">{row['Symbol']}</a></td>
            <td>{row['Status']}</td>
            <td>{row['Name']}</td>
            <td>{row['Industry']}</td>
            <td>{row['Price']}</td>
            <td>{row['Change%']}</td>
            <td>{row['RS-Ratio']}</td>
            <td>{row['RS-Momentum']}</td>
        </tr>'''
    
    html_table += '</tbody></table>'
    
    with st.expander("Expand Table", expanded=True):
        st.markdown(f'<div class="rrg-wrap">{html_table}</div>', unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    f"""
    <div style="text-align: center; font-size: 11px; color: #666;">
    <b>RRG Dashboard v2.1</b> | Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')} |
    Data Source: Yahoo Finance | Benchmark: {bench_name} | Window: {WINDOW}
    </div>
    """,
    unsafe_allow_html=True
)

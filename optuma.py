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
    page_title="RRG Dashboard - Optuma Style",
    initial_sidebar_state="collapsed"
)

# Optuma-inspired light theme with professional styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #f5f5f5;
  --bg-card: #ffffff;
  --border: #e0e0e0;
  --text: #333333;
  --text-dim: #666666;
  --accent: #2196F3;
}

html, body, .stApp {
  background: var(--bg) !important;
  font-family: 'Inter', system-ui, sans-serif !important;
}

.block-container {
  padding-top: 1rem;
  max-width: 100% !important;
  padding-left: 1rem !important;
  padding-right: 1rem !important;
}

/* Control bar styling */
.control-bar {
  background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
  border: 1px solid #dee2e6;
  border-radius: 8px;
  padding: 12px 20px;
  margin-bottom: 10px;
  display: flex;
  align-items: center;
  gap: 20px;
  flex-wrap: wrap;
}

.control-group {
  display: flex;
  align-items: center;
  gap: 8px;
}

.control-label {
  font-size: 12px;
  font-weight: 600;
  color: #495057;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.control-value {
  background: #ffffff;
  border: 1px solid #ced4da;
  border-radius: 4px;
  padding: 6px 12px;
  font-size: 13px;
  font-weight: 500;
  color: #212529;
}

.date-display {
  background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
  border: 1px solid #90caf9;
  border-radius: 6px;
  padding: 8px 16px;
  font-size: 13px;
  font-weight: 600;
  color: #1565c0;
}

/* Timeline styling */
.timeline-container {
  background: #ffffff;
  border: 1px solid #dee2e6;
  border-radius: 6px;
  padding: 8px 16px;
  margin-bottom: 10px;
}

/* Quadrant labels */
.quadrant-label {
  font-weight: 700;
  font-size: 14px;
  text-transform: uppercase;
  letter-spacing: 1px;
}

/* Hide streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Slider styling */
.stSlider > div > div > div {
  background: #2196F3 !important;
}

/* Button styling */
.stButton > button {
  background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
  border: 1px solid #dee2e6;
  color: #495057;
  font-weight: 600;
  border-radius: 6px;
  transition: all 0.2s;
}

.stButton > button:hover {
  background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
  border-color: #2196F3;
  color: #2196F3;
}

/* Play button special */
.play-btn button {
  background: linear-gradient(180deg, #4CAF50 0%, #43A047 100%) !important;
  color: white !important;
  border: none !important;
}

.play-btn button:hover {
  background: linear-gradient(180deg, #43A047 0%, #388E3C 100%) !important;
}

/* Selectbox styling */
div[data-baseweb="select"] {
  font-size: 13px;
}
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
    "5 min": ("5m", "60d"),
    "15 min": ("15m", "60d"),
    "30 min": ("30m", "60d"),
    "1 hr": ("60m", "90d"),
    "4 hr": ("240m", "120d"),
    "Daily": ("1d", "5y"),
    "Weekly": ("1wk", "10y"),
    "Monthly": ("1mo", "20y")
}

DATE_RANGES = {
    "1 Month": 21,
    "3 Months": 63,
    "6 Months": 126,
    "1 Year": 252,
    "2 Years": 504,
    "3 Years": 756,
}

# Optuma-style quadrant colors
QUADRANT_COLORS = {
    "Leading": "#228B22",    # Forest green
    "Improving": "#8B5CF6",  # Purple
    "Weakening": "#D97706",  # Orange/amber
    "Lagging": "#DC2626"     # Red
}

QUADRANT_BG_COLORS = {
    "Leading": "rgba(144, 238, 144, 0.5)",     # Light green
    "Improving": "rgba(221, 214, 254, 0.5)",   # Light purple
    "Weakening": "rgba(254, 215, 170, 0.5)",   # Light orange
    "Lagging": "rgba(254, 178, 178, 0.5)"      # Light red
}

WINDOW = 14

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
@st.cache_data(ttl=600)
def list_csv_from_github():
    url = "https://api.github.com/repos/anki1007/rrg-stocks/contents/ticker"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, list):
            files = [f['name'].replace('.csv', '').upper() for f in data
                    if isinstance(f, dict) and f.get('name', '').endswith('.csv')]
            return sorted(files) if files else []
    except Exception:
        return []
    return []

@st.cache_data(ttl=600)
def load_universe(csv_name):
    url = f"https://raw.githubusercontent.com/anki1007/rrg-stocks/main/ticker/{csv_name.lower()}.csv"
    try:
        return pd.read_csv(url)
    except Exception:
        return pd.DataFrame()

def calculate_jdk_rrg(ticker_series, benchmark_series, window=WINDOW):
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
    if x > 100 and y > 100:
        return "Leading"
    elif x < 100 and y > 100:
        return "Improving"
    elif x < 100 and y < 100:
        return "Lagging"
    else:
        return "Weakening"

def get_quadrant_color(x, y):
    status = quadrant(x, y)
    return QUADRANT_COLORS[status], status

def smooth_spline_curve(x_points, y_points, points_per_segment=8):
    if len(x_points) < 3:
        return np.array(x_points), np.array(y_points)
    
    x_points, y_points = np.array(x_points, dtype=float), np.array(y_points, dtype=float)
    
    def catmull_rom_segment(p0, p1, p2, p3, num_points):
        t = np.linspace(0, 1, num_points, endpoint=False)
        t2, t3 = t * t, t * t * t
        x = 0.5 * ((2*p1[0]) + (-p0[0]+p2[0])*t + (2*p0[0]-5*p1[0]+4*p2[0]-p3[0])*t2 + (-p0[0]+3*p1[0]-3*p2[0]+p3[0])*t3)
        y = 0.5 * ((2*p1[1]) + (-p0[1]+p2[1])*t + (2*p0[1]-5*p1[1]+4*p2[1]-p3[1])*t2 + (-p0[1]+3*p1[1]-3*p2[1]+p3[1])*t3)
        return x, y
    
    points = np.column_stack([x_points, y_points])
    padded = np.vstack([2*points[0]-points[1], points, 2*points[-1]-points[-2]])
    x_smooth, y_smooth = [], []
    
    for i in range(len(points)-1):
        seg_x, seg_y = catmull_rom_segment(padded[i], padded[i+1], padded[i+2], padded[i+3], points_per_segment)
        x_smooth.extend(seg_x)
        y_smooth.extend(seg_y)
    
    x_smooth.append(x_points[-1])
    y_smooth.append(y_points[-1])
    return np.array(x_smooth), np.array(y_smooth)

def get_tv_link(sym):
    clean_sym = sym.replace('.NS', '')
    return f"https://www.tradingview.com/chart/?symbol=NSE:{clean_sym}"

def format_symbol(sym):
    return sym.replace('.NS', '')

def select_graph_stocks(df, min_stocks=50):
    graph_stocks = []
    
    for status in ["Leading", "Improving", "Weakening", "Lagging"]:
        df_quad = df[df['Status'] == status].copy()
        if len(df_quad) == 0:
            continue
        elif len(df_quad) < 15:
            graph_stocks.extend(df_quad.index.tolist())
        else:
            if status in ["Leading", "Improving"]:
                top = df_quad.nlargest(15, 'RRG Power')
            else:
                top = df_quad.nsmallest(15, 'RRG Power')
            graph_stocks.extend(top.index.tolist())
    
    if len(graph_stocks) < min_stocks:
        remaining = df.index.difference(graph_stocks)
        additional = df.loc[remaining].nlargest(min_stocks - len(graph_stocks), 'RRG Power')
        graph_stocks.extend(additional.index.tolist())
    
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
if "dates_cache" not in st.session_state:
    st.session_state.dates_cache = []
if "current_frame" not in st.session_state:
    st.session_state.current_frame = 0
if "is_playing" not in st.session_state:
    st.session_state.is_playing = False

# ============================================================================
# CONTROL BAR (Optuma-style above graph)
# ============================================================================
csv_files = list_csv_from_github()
if not csv_files:
    csv_files = ["NIFTY200"]

# Find default NIFTY200
default_csv_idx = 0
for i, csv in enumerate(csv_files):
    if 'NIFTY200' in csv.upper():
        default_csv_idx = i
        break

# Control bar row 1: Main controls
ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4, ctrl_col5, ctrl_col6, ctrl_col7 = st.columns([1.5, 1.2, 1.2, 1.2, 0.8, 0.8, 2])

with ctrl_col1:
    st.markdown('<p style="font-size:11px; font-weight:600; color:#666; margin-bottom:2px;">BENCHMARK</p>', unsafe_allow_html=True)
    csv_selected = st.selectbox("", csv_files, index=default_csv_idx, key="csv_sel", label_visibility="collapsed")

with ctrl_col2:
    st.markdown('<p style="font-size:11px; font-weight:600; color:#666; margin-bottom:2px;">VS INDEX</p>', unsafe_allow_html=True)
    bench_name = st.selectbox("", list(BENCHMARKS.keys()), index=2, key="bench_sel", label_visibility="collapsed")

with ctrl_col3:
    st.markdown('<p style="font-size:11px; font-weight:600; color:#666; margin-bottom:2px;">TIMEFRAME</p>', unsafe_allow_html=True)
    tf_name = st.selectbox("", list(TIMEFRAMES.keys()), index=5, key="tf_sel", label_visibility="collapsed")

with ctrl_col4:
    st.markdown('<p style="font-size:11px; font-weight:600; color:#666; margin-bottom:2px;">DATE RANGE</p>', unsafe_allow_html=True)
    date_range = st.selectbox("", list(DATE_RANGES.keys()), index=1, key="date_sel", label_visibility="collapsed")

with ctrl_col5:
    st.markdown('<p style="font-size:11px; font-weight:600; color:#666; margin-bottom:2px;">COUNTS</p>', unsafe_allow_html=True)
    trail_length = st.number_input("", min_value=1, max_value=14, value=5, key="trail_input", label_visibility="collapsed")

with ctrl_col6:
    st.markdown('<p style="font-size:11px; font-weight:600; color:#666; margin-bottom:2px;">&nbsp;</p>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:#e3f2fd; border:1px solid #90caf9; border-radius:4px; padding:6px 10px; text-align:center; font-weight:600; color:#1565c0; font-size:13px;">{trail_length} Days</div>', unsafe_allow_html=True)

with ctrl_col7:
    st.markdown('<p style="font-size:11px; font-weight:600; color:#666; margin-bottom:2px;">&nbsp;</p>', unsafe_allow_html=True)
    btn_cols = st.columns([1, 1, 1, 1])
    with btn_cols[0]:
        load_btn = st.button("📥 Load", key="load_btn", use_container_width=True)
    with btn_cols[1]:
        play_btn = st.button("▶ Play", key="play_btn", use_container_width=True)
    with btn_cols[2]:
        pause_btn = st.button("⏸ Stop", key="pause_btn", use_container_width=True)
    with btn_cols[3]:
        label_toggle = st.checkbox("Label", value=True, key="label_chk")

if load_btn:
    st.session_state.load_clicked = True
    st.session_state.is_playing = False
    st.session_state.current_frame = trail_length - 1

if play_btn:
    st.session_state.is_playing = True

if pause_btn:
    st.session_state.is_playing = False

# ============================================================================
# DATA LOADING
# ============================================================================
if st.session_state.load_clicked:
    try:
        interval, yf_period = TIMEFRAMES[tf_name]
        universe = load_universe(csv_selected)
        
        if universe.empty:
            st.error("❌ Failed to load universe data.")
            st.stop()
        
        symbols = universe['Symbol'].tolist()
        names_dict = dict(zip(universe['Symbol'], universe['Company Name']))
        industries_dict = dict(zip(universe['Symbol'], universe['Industry']))
        
        with st.spinner(f"📥 Loading {len(symbols)} symbols..."):
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
        dates_list = raw.index.tolist()[-DATE_RANGES[date_range]:]
        
        for s in symbols:
            if s not in raw['Close'].columns:
                continue
            
            try:
                rs_ratio, rs_momentum, distance, heading, velocity = calculate_jdk_rrg(
                    raw['Close'][s], bench, window=WINDOW
                )
                
                if rs_ratio is None or len(rs_ratio) < 3:
                    continue
                
                # Store full history for animation
                max_hist = min(DATE_RANGES[date_range], len(rs_ratio))
                rs_history[format_symbol(s)] = {
                    'rs_ratio': rs_ratio.iloc[-max_hist:].tolist(),
                    'rs_momentum': rs_momentum.iloc[-max_hist:].tolist(),
                    'dates': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10] for d in dates_list[-max_hist:]]
                }
                
                rsr_current = rs_ratio.iloc[-1]
                rsm_current = rs_momentum.iloc[-1]
                dist_current = distance.iloc[-1]
                head_current = heading.iloc[-1]
                
                power = np.sqrt((rsr_current - 100) ** 2 + (rsm_current - 100) ** 2)
                current_price = raw['Close'][s].iloc[-1]
                status = quadrant(rsr_current, rsm_current)
                
                rows.append({
                    'Symbol': format_symbol(s),
                    'Name': names_dict.get(s, s),
                    'Industry': industries_dict.get(s, 'N/A'),
                    'Price': round(current_price, 2),
                    'RS-Ratio': round(rsr_current, 2),
                    'RS-Momentum': round(rsm_current, 2),
                    'RRG Power': round(power, 2),
                    'Distance': round(dist_current, 2),
                    'Status': status,
                    'TV Link': get_tv_link(s)
                })
            except Exception:
                continue
        
        if rows:
            df = pd.DataFrame(rows)
            df['Rank'] = df['RRG Power'].rank(ascending=False, method='min').astype(int)
            df = df.sort_values('Rank')
            df['Sl No.'] = range(1, len(df) + 1)
            
            st.session_state.df_cache = df
            st.session_state.rs_history_cache = rs_history
            st.session_state.dates_cache = dates_list
            st.session_state.current_frame = trail_length - 1
            
            st.success(f"✅ Loaded {len(df)} symbols")
        else:
            st.error("No data available.")
            st.stop()
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.stop()

# ============================================================================
# DISPLAY RRG GRAPH WITH ANIMATION
# ============================================================================
if st.session_state.df_cache is not None:
    df = st.session_state.df_cache
    rs_history = st.session_state.rs_history_cache
    
    # Timeline slider and animation controls
    if rs_history:
        max_history_len = max([len(rs_history[sym]['rs_ratio']) for sym in rs_history], default=1)
        
        # Get dates for display
        sample_sym = list(rs_history.keys())[0] if rs_history else None
        dates_available = rs_history[sample_sym].get('dates', []) if sample_sym else []
        
        st.markdown("---")
        
        # Animation mode selection
        anim_cols = st.columns([2, 1, 6, 1])
        
        with anim_cols[0]:
            view_mode = st.radio("View Mode", ["Static (Current)", "Animated"], horizontal=True, key="view_mode")
        
        with anim_cols[1]:
            st.write("")  # spacer
        
        if view_mode == "Static (Current)":
            with anim_cols[2]:
                frame_idx = st.slider(
                    "Timeline Position",
                    min_value=trail_length - 1,
                    max_value=max_history_len - 1,
                    value=max_history_len - 1,
                    key="static_slider"
                )
            with anim_cols[3]:
                if dates_available and len(dates_available) > frame_idx:
                    st.markdown(f"<div style='background:#e3f2fd; padding:8px; border-radius:4px; text-align:center; font-size:12px; font-weight:600; color:#1565c0; margin-top:22px;'>{dates_available[frame_idx]}</div>", unsafe_allow_html=True)
        else:
            frame_idx = max_history_len - 1  # Will use animation frames
        
        st.session_state.current_frame = frame_idx
    else:
        frame_idx = trail_length - 1
        view_mode = "Static (Current)"
        max_history_len = trail_length
        dates_available = []
    
    # Select stocks for graph
    df_graph = select_graph_stocks(df, min_stocks=50)
    
    # Calculate dynamic range
    x_min = df['RS-Ratio'].min() - 2
    x_max = df['RS-Ratio'].max() + 2
    y_min = df['RS-Momentum'].min() - 2
    y_max = df['RS-Momentum'].max() + 2
    x_range = max(abs(100 - x_min), abs(x_max - 100))
    y_range = max(abs(100 - y_min), abs(y_max - 100))
    
    # Create figure based on view mode
    if view_mode == "Animated" and max_history_len > trail_length:
        # ========================================================================
        # ANIMATED VIEW - Using Plotly Animation Frames
        # ========================================================================
        fig = go.Figure()
        
        # Quadrant backgrounds (static)
        fig.add_shape(type="rect", x0=100, y0=100, x1=100+x_range+2, y1=100+y_range+2,
                      fillcolor=QUADRANT_BG_COLORS["Leading"], line_width=0, layer="below")
        fig.add_shape(type="rect", x0=100-x_range-2, y0=100, x1=100, y1=100+y_range+2,
                      fillcolor=QUADRANT_BG_COLORS["Improving"], line_width=0, layer="below")
        fig.add_shape(type="rect", x0=100-x_range-2, y0=100-y_range-2, x1=100, y1=100,
                      fillcolor=QUADRANT_BG_COLORS["Lagging"], line_width=0, layer="below")
        fig.add_shape(type="rect", x0=100, y0=100-y_range-2, x1=100+x_range+2, y1=100,
                      fillcolor=QUADRANT_BG_COLORS["Weakening"], line_width=0, layer="below")
        
        # Center lines (static)
        fig.add_hline(y=100, line_color="rgba(100,100,100,0.6)", line_width=1.5)
        fig.add_vline(x=100, line_color="rgba(100,100,100,0.6)", line_width=1.5)
        
        # Quadrant labels (static)
        label_offset_x = x_range * 0.65
        label_offset_y = y_range * 0.75
        fig.add_annotation(x=100+label_offset_x, y=100+label_offset_y, text="<b>LEADING</b>",
                           showarrow=False, font=dict(size=14, color=QUADRANT_COLORS["Leading"]))
        fig.add_annotation(x=100-label_offset_x, y=100+label_offset_y, text="<b>IMPROVING</b>",
                           showarrow=False, font=dict(size=14, color=QUADRANT_COLORS["Improving"]))
        fig.add_annotation(x=100-label_offset_x, y=100-label_offset_y, text="<b>LAGGING</b>",
                           showarrow=False, font=dict(size=14, color=QUADRANT_COLORS["Lagging"]))
        fig.add_annotation(x=100+label_offset_x, y=100-label_offset_y, text="<b>WEAKENING</b>",
                           showarrow=False, font=dict(size=14, color=QUADRANT_COLORS["Weakening"]))
        
        # Prepare animation frames
        frames = []
        slider_steps = []
        
        # Create initial traces (one for each stock - head markers and trails)
        symbols_list = [row['Symbol'] for _, row in df_graph.iterrows() if row['Symbol'] in rs_history]
        
        # Initial frame data
        initial_frame_idx = trail_length - 1
        for sym in symbols_list:
            hist = rs_history[sym]
            end_idx = min(initial_frame_idx + 1, len(hist['rs_ratio']))
            start_idx = max(0, end_idx - trail_length)
            
            x_pts = np.array(hist['rs_ratio'][start_idx:end_idx], dtype=float)
            y_pts = np.array(hist['rs_momentum'][start_idx:end_idx], dtype=float)
            
            if len(x_pts) > 0:
                head_x, head_y = x_pts[-1], y_pts[-1]
                color, status = get_quadrant_color(head_x, head_y)
                
                # Trail line
                fig.add_trace(go.Scatter(
                    x=x_pts,
                    y=y_pts,
                    mode='lines+markers',
                    line=dict(color=color, width=2),
                    marker=dict(size=6, color=color, opacity=0.7, line=dict(color='white', width=1)),
                    hoverinfo='skip',
                    showlegend=False,
                    name=f"trail_{sym}"
                ))
                
                # Head marker
                fig.add_trace(go.Scatter(
                    x=[head_x],
                    y=[head_y],
                    mode='markers+text',
                    marker=dict(size=12, color=color, line=dict(color='white', width=2)),
                    text=[f"<b>{sym}</b>"] if label_toggle else [""],
                    textposition="top center",
                    textfont=dict(size=9, color=color),
                    hovertemplate=f"<b>{sym}</b><br>RS-Ratio: %{{x:.2f}}<br>RS-Mom: %{{y:.2f}}<extra></extra>",
                    showlegend=False,
                    name=f"head_{sym}"
                ))
        
        # Generate frames for animation
        for f_idx in range(trail_length - 1, max_history_len):
            frame_data = []
            
            for sym in symbols_list:
                hist = rs_history[sym]
                end_idx = min(f_idx + 1, len(hist['rs_ratio']))
                start_idx = max(0, end_idx - trail_length)
                
                x_pts = hist['rs_ratio'][start_idx:end_idx]
                y_pts = hist['rs_momentum'][start_idx:end_idx]
                
                if len(x_pts) > 0:
                    head_x, head_y = x_pts[-1], y_pts[-1]
                    color, _ = get_quadrant_color(head_x, head_y)
                    
                    # Trail
                    frame_data.append(go.Scatter(
                        x=x_pts,
                        y=y_pts,
                        mode='lines+markers',
                        line=dict(color=color, width=2),
                        marker=dict(size=6, color=color, opacity=0.7, line=dict(color='white', width=1)),
                    ))
                    
                    # Head
                    frame_data.append(go.Scatter(
                        x=[head_x],
                        y=[head_y],
                        mode='markers+text',
                        marker=dict(size=12, color=color, line=dict(color='white', width=2)),
                        text=[f"<b>{sym}</b>"] if label_toggle else [""],
                        textposition="top center",
                        textfont=dict(size=9, color=color),
                    ))
                else:
                    frame_data.append(go.Scatter(x=[], y=[]))
                    frame_data.append(go.Scatter(x=[], y=[]))
            
            # Get frame date
            frame_date = dates_available[f_idx] if f_idx < len(dates_available) else f"Frame {f_idx}"
            
            frames.append(go.Frame(data=frame_data, name=str(f_idx)))
            slider_steps.append({
                "args": [[str(f_idx)], {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"}],
                "label": frame_date[-5:] if len(frame_date) > 5 else frame_date,  # Short date label
                "method": "animate"
            })
        
        fig.frames = frames
        
        # Legend (static)
        for status in ["Leading", "Improving", "Weakening", "Lagging"]:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=QUADRANT_COLORS[status]),
                name=status,
                showlegend=True
            ))
        
        # Animation controls
        fig.update_layout(
            height=700,
            plot_bgcolor='#fafafa',
            paper_bgcolor='#f5f5f5',
            font=dict(color='#333333', size=12, family='Inter, sans-serif'),
            xaxis=dict(
                title=dict(text="<b>JdK RS-RATIO</b>", font=dict(size=12, color='#666')),
                range=[100-x_range-1, 100+x_range+1],
                gridcolor='rgba(150,150,150,0.15)',
                zeroline=False,
                tickfont=dict(color='#666')
            ),
            yaxis=dict(
                title=dict(text="<b>JdK RS-MOMENTUM</b>", font=dict(size=12, color='#666')),
                range=[100-y_range-1, 100+y_range+1],
                gridcolor='rgba(150,150,150,0.15)',
                zeroline=False,
                tickfont=dict(color='#666')
            ),
            legend=dict(
                x=1.02, y=1,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#dee2e6',
                borderwidth=1,
                font=dict(size=11)
            ),
            hovermode='closest',
            margin=dict(l=60, r=120, t=60, b=100),
            updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, {"frame": {"duration": 400, "redraw": True},
                                          "fromcurrent": True,
                                          "transition": {"duration": 200}}],
                            "label": "▶ Play",
                            "method": "animate"
                        },
                        {
                            "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                            "mode": "immediate",
                                            "transition": {"duration": 0}}],
                            "label": "⏸ Pause",
                            "method": "animate"
                        }
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "showactive": False,
                    "type": "buttons",
                    "x": 0.1,
                    "xanchor": "right",
                    "y": 0,
                    "yanchor": "top"
                }
            ],
            sliders=[{
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 14, "color": "#1565c0"},
                    "prefix": "Date: ",
                    "visible": True,
                    "xanchor": "right"
                },
                "transition": {"duration": 200},
                "pad": {"b": 10, "t": 50},
                "len": 0.9,
                "x": 0.1,
                "y": 0,
                "steps": slider_steps
            }]
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
        st.info("🎬 Use the Play/Pause buttons and timeline slider below the chart to animate stock rotation over time.")
    
    else:
        # ========================================================================
        # STATIC VIEW
        # ========================================================================
        fig = go.Figure()
        
        # Quadrant backgrounds
        fig.add_shape(type="rect", x0=100, y0=100, x1=100+x_range+2, y1=100+y_range+2,
                      fillcolor=QUADRANT_BG_COLORS["Leading"], line_width=0, layer="below")
        fig.add_shape(type="rect", x0=100-x_range-2, y0=100, x1=100, y1=100+y_range+2,
                      fillcolor=QUADRANT_BG_COLORS["Improving"], line_width=0, layer="below")
        fig.add_shape(type="rect", x0=100-x_range-2, y0=100-y_range-2, x1=100, y1=100,
                      fillcolor=QUADRANT_BG_COLORS["Lagging"], line_width=0, layer="below")
        fig.add_shape(type="rect", x0=100, y0=100-y_range-2, x1=100+x_range+2, y1=100,
                      fillcolor=QUADRANT_BG_COLORS["Weakening"], line_width=0, layer="below")
        
        # Center lines
        fig.add_hline(y=100, line_color="rgba(100,100,100,0.6)", line_width=1.5)
        fig.add_vline(x=100, line_color="rgba(100,100,100,0.6)", line_width=1.5)
        
        # Quadrant labels
        label_offset_x = x_range * 0.65
        label_offset_y = y_range * 0.75
        fig.add_annotation(x=100+label_offset_x, y=100+label_offset_y, text="<b>LEADING</b>",
                           showarrow=False, font=dict(size=14, color=QUADRANT_COLORS["Leading"]))
        fig.add_annotation(x=100-label_offset_x, y=100+label_offset_y, text="<b>IMPROVING</b>",
                           showarrow=False, font=dict(size=14, color=QUADRANT_COLORS["Improving"]))
        fig.add_annotation(x=100-label_offset_x, y=100-label_offset_y, text="<b>LAGGING</b>",
                           showarrow=False, font=dict(size=14, color=QUADRANT_COLORS["Lagging"]))
        fig.add_annotation(x=100+label_offset_x, y=100-label_offset_y, text="<b>WEAKENING</b>",
                           showarrow=False, font=dict(size=14, color=QUADRANT_COLORS["Weakening"]))
        
        # Current frame index
        frame_idx = st.session_state.current_frame
        
        # Plot each stock with trail
        for _, row in df_graph.iterrows():
            sym = row['Symbol']
            
            if sym not in rs_history:
                continue
            
            hist = rs_history[sym]
            
            # Get data up to current frame
            end_idx = min(frame_idx + 1, len(hist['rs_ratio']))
            start_idx = max(0, end_idx - trail_length)
            
            x_pts = np.array(hist['rs_ratio'][start_idx:end_idx], dtype=float)
            y_pts = np.array(hist['rs_momentum'][start_idx:end_idx], dtype=float)
            n_pts = len(x_pts)
            
            if n_pts == 0:
                continue
            
            # Color based on head position
            head_x = x_pts[-1]
            head_y = y_pts[-1]
            color, status = get_quadrant_color(head_x, head_y)
            
            # Draw trail with smooth curve
            if n_pts >= 2:
                # Smooth spline if enough points
                if n_pts >= 3:
                    x_smooth, y_smooth = smooth_spline_curve(x_pts, y_pts, points_per_segment=6)
                else:
                    x_smooth, y_smooth = x_pts, y_pts
                
                n_smooth = len(x_smooth)
                
                # Draw gradient trail
                for i in range(n_smooth - 1):
                    prog = i / max(1, n_smooth - 2)
                    line_width = 2 + prog * 3
                    opacity = 0.3 + prog * 0.7
                    fig.add_trace(go.Scatter(
                        x=[x_smooth[i], x_smooth[i+1]],
                        y=[y_smooth[i], y_smooth[i+1]],
                        mode='lines',
                        line=dict(color=color, width=line_width),
                        opacity=opacity,
                        hoverinfo='skip',
                        showlegend=False,
                    ))
                
                # Trail markers
                trail_sizes = [4 + (i / max(1, n_pts - 1)) * 4 for i in range(n_pts)]
                if n_pts > 1:
                    fig.add_trace(go.Scatter(
                        x=x_pts[:-1],
                        y=y_pts[:-1],
                        mode='markers',
                        marker=dict(size=trail_sizes[:-1], color=color, opacity=0.6,
                                   line=dict(color='white', width=1)),
                        hoverinfo='skip',
                        showlegend=False,
                    ))
                
                # Direction arrow
                dx = x_pts[-1] - x_pts[-2]
                dy = y_pts[-1] - y_pts[-2]
                length = np.sqrt(dx**2 + dy**2)
                if length > 0.01:
                    fig.add_annotation(
                        x=x_pts[-1],
                        y=y_pts[-1],
                        ax=x_pts[-1] - dx/length * 0.35,
                        ay=y_pts[-1] - dy/length * 0.35,
                        xref='x', yref='y', axref='x', ayref='y',
                        showarrow=True,
                        arrowhead=2,
                        arrowsize=1.5,
                        arrowwidth=2.5,
                        arrowcolor=color,
                    )
            
            # Head marker
            hover_text = f"<b>{sym}</b><br>Status: {status}<br>RS-Ratio: {head_x:.2f}<br>RS-Mom: {head_y:.2f}"
            fig.add_trace(go.Scatter(
                x=[head_x],
                y=[head_y],
                mode='markers',
                marker=dict(size=12, color=color, line=dict(color='white', width=2)),
                text=[hover_text],
                hoverinfo='text',
                showlegend=False,
            ))
            
            # Label
            if label_toggle:
                fig.add_annotation(
                    x=head_x,
                    y=head_y,
                    text=f"<b>{sym}</b>",
                    showarrow=False,
                    font=dict(size=9, color=color),
                    yshift=12,
                )
        
        # Legend
        for status in ["Leading", "Improving", "Weakening", "Lagging"]:
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=QUADRANT_COLORS[status]),
                name=status,
                showlegend=True
            ))
        
        # Layout
        fig.update_layout(
            height=650,
            plot_bgcolor='#fafafa',
            paper_bgcolor='#f5f5f5',
            font=dict(color='#333333', size=12, family='Inter, sans-serif'),
            xaxis=dict(
                title=dict(text="<b>JdK RS-RATIO</b>", font=dict(size=12, color='#666')),
                range=[100-x_range-1, 100+x_range+1],
                gridcolor='rgba(150,150,150,0.15)',
                zeroline=False,
                tickfont=dict(color='#666')
            ),
            yaxis=dict(
                title=dict(text="<b>JdK RS-MOMENTUM</b>", font=dict(size=12, color='#666')),
                range=[100-y_range-1, 100+y_range+1],
                gridcolor='rgba(150,150,150,0.15)',
                zeroline=False,
                tickfont=dict(color='#666')
            ),
            legend=dict(
                x=1.02, y=1,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#dee2e6',
                borderwidth=1,
                font=dict(size=11)
            ),
            hovermode='closest',
            margin=dict(l=60, r=120, t=40, b=60),
        )
        
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
    
    # Stats bar
    stats_cols = st.columns(5)
    status_counts = df['Status'].value_counts()
    
    with stats_cols[0]:
        st.metric("Total Stocks", len(df))
    with stats_cols[1]:
        st.metric("🟢 Leading", status_counts.get("Leading", 0))
    with stats_cols[2]:
        st.metric("🟣 Improving", status_counts.get("Improving", 0))
    with stats_cols[3]:
        st.metric("🟡 Weakening", status_counts.get("Weakening", 0))
    with stats_cols[4]:
        st.metric("🔴 Lagging", status_counts.get("Lagging", 0))
    
    # Quadrant details expanders
    st.markdown("---")
    st.markdown("### 📊 Quadrant Details")
    
    quad_cols = st.columns(4)
    
    for idx, status in enumerate(["Leading", "Improving", "Weakening", "Lagging"]):
        with quad_cols[idx]:
            df_status = df[df['Status'] == status].head(20)
            color = QUADRANT_COLORS[status]
            icon = {"Leading": "🟢", "Improving": "🟣", "Weakening": "🟡", "Lagging": "🔴"}[status]
            
            with st.expander(f"{icon} {status} ({len(df[df['Status'] == status])})", expanded=(status == "Leading")):
                for _, row in df_status.iterrows():
                    st.markdown(f"""
                    <div style="padding:4px 8px; margin:2px 0; background:#f8f9fa; border-left:3px solid {color}; border-radius:4px;">
                        <a href="{row['TV Link']}" target="_blank" style="font-weight:600; color:{color}; text-decoration:none;">{row['Symbol']}</a>
                        <span style="font-size:11px; color:#666;"> | RS: {row['RS-Ratio']:.1f} | Mom: {row['RS-Momentum']:.1f}</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Export
    if st.checkbox("📥 Export CSV"):
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="Download Data",
            data=csv_buffer.getvalue(),
            file_name=f"RRG_{csv_selected}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align:center; padding:60px 20px; background:#ffffff; border-radius:12px; border:1px solid #dee2e6; margin-top:40px;">
        <h2 style="color:#333; margin-bottom:20px;">📈 Relative Rotation Graph Dashboard</h2>
        <p style="color:#666; font-size:16px; margin-bottom:30px;">
            Analyze stock rotation patterns with Optuma-style RRG visualization.<br>
            Select your parameters above and click <b>Load</b> to begin.
        </p>
        <div style="display:flex; justify-content:center; gap:40px; margin-top:30px;">
            <div style="text-align:center;">
                <div style="width:50px; height:50px; background:#90EE90; border-radius:8px; margin:0 auto 10px;"></div>
                <p style="font-weight:600; color:#228B22;">Leading</p>
            </div>
            <div style="text-align:center;">
                <div style="width:50px; height:50px; background:#DDD6FE; border-radius:8px; margin:0 auto 10px;"></div>
                <p style="font-weight:600; color:#8B5CF6;">Improving</p>
            </div>
            <div style="text-align:center;">
                <div style="width:50px; height:50px; background:#FED7AA; border-radius:8px; margin:0 auto 10px;"></div>
                <p style="font-weight:600; color:#D97706;">Weakening</p>
            </div>
            <div style="text-align:center;">
                <div style="width:50px; height:50px; background:#FEB2B2; border-radius:8px; margin:0 auto 10px;"></div>
                <p style="font-weight:600; color:#DC2626;">Lagging</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#999; font-size:11px;">
    RRG Analysis | Data: Yahoo Finance | Reference: <a href="https://www.optuma.com/blog/scripting-for-rrgs" style="color:#2196F3;">Optuma RRG Guide</a>
</div>
""", unsafe_allow_html=True)

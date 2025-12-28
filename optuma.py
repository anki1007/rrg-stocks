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
    page_title="RRG Dashboard - Multi Theme",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# THEME CONFIGURATION - Multiple Themes
# ============================================================================
THEMES = {
    "dark": {
        "name": "Dark Pro",
        "bg": "#0b0e13",
        "bg_card": "#10141b",
        "bg_secondary": "#1a1f2e",
        "bg_control": "#151a24",
        "border": "#2d3748",
        "text": "#e6eaee",
        "text_secondary": "#a0aec0",
        "text_muted": "#718096",
        "accent": "#4299e1",
        "accent_hover": "#3182ce",
        "success": "#48bb78",
        "table_header": "#1a2230",
        "table_row": "#0d1117",
        "table_row_alt": "#121823",
    },
    "light": {
        "name": "Light Classic",
        "bg": "#f7fafc",
        "bg_card": "#ffffff",
        "bg_secondary": "#9D00FF",
        "bg_control": "#87CEEB",
        "border": "#cbd5e0",
        "text": "#1a202c",
        "text_secondary": "#4a5568",
        "text_muted": "#718096",
        "accent": "#3182ce",
        "accent_hover": "#2c5282",
        "success": "#38a169",
        "table_header": "#e2e8f0",
        "table_row": "#ffffff",
        "table_row_alt": "#f7fafc",
    },
    "matrix": {
        "name": "Matrix Green",
        "bg": "#0a0f0a",
        "bg_card": "#0d140d",
        "bg_secondary": "#142014",
        "bg_control": "#0f1a0f",
        "border": "#1a3d1a",
        "text": "#00ff41",
        "text_secondary": "#00cc33",
        "text_muted": "#008f26",
        "accent": "#00ff41",
        "accent_hover": "#00cc33",
        "success": "#00ff41",
        "table_header": "#142014",
        "table_row": "#0a0f0a",
        "table_row_alt": "#0d140d",
    },
    "arctic": {
        "name": "Arctic Blue",
        "bg": "#0a1628",
        "bg_card": "#0f1f3d",
        "bg_secondary": "#142952",
        "bg_control": "#0d1a33",
        "border": "#1e3a5f",
        "text": "#e0f2fe",
        "text_secondary": "#7dd3fc",
        "text_muted": "#38bdf8",
        "accent": "#0ea5e9",
        "accent_hover": "#0284c7",
        "success": "#22d3ee",
        "table_header": "#142952",
        "table_row": "#0a1628",
        "table_row_alt": "#0f1f3d",
    },
    "sunset": {
        "name": "Sunset Trading",
        "bg": "#1a1018",
        "bg_card": "#261620",
        "bg_secondary": "#331c28",
        "bg_control": "#1f1218",
        "border": "#4a2838",
        "text": "#fce7f3",
        "text_secondary": "#f9a8d4",
        "text_muted": "#ec4899",
        "accent": "#f97316",
        "accent_hover": "#ea580c",
        "success": "#fb923c",
        "table_header": "#331c28",
        "table_row": "#1a1018",
        "table_row_alt": "#261620",
    },
    "cyberpunk": {
        "name": "Cyberpunk Neon",
        "bg": "#0d0221",
        "bg_card": "#150530",
        "bg_secondary": "#1a0a3e",
        "bg_control": "#120428",
        "border": "#6b21a8",
        "text": "#f0abfc",
        "text_secondary": "#e879f9",
        "text_muted": "#c026d3",
        "accent": "#f0f",
        "accent_hover": "#d946ef",
        "success": "#22d3ee",
        "table_header": "#1a0a3e",
        "table_row": "#0d0221",
        "table_row_alt": "#150530",
    },
}

# Chart colors - ALWAYS CONSISTENT (white bg with pastel quadrants)
CHART_COLORS = {
    "plot_bg": "#fafafa",
    "paper_bg": "#ffffff",
    "grid": "rgba(150,150,150,0.2)",
    "axis_text": "#4a5568",
    "axis_title": "#2d3748",
}

# Quadrant colors - same as Optuma reference
QUADRANT_COLORS = {
    "Leading": "#228B22",
    "Improving": "#7c3aed",
    "Weakening": "#d97706",
    "Lagging": "#dc2626"
}

QUADRANT_BG_COLORS = {
    "Leading": "rgba(187, 247, 208, 0.6)",
    "Improving": "rgba(233, 213, 255, 0.6)",
    "Weakening": "rgba(254, 243, 199, 0.6)",
    "Lagging": "rgba(254, 202, 202, 0.6)"
}

# Initialize theme
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

theme = THEMES[st.session_state.theme]

# ============================================================================
# DYNAMIC CSS
# ============================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp {{
    background: {theme['bg']} !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}}

.block-container {{
    padding: 0.5rem 1rem !important;
    max-width: 100% !important;
}}

/* Control Bar Container */
.control-bar {{
    background: {theme['bg_control']};
    border: 1px solid {theme['border']};
    border-radius: 8px;
    padding: 8px 16px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 16px;
    flex-wrap: nowrap;
    overflow-x: auto;
}}

.control-group {{
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: fit-content;
}}

.control-label {{
    font-size: 10px;
    font-weight: 600;
    color: {theme['text_muted']};
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}

.control-select {{
    background: {theme['bg_card']};
    border: 1px solid {theme['border']};
    border-radius: 4px;
    padding: 6px 10px;
    color: {theme['text']};
    font-size: 12px;
    font-weight: 500;
    min-width: 100px;
}}

.counts-display {{
    background: {theme['accent']};
    color: white;
    padding: 6px 16px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 13px;
    white-space: nowrap;
}}

.date-badge {{
    background: {theme['accent']};
    color: white;
    padding: 8px 16px;
    border-radius: 4px;
    font-weight: 600;
    font-size: 12px;
    white-space: nowrap;
}}

.btn-group {{
    display: flex;
    gap: 4px;
}}

.ctrl-btn {{
    background: {theme['bg_card']};
    border: 1px solid {theme['border']};
    color: {theme['text']};
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    white-space: nowrap;
}}

.ctrl-btn:hover {{
    border-color: {theme['accent']};
    color: {theme['accent']};
}}

.ctrl-btn.active {{
    background: {theme['accent']};
    color: white;
    border-color: {theme['accent']};
}}

/* Streamlit overrides */
div[data-baseweb="select"] > div {{
    background: {theme['bg_card']} !important;
    border-color: {theme['border']} !important;
    min-height: 32px !important;
}}

div[data-baseweb="select"] span {{
    color: {theme['text']} !important;
    font-size: 12px !important;
}}

.stSelectbox, .stNumberInput {{
    min-width: 80px;
}}

.stSelectbox > div > div {{
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}}

.stButton > button {{
    background: {theme['bg_card']} !important;
    border: 1px solid {theme['border']} !important;
    color: {theme['text']} !important;
    padding: 4px 12px !important;
    font-size: 12px !important;
    min-height: 32px !important;
}}

.stButton > button:hover {{
    border-color: {theme['accent']} !important;
    color: {theme['accent']} !important;
}}

.stCheckbox label span {{
    color: {theme['text']} !important;
    font-size: 12px !important;
}}

/* Success/info messages */
.stSuccess, .stInfo {{
    background: {theme['bg_secondary']} !important;
    border: 1px solid {theme['border']} !important;
    color: {theme['text']} !important;
}}

/* Slider */
.stSlider > div > div > div {{
    background: {theme['accent']} !important;
}}

.stSlider label {{
    color: {theme['text']} !important;
}}

/* Expander */
.streamlit-expanderHeader {{
    background: {theme['bg_secondary']} !important;
    color: {theme['text']} !important;
    border-radius: 6px !important;
}}

/* Metrics */
[data-testid="stMetric"] {{
    background: {theme['bg_card']};
    padding: 10px;
    border-radius: 6px;
    border: 1px solid {theme['border']};
}}

[data-testid="stMetricValue"] {{
    color: {theme['text']} !important;
    font-size: 18px !important;
}}

[data-testid="stMetricLabel"] {{
    color: {theme['text_secondary']} !important;
}}

/* Hide defaults */
#MainMenu, footer, header {{visibility: hidden;}}

/* Radio buttons inline */
.stRadio > div {{
    flex-direction: row !important;
    gap: 8px !important;
}}

.stRadio label {{
    color: {theme['text']} !important;
    font-size: 12px !important;
}}

/* Number input compact */
.stNumberInput > div > div > input {{
    background: {theme['bg_card']} !important;
    color: {theme['text']} !important;
    border-color: {theme['border']} !important;
}}
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
    "1M": 21,
    "3M": 63,
    "6M": 126,
    "1Y": 252,
    "2Y": 504,
    "3Y": 756,
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
    return f"https://www.tradingview.com/chart/?symbol=NSE:{sym.replace('.NS', '')}"

def format_symbol(sym):
    return sym.replace('.NS', '')

def get_heading_direction(heading):
    dirs = [("→ E", 0), ("↗ NE", 45), ("↑ N", 90), ("↖ NW", 135), 
            ("← W", 180), ("↙ SW", 225), ("↓ S", 270), ("↘ SE", 315)]
    for d, angle in dirs:
        if abs(heading - angle) < 22.5 or abs(heading - angle - 360) < 22.5:
            return d
    return "→ E"

def select_graph_stocks(df, min_stocks=50):
    graph_stocks = []
    for status in ["Leading", "Improving", "Weakening", "Lagging"]:
        df_quad = df[df['Status'] == status].copy()
        if len(df_quad) == 0:
            continue
        elif len(df_quad) < 15:
            graph_stocks.extend(df_quad.index.tolist())
        else:
            top = df_quad.nlargest(15, 'RRG Power') if status in ["Leading", "Improving"] else df_quad.nsmallest(15, 'RRG Power')
            graph_stocks.extend(top.index.tolist())
    
    if len(graph_stocks) < min_stocks:
        remaining = df.index.difference(graph_stocks)
        additional = df.loc[remaining].nlargest(min_stocks - len(graph_stocks), 'RRG Power')
        graph_stocks.extend(additional.index.tolist())
    
    return df.loc[graph_stocks]

def generate_table_html(df, theme):
    industries = sorted(df['Industry'].unique().tolist())
    ind_opts = ''.join([f'<option value="{i}">{i}</option>' for i in industries])
    
    rows = ""
    for _, r in df.iterrows():
        c, s = get_quadrant_color(r['RS-Ratio'], r['RS-Momentum'])
        rows += f"""<tr>
            <td>{int(r['Sl No.'])}</td>
            <td><a href="{r['TV Link']}" target="_blank" style="color:{theme['accent']};font-weight:600;">{r['Symbol']}</a></td>
            <td>{r['Name'][:22]}</td>
            <td>{r['Industry'][:18]}</td>
            <td style="text-align:right;">₹{r['Price']:,.2f}</td>
            <td><span style="background:{c};color:#fff;padding:2px 6px;border-radius:3px;font-size:10px;">{s}</span></td>
            <td style="text-align:right;">{r['RS-Ratio']:.2f}</td>
            <td style="text-align:right;">{r['RS-Momentum']:.2f}</td>
            <td style="text-align:right;font-weight:600;">{r['RRG Power']:.2f}</td>
            <td style="text-align:center;">{r['Direction']}</td>
        </tr>"""
    
    return f"""<!DOCTYPE html><html><head><style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        *{{box-sizing:border-box;margin:0;padding:0;}}
        body{{font-family:'Inter',sans-serif;background:{theme['bg_card']};color:{theme['text']};}}
        .controls{{padding:10px;background:{theme['bg_secondary']};border-bottom:1px solid {theme['border']};display:flex;gap:10px;flex-wrap:wrap;}}
        input,select{{padding:6px 10px;background:{theme['bg_card']};border:1px solid {theme['border']};border-radius:4px;color:{theme['text']};font-size:12px;}}
        input:focus,select:focus{{outline:none;border-color:{theme['accent']};}}
        .badge{{background:{theme['accent']};color:#fff;padding:5px 12px;border-radius:12px;font-size:12px;font-weight:600;}}
        .wrap{{max-height:400px;overflow:auto;}}
        table{{width:100%;border-collapse:collapse;font-size:12px;}}
        th{{position:sticky;top:0;background:{theme['table_header']};color:{theme['text_secondary']};padding:10px 8px;text-align:left;font-weight:600;border-bottom:2px solid {theme['border']};cursor:pointer;}}
        th:hover{{background:{theme['bg_secondary']};}}
        td{{padding:8px;border-bottom:1px solid {theme['border']};}}
        tr:nth-child(odd){{background:{theme['table_row']};}}
        tr:nth-child(even){{background:{theme['table_row_alt']};}}
        tr:hover{{background:{theme['bg_secondary']};}}
        a{{text-decoration:none;}}
        .hidden{{display:none;}}
    </style></head><body>
    <div class="controls">
        <input type="text" id="search" placeholder="🔍 Search..." onkeyup="filter()" style="min-width:180px;">
        <select id="statusF" onchange="filter()"><option value="">All Status</option><option>Leading</option><option>Improving</option><option>Weakening</option><option>Lagging</option></select>
        <select id="indF" onchange="filter()"><option value="">All Industries</option>{ind_opts}</select>
        <span class="badge" id="cnt">{len(df)}/{len(df)}</span>
    </div>
    <div class="wrap"><table><thead><tr>
        <th onclick="sort(0)">#</th><th onclick="sort(1)">Symbol</th><th onclick="sort(2)">Name</th><th onclick="sort(3)">Industry</th>
        <th onclick="sort(4)">Price</th><th onclick="sort(5)">Status</th><th onclick="sort(6)">RS-Ratio</th>
        <th onclick="sort(7)">RS-Mom</th><th onclick="sort(8)">Power</th><th onclick="sort(9)">Dir</th>
    </tr></thead><tbody id="tb">{rows}</tbody></table></div>
    <script>
        const total={len(df)},dir={{}};
        function sort(c){{const tb=document.getElementById('tb'),rs=Array.from(tb.rows);dir[c]=!dir[c];rs.sort((a,b)=>{{let av=a.cells[c].textContent.replace(/[₹,%]/g,''),bv=b.cells[c].textContent.replace(/[₹,%]/g,'');const an=parseFloat(av),bn=parseFloat(bv);if(!isNaN(an)&&!isNaN(bn))return dir[c]?an-bn:bn-an;return dir[c]?av.localeCompare(bv):bv.localeCompare(av);}});tb.innerHTML='';rs.forEach(r=>tb.appendChild(r));}}
        function filter(){{const s=document.getElementById('search').value.toLowerCase(),st=document.getElementById('statusF').value,ind=document.getElementById('indF').value,tb=document.getElementById('tb'),rs=tb.rows;let v=0;for(let r of rs){{const sym=r.cells[1].textContent.toLowerCase(),nm=r.cells[2].textContent.toLowerCase(),i=r.cells[3].textContent,stat=r.cells[5].textContent.trim();const m=(sym.includes(s)||nm.includes(s))&&(!st||stat===st)&&(!ind||i===ind);r.classList.toggle('hidden',!m);if(m)v++;}}document.getElementById('cnt').textContent=v+'/'+total;}}
    </script></body></html>"""

# ============================================================================
# SESSION STATE
# ============================================================================
for key, default in [("load_clicked", False), ("df_cache", None), ("rs_history_cache", {}), 
                     ("dates_cache", []), ("current_frame", 0), ("is_playing", False)]:
    if key not in st.session_state:
        st.session_state[key] = default

# ============================================================================
# CONTROL BAR - Compact Inline Layout
# ============================================================================
csv_files = list_csv_from_github() or ["NIFTY200"]
default_csv = next((i for i, c in enumerate(csv_files) if 'NIFTY200' in c.upper()), 0)

# Row 1: All controls inline
c1, c2, c3, c4, c5, c6, c7, c8 = st.columns([1.2, 1.2, 1, 1, 0.8, 0.8, 1.8, 1.2])

with c1:
    st.markdown(f"<span style='font-size:10px;color:{theme['text_muted']};font-weight:600;'>BENCHMARK</span>", unsafe_allow_html=True)
    csv_selected = st.selectbox("b", csv_files, index=default_csv, key="csv", label_visibility="collapsed")

with c2:
    st.markdown(f"<span style='font-size:10px;color:{theme['text_muted']};font-weight:600;'>VS INDEX</span>", unsafe_allow_html=True)
    bench_name = st.selectbox("v", list(BENCHMARKS.keys()), index=2, key="bench", label_visibility="collapsed")

with c3:
    st.markdown(f"<span style='font-size:10px;color:{theme['text_muted']};font-weight:600;'>TIMEFRAME</span>", unsafe_allow_html=True)
    tf_name = st.selectbox("t", list(TIMEFRAMES.keys()), index=5, key="tf", label_visibility="collapsed")

with c4:
    st.markdown(f"<span style='font-size:10px;color:{theme['text_muted']};font-weight:600;'>DATE RANGE</span>", unsafe_allow_html=True)
    date_range = st.selectbox("d", list(DATE_RANGES.keys()), index=1, key="dr", label_visibility="collapsed")

with c5:
    st.markdown(f"<span style='font-size:10px;color:{theme['text_muted']};font-weight:600;'>COUNTS</span>", unsafe_allow_html=True)
    trail_length = st.number_input("c", min_value=1, max_value=14, value=5, key="trail", label_visibility="collapsed")

with c6:
    st.markdown(f"<span style='font-size:10px;color:{theme['text_muted']};'>&nbsp;</span>", unsafe_allow_html=True)
    st.markdown(f"<div style='background:{theme['accent']};color:white;padding:6px 12px;border-radius:4px;text-align:center;font-weight:600;font-size:13px;margin-top:2px;'>{trail_length} Days</div>", unsafe_allow_html=True)

with c7:
    st.markdown(f"<span style='font-size:10px;color:{theme['text_muted']};'>&nbsp;</span>", unsafe_allow_html=True)
    bc1, bc2, bc3, bc4 = st.columns(4)
    with bc1:
        load_btn = st.button("📥 Load", key="load", use_container_width=True)
    with bc2:
        play_btn = st.button("▶ Play", key="play", use_container_width=True)
    with bc3:
        stop_btn = st.button("⏹ Stop", key="stop", use_container_width=True)
    with bc4:
        label_on = st.checkbox("Label", value=True, key="lbl")

with c8:
    st.markdown(f"<span style='font-size:10px;color:{theme['text_muted']};font-weight:600;'>THEME</span>", unsafe_allow_html=True)
    theme_choice = st.selectbox("th", list(THEMES.keys()), 
                                index=list(THEMES.keys()).index(st.session_state.theme),
                                format_func=lambda x: THEMES[x]['name'],
                                key="theme_sel", label_visibility="collapsed")

# Handle theme change
if theme_choice != st.session_state.theme:
    st.session_state.theme = theme_choice
    st.rerun()

if load_btn:
    st.session_state.load_clicked = True
    st.session_state.current_frame = trail_length - 1

if play_btn:
    st.session_state.is_playing = True

if stop_btn:
    st.session_state.is_playing = False

# ============================================================================
# DATA LOADING
# ============================================================================
if st.session_state.load_clicked:
    try:
        interval, yf_period = TIMEFRAMES[tf_name]
        universe = load_universe(csv_selected)
        
        if universe.empty:
            st.error("❌ Failed to load data")
            st.stop()
        
        symbols = universe['Symbol'].tolist()
        names_dict = dict(zip(universe['Symbol'], universe['Company Name']))
        industries_dict = dict(zip(universe['Symbol'], universe['Industry']))
        
        with st.spinner(f"Loading {len(symbols)} symbols..."):
            raw = yf.download(symbols + [BENCHMARKS[bench_name]], interval=interval, 
                            period=yf_period, auto_adjust=True, progress=False, threads=True)
        
        if BENCHMARKS[bench_name] not in raw['Close'].columns:
            st.error("❌ Benchmark unavailable")
            st.stop()
        
        bench = raw['Close'][BENCHMARKS[bench_name]]
        rows, rs_history = [], {}
        dates_list = raw.index.tolist()[-DATE_RANGES[date_range]:]
        
        for s in symbols:
            if s not in raw['Close'].columns:
                continue
            try:
                rs_ratio, rs_momentum, distance, heading, velocity = calculate_jdk_rrg(raw['Close'][s], bench)
                if rs_ratio is None or len(rs_ratio) < 3:
                    continue
                
                max_hist = min(DATE_RANGES[date_range], len(rs_ratio))
                rs_history[format_symbol(s)] = {
                    'rs_ratio': rs_ratio.iloc[-max_hist:].tolist(),
                    'rs_momentum': rs_momentum.iloc[-max_hist:].tolist(),
                    'dates': [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d)[:10] for d in dates_list[-max_hist:]]
                }
                
                rsr, rsm = rs_ratio.iloc[-1], rs_momentum.iloc[-1]
                rows.append({
                    'Symbol': format_symbol(s), 'Name': names_dict.get(s, s),
                    'Industry': industries_dict.get(s, 'N/A'), 'Price': round(raw['Close'][s].iloc[-1], 2),
                    'RS-Ratio': round(rsr, 2), 'RS-Momentum': round(rsm, 2),
                    'RRG Power': round(np.sqrt((rsr-100)**2 + (rsm-100)**2), 2),
                    'Distance': round(distance.iloc[-1], 2), 'Direction': get_heading_direction(heading.iloc[-1]),
                    'Status': quadrant(rsr, rsm), 'TV Link': get_tv_link(s)
                })
            except:
                continue
        
        if rows:
            df = pd.DataFrame(rows)
            df['Rank'] = df['RRG Power'].rank(ascending=False, method='min').astype(int)
            df = df.sort_values('Rank')
            df['Sl No.'] = range(1, len(df)+1)
            st.session_state.df_cache = df
            st.session_state.rs_history_cache = rs_history
            st.session_state.current_frame = trail_length - 1
            st.success(f"✅ Loaded {len(df)} symbols")
        else:
            st.error("No data")
            st.stop()
    except Exception as e:
        st.error(f"Error: {e}")
        st.stop()

# ============================================================================
# DISPLAY
# ============================================================================
if st.session_state.df_cache is not None:
    df = st.session_state.df_cache
    rs_history = st.session_state.rs_history_cache
    
    # Timeline
    if rs_history:
        max_hist = max(len(rs_history[s]['rs_ratio']) for s in rs_history)
        sample = list(rs_history.keys())[0]
        dates_avail = rs_history[sample].get('dates', [])
        
        tc1, tc2, tc3 = st.columns([1, 7, 1.5])
        with tc1:
            mode = st.radio("", ["Static", "Animate"], horizontal=True, key="mode", label_visibility="collapsed")
        with tc2:
            frame_idx = st.slider("Timeline", trail_length-1, max_hist-1, 
                                 max_hist-1 if mode == "Static" else st.session_state.current_frame,
                                 key="timeline", label_visibility="collapsed")
            if mode == "Animate":
                st.session_state.current_frame = frame_idx
        with tc3:
            if dates_avail and frame_idx < len(dates_avail):
                st.markdown(f"<div style='background:{theme['accent']};color:white;padding:8px 12px;border-radius:4px;text-align:center;font-weight:600;font-size:12px;margin-top:18px;'>{dates_avail[frame_idx]}</div>", unsafe_allow_html=True)
        
        # Auto-play
        if mode == "Animate" and st.session_state.is_playing:
            import time
            if frame_idx < max_hist - 1:
                time.sleep(0.25)
                st.session_state.current_frame = frame_idx + 1
                st.rerun()
            else:
                st.session_state.is_playing = False
    else:
        frame_idx = trail_length - 1
        max_hist = trail_length
    
    # Graph
    df_graph = select_graph_stocks(df, 50)
    
    x_min, x_max = df['RS-Ratio'].min() - 2, df['RS-Ratio'].max() + 2
    y_min, y_max = df['RS-Momentum'].min() - 2, df['RS-Momentum'].max() + 2
    x_range = max(abs(100 - x_min), abs(x_max - 100))
    y_range = max(abs(100 - y_min), abs(y_max - 100))
    
    fig = go.Figure()
    
    # Quadrants (always same colors)
    fig.add_shape(type="rect", x0=100, y0=100, x1=100+x_range+2, y1=100+y_range+2,
                  fillcolor=QUADRANT_BG_COLORS["Leading"], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=100-x_range-2, y0=100, x1=100, y1=100+y_range+2,
                  fillcolor=QUADRANT_BG_COLORS["Improving"], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=100-x_range-2, y0=100-y_range-2, x1=100, y1=100,
                  fillcolor=QUADRANT_BG_COLORS["Lagging"], line_width=0, layer="below")
    fig.add_shape(type="rect", x0=100, y0=100-y_range-2, x1=100+x_range+2, y1=100,
                  fillcolor=QUADRANT_BG_COLORS["Weakening"], line_width=0, layer="below")
    
    # Center lines
    fig.add_hline(y=100, line_color="rgba(100,100,100,0.5)", line_width=1.5)
    fig.add_vline(x=100, line_color="rgba(100,100,100,0.5)", line_width=1.5)
    
    # Labels
    ox, oy = x_range * 0.65, y_range * 0.75
    for txt, x, y, c in [("LEADING", 100+ox, 100+oy, "Leading"), ("IMPROVING", 100-ox, 100+oy, "Improving"),
                         ("LAGGING", 100-ox, 100-oy, "Lagging"), ("WEAKENING", 100+ox, 100-oy, "Weakening")]:
        fig.add_annotation(x=x, y=y, text=f"<b>{txt}</b>", showarrow=False, 
                          font=dict(size=13, color=QUADRANT_COLORS[c]))
    
    # Stocks
    for _, row in df_graph.iterrows():
        sym = row['Symbol']
        if sym not in rs_history:
            continue
        
        h = rs_history[sym]
        end = min(frame_idx + 1, len(h['rs_ratio']))
        start = max(0, end - trail_length)
        
        xp = np.array(h['rs_ratio'][start:end], dtype=float)
        yp = np.array(h['rs_momentum'][start:end], dtype=float)
        
        if len(xp) == 0:
            continue
        
        hx, hy = xp[-1], yp[-1]
        color, status = get_quadrant_color(hx, hy)
        
        # Trail
        if len(xp) >= 2:
            xs, ys = (smooth_spline_curve(xp, yp, 6) if len(xp) >= 3 else (xp, yp))
            for i in range(len(xs)-1):
                p = i / max(1, len(xs)-2)
                fig.add_trace(go.Scatter(x=[xs[i], xs[i+1]], y=[ys[i], ys[i+1]], mode='lines',
                    line=dict(color=color, width=2+p*3), opacity=0.3+p*0.7, hoverinfo='skip', showlegend=False))
            
            # Arrow
            dx, dy = xp[-1]-xp[-2], yp[-1]-yp[-2]
            l = np.sqrt(dx**2+dy**2)
            if l > 0.01:
                fig.add_annotation(x=xp[-1], y=yp[-1], ax=xp[-1]-dx/l*0.3, ay=yp[-1]-dy/l*0.3,
                    xref='x', yref='y', axref='x', ayref='y', showarrow=True, arrowhead=2, 
                    arrowsize=1.5, arrowwidth=2.5, arrowcolor=color)
        
        # Head
        fig.add_trace(go.Scatter(x=[hx], y=[hy], mode='markers',
            marker=dict(size=11, color=color, line=dict(color='white', width=2)),
            hovertemplate=f"<b>{sym}</b><br>RS:{hx:.2f}<br>Mom:{hy:.2f}<extra></extra>", showlegend=False))
        
        if label_on:
            fig.add_annotation(x=hx, y=hy, text=f"<b>{sym}</b>", showarrow=False,
                              font=dict(size=9, color=color), yshift=12)
    
    # Legend
    for s in ["Leading", "Improving", "Weakening", "Lagging"]:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
            marker=dict(size=10, color=QUADRANT_COLORS[s]), name=s, showlegend=True))
    
    # Layout - WHITE CHART ALWAYS
    fig.update_layout(
        height=520,
        plot_bgcolor=CHART_COLORS['plot_bg'],
        paper_bgcolor=CHART_COLORS['paper_bg'],
        font=dict(color=CHART_COLORS['axis_text'], size=11, family='Inter'),
        xaxis=dict(title="<b>JdK RS-RATIO</b>", range=[100-x_range-1, 100+x_range+1],
                   gridcolor=CHART_COLORS['grid'], zeroline=False, tickfont=dict(color=CHART_COLORS['axis_text'])),
        yaxis=dict(title="<b>JdK RS-MOMENTUM</b>", range=[100-y_range-1, 100+y_range+1],
                   gridcolor=CHART_COLORS['grid'], zeroline=False, tickfont=dict(color=CHART_COLORS['axis_text'])),
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center", bgcolor='rgba(255,255,255,0.8)'),
        hovermode='closest',
        margin=dict(l=50, r=20, t=10, b=70),
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
    
    # Quadrant summary
    qc = st.columns(4)
    counts = df['Status'].value_counts()
    for i, (s, icon) in enumerate([("Leading","🟢"), ("Improving","🟣"), ("Weakening","🟡"), ("Lagging","🔴")]):
        with qc[i]:
            df_s = df[df['Status']==s].head(8)
            with st.expander(f"{icon} {s} ({counts.get(s,0)})", expanded=(s=="Leading")):
                for _, r in df_s.iterrows():
                    st.markdown(f"<div style='padding:3px 6px;background:{theme['bg_secondary']};border-left:3px solid {QUADRANT_COLORS[s]};border-radius:3px;margin:2px 0;font-size:11px;'><a href='{r['TV Link']}' target='_blank' style='color:{QUADRANT_COLORS[s]};font-weight:600;'>{r['Symbol']}</a> <span style='color:{theme['text_muted']}'>RS:{r['RS-Ratio']:.1f} Mom:{r['RS-Momentum']:.1f}</span></div>", unsafe_allow_html=True)
    
    # Table
    st.markdown("---")
    st.markdown(f"<h4 style='color:{theme['text']};margin:10px 0;'>📊 Analysis Table ({len(df)} stocks)</h4>", unsafe_allow_html=True)
    st.components.v1.html(generate_table_html(df, theme), height=480, scrolling=False)
    
    # Export
    col1, col2 = st.columns([1, 5])
    with col1:
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button("📥 CSV", buf.getvalue(), f"RRG_{csv_selected}_{datetime.now():%Y%m%d}.csv", "text/csv")

else:
    st.markdown(f"""
    <div style="text-align:center;padding:50px;background:{theme['bg_card']};border-radius:10px;border:1px solid {theme['border']};margin-top:30px;">
        <h2 style="color:{theme['text']};">📈 RRG Dashboard</h2>
        <p style="color:{theme['text_secondary']};">Select parameters and click <b>Load</b> to start</p>
        <div style="display:flex;justify-content:center;gap:30px;margin-top:25px;">
            <div><div style="width:40px;height:40px;background:#bbf7d0;border-radius:6px;margin:auto;"></div><p style="color:#228B22;font-weight:600;margin-top:5px;">Leading</p></div>
            <div><div style="width:40px;height:40px;background:#e9d5ff;border-radius:6px;margin:auto;"></div><p style="color:#7c3aed;font-weight:600;margin-top:5px;">Improving</p></div>
            <div><div style="width:40px;height:40px;background:#fef3c7;border-radius:6px;margin:auto;"></div><p style="color:#d97706;font-weight:600;margin-top:5px;">Weakening</p></div>
            <div><div style="width:40px;height:40px;background:#fecaca;border-radius:6px;margin:auto;"></div><p style="color:#dc2626;font-weight:600;margin-top:5px;">Lagging</p></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(f"<div style='text-align:center;color:{theme['text_muted']};font-size:10px;margin-top:10px;'>RRG Dashboard | Theme: {theme['name']} | <a href='https://www.optuma.com/blog/scripting-for-rrgs' style='color:{theme['accent']}'>Optuma Reference</a></div>", unsafe_allow_html=True)




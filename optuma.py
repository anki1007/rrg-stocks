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

# ============================================================================
# THEME CONFIGURATION
# ============================================================================
THEMES = {
    "light": {
        "bg": "#f5f5f5",
        "bg_card": "#ffffff",
        "bg_secondary": "#f8f9fa",
        "border": "#dee2e6",
        "text": "#212529",
        "text_secondary": "#495057",
        "text_muted": "#6c757d",
        "plot_bg": "#fafafa",
        "paper_bg": "#f5f5f5",
        "grid": "rgba(100,100,100,0.15)",
        "table_header": "#e9ecef",
        "table_row": "#ffffff",
        "table_row_alt": "#f8f9fa",
        "table_border": "#dee2e6",
    },
    "dark": {
        "bg": "#0b0e13",
        "bg_card": "#10141b",
        "bg_secondary": "#1a1f2e",
        "border": "#1f2732",
        "text": "#e6eaee",
        "text_secondary": "#b3bdc7",
        "text_muted": "#8892a0",
        "plot_bg": "#10141b",
        "paper_bg": "#0b0e13",
        "grid": "rgba(150,150,150,0.2)",
        "table_header": "#1a2230",
        "table_row": "#0d1117",
        "table_row_alt": "#121823",
        "table_border": "#1f2732",
    }
}

# Initialize theme in session state
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# Get current theme
theme = THEMES[st.session_state.theme]

# Dynamic CSS based on theme
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, .stApp {{
  background: {theme['bg']} !important;
  font-family: 'Inter', system-ui, sans-serif !important;
}}

.block-container {{
  padding-top: 1rem;
  max-width: 100% !important;
  padding-left: 1rem !important;
  padding-right: 1rem !important;
}}

/* All text elements */
p, span, label, div {{
  color: {theme['text']} !important;
}}

/* Selectbox and inputs */
div[data-baseweb="select"] {{
  background: {theme['bg_card']} !important;
}}

div[data-baseweb="select"] > div {{
  background: {theme['bg_card']} !important;
  border-color: {theme['border']} !important;
}}

.stSelectbox label, .stNumberInput label {{
  color: {theme['text_secondary']} !important;
}}

/* Buttons */
.stButton > button {{
  background: {theme['bg_card']} !important;
  border: 1px solid {theme['border']} !important;
  color: {theme['text']} !important;
}}

.stButton > button:hover {{
  border-color: #2196F3 !important;
  color: #2196F3 !important;
}}

/* Metrics */
[data-testid="stMetric"] {{
  background: {theme['bg_card']};
  padding: 12px;
  border-radius: 8px;
  border: 1px solid {theme['border']};
}}

[data-testid="stMetricLabel"] {{
  color: {theme['text_secondary']} !important;
}}

[data-testid="stMetricValue"] {{
  color: {theme['text']} !important;
}}

/* Expanders */
.streamlit-expanderHeader {{
  background: {theme['bg_secondary']} !important;
  color: {theme['text']} !important;
  border-radius: 8px !important;
}}

/* Checkbox */
.stCheckbox label {{
  color: {theme['text']} !important;
}}

/* Slider */
.stSlider label {{
  color: {theme['text']} !important;
}}

/* Radio */
.stRadio label {{
  color: {theme['text']} !important;
}}

/* Info box */
.stAlert {{
  background: {theme['bg_secondary']} !important;
  border: 1px solid {theme['border']} !important;
}}

/* Success message */
div[data-testid="stNotification"] {{
  background: {theme['bg_card']} !important;
}}

/* Hide elements */
#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

/* Data table container */
.table-container {{
  background: {theme['bg_card']};
  border: 1px solid {theme['border']};
  border-radius: 8px;
  overflow: hidden;
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
    "1 Month": 21,
    "3 Months": 63,
    "6 Months": 126,
    "1 Year": 252,
    "2 Years": 504,
    "3 Years": 756,
}

QUADRANT_COLORS = {
    "Leading": "#228B22",
    "Improving": "#8B5CF6",
    "Weakening": "#D97706",
    "Lagging": "#DC2626"
}

QUADRANT_BG_COLORS = {
    "Leading": "rgba(144, 238, 144, 0.5)",
    "Improving": "rgba(221, 214, 254, 0.5)",
    "Weakening": "rgba(254, 215, 170, 0.5)",
    "Lagging": "rgba(254, 178, 178, 0.5)"
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

def get_heading_direction(heading):
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

def generate_data_table_html(df, theme):
    """Generate HTML table with theme support"""
    industries = sorted(df['Industry'].unique().tolist())
    industry_options = ''.join([f'<option value="{ind}">{ind}</option>' for ind in industries])
    
    table_rows = ""
    for _, row in df.iterrows():
        color, status = get_quadrant_color(row['RS-Ratio'], row['RS-Momentum'])
        
        table_rows += f"""
        <tr>
            <td style="text-align: center;">{int(row['Sl No.'])}</td>
            <td><a href="{row['TV Link']}" target="_blank" style="color: #2196F3; font-weight: 600;">{row['Symbol']}</a></td>
            <td>{row['Name'][:25]}</td>
            <td>{row['Industry'][:20]}</td>
            <td style="text-align: right;">₹{row['Price']:,.2f}</td>
            <td style="text-align: center;"><span style="background:{color}; color:white; padding:2px 8px; border-radius:4px; font-size:11px;">{status}</span></td>
            <td style="text-align: right;">{row['RS-Ratio']:.2f}</td>
            <td style="text-align: right;">{row['RS-Momentum']:.2f}</td>
            <td style="text-align: right; font-weight:600;">{row['RRG Power']:.2f}</td>
            <td style="text-align: right;">{row['Distance']:.2f}</td>
            <td style="text-align: center; color: #fbbf24;">{row['Direction']}</td>
        </tr>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Inter', system-ui, sans-serif;
            background: {theme['bg_card']};
            color: {theme['text']};
        }}
        
        .controls {{
            padding: 12px 16px;
            background: {theme['bg_secondary']};
            border-bottom: 1px solid {theme['table_border']};
            display: flex;
            gap: 12px;
            align-items: center;
            flex-wrap: wrap;
        }}
        
        .search-box, .filter-select {{
            padding: 8px 12px;
            background: {theme['bg_card']};
            border: 1px solid {theme['border']};
            border-radius: 6px;
            color: {theme['text']};
            font-size: 13px;
            font-family: inherit;
        }}
        
        .search-box {{ min-width: 200px; }}
        .filter-select {{ min-width: 140px; }}
        
        .search-box:focus, .filter-select:focus {{
            outline: none;
            border-color: #2196F3;
        }}
        
        .count-badge {{
            background: #2196F3;
            color: white;
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 600;
        }}
        
        .table-wrapper {{
            max-height: 500px;
            overflow: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        
        th {{
            position: sticky;
            top: 0;
            background: {theme['table_header']};
            color: {theme['text_secondary']};
            padding: 12px 10px;
            text-align: left;
            font-weight: 700;
            border-bottom: 2px solid {theme['table_border']};
            cursor: pointer;
            white-space: nowrap;
        }}
        
        th:hover {{ background: {theme['bg_secondary']}; }}
        
        td {{
            padding: 10px;
            border-bottom: 1px solid {theme['table_border']};
            color: {theme['text']};
        }}
        
        tr:nth-child(odd) {{ background: {theme['table_row']}; }}
        tr:nth-child(even) {{ background: {theme['table_row_alt']}; }}
        tr:hover {{ background: {theme['bg_secondary']}; }}
        
        .hidden {{ display: none; }}
        
        a {{ text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
    </head>
    <body>
    <div class="controls">
        <input type="text" class="search-box" id="searchBox" placeholder="🔍 Search symbol or name..." onkeyup="filterTable()">
        <select class="filter-select" id="statusFilter" onchange="filterTable()">
            <option value="">All Status</option>
            <option value="Leading">Leading</option>
            <option value="Improving">Improving</option>
            <option value="Weakening">Weakening</option>
            <option value="Lagging">Lagging</option>
        </select>
        <select class="filter-select" id="industryFilter" onchange="filterTable()">
            <option value="">All Industries</option>
            {industry_options}
        </select>
        <span class="count-badge" id="countBadge">{len(df)} / {len(df)}</span>
    </div>
    
    <div class="table-wrapper">
        <table>
            <thead>
                <tr>
                    <th onclick="sortTable(0)">#</th>
                    <th onclick="sortTable(1)">Symbol</th>
                    <th onclick="sortTable(2)">Name</th>
                    <th onclick="sortTable(3)">Industry</th>
                    <th onclick="sortTable(4)">Price</th>
                    <th onclick="sortTable(5)">Status</th>
                    <th onclick="sortTable(6)">RS-Ratio</th>
                    <th onclick="sortTable(7)">RS-Mom</th>
                    <th onclick="sortTable(8)">Power</th>
                    <th onclick="sortTable(9)">Distance</th>
                    <th onclick="sortTable(10)">Direction</th>
                </tr>
            </thead>
            <tbody id="tableBody">
                {table_rows}
            </tbody>
        </table>
    </div>
    
    <script>
        const totalRows = {len(df)};
        const sortDirection = {{}};
        
        function sortTable(columnIndex) {{
            const tbody = document.getElementById('tableBody');
            const rows = Array.from(tbody.getElementsByTagName('tr'));
            
            sortDirection[columnIndex] = !sortDirection[columnIndex];
            const ascending = sortDirection[columnIndex];
            
            rows.sort((a, b) => {{
                let aValue = a.cells[columnIndex].textContent.trim();
                let bValue = b.cells[columnIndex].textContent.trim();
                
                aValue = aValue.replace(/[₹,%]/g, '');
                bValue = bValue.replace(/[₹,%]/g, '');
                
                const aNum = parseFloat(aValue);
                const bNum = parseFloat(bValue);
                
                if (!isNaN(aNum) && !isNaN(bNum)) {{
                    return ascending ? aNum - bNum : bNum - aNum;
                }}
                return ascending ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
            }});
            
            tbody.innerHTML = '';
            rows.forEach(row => tbody.appendChild(row));
        }}
        
        function filterTable() {{
            const searchBox = document.getElementById("searchBox").value.toLowerCase();
            const statusFilter = document.getElementById("statusFilter").value;
            const industryFilter = document.getElementById("industryFilter").value;
            const tbody = document.getElementById("tableBody");
            const rows = tbody.getElementsByTagName("tr");
            let visibleCount = 0;
            
            for (let row of rows) {{
                const symbol = row.cells[1].textContent.toLowerCase();
                const name = row.cells[2].textContent.toLowerCase();
                const industry = row.cells[3].textContent;
                const status = row.cells[5].textContent.trim();
                
                const matchesSearch = symbol.includes(searchBox) || name.includes(searchBox);
                const matchesStatus = !statusFilter || status === statusFilter;
                const matchesIndustry = !industryFilter || industry === industryFilter;
                
                if (matchesSearch && matchesStatus && matchesIndustry) {{
                    row.classList.remove('hidden');
                    visibleCount++;
                }} else {{
                    row.classList.add('hidden');
                }}
            }}
            
            document.getElementById('countBadge').textContent = visibleCount + ' / ' + totalRows;
        }}
    </script>
    </body>
    </html>
    """
    return html

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

# ============================================================================
# CONTROL BAR
# ============================================================================
csv_files = list_csv_from_github()
if not csv_files:
    csv_files = ["NIFTY200"]

default_csv_idx = 0
for i, csv in enumerate(csv_files):
    if 'NIFTY200' in csv.upper():
        default_csv_idx = i
        break

# Row 1: Main controls
ctrl_cols = st.columns([1.5, 1.2, 1.2, 1.2, 1.2, 1.5, 2])

with ctrl_cols[0]:
    st.markdown(f'<p style="font-size:11px; font-weight:600; color:{theme["text_muted"]}; margin-bottom:2px;">BENCHMARK</p>', unsafe_allow_html=True)
    csv_selected = st.selectbox("", csv_files, index=default_csv_idx, key="csv_sel", label_visibility="collapsed")

with ctrl_cols[1]:
    st.markdown(f'<p style="font-size:11px; font-weight:600; color:{theme["text_muted"]}; margin-bottom:2px;">VS INDEX</p>', unsafe_allow_html=True)
    bench_name = st.selectbox("", list(BENCHMARKS.keys()), index=2, key="bench_sel", label_visibility="collapsed")

with ctrl_cols[2]:
    st.markdown(f'<p style="font-size:11px; font-weight:600; color:{theme["text_muted"]}; margin-bottom:2px;">TIMEFRAME</p>', unsafe_allow_html=True)
    tf_name = st.selectbox("", list(TIMEFRAMES.keys()), index=5, key="tf_sel", label_visibility="collapsed")

with ctrl_cols[3]:
    st.markdown(f'<p style="font-size:11px; font-weight:600; color:{theme["text_muted"]}; margin-bottom:2px;">DATE RANGE</p>', unsafe_allow_html=True)
    date_range = st.selectbox("", list(DATE_RANGES.keys()), index=1, key="date_sel", label_visibility="collapsed")

with ctrl_cols[4]:
    st.markdown(f'<p style="font-size:11px; font-weight:600; color:{theme["text_muted"]}; margin-bottom:2px;">COUNTS</p>', unsafe_allow_html=True)
    trail_length = st.number_input("", min_value=1, max_value=14, value=5, key="trail_input", label_visibility="collapsed")

with ctrl_cols[5]:
    st.markdown(f'<p style="font-size:11px; font-weight:600; color:{theme["text_muted"]}; margin-bottom:2px;">&nbsp;</p>', unsafe_allow_html=True)
    st.markdown(f'<div style="background:#1565c0; border-radius:4px; padding:8px 12px; text-align:center; font-weight:600; color:white; font-size:13px;">{trail_length} Days</div>', unsafe_allow_html=True)

with ctrl_cols[6]:
    st.markdown(f'<p style="font-size:11px; font-weight:600; color:{theme["text_muted"]}; margin-bottom:2px;">&nbsp;</p>', unsafe_allow_html=True)
    btn_cols = st.columns([1, 1, 1, 1, 1])
    with btn_cols[0]:
        load_btn = st.button("📥 Load", key="load_btn", use_container_width=True)
    with btn_cols[1]:
        play_btn = st.button("▶ Play", key="play_btn", use_container_width=True)
    with btn_cols[2]:
        stop_btn = st.button("⏹ Stop", key="stop_btn", use_container_width=True)
    with btn_cols[3]:
        label_toggle = st.checkbox("Label", value=True, key="label_chk")
    with btn_cols[4]:
        # Theme toggle
        theme_btn = st.button("🌓", key="theme_btn", help="Toggle Dark/Light Theme")

if load_btn:
    st.session_state.load_clicked = True
    st.session_state.current_frame = trail_length - 1

if theme_btn:
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
    st.rerun()

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
                direction = get_heading_direction(head_current)
                
                rows.append({
                    'Symbol': format_symbol(s),
                    'Name': names_dict.get(s, s),
                    'Industry': industries_dict.get(s, 'N/A'),
                    'Price': round(current_price, 2),
                    'RS-Ratio': round(rsr_current, 2),
                    'RS-Momentum': round(rsm_current, 2),
                    'RRG Power': round(power, 2),
                    'Distance': round(dist_current, 2),
                    'Heading': round(head_current, 1),
                    'Direction': direction,
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
# DISPLAY
# ============================================================================
if st.session_state.df_cache is not None:
    df = st.session_state.df_cache
    rs_history = st.session_state.rs_history_cache
    
    # Timeline controls
    if rs_history:
        max_history_len = max([len(rs_history[sym]['rs_ratio']) for sym in rs_history], default=1)
        sample_sym = list(rs_history.keys())[0] if rs_history else None
        dates_available = rs_history[sample_sym].get('dates', []) if sample_sym else []
        
        st.markdown("---")
        
        timeline_cols = st.columns([1, 8, 1])
        with timeline_cols[0]:
            view_mode = st.radio("", ["Static", "Animate"], horizontal=True, key="view_mode", label_visibility="collapsed")
        
        with timeline_cols[1]:
            if view_mode == "Static":
                frame_idx = st.slider(
                    "Timeline",
                    min_value=trail_length - 1,
                    max_value=max_history_len - 1,
                    value=max_history_len - 1,
                    key="timeline_slider",
                    label_visibility="collapsed"
                )
            else:
                frame_idx = st.slider(
                    "Timeline",
                    min_value=trail_length - 1,
                    max_value=max_history_len - 1,
                    value=st.session_state.current_frame,
                    key="anim_slider",
                    label_visibility="collapsed"
                )
                st.session_state.current_frame = frame_idx
        
        with timeline_cols[2]:
            if dates_available and len(dates_available) > frame_idx:
                st.markdown(f'<div style="background:#1565c0; padding:8px 12px; border-radius:4px; text-align:center; font-weight:600; color:white; font-size:12px; margin-top:5px;">{dates_available[frame_idx]}</div>', unsafe_allow_html=True)
        
        # Auto-play for animation mode
        if view_mode == "Animate" and play_btn:
            import time
            if frame_idx < max_history_len - 1:
                time.sleep(0.3)
                st.session_state.current_frame = frame_idx + 1
                st.rerun()
    else:
        frame_idx = trail_length - 1
        max_history_len = trail_length
    
    # Select stocks for graph
    df_graph = select_graph_stocks(df, min_stocks=50)
    
    # Calculate range
    x_min = df['RS-Ratio'].min() - 2
    x_max = df['RS-Ratio'].max() + 2
    y_min = df['RS-Momentum'].min() - 2
    y_max = df['RS-Momentum'].max() + 2
    x_range = max(abs(100 - x_min), abs(x_max - 100))
    y_range = max(abs(100 - y_min), abs(y_max - 100))
    
    # Create figure
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
    
    # Plot stocks
    for _, row in df_graph.iterrows():
        sym = row['Symbol']
        
        if sym not in rs_history:
            continue
        
        hist = rs_history[sym]
        end_idx = min(frame_idx + 1, len(hist['rs_ratio']))
        start_idx = max(0, end_idx - trail_length)
        
        x_pts = np.array(hist['rs_ratio'][start_idx:end_idx], dtype=float)
        y_pts = np.array(hist['rs_momentum'][start_idx:end_idx], dtype=float)
        n_pts = len(x_pts)
        
        if n_pts == 0:
            continue
        
        head_x, head_y = x_pts[-1], y_pts[-1]
        color, status = get_quadrant_color(head_x, head_y)
        
        # Trail
        if n_pts >= 2:
            if n_pts >= 3:
                x_smooth, y_smooth = smooth_spline_curve(x_pts, y_pts, points_per_segment=6)
            else:
                x_smooth, y_smooth = x_pts, y_pts
            
            for i in range(len(x_smooth) - 1):
                prog = i / max(1, len(x_smooth) - 2)
                fig.add_trace(go.Scatter(
                    x=[x_smooth[i], x_smooth[i+1]],
                    y=[y_smooth[i], y_smooth[i+1]],
                    mode='lines',
                    line=dict(color=color, width=2 + prog * 3),
                    opacity=0.3 + prog * 0.7,
                    hoverinfo='skip',
                    showlegend=False,
                ))
            
            # Arrow
            dx, dy = x_pts[-1] - x_pts[-2], y_pts[-1] - y_pts[-2]
            length = np.sqrt(dx**2 + dy**2)
            if length > 0.01:
                fig.add_annotation(
                    x=x_pts[-1], y=y_pts[-1],
                    ax=x_pts[-1] - dx/length * 0.35,
                    ay=y_pts[-1] - dy/length * 0.35,
                    xref='x', yref='y', axref='x', ayref='y',
                    showarrow=True, arrowhead=2, arrowsize=1.5, arrowwidth=2.5, arrowcolor=color,
                )
        
        # Head
        fig.add_trace(go.Scatter(
            x=[head_x], y=[head_y],
            mode='markers',
            marker=dict(size=12, color=color, line=dict(color='white', width=2)),
            hovertemplate=f"<b>{sym}</b><br>RS: {head_x:.2f}<br>Mom: {head_y:.2f}<extra></extra>",
            showlegend=False,
        ))
        
        # Label
        if label_toggle:
            fig.add_annotation(
                x=head_x, y=head_y,
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
            marker=dict(size=12, color=QUADRANT_COLORS[status]),
            name=status,
            showlegend=True
        ))
    
    # Layout
    fig.update_layout(
        height=550,
        plot_bgcolor=theme['plot_bg'],
        paper_bgcolor=theme['paper_bg'],
        font=dict(color=theme['text'], size=12, family='Inter, sans-serif'),
        xaxis=dict(
            title=dict(text="<b>JdK RS-RATIO</b>", font=dict(size=12, color=theme['text_secondary'])),
            range=[100-x_range-1, 100+x_range+1],
            gridcolor=theme['grid'],
            zeroline=False,
            tickfont=dict(color=theme['text_muted'])
        ),
        yaxis=dict(
            title=dict(text="<b>JdK RS-MOMENTUM</b>", font=dict(size=12, color=theme['text_secondary'])),
            range=[100-y_range-1, 100+y_range+1],
            gridcolor=theme['grid'],
            zeroline=False,
            tickfont=dict(color=theme['text_muted'])
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.08,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, color=theme['text'])
        ),
        hovermode='closest',
        margin=dict(l=60, r=30, t=20, b=80),
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True, 'displaylogo': False})
    
    # ========================================================================
    # QUADRANT SUMMARY (horizontal bar)
    # ========================================================================
    quad_cols = st.columns(4)
    status_counts = df['Status'].value_counts()
    
    for idx, (status, icon) in enumerate([("Leading", "🟢"), ("Improving", "🟣"), ("Weakening", "🟡"), ("Lagging", "🔴")]):
        with quad_cols[idx]:
            count = status_counts.get(status, 0)
            df_status = df[df['Status'] == status].head(10)
            color = QUADRANT_COLORS[status]
            
            with st.expander(f"{icon} {status} ({count})", expanded=(status == "Leading")):
                for _, row in df_status.iterrows():
                    st.markdown(f"""
                    <div style="padding:4px 8px; margin:2px 0; background:{theme['bg_secondary']}; border-left:3px solid {color}; border-radius:4px;">
                        <a href="{row['TV Link']}" target="_blank" style="font-weight:600; color:{color}; text-decoration:none;">{row['Symbol']}</a>
                        <span style="font-size:10px; color:{theme['text_muted']};"> RS: {row['RS-Ratio']:.1f} | Mom: {row['RS-Momentum']:.1f}</span>
                    </div>
                    """, unsafe_allow_html=True)
    
    # ========================================================================
    # DATA TABLE
    # ========================================================================
    st.markdown("---")
    st.markdown(f"### 📊 Detailed Analysis ({len(df)} stocks)")
    
    # Generate and display table
    table_html = generate_data_table_html(df, theme)
    st.components.v1.html(table_html, height=600, scrolling=False)
    
    # Export
    st.markdown("---")
    exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 4])
    with exp_col1:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_buffer.getvalue(),
            file_name=f"RRG_{csv_selected}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    with exp_col2:
        theme_label = '🌙 Dark' if st.session_state.theme == 'dark' else '☀️ Light'
        st.markdown(f"<small style='color:{theme['text_muted']}'>Theme: {theme_label}</small>", unsafe_allow_html=True)

else:
    # Welcome screen
    st.markdown(f"""
    <div style="text-align:center; padding:60px 20px; background:{theme['bg_card']}; border-radius:12px; border:1px solid {theme['border']}; margin-top:40px;">
        <h2 style="color:{theme['text']}; margin-bottom:20px;">📈 Relative Rotation Graph Dashboard</h2>
        <p style="color:{theme['text_secondary']}; font-size:16px; margin-bottom:30px;">
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
st.markdown(f"""
<div style="text-align:center; color:{theme['text_muted']}; font-size:11px;">
    RRG Analysis | Data: Yahoo Finance | Reference: <a href="https://www.optuma.com/blog/scripting-for-rrgs" style="color:#2196F3;">Optuma RRG Guide</a>
</div>
""", unsafe_allow_html=True)

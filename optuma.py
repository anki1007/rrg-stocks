import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import requests
from datetime import datetime
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
# SESSION STATE INITIALIZATION
# ============================================================================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.df_top = None

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

def load_data(csv_selected, bench_name, tf_name, period_name, rank_by, top_n):
    """Load and process all data based on current selections"""
    interval, yf_period = TIMEFRAMES[tf_name]
    universe = load_universe(csv_selected)

    if universe.empty:
        st.error("Failed to load universe data")
        return None, None

    symbols = universe['Symbol'].tolist()
    names_dict = dict(zip(universe['Symbol'], universe['Company Name']))
    industries_dict = dict(zip(universe['Symbol'], universe['Industry']))

    with st.spinner(f"📥 Downloading data for {len(symbols)} symbols from {tf_name}..."):
        try:
            raw = yf.download(
                symbols + [BENCHMARKS[bench_name]],
                interval=interval,
                period=yf_period,
                auto_adjust=True,
                progress=False,
                threads=True
            )
        except Exception as e:
            st.error(f"Error downloading data: {e}")
            return None, None

    if BENCHMARKS[bench_name] not in raw['Close'].columns:
        st.error(f"Benchmark {bench_name} ({BENCHMARKS[bench_name]}) data not available")
        return None, None

    bench = raw['Close'][BENCHMARKS[bench_name]]

    rows = []
    success_count = 0
    failed_count = 0

    for s in symbols:
        if s not in raw['Close'].columns:
            failed_count += 1
            continue

        try:
            rr, mm = rrg_calc(raw['Close'][s], bench)
            if rr is None or mm is None:
                failed_count += 1
                continue

            rr_tail = rr.iloc[-TAIL:]
            mm_tail = mm.iloc[-TAIL:]

            if len(rr_tail) < 3:
                failed_count += 1
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
            success_count += 1
        except Exception:
            failed_count += 1
            continue

    if success_count > 0:
        st.success(f"✅ Loaded {success_count} symbols | ⚠️ Skipped {failed_count} unavailable")
    else:
        st.error("No data available. Try another combination.")
        return None, None

    df = pd.DataFrame(rows)

    if df.empty:
        st.error("No data available after processing.")
        return None, None

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

    return df, df_top

# ============================================================================
# SIDEBAR - CONTROLS
# ============================================================================

st.sidebar.markdown("### ⚙️ Controls")

csv_files = list_csv_from_github()
default_index = 0
if 'NIFTY200' in csv_files:
    default_index = csv_files.index('NIFTY200')

csv_selected = st.sidebar.selectbox("Indices", csv_files, index=default_index)

bench_options = list(BENCHMARKS.keys())
default_bench = bench_options.index("NIFTY 500") if "NIFTY 500" in bench_options else 2
bench_name = st.sidebar.selectbox("Benchmark", bench_options, index=default_bench)

tf_name = st.sidebar.selectbox("Strength vs Timeframe", list(TIMEFRAMES.keys()))
period_name = st.sidebar.selectbox("Period", list(PERIOD_MAP.keys()), index=0)
rank_by = st.sidebar.selectbox(
    "Rank by",
    ["RRG Power", "RS-Ratio", "RS-Momentum", "Price % Change", "Momentum Slope"],
    index=0
)

st.sidebar.markdown("---")

top_n = st.sidebar.slider("Show Top N", min_value=5, max_value=50, value=15)

st.sidebar.markdown("---")

st.sidebar.markdown("### 🔄 Load Data")
load_button = st.sidebar.button("📊 LOAD DATA", use_container_width=True, type="primary")

if load_button:
    st.session_state.data_loaded = True
    df, df_top = load_data(csv_selected, bench_name, tf_name, period_name, rank_by, top_n)
    st.session_state.df = df
    st.session_state.df_top = df_top

st.sidebar.markdown("---")

export_csv = st.sidebar.checkbox("Export CSV", value=True)

st.sidebar.markdown("---")

st.sidebar.markdown("### 📍 Legend")
st.sidebar.markdown("🟢 **Leading**: Strong RS, ↑ Momentum")
st.sidebar.markdown("🔵 **Improving**: Weak RS, ↑ Momentum")
st.sidebar.markdown("🟡 **Weakening**: Weak RS, ↓ Momentum")
st.sidebar.markdown("🔴 **Lagging**: Strong RS, ↓ Momentum")

# ============================================================================
# MAIN CONTENT
# ============================================================================

if st.session_state.data_loaded and st.session_state.df is not None:
    df = st.session_state.df
    df_top = st.session_state.df_top

    st.markdown("## Relative Rotation Graph")
    st.markdown(
        f"**{csv_selected} | {tf_name} | {period_name} | "
        f"Benchmark: {bench_name} ({BENCHMARKS[bench_name]})**"
    )

    fig_rrg = go.Figure()

    x_min = df['RS-Ratio'].min() - 5
    x_max = df['RS-Ratio'].max() + 5
    y_min = df['RS-Momentum'].min() - 5
    y_max = df['RS-Momentum'].max() + 5

    quadrant_size = max(
        abs(100 - x_min),
        abs(x_max - 100),
        abs(100 - y_min),
        abs(y_max - 100)
    )

    fig_rrg.add_shape(
        type="rect", x0=100, y0=100, x1=100 + quadrant_size, y1=100 + quadrant_size,
        fillcolor="rgba(34, 197, 94, 0.1)",
        line=dict(color="rgba(34, 197, 94, 0.3)", width=2), layer="below"
    )
    fig_rrg.add_shape(
        type="rect", x0=100 - quadrant_size, y0=100, x1=100, y1=100 + quadrant_size,
        fillcolor="rgba(59, 130, 246, 0.1)",
        line=dict(color="rgba(59, 130, 246, 0.3)", width=2), layer="below"
    )
    fig_rrg.add_shape(
        type="rect", x0=100 - quadrant_size, y0=100 - quadrant_size, x1=100, y1=100,
        fillcolor="rgba(251, 191, 36, 0.1)",
        line=dict(color="rgba(251, 191, 36, 0.3)", width=2), layer="below"
    )
    fig_rrg.add_shape(
        type="rect", x0=100, y0=100 - quadrant_size, x1=100 + quadrant_size, y1=100,
        fillcolor="rgba(239, 68, 68, 0.1)",
        line=dict(color="rgba(239, 68, 68, 0.3)", width=2), layer="below"
    )

    fig_rrg.add_annotation(
        x=100 + quadrant_size * 0.5, y=100 + quadrant_size * 0.5, text="Leading",
        showarrow=False, font=dict(size=12, color="rgba(34, 197, 94, 0.5)")
    )
    fig_rrg.add_annotation(
        x=100 - quadrant_size * 0.5, y=100 + quadrant_size * 0.5, text="Improving",
        showarrow=False, font=dict(size=12, color="rgba(59, 130, 246, 0.5)")
    )
    fig_rrg.add_annotation(
        x=100 - quadrant_size * 0.5, y=100 - quadrant_size * 0.5, text="Weakening",
        showarrow=False, font=dict(size=12, color="rgba(251, 191, 36, 0.5)")
    )
    fig_rrg.add_annotation(
        x=100 + quadrant_size * 0.5, y=100 - quadrant_size * 0.5, text="Lagging",
        showarrow=False, font=dict(size=12, color="rgba(239, 68, 68, 0.5)")
    )

    fig_rrg.add_hline(y=100, line_dash="dash",
                      line_color="rgba(150, 150, 150, 0.5)", layer="below")
    fig_rrg.add_vline(x=100, line_dash="dash",
                      line_color="rgba(150, 150, 150, 0.5)", layer="below")

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
                    f"Price: ₹{row['Price']:.2f}<br>"
                    f"Change %: {row['Change %']:+.2f}%<br>"
                    f"RS-Ratio: {row['RS-Ratio']:.2f}<br>"
                    f"RS-Momentum: {row['RS-Momentum']:.2f}<br>"
                    f"RRG Power: {row['RRG Power']:.2f}<br>"
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
        height=700,
        xaxis_title="RS-Ratio",
        yaxis_title="RS-Momentum",
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
        'Sl No.', 'Symbol', 'Industry', 'Price', 'Change %',
        'RRG Power', 'Status', 'RS-Ratio', 'RS-Momentum'
    ]].copy()

    display_data = []
    for _, row in df_display.iterrows():
        tv_link = row['TV Link']
        display_data.append({
            'Sl No.': int(row['Sl No.']),
            'Symbol': f"[{row['Symbol']}]({tv_link})",
            'Industry': row['Industry'],
            'Price': f"₹{row['Price']:.2f}",
            'Change %': f"{row['Change %']:+.2f}%",
            'Strength': f"{row['RRG Power']:.2f}",
            'Status': row['Status'],
            'RS-Ratio': f"{row['RS-Ratio']:.2f}",
            'RS-Momentum': f"{row['RS-Momentum']:.2f}"
        })

    st.dataframe(pd.DataFrame(display_data), use_container_width=True, height=600)

    st.markdown("---")

    if export_csv:
        df_export = df_top[[
            'Sl No.', 'Symbol', 'Industry', 'Price', 'Change %',
            'RRG Power', 'RS-Ratio', 'RS-Momentum', 'Status'
        ]].copy()

        csv_buffer = io.StringIO()
        df_export.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        b64_csv = base64.b64encode(csv_data.encode()).decode()

        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.markdown(
                f'<a href="data:file/csv;base64,{b64_csv}" '
                f'download="RRG_Analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv" '
                f'style="display: inline-block; padding: 12px 24px; background: #22c55e; '
                f'color: #000; border-radius: 6px; text-decoration: none; font-weight: 600; '
                f'text-align: center; width: 200px;">📥 Download CSV</a>',
                unsafe_allow_html=True
            )

else:
    st.markdown("---")
    st.info("👈 **Select all parameters in the sidebar and click the 📊 LOAD DATA button to start analysis**")
    st.markdown("---")

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #888; font-size: 11px;'>
    💡 RRG Analysis Dashboard • Data: Yahoo Finance • Charts: TradingView<br>
    <i>Disclaimer: For educational purposes only. Not financial advice.</i>
</div>
""", unsafe_allow_html=True)

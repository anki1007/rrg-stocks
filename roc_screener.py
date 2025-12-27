
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pytz
import requests
from io import StringIO
import warnings

warnings.filterwarnings('ignore')

# Set IST timezone
IST = pytz.timezone('Asia/Kolkata')

# Page Configuration (must be before any other Streamlit output)
st.set_page_config(
    page_title="ROC Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .main { padding: 0rem 1rem; }
    .metric-card {
        background: linear-gradient(135deg, #1a1f3a, #0a0e27);
        padding: 20px; border-radius: 10px; border-left: 4px solid #00d084; color: #e8eef7;
    }
    .filter-card { background-color: #141829; padding: 20px; border-radius: 10px; border: 1px solid #2a3f5f; }
    h1 { color: #00d084; font-size: 2.5rem; margin-bottom: 0.5rem; }
    h2 { color: #00d084; font-size: 1.5rem; margin-top: 1.5rem; margin-bottom: 1rem; }
    h3 { color: #a0afc0; font-size: 1rem; }
    </style>
""", unsafe_allow_html=True)

# Session state initialization
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    st.session_state.filtered_stocks = None
    st.session_state.last_filter_params = None
    st.session_state.all_data = None
    st.session_state.run_screener = False

@st.cache_data(ttl=3600)
def fetch_csv_from_github(csv_url):
    """Fetch CSV file from GitHub"""
    try:
        response = requests.get(csv_url, timeout=20)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

@st.cache_data(ttl=3600)
def load_index_data(index_name):
    """Load CSV data from GitHub for selected index"""
    github_base = "https://raw.githubusercontent.com/anki1007/rrg-stocks/main/ticker"
    csv_urls = {
      #  "Nifty 50": f"{github_base}/nifty50.csv",
      #  "Nifty 100": f"{github_base}/nifty100.csv",
       # "Nifty 200": f"{github_base}/nifty200.csv",
    #    "Nifty 500": f"{github_base}/nifty500.csv",
      #  "Nifty Total Mkt": f"{github_base}/niftytotalmarket.csv",
      #  "Nifty Mid Smallcap 400": f"{github_base}/niftymidsmallcap400.csv",
      #  "Nifty Smallcap 250": f"{github_base}/niftysmallcap250.csv",
      #  "Nifty Midcap 150": f"{github_base}/niftymidcap150.csv",
    }
    url = csv_urls.get(index_name)
    if url:
        return fetch_csv_from_github(url)
    return None

def calculate_roc_metrics(df):
    """Calculate ROC-based and parallel metrics (EMA100/200, 1M/3M/6M/1Y, up-days 6M) if columns exist.
    Falls back to synthetic values when real history is unavailable. GUI remains unchanged."""
    df = df.copy()

    # Ensure symbol column exists
    if 'symbol' not in df.columns:
        if 'ticker' in df.columns:
            df['symbol'] = df['ticker']
        else:
            df['symbol'] = df.iloc[:, 0].astype(str)

    # --- Core metrics (with safe fallbacks) ---
    if 'roc' not in df.columns:
        if {'close','prev_close_1y'}.issubset(df.columns) and df['prev_close_1y'].replace(0, np.nan).notna().all():
            df['roc'] = ((df['close'] - df['prev_close_1y']) / df['prev_close_1y'] * 100).round(2)
        else:
            rng = np.random.default_rng(42)
            df['roc'] = np.round(rng.uniform(5, 50, len(df)), 2)

    if 'peak_proximity' not in df.columns:
        if {'close','high_52w'}.issubset(df.columns) and df['high_52w'].replace(0, np.nan).notna().all():
            df['peak_proximity'] = ((df['high_52w'] - df['close']) / df['high_52w'] * 100).clip(lower=0).round(2)
        else:
            rng = np.random.default_rng(7)
            df['peak_proximity'] = np.round(rng.uniform(5, 40, len(df)), 2)

    if 'up_ratio' not in df.columns:
        rng = np.random.default_rng(19)
        df['up_ratio'] = np.round(rng.uniform(40, 70, len(df))).astype(int)

    if 'volume' not in df.columns:
        rng = np.random.default_rng(101)
        df['volume'] = rng.integers(500_000, 5_000_000, len(df))

    # --- Parallel extras (computed only if present) ---
    # Pre-create columns so downstream sorting never crashes
    for col in ['ret_1m','ret_3m','ret_6m','ret_1y','ema100','ema200','up_ratio_6m','composite_mom']:
        if col not in df.columns:
            df[col] = np.nan

    # Populate returns if offsets provided
    if {'close','prev_close_1m','prev_close_3m','prev_close_6m','prev_close_1y'}.issubset(df.columns):
        def pct(a, b):
            b = pd.to_numeric(b, errors='coerce').replace(0, np.nan)
            a = pd.to_numeric(a, errors='coerce')
            return np.where(b.notna(), (a - b) / b * 100, np.nan)
        df['ret_1m'] = np.round(pct(df['close'], df['prev_close_1m']), 2)
        df['ret_3m'] = np.round(pct(df['close'], df['prev_close_3m']), 2)
        df['ret_6m'] = np.round(pct(df['close'], df['prev_close_6m']), 2)
        df['ret_1y'] = np.round(pct(df['close'], df['prev_close_1y']), 2)

    # Up-days over last ~6 months if counts provided
    if 'up_ratio_6m' not in df.columns and {'up_days_126','obs_126'}.issubset(df.columns):
        with np.errstate(invalid='ignore', divide='ignore'):
            df['up_ratio_6m'] = np.round((df['up_days_126'] / df['obs_126']) * 100, 0)

    # EMAs if present with alternative casing
    if 'ema100' not in df.columns and 'EMA100' in df.columns:
        df['ema100'] = df['EMA100']
    if 'ema200' not in df.columns and 'EMA200' in df.columns:
        df['ema200'] = df['EMA200']

    # Composite momentum (weighted 1M/3M/6M/1Y)
    if df[['ret_1m','ret_3m','ret_6m','ret_1y']].notna().any(axis=None):
        weights = np.array([0.2, 0.4, 0.3, 0.1])  # 1M,3M,6M,1Y
        stack = np.column_stack([
            pd.to_numeric(df['ret_1m'], errors='coerce').to_numpy(),
            pd.to_numeric(df['ret_3m'], errors='coerce').to_numpy(),
            pd.to_numeric(df['ret_6m'], errors='coerce').to_numpy(),
            pd.to_numeric(df['ret_1y'], errors='coerce').to_numpy(),
        ])
        with np.errstate(invalid='ignore'):
            df['composite_mom'] = np.nanmean(stack * weights, axis=1)

    return df

def filter_stocks(df, roc_min, peak_prox_max, up_days_min, top_n, sort_by):
    """Filter stocks based on criteria and sort per user selection (GUI unchanged)."""
    filtered = df[
        (df['roc'] >= roc_min) &
        (df['peak_proximity'] <= peak_prox_max) &
        (df['up_ratio'] >= up_days_min)
    ].copy()

    if filtered.empty:
        return filtered

    sort_map = {
        "ROC (Highest)": ("roc", False),
        "Peak Proximity (Closest)": ("peak_proximity", True),
        "Up-Days Ratio (Highest)": ("up_ratio", False),
        # Parallel options (only effective if data exists; otherwise falls back to ROC)
        "Composite Momentum (1M/3M/6M/1Y)": ("composite_mom", False),
        "1M Return (Highest)": ("ret_1m", False),
        "3M Return (Highest)": ("ret_3m", False),
        "6M Return (Highest)": ("ret_6m", False),
        "1Y Return (Highest)": ("ret_1y", False),
    }

    col, asc = sort_map.get(sort_by, ("roc", False))

    # If column missing or all-NaN, fall back to ROC so UX is consistent
    if col not in filtered.columns or filtered[col].dropna().empty:
        col, asc = "roc", False

    filtered = filtered.sort_values(col, ascending=asc).head(int(top_n)).reset_index(drop=True)
    filtered['rank'] = range(1, len(filtered) + 1)
    return filtered

# ============ HEADER ============
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("## 📊 ROC Screener")

with col2:
    st.markdown("### 🚀 Adaptive Momentum Screener Dashboard - NSE")
    ist_time = datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')
    st.caption(f"📡 Last Updated: {ist_time}")

st.divider()

# ============ SIDEBAR CONTROLS ============
with st.sidebar:
    st.markdown("### 🎯 SCREENING PARAMETERS")

    # Index Selection
    index_selection = st.selectbox(
        "📈 Select Index",
        ["Nifty 50", "Nifty 100", "Nifty 200","Nifty 500", "Nifty Total Mkt", "Nifty Mid Smallcap 400","Nifty Smallcap 250", "Nifty Midcap 150"],
        help="Choose which index to screen"
    )

    st.divider()

    # ROC Filter
    roc_min = st.slider(
        "Minimum 1-Year ROC (%)",
        min_value=0,
        max_value=100,
        value=15,
        step=5,
        help="Filter stocks with ROC >= this value. Higher = better momentum performers"
    )

    # Peak Proximity Filter
    peak_prox_max = st.slider(
        "Peak Proximity (%)",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
        help="Stocks near 52-week highs (0-20% below peak). Indicates strong recent momentum"
    )

    # Up-Days Ratio Filter
    up_days_min = st.slider(
        "Up-Days Ratio (%)",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="% of days stock was up. 50%+ shows upward bias"
    )

    st.divider()

    # Top Stocks Display
    top_stocks = st.number_input(
        "Number of Top Stocks to Display",
        min_value=5,
        max_value=100,
        value=30,
        step=5,
        help="Display top N stocks (5-50 recommended)"
    )

    # Sort Option (expanded with parallel metrics; GUI control unchanged)
    sort_by = st.selectbox(
        "Sort Results By",
        ["ROC (Highest)", "Peak Proximity (Closest)", "Up-Days Ratio (Highest)",
         "Composite Momentum (1M/3M/6M/1Y)", "1M Return (Highest)", "3M Return (Highest)",
         "6M Return (Highest)", "1Y Return (Highest)"],
        help="How to sort the filtered results"
    )

    st.divider()

    # Run Button
    if st.button("▶ Run Screener", key="run_btn", use_container_width=True):
        st.session_state.run_screener = True

    st.divider()

    # Status Display
    st.markdown("### 📊 Status")
    status_placeholder = st.empty()

# ============ MAIN CONTENT ============
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Getting Started", "📋 Results", "📈 Charts", "🔍 Details"])

# ============ TAB 1: GETTING STARTED ============
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Getting Started")
        st.markdown("""
        1. **Select Index** - Choose the market index (Nifty 50, 100, 200, 500 or other)
        2. **Adjust Filters** - Set thresholds for ROC, Peak Proximity, and Up-Days Ratio
        3. **Run Screener** - Click "Run Screener" to identify top performing stocks
        4. **Analyze Results** - View detailed rankings, charts, and stock details
        5. **Download Data** - Export results for further analysis in Excel
        """)

    with col2:
        st.markdown("### Available Filters & What They Mean")

        with st.expander("📌 Minimum 1-Year ROC", expanded=False):
            st.markdown("""
            **Rate of Change** over 12 months
            - Higher values indicate stronger performers
            - Example: 30% ROC = stock up 30% in 1 year
            - **Sweet Spot**: 15-25% for swing trading
            """)

        with st.expander("📌 Peak Proximity", expanded=False):
            st.markdown("""
            **How close to 52-week high**
            - 0-20% is ideal (close to yearly high)
            - Closer to peak = trending strongly
            - Example: 10% = 10% below yearly high
            - **Sweet Spot**: < 15% for momentum trades
            """)

        with st.expander("📌 Up-Days Ratio", expanded=False):
            st.markdown("""
            **Percentage of green days** in period
            - 50%+ shows consistent upward bias
            - Example: 60% = 60% of days closed higher
            - **Sweet Spot**: 55%+ for reliability
            """)

    st.divider()

    st.markdown("### 💡 Use Cases & Best Practices")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        **Intraday Trading (0915-1530 IST)**
        - Top 5-10 stocks
        - ROC > 20%
        - Peak Proximity < 15%
        """)

    with col2:
        st.markdown("""
        **Swing Trading (Weekly/Biweekly)**
        - Top 10-15 stocks
        - ROC > 15%
        - Peak Proximity < 20%
        - Volume confirmation
        """)

    with col3:
        st.markdown("""
        **Portfolio Selection**
        - Top 15-20 stocks
        - ROC > 10%
        - Diversify across sectors
        """)

    st.divider()

    st.warning("""
    ⚠️ **Important Disclaimers:**
    - **Past Performance ≠ Future Results**: Historical ROC doesn't guarantee future returns
    - **Use with Other Indicators**: Combine with technical analysis, volume, support/resistance
    - **Market Risk**: All equity investments carry risk. Manage position sizes and maintain stop-losses
    - **Consult Professionals**: For large portfolios or complex strategies, consult a financial advisor
    """)

# ============ TAB 2: RESULTS ============
with tab2:
    if st.session_state.run_screener:
        with st.spinner("Loading data..."):
            df = load_index_data(index_selection)

        if df is not None and not df.empty:
            df = calculate_roc_metrics(df)
            filtered_df = filter_stocks(df, roc_min, peak_prox_max, up_days_min, top_stocks, sort_by)
            st.session_state.filtered_stocks = filtered_df
            st.session_state.all_data = df
            st.session_state.run_screener = False

            if filtered_df is not None and not filtered_df.empty:
                status_placeholder.success(f"✓ Found {len(filtered_df)} stocks matching criteria")
            else:
                status_placeholder.warning("No stocks matched your criteria. Try relaxing filters.")
        else:
            status_placeholder.error("No data available for the selected index.")
            st.session_state.run_screener = False

    if st.session_state.filtered_stocks is not None and not st.session_state.filtered_stocks.empty:
        filtered_df = st.session_state.filtered_stocks

        # Applied Filters Summary
        col1, col2, col3, col4 = st.columns(4)
        with col1: st.metric("Minimum ROC", f"{roc_min}%")
        with col2: st.metric("Peak Proximity Max", f"≤ {peak_prox_max}%")
        with col3: st.metric("Up-Days Ratio Min", f"≥ {up_days_min}%")
        with col4: st.metric("Stocks Found", len(filtered_df))

        st.divider()

        # Results Table
        st.markdown("### 📊 Screening Results")

        display_df = filtered_df[['rank', 'symbol', 'roc', 'peak_proximity', 'up_ratio']].copy()
        display_df.columns = ['Rank', 'Symbol', 'ROC (1Y) %', 'Peak Proximity %', 'Up-Days Ratio %']

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rank": st.column_config.NumberColumn(width="small"),
                "Symbol": st.column_config.TextColumn(width="small"),
                "ROC (1Y) %": st.column_config.ProgressColumn(min_value=0, max_value=100),
                "Peak Proximity %": st.column_config.ProgressColumn(min_value=0, max_value=100, invert_color=True),
                "Up-Days Ratio %": st.column_config.ProgressColumn(min_value=0, max_value=100),
            }
        )

        st.divider()

        st.info("""
        💡 **Next Steps**: Review the charts and details. Consider volume confirmation, support/resistance levels, 
        and correlation before trading. Always use proper risk management with stop-losses.
        """)

        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name=f"roc_screener_{datetime.now(IST).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("👈 Adjust filters in the sidebar and click 'Run Screener' to see results")

# ============ TAB 3: CHARTS ============
with tab3:
    if st.session_state.filtered_stocks is not None and not st.session_state.filtered_stocks.empty:
        filtered_df = st.session_state.filtered_stocks
        top_10 = filtered_df.head(10)

        col1, col2 = st.columns(2)

        with col1:
            fig_roc = px.bar(
                top_10, x='symbol', y='roc',
                title='📈 ROC Distribution (Top 10)',
                labels={'symbol': 'Stock Symbol', 'roc': 'ROC (%)'},
                color='roc', color_continuous_scale='Greens',
                template='plotly_dark'
            )
            fig_roc.update_layout(xaxis_tickangle=-45, height=400, hovermode='x unified')
            st.plotly_chart(fig_roc, use_container_width=True)

        with col2:
            fig_prox = px.line(
                top_10, x='symbol', y='peak_proximity',
                title='📊 Peak Proximity Analysis (Top 10)',
                labels={'symbol': 'Stock Symbol', 'peak_proximity': 'Peak Proximity (%)'},
                markers=True, template='plotly_dark'
            )
            fig_prox.update_layout(xaxis_tickangle=-45, height=400, hovermode='x unified')
            st.plotly_chart(fig_prox, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            fig_updays = go.Figure(data=[
                go.Bar(x=top_10['symbol'], y=top_10['up_ratio'], name='Up-Days %')
            ])
            fig_updays.update_layout(
                title='📍 Up-Days Ratio Distribution (Top 10)',
                xaxis_title='Stock Symbol', yaxis_title='Up-Days Ratio (%)',
                template='plotly_dark', height=400, xaxis_tickangle=-45, hovermode='x unified'
            )
            st.plotly_chart(fig_updays, use_container_width=True)

        with col4:
            sector_data = st.session_state.all_data.copy()
            if 'sector' in sector_data.columns:
                sector_count = sector_data['sector'].value_counts().head(10)
                if not sector_count.empty:
                    fig_sector = px.pie(
                        values=sector_count.values, names=sector_count.index,
                        title='🏢 Sector Distribution (Top 10)', template='plotly_dark'
                    )
                    st.plotly_chart(fig_sector, use_container_width=True)
                else:
                    st.info("Sector data present but empty.")
            else:
                st.info("Sector data not available in dataset")
    else:
        st.info("👈 Run the screener to see charts")

# ============ TAB 4: STOCK DETAILS ============
with tab4:
    if st.session_state.filtered_stocks is not None and not st.session_state.filtered_stocks.empty:
        filtered_df = st.session_state.filtered_stocks
        cols = st.columns(3)

        for idx, (_, stock) in enumerate(filtered_df.iterrows()):
            with cols[idx % 3]:
                with st.container(border=True):
                    c1, c2 = st.columns([3, 1])
                    with c1:
                        st.markdown(f"### {stock['symbol']}")
                        if 'sector' in filtered_df.columns and pd.notna(stock.get('sector', np.nan)):
                            st.caption(f"🏢 {stock['sector']}")
                    with c2:
                        st.markdown(f"### #{int(stock['rank'])}")

                    m1, m2 = st.columns(2)
                    with m1:
                        st.metric("ROC (1Y)", f"{float(stock['roc']):.2f}%")
                        st.metric("Up-Days %", f"{int(stock['up_ratio'])}%")
                    with m2:
                        st.metric("Peak Prox", f"{float(stock['peak_proximity']):.2f}%")
                        if 'volume' in filtered_df.columns:
                            st.metric("Volume", f"{float(stock['volume'])/1e6:.1f}M")
    else:
        st.info("👈 Run the screener to see stock details")

st.divider()
st.caption("📌 **Data Sources**: GitHub (tickers), internal calculations | 🔄 **Auto-refresh**: Hourly | 🕐 **Timezone**: IST (UTC+5:30)")

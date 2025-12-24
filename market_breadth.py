import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import pytz
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Market Breadth Terminal",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bloomberg Terminal-inspired themes
THEMES = {
    "Dark": {
        "bg_color": "#161614",
        "text_color": "#FFD700",
        "grid_color": "#2a2a2a",
        "ema_colors": {"20": "#FF6B9D", "50": "#4ECDC4", "100": "#95E1D3", "200": "#FF6348"},
        "up_color": "#00D084",
        "down_color": "#FF5E78",
        "neutral_color": "#FFD700"
    },
    "Terminal Green": {
        "bg_color": "#0B0E11",
        "text_color": "#00FF41",
        "grid_color": "#1a1a1a",
        "ema_colors": {"20": "#39FF14", "50": "#00FF41", "100": "#0FFF50", "200": "#3FFF00"},
        "up_color": "#39FF14",
        "down_color": "#FF0000",
        "neutral_color": "#00FF41"
    },
    "Neon Purple": {
        "bg_color": "#0a0015",
        "text_color": "#00D9FF",
        "grid_color": "#1a0033",
        "ema_colors": {"20": "#FF10F0", "50": "#00D9FF", "100": "#B537F2", "200": "#FF006E"},
        "up_color": "#00FF88",
        "down_color": "#FF0055",
        "neutral_color": "#00D9FF"
    },
    "Professional": {
        "bg_color": "#FFFFFF",
        "text_color": "#1F2937",
        "grid_color": "#E5E7EB",
        "ema_colors": {"20": "#EF4444", "50": "#3B82F6", "100": "#10B981", "200": "#F59E0B"},
        "up_color": "#10B981",
        "down_color": "#EF4444",
        "neutral_color": "#6B7280"
    }
}

# Index configuration - Maps to CSV files in same directory
INDEX_CONFIG = {
    "Nifty 50": {
        "csv_name": "ticker/nifty50.csv",
        "description": "Top 50 Large Cap Stocks",
        "count": 50
    },
    "Nifty 100": {
        "csv_name": "ticker/nifty100.csv",
        "description": "Top 100 Large Cap Stocks",
        "count": 100
    },
    "Nifty 200": {
        "csv_name": "ticker/nifty200.csv",
        "description": "Top 200 Large Cap and Mid Cap Stocks",
        "count": 200
    },
    "Nifty 500": {
        "csv_name": "ticker/nifty500.csv",
        "description": "Nifty 500 - Broad Market Index",
        "count": 500
    },
    "Nifty Total Market": {
        "csv_name": "ticker/niftytotalmarket.csv",
        "description": "Nifty Total Market Index",
        "count": 750
    },
    "Nifty Midcap 150": {
        "csv_name": "ticker/niftymidcap150.csv",
        "description": "Mid Cap 150 Stocks",
        "count": 150
    },
    "Nifty Smallcap 250": {
        "csv_name": "ticker/niftysmallcap250.csv",
        "description": "Small Cap 250 Stocks",
        "count": 250
    },
    "Nifty Mid-Smallcap 400": {
        "csv_name": "ticker/niftymidsmallcap400.csv",
        "description": "Mid-Small Cap 400 Stocks",
        "count": 400
    },
}

st.divider()
    
    st.markdown("### 📊 EMA Reference")
    st.markdown(":red[**EMA-20**] — Short-term trend")
    st.markdown(":orange[**EMA-50**] — Medium-term trend")
    st.markdown(":green[**EMA-100**] — Long-term trend")
    st.markdown(":blue[**EMA-200**] — Major trend")
    
    st.divider()
}

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ TERMINAL SETTINGS")
    
    selected_theme = st.selectbox("🎨 Select Theme", list(THEMES.keys()), index=0)
    theme = THEMES[selected_theme]
    
    st.divider()
    
    lookback_years = st.slider("📅 Historical Data (Years)", min_value=1, max_value=25, value=10, step=1)
    lookback_days = lookback_years * 365
    
    max_workers = st.slider("⚡ Data Fetch Threads", min_value=5, max_value=30, value=10)
    
    st.divider()
    
    # FIX #2: Display EMA colors with proper color swatches and descriptions
    st.markdown("### 🎨 ACTIVE THEME COLORS")
    
    for period in ["20", "50", "100", "200"]:
        color = theme['ema_colors'][period]
        st.markdown(
            f"""
            <div style='display: flex; align-items: center; margin-bottom: 8px;'>
                <div style='width: 20px; height: 20px; background-color: {color}; border-radius: 4px; margin-right: 10px;'></div>
                <div>
                    <span style='color: {theme["text_color"]}; font-weight: bold;'>EMA {period}</span>
                    <br>
                    <span style='color: #888; font-size: 11px;'>{EMA_DESCRIPTIONS[period]}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================================
# LOAD TICKERS FROM CSV
# ============================================================================

def load_tickers_from_csv(csv_filename):
    """Load tickers from CSV file"""
    try:
        df = pd.read_csv(csv_filename)
        
        # Try different column names
        symbol_col = None
        for col_name in ['Symbol', 'SYMBOL', 'Ticker', 'ticker', 'symbol']:
            if col_name in df.columns:
                symbol_col = col_name
                break
        
        if symbol_col is None:
            return None, f"Could not find Symbol column in {csv_filename}"
        
        tickers = sorted(df[symbol_col].unique().tolist())
        return tickers, None
        
    except FileNotFoundError:
        return None, f"File not found: {csv_filename}"
    except Exception as e:
        return None, f"Error loading {csv_filename}: {str(e)}"


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def convert_to_ist(dt_index):
    """Convert datetime index to IST and format cleanly"""
    ist = pytz.timezone('Asia/Kolkata')
    
    # If already timezone aware, convert; otherwise localize to UTC first
    if dt_index.tz is None:
        dt_index = dt_index.tz_localize('UTC')
    
    return dt_index.tz_convert(ist)


def format_date_ist(dt):
    """Format a datetime object to clean IST string"""
    if pd.isna(dt):
        return ""
    
    ist = pytz.timezone('Asia/Kolkata')
    
    if hasattr(dt, 'tz') and dt.tz is not None:
        dt_ist = dt.astimezone(ist)
    else:
        dt_ist = ist.localize(dt) if hasattr(dt, 'tzinfo') else dt
    
    return dt_ist.strftime('%Y-%m-%d')


# ============================================================================
# MARKET BREADTH ANALYZER
# ============================================================================

class MarketBreadthAnalyzer:
    def __init__(self, lookback_days, max_workers, theme):
        self.lookback_days = lookback_days
        self.max_workers = max_workers
        self.ema_periods = [20, 50, 100, 200]
        self.theme = theme
    
    def get_stock_data(self, ticker, max_retries=3):
        """Fetch stock data with retry logic - FIXED to get full historical data"""
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.lookback_days)
                
                # FIX: Use period parameter as fallback for better data retrieval
                # yfinance sometimes has issues with start/end dates
                hist = stock.history(start=start_date, end=end_date, auto_adjust=True)
                
                # If we got very limited data, try using period parameter instead
                if len(hist) < 200:
                    # Map lookback_days to period string
                    if self.lookback_days >= 3650:  # 10+ years
                        period_str = "max"
                    elif self.lookback_days >= 1825:  # 5+ years
                        period_str = "10y"
                    elif self.lookback_days >= 730:  # 2+ years
                        period_str = "5y"
                    elif self.lookback_days >= 365:  # 1+ year
                        period_str = "2y"
                    else:
                        period_str = "1y"
                    
                    hist = stock.history(period=period_str, auto_adjust=True)
                
                if hist.empty or len(hist) < 200:
                    return None
                
                # Trim to requested lookback if we got more data
                if len(hist) > 0:
                    cutoff_date = datetime.now() - timedelta(days=self.lookback_days)
                    # Make cutoff_date timezone-aware if hist.index is timezone-aware
                    if hist.index.tz is not None:
                        cutoff_date = cutoff_date.replace(tzinfo=hist.index.tz)
                    hist = hist[hist.index >= cutoff_date]
                
                return hist['Close'].dropna()
            except Exception as e:
                if attempt == max_retries - 1:
                    return None
        return None
    
    @staticmethod
    def calculate_ema(data, period):
        """Calculate EMA"""
        return data.ewm(span=period, adjust=False).mean()
    
    def analyze_ticker(self, ticker):
        """Analyze single ticker against all EMAs - FIXED to return ALL data after warm-up"""
        data = self.get_stock_data(ticker)
        if data is None or len(data) < 200:
            return None
        
        results = {}
        for period in self.ema_periods:
            ema = self.calculate_ema(data, period)
            # Start from index 199 (after EMA warm-up for 200-day EMA) to ensure accurate EMAs
            # But keep ALL subsequent data points for full historical analysis
            start_idx = 199
            if len(data) <= start_idx:
                return None
            
            above = (data.iloc[start_idx:].values > ema.iloc[start_idx:].values).astype(int)
            common_index = data.index[start_idx:]
            results[period] = pd.Series(above, index=common_index)
        
        return results
    
    def calculate_breadth(self, tickers):
        """Calculate market breadth for all tickers - FIXED to preserve full date range"""
        container = st.container()
        progress_bar = container.progress(0)
        status_text = container.empty()
        
        all_results = {}
        completed = 0
        failed_tickers = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.analyze_ticker, ticker): ticker for ticker in tickers}
            
            for future in futures:
                ticker = futures[future]
                try:
                    result = future.result(timeout=60)  # Increased timeout for more data
                    if result is not None:
                        all_results[ticker] = result
                    else:
                        failed_tickers.append(ticker)
                except Exception as e:
                    failed_tickers.append(ticker)
                
                completed += 1
                progress = completed / len(tickers)
                progress_bar.progress(progress)
                status_text.text(f"📊 Fetched: {completed}/{len(tickers)} tickers | Success: {len(all_results)} | Failed: {len(failed_tickers)}")
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_results:
            st.error("❌ No valid data retrieved. Please try again.")
            return None
        
        # Debug info
        sample_ticker = list(all_results.keys())[0]
        sample_data = all_results[sample_ticker][20]
        st.info(f"📊 Sample ticker '{sample_ticker}' has {len(sample_data)} data points from {sample_data.index.min().strftime('%Y-%m-%d')} to {sample_data.index.max().strftime('%Y-%m-%d')}")
        
        breadth_data = {}
        for period in self.ema_periods:
            # Collect all unique dates across all tickers
            all_dates = set()
            for ticker_results in all_results.values():
                if period in ticker_results:
                    all_dates.update(ticker_results[period].index)
            
            all_dates = sorted(list(all_dates))
            
            if not all_dates:
                continue
            
            # Create aligned series for each ticker
            series_list = []
            for ticker, ticker_results in all_results.items():
                if period in ticker_results:
                    series = ticker_results[period].reindex(all_dates, method='ffill')
                    series_list.append(series)
            
            if not series_list:
                continue
            
            df_combined = pd.concat(series_list, axis=1, ignore_index=True)
            # Don't drop all NaN rows - just fill forward to maintain data continuity
            df_combined = df_combined.ffill().bfill()
            
            # Calculate breadth metrics
            count_per_day = df_combined.sum(axis=1)
            total_stocks_per_day = df_combined.notna().sum(axis=1)
            percent_above = (count_per_day / total_stocks_per_day * 100).round(2)
            
            breadth_data[period] = {
                'percent': percent_above,
                'count': count_per_day.astype(int),
                'total': len(all_results)
            }
        
        if failed_tickers:
            st.warning(f"⚠️ Could not fetch data for {len(failed_tickers)} tickers")
        
        st.success(f"✅ Analysis Complete! Analyzed {len(all_results)} tickers with {len(all_dates)} trading days of data")
        return breadth_data


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("## 📈 NSE MARKET BREADTH TERMINAL")
    with col2:
        st.markdown(f"**Theme:** {selected_theme}")
    
    st.divider()
    
    # Index selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        selected_index = st.selectbox(
            "📊 Select Index",
            list(INDEX_CONFIG.keys()),
            index=0,
            help="Choose which index to analyze"
        )
    
    index_info = INDEX_CONFIG[selected_index]
    
    with col2:
        st.metric(label="📊 Index", value=selected_index)
    
    with col3:
        st.metric(label="📍 Stocks", value=index_info['count'])
    
    st.info(f"ℹ️ {index_info['description']}")
    st.divider()
    
    # Load tickers from CSV
    selected_tickers, error = load_tickers_from_csv(index_info['csv_name'])
    
    if error:
        st.error(f"❌ {error}")
        st.stop()
    
    if selected_tickers:
        st.success(f"✅ Loaded {len(selected_tickers)} {selected_index} tickers from CSV")
    else:
        st.error("❌ No tickers found in CSV file")
        st.stop()
    
    # Analysis options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="📊 Selected", value=len(selected_tickers))
    
    with col2:
        st.metric(label="📅 Period", value=f"{lookback_years}Y")
    
    with col3:
        st.metric(label="⚡ Threads", value=max_workers)
    
    st.divider()
    
    # Advanced options
    with st.expander("🎛️ Advanced Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            custom_tickers = st.text_area(
                "🎯 Or paste custom tickers (comma-separated)",
                height=100,
                help="Example: HDFCBANK.NS, ICICIBANK.NS, KOTAKBANK.NS"
            )
        
        with col2:
            st.info("""
            📌 **Custom Ticker Format:**
            - One ticker per line or comma-separated
            - Use .NS suffix (e.g., HDFCBANK.NS)
            - Leave empty to use index tickers
            """)
        
        if custom_tickers.strip():
            custom_list = [t.strip() for t in custom_tickers.replace('\n', ',').split(',') if t.strip()]
            selected_tickers = custom_list
            st.success(f"✅ Using {len(selected_tickers)} custom tickers")
    
    # Analyze Button
    if st.button("🚀 ANALYZE MARKET BREADTH", type="primary"):
        analyzer = MarketBreadthAnalyzer(lookback_days, max_workers, theme)
        breadth_data = analyzer.calculate_breadth(selected_tickers)
        
        if breadth_data:
            st.session_state['breadth_data'] = breadth_data
            st.session_state['selected_tickers'] = selected_tickers
            st.session_state['selected_index'] = selected_index
            st.session_state['lookback_years'] = lookback_years
    
    # Display Results
    if 'breadth_data' in st.session_state:
        breadth_data = st.session_state['breadth_data']
        
        # Show data range info
        sample_period = list(breadth_data.keys())[0]
        date_range = breadth_data[sample_period]['percent'].index
        total_days = len(date_range)
        start_date = format_date_ist(date_range.min())
        end_date = format_date_ist(date_range.max())
        
        st.info(f"📅 **Data Range:** {start_date} to {end_date} ({total_days} trading days)")
        
        # =====================================================================
        # METRICS
        # =====================================================================
        
        st.markdown("### 📊 CURRENT MARKET STATUS")
        
        metrics_cols = st.columns(4)
        
        for idx, period in enumerate([20, 50, 100, 200]):
            with metrics_cols[idx]:
                pct = breadth_data[period]['percent'].iloc[-1]
                count = int(breadth_data[period]['count'].iloc[-1])
                total = breadth_data[period]['total']
                
                if pct >= 70:
                    color = theme['up_color']
                    signal = "🟢 STRONG"
                elif pct >= 50:
                    color = theme['neutral_color']
                    signal = "🟡 BULLISH"
                elif pct >= 30:
                    color = theme['down_color']
                    signal = "🟠 BEARISH"
                else:
                    color = theme['down_color']
                    signal = "🔴 WEAK"
                
                st.markdown(f"""
                <div style='background-color: {theme['grid_color']}; padding: 20px; border-radius: 10px; border-left: 5px solid {theme['ema_colors'][str(period)]}'>
                    <p style='color: {theme['text_color']}; margin: 0; font-size: 13px; font-weight: bold; text-transform: uppercase;'>EMA {period}</p>
                    <p style='color: {color}; margin: 10px 0 0 0; font-size: 28px; font-weight: bold;'>{pct:.1f}%</p>
                    <p style='color: {theme['text_color']}; margin: 5px 0 0 0; font-size: 12px;'>{count}/{total} stocks above</p>
                    <p style='color: {color}; margin: 5px 0 0 0; font-size: 11px; font-weight: bold;'>{signal}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # =====================================================================
        # CHARTS - FIXED: Display ALL historical data with proper x-axis range
        # =====================================================================
        
        st.markdown("### 📈 MARKET BREADTH EVOLUTION")
        
        # Add chart time range selector
        chart_range = st.selectbox(
            "📅 Chart Display Range",
            ["Full History", "Last 5 Years", "Last 3 Years", "Last 1 Year", "Last 6 Months", "Last 3 Months"],
            index=0,
            key="chart_range"
        )
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4],
            subplot_titles=("Percentage of Stocks Above EMA", "Stock Count Above EMA")
        )
        
        # Determine x-axis range based on selection
        x_range = None
        if chart_range != "Full History":
            end_dt = datetime.now()
            if chart_range == "Last 5 Years":
                start_dt = end_dt - timedelta(days=5*365)
            elif chart_range == "Last 3 Years":
                start_dt = end_dt - timedelta(days=3*365)
            elif chart_range == "Last 1 Year":
                start_dt = end_dt - timedelta(days=365)
            elif chart_range == "Last 6 Months":
                start_dt = end_dt - timedelta(days=180)
            elif chart_range == "Last 3 Months":
                start_dt = end_dt - timedelta(days=90)
            x_range = [start_dt, end_dt]
        
        # Percentage lines - ALL DATA
        for period in [20, 50, 100, 200]:
            data = breadth_data[period]['percent']
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data.values,
                    name=f"EMA {period}",
                    line=dict(color=theme['ema_colors'][str(period)], width=2),
                    hovertemplate=f"<b>EMA {period}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}%<extra></extra>"
                ),
                row=1, col=1
            )
        
        # Reference lines
        fig.add_hline(y=70, line_dash="dash", line_color=theme['up_color'], line_width=1.5, annotation_text="70% (Strong)", annotation_position="right", row=1)
        fig.add_hline(y=50, line_dash="dot", line_color=theme['neutral_color'], line_width=1.5, annotation_text="50% (Neutral)", annotation_position="right", row=1)
        fig.add_hline(y=30, line_dash="dash", line_color=theme['down_color'], line_width=1.5, annotation_text="30% (Weak)", annotation_position="right", row=1)
        
        # Count lines (area fill for better visualization)
        for period in [20, 50, 100, 200]:
            data = breadth_data[period]['count']
            color_hex = theme['ema_colors'][str(period)].lstrip('#')
            r, g, b = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data.values,
                    name=f"Count EMA {period}",
                    line=dict(color=theme['ema_colors'][str(period)], width=1.5),
                    fill='tozeroy',
                    fillcolor=f"rgba({r},{g},{b},0.2)",
                    showlegend=False,
                    hovertemplate=f"<b>EMA {period}</b><br>%{{x|%Y-%m-%d}}<br>Stocks: %{{y}}<extra></extra>"
                ),
                row=2, col=1
            )
        
        # Update axes with optional range
        fig.update_xaxes(
            title_text="Date", 
            row=2, col=1, 
            gridcolor=theme['grid_color'], 
            showgrid=True,
            range=x_range,
            rangeslider=dict(visible=True, thickness=0.05)
        )
        fig.update_yaxes(title_text="% of Stocks", row=1, col=1, gridcolor=theme['grid_color'], range=[0, 100], showgrid=True)
        fig.update_yaxes(title_text="Stock Count", row=2, col=1, gridcolor=theme['grid_color'], showgrid=True)
        
        # If range is set, also update the first x-axis
        if x_range:
            fig.update_xaxes(range=x_range, row=1, col=1)
        
        fig.update_layout(
            height=800,
            plot_bgcolor=theme['bg_color'],
            paper_bgcolor=theme['bg_color'],
            font=dict(color=theme['text_color'], family="Courier New", size=11),
            hovermode='x unified',
            margin=dict(l=60, r=60, t=80, b=60),
            showlegend=True,
            legend=dict(bgcolor=f"rgba(0,0,0,0.5)", bordercolor=theme['grid_color'], borderwidth=1, yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # =====================================================================
        # DETAILED STATISTICS - FIXED: Proper scrollable table with column config
        # =====================================================================
        
        st.markdown("### 📋 DETAILED BREADTH ANALYSIS BY EMA")
        
        tab1, tab2, tab3, tab4 = st.tabs(["EMA 20", "EMA 50", "EMA 100", "EMA 200"])
        
        for period, tab in zip([20, 50, 100, 200], [tab1, tab2, tab3, tab4]):
            with tab:
                data = breadth_data[period]
                
                # Statistics
                col1, col2, col3, col4, col5 = st.columns(5)
                
                latest_pct = data['percent'].iloc[-1]
                latest_count = int(data['count'].iloc[-1])
                highest = data['percent'].max()
                lowest = data['percent'].min()
                avg = data['percent'].mean()
                
                with col1:
                    st.metric("Current %", f"{latest_pct:.2f}%")
                with col2:
                    st.metric("Stocks Above", f"{latest_count}/{data['total']}")
                with col3:
                    st.metric("Highest", f"{highest:.2f}%")
                with col4:
                    st.metric("Lowest", f"{lowest:.2f}%")
                with col5:
                    st.metric("Average", f"{avg:.2f}%")
                
                st.divider()
                
                # FIX #3: Format dates in clean IST format
                df_display = pd.DataFrame({
                    'Date': [format_date_ist(dt) for dt in data['percent'].index],
                    'Percentage (%)': data['percent'].values.round(2),
                    'Count Above': data['count'].values.astype(int),
                    'Total': [data['total']] * len(data['percent']),
                    'Signal': ['🟢 STRONG' if x >= 70 else '🟡 BULLISH' if x >= 50 else '🟠 BEARISH' if x >= 30 else '🔴 WEAK' for x in data['percent'].values]
                })
                
                # Show data summary
                st.caption(f"📊 Total: {len(df_display)} trading days of data")
                
                # Add date range filter
                col1, col2 = st.columns(2)
                with col1:
                    show_recent = st.selectbox(
                        f"View range (EMA {period})",
                        ["Last 30 days", "Last 90 days", "Last 1 year", "Last 3 years", "Last 5 years", "All data"],
                        key=f"range_{period}"
                    )
                
                # Filter based on selection
                if show_recent == "Last 30 days":
                    df_filtered = df_display.tail(30)
                elif show_recent == "Last 90 days":
                    df_filtered = df_display.tail(90)
                elif show_recent == "Last 1 year":
                    df_filtered = df_display.tail(252)
                elif show_recent == "Last 3 years":
                    df_filtered = df_display.tail(756)
                elif show_recent == "Last 5 years":
                    df_filtered = df_display.tail(1260)
                else:
                    df_filtered = df_display
                
                st.caption(f"📋 Showing {len(df_filtered)} rows")
                
                # FIX #1: Use proper dataframe configuration for scrolling
                st.dataframe(
                    df_filtered.sort_values('Date', ascending=False).reset_index(drop=True),
                    use_container_width=True,
                    hide_index=True,
                    height=500,  # Increased height for better visibility
                    column_config={
                        "Date": st.column_config.TextColumn("Date", width="medium"),
                        "Percentage (%)": st.column_config.NumberColumn("Percentage (%)", format="%.2f", width="medium"),
                        "Count Above": st.column_config.NumberColumn("Count Above", width="small"),
                        "Total": st.column_config.NumberColumn("Total", width="small"),
                        "Signal": st.column_config.TextColumn("Signal", width="medium")
                    }
                )
                
                # Download button - full data
                csv = df_display.to_csv(index=False)
                st.download_button(
                    f"⬇️ Download EMA {period} Full History CSV",
                    csv,
                    f"breadth_ema{period}_{st.session_state.get('selected_index', 'custom').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    key=f"download_{period}"
                )
        
        # =====================================================================
        # INTERPRETATION GUIDE
        # =====================================================================
        
        st.divider()
        st.markdown("### 📖 INTERPRETATION GUIDE")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Market Breadth Signals:**
            
            🟢 **Above 70%** - Strong trend
            - Majority of stocks advancing
            - Risk-on environment
            - Support bullish positions
            
            🟡 **50-70%** - Bullish participation
            - Healthy uptrend
            - Most stocks supporting
            - Trade carefully
            
            🟠 **30-50%** - Weak participation  
            - Bear market developing
            - Few stocks leading
            - Reduce long positions
            
            🔴 **Below 30%** - Capitulation zone
            - Very weak market
            - Extreme pessimism
            - Potential reversal setup
            """)
        
        with col2:
            st.markdown(f"""
            **Current Index: {st.session_state.get('selected_index', 'Nifty 50')}**
            
            📍 **EMA Period Meanings:**
            
            📍 **EMA 20** - Short-term momentum
            - Quick trend reversals
            - Useful for intraday
            - Higher volatility
            
            📍 **EMA 50** - Medium-term
            - Standard trend indicator
            - Swing trading reference
            - Balanced perspective
            
            📍 **EMA 100** - Intermediate support
            - Swing trend identifier
            - Position trading
            - Strong support/resistance
            
            📍 **EMA 200** - Long-term trend
            - Strategic direction
            - Major market strength
            - Long-term bias indicator
            """)


if __name__ == "__main__":
    main()

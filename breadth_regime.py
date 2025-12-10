import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import warnings
import time
import random
import os

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Historical Breadth Regime Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

THEMES = {
    "Dark": {
        "bg_color": "#161614",
        "text_color": "#FFD700",
        "grid_color": "#2a2a2a",
        "ema_colors": {"20": "#FF6B9D", "50": "#4ECDC4", "100": "#95E1D3", "200": "#FF6348"},
        "up_color": "#00D084",
        "down_color": "#FF5E78",
    },
    "Terminal Green": {
        "bg_color": "#0B0E11",
        "text_color": "#00FF41",
        "grid_color": "#1a1a1a",
        "ema_colors": {"20": "#39FF14", "50": "#00FF41", "100": "#0FFF50", "200": "#3FFF00"},
        "up_color": "#39FF14",
        "down_color": "#FF0000",
    },
}

INDEX_CONFIG = {
    "Nifty 50": {
        "csv_name": "ticker/nifty50.csv",
        "description": "Top 50 Large Cap Stocks",
    },
    "Nifty 100": {
        "csv_name": "ticker/nifty100.csv",
        "description": "Top 100 Large Cap Stocks",
    },
    "Nifty 200": {
        "csv_name": "ticker/nifty200.csv",
        "description": "Top 200 Large Cap and Mid Cap Stocks",
    },
    "Nifty Total Market": {
        "csv_name": "ticker/niftytotalmarket.csv",
        "description": "Nifty Total Market Index",
    },
}

# Configuration constants
MIN_SUCCESS_RATE = 0.3
DEFAULT_EMA_PERIOD = 50
MAX_RETRIES = 3
BASE_TIMEOUT = 15

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    selected_theme = st.selectbox("🎨 Theme", list(THEMES.keys()), index=0)
    theme = THEMES[selected_theme]
    
    st.markdown("---")
    st.markdown("### 🔧 Advanced Settings")
    max_workers = st.slider("Concurrent Downloads", min_value=1, max_value=3, value=1, 
                           help="Lower values = more reliable but slower")
    request_delay = st.slider("Request Delay (sec)", min_value=0.5, max_value=3.0, value=1.5, step=0.1,
                             help="Higher values help avoid rate limits")
    ema_period = st.slider("EMA Period", min_value=20, max_value=200, value=50, step=10,
                          help="Moving average period for breadth calculation")
    
    st.markdown("---")
    st.markdown("### ℹ️ Info")
    st.caption("💡 Tip: If you see rate limit errors, try:")
    st.caption("• Reduce concurrent downloads to 1")
    st.caption("• Increase request delay to 2-3 seconds")
    st.caption("• Wait a few minutes before retrying")

@st.cache_data(ttl=3600)
def load_tickers_from_csv(csv_filename):
    """Load ticker symbols from CSV file with validation."""
    if not os.path.exists(csv_filename):
        return None, f"File not found: {csv_filename}"
    
    try:
        df = pd.read_csv(csv_filename)
        symbol_col = None
        for col_name in ['Symbol', 'SYMBOL', 'Ticker', 'ticker', 'symbol']:
            if col_name in df.columns:
                symbol_col = col_name
                break
        
        if symbol_col is None:
            return None, "Could not find Symbol column. Expected columns: Symbol, SYMBOL, Ticker, ticker, or symbol"
        
        tickers = sorted(df[symbol_col].unique().tolist())
        return tickers, None
    except Exception as e:
        return None, f"Error reading CSV: {str(e)}"

class HistoricalBreadthAnalyzer:
    """Analyzer for calculating market breadth indicators."""
    
    def __init__(self, max_workers, theme, request_delay=1.5, ema_period=50):
        self.max_workers = max_workers
        self.ema_period = ema_period
        self.theme = theme
        self.request_delay = request_delay
        self.failed_tickers = []
        self.rate_limited_tickers = []
    
    def get_stock_data(self, ticker, start_date, end_date, max_retries=MAX_RETRIES):
        """Fetch stock data with exponential backoff and rate limit handling."""
        for attempt in range(max_retries):
            try:
                # Progressive delay increase
                base_delay = self.request_delay * (1.5 ** attempt)
                time.sleep(base_delay + random.uniform(0, 0.5))
                
                hist = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    timeout=BASE_TIMEOUT
                )
                
                if hist.empty or len(hist) < self.ema_period:
                    return None
                
                return hist['Close'].dropna()
            
            except Exception as e:
                error_str = str(e).lower()
                
                # Handle rate limiting with longer waits
                if 'rate limit' in error_str or '429' in error_str or 'too many requests' in error_str:
                    wait_time = (attempt + 1) * 5 + random.uniform(0, 2)
                    if attempt == max_retries - 1:
                        self.rate_limited_tickers.append(ticker)
                        return None
                    time.sleep(wait_time)
                    continue
                
                # Handle timeouts and connection issues
                if 'timeout' in error_str or 'connection' in error_str:
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt + random.uniform(0, 1))
                        continue
                
                # Final attempt failed
                if attempt == max_retries - 1:
                    self.failed_tickers.append(ticker)
                    return None
        
        return None
    
    @staticmethod
    def calculate_ema(data, period):
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period, adjust=False).mean()
    
    def analyze_ticker(self, ticker, start_date, end_date):
        """Analyze single ticker for breadth calculation."""
        data = self.get_stock_data(ticker, start_date, end_date)
        if data is None or len(data) < self.ema_period:
            return None
        
        ema = self.calculate_ema(data, self.ema_period)
        above = (data.values > ema.values).astype(int)
        return pd.Series(above, index=data.index)
    
    def calculate_breadth(self, tickers, start_date, end_date):
        """Calculate market breadth with progress tracking."""
        container = st.container()
        progress_bar = container.progress(0)
        status_text = container.empty()
        
        all_results = {}
        completed = 0
        self.failed_tickers = []
        self.rate_limited_tickers = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.analyze_ticker, ticker, start_date, end_date): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(future_to_ticker):
                try:
                    result = future.result(timeout=60)
                    ticker = future_to_ticker[future]
                    if result is not None:
                        all_results[ticker] = result
                except Exception:
                    pass
                
                completed += 1
                progress = completed / len(tickers)
                progress_bar.progress(progress)
                
                # Enhanced status display
                success_count = len(all_results)
                fail_count = len(self.failed_tickers)
                rate_limit_count = len(self.rate_limited_tickers)
                status_text.text(
                    f"✅ Success: {success_count} | ⏳ Processing: {completed}/{len(tickers)} | "
                    f"❌ Failed: {fail_count} | ⚠️ Rate Limited: {rate_limit_count}"
                )
        
        progress_bar.empty()
        status_text.empty()
        
        # Show detailed warnings
        if self.rate_limited_tickers:
            st.warning(
                f"⚠️ {len(self.rate_limited_tickers)} stocks hit rate limits. "
                f"Consider increasing delay or reducing workers.\n\n"
                f"Rate limited: {', '.join(self.rate_limited_tickers[:10])}"
                f"{' and more...' if len(self.rate_limited_tickers) > 10 else ''}"
            )
        
        if self.failed_tickers:
            with st.expander(f"❌ {len(self.failed_tickers)} stocks failed to load"):
                st.text(', '.join(self.failed_tickers))
        
        # Check minimum success threshold
        success_rate = len(all_results) / len(tickers)
        if success_rate < MIN_SUCCESS_RATE:
            st.error(
                f"❌ Only {len(all_results)} out of {len(tickers)} stocks loaded "
                f"({success_rate:.1%}). Results may be unreliable."
            )
            return None
        elif success_rate < 0.7:
            st.warning(
                f"⚠️ Partial data: {len(all_results)}/{len(tickers)} stocks loaded "
                f"({success_rate:.1%})"
            )
        
        if not all_results:
            st.error("No data could be fetched. Please try again later.")
            return None
        
        # Combine results efficiently
        df_combined = pd.DataFrame(all_results).dropna()
        
        breadth_percent = (df_combined.mean(axis=1) * 100).round(2)
        breadth_count = df_combined.sum(axis=1).astype(int)
        
        return {
            'percent': breadth_percent,
            'count': breadth_count,
            'total': len(all_results),
            'failed': len(self.failed_tickers),
            'rate_limited': len(self.rate_limited_tickers)
        }

def format_time_ago(timestamp):
    """Format timestamp as human-readable time ago."""
    if timestamp is None:
        return "Never"
    delta = datetime.now() - timestamp
    if delta.seconds < 60:
        return f"{delta.seconds} seconds ago"
    elif delta.seconds < 3600:
        return f"{delta.seconds // 60} minutes ago"
    elif delta.days == 0:
        return f"{delta.seconds // 3600} hours ago"
    elif delta.days == 1:
        return "1 day ago"
    else:
        return f"{delta.days} days ago"

def check_session_state_validity(selected_index, data_key):
    """Check if cached data is valid for current selection."""
    return (data_key in st.session_state and 
            st.session_state.get('selected_index') == selected_index)

def main():
    st.markdown("## 📊 Historical Breadth Regime Analysis")
    st.markdown("*Compare 5-year vs 10-year breadth patterns | Track all-time lows since 2000*")
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_index = st.selectbox("Select Index", list(INDEX_CONFIG.keys()), index=0)
    with col2:
        st.metric("Index", selected_index)
    
    st.info(f"ℹ️ {INDEX_CONFIG[selected_index]['description']}")
    
    # Check if index changed - clear old data
    if 'selected_index' in st.session_state and st.session_state['selected_index'] != selected_index:
        for key in ['breadth_5y', 'breadth_10y', 'breadth_alltime', 
                    'breadth_5y_timestamp', 'breadth_10y_timestamp', 'breadth_alltime_timestamp']:
            if key in st.session_state:
                del st.session_state[key]
        st.info("🔄 Index changed - cleared cached data")
    
    st.session_state['selected_index'] = selected_index
    st.divider()
    
    # Load tickers with validation
    selected_tickers, error = load_tickers_from_csv(INDEX_CONFIG[selected_index]['csv_name'])
    
    if error:
        st.error(f"❌ {error}")
        st.info("**Troubleshooting:**\n"
                "1. Ensure CSV file exists in 'ticker/' directory\n"
                "2. Check CSV has 'Symbol' or 'SYMBOL' column\n"
                "3. Verify file permissions")
        st.stop()
    
    if selected_tickers:
        st.success(f"✅ Loaded {len(selected_tickers)} tickers from {INDEX_CONFIG[selected_index]['csv_name']}")
    else:
        st.error("❌ No tickers found in CSV file")
        st.stop()
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["📈 5Y vs 10Y Comparison", "📉 All-Time Lows (2000-2025)", "📋 Detailed Table"])
    
    with tab1:
        st.markdown("### 5-Year vs 10-Year Breadth Comparison")
        
        # Show data freshness
        if 'breadth_5y_timestamp' in st.session_state:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"🕐 Last fetched: {format_time_ago(st.session_state.get('breadth_5y_timestamp'))}")
            with col2:
                if st.button("🗑️ Clear Cache", key="clear_5y10y"):
                    for key in ['breadth_5y', 'breadth_10y', 'breadth_5y_timestamp', 'breadth_10y_timestamp']:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()
        
        if st.button("🔄 FETCH 5Y & 10Y DATA", key="fetch_5y_10y", type="primary", use_container_width=True):
            analyzer = HistoricalBreadthAnalyzer(max_workers, theme, request_delay, ema_period)
            
            today = datetime.now()
            start_5y = today - timedelta(days=365*5)
            start_10y = today - timedelta(days=365*10)
            
            with st.spinner("📥 Fetching 5-year data... (this may take a few minutes)"):
                breadth_5y = analyzer.calculate_breadth(selected_tickers, start_5y, today)
            
            if breadth_5y:
                with st.spinner("📥 Fetching 10-year data... (this may take a few minutes)"):
                    breadth_10y = analyzer.calculate_breadth(selected_tickers, start_10y, today)
                
                if breadth_10y:
                    st.success("✅ Data fetched successfully!")
                    st.session_state['breadth_5y'] = breadth_5y
                    st.session_state['breadth_10y'] = breadth_10y
                    st.session_state['breadth_5y_timestamp'] = datetime.now()
                    st.session_state['breadth_10y_timestamp'] = datetime.now()
                    st.rerun()
        
        if check_session_state_validity(selected_index, 'breadth_5y') and \
           check_session_state_validity(selected_index, 'breadth_10y'):
            
            breadth_5y = st.session_state['breadth_5y']
            breadth_10y = st.session_state['breadth_10y']
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Last 5 Years", "Last 10 Years"),
                shared_xaxes=False,
                vertical_spacing=0.12,
                specs=[[{"secondary_y": False}], [{"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Scatter(
                    x=breadth_5y['percent'].index,
                    y=breadth_5y['percent'].values,
                    name="5Y Breadth %",
                    line=dict(color=theme['ema_colors']['50'], width=2),
                    fill='tozeroy',
                    fillcolor='rgba(78, 205, 196, 0.2)',
                    hovertemplate="<b>5Y</b><br>%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>"
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=breadth_10y['percent'].index,
                    y=breadth_10y['percent'].values,
                    name="10Y Breadth %",
                    line=dict(color=theme['ema_colors']['100'], width=2),
                    fill='tozeroy',
                    fillcolor='rgba(149, 225, 211, 0.2)',
                    hovertemplate="<b>10Y</b><br>%{x|%Y-%m-%d}<br>%{y:.2f}%<extra></extra>"
                ),
                row=2, col=1
            )
            
            fig.add_hline(y=70, line_dash="dash", line_color=theme['up_color'], line_width=1, row=1)
            fig.add_hline(y=30, line_dash="dash", line_color=theme['down_color'], line_width=1, row=1)
            fig.add_hline(y=70, line_dash="dash", line_color=theme['up_color'], line_width=1, row=2)
            fig.add_hline(y=30, line_dash="dash", line_color=theme['down_color'], line_width=1, row=2)
            
            fig.update_xaxes(title_text="Date", row=2, col=1, gridcolor=theme['grid_color'])
            fig.update_yaxes(title_text="Breadth %", row=1, col=1, gridcolor=theme['grid_color'], range=[0, 100])
            fig.update_yaxes(title_text="Breadth %", row=2, col=1, gridcolor=theme['grid_color'], range=[0, 100])
            
            fig.update_layout(
                height=800,
                plot_bgcolor=theme['bg_color'],
                paper_bgcolor=theme['bg_color'],
                font=dict(color=theme['text_color'], size=11),
                hovermode='x unified',
                showlegend=True,
                margin=dict(l=60, r=60, t=80, b=60),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 📊 STATISTICS COMPARISON")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("5Y Current", f"{breadth_5y['percent'].iloc[-1]:.2f}%")
            with col2:
                st.metric("5Y Average", f"{breadth_5y['percent'].mean():.2f}%")
            with col3:
                st.metric("5Y Highest", f"{breadth_5y['percent'].max():.2f}%")
            with col4:
                st.metric("5Y Lowest", f"{breadth_5y['percent'].min():.2f}%")
            with col5:
                st.metric("5Y Std Dev", f"{breadth_5y['percent'].std():.2f}%")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("10Y Current", f"{breadth_10y['percent'].iloc[-1]:.2f}%")
            with col2:
                st.metric("10Y Average", f"{breadth_10y['percent'].mean():.2f}%")
            with col3:
                st.metric("10Y Highest", f"{breadth_10y['percent'].max():.2f}%")
            with col4:
                st.metric("10Y Lowest", f"{breadth_10y['percent'].min():.2f}%")
            with col5:
                st.metric("10Y Std Dev", f"{breadth_10y['percent'].std():.2f}%")
    
    with tab2:
        st.markdown("### All-Time Lowest Breadth Levels (2000-2025)")
        
        # Show data freshness
        if 'breadth_alltime_timestamp' in st.session_state:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption(f"🕐 Last fetched: {format_time_ago(st.session_state.get('breadth_alltime_timestamp'))}")
            with col2:
                if st.button("🗑️ Clear Cache", key="clear_alltime"):
                    if 'breadth_alltime' in st.session_state:
                        del st.session_state['breadth_alltime']
                    if 'breadth_alltime_timestamp' in st.session_state:
                        del st.session_state['breadth_alltime_timestamp']
                    st.rerun()
        
        st.warning("⚠️ **Note:** Fetching 25 years of data takes 10-20 minutes and may hit rate limits. "
                  "Consider using 10Y data for faster results.")
        
        if st.button("🔄 FETCH ALL-TIME DATA (2000-2025)", key="fetch_alltime", type="primary", use_container_width=True):
            analyzer = HistoricalBreadthAnalyzer(max_workers, theme, request_delay, ema_period)
            
            start_date = datetime(2000, 1, 1)
            today = datetime.now()
            
            with st.spinner("📥 Fetching 25-year data... (this will take 10-20 minutes)"):
                breadth_alltime = analyzer.calculate_breadth(selected_tickers, start_date, today)
            
            if breadth_alltime:
                st.success("✅ All-time data fetched successfully!")
                st.session_state['breadth_alltime'] = breadth_alltime
                st.session_state['breadth_alltime_timestamp'] = datetime.now()
                st.rerun()
        
        if check_session_state_validity(selected_index, 'breadth_alltime'):
            breadth_alltime = st.session_state['breadth_alltime']
            
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=breadth_alltime['percent'].index,
                    y=breadth_alltime['percent'].values,
                    name="Breadth %",
                    line=dict(color=theme['ema_colors']['100'], width=1.5),
                    fill='tozeroy',
                    fillcolor='rgba(149, 225, 211, 0.2)',
                    hovertemplate="<b>Date</b><br>%{x|%Y-%m-%d}<br>Breadth: %{y:.2f}%<extra></extra>"
                )
            )
            
            fig.add_hline(y=70, line_dash="dash", line_color=theme['up_color'], annotation_text="70%")
            fig.add_hline(y=30, line_dash="dash", line_color=theme['down_color'], annotation_text="30%")
            
            fig.update_layout(
                title=f"25-Year Breadth History ({selected_index})",
                height=600,
                plot_bgcolor=theme['bg_color'],
                paper_bgcolor=theme['bg_color'],
                font=dict(color=theme['text_color'], size=11),
                hovermode='x unified',
                xaxis_title="Date",
                yaxis_title="Breadth %",
                yaxis=dict(range=[0, 100], gridcolor=theme['grid_color']),
                xaxis=dict(gridcolor=theme['grid_color']),
                margin=dict(l=60, r=60, t=80, b=60),
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 📉 LOWEST BREADTH PERIODS")
            
            lowest_breadth = breadth_alltime['percent'].nsmallest(20)
            
            df_lowest = pd.DataFrame({
                'Date': lowest_breadth.index.strftime('%Y-%m-%d'),
                'Year': lowest_breadth.index.strftime('%Y'),
                'Breadth %': lowest_breadth.values.round(2),
                'Stocks Above': breadth_alltime['count'].loc[lowest_breadth.index].values,
                'Total Stocks': [breadth_alltime['total']] * len(lowest_breadth)
            }).reset_index(drop=True)
            
            st.dataframe(df_lowest, use_container_width=True, hide_index=True)
            
            csv = df_lowest.to_csv(index=False)
            st.download_button(
                "📥 Download Lowest Breadth Data (CSV)",
                csv,
                f"lowest_breadth_2000_2025_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    
    with tab3:
        st.markdown("### 📋 DETAILED BREADTH TABLE")
        
        if check_session_state_validity(selected_index, 'breadth_5y'):
            breadth_5y = st.session_state['breadth_5y']
            
            df_table = pd.DataFrame({
                'Date': breadth_5y['percent'].index.strftime('%Y-%m-%d'),
                'Breadth %': breadth_5y['percent'].values.round(2),
                'Stocks Above': breadth_5y['count'].values.astype(int),
                'Total Stocks': [breadth_5y['total']] * len(breadth_5y['percent']),
                'Regime': ['Strong' if x >= 70 else 'Bullish' if x >= 50 else 'Bearish' if x >= 30 else 'Weak' 
                          for x in breadth_5y['percent'].values]
            })
            
            st.dataframe(
                df_table.sort_values('Date', ascending=False), 
                use_container_width=True, 
                hide_index=True, 
                height=600
            )
            
            csv = df_table.to_csv(index=False)
            st.download_button(
                "📥 Download 5Y Detailed Data (CSV)",
                csv,
                f"breadth_5y_detailed_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        else:
            st.info("ℹ️ Fetch 5Y data from the '5Y vs 10Y Comparison' tab to view detailed table")
    
    # Footer
    st.divider()
    st.caption(f"💡 Using EMA-{ema_period} | Max Workers: {max_workers} | Delay: {request_delay}s")

if __name__ == "__main__":
    main()

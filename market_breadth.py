import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
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
        "csv_name": "nifty50.csv",
        "description": "Top 50 Large Cap Stocks",
        "count": 50
    },
    "Nifty 100": {
        "csv_name": "nifty100.csv",
        "description": "Top 100 Large Cap Stocks",
        "count": 100
    },
    "Nifty 200": {
        "csv_name": "nifty200.csv",
        "description": "Top 200 Large Cap and Mid Cap Stocks",
        "count": 200
    },
    "Nifty 500": {
        "csv_name": "nifty500.csv",
        "description": "Nifty 500 - Broad Market Index",
        "count": 500
    },
    "Nifty Total Market": {
        "csv_name": "niftytotalmarket.csv",
        "description": "Nifty Total Market Index",
        "count": 750
    },
    "Nifty Midcap 150": {
        "csv_name": "niftymidcap150.csv",
        "description": "Mid Cap 150 Stocks",
        "count": 150
    },
    "Nifty Smallcap 250": {
        "csv_name": "niftysmallcap250.csv",
        "description": "Small Cap 250 Stocks",
        "count": 250
    },
    "Nifty Mid-Smallcap 400": {
        "csv_name": "niftymidsmallcap400.csv",
        "description": "Mid-Small Cap 400 Stocks",
        "count": 400
    },
}

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("### ⚙️ TERMINAL SETTINGS")
    
    selected_theme = st.selectbox("🎨 Select Theme", list(THEMES.keys()), index=0)
    theme = THEMES[selected_theme]
    
    st.divider()
    
    lookback_days = st.slider("📅 Historical Data (Years)", min_value=1, max_value=10, value=5, step=1)
    lookback_days = lookback_days * 365
    
    max_workers = st.slider("⚡ Data Fetch Threads", min_value=5, max_value=20, value=10)
    
    st.divider()
    
    st.markdown("### 🎨 ACTIVE THEME COLORS")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"**EMA 20**")
    with col2:
        st.markdown(f"**EMA 50**")
    with col3:
        st.markdown(f"**EMA 100**")
    with col4:
        st.markdown(f"**EMA 200**")

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
# MARKET BREADTH ANALYZER
# ============================================================================

class MarketBreadthAnalyzer:
    def __init__(self, lookback_days, max_workers, theme):
        self.lookback_days = lookback_days
        self.max_workers = max_workers
        self.ema_periods = [20, 50, 100, 200]
        self.theme = theme
    
    def get_stock_data(self, ticker, max_retries=2):
        """Fetch stock data with retry logic"""
        for attempt in range(max_retries):
            try:
                stock = yf.Ticker(ticker)
                end_date = datetime.now()
                start_date = end_date - timedelta(days=self.lookback_days)
                hist = stock.history(start=start_date, end=end_date)
                
                if hist.empty or len(hist) < 200:
                    return None
                
                return hist['Close'].dropna()
            except:
                if attempt == max_retries - 1:
                    return None
        return None
    
    @staticmethod
    def calculate_ema(data, period):
        """Calculate EMA"""
        return data.ewm(span=period, adjust=False).mean()
    
    def analyze_ticker(self, ticker):
        """Analyze single ticker against all EMAs"""
        data = self.get_stock_data(ticker)
        if data is None or len(data) < 200:
            return None
        
        results = {}
        for period in self.ema_periods:
            ema = self.calculate_ema(data, period)
            start_idx = 199
            above = (data.iloc[start_idx:].values > ema.iloc[start_idx:].values).astype(int)
            common_index = data.index[start_idx:]
            results[period] = pd.Series(above, index=common_index)
        
        return results
    
    def calculate_breadth(self, tickers):
        """Calculate market breadth for all tickers"""
        container = st.container()
        progress_bar = container.progress(0)
        status_text = container.empty()
        
        all_results = {}
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.analyze_ticker, ticker) for ticker in tickers]
            
            for idx, future in enumerate(futures):
                try:
                    result = future.result(timeout=10)
                    if result is not None:
                        all_results[tickers[idx]] = result
                except:
                    pass
                
                completed += 1
                progress = completed / len(tickers)
                progress_bar.progress(progress)
                status_text.text(f"📊 Fetched: {completed}/{len(tickers)} tickers")
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_results:
            return None
        
        breadth_data = {}
        for period in self.ema_periods:
            all_dates = set()
            for ticker_results in all_results.values():
                all_dates.update(ticker_results[period].index)
            
            all_dates = sorted(list(all_dates))
            series_list = []
            for ticker, ticker_results in all_results.items():
                series = ticker_results[period].reindex(all_dates, method='ffill')
                series_list.append(series)
            
            df_combined = pd.concat(series_list, axis=1, ignore_index=True)
            df_combined = df_combined.dropna()
            
            breadth_data[period] = {
                'percent': (df_combined.mean(axis=1) * 100).round(2),
                'count': df_combined.sum(axis=1).astype(int),
                'total': len(all_results)
            }
        
        st.success(f"✅ Analysis Complete! Analyzed {len(all_results)} tickers successfully")
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
        st.metric(label="📅 Period", value=f"{lookback_days/365:.0f}Y")
    
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
    if st.button("🚀 ANALYZE MARKET BREADTH", width="stretch", type="primary"):
        analyzer = MarketBreadthAnalyzer(lookback_days, max_workers, theme)
        breadth_data = analyzer.calculate_breadth(selected_tickers)
        
        if breadth_data:
            st.session_state['breadth_data'] = breadth_data
            st.session_state['selected_tickers'] = selected_tickers
            st.session_state['selected_index'] = selected_index
    
    # Display Results
    if 'breadth_data' in st.session_state:
        breadth_data = st.session_state['breadth_data']
        
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
        # CHARTS
        # =====================================================================
        
        st.markdown("### 📈 MARKET BREADTH EVOLUTION")
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.15,
            row_heights=[0.6, 0.4],
            subplot_titles=("Percentage of Stocks Above EMA", "Stock Count Above EMA")
        )
        
        # Percentage lines
        for period in [20, 50, 100, 200]:
            data = breadth_data[period]['percent']
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data.values,
                    name=f"EMA {period}",
                    line=dict(color=theme['ema_colors'][str(period)], width=2.5),
                    hovertemplate=f"<b>EMA {period}</b><br>%{{x|%Y-%m-%d}}<br>%{{y:.2f}}%<extra></extra>"
                ),
                row=1, col=1
            )
        
        # Reference lines
        fig.add_hline(y=70, line_dash="dash", line_color=theme['up_color'], line_width=1.5, annotation_text="70% (Strong)", annotation_position="right", row=1)
        fig.add_hline(y=50, line_dash="dot", line_color=theme['neutral_color'], line_width=1.5, annotation_text="50% (Neutral)", annotation_position="right", row=1)
        fig.add_hline(y=30, line_dash="dash", line_color=theme['down_color'], line_width=1.5, annotation_text="30% (Weak)", annotation_position="right", row=1)
        
        # Count bars
        for idx, period in enumerate([20, 50, 100, 200]):
            data = breadth_data[period]['count']
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data.values,
                    name=f"Count EMA {period}",
                    marker=dict(color=theme['ema_colors'][str(period)], opacity=0.7),
                    showlegend=False,
                    hovertemplate=f"<b>EMA {period}</b><br>%{{x|%Y-%m-%d}}<br>Stocks: %{{y}}<extra></extra>"
                ),
                row=2, col=1
            )
        
        # Update axes
        fig.update_xaxes(title_text="Date", row=2, col=1, gridcolor=theme['grid_color'], showgrid=True)
        fig.update_yaxes(title_text="% of Stocks", row=1, col=1, gridcolor=theme['grid_color'], range=[0, 100], showgrid=True)
        fig.update_yaxes(title_text="Stock Count", row=2, col=1, gridcolor=theme['grid_color'], showgrid=True)
        
        fig.update_layout(
            height=700,
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
        # DETAILED STATISTICS
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
                
                # Time series data
                df_display = pd.DataFrame({
                    'Date': data['percent'].index,
                    'Percentage (%)': data['percent'].values,
                    'Count Above': data['count'].values.astype(int),
                    'Total': [data['total']] * len(data['percent']),
                    'Signal': ['🟢 STRONG' if x >= 70 else '🟡 BULLISH' if x >= 50 else '🟠 BEARISH' if x >= 30 else '🔴 WEAK' for x in data['percent'].values]
                })
                
                st.dataframe(
                    df_display.sort_values('Date', ascending=False).head(30),
                    use_container_width=True,
                    hide_index=True,
                    height=400
                )
                
                # Download button
                csv = df_display.to_csv(index=False)
                st.download_button(
                    f"⬇️ Download EMA {period} CSV",
                    csv,
                    f"breadth_ema{period}_{st.session_state.get('selected_index', 'custom').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv",
                    width="stretch"
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

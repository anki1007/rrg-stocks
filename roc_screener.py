import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Momentum Stock Screener",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM THEME
# ============================================================================
st.markdown("""
<style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #252d3d 100%);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1DB954 !important;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.3);
    }
    
    /* Text colors */
    p, span, label {
        color: #e8e8e8 !important;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(29, 185, 84, 0.1);
        border-left: 4px solid #1DB954;
        padding: 15px;
        border-radius: 8px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #1DB954 0%, #1ed760 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 10px 20px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #1ed760 0%, #1DB954 100%);
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# UTILITY FUNCTIONS - Load CSV Files
# ============================================================================

@st.cache_data
def get_available_csv_files(ticker_folder="ticker"):
    """Get list of available CSV files in ticker folder"""
    try:
        ticker_path = Path(ticker_folder)
        if ticker_path.exists():
            csv_files = sorted([f.stem for f in ticker_path.glob("*.csv")])
            return csv_files if csv_files else []
    except Exception as e:
        pass
    return []


@st.cache_data
def load_tickers_from_csv(csv_filename, ticker_folder="ticker"):
    """Load tickers from selected CSV file"""
    try:
        csv_path = Path(ticker_folder) / f"{csv_filename}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
          
            columns_lower = [col.lower() for col in df.columns]
            
            if 'symbol' in columns_lower:
                idx = columns_lower.index('symbol')
                return df[df.columns[idx]].dropna().unique().tolist()
            elif 'ticker' in columns_lower:
                idx = columns_lower.index('ticker')
                return df[df.columns[idx]].dropna().unique().tolist()
            elif len(df.columns) > 0:
             
                return df.iloc[:, 0].dropna().unique().tolist()
    except Exception as e:
        pass
    return []


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class ScreenerConfig:
    """Configuration for momentum screener"""
    
    # Filter ranges
    MIN_1Y_RETURN = 0.065
    PEAK_RATIO = 0.70
    MIN_UPDAYS_PCT = 0.20
    
    # Data lookback
    LOOKBACK_DAYS = 756
    LOOKBACK_DAYS = 504
    LOOKBACK_52W = 252
    LOOKBACK_6M = 126
    LOOKBACK_3M = 63
    LOOKBACK_1M = 21


# ============================================================================
# SCREENER CLASS
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_screener_data(tickers, min_return, peak_ratio, updays_pct):
    """Fetch and screen stock data"""
    results = []
    failed = []
    
    for ticker in tickers:
        try:
            end_date = datetime.today()
            start_date = end_date - timedelta(days=ScreenerConfig.LOOKBACK_DAYS)
            
            # Fetch data
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty or len(df) < ScreenerConfig.LOOKBACK_6M:
                failed.append(ticker)
                continue
            
            close = df['Close']
            
            # Calculate metrics
            ema100 = close.ewm(span=100).mean()
            ema200 = close.ewm(span=200).mean()
            
            ret_1y = (close.iloc[-1] / close.iloc[-252] - 1) if len(close) >= 252 else np.nan
            ret_6m = (close.iloc[-1] / close.iloc[-126] - 1) if len(close) >= 126 else np.nan
            ret_3m = (close.iloc[-1] / close.iloc[-63] - 1) if len(close) >= 63 else np.nan
            ret_1m = (close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else np.nan
            
            high_52w = close.iloc[-252:].max() if len(close) >= 252 else close.max()
            peak_ratio_val = close.iloc[-1] / high_52w
            
            pct_change = close.pct_change()
            up_days_6m = (pct_change.iloc[-126:] > 0).sum() if len(pct_change) >= 126 else 0
            updays_pct_val = up_days_6m / min(126, len(pct_change))
            
            # Apply filters
            if (close.iloc[-1] > ema100.iloc[-1] and
                ema100.iloc[-1] > ema200.iloc[-1] and
                ret_1y >= min_return and
                peak_ratio_val >= peak_ratio and
                updays_pct_val >= updays_pct):
                
                results.append({
                    'Ticker': ticker,
                    'Price': round(close.iloc[-1], 2),
                    'Return_6M': round(ret_6m * 100, 2),
                    'Return_3M': round(ret_3m * 100, 2),
                    'Return_1M': round(ret_1m * 100, 2),
                    'EMA100': round(ema100.iloc[-1], 2),
                    'EMA200': round(ema200.iloc[-1], 2),
                    'Peak_Ratio': round(peak_ratio_val * 100, 2),
                    'UpDays_Pct': round(updays_pct_val * 100, 2),
                    'Volatility': round(close.pct_change().std() * np.sqrt(252) * 100, 2)
                })
        
        except Exception as e:
            failed.append(ticker)
    
    # Create DataFrame and rank
    df_results = pd.DataFrame(results)
    
    if len(df_results) > 0:
        df_results['Rank_6M'] = df_results['Return_6M'].rank(ascending=False)
        df_results['Rank_3M'] = df_results['Return_3M'].rank(ascending=False)
        df_results['Rank_1M'] = df_results['Return_1M'].rank(ascending=False)
        df_results['Rank_Final'] = (df_results['Rank_6M'] +
                                    df_results['Rank_3M'] +
                                    df_results['Rank_1M'])
        df_results = df_results.sort_values('Rank_Final').reset_index(drop=True)
        df_results['Position'] = range(1, len(df_results) + 1)
    
    return df_results, len(results), len(failed)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Initialize session state for last update time
    if 'last_scan_time' not in st.session_state:
        st.session_state.last_scan_time = None
    
    # Title
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.title("📈 ROC Screener")
        st.markdown("**Interactive Dashboard for Indian Markets**")
    
    with col2:
        # Display last scan time or current time
        if st.session_state.last_scan_time:
            st.metric("Last Updated", st.session_state.last_scan_time)
        else:
            st.metric("Last Updated", datetime.now().strftime("%H:%M IST"))
    
    st.markdown("---")
    
    # ========================================================================
    # SIDEBAR CONTROLS
    # ========================================================================
    
    with st.sidebar:
        st.header("⚙️ Screening Parameters")
        
        # ====== CSV SELECTION ======
        st.subheader("Select Index")
        
        available_csvs = get_available_csv_files("ticker")
        
        if available_csvs:
            selected_csv = st.selectbox(
                "Choose CSV Index File",
                options=available_csvs,
                help="Select the index/watchlist to screen",
                key="csv_select"
            )
            
            # Load tickers from selected CSV
            tickers = load_tickers_from_csv(selected_csv, "ticker")
            
            if tickers:
                st.success(f"✅ Loaded {len(tickers)} symbols from {selected_csv}")
            else:
                st.error(f"⚠️ No symbols found in {selected_csv}")
                tickers = []
        else:
            st.error("❌ No CSV files found in 'ticker' folder")
            st.info("📌 **Setup Instructions:**\n1. Create a folder named `ticker` in your project\n2. Add CSV files with Symbol column\n3. Restart the app")
            tickers = []
        
        st.markdown("---")
        
        # ====== FILTER THRESHOLDS ======
        st.subheader("Filter Thresholds")
        
        min_1y_ret = st.slider(
            "Minimum 1-Year Return (%)",
            min_value=0.0,
            max_value=50.0,
            value=6.5,
            step=1.0,
            help="Minimum annual return filter"
        )
        
        peak_ratio = st.slider(
            "Peak Proximity (%)",
            min_value=50.0,
            max_value=100.0,
            value=80.0,
            step=5.0,
            help="Distance from 52-week high (80% = near peak)"
        )
        
        updays_pct = st.slider(
            "Up-Days Ratio (%)",
            min_value=5.0,
            max_value=50.0,
            value=20.0,
            step=2.0,
            help="Minimum % of up-days in last 6 months"
        )
        
        st.markdown("---")
        
        # ====== DISPLAY OPTIONS ======
        st.subheader("Display Options")
        
        top_n = st.slider(
            "Number of Top Stocks",
            min_value=5,
            max_value=50,
            value=15,
            step=5
        )
        
        sort_by = st.selectbox(
            "Sort By",
            options=["Rank_Final", "Return_6M", "Return_3M", "Return_1M", "Volatility"],
            help="Column to sort results by"
        )
        
        st.markdown("---")
        
        # Run button
        run_screener = st.button(
            "🔍 Run Screener",
            use_container_width=True,
            key="run_btn",
            disabled=(len(tickers) == 0)
        )
    
    # ========================================================================
    # MAIN CONTENT
    # ========================================================================
    
    if run_screener and tickers:
        # Update the scan time
        scan_start = datetime.now()
        st.session_state.last_scan_time = scan_start.strftime("%H:%M IST")
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        status_text.text("Fetching data and screening stocks...")
        progress_bar.progress(50)
        
        # Run screening
        df_results, passed, failed = fetch_screener_data(
            tickers,
            min_1y_ret / 100,
            peak_ratio / 100,
            updays_pct / 100
        )
        
        progress_bar.progress(100)
        status_text.text("✅ Screening complete!")
        
        # Display status with last update time
        st.markdown(f"**Status**: Ready for screening | Last updated: {st.session_state.last_scan_time}")
        
        if len(df_results) == 0:
            st.error("❌ No stocks passed the selected filters. Try adjusting parameters.")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Stocks Passed",
                f"{passed}",
                f"+{passed - failed}",
                delta_color="normal"
            )
        
        with col2:
            st.metric(
                "Avg Return (6M)",
                f"{df_results['Return_6M'].mean():.1f}%",
                delta_color="normal"
            )
        
        with col3:
            st.metric(
                "Avg Volatility",
                f"{df_results['Volatility'].mean():.1f}%",
                delta_color="inverse"
            )
        
        with col4:
            st.metric(
                "Success Rate",
                f"{(passed/(passed+failed)*100):.1f}%",
                delta_color="normal"
            )
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Rankings",
            "📈 Charts",
            "🔍 Details",
            "⚡ Quick Stats"
        ])
        
        # ====================================================================
        # TAB 1: RANKINGS
        # ====================================================================
        
        with tab1:
            df_display = df_results.head(top_n)[
                ['Position', 'Ticker', 'Price', 'Return_6M', 'Return_3M',
                 'Return_1M', 'Peak_Ratio', 'Volatility', 'Rank_Final']
            ].copy()
            
            # Style DataFrame
            def highlight_returns(val):
                if isinstance(val, float):
                    if val > 0:
                        return 'color: #1DB954; font-weight: 600'
                    elif val < 0:
                        return 'color: #E74C3C; font-weight: 600'
                return ''
            
            styled_df = df_display.style.applymap(highlight_returns)
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=400
            )
            
            # Download button
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Results (CSV)",
                data=csv,
                file_name=f"momentum_screening_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # ====================================================================
        # TAB 2: CHARTS
        # ====================================================================
        
        with tab2:
            col1, col2 = st.columns(2)
            
            # Returns comparison
            with col1:
                fig_returns = go.Figure()
                
                top_tickers = df_results.head(10)
                
                fig_returns.add_trace(go.Bar(
                    name='6M Return',
                    x=top_tickers['Ticker'],
                    y=top_tickers['Return_6M'],
                    marker_color='#1DB954'
                ))
                
                fig_returns.add_trace(go.Bar(
                    name='3M Return',
                    x=top_tickers['Ticker'],
                    y=top_tickers['Return_3M'],
                    marker_color='#1ed760'
                ))
                
                fig_returns.update_layout(
                    title="Returns Comparison (Top 10)",
                    xaxis_title="Ticker",
                    yaxis_title="Return (%)",
                    barmode='group',
                    template='plotly_dark',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig_returns, use_container_width=True)
            
            # Volatility vs Return
            with col2:
                fig_scatter = go.Figure()
                
                fig_scatter.add_trace(go.Scatter(
                    x=df_results['Volatility'],
                    y=df_results['Return_6M'],
                    mode='markers+text',
                    text=df_results['Ticker'],
                    textposition='top center',
                    marker=dict(
                        size=10,
                        color=df_results['Return_6M'],
                        colorscale='Greens',
                        showscale=True
                    ),
                    textfont=dict(size=8)
                ))
                
                fig_scatter.update_layout(
                    title="Risk-Return Profile",
                    xaxis_title="Volatility (%)",
                    yaxis_title="6M Return (%)",
                    template='plotly_dark',
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
            # Distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig_dist = px.histogram(
                    df_results,
                    x='Return_6M',
                    nbins=20,
                    title="6M Return Distribution",
                    color_discrete_sequence=['#1DB954']
                )
                
                fig_dist.update_layout(template='plotly_dark')
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                fig_vol = px.box(
                    df_results,
                    y='Volatility',
                    title="Volatility Distribution",
                    color_discrete_sequence=['#1DB954']
                )
                
                fig_vol.update_layout(template='plotly_dark')
                st.plotly_chart(fig_vol, use_container_width=True)
        
        # ====================================================================
        # TAB 3: DETAILS
        # ====================================================================
        
        with tab3:
            selected_ticker = st.selectbox(
                "Select Ticker for Details",
                options=df_results['Ticker'].head(20)
            )
            
            ticker_data = df_results[df_results['Ticker'] == selected_ticker].iloc[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"### {selected_ticker}")
                st.metric("Current Price", f"₹ {ticker_data['Price']}")
                st.metric("Position", f"#{ticker_data['Position']:.0f}")
            
            with col2:
                st.metric("6M Return", f"{ticker_data['Return_6M']:.1f}%")
                st.metric("3M Return", f"{ticker_data['Return_3M']:.1f}%")
                st.metric("1M Return", f"{ticker_data['Return_1M']:.1f}%")
            
            with col3:
                st.metric("Final Rank Score", f"{ticker_data['Rank_Final']:.0f}")
                st.metric("Peak Ratio", f"{ticker_data['Peak_Ratio']:.1f}%")
                st.metric("Volatility", f"{ticker_data['Volatility']:.1f}%")
            
            # EMA levels
            st.subheader("EMA Levels")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("EMA 100", f"₹ {ticker_data['EMA100']:.2f}")
            
            with col2:
                st.metric("EMA 200", f"₹ {ticker_data['EMA200']:.2f}")
        
        # ====================================================================
        # TAB 4: QUICK STATS
        # ====================================================================
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Statistical Summary")
                st.write(df_results[['Return_6M', 'Return_3M', 'Volatility']].describe())
            
            with col2:
                st.subheader("Correlation Matrix")
                
                corr_cols = ['Return_6M', 'Return_3M', 'Return_1M', 'Volatility', 'Peak_Ratio']
                corr_matrix = df_results[corr_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    color_continuous_scale='RdYlGn',
                    zmin=-1, zmax=1,
                    title="Correlation Matrix"
                )
                
                fig_corr.update_layout(template='plotly_dark')
                st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("---")
        
        # Footer info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"✅ Passed: {passed} / Total: {passed + failed}")
        
        with col2:
            st.warning(f"📊 Displayed: Top {top_n} of {len(df_results)}")
        
        with col3:
            st.success(f"🔄 Updated: {datetime.now().strftime('%d-%b-%Y %H:%M:%S IST')}")
    
    else:
        # Initial instruction screen
        st.info(
            """
            ### 🚀 Getting Started
            
            1. **Adjust Filters** in the left sidebar to customize your screening criteria
            
            2. **Click "Run Screener"** to identify top-performing stocks
            
            3. **Analyze Results** using multiple views (Rankings, Charts, Details)
            
            4. **Download Data** for further analysis
            
            ### 📊 Available Filters
            
            - **Minimum 1-Year Return**: Select stocks with at least X% annual return
            
            - **Peak Proximity**: Filter stocks near their 52-week highs
            
            - **Up-Days Ratio**: Select stocks with consistent upward bias
            
            ### 🎯 Use Cases
            
            - **Intraday Trading**: Use top 5-10 stocks at market close
            
            - **Swing Trading**: Use top 10-15 stocks with volume confirmation
            
            - **Portfolio Selection**: Use top 20 stocks across sectors
            
            **Status**: Ready for screening | Last updated: {datetime.now().strftime('%H:%M IST')}
            """
        )


if __name__ == "__main__":
    main()

# Streamlit Momentum & ROC Screener - OPTIMIZED FOR YOUR CSV FORMAT
# ============================================================================
# This version is specifically optimized for your ticker CSV structure
# (Symbol, Company Name, Industry)
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
import os
import logging
from pathlib import Path

warnings.filterwarnings('ignore')

# Setup logging for debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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
# UTILITY FUNCTIONS - OPTIMIZED FOR YOUR CSV FORMAT
# ============================================================================

def find_ticker_folder():
    """Find the ticker folder in multiple possible locations"""
    possible_paths = [
        Path("ticker"),
        Path.cwd() / "ticker",
        Path(__file__).parent / "ticker" if "__file__" in dir() else None,
    ]
    
    for path in possible_paths:
        if path and path.exists() and path.is_dir():
            logger.info(f"✅ Found ticker folder at: {path.absolute()}")
            return path
    
    logger.error(f"❌ Ticker folder not found. Checked: {[str(p) for p in possible_paths if p]}")
    return None


@st.cache_data
def get_available_csv_files(ticker_folder="ticker"):
    """
    Get list of available CSV files in ticker folder
    Optimized for: nifty100.csv, nifty200.csv, etc.
    """
    try:
        ticker_path = Path(ticker_folder)
        
        # Try to find the folder
        if not ticker_path.exists():
            ticker_path = find_ticker_folder()
        
        if not ticker_path:
            logger.error("Ticker folder not found in any location")
            st.error(
                "❌ **Ticker folder not found!**\n\n"
                "Please ensure the `ticker/` folder exists in your project root with CSV files.\n\n"
                f"Current working directory: `{Path.cwd()}`"
            )
            return [], None
        
        csv_files = sorted([f.stem for f in ticker_path.glob("*.csv")])
        
        if not csv_files:
            logger.warning(f"No CSV files found in {ticker_path}")
            st.warning(f"⚠️ No CSV files found in {ticker_path}")
            return [], None
        
        logger.info(f"✅ Found {len(csv_files)} CSV files: {csv_files}")
        return csv_files, str(ticker_path)
        
    except Exception as e:
        logger.error(f"Error in get_available_csv_files: {str(e)}", exc_info=True)
        st.error(f"Error reading ticker folder: {str(e)}")
        return [], None


@st.cache_data
def load_tickers_from_csv(csv_filename, ticker_folder="ticker"):
    """
    Load tickers from CSV file
    Optimized for format: Symbol | Company Name | Industry
    """
    try:
        # Find the CSV file
        csv_path = Path(ticker_folder) / f"{csv_filename}.csv"
        
        if not csv_path.exists():
            # Try alternative path
            alt_path = find_ticker_folder()
            if alt_path:
                csv_path = alt_path / f"{csv_filename}.csv"
        
        if not csv_path.exists():
            logger.error(f"CSV file not found: {csv_path}")
            st.error(f"CSV file not found: {csv_filename}.csv")
            return []
        
        logger.info(f"📂 Loading CSV from: {csv_path}")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        logger.info(f"CSV columns: {df.columns.tolist()}")
        logger.info(f"CSV rows: {len(df)}")
        
        # Get column names (case-insensitive)
        columns_lower = {col.lower(): col for col in df.columns}
        
        # Find the Symbol column
        symbol_col = None
        for key in ['symbol', 'ticker', 'code']:
            if key in columns_lower:
                symbol_col = columns_lower[key]
                break
        
        if not symbol_col:
            logger.error(f"No Symbol/Ticker column found. Available columns: {df.columns.tolist()}")
            st.error(
                f"❌ No Symbol column found in {csv_filename}.csv\n\n"
                f"Available columns: {', '.join(df.columns.tolist())}\n\n"
                "Please ensure your CSV has a 'Symbol' column."
            )
            return []
        
        # Extract symbols
        symbols = df[symbol_col].dropna().astype(str).unique().tolist()
        symbols = [s.strip() for s in symbols if s.strip()]  # Remove whitespace
        
        logger.info(f"✅ Loaded {len(symbols)} symbols from '{symbol_col}' column")
        st.success(f"✅ Loaded {len(symbols)} symbols from {csv_filename}")
        
        return symbols
        
    except Exception as e:
        logger.error(f"Error loading {csv_filename}: {str(e)}", exc_info=True)
        st.error(f"Error loading {csv_filename}.csv: {str(e)}")
        return []


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class ScreenerConfig:
    """Configuration for momentum screener"""
    
    # Filter ranges
    MIN_1Y_RETURN = 0.065
    PEAK_RATIO = 0.80
    MIN_UPDAYS_PCT = 0.20
    
    # Data lookback
    LOOKBACK_DAYS = 730
    LOOKBACK_52W = 252
    LOOKBACK_6M = 126
    LOOKBACK_3M = 63
    LOOKBACK_1M = 21


# ============================================================================
# SCREENER FUNCTION
# ============================================================================

@st.cache_data(ttl=3600)
def fetch_screener_data(tickers, min_return, peak_ratio, updays_pct):
    """Fetch and screen stock data"""
    results = []
    failed = []
    
    # Progress tracking
    progress_placeholder = st.empty()
    
    for idx, ticker in enumerate(tickers):
        try:
            # Update progress
            progress = (idx + 1) / len(tickers)
            progress_placeholder.progress(
                progress,
                text=f"Processing {idx + 1}/{len(tickers)} - {ticker}"
            )
            
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
            updays_pct_val = up_days_6m / min(126, len(pct_change)) if len(pct_change) > 0 else 0
            
            # Apply filters
            if (close.iloc[-1] > ema100.iloc[-1] and
                ema100.iloc[-1] > ema200.iloc[-1] and
                ret_1y >= min_return and
                peak_ratio_val >= peak_ratio and
                updays_pct_val >= updays_pct):
                
                results.append({
                    'Ticker': ticker,
                    'Price': round(close.iloc[-1], 2),
                    'Return_1Y': round(ret_1y * 100, 2),
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
            logger.warning(f"Failed to fetch {ticker}: {str(e)}")
            failed.append(ticker)
    
    progress_placeholder.empty()
    
    # Create DataFrame and rank
    df_results = pd.DataFrame(results)
    
    if len(df_results) > 0:
        df_results['Rank_1Y'] = df_results['Return_1Y'].rank(ascending=False)
        df_results['Rank_6M'] = df_results['Return_6M'].rank(ascending=False)
        df_results['Rank_3M'] = df_results['Return_3M'].rank(ascending=False)
        df_results['Rank_1M'] = df_results['Return_1M'].rank(ascending=False)
        
        df_results['Rank_Final'] = (
            df_results['Rank_1Y'] +
            df_results['Rank_6M'] +
            df_results['Rank_3M'] +
            df_results['Rank_1M']
        )
        
        df_results = df_results.sort_values('Rank_Final').reset_index(drop=True)
        df_results['Position'] = range(1, len(df_results) + 1)
    
    return df_results, len(results), len(failed)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Initialize session state
    if 'last_scan_time' not in st.session_state:
        st.session_state.last_scan_time = None
    
    # Title
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        st.title("📈 Momentum Stock Screener")
        st.markdown("**Bloomberg-Style Interactive Dashboard for Indian Markets**")
    
    with col2:
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
        st.subheader("📁 Select Index")
        
        available_csvs, ticker_folder_path = get_available_csv_files("ticker")
        tickers = []
        
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
                st.info(f"📊 Index: **{selected_csv}** | Symbols: **{len(tickers)}**")
            else:
                st.error(f"Failed to load symbols from {selected_csv}")
        
        st.markdown("---")
        
        # ====== FILTER THRESHOLDS ======
        st.subheader("🎯 Filter Thresholds")
        
        min_1y_ret = st.slider(
            "Minimum 1-Year Return (%)",
            min_value=0.0,
            max_value=100.0,
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
        st.subheader("📊 Display Options")
        
        top_n = st.slider(
            "Number of Top Stocks",
            min_value=5,
            max_value=min(100, len(tickers)) if tickers else 50,
            value=15,
            step=5
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
        
        # Run screening
        df_results, passed, failed = fetch_screener_data(
            tickers,
            min_1y_ret / 100,
            peak_ratio / 100,
            updays_pct / 100
        )
        
        # Display status
        st.markdown(f"**Last updated**: {st.session_state.last_scan_time}")
        
        if len(df_results) == 0:
            st.error("❌ No stocks passed the selected filters. Try adjusting parameters.")
            return
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Stocks Passed", f"{passed}/{len(tickers)}")
        
        with col2:
            st.metric(
                "Avg Return (6M)",
                f"{df_results['Return_6M'].mean():.1f}%"
            )
        
        with col3:
            st.metric(
                "Avg Volatility",
                f"{df_results['Volatility'].mean():.1f}%"
            )
        
        with col4:
            st.metric(
                "Success Rate",
                f"{(passed/(passed+failed)*100):.1f}%"
            )
        
        st.markdown("---")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Rankings",
            "📈 Charts",
            "🔍 Details",
            "⚡ Stats"
        ])
        
        # ====================================================================
        # TAB 1: RANKINGS
        # ====================================================================
        
        with tab1:
            df_display = df_results.head(top_n)[[
                'Position', 'Ticker', 'Price', 'Return_1Y', 'Return_6M',
                'Return_3M', 'Return_1M', 'Peak_Ratio', 'Volatility'
            ]].copy()
            
            st.dataframe(
                df_display.style.format({
                    'Price': '₹{:,.2f}',
                    'Return_1Y': '{:.1f}%',
                    'Return_6M': '{:.1f}%',
                    'Return_3M': '{:.1f}%',
                    'Return_1M': '{:.1f}%',
                    'Peak_Ratio': '{:.1f}%',
                    'Volatility': '{:.1f}%'
                }),
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
            
            with col1:
                fig_returns = go.Figure()
                top_tickers = df_results.head(10)
                
                fig_returns.add_trace(go.Bar(
                    name='1Y Return',
                    x=top_tickers['Ticker'],
                    y=top_tickers['Return_1Y'],
                    marker_color='#1DB954'
                ))
                
                fig_returns.add_trace(go.Bar(
                    name='6M Return',
                    x=top_tickers['Ticker'],
                    y=top_tickers['Return_6M'],
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
            
            with col2:
                fig_scatter = go.Figure()
                
                fig_scatter.add_trace(go.Scatter(
                    x=df_results['Volatility'],
                    y=df_results['Return_1Y'],
                    mode='markers+text',
                    text=df_results['Ticker'],
                    textposition='top center',
                    marker=dict(
                        size=10,
                        color=df_results['Return_1Y'],
                        colorscale='Greens',
                        showscale=True
                    ),
                    textfont=dict(size=8)
                ))
                
                fig_scatter.update_layout(
                    title="Risk-Return Profile (1Y)",
                    xaxis_title="Volatility (%)",
                    yaxis_title="1Y Return (%)",
                    template='plotly_dark',
                    hovermode='closest'
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
            
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
                options=df_results['Ticker'].head(20).tolist()
            )
            
            ticker_data = df_results[df_results['Ticker'] == selected_ticker].iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Symbol", selected_ticker)
                st.metric("Current Price", f"₹{ticker_data['Price']:,.2f}")
            
            with col2:
                st.metric("1Y Return", f"{ticker_data['Return_1Y']:.1f}%")
                st.metric("6M Return", f"{ticker_data['Return_6M']:.1f}%")
            
            with col3:
                st.metric("3M Return", f"{ticker_data['Return_3M']:.1f}%")
                st.metric("1M Return", f"{ticker_data['Return_1M']:.1f}%")
            
            with col4:
                st.metric("Position", f"#{ticker_data['Position']:.0f}")
                st.metric("Volatility", f"{ticker_data['Volatility']:.1f}%")
            
            st.subheader("Technical Indicators")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("EMA 100", f"₹{ticker_data['EMA100']:,.2f}")
            
            with col2:
                st.metric("EMA 200", f"₹{ticker_data['EMA200']:,.2f}")
            
            with col3:
                st.metric("From Peak", f"{ticker_data['Peak_Ratio']:.1f}%")
        
        # ====================================================================
        # TAB 4: STATS
        # ====================================================================
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Statistical Summary")
                st.write(
                    df_results[[
                        'Return_1Y', 'Return_6M', 'Return_3M',
                        'Return_1M', 'Volatility', 'Peak_Ratio'
                    ]].describe().round(2)
                )
            
            with col2:
                st.subheader("📊 Correlation Matrix")
                corr_cols = [
                    'Return_1Y', 'Return_6M', 'Return_3M',
                    'Return_1M', 'Volatility', 'Peak_Ratio'
                ]
                corr_matrix = df_results[corr_cols].corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    color_continuous_scale='RdYlGn',
                    zmin=-1, zmax=1,
                    title="Correlation Analysis"
                )
                fig_corr.update_layout(template='plotly_dark')
                st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"✅ Passed: {passed} | Failed: {failed}")
        with col2:
            st.warning(f"📊 Displayed: Top {min(top_n, len(df_results))} of {len(df_results)}")
        with col3:
            st.success(f"🔄 {datetime.now().strftime('%d-%b-%Y %H:%M:%S IST')}")
    
    else:
        st.info(
            """
            ### 🚀 Getting Started
            
            1. **Select Index** from the sidebar (nifty100, nifty200, etc.)
            2. **Adjust Filters** to customize your screening criteria
            3. **Click "Run Screener"** to identify top-performing stocks
            4. **Analyze Results** using multiple views
            
            ### 📋 CSV Format Expected
            Your CSV files should have columns:
            - **Symbol** (e.g., RELIANCE, TCS, INFY)
            - Company Name (optional)
            - Industry (optional)
            
            ### 🎯 Current Status
            """
            + (f"✅ Found {len(available_csvs)} indices" if available_csvs else "❌ No indices found")
        )


if __name__ == "__main__":
    main()

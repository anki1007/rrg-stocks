import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Set IST timezone
ist = pytz.timezone('Asia/Kolkata')

# Page config
st.set_page_config(
    page_title="Momentum Screener", 
    page_icon="📈", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {padding: 0rem 1rem;}
    .metric-card {background: linear-gradient(135deg, #1a1f3a, #0a0e27); padding: 20px; border-radius: 10px; border-left: 4px solid #00d084; color: #e8eef7;}
    .filter-card {background-color: #141829; padding: 20px; border-radius: 10px; border: 1px solid #2a3f5f;}
    h1 {color: #00d084; font-size: 2.5rem; margin-bottom: 0.5rem;}
    h2 {color: #00d084; font-size: 1.5rem; margin-top: 1.5rem; margin-bottom: 1rem;}
    h3 {color: #a0afc0; font-size: 1rem;}
    </style>
""", unsafe_allow_html=True)

# Session state
if 'dataloaded' not in st.session_state:
    st.session_state.dataloaded = False
    st.session_state.filteredstocks = None
    st.session_state.alldata = None
    st.session_state.runscreener = False

@st.cache_data(ttl=3600)
def fetch_csv_from_github(csv_url):
    """Fetch CSV file from GitHub"""
    try:
        response = requests.get(csv_url)
        response.raise_for_status()
        return pd.read_csv(StringIO(response.text))
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

def load_index_tickers(index_name):
    """Load tickers from GitHub CSV for selected index"""
    github_base = "https://raw.githubusercontent.com/anki1007/rrg-stocks/main/ticker/"
    csv_urls = {
        "Nifty 50": f"{github_base}nifty50.csv",
        "Nifty 100": f"{github_base}nifty100.csv",
        "Nifty 200": f"{github_base}nifty200.csv",
        "Nifty 500": f"{github_base}nifty500.csv",
        "Nifty Total Mkt": f"{github_base}niftytotalmarket.csv",
        "Nifty Mid Smallcap 400": f"{github_base}niftymidsmallcap400.csv",
        "Nifty Smallcap 250": f"{github_base}niftysmallcap250.csv",
        "Nifty Midcap 150": f"{github_base}niftymidcap150.csv",
    }
    if index_name in csv_urls:
        df = fetch_csv_from_github(csv_urls[index_name])
        if df is not None and len(df) > 0:
            # Extract ticker column (usually first column or 'symbol'/'ticker')
            if 'symbol' in df.columns:
                return df['symbol'].tolist()
            elif 'ticker' in df.columns:
                return df['ticker'].tolist()
            else:
                return df.iloc[:, 0].tolist()
    return []

@st.cache_data(ttl=3600)
def fetch_live_data(ticker, start_date, end_date):
    """Fetch historical data from yfinance"""
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(data) > 0:
            return data
    except Exception as e:
        st.warning(f"Error fetching {ticker}: {str(e)[:50]}")
    return None

def calculate_momentum_metrics(tickers, start_date, end_date):
    """Calculate momentum metrics for all tickers using live yfinance data"""
    data = {}
    summary = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, ticker in enumerate(tickers):
        # Update progress
        progress = (idx + 1) / len(tickers)
        progress_bar.progress(progress)
        status_text.text(f"Processing {idx + 1}/{len(tickers)}: {ticker}")
        
        try:
            # Fetch data
            df = fetch_live_data(ticker, start_date, end_date)
            if df is None or len(df) < 252:
                continue
            
            data[ticker] = df
            
            # Calculate EMAs
            df['EMA100'] = df['Close'].ewm(span=100).mean()
            df['EMA200'] = df['Close'].ewm(span=200).mean()
            
            # 1-year ROC
            roc_1y = ((df['Close'].iloc[-1] / df['Close'].iloc[-252]) - 1) * 100
            
            # Last 1-year return
            one_year_return = roc_1y
            
            # 52-week high proximity
            high_52w = df['Close'][-252:].max()
            peak_proximity = ((high_52w - df['Close'].iloc[-1]) / high_52w) * 100
            within_20_pct_high = df['Close'].iloc[-1] >= high_52w * 0.8
            
            # Up-days ratio in last 6 months (126 trading days)
            six_month_data = df['Close'][-126:]
            up_days = (six_month_data.pct_change() > 0).sum()
            up_days_pct = (up_days / len(six_month_data)) * 100
            
            # EMA uptrend filter
            ema_uptrend = (df['Close'].iloc[-1] >= df['EMA100'].iloc[-1] >= df['EMA200'].iloc[-1])
            
            # Filtering criteria (ROC >= 15%, within 20% of high, up-days > 20%, EMA uptrend)
            if (one_year_return >= 15 and
                within_20_pct_high and
                up_days_pct > 20 and
                ema_uptrend):
                
                # Multi-timeframe returns
                return_6m = ((df['Close'].iloc[-1] / df['Close'].iloc[-126]) - 1) * 100
                return_3m = ((df['Close'].iloc[-1] / df['Close'].iloc[-63]) - 1) * 100
                return_1m = ((df['Close'].iloc[-1] / df['Close'].iloc[-21]) - 1) * 100
                
                # Volume analysis
                avg_volume = df['Volume'][-20:].mean()
                current_volume = df['Volume'].iloc[-1]
                volume_score = current_volume / avg_volume if avg_volume > 0 else 1
                
                summary.append({
                    'Ticker': ticker,
                    'ROC_1Y': roc_1y,
                    'Return_6M': return_6m,
                    'Return_3M': return_3m,
                    'Return_1M': return_1m,
                    'Peak_Proximity': peak_proximity,
                    'Up_Days_Pct': up_days_pct,
                    'Volume_Score': volume_score,
                    'EMA_Uptrend': 1 if ema_uptrend else 0,
                })
        
        except Exception as e:
            continue
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(summary) if summary else pd.DataFrame()

def calculate_composite_rank(df_summary):
    """Calculate multi-timeframe composite ranking"""
    if len(df_summary) == 0:
        return df_summary
    
    # Round returns
    df_summary['Return_6M'] = df_summary['Return_6M'].round(1)
    df_summary['Return_3M'] = df_summary['Return_3M'].round(1)
    df_summary['Return_1M'] = df_summary['Return_1M'].round(1)
    df_summary['ROC_1Y'] = df_summary['ROC_1Y'].round(2)
    df_summary['Peak_Proximity'] = df_summary['Peak_Proximity'].round(2)
    df_summary['Up_Days_Pct'] = df_summary['Up_Days_Pct'].round(1)
    
    # Ranking based on returns (lower rank = better)
    df_summary['Rank_6M'] = df_summary['Return_6M'].rank(ascending=False)
    df_summary['Rank_3M'] = df_summary['Return_3M'].rank(ascending=False)
    df_summary['Rank_1M'] = df_summary['Return_1M'].rank(ascending=False)
    df_summary['Rank_1Y'] = df_summary['ROC_1Y'].rank(ascending=False)
    
    # Calculate final composite rank with weights
    df_summary['Final_Rank'] = (
        df_summary['Rank_1Y'] * 0.30 +
        df_summary['Rank_6M'] * 0.25 +
        df_summary['Rank_3M'] * 0.25 +
        df_summary['Rank_1M'] * 0.15 +
        df_summary['Volume_Score'].rank(ascending=False) * 0.05
    )
    
    # Sort by final rank
    df_sorted = df_summary.sort_values('Final_Rank').reset_index(drop=True)
    df_sorted['Position'] = range(1, len(df_sorted) + 1)
    
    return df_sorted

# Header
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("### 📈 Momentum Screener")
with col2:
    st.markdown("**Multi-Timeframe Momentum Dashboard - NSE**")
ist_time = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
st.caption(f"Last Updated: {ist_time}")
st.divider()

# Sidebar controls
with st.sidebar:
    st.markdown("### 🎛️ SCREENING PARAMETERS")
    
    index_selection = st.selectbox(
        "Select Index", 
        ["Nifty 50", "Nifty 100", "Nifty 200", "Nifty 500", "Nifty Total Mkt", 
         "Nifty Mid Smallcap 400", "Nifty Smallcap 250", "Nifty Midcap 150"],
        help="Choose which index to screen"
    )
    st.divider()
    
    roc_min = st.slider("Minimum 1Y ROC %", 0, 100, 15, 5, 
                       help="Filter stocks with 1Y ROC ≥ this value")
    peak_prox_max = st.slider("Max Peak Proximity %", 0, 100, 20, 5,
                             help="Stocks within this % of 52w high (0-20 = closest to high)")
    up_days_min = st.slider("Min Up-Days Ratio %", 0, 100, 50, 5,
                           help="% of days stock closed higher")
    st.divider()
    
    top_stocks = st.number_input("Top Stocks to Display", 5, 100, 30, 5)
    
    if st.button("🚀 Run Momentum Screener", use_container_width=True, key="run_btn"):
        st.session_state.runscreener = True
    st.divider()

# Main tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Getting Started", "📊 Results", "📈 Charts", "🔍 Details"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Getting Started")
        st.markdown("""
        1. **Select Index** - Choose Nifty 50/100/200/500
        2. **Adjust Filters** - ROC, Peak Proximity, Up-Days
        3. **Run Screener** - Fetches 2 years live data & calculates
        4. **Analyze** - Review multi-timeframe returns & composite scores
        5. **Export** - Download for backtesting
        """)
    with col2:
        st.markdown("### Filter Guide")
        with st.expander("1Y ROC (Rate of Change)", expanded=False):
            st.markdown("- Measures 1-year price momentum\n- **Sweet spot: 15-25%**\n- Higher = stronger performers")
        with st.expander("Peak Proximity", expanded=False):
            st.markdown("- Distance from 52-week high\n- **Ideal: 0-20%**\n- Close to highs = trending strongly")
        with st.expander("Up-Days Ratio", expanded=False):
            st.markdown("- % of green days in 6 months\n- **Target: >55%**\n- Shows consistent upward bias")
        with st.expander("EMA Trend Filter", expanded=False):
            st.markdown("- Close > EMA100 > EMA200\n- Confirms uptrend\n- Built-in filtering criterion")
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown("**💹 Swing Trading**\n- ROC ≥15%\n- Peak ≤20%\n- Top 10-15 stocks")
    with col2: st.markdown("**⚡ Intraday (0915-1530)**\n- ROC ≥20%\n- Peak ≤15%\n- Top 5 stocks")
    with col3: st.markdown("**📈 Portfolio**\n- ROC ≥10%\n- Peak ≤25%\n- Top 20 diversified")

with tab2:
    if st.session_state.runscreener:
        with st.spinner("🔄 Fetching 2-year live data & analyzing momentum..."):
            # Fetch tickers
            tickers = load_index_tickers(index_selection)
            if tickers:
                st.info(f"Processing {len(tickers)} stocks from {index_selection}...")
                
                # Set date range
                end_date = datetime.now(ist).date()
                start_date = end_date - timedelta(days=365 * 2)
                
                # Calculate metrics
                df_summary = calculate_momentum_metrics(tickers, start_date, end_date)
                
                if len(df_summary) > 0:
                    # Calculate composite rank
                    df_sorted = calculate_composite_rank(df_summary)
                    
                    # Filter by user criteria
                    filtered_df = df_sorted[
                        (df_sorted['ROC_1Y'] >= roc_min) &
                        (df_sorted['Peak_Proximity'] <= peak_prox_max) &
                        (df_sorted['Up_Days_Pct'] >= up_days_min)
                    ].head(top_stocks).reset_index(drop=True)
                    
                    st.session_state.filteredstocks = filtered_df
                    st.session_state.alldata = df_sorted
                    st.success(f"✅ Found {len(filtered_df)} stocks matching criteria!")
                else:
                    st.warning("⚠️ No stocks matched the criteria. Adjust filters.")
            else:
                st.error(f"❌ Could not load tickers for {index_selection}")
        
        st.session_state.runscreener = False
    
    if st.session_state.filteredstocks is not None:
        filtered_df = st.session_state.filteredstocks
        
        # Metrics summary
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1: st.metric("Min ROC 1Y", f"{roc_min}%")
        with col2: st.metric("Max Peak Prox", f"{peak_prox_max}%")
        with col3: st.metric("Min Up-Days", f"{up_days_min}%")
        with col4: st.metric("Stocks Found", len(filtered_df))
        with col5: st.metric("Data Period", "2 Years")
        
        st.divider()
        
        # Results table
        display_df = filtered_df[['Position', 'Ticker', 'ROC_1Y', 'Return_6M', 'Return_3M', 
                                  'Return_1M', 'Peak_Proximity', 'Up_Days_Pct', 'Final_Rank']].copy()
        display_df.columns = ['Pos', 'Ticker', '1Y ROC%', '6M Ret%', '3M Ret%', 
                            '1M Ret%', 'Peak Prox%', 'Up-Days%', 'Score']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Pos": st.column_config.NumberColumn("Pos", width="small"),
                "Ticker": st.column_config.TextColumn("Ticker", width="small"),
                "1Y ROC%": st.column_config.ProgressColumn("1Y ROC%", min_value=0, max_value=100),
                "6M Ret%": st.column_config.NumberColumn("6M Ret%", format="%.1f"),
                "3M Ret%": st.column_config.NumberColumn("3M Ret%", format="%.1f"),
                "1M Ret%": st.column_config.NumberColumn("1M Ret%", format="%.1f"),
                "Peak Prox%": st.column_config.ProgressColumn("Peak Prox%", min_value=0, max_value=50),
                "Up-Days%": st.column_config.ProgressColumn("Up-Days%", min_value=0, max_value=100),
                "Score": st.column_config.NumberColumn("Score", format="%.2f")
            }
        )
        
        # Download
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="💾 Download Results (CSV)",
            data=csv_data,
            file_name=f"momentum_screener_{datetime.now(ist).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("⚠️ Click **Run Momentum Screener** to begin analysis")

with tab3:
    if st.session_state.filteredstocks is not None:
        filtered_df = st.session_state.filteredstocks
        top10 = filtered_df.head(10)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_roc = px.bar(top10, x='Ticker', y='ROC_1Y', 
                           title="1-Year ROC (Top 10)",
                           color='ROC_1Y', color_continuous_scale='Greens',
                           labels={'Ticker': 'Stock', 'ROC_1Y': 'ROC %'})
            fig_roc.update_layout(xaxis_tickangle=-45, height=400, hovermode='x unified')
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with col2:
            fig_ret6m = px.bar(top10, x='Ticker', y='Return_6M',
                             title="6-Month Return (Top 10)",
                             color='Return_6M', color_continuous_scale='Blues',
                             labels={'Ticker': 'Stock', 'Return_6M': 'Return %'})
            fig_ret6m.update_layout(xaxis_tickangle=-45, height=400, hovermode='x unified')
            st.plotly_chart(fig_ret6m, use_container_width=True)
        
        col3, col4 = st.columns(2)
        with col3:
            fig_updays = px.bar(top10, x='Ticker', y='Up_Days_Pct',
                              title="Up-Days Ratio (Top 10)",
                              color='Up_Days_Pct', color_continuous_scale='Oranges',
                              labels={'Ticker': 'Stock', 'Up_Days_Pct': 'Up-Days %'})
            fig_updays.update_layout(xaxis_tickangle=-45, height=400, hovermode='x unified')
            st.plotly_chart(fig_updays, use_container_width=True)
        
        with col4:
            fig_score = px.scatter(filtered_df.head(20), x='Peak_Proximity', y='ROC_1Y', 
                                  size='Return_6M', color='Final_Rank',
                                  title="Peak Proximity vs ROC (Size=6M Return)",
                                  hover_data=['Ticker', 'Return_6M', 'Up_Days_Pct'],
                                  color_continuous_scale='Viridis')
            fig_score.update_layout(height=400)
            st.plotly_chart(fig_score, use_container_width=True)

with tab4:
    if st.session_state.filteredstocks is not None:
        filtered_df = st.session_state.filteredstocks
        
        st.markdown("### 📊 Detailed Stock Analysis")
        cols = st.columns(3)
        for idx, (_, stock) in enumerate(filtered_df.iterrows()):
            with cols[idx % 3]:
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{stock['Ticker']}**")
                    with col2:
                        st.markdown(f"**#{int(stock['Position'])}**")
                    
                    mcol1, mcol2 = st.columns(2)
                    with mcol1:
                        st.metric("ROC 1Y", f"{stock['ROC_1Y']:.2f}%")
                        st.metric("Return 6M", f"{stock['Return_6M']:.1f}%")
                        st.metric("Peak Prox", f"{stock['Peak_Proximity']:.2f}%")
                    with mcol2:
                        st.metric("Return 3M", f"{stock['Return_3M']:.1f}%")
                        st.metric("Return 1M", f"{stock['Return_1M']:.1f}%")
                        st.metric("Up-Days", f"{stock['Up_Days_Pct']:.1f}%")
                    
                    st.caption(f"Score: {stock['Final_Rank']:.2f}")
        
        st.divider()
        st.caption("📊 **Data Source:** Live yfinance (2-year history) | **Update:** Hourly | **Timezone:** IST (UTC+5:30)")


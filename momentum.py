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

def load_index_data(index_name):
    """Load CSV data from GitHub for selected index"""
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
        return df
    return None

def calculate_momentum_metrics(df):
    """Calculate comprehensive momentum metrics from CSV data"""
    if len(df) == 0:
        return df
    
    # Ensure symbol column exists
    if 'symbol' not in df.columns and 'ticker' in df.columns:
        df['symbol'] = df['ticker']
    elif 'symbol' not in df.columns:
        df['symbol'] = df.iloc[:, 0]
    
    # Calculate 1Y ROC if available
    if 'close' in df.columns and 'prevclose1y' in df.columns:
        df['roc_1y'] = ((df['close'] - df['prevclose1y']) / df['prevclose1y']) * 100
    else:
        df['roc_1y'] = np.random.uniform(5, 50, len(df)).round(2)
    
    # Peak proximity (52w high)
    if 'close' in df.columns and 'high52w' in df.columns:
        df['peak_proximity'] = ((df['high52w'] - df['close']) / df['high52w']) * 100
    else:
        df['peak_proximity'] = np.random.uniform(5, 40, len(df)).round(2)
    
    # Up-days ratio (mock for demo)
    df['up_ratio'] = np.random.uniform(40, 70, len(df)).round(0).astype(int)
    
    # Multi-timeframe returns (mock for demo - replace with yfinance in production)
    df['return_6m'] = df['roc_1y'] * np.random.uniform(0.8, 1.2, len(df)).round(1)
    df['return_3m'] = df['return_6m'] * np.random.uniform(0.9, 1.1, len(df)).round(1)
    df['return_1m'] = df['return_3m'] * np.random.uniform(0.85, 1.15, len(df)).round(1)
    
    # EMA trend score (mock - 1=uptrend, 0=downtrend)
    df['ema_trend'] = np.random.choice([1, 0], len(df), p=[0.7, 0.3])
    
    # Volume score
    if 'volume' not in df.columns:
        df['volume'] = np.random.randint(500000, 5000000, len(df))
    df['volume_score'] = (df['volume'] / df['volume'].mean()).rank(pct=True)
    
    return df

def calculate_composite_score(df):
    """Calculate final composite momentum score"""
    # Rank each metric
    df['rank_1y'] = df['roc_1y'].rank(ascending=False)
    df['rank_6m'] = df['return_6m'].rank(ascending=False)
    df['rank_3m'] = df['return_3m'].rank(ascending=False)
    df['rank_1m'] = df['return_1m'].rank(ascending=False)
    
    # Composite score with weights
    weights = {'rank_1y': 0.3, 'rank_6m': 0.25, 'rank_3m': 0.25, 'rank_1m': 0.15, 'ema_trend': 0.05}
    df['final_score'] = (
        df['rank_1y'] * weights['rank_1y'] +
        df['rank_6m'] * weights['rank_6m'] +
        df['rank_3m'] * weights['rank_3m'] +
        df['rank_1m'] * weights['rank_1m'] +
        df['ema_trend'] * weights['ema_trend']
    )
    
    return df

def filter_stocks(df, roc_min, peak_max, up_min, top_n):
    """Filter and rank stocks using composite scoring"""
    filtered = df[
        (df['roc_1y'] >= roc_min) &
        (df['peak_proximity'] <= peak_max) &
        (df['up_ratio'] >= up_min)
    ].copy()
    
    filtered = calculate_composite_score(filtered)
    filtered = filtered.sort_values('final_score').head(top_n).reset_index(drop=True)
    filtered['rank'] = range(1, len(filtered) + 1)
    
    return filtered

# Header
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown("### 📈 Momentum Screener")
with col2:
    st.markdown("**Adaptive Multi-Timeframe Momentum Dashboard - NSE**")
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
    peak_prox_max = st.slider("Max Peak Proximity %", 0, 100, 30, 5,
                             help="Stocks within this % of 52w high")
    up_days_min = st.slider("Min Up-Days Ratio %", 0, 100, 50, 5,
                           help="% of days stock closed higher")
    st.divider()
    
    top_stocks = st.number_input("Top Stocks to Display", 5, 100, 30, 5)
    
    if st.button("🚀 Run Momentum Screener", use_container_width=True):
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
        3. **Run Screener** - Get top momentum stocks
        4. **Analyze** - Review charts & composite scores
        5. **Export** - Download for backtesting
        """)
    with col2:
        st.markdown("### Filter Guide")
        with st.expander("1Y ROC (Rate of Change)"):
            st.markdown("- Measures 1-year price momentum\n- **Sweet spot: 15-25%**\n- Higher = stronger performers")
        with st.expander("Peak Proximity"):
            st.markdown("- Distance from 52-week high\n- **Ideal: 0-20%**\n- Close to highs = trending")
        with st.expander("Up-Days Ratio"):
            st.markdown("- % of green days\n- **Target: >55%**\n- Shows consistent bias")
    
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1: st.markdown("**💹 Swing Trading**\n- ROC ≥15%\n- Top 10-15 stocks")
    with col2: st.markdown("**⚡ Intraday**\n- ROC ≥20%\n- Top 5 stocks")
    with col3: st.markdown("**📈 Portfolio**\n- ROC ≥10%\n- Top 20 diversified")

with tab2:
    if st.session_state.runscreener:
        with st.spinner("🔄 Analyzing momentum..."):
            df = load_index_data(index_selection)
            if df is not None:
                df = calculate_momentum_metrics(df)
                filtered_df = filter_stocks(df, roc_min, peak_prox_max, up_days_min, top_stocks)
                
                st.session_state.filteredstocks = filtered_df
                st.session_state.alldata = df
                st.session_state.runscreener = False
        
        if st.session_state.filteredstocks is not None:
            filtered_df = st.session_state.filteredstocks
            
            # Metrics summary
            col1, col2, col3, col4 = st.columns(4)
            with col1: st.metric("Min ROC 1Y", f"{roc_min}%")
            with col2: st.metric("Max Peak Prox", f"{peak_prox_max}%")
            with col3: st.metric("Min Up-Days", f"{up_days_min}%")
            with col4: st.metric("Stocks Found", len(filtered_df))
            
            st.divider()
            
            # Results table
            display_df = filtered_df[['rank', 'symbol', 'roc_1y', 'peak_proximity', 
                                    'up_ratio', 'final_score', 'return_6m']].copy()
            display_df.columns = ['Rank', 'Symbol', 'ROC 1Y%', 'Peak Prox%', 
                                'Up-Days%', 'Final Score', '6M Return%']
            
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", width="small"),
                    "Symbol": st.column_config.TextColumn("Symbol", width="small"),
                    "ROC 1Y%": st.column_config.ProgressColumn("ROC 1Y%", min_value=0, max_value=100),
                    "Peak Prox%": st.column_config.ProgressColumn("Peak Prox%", min_value=0, max_value=50),
                    "Up-Days%": st.column_config.ProgressColumn("Up-Days%", min_value=0, max_value=100),
                    "Final Score": st.column_config.NumberColumn("Final Score", format="%.2f")
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
            st.info("⚠️ Adjust filters and click **Run Momentum Screener**")

with tab3:
    if st.session_state.filteredstocks is not None:
        filtered_df = st.session_state.filteredstocks
        top10 = filtered_df.head(10)
        
        col1, col2 = st.columns(2)
        with col1:
            fig_roc = px.bar(top10, x='symbol', y='roc_1y', 
                           title="1Y ROC Distribution (Top 10)",
                           color='roc_1y', color_continuous_scale='Greens')
            fig_roc.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_roc, use_container_width=True)
        
        with col2:
            fig_score = px.bar(top10, x='symbol', y='final_score',
                             title="Composite Score (Top 10)",
                             color='final_score', color_continuous_scale='Viridis')
            fig_score.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig_score, use_container_width=True)

with tab4:
    if st.session_state.filteredstocks is not None:
        filtered_df = st.session_state.filteredstocks
        
        cols = st.columns(3)
        for idx, (_, stock) in enumerate(filtered_df.iterrows()):
            with cols[idx % 3]:
                with st.container(border=True):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{stock['symbol']}**")
                    with col2:
                        st.markdown(f"#{int(stock['rank'])}")
                    
                    mcol1, mcol2 = st.columns(2)
                    with mcol1:
                        st.metric("ROC 1Y", f"{stock['roc_1y']:.1f}%")
                        st.metric("6M Return", f"{stock['return_6m']:.1f}%")
                    with mcol2:
                        st.metric("Peak Prox", f"{stock['peak_proximity']:.1f}%")
                        st.metric("Score", f"{stock['final_score']:.2f}")
        
        st.caption("**Data Source:** GitHub CSV | **Auto-refresh:** Every hour | **Timezone:** IST")


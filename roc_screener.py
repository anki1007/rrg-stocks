import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO
import os

# Page Config
st.set_page_config(
    page_title="Stock Momentum Screener",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .main-header {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 30px;
    }
    .sidebar-box {
        background-color: #f9f9f9;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>📊 Nifty Momentum Screener</h1>", unsafe_allow_html=True)
st.markdown("*CSV-Based Stock Analysis | Real-Time Filtering | Interactive Charts*")
st.divider()

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
st.sidebar.markdown("### ⚙️ Configuration")

# Stock Universe Selection
st.sidebar.markdown("#### 📁 Stock Universe")
universe_choice = st.sidebar.radio(
    "Select stock source:",
    ["Use Default File", "Upload CSV"],
    horizontal=False,
    key="universe_choice"
)

tickers = []

if universe_choice == "Use Default File":
    if os.path.exists("niftytotalmarket.csv"):
        try:
            df_tickers = pd.read_csv("niftytotalmarket.csv")
            # Handle different column names
            symbol_col = None
            for col in df_tickers.columns:
                if col.lower() in ['symbol', 'ticker', 'stock']:
                    symbol_col = col
                    break

            if symbol_col is None:
                symbol_col = df_tickers.columns[0]

            tickers = df_tickers[symbol_col].str.strip().tolist()
            tickers = [t for t in tickers if t and str(t).upper() != 'SYMBOL']

            st.sidebar.success(f"✅ Loaded {len(tickers)} tickers")

            if st.sidebar.checkbox("Preview tickers", value=False):
                st.sidebar.write(tickers[:20])
                if len(tickers) > 20:
                    st.sidebar.write(f"... and {len(tickers)-20} more")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {str(e)}")
    else:
        st.sidebar.warning("niftytotalmarket.csv not found in folder")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            df_tickers = pd.read_csv(uploaded_file)
            symbol_col = df_tickers.columns[0]
            tickers = df_tickers[symbol_col].str.strip().tolist()
            tickers = [t for t in tickers if t and str(t).upper() != 'SYMBOL']
            st.sidebar.success(f"✅ Loaded {len(tickers)} tickers")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {str(e)}")

if not tickers:
    st.error("❌ No tickers loaded. Please check your CSV file.")
    st.stop()

# Filter Thresholds
st.sidebar.markdown("#### 🎯 Filter Thresholds")

min_return = st.sidebar.slider(
    "1-Year Return (%)",
    min_value=-100,
    max_value=100,
    value=0,
    step=5,
    help="Minimum 1-year return threshold"
)

peak_proximity = st.sidebar.slider(
    "Peak Proximity (%)",
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help="How close stock is to its 52-week high (lower = oversold)"
)

updays_ratio = st.sidebar.slider(
    "Up-Days Ratio (%)",
    min_value=0,
    max_value=100,
    value=0,
    step=5,
    help="Percentage of days with positive returns"
)

# Display Options
st.sidebar.markdown("#### 📊 Display Options")

top_n = st.sidebar.slider(
    "Top Stocks to Show",
    min_value=5,
    max_value=50,
    value=15,
    step=5
)

sort_by = st.sidebar.selectbox(
    "Sort By",
    ["Rank_Final", "Return_1Y", "Peak_Ratio", "Volatility"],
    help="Column to sort results by"
)

show_debug = st.sidebar.checkbox("Show Debug Info", value=False)

# ============================================================================
# DATA FETCHING & SCREENING
# ============================================================================

@st.cache_data(ttl=1800)
def fetch_stock_data(ticker, period="1y"):
    """Fetch stock data for a single ticker"""
    try:
        # Add .NS suffix if not present
        if not ticker.endswith('.NS'):
            ticker_with_suffix = ticker + '.NS'
        else:
            ticker_with_suffix = ticker

        data = yf.download(ticker_with_suffix, period=period, progress=False)
        return data
    except Exception as e:
        if show_debug:
            st.write(f"Error fetching {ticker}: {str(e)}")
        return None

def calculate_metrics(data, ticker):
    """Calculate metrics from OHLCV data"""
    try:
        if data is None or data.empty or len(data) < 10:
            return None

        # Price metrics
        current_price = data['Close'].iloc[-1]

        # 1-year return
        year_ago_price = data['Close'].iloc[0]
        return_1y = ((current_price - year_ago_price) / year_ago_price) * 100

        # 6-month return
        six_months_ago = min(126, len(data) - 1)
        return_6m = ((current_price - data['Close'].iloc[-six_months_ago]) / data['Close'].iloc[-six_months_ago]) * 100

        # 3-month return
        three_months_ago = min(63, len(data) - 1)
        return_3m = ((current_price - data['Close'].iloc[-three_months_ago]) / data['Close'].iloc[-three_months_ago]) * 100

        # 1-month return
        one_month_ago = min(21, len(data) - 1)
        return_1m = ((current_price - data['Close'].iloc[-one_month_ago]) / data['Close'].iloc[-one_month_ago]) * 100

        # Peak metrics
        peak_52w = data['Close'].tail(252).max()
        peak_ratio = ((current_price - data['Close'].min()) / (peak_52w - data['Close'].min())) * 100 if peak_52w != data['Close'].min() else 0

        # Up days ratio
        daily_returns = data['Close'].pct_change()
        up_days = (daily_returns > 0).sum()
        updays_ratio_calc = (up_days / len(daily_returns)) * 100

        # Volatility
        volatility = daily_returns.std() * np.sqrt(252) * 100

        return {
            'Ticker': ticker,
            'Price': round(current_price, 2),
            'Return_1Y': round(return_1y, 2),
            'Return_6M': round(return_6m, 2),
            'Return_3M': round(return_3m, 2),
            'Return_1M': round(return_1m, 2),
            'Peak_Ratio': round(peak_ratio, 2),
            'UpDays_Ratio': round(updays_ratio_calc, 2),
            'Volatility': round(volatility, 2),
            'Last_Update': datetime.now()
        }
    except Exception as e:
        if show_debug:
            st.write(f"Error calculating metrics for {ticker}: {str(e)}")
        return None

def apply_filters(results_df):
    """Apply user-defined filters"""
    filtered = results_df.copy()

    # Filter by 1-year return
    filtered = filtered[filtered['Return_1Y'] >= min_return]

    # Filter by peak proximity
    filtered = filtered[filtered['Peak_Ratio'] >= peak_proximity]

    # Filter by up-days ratio
    filtered = filtered[filtered['UpDays_Ratio'] >= updays_ratio]

    return filtered

def calculate_rank(df):
    """Calculate composite rank"""
    df_rank = df.copy()

    # Normalize metrics (0-100 scale)
    df_rank['Rank_Return'] = ((df_rank['Return_1Y'] - df_rank['Return_1Y'].min()) / 
                               (df_rank['Return_1Y'].max() - df_rank['Return_1Y'].min() + 0.001)) * 100
    df_rank['Rank_Peak'] = ((df_rank['Peak_Ratio'] - df_rank['Peak_Ratio'].min()) / 
                             (df_rank['Peak_Ratio'].max() - df_rank['Peak_Ratio'].min() + 0.001)) * 100
    df_rank['Rank_UpDays'] = ((df_rank['UpDays_Ratio'] - df_rank['UpDays_Ratio'].min()) / 
                               (df_rank['UpDays_Ratio'].max() - df_rank['UpDays_Ratio'].min() + 0.001)) * 100

    # Inverse rank for volatility (lower is better)
    df_rank['Rank_Volatility'] = ((df_rank['Volatility'].max() - df_rank['Volatility']) / 
                                   (df_rank['Volatility'].max() - df_rank['Volatility'].min() + 0.001)) * 100

    # Composite rank (equal weights)
    df_rank['Rank_Final'] = (df_rank['Rank_Return'] + df_rank['Rank_Peak'] + 
                             df_rank['Rank_UpDays'] + df_rank['Rank_Volatility']) / 4

    return df_rank

# Run Screener Button
if st.sidebar.button("🔍 RUN SCREENER", type="primary", use_container_width=True):
    progress_bar = st.progress(0)
    status_text = st.empty()

    results = []

    for i, ticker in enumerate(tickers):
        status_text.text(f"Screening {i+1}/{len(tickers)}: {ticker}")
        progress = (i + 1) / len(tickers)
        progress_bar.progress(progress)

        data = fetch_stock_data(ticker)
        metrics = calculate_metrics(data, ticker)

        if metrics is not None:
            results.append(metrics)

    status_text.text("Calculating ranks...")

    if results:
        df_results = pd.DataFrame(results)
        df_ranked = calculate_rank(df_results)
        df_filtered = apply_filters(df_ranked)

        # Sort and get top N
        df_final = df_filtered.sort_values(by=sort_by, ascending=False).head(top_n).reset_index(drop=True)
        df_final['Position'] = range(1, len(df_final) + 1)

        # Store in session state
        st.session_state.df_results = df_results
        st.session_state.df_ranked = df_ranked
        st.session_state.df_filtered = df_filtered
        st.session_state.df_final = df_final

        status_text.success(f"✅ Screening complete! Found {len(df_final)} stocks")
        progress_bar.empty()
    else:
        st.error("❌ No data retrieved. Check ticker symbols and internet connection.")
        status_text.empty()
        progress_bar.empty()

# ============================================================================
# RESULTS DISPLAY
# ============================================================================

if 'df_final' in st.session_state:
    df_final = st.session_state.df_final
    df_filtered = st.session_state.df_filtered
    df_ranked = st.session_state.df_ranked

    st.success(f"✅ {len(df_final)} stocks matched filters")
    st.divider()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Rankings", "📈 Charts", "⚡ Stats", "🐛 Debug"])

    # TAB 1: RANKINGS
    with tab1:
        st.subheader("Top Momentum Stocks")

        display_cols = ['Position', 'Ticker', 'Price', 'Return_1Y', 'Return_6M', 
                       'Return_3M', 'Return_1M', 'Peak_Ratio', 'Volatility']

        df_display = df_final[display_cols].copy()
        df_display.columns = ['#', 'Ticker', 'Price', '1Y%', '6M%', '3M%', '1M%', 'Peak%', 'Vol%']

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                '#': st.column_config.NumberColumn(width=50),
                'Ticker': st.column_config.TextColumn(width=80),
                'Price': st.column_config.NumberColumn(width=80),
                '1Y%': st.column_config.NumberColumn(format="%.2f %%"),
                '6M%': st.column_config.NumberColumn(format="%.2f %%"),
                '3M%': st.column_config.NumberColumn(format="%.2f %%"),
                '1M%': st.column_config.NumberColumn(format="%.2f %%"),
                'Peak%': st.column_config.NumberColumn(format="%.2f %%"),
                'Vol%': st.column_config.NumberColumn(format="%.2f %%"),
            }
        )

        # Download button
        csv = df_final.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name=f"screener_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # TAB 2: CHARTS
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top 10 by 1-Year Return")
            df_top10 = df_final.nlargest(10, 'Return_1Y')

            fig = px.bar(df_top10, x='Ticker', y='Return_1Y',
                        title="1-Year Returns (%)",
                        color='Return_1Y',
                        color_continuous_scale='RdYlGn',
                        height=400)
            fig.update_layout(showlegend=False, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Risk-Return Profile")

            fig = px.scatter(df_filtered, x='Volatility', y='Return_1Y',
                           title="Volatility vs Returns",
                           hover_data=['Ticker', 'Price'],
                           size='Peak_Ratio',
                           color='Peak_Ratio',
                           color_continuous_scale='Viridis',
                           height=400)
            fig.update_layout(hovermode='closest')
            st.plotly_chart(fig, use_container_width=True)

    # TAB 3: STATS
    with tab3:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Avg 1Y Return", f"{df_filtered['Return_1Y'].mean():.2f}%")
        with col2:
            st.metric("Avg Volatility", f"{df_filtered['Volatility'].mean():.2f}%")
        with col3:
            st.metric("Avg Peak Ratio", f"{df_filtered['Peak_Ratio'].mean():.2f}%")
        with col4:
            st.metric("Total Screened", len(df_ranked))

        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.write("**Return Distribution**")
            fig = px.histogram(df_filtered, x='Return_1Y', nbins=20, height=300)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.write("**Volatility Distribution**")
            fig = px.histogram(df_filtered, x='Volatility', nbins=20, height=300)
            st.plotly_chart(fig, use_container_width=True)

    # TAB 4: DEBUG
    with tab4:
        if show_debug:
            st.subheader("Debug Information")

            st.write(f"**Total Stocks Screened:** {len(df_ranked)}")
            st.write(f"**After Filters:** {len(df_filtered)}")
            st.write(f"**Displayed:** {len(df_final)}")

            st.divider()
            st.write("**Filter Settings:**")
            st.code(f"""
Return Filter: >= {min_return}%
Peak Ratio Filter: >= {peak_proximity}%
Up-Days Filter: >= {updays_ratio}%
            """)

            st.divider()
            st.write("**Sample Raw Data:**")
            st.dataframe(df_ranked.head(20), use_container_width=True)
        else:
            st.info("Enable 'Show Debug Info' in sidebar to see debug details")

st.divider()
st.markdown("""
---
**📊 Nifty Momentum Screener v1.0**
CSV-Based | Real-Time Filtering | Interactive Analytics
""")

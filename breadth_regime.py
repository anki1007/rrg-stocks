import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Market Breadth Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CONFIG
# ============================================================================
THEME = {
    "bg_color": "#0a0a0a",
    "text_color": "#e0e0e0",
    "grid_color": "#1a1a1a",
    "ema_colors": {
        20: "#FF6B9D",
        50: "#4ECDC4", 
        100: "#95E1D3",
        200: "#FFA07A"
    },
}

INDEX_CONFIG = {
    "Nifty 50": {"csv_name": "ticker/nifty50.csv"},
    "Nifty 100": {"csv_name": "ticker/nifty100.csv"},
    "Nifty 200": {"csv_name": "ticker/nifty200.csv"},
    "Nifty 500": {"csv_name": "ticker/nifty500.csv"},
}

EMA_PERIODS = [20, 50, 100, 200]
TODAY = datetime.now().date()
FIVE_YEARS_AGO = TODAY - timedelta(days=365*5)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_data(ttl=86400)
def load_tickers_from_csv(csv_filename):
    """Load ticker symbols from CSV file."""
    try:
        df = pd.read_csv(csv_filename)
        for col_name in ['Symbol', 'SYMBOL', 'Ticker', 'ticker', 'symbol']:
            if col_name in df.columns:
                return sorted(df[col_name].unique().tolist()), None
        return None, "Symbol column not found"
    except Exception as e:
        return None, str(e)

def get_cache_key(index_name, date_range):
    """Generate cache key."""
    import hashlib
    key = f"{index_name}_{date_range}"
    return hashlib.md5(key.encode()).hexdigest()

@st.cache_data(ttl=86400)
def load_cached_data(cache_key):
    """Load from cache."""
    cache_dir = Path("cache")
    cache_file = cache_dir / f"{cache_key}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def save_to_cache(cache_key, data):
    """Save to cache."""
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    with open(cache_dir / f"{cache_key}.pkl", 'wb') as f:
        pickle.dump(data, f)

# ============================================================================
# DATA DOWNLOAD
# ============================================================================

def download_ticker_data(ticker, start_date, end_date):
    """Download single ticker with error handling."""
    try:
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            timeout=10
        )
        if not data.empty and len(data) > 0:
            return ticker, data['Close'], None
        return ticker, None, "Empty data"
    except Exception as e:
        return ticker, None, str(e)

@st.cache_data(ttl=86400)
def download_all_tickers(tickers, start_date, end_date):
    """Parallel download using ThreadPoolExecutor."""
    all_data = {}
    failed = []
    
    progress_bar = st.progress(0)
    status = st.empty()
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(download_ticker_data, ticker, start_date, end_date): ticker 
            for ticker in tickers
        }
        
        completed = 0
        for future in as_completed(futures):
            ticker, data, error = future.result()
            completed += 1
            
            if error:
                failed.append((ticker, error))
            else:
                all_data[ticker] = data
            
            progress_bar.progress(completed / len(tickers))
            status.text(f"Downloaded: {completed}/{len(tickers)}")
    
    progress_bar.empty()
    status.empty()
    
    return all_data, failed

# ============================================================================
# BREADTH CALCULATION
# ============================================================================

def calculate_ema(series, period):
    """Calculate EMA."""
    return series.ewm(span=period, adjust=False).mean()

def calculate_breadth_metrics(all_data, ema_period, start_date, end_date):
    """
    Calculate breadth for single EMA period.
    Returns: daily % below, daily count below
    """
    below_ema_dict = {}
    
    for ticker, close_data in all_data.items():
        if close_data is None or len(close_data) < ema_period:
            continue
        
        ema = calculate_ema(close_data, ema_period)
        below_ema = (close_data < ema).astype(int)
        below_ema_dict[ticker] = below_ema
    
    if not below_ema_dict:
        return None, None, 0
    
    df_combined = pd.DataFrame(below_ema_dict).dropna()
    total_valid = len(below_ema_dict)
    
    # Daily counts and percentages
    daily_count = df_combined.sum(axis=1).astype(int)
    daily_pct = (df_combined.mean(axis=1) * 100).round(2)
    
    return daily_pct, daily_count, total_valid

# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_5y_vs_10y(all_breadth_data):
    """Create 5Y vs 10Y comparison for all EMAs."""
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"EMA-{ema}" for ema in EMA_PERIODS],
        vertical_spacing=0.12,
        horizontal_spacing=0.10
    )
    
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for idx, ema in enumerate(EMA_PERIODS):
        row, col = positions[idx]
        pct_data = all_breadth_data[ema]['pct']
        
        if pct_data is not None:
            # Full 10Y
            fig.add_trace(
                go.Scatter(
                    x=pct_data.index,
                    y=pct_data.values,
                    name=f"10Y",
                    line=dict(color=THEME['ema_colors'][ema], width=1, dash='dot'),
                    showlegend=(idx == 0),
                    hovertemplate="%{y:.1f}%<extra></extra>"
                ),
                row=row, col=col
            )
            
            # Last 5Y overlay
            pct_5y = pct_data[pct_data.index >= FIVE_YEARS_AGO]
            fig.add_trace(
                go.Scatter(
                    x=pct_5y.index,
                    y=pct_5y.values,
                    name=f"5Y",
                    line=dict(color=THEME['ema_colors'][ema], width=2.5),
                    showlegend=(idx == 0),
                    hovertemplate="%{y:.1f}%<extra></extra>"
                ),
                row=row, col=col
            )
            
            # Reference levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, 
                         row=row, col=col, opacity=0.3)
            fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, 
                         row=row, col=col, opacity=0.3)
        
        fig.update_yaxes(title_text="% Below" if col==1 else "", row=row, col=col, 
                        range=[0, 100], gridcolor=THEME['grid_color'])
        fig.update_xaxes(gridcolor=THEME['grid_color'], row=row, col=col)
    
    fig.update_layout(
        height=800,
        plot_bgcolor=THEME['bg_color'],
        paper_bgcolor=THEME['bg_color'],
        font=dict(color=THEME['text_color'], size=10),
        hovermode='x unified',
        margin=dict(l=50, r=50, t=70, b=50),
        showlegend=True
    )
    
    return fig

def plot_yearly_comparison(yearly_lows_df):
    """Create yearly lows comparison chart."""
    
    fig = go.Figure()
    
    for ema in EMA_PERIODS:
        df_ema = yearly_lows_df[yearly_lows_df['EMA'] == ema].sort_values('Year')
        
        if not df_ema.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_ema['Year'],
                    y=df_ema['Max_Below_Count'],
                    name=f"EMA-{ema}",
                    line=dict(color=THEME['ema_colors'][ema], width=2.5),
                    mode='lines+markers',
                    marker=dict(size=5),
                    hovertemplate="Year: %{x}<br>Max Below: %{y}<extra></extra>"
                )
            )
    
    fig.update_layout(
        title="Yearly Peak Breadth (Max Stocks Below EMA)",
        height=500,
        plot_bgcolor=THEME['bg_color'],
        paper_bgcolor=THEME['bg_color'],
        font=dict(color=THEME['text_color'], size=11),
        hovermode='x unified',
        xaxis=dict(title="Year", gridcolor=THEME['grid_color'], dtick=1),
        yaxis=dict(title="Stock Count", gridcolor=THEME['grid_color']),
        margin=dict(l=60, r=60, t=70, b=50),
    )
    
    return fig

# ============================================================================
# YEARLY LOWS CALCULATION
# ============================================================================

def calculate_yearly_lows(all_breadth_data):
    """Calculate max stocks below EMA for each year."""
    
    yearly_data = []
    
    for ema in EMA_PERIODS:
        count_data = all_breadth_data[ema]['count']
        total_stocks = all_breadth_data[ema]['total']
        
        if count_data is not None:
            # Group by year and find max
            yearly_max = count_data.groupby(count_data.index.year).max()
            
            for year, max_count in yearly_max.items():
                yearly_data.append({
                    'Year': int(year),
                    'EMA': ema,
                    'Max_Below_Count': int(max_count),
                    'Total_Stocks': total_stocks,
                    'Max_Below_Pct': round((max_count / total_stocks) * 100, 2)
                })
    
    return pd.DataFrame(yearly_data).sort_values(['Year', 'EMA'], ascending=[False, True])

# ============================================================================
# CURRENT STATS TABLE
# ============================================================================

def create_current_stats_table(all_breadth_data):
    """Create comparison table: Current vs Historical Yearly Max."""
    
    current_date = TODAY
    yearly_df = calculate_yearly_lows(all_breadth_data)
    
    stats = []
    
    for ema in EMA_PERIODS:
        pct_data = all_breadth_data[ema]['pct']
        count_data = all_breadth_data[ema]['count']
        total = all_breadth_data[ema]['total']
        
        if pct_data is not None:
            current_pct = pct_data.iloc[-1]
            current_count = count_data.iloc[-1]
            
            # Get yearly max for this EMA across all years
            yearly_ema = yearly_df[yearly_df['EMA'] == ema]
            if not yearly_ema.empty:
                yearly_breakdown = yearly_ema[['Year', 'Max_Below_Count']].set_index('Year')['Max_Below_Count'].to_dict()
            else:
                yearly_breakdown = {}
            
            stats.append({
                'EMA': ema,
                'Current_%': f"{current_pct:.1f}%",
                'Current_Count': f"{int(current_count)}/{total}",
                'Yearly_Breakdown': yearly_breakdown
            })
    
    return stats

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.markdown("## 📊 Market Breadth Analysis")
    st.markdown("*5Y vs 10Y Comparison | Yearly Historical Data | Current Status*")
    st.divider()
    
    # Index Selection
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_index = st.selectbox("Select Index", list(INDEX_CONFIG.keys()), index=0)
    
    # Load tickers
    tickers, error = load_tickers_from_csv(INDEX_CONFIG[selected_index]['csv_name'])
    if error:
        st.error(f"❌ {error}")
        st.stop()
    
    st.success(f"✅ {len(tickers)} stocks loaded")
    st.divider()
    
    # Fetch Button
    if st.button("🔄 ANALYZE (2008 - Present)", type="primary", use_container_width=True):
        cache_key = get_cache_key(selected_index, "2008_present")
        cached = load_cached_data(cache_key)
        
        if cached:
            st.info("✅ Loaded from cache")
            all_breadth_data = cached
        else:
            with st.spinner("📥 Downloading & analyzing..."):
                start_date = datetime(2008, 1, 1)
                end_date = datetime.now()
                
                all_data, failed = download_all_tickers(tickers, start_date, end_date)
                
                if failed:
                    st.warning(f"⚠️ Failed to download {len(failed)} stocks")
                
                # Calculate breadth for all EMAs
                all_breadth_data = {}
                for ema in EMA_PERIODS:
                    pct, count, total = calculate_breadth_metrics(all_data, ema, start_date, end_date)
                    all_breadth_data[ema] = {'pct': pct, 'count': count, 'total': total}
                
                save_to_cache(cache_key, all_breadth_data)
                st.success("✅ Analysis complete!")
        
        st.session_state['breadth_data'] = all_breadth_data
        st.rerun()
    
    if 'breadth_data' in st.session_state:
        breadth_data = st.session_state['breadth_data']
        
        # ====================================================================
        # TAB 1: 5Y vs 10Y COMPARISON
        # ====================================================================
        st.markdown("### 5-Year vs 10-Year Breadth Comparison")
        st.caption("**Thick line** = Last 5 years | **Dotted line** = Full history | Higher % = More stocks below EMA")
        
        fig_5y_10y = plot_5y_vs_10y(breadth_data)
        st.plotly_chart(fig_5y_10y, use_container_width=True)
        
        # Current metrics
        st.markdown("#### 📊 Current Status")
        cols = st.columns(4)
        for idx, ema in enumerate(EMA_PERIODS):
            with cols[idx]:
                if breadth_data[ema]['pct'] is not None:
                    pct = breadth_data[ema]['pct'].iloc[-1]
                    count = int(breadth_data[ema]['count'].iloc[-1])
                    total = breadth_data[ema]['total']
                    
                    st.metric(
                        f"EMA-{ema}",
                        f"{pct:.1f}%",
                        f"{count}/{total} stocks"
                    )
        
        st.divider()
        
        # ====================================================================
        # TAB 2: YEARLY LOWS COMPARISON
        # ====================================================================
        st.markdown("### Yearly Peak Breadth (2008-2025)")
        st.caption("Maximum number of stocks trading below each EMA per year")
        
        yearly_df = calculate_yearly_lows(breadth_data)
        
        fig_yearly = plot_yearly_comparison(yearly_df)
        st.plotly_chart(fig_yearly, use_container_width=True)
        
        st.divider()
        
        # ====================================================================
        # TAB 3: CURRENT vs HISTORICAL YEARLY MAX
        # ====================================================================
        st.markdown("### Current Breadth vs Historical Yearly Peak")
        
        # Create pivot table: Years as columns, EMAs as rows
        pivot_data = []
        
        for ema in EMA_PERIODS:
            row_data = {'EMA': ema}
            
            # Current
            pct_data = breadth_data[ema]['pct']
            count_data = breadth_data[ema]['count']
            total = breadth_data[ema]['total']
            
            if pct_data is not None:
                current_count = int(count_data.iloc[-1])
                row_data['Current'] = f"{current_count}/{total}"
            
            # Yearly max
            yearly_ema = yearly_df[yearly_df['EMA'] == ema].sort_values('Year', ascending=False)
            for _, row in yearly_ema.head(17).iterrows():  # 2008-2024
                year = int(row['Year'])
                count = int(row['Max_Below_Count'])
                row_data[str(year)] = count
            
            pivot_data.append(row_data)
        
        comparison_df = pd.DataFrame(pivot_data)
        
        # Display as styled table
        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True,
            height=250
        )
        
        # Download option
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            "📥 Download Yearly Comparison (CSV)",
            csv,
            f"breadth_yearly_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv"
        )
        
        st.divider()
        
        # ====================================================================
        # TAB 4: DETAILED DAILY DATA
        # ====================================================================
        st.markdown("### Daily Breadth Data")
        
        selected_ema = st.selectbox("Select EMA", EMA_PERIODS, format_func=lambda x: f"EMA-{x}")
        
        if breadth_data[selected_ema]['pct'] is not None:
            pct_data = breadth_data[selected_ema]['pct']
            count_data = breadth_data[selected_ema]['count']
            total = breadth_data[selected_ema]['total']
            
            df_display = pd.DataFrame({
                'Date': pct_data.index.strftime('%Y-%m-%d'),
                '% Below EMA': pct_data.values,
                'Stocks Below': count_data.values,
                'Total': total
            })
            
            st.dataframe(
                df_display.sort_values('Date', ascending=False),
                use_container_width=True,
                hide_index=True,
                height=500
            )
            
            csv = df_display.to_csv(index=False)
            st.download_button(
                f"📥 Download EMA-{selected_ema} Data (CSV)",
                csv,
                f"ema{selected_ema}_daily_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
    else:
        st.info("👆 Click **ANALYZE** button to fetch data")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
import pickle
from pathlib import Path
import hashlib

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Multi-EMA Breadth Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Theme Configuration
THEME = {
    "bg_color": "#0a0a0a",
    "text_color": "#e0e0e0",
    "grid_color": "#1a1a1a",
    "ema_colors": {
        "20": "#FF6B9D",
        "50": "#4ECDC4", 
        "100": "#95E1D3",
        "200": "#FFA07A"
    },
}

INDEX_CONFIG = {
    "Nifty 50": {"csv_name": "ticker/nifty50.csv", "description": "Top 50 Large Cap Stocks"},
    "Nifty 100": {"csv_name": "ticker/nifty100.csv", "description": "Top 100 Large Cap Stocks"},
    "Nifty 200": {"csv_name": "ticker/nifty200.csv", "description": "Top 200 Large Cap and Mid Cap Stocks"},
    "Nifty Total Market": {"csv_name": "ticker/niftytotalmarket.csv", "description": "Nifty Total Market Index"},
}

EMA_PERIODS = [20, 50, 100, 200]

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    clear_cache = st.button("🗑️ Clear Cache", help="Clear downloaded data cache")
    if clear_cache:
        cache_dir = Path("cache")
        if cache_dir.exists():
            for file in cache_dir.glob("*.pkl"):
                file.unlink()
            st.success("✅ Cache cleared!")
            st.rerun()

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

def get_cache_key(tickers, start_date, end_date):
    """Generate unique cache key for dataset."""
    key_str = f"{'-'.join(sorted(tickers))}_{start_date}_{end_date}"
    return hashlib.md5(key_str.encode()).hexdigest()

def load_cached_data(cache_key):
    """Load data from cache if exists."""
    cache_dir = Path("cache")
    cache_file = cache_dir / f"{cache_key}.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

def save_to_cache(cache_key, data):
    """Save data to cache."""
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.pkl"
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

class MultiBreadthAnalyzer:
    """Analyzer for multiple EMA breadth indicators."""
    
    @staticmethod
    def calculate_ema(data, period):
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period, adjust=False).mean()
    
    def download_all_data(self, tickers, start_date, end_date):
        """Download all ticker data in batch."""
        progress_bar = st.progress(0)
        status = st.empty()
        
        status.text("📥 Downloading data for all stocks (batch mode)...")
        
        try:
            # Download all at once - yfinance handles this better
            data = yf.download(
                tickers,
                start=start_date,
                end=end_date,
                progress=False,
                group_by='ticker',
                threads=False
            )
            
            progress_bar.progress(100)
            status.text("✅ Download complete!")
            
            return data
        except Exception as e:
            st.error(f"❌ Download failed: {str(e)}")
            return None
        finally:
            progress_bar.empty()
            status.empty()
    
    def calculate_multi_ema_breadth(self, tickers, start_date, end_date):
        """Calculate breadth for multiple EMA periods."""
        
        # Try cache first
        cache_key = get_cache_key(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        cached = load_cached_data(cache_key)
        if cached:
            st.info("✅ Loaded from cache")
            return cached
        
        # Download data
        data = self.download_all_data(tickers, start_date, end_date)
        if data is None:
            return None
        
        results = {ema: {'percent': None, 'count': None, 'total': 0} for ema in EMA_PERIODS}
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        for idx, ema_period in enumerate(EMA_PERIODS):
            status.text(f"📊 Calculating EMA-{ema_period} breadth...")
            
            ema_results = {}
            valid_count = 0
            
            for ticker in tickers:
                try:
                    # Handle both single and multi-ticker dataframes
                    if len(tickers) == 1:
                        close_data = data['Close'].dropna()
                    else:
                        if ticker not in data.columns.levels[0]:
                            continue
                        close_data = data[ticker]['Close'].dropna()
                    
                    if len(close_data) < ema_period:
                        continue
                    
                    ema = self.calculate_ema(close_data, ema_period)
                    above = (close_data.values > ema.values).astype(int)
                    ema_results[ticker] = pd.Series(above, index=close_data.index)
                    valid_count += 1
                    
                except:
                    continue
            
            if ema_results:
                df_combined = pd.DataFrame(ema_results).dropna()
                results[ema_period]['percent'] = (df_combined.mean(axis=1) * 100).round(2)
                results[ema_period]['count'] = df_combined.sum(axis=1).astype(int)
                results[ema_period]['total'] = valid_count
            
            progress_bar.progress((idx + 1) / len(EMA_PERIODS))
        
        progress_bar.empty()
        status.empty()
        
        # Save to cache
        save_to_cache(cache_key, results)
        
        return results

def plot_comparison_chart(data_5y, data_10y, theme):
    """Create comparison chart for 5Y vs 10Y."""
    
    fig = make_subplots(
        rows=len(EMA_PERIODS), cols=2,
        subplot_titles=[f"EMA-{ema} (5Y)" for ema in EMA_PERIODS] + [f"EMA-{ema} (10Y)" for ema in EMA_PERIODS],
        vertical_spacing=0.08,
        horizontal_spacing=0.08,
        specs=[[{"secondary_y": False}, {"secondary_y": False}] for _ in EMA_PERIODS]
    )
    
    for idx, ema in enumerate(EMA_PERIODS):
        row = idx + 1
        
        # 5Y data
        if data_5y[ema]['percent'] is not None:
            fig.add_trace(
                go.Scatter(
                    x=data_5y[ema]['percent'].index,
                    y=data_5y[ema]['percent'].values,
                    name=f"EMA-{ema} 5Y",
                    line=dict(color=theme['ema_colors'][str(ema)], width=1.5),
                    fill='tozeroy',
                    fillcolor=f"rgba{tuple(list(int(theme['ema_colors'][str(ema)][i:i+2], 16) for i in (1, 3, 5)) + [0.2])}",
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
                    showlegend=(idx == 0)
                ),
                row=row, col=1
            )
        
        # 10Y data
        if data_10y[ema]['percent'] is not None:
            fig.add_trace(
                go.Scatter(
                    x=data_10y[ema]['percent'].index,
                    y=data_10y[ema]['percent'].values,
                    name=f"EMA-{ema} 10Y",
                    line=dict(color=theme['ema_colors'][str(ema)], width=1.5),
                    fill='tozeroy',
                    fillcolor=f"rgba{tuple(list(int(theme['ema_colors'][str(ema)][i:i+2], 16) for i in (1, 3, 5)) + [0.2])}",
                    hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>",
                    showlegend=(idx == 0)
                ),
                row=row, col=2
            )
        
        # Add reference lines
        for col in [1, 2]:
            fig.add_hline(y=70, line_dash="dash", line_color="green", line_width=1, row=row, col=col, opacity=0.5)
            fig.add_hline(y=30, line_dash="dash", line_color="red", line_width=1, row=row, col=col, opacity=0.5)
        
        # Update axes
        fig.update_yaxes(title_text="Breadth %", row=row, col=1, range=[0, 100], gridcolor=theme['grid_color'])
        fig.update_yaxes(title_text="Breadth %", row=row, col=2, range=[0, 100], gridcolor=theme['grid_color'])
        fig.update_xaxes(gridcolor=theme['grid_color'], row=row, col=1)
        fig.update_xaxes(gridcolor=theme['grid_color'], row=row, col=2)
    
    fig.update_layout(
        height=350 * len(EMA_PERIODS),
        plot_bgcolor=theme['bg_color'],
        paper_bgcolor=theme['bg_color'],
        font=dict(color=theme['text_color'], size=10),
        hovermode='x unified',
        margin=dict(l=60, r=60, t=80, b=60),
    )
    
    return fig

def main():
    st.markdown("## 📊 Multi-EMA Breadth Analysis")
    st.markdown("*Compare 5Y vs 10Y breadth across EMA-20, 50, 100, 200 | Track yearly lows since 2008*")
    st.divider()
    
    # Index Selection
    selected_index = st.selectbox("Select Index", list(INDEX_CONFIG.keys()), index=0)
    st.info(f"ℹ️ {INDEX_CONFIG[selected_index]['description']}")
    
    # Load tickers
    tickers, error = load_tickers_from_csv(INDEX_CONFIG[selected_index]['csv_name'])
    if error:
        st.error(f"❌ {error}")
        st.stop()
    
    st.success(f"✅ Loaded {len(tickers)} tickers")
    st.divider()
    
    analyzer = MultiBreadthAnalyzer()
    
    tab1, tab2 = st.tabs(["📈 5Y vs 10Y Comparison", "📉 Yearly Lows (2008-2025)"])
    
    with tab1:
        st.markdown("### 5-Year vs 10-Year Breadth Comparison")
        
        if st.button("🔄 FETCH 5Y & 10Y DATA", type="primary", use_container_width=True):
            today = datetime.now()
            start_5y = today - timedelta(days=365*5)
            start_10y = today - timedelta(days=365*10)
            
            with st.spinner("📥 Fetching 5-year data..."):
                breadth_5y = analyzer.calculate_multi_ema_breadth(tickers, start_5y, today)
            
            if breadth_5y:
                with st.spinner("📥 Fetching 10-year data..."):
                    breadth_10y = analyzer.calculate_multi_ema_breadth(tickers, start_10y, today)
                
                if breadth_10y:
                    st.success("✅ Data fetched successfully!")
                    st.session_state['breadth_5y'] = breadth_5y
                    st.session_state['breadth_10y'] = breadth_10y
                    st.rerun()
        
        if 'breadth_5y' in st.session_state and 'breadth_10y' in st.session_state:
            breadth_5y = st.session_state['breadth_5y']
            breadth_10y = st.session_state['breadth_10y']
            
            # Display charts
            fig = plot_comparison_chart(breadth_5y, breadth_10y, THEME)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            st.markdown("### 📊 Statistics Summary")
            
            for ema in EMA_PERIODS:
                with st.expander(f"EMA-{ema} Statistics", expanded=(ema==50)):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**5-Year Stats**")
                        if breadth_5y[ema]['percent'] is not None:
                            data = breadth_5y[ema]
                            col1a, col1b, col1c = st.columns(3)
                            col1a.metric("Current", f"{data['percent'].iloc[-1]:.1f}%")
                            col1b.metric("Average", f"{data['percent'].mean():.1f}%")
                            col1c.metric("Stocks", f"{data['count'].iloc[-1]}/{data['total']}")
                            
                            col1a.metric("Highest", f"{data['percent'].max():.1f}%")
                            col1b.metric("Lowest", f"{data['percent'].min():.1f}%")
                            col1c.metric("Std Dev", f"{data['percent'].std():.1f}%")
                    
                    with col2:
                        st.markdown("**10-Year Stats**")
                        if breadth_10y[ema]['percent'] is not None:
                            data = breadth_10y[ema]
                            col2a, col2b, col2c = st.columns(3)
                            col2a.metric("Current", f"{data['percent'].iloc[-1]:.1f}%")
                            col2b.metric("Average", f"{data['percent'].mean():.1f}%")
                            col2c.metric("Stocks", f"{data['count'].iloc[-1]}/{data['total']}")
                            
                            col2a.metric("Highest", f"{data['percent'].max():.1f}%")
                            col2b.metric("Lowest", f"{data['percent'].min():.1f}%")
                            col2c.metric("Std Dev", f"{data['percent'].std():.1f}%")
    
    with tab2:
        st.markdown("### Yearly Lowest Breadth (2008-2025)")
        
        if st.button("🔄 FETCH HISTORICAL DATA (2008-2025)", type="primary", use_container_width=True):
            start_date = datetime(2008, 1, 1)
            today = datetime.now()
            
            with st.spinner("📥 Fetching 17 years of data... (may take 5-10 minutes)"):
                breadth_historical = analyzer.calculate_multi_ema_breadth(tickers, start_date, today)
            
            if breadth_historical:
                st.success("✅ Historical data fetched!")
                st.session_state['breadth_historical'] = breadth_historical
                st.rerun()
        
        if 'breadth_historical' in st.session_state:
            breadth_hist = st.session_state['breadth_historical']
            
            for ema in EMA_PERIODS:
                if breadth_hist[ema]['percent'] is not None:
                    st.markdown(f"#### EMA-{ema} Lowest Breadth by Year")
                    
                    data = breadth_hist[ema]['percent']
                    yearly_lows = data.groupby(data.index.year).agg(['min', 'idxmin'])
                    
                    df_yearly = pd.DataFrame({
                        'Year': yearly_lows.index,
                        'Lowest Breadth %': yearly_lows['min'].round(2),
                        'Date': yearly_lows['idxmin'].dt.strftime('%Y-%m-%d'),
                        'Stocks Above': [breadth_hist[ema]['count'].loc[date] for date in yearly_lows['idxmin']],
                        'Total Stocks': breadth_hist[ema]['total']
                    })
                    
                    st.dataframe(df_yearly, use_container_width=True, hide_index=True)
                    
                    csv = df_yearly.to_csv(index=False)
                    st.download_button(
                        f"📥 Download EMA-{ema} Data",
                        csv,
                        f"ema{ema}_yearly_lows_2008_2025.csv",
                        "text/csv"
                    )
                    
                    st.divider()

if __name__ == "__main__":
    main()

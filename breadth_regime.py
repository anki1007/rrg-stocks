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
    page_title="Market Breadth Analysis",
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
        20: "#FF6B9D",
        50: "#4ECDC4", 
        100: "#95E1D3",
        200: "#FFA07A"
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
    
    st.markdown("**Analysis Period**")
    analysis_start = st.date_input("Start Date", value=datetime(2008, 1, 1))
    
    st.markdown("---")
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

class BreadthAnalyzer:
    """Analyzer for market breadth indicators."""
    
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
        """Calculate breadth for multiple EMA periods - stocks BELOW EMA."""
        
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
            status.text(f"📊 Calculating EMA-{ema_period} breadth (stocks BELOW)...")
            
            ema_results = {}
            valid_count = 0
            
            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        close_data = data['Close'].dropna()
                    else:
                        if ticker not in data.columns.levels[0]:
                            continue
                        close_data = data[ticker]['Close'].dropna()
                    
                    if len(close_data) < ema_period:
                        continue
                    
                    ema = self.calculate_ema(close_data, ema_period)
                    # Calculate stocks BELOW EMA (inverted logic)
                    below = (close_data.values < ema.values).astype(int)
                    ema_results[ticker] = pd.Series(below, index=close_data.index)
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

def plot_5y_10y_comparison(full_data, theme):
    """Create 5Y vs 10Y comparison line charts for each EMA."""
    
    today = datetime.now()
    cutoff_5y = today - timedelta(days=365*5)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"EMA-{ema}: 5Y vs 10Y (% Stocks Below)" for ema in EMA_PERIODS],
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    positions = [(1,1), (1,2), (2,1), (2,2)]
    
    for idx, ema in enumerate(EMA_PERIODS):
        row, col = positions[idx]
        
        if full_data[ema]['percent'] is not None:
            data_full = full_data[ema]['percent']
            
            # Split into 5Y and 10Y
            data_5y = data_full[data_full.index >= cutoff_5y]
            data_10y = data_full
            
            # 10Y line (full data)
            fig.add_trace(
                go.Scatter(
                    x=data_10y.index,
                    y=data_10y.values,
                    name=f"10Y",
                    line=dict(color=theme['ema_colors'][ema], width=1.2, dash='dot'),
                    hovertemplate="<b>10Y</b><br>%{x|%Y-%m-%d}<br>%{y:.1f}% below<extra></extra>",
                    showlegend=(idx == 0),
                    legendgroup=f"ema{ema}"
                ),
                row=row, col=col
            )
            
            # 5Y line (overlay)
            fig.add_trace(
                go.Scatter(
                    x=data_5y.index,
                    y=data_5y.values,
                    name=f"5Y",
                    line=dict(color=theme['ema_colors'][ema], width=2.5),
                    hovertemplate="<b>5Y</b><br>%{x|%Y-%m-%d}<br>%{y:.1f}% below<extra></extra>",
                    showlegend=(idx == 0),
                    legendgroup=f"ema{ema}"
                ),
                row=row, col=col
            )
            
            # Reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, 
                         row=row, col=col, opacity=0.4, annotation_text="70%" if col==1 else "")
            fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, 
                         row=row, col=col, opacity=0.4, annotation_text="30%" if col==1 else "")
        
        # Update axes
        fig.update_yaxes(title_text="% Below EMA" if col==1 else "", row=row, col=col, 
                        range=[0, 100], gridcolor=theme['grid_color'])
        fig.update_xaxes(gridcolor=theme['grid_color'], row=row, col=col)
    
    fig.update_layout(
        height=800,
        plot_bgcolor=theme['bg_color'],
        paper_bgcolor=theme['bg_color'],
        font=dict(color=theme['text_color'], size=11),
        hovermode='x unified',
        margin=dict(l=60, r=60, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_yearly_lows_comparison(full_data):
    """Create yearly lowest breadth comparison table."""
    
    yearly_data = []
    
    for ema in EMA_PERIODS:
        if full_data[ema]['percent'] is not None:
            data = full_data[ema]['percent']
            yearly_lows = data.groupby(data.index.year).agg(['min', 'idxmin'])
            
            for year in yearly_lows.index:
                yearly_data.append({
                    'Year': int(year),
                    'EMA': ema,
                    'Lowest Breadth %': round(yearly_lows.loc[year, 'min'], 2),
                    'Date': yearly_lows.loc[year, 'idxmin'].strftime('%Y-%m-%d'),
                    'Stocks Below': int(full_data[ema]['count'].loc[yearly_lows.loc[year, 'idxmin']]),
                    'Total': full_data[ema]['total']
                })
    
    return pd.DataFrame(yearly_data)

def plot_yearly_lows_chart(df_yearly, theme):
    """Plot yearly lows comparison across EMAs."""
    
    fig = go.Figure()
    
    for ema in EMA_PERIODS:
        df_ema = df_yearly[df_yearly['EMA'] == ema].sort_values('Year')
        
        fig.add_trace(
            go.Scatter(
                x=df_ema['Year'],
                y=df_ema['Lowest Breadth %'],
                name=f"EMA-{ema}",
                line=dict(color=theme['ema_colors'][ema], width=2.5),
                mode='lines+markers',
                marker=dict(size=6),
                hovertemplate="<b>EMA-%{fullData.name}</b><br>Year: %{x}<br>Lowest: %{y:.1f}%<extra></extra>"
            )
        )
    
    fig.update_layout(
        title="Yearly Lowest Breadth Comparison (% Stocks Below EMA)",
        height=500,
        plot_bgcolor=theme['bg_color'],
        paper_bgcolor=theme['bg_color'],
        font=dict(color=theme['text_color'], size=11),
        hovermode='x unified',
        xaxis=dict(title="Year", gridcolor=theme['grid_color'], dtick=1),
        yaxis=dict(title="% Stocks Below EMA", gridcolor=theme['grid_color'], range=[0, 100]),
        margin=dict(l=60, r=60, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def main():
    st.markdown("## 📊 Market Breadth Analysis - Stocks Below EMA")
    st.markdown("*Compare 5Y vs 10Y breadth patterns | Track yearly lows from 2008*")
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
    
    analyzer = BreadthAnalyzer()
    
    # Fetch Data Button
    if st.button("🔄 FETCH BREADTH DATA (2008 - Present)", type="primary", use_container_width=True):
        start_date = datetime(2008, 1, 1)
        today = datetime.now()
        
        with st.spinner("📥 Fetching historical data... (this may take 5-10 minutes)"):
            breadth_data = analyzer.calculate_multi_ema_breadth(tickers, start_date, today)
        
        if breadth_data:
            st.success("✅ Data fetched successfully!")
            st.session_state['breadth_full'] = breadth_data
            st.rerun()
    
    if 'breadth_full' in st.session_state:
        breadth_full = st.session_state['breadth_full']
        
        tab1, tab2, tab3 = st.tabs(["📈 5Y vs 10Y Comparison", "📉 Yearly Lows Comparison", "📋 Detailed Data"])
        
        with tab1:
            st.markdown("### 5-Year vs 10-Year Breadth Overlay")
            st.caption("**Thick line** = Last 5 years | **Dotted line** = Full 10+ years | Higher % = More stocks below EMA (bearish)")
            
            fig_comparison = plot_5y_10y_comparison(breadth_full, THEME)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Current Stats
            st.markdown("### 📊 Current Breadth Statistics")
            
            cols = st.columns(4)
            for idx, ema in enumerate(EMA_PERIODS):
                with cols[idx]:
                    if breadth_full[ema]['percent'] is not None:
                        current = breadth_full[ema]['percent'].iloc[-1]
                        count = breadth_full[ema]['count'].iloc[-1]
                        total = breadth_full[ema]['total']
                        
                        st.metric(
                            f"EMA-{ema}",
                            f"{current:.1f}%",
                            f"{count}/{total} stocks"
                        )
        
        with tab2:
            st.markdown("### Yearly Lowest Breadth Comparison")
            st.caption("Shows the lowest breadth % reached each year for each EMA period")
            
            df_yearly = create_yearly_lows_comparison(breadth_full)
            
            # Chart
            fig_yearly = plot_yearly_lows_chart(df_yearly, THEME)
            st.plotly_chart(fig_yearly, use_container_width=True)
            
            # Table with filters
            st.markdown("#### 📋 Detailed Yearly Lows Table")
            
            col1, col2 = st.columns([1, 3])
            with col1:
                selected_ema = st.multiselect(
                    "Filter EMA",
                    options=EMA_PERIODS,
                    default=EMA_PERIODS
                )
            with col2:
                year_range = st.slider(
                    "Year Range",
                    min_value=int(df_yearly['Year'].min()),
                    max_value=int(df_yearly['Year'].max()),
                    value=(int(df_yearly['Year'].min()), int(df_yearly['Year'].max()))
                )
            
            df_filtered = df_yearly[
                (df_yearly['EMA'].isin(selected_ema)) &
                (df_yearly['Year'] >= year_range[0]) &
                (df_yearly['Year'] <= year_range[1])
            ].sort_values(['Year', 'EMA'], ascending=[False, True])
            
            st.dataframe(df_filtered, use_container_width=True, hide_index=True, height=600)
            
            # Download
            csv = df_filtered.to_csv(index=False)
            st.download_button(
                "📥 Download Yearly Lows Data (CSV)",
                csv,
                f"yearly_lows_comparison_{datetime.now().strftime('%Y%m%d')}.csv",
                "text/csv"
            )
        
        with tab3:
            st.markdown("### 📋 Detailed Daily Breadth Data")
            
            selected_ema_detail = st.selectbox("Select EMA Period", EMA_PERIODS, index=1)
            
            if breadth_full[selected_ema_detail]['percent'] is not None:
                df_detail = pd.DataFrame({
                    'Date': breadth_full[selected_ema_detail]['percent'].index.strftime('%Y-%m-%d'),
                    '% Below EMA': breadth_full[selected_ema_detail]['percent'].values,
                    'Stocks Below': breadth_full[selected_ema_detail]['count'].values,
                    'Total Stocks': breadth_full[selected_ema_detail]['total']
                })
                
                st.dataframe(
                    df_detail.sort_values('Date', ascending=False),
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )
                
                csv = df_detail.to_csv(index=False)
                st.download_button(
                    f"📥 Download EMA-{selected_ema_detail} Daily Data (CSV)",
                    csv,
                    f"ema{selected_ema_detail}_daily_breadth_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )
    
    else:
        st.info("👆 Click the button above to fetch breadth data and start analysis")

if __name__ == "__main__":
    main()

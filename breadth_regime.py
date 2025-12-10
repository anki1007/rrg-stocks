"""
Institutional-Grade EMA Breadth Analysis Dashboard
===================================================
Professional market breadth analysis tool tracking stocks below EMA levels
across multiple timeframes with historical comparison since 2008.

Author: Quantitative Analytics Team
Version: 2.0
"""

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
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

warnings.filterwarnings('ignore')

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="EMA Breadth Analysis | Institutional Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =============================================================================
# PROFESSIONAL THEME CONFIGURATION
# =============================================================================
THEME = {
    "bg_primary": "#0D1117",
    "bg_secondary": "#161B22", 
    "bg_card": "#21262D",
    "border": "#30363D",
    "text_primary": "#F0F6FC",
    "text_secondary": "#8B949E",
    "accent_green": "#3FB950",
    "accent_red": "#F85149",
    "accent_blue": "#58A6FF",
    "accent_purple": "#A371F7",
    "accent_orange": "#D29922",
    "accent_cyan": "#39C5CF",
    "ema_colors": {
        20: "#F85149",    # Red - Short term
        50: "#D29922",    # Orange - Medium term  
        100: "#3FB950",   # Green - Long term
        200: "#58A6FF"    # Blue - Very long term
    }
}

# =============================================================================
# CUSTOM CSS FOR INSTITUTIONAL LOOK
# =============================================================================
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main background */
    .stApp {{
        background: linear-gradient(180deg, {THEME['bg_primary']} 0%, #0a0d12 100%);
    }}
    
    /* Headers */
    h1, h2, h3 {{
        font-family: 'Inter', sans-serif !important;
        color: {THEME['text_primary']} !important;
        font-weight: 600 !important;
    }}
    
    /* Metrics styling */
    [data-testid="stMetricValue"] {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }}
    
    [data-testid="stMetricLabel"] {{
        font-family: 'Inter', sans-serif !important;
        font-size: 0.85rem !important;
        color: {THEME['text_secondary']} !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    /* Card styling */
    .metric-card {{
        background: {THEME['bg_card']};
        border: 1px solid {THEME['border']};
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
    }}
    
    .metric-card-critical {{
        border-left: 4px solid {THEME['accent_red']};
    }}
    
    .metric-card-warning {{
        border-left: 4px solid {THEME['accent_orange']};
    }}
    
    .metric-card-healthy {{
        border-left: 4px solid {THEME['accent_green']};
    }}
    
    /* Table styling */
    .dataframe {{
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.85rem !important;
    }}
    
    /* Button styling */
    .stButton > button {{
        background: linear-gradient(135deg, {THEME['accent_blue']} 0%, {THEME['accent_purple']} 100%);
        color: white;
        border: none;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .stButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(88, 166, 255, 0.3);
    }}
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        background: {THEME['bg_secondary']};
        padding: 8px;
        border-radius: 12px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px;
        color: {THEME['text_secondary']};
        font-family: 'Inter', sans-serif;
        font-weight: 500;
    }}
    
    .stTabs [aria-selected="true"] {{
        background: {THEME['bg_card']};
        color: {THEME['text_primary']};
    }}
    
    /* Expander styling */
    .streamlit-expanderHeader {{
        background: {THEME['bg_card']} !important;
        border-radius: 8px !important;
        font-family: 'Inter', sans-serif !important;
    }}
    
    /* Progress bar */
    .stProgress > div > div {{
        background: linear-gradient(90deg, {THEME['accent_blue']} 0%, {THEME['accent_cyan']} 100%);
    }}
    
    /* Divider */
    hr {{
        border-color: {THEME['border']} !important;
        margin: 2rem 0 !important;
    }}
    
    /* Status indicator */
    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }}
    
    .status-critical {{ background: {THEME['accent_red']}; }}
    .status-warning {{ background: {THEME['accent_orange']}; }}
    .status-healthy {{ background: {THEME['accent_green']}; }}
    
    /* Hide Streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    /* Selectbox styling */
    .stSelectbox [data-baseweb="select"] {{
        background: {THEME['bg_card']};
        border-color: {THEME['border']};
    }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# INDEX CONFIGURATION
# =============================================================================
INDEX_CONFIG = {
    "Nifty 50": {"csv_name": "ticker/nifty50.csv", "description": "India's flagship index - Top 50 blue-chip companies"},
    "Nifty 100": {"csv_name": "ticker/nifty100.csv", "description": "Top 100 companies by market capitalization"},
    "Nifty 200": {"csv_name": "ticker/nifty200.csv", "description": "Broad market representation - Large & Mid Cap"},
    "Nifty Total Market": {"csv_name": "ticker/niftytotalmarket.csv", "description": "Comprehensive market coverage"},
}

EMA_PERIODS = [20, 50, 100, 200]

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

@st.cache_data(ttl=86400)
def load_tickers_from_csv(csv_filename):
    """Load ticker symbols from CSV file with .NS suffix for NSE."""
    try:
        df = pd.read_csv(csv_filename)
        # Find the symbol column
        symbol_col = None
        for col_name in ['Symbol', 'SYMBOL', 'Ticker', 'ticker', 'symbol']:
            if col_name in df.columns:
                symbol_col = col_name
                break
        
        if symbol_col is None:
            return None, None, "Symbol column not found"
        
        # Get unique symbols (already includes .NS suffix in CSV)
        tickers = df[symbol_col].unique().tolist()
        
        # Get company info if available
        company_info = {}
        name_col = None
        industry_col = None
        
        for col in ['Company Name', 'Company', 'Name', 'company_name']:
            if col in df.columns:
                name_col = col
                break
        
        for col in ['Industry', 'Sector', 'industry', 'sector']:
            if col in df.columns:
                industry_col = col
                break
        
        for _, row in df.iterrows():
            sym = row[symbol_col]
            company_info[sym] = {
                'name': row[name_col] if name_col else row[symbol_col],
                'industry': row[industry_col] if industry_col else 'N/A'
            }
        
        return sorted(tickers), company_info, None
    except Exception as e:
        return None, None, str(e)


def get_cache_key(tickers, start_date, end_date, prefix="breadth"):
    """Generate unique cache key for dataset."""
    key_str = f"{prefix}_{len(tickers)}_{start_date}_{end_date}"
    return hashlib.md5(key_str.encode()).hexdigest()


def load_cached_data(cache_key):
    """Load data from cache if exists and is recent."""
    cache_dir = Path("cache")
    cache_file = cache_dir / f"{cache_key}.pkl"
    if cache_file.exists():
        # Check if cache is less than 24 hours old
        if (time.time() - cache_file.stat().st_mtime) < 86400:
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return None
    return None


def save_to_cache(cache_key, data):
    """Save data to cache."""
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"{cache_key}.pkl"
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except:
        pass


# =============================================================================
# MAIN ANALYZER CLASS
# =============================================================================

class InstitutionalBreadthAnalyzer:
    """
    Professional-grade breadth analyzer for institutional analysis.
    Tracks stocks trading BELOW EMA levels across multiple timeframes.
    """
    
    def __init__(self):
        self.data_cache = {}
    
    @staticmethod
    def calculate_ema(data, period):
        """Calculate Exponential Moving Average."""
        return data.ewm(span=period, adjust=False).mean()
    
    def download_stock_data(self, ticker, start_date, end_date, retries=3):
        """Download data for a single stock with retry logic."""
        for attempt in range(retries):
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True
                )
                if not data.empty and len(data) > 50:
                    return ticker, data['Close']
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(0.5)
                continue
        return ticker, None
    
    def download_all_data_parallel(self, tickers, start_date, end_date, max_workers=10):
        """Download all ticker data in parallel with progress tracking."""
        progress_bar = st.progress(0)
        status = st.empty()
        
        all_data = {}
        completed = 0
        total = len(tickers)
        failed = []
        
        status.markdown(f"📥 **Downloading data for {total} stocks...**")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.download_stock_data, ticker, start_date, end_date): ticker 
                for ticker in tickers
            }
            
            for future in as_completed(futures):
                ticker, data = future.result()
                completed += 1
                
                if data is not None:
                    all_data[ticker] = data
                else:
                    failed.append(ticker)
                
                progress_bar.progress(completed / total)
                
                if completed % 20 == 0 or completed == total:
                    status.markdown(f"📥 **Progress: {completed}/{total} stocks** | ✅ {len(all_data)} successful | ❌ {len(failed)} failed")
        
        progress_bar.empty()
        status.empty()
        
        return all_data, failed
    
    def calculate_breadth_below_ema(self, tickers, start_date, end_date):
        """
        Calculate breadth for stocks BELOW EMA for each period.
        Returns percentage and count of stocks below each EMA level over time.
        """
        # Try cache first
        cache_key = get_cache_key(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        cached = load_cached_data(cache_key)
        if cached:
            st.success("✅ **Loaded from cache** | Data is less than 24 hours old")
            return cached
        
        # Download all data
        all_data, failed = self.download_all_data_parallel(tickers, start_date, end_date)
        
        if not all_data:
            st.error("❌ No data could be downloaded")
            return None
        
        st.info(f"📊 **Processing {len(all_data)} stocks** | {len(failed)} failed to download")
        
        results = {ema: {'percent_below': None, 'count_below': None, 'total': 0} for ema in EMA_PERIODS}
        
        progress_bar = st.progress(0)
        status = st.empty()
        
        for idx, ema_period in enumerate(EMA_PERIODS):
            status.markdown(f"⚙️ **Calculating EMA-{ema_period} breadth...**")
            
            below_ema_series = {}
            valid_count = 0
            
            for ticker, close_data in all_data.items():
                try:
                    if len(close_data) < ema_period + 20:
                        continue
                    
                    # Calculate EMA
                    ema = self.calculate_ema(close_data, ema_period)
                    
                    # Check if price is BELOW EMA (1 = below, 0 = above or equal)
                    below = (close_data.values < ema.values).astype(int)
                    below_ema_series[ticker] = pd.Series(below, index=close_data.index)
                    valid_count += 1
                    
                except Exception:
                    continue
            
            if below_ema_series:
                # Combine all series
                df_combined = pd.DataFrame(below_ema_series)
                df_combined = df_combined.dropna(how='all')
                
                # Forward fill missing values within reasonable limits
                df_combined = df_combined.ffill(limit=5)
                
                # Calculate percentage and count of stocks BELOW EMA
                results[ema_period]['percent_below'] = (df_combined.mean(axis=1) * 100).round(2)
                results[ema_period]['count_below'] = df_combined.sum(axis=1).astype(int)
                results[ema_period]['total'] = valid_count
            
            progress_bar.progress((idx + 1) / len(EMA_PERIODS))
        
        progress_bar.empty()
        status.empty()
        
        # Save to cache
        save_to_cache(cache_key, results)
        
        return results
    
    def get_current_breadth(self, results):
        """Get the most recent breadth values."""
        current = {}
        for ema in EMA_PERIODS:
            if results[ema]['percent_below'] is not None and len(results[ema]['percent_below']) > 0:
                current[ema] = {
                    'percent': results[ema]['percent_below'].iloc[-1],
                    'count': results[ema]['count_below'].iloc[-1],
                    'total': results[ema]['total'],
                    'date': results[ema]['percent_below'].index[-1].strftime('%Y-%m-%d')
                }
            else:
                current[ema] = {'percent': 0, 'count': 0, 'total': 0, 'date': 'N/A'}
        return current
    
    def get_yearly_extremes(self, results):
        """Get the highest percentage of stocks below EMA for each year."""
        yearly_data = {}
        
        for ema in EMA_PERIODS:
            if results[ema]['percent_below'] is not None:
                data = results[ema]['percent_below']
                count_data = results[ema]['count_below']
                
                # Group by year and get maximum (worst breadth)
                yearly_max = data.groupby(data.index.year).agg(['max', 'idxmax'])
                yearly_count = count_data.groupby(count_data.index.year).max()
                
                yearly_data[ema] = {
                    'max_percent': yearly_max['max'],
                    'max_date': yearly_max['idxmax'],
                    'max_count': yearly_count,
                    'total': results[ema]['total']
                }
        
        return yearly_data


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_comparison_chart(data_5y, data_10y, ema_period, theme):
    """Create a professional side-by-side comparison chart for a single EMA period."""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'<b>5-Year History</b>', f'<b>10-Year History</b>'),
        horizontal_spacing=0.08
    )
    
    color = theme['ema_colors'][ema_period]
    
    # 5Y Chart
    if data_5y[ema_period]['percent_below'] is not None:
        fig.add_trace(
            go.Scatter(
                x=data_5y[ema_period]['percent_below'].index,
                y=data_5y[ema_period]['percent_below'].values,
                name='5Y',
                line=dict(color=color, width=1.5),
                fill='tozeroy',
                fillcolor=f"rgba{tuple(list(int(color[i:i+2], 16) for i in (1, 3, 5)) + [0.15])}",
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Below EMA: %{y:.1f}%<extra></extra>",
            ),
            row=1, col=1
        )
    
    # 10Y Chart
    if data_10y[ema_period]['percent_below'] is not None:
        fig.add_trace(
            go.Scatter(
                x=data_10y[ema_period]['percent_below'].index,
                y=data_10y[ema_period]['percent_below'].values,
                name='10Y',
                line=dict(color=color, width=1.5),
                fill='tozeroy',
                fillcolor=f"rgba{tuple(list(int(color[i:i+2], 16) for i in (1, 3, 5)) + [0.15])}",
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Below EMA: %{y:.1f}%<extra></extra>",
            ),
            row=1, col=2
        )
    
    # Add critical threshold lines
    for col in [1, 2]:
        # 70% threshold - Critical weakness
        fig.add_hline(y=70, line_dash="dash", line_color=theme['accent_red'], 
                     line_width=1, row=1, col=col, opacity=0.7,
                     annotation_text="Critical (70%)" if col == 1 else None,
                     annotation_position="left")
        # 50% threshold - Warning
        fig.add_hline(y=50, line_dash="dot", line_color=theme['accent_orange'], 
                     line_width=1, row=1, col=col, opacity=0.5)
        # 30% threshold - Healthy
        fig.add_hline(y=30, line_dash="dash", line_color=theme['accent_green'], 
                     line_width=1, row=1, col=col, opacity=0.7,
                     annotation_text="Healthy (30%)" if col == 1 else None,
                     annotation_position="left")
    
    # Update layout
    fig.update_layout(
        height=350,
        plot_bgcolor=theme['bg_secondary'],
        paper_bgcolor=theme['bg_primary'],
        font=dict(family="Inter, sans-serif", color=theme['text_primary'], size=11),
        hovermode='x unified',
        showlegend=False,
        margin=dict(l=60, r=40, t=50, b=50),
    )
    
    # Update axes
    for col in [1, 2]:
        fig.update_xaxes(
            gridcolor=theme['border'], 
            gridwidth=0.5,
            showline=True,
            linecolor=theme['border'],
            row=1, col=col
        )
        fig.update_yaxes(
            title_text="% Below EMA" if col == 1 else "",
            range=[0, 100], 
            gridcolor=theme['border'],
            gridwidth=0.5,
            showline=True,
            linecolor=theme['border'],
            ticksuffix="%",
            row=1, col=col
        )
    
    return fig


def create_historical_table(yearly_data, current_breadth, ema_period, theme):
    """Create a professional comparison table for historical vs current breadth."""
    
    if ema_period not in yearly_data:
        return None
    
    data = yearly_data[ema_period]
    total = data['total']
    
    rows = []
    years = sorted([y for y in data['max_percent'].index if 2008 <= y <= 2024])
    
    for year in years:
        max_pct = data['max_percent'].get(year, 0)
        max_count = data['max_count'].get(year, 0)
        max_date = data['max_date'].get(year)
        date_str = max_date.strftime('%d %b') if pd.notna(max_date) else 'N/A'
        
        rows.append({
            'Year': year,
            'Worst Breadth %': f"{max_pct:.1f}%",
            'Stocks Below': f"{int(max_count)}/{total}",
            'Date': date_str
        })
    
    df = pd.DataFrame(rows)
    return df


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem; background: linear-gradient(135deg, #58A6FF 0%, #A371F7 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            📊 EMA Breadth Analysis
        </h1>
        <p style="color: #8B949E; font-size: 1rem; font-family: 'Inter', sans-serif;">
            Institutional-Grade Market Breadth Dashboard | Stocks Trading Below EMA Levels
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Index Selection
    col1, col2 = st.columns([2, 3])
    
    with col1:
        selected_index = st.selectbox(
            "**Select Index**",
            list(INDEX_CONFIG.keys()),
            index=0,
            help="Choose the market index to analyze"
        )
    
    with col2:
        st.markdown(f"""
        <div style="background: {THEME['bg_card']}; padding: 1rem; border-radius: 8px; margin-top: 0.5rem;">
            <span style="color: {THEME['accent_blue']};">ℹ️</span>
            <span style="color: {THEME['text_secondary']};">{INDEX_CONFIG[selected_index]['description']}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Load tickers
    tickers, company_info, error = load_tickers_from_csv(INDEX_CONFIG[selected_index]['csv_name'])
    
    if error:
        st.error(f"❌ Error loading tickers: {error}")
        st.stop()
    
    st.success(f"✅ **{len(tickers)} stocks loaded** from {selected_index}")
    
    st.divider()
    
    # Initialize analyzer
    analyzer = InstitutionalBreadthAnalyzer()
    
    # Main Tabs
    tab1, tab2 = st.tabs(["📈 **5Y vs 10Y Comparison**", "📉 **Historical Extremes (2008-2025)**"])
    
    # =========================================================================
    # TAB 1: 5Y vs 10Y COMPARISON
    # =========================================================================
    with tab1:
        st.markdown("### 5-Year vs 10-Year Breadth Analysis")
        st.markdown("""
        <p style="color: #8B949E; margin-bottom: 1.5rem;">
            Compare the percentage of stocks trading <b>below</b> each EMA level across different timeframes.
            Higher values indicate market weakness.
        </p>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 **FETCH 5Y & 10Y DATA**", type="primary", use_container_width=True):
            today = datetime.now()
            start_5y = today - timedelta(days=365*5 + 30)
            start_10y = today - timedelta(days=365*10 + 30)
            
            with st.spinner("Fetching 5-year data..."):
                breadth_5y = analyzer.calculate_breadth_below_ema(tickers, start_5y, today)
            
            if breadth_5y:
                with st.spinner("Fetching 10-year data..."):
                    breadth_10y = analyzer.calculate_breadth_below_ema(tickers, start_10y, today)
                
                if breadth_10y:
                    st.session_state['breadth_5y'] = breadth_5y
                    st.session_state['breadth_10y'] = breadth_10y
                    st.session_state['current_5y'] = analyzer.get_current_breadth(breadth_5y)
                    st.session_state['current_10y'] = analyzer.get_current_breadth(breadth_10y)
                    st.rerun()
        
        # Display data if available
        if 'breadth_5y' in st.session_state and 'breadth_10y' in st.session_state:
            breadth_5y = st.session_state['breadth_5y']
            breadth_10y = st.session_state['breadth_10y']
            current_5y = st.session_state['current_5y']
            current_10y = st.session_state['current_10y']
            
            # Current Status Summary
            st.markdown("#### 📊 Current Market Breadth Status")
            
            cols = st.columns(4)
            for idx, ema in enumerate(EMA_PERIODS):
                with cols[idx]:
                    pct = current_10y[ema]['percent']
                    count = current_10y[ema]['count']
                    total = current_10y[ema]['total']
                    
                    # Determine status
                    if pct >= 70:
                        status_class = "critical"
                        status_color = THEME['accent_red']
                        status_text = "CRITICAL"
                    elif pct >= 50:
                        status_class = "warning"
                        status_color = THEME['accent_orange']
                        status_text = "WARNING"
                    else:
                        status_class = "healthy"
                        status_color = THEME['accent_green']
                        status_text = "HEALTHY"
                    
                    st.markdown(f"""
                    <div class="metric-card metric-card-{status_class}">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                            <span style="font-family: 'Inter', sans-serif; font-weight: 600; color: {THEME['ema_colors'][ema]};">
                                EMA-{ema}
                            </span>
                            <span style="font-size: 0.7rem; padding: 2px 8px; background: {status_color}22; color: {status_color}; border-radius: 4px; font-weight: 600;">
                                {status_text}
                            </span>
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 2rem; font-weight: 700; color: {THEME['text_primary']};">
                            {pct:.1f}%
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; color: {THEME['text_secondary']};">
                            {count}/{total} stocks below
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.divider()
            
            # Individual EMA Charts
            for ema in EMA_PERIODS:
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 10px; margin: 1.5rem 0 1rem 0;">
                    <div style="width: 4px; height: 24px; background: {THEME['ema_colors'][ema]}; border-radius: 2px;"></div>
                    <h4 style="margin: 0; color: {THEME['text_primary']};">EMA-{ema} Breadth Analysis</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Stats row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "5Y Current",
                        f"{current_5y[ema]['percent']:.1f}%",
                        f"{current_5y[ema]['count']}/{current_5y[ema]['total']}"
                    )
                
                with col2:
                    if breadth_5y[ema]['percent_below'] is not None:
                        avg_5y = breadth_5y[ema]['percent_below'].mean()
                        st.metric("5Y Average", f"{avg_5y:.1f}%")
                
                with col3:
                    st.metric(
                        "10Y Current",
                        f"{current_10y[ema]['percent']:.1f}%",
                        f"{current_10y[ema]['count']}/{current_10y[ema]['total']}"
                    )
                
                with col4:
                    if breadth_10y[ema]['percent_below'] is not None:
                        avg_10y = breadth_10y[ema]['percent_below'].mean()
                        st.metric("10Y Average", f"{avg_10y:.1f}%")
                
                # Chart
                fig = create_comparison_chart(breadth_5y, breadth_10y, ema, THEME)
                st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
    
    # =========================================================================
    # TAB 2: HISTORICAL EXTREMES
    # =========================================================================
    with tab2:
        st.markdown("### Historical Breadth Extremes (2008-2025)")
        st.markdown("""
        <p style="color: #8B949E; margin-bottom: 1.5rem;">
            Track the <b>worst market breadth</b> (highest % of stocks below EMA) for each year since 2008.
            Compare historical extremes against current levels to assess market conditions.
        </p>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 **FETCH HISTORICAL DATA (2008-2025)**", type="primary", use_container_width=True):
            start_date = datetime(2008, 1, 1)
            today = datetime.now()
            
            with st.spinner("Fetching 17 years of historical data... This may take several minutes."):
                breadth_historical = analyzer.calculate_breadth_below_ema(tickers, start_date, today)
            
            if breadth_historical:
                st.session_state['breadth_historical'] = breadth_historical
                st.session_state['yearly_extremes'] = analyzer.get_yearly_extremes(breadth_historical)
                st.session_state['current_historical'] = analyzer.get_current_breadth(breadth_historical)
                st.rerun()
        
        # Display data if available
        if 'breadth_historical' in st.session_state:
            yearly_extremes = st.session_state['yearly_extremes']
            current_hist = st.session_state['current_historical']
            
            # Current vs Historical Comparison
            st.markdown("#### 📊 Current vs Historical Extremes")
            
            for ema in EMA_PERIODS:
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 10px; margin: 2rem 0 1rem 0;">
                    <div style="width: 4px; height: 24px; background: {THEME['ema_colors'][ema]}; border-radius: 2px;"></div>
                    <h4 style="margin: 0; color: {THEME['text_primary']};">EMA-{ema} — Worst Breadth by Year</h4>
                </div>
                """, unsafe_allow_html=True)
                
                # Current status card
                curr = current_hist[ema]
                
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    pct = curr['percent']
                    if pct >= 70:
                        status_color = THEME['accent_red']
                        status_text = "CRITICAL"
                    elif pct >= 50:
                        status_color = THEME['accent_orange']
                        status_text = "WARNING"
                    else:
                        status_color = THEME['accent_green']
                        status_text = "HEALTHY"
                    
                    st.markdown(f"""
                    <div style="background: {THEME['bg_card']}; padding: 1.5rem; border-radius: 12px; border: 1px solid {THEME['border']}; text-align: center;">
                        <div style="font-size: 0.8rem; color: {THEME['text_secondary']}; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">
                            Current Status
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 2.5rem; font-weight: 700; color: {status_color};">
                            {pct:.1f}%
                        </div>
                        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1rem; color: {THEME['text_secondary']}; margin-top: 0.5rem;">
                            {curr['count']}/{curr['total']} below
                        </div>
                        <div style="margin-top: 1rem; padding: 0.25rem 0.75rem; background: {status_color}22; color: {status_color}; border-radius: 4px; display: inline-block; font-weight: 600; font-size: 0.75rem;">
                            {status_text}
                        </div>
                        <div style="font-size: 0.75rem; color: {THEME['text_secondary']}; margin-top: 0.5rem;">
                            as of {curr['date']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    # Historical table
                    df = create_historical_table(yearly_extremes, current_hist, ema, THEME)
                    if df is not None:
                        # Style the dataframe
                        st.dataframe(
                            df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Year": st.column_config.NumberColumn("Year", format="%d"),
                                "Worst Breadth %": st.column_config.TextColumn("Worst Breadth"),
                                "Stocks Below": st.column_config.TextColumn("Count"),
                                "Date": st.column_config.TextColumn("Date")
                            }
                        )
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            f"📥 Download EMA-{ema} Historical Data",
                            csv,
                            f"ema{ema}_yearly_extremes_2008_2025.csv",
                            "text/csv",
                            use_container_width=True
                        )
                
                st.divider()
    
    # Footer
    st.markdown(f"""
    <div style="text-align: center; padding: 2rem 0; color: {THEME['text_secondary']}; font-size: 0.8rem;">
        <p>📊 EMA Breadth Analysis Dashboard | Data sourced from Yahoo Finance</p>
        <p>Market breadth measures the percentage of stocks trading below their EMA levels.</p>
        <p>Higher values indicate broader market weakness.</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# SIDEBAR - CACHE MANAGEMENT
# =============================================================================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    
    if st.button("🗑️ Clear Cache", help="Clear all cached data"):
        cache_dir = Path("cache")
        if cache_dir.exists():
            for file in cache_dir.glob("*.pkl"):
                file.unlink()
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("✅ Cache cleared!")
        st.rerun()
    
    st.divider()
    
    st.markdown("### 📖 Interpretation Guide")
    
    st.markdown(f"""
    <div style="font-size: 0.85rem; color: {THEME['text_secondary']};">
        <p><b style="color: {THEME['accent_green']};">● Healthy (&lt;30%)</b><br>
        Strong market breadth. Most stocks above EMA.</p>
        
        <p><b style="color: {THEME['accent_orange']};">● Warning (30-70%)</b><br>
        Deteriorating breadth. Caution advised.</p>
        
        <p><b style="color: {THEME['accent_red']};">● Critical (&gt;70%)</b><br>
        Severe weakness. Potential capitulation or bear market.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("### 📊 EMA Reference")
    st.markdown(f"""
    <div style="font-size: 0.85rem; color: {THEME['text_secondary']};">
        <p><b style="color: {THEME['ema_colors'][20]};">EMA-20</b> — Short-term trend</p>
        <p><b style="color: {THEME['ema_colors'][50]};">EMA-50</b> — Medium-term trend</p>
        <p><b style="color: {THEME['ema_colors'][100]};">EMA-100</b> — Long-term trend</p>
        <p><b style="color: {THEME['ema_colors'][200]};">EMA-200</b> — Major trend</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

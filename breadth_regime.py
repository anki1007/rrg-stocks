import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Historical Breadth Regime Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

THEMES = {
    "Dark": {
        "bg_color": "#161614",
        "text_color": "#FFD700",
        "grid_color": "#2a2a2a",
        "ema_colors": {"20": "#FF6B9D", "50": "#4ECDC4", "100": "#95E1D3", "200": "#FF6348"},
        "up_color": "#00D084",
        "down_color": "#FF5E78",
    },
}

INDEX_CONFIG = {
    "Nifty 50": {"csv_name": "ticker/nifty50.csv", "description": "Top 50 Large Cap Stocks"},
    "Nifty 100": {"csv_name": "ticker/nifty100.csv", "description": "Top 100 Large Cap Stocks"},
    "Nifty 200": {"csv_name": "ticker/nifty200.csv", "description": "Top 200 Large Cap and Mid Cap Stocks"},
    "Nifty Total Market": {"csv_name": "ticker/niftytotalmarket.csv", "description": "Nifty Total Market Index"},
}

with st.sidebar:
    st.markdown("Settings")
    selected_theme = st.selectbox("Theme", list(THEMES.keys()), index=0)
    theme = THEMES[selected_theme]
    max_workers = st.slider("Data Fetch Threads", 5, 20, 10)

@st.cache_data(ttl=3600)
def load_tickers_from_csv(csv_filename):
    try:
        df = pd.read_csv(csv_filename)
        for col_name in ['Symbol', 'SYMBOL', 'Ticker', 'ticker', 'symbol']:
            if col_name in df.columns:
                tickers = sorted(df[col_name].unique().tolist())
                return tickers, None
        return None, "Could not find Symbol column"
    except Exception as e:
        return None, f"Error: {str(e)}"

class HistoricalBreadthAnalyzer:
    def __init__(self, max_workers, theme):
        self.max_workers = max_workers
        self.ema_period = 50
        self.theme = theme
    
    def get_stock_data(self, ticker, start_date, end_date, max_retries=1):
        for attempt in range(max_retries):
            try:
                hist = yf.download(ticker, start=start_date, end=end_date, progress=False, timeout=5)
                if hist.empty or len(hist) < 100:
                    return None
                return hist['Close'].dropna()
            except:
                if attempt == max_retries - 1:
                    return None
        return None
    
    @staticmethod
    def calculate_ema(data, period):
        return data.ewm(span=period, adjust=False).mean()
    
    def analyze_ticker(self, ticker, start_date, end_date):
        data = self.get_stock_data(ticker, start_date, end_date)
        if data is None or len(data) < 100:
            return None
        ema = self.calculate_ema(data, self.ema_period)
        above = (data.values > ema.values).astype(int)
        return pd.Series(above, index=data.index)
    
    def calculate_breadth(self, tickers, start_date, end_date):
        container = st.container()
        progress_bar = container.progress(0)
        status_text = container.empty()
        
        all_results = {}
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self.analyze_ticker, ticker, start_date, end_date) for ticker in tickers]
            for idx, future in enumerate(futures):
                try:
                    result = future.result(timeout=10)
                    if result is not None:
                        all_results[tickers[idx]] = result
                except:
                    pass
                completed += 1
                progress_bar.progress(completed / len(tickers))
                status_text.text(f"Fetched: {completed}/{len(tickers)}")
        
        progress_bar.empty()
        status_text.empty()
        
        if not all_results:
            return None
        
        all_dates = set()
        for series in all_results.values():
            all_dates.update(series.index)
        all_dates = sorted(list(all_dates))
        series_list = [series.reindex(all_dates, method='ffill') for series in all_results.values()]
        df_combined = pd.concat(series_list, axis=1, ignore_index=True).dropna()
        
        breadth_percent = (df_combined.mean(axis=1) * 100).round(2)
        breadth_count = df_combined.sum(axis=1).astype(int)
        
        return {'percent': breadth_percent, 'count': breadth_count, 'total': len(all_results)}

def main():
    st.markdown("Historical Breadth Regime Analysis")
    st.markdown("Compare 5-year vs 10-year breadth patterns. Track all-time lows since 2000.")
    st.divider()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_index = st.selectbox("Select Index", list(INDEX_CONFIG.keys()), index=0)
    with col2:
        st.metric("Index", selected_index)
    
    st.info(f"Info: {INDEX_CONFIG[selected_index]['description']}")
    st.divider()
    
    selected_tickers, error = load_tickers_from_csv(INDEX_CONFIG[selected_index]['csv_name'])
    if error:
        st.error(f"Error: {error}")
        st.stop()
    if selected_tickers:
        st.success(f"Loaded {len(selected_tickers)} tickers")
    else:
        st.error("No tickers found")
        st.stop()
    
    st.divider()
    
    tab1, tab2, tab3 = st.tabs(["5Y vs 10Y Comparison", "All-Time Lows 2000-2025", "Detailed Table"])
    
    with tab1:
        st.markdown("5-Year vs 10-Year Breadth Comparison")
        if st.button("FETCH 5Y & 10Y DATA", key="fetch_5y_10y", type="primary", use_container_width=True):
            analyzer = HistoricalBreadthAnalyzer(max_workers, theme)
            today = datetime.now()
            start_5y = today - timedelta(days=365*5)
            start_10y = today - timedelta(days=365*10)
            
            with st.spinner("Fetching 5-year data..."):
                breadth_5y = analyzer.calculate_breadth(selected_tickers, start_5y, today)
            with st.spinner("Fetching 10-year data..."):
                breadth_10y = analyzer.calculate_breadth(selected_tickers, start_10y, today)
            
            if breadth_5y and breadth_10y:
                st.success("Data fetched successfully")
                st.session_state['breadth_5y'] = breadth_5y
                st.session_state['breadth_10y'] = breadth_10y
                st.session_state['selected_index'] = selected_index
        
        if 'breadth_5y' in st.session_state and 'breadth_10y' in st.session_state:
            breadth_5y = st.session_state['breadth_5y']
            breadth_10y = st.session_state['breadth_10y']
            
            fig = make_subplots(rows=2, cols=1, subplot_titles=("Last 5 Years", "Last 10 Years"), vertical_spacing=0.12)
            fig.add_trace(go.Scatter(x=breadth_5y['percent'].index, y=breadth_5y['percent'].values, name="5Y", line=dict(color="#4ECDC4", width=2), fill='tozeroy'), row=1, col=1)
            fig.add_trace(go.Scatter(x=breadth_10y['percent'].index, y=breadth_10y['percent'].values, name="10Y", line=dict(color="#95E1D3", width=2), fill='tozeroy'), row=2, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#00D084", row=1)
            fig.add_hline(y=30, line_dash="dash", line_color="#FF5E78", row=1)
            fig.add_hline(y=70, line_dash="dash", line_color="#00D084", row=2)
            fig.add_hline(y=30, line_dash="dash", line_color="#FF5E78", row=2)
            fig.update_xaxes(title_text="Date", row=2, col=1, gridcolor="#2a2a2a")
            fig.update_yaxes(title_text="Breadth %", row=1, col=1, gridcolor="#2a2a2a", range=[0, 100])
            fig.update_yaxes(title_text="Breadth %", row=2, col=1, gridcolor="#2a2a2a", range=[0, 100])
            fig.update_layout(height=800, plot_bgcolor="#161614", paper_bgcolor="#161614", font=dict(color="#FFD700", size=11), hovermode='x unified', margin=dict(l=60, r=60, t=80, b=60))
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("STATISTICS COMPARISON")
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("5Y Current", f"{breadth_5y['percent'].iloc[-1]:.2f}%")
            with col2:
                st.metric("5Y Average", f"{breadth_5y['percent'].mean():.2f}%")
            with col3:
                st.metric("5Y High", f"{breadth_5y['percent'].max():.2f}%")
            with col4:
                st.metric("5Y Low", f"{breadth_5y['percent'].min():.2f}%")
            with col5:
                st.metric("5Y StdDev", f"{breadth_5y['percent'].std():.2f}%")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("10Y Current", f"{breadth_10y['percent'].iloc[-1]:.2f}%")
            with col2:
                st.metric("10Y Average", f"{breadth_10y['percent'].mean():.2f}%")
            with col3:
                st.metric("10Y High", f"{breadth_10y['percent'].max():.2f}%")
            with col4:
                st.metric("10Y Low", f"{breadth_10y['percent'].min():.2f}%")
            with col5:
                st.metric("10Y StdDev", f"{breadth_10y['percent'].std():.2f}%")

if __name__ == "__main__":
    main()

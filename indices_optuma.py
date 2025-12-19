import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ==================== YOUR CONFIG ====================
GITHUB_USER = "anki1007"
GITHUB_REPO = "rrg-stocks"
GITHUB_BRANCH = "main"
GITHUB_TICKER_DIR = "ticker"
CSV_BASENAME = "niftyindices.csv"

BENCHMARKS = {
    "Nifty 50": "^NSEI",
    "Nifty 200": "^CNX200", 
    "Nifty 500": "^CRSLDX"
}

TIMEFRAMES = {
    "Daily": ("1d", "5y"),
    "Weekly": ("1wk", "10y"),
    "Monthly": ("1mo", "20y")
}

WINDOW = 14

# Streamlit page config
st.set_page_config(page_title="RRG Dashboard", layout="wide")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #111827 !important; }
h1, h2, h3 { color: #ffffff !important; }
</style>
""", unsafe_allow_html=True)

st.title("📊 RRG Dashboard - Indian Indices")

# ==================== LOAD CSV ====================
@st.cache_data(ttl=600)
def load_universe():
    try:
        url = f"https://raw.githubusercontent.com/{GITHUB_USER}/{GITHUB_REPO}/{GITHUB_BRANCH}/{GITHUB_TICKER_DIR}/{CSV_BASENAME}"
        df = pd.read_csv(url)
        symbols = df['Symbol'].dropna().str.strip().tolist()
        symbols = [s if s.startswith('^') else '^' + s for s in symbols]
        return symbols[:50]  # Top 50 for speed
    except:
        return ["^NSEI", "^NIFTY_BANK", "^CNXIT", "^CNXAUTO"]

# ==================== MAIN APP ====================
universe = load_universe()
bench_name = st.sidebar.selectbox("Benchmark", list(BENCHMARKS.keys()))
tf_name = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES.keys()))
period_name = st.sidebar.selectbox("Period", ["1y", "2y", "5y"])

if st.sidebar.button("🚀 LOAD DATA", type="primary"):
    interval, period = TIMEFRAMES[tf_name]
    bench_symbol = BENCHMARKS[bench_name]
    
    with st.spinner('Downloading data...'):
        try:
            # Download data
            data = yf.download(universe + [bench_symbol], period=period, interval=interval, progress=False)
            
            # Extract benchmark
            bench_close = data['Close'][bench_symbol].dropna()
            
            results = []
            for sym in universe:
                if sym in data['Close'].columns:
                    sym_close = data['Close'][sym].dropna()
                    
                    # Align data
                    min_len = min(len(sym_close), len(bench_close))
                    sym_close = sym_close[-min_len:]
                    bench_close_aligned = bench_close[-min_len:]
                    
                    if len(sym_close) > WINDOW * 2:
                        # RS Ratio
                        rs = 100 * (sym_close / bench_close_aligned)
                        rs_ratio = 100 + (rs - rs.rolling(WINDOW).mean()) / rs.rolling(WINDOW).std()
                        
                        # RS Momentum  
                        rroc = rs_ratio.pct_change() * 100
                        rs_momentum = 101 + (rroc - rroc.rolling(WINDOW).mean()) / rroc.rolling(WINDOW).std()
                        
                        latest_rr = rs_ratio.iloc[-1]
                        latest_rm = rs_momentum.iloc[-1]
                        
                        # Quadrant
                        if latest_rr >= 100 and latest_rm >= 100:
                            quadrant = "🟢 Leading"
                        elif latest_rr < 100 and latest_rm >= 100:
                            quadrant = "🔵 Improving" 
                        elif latest_rr >= 100 and latest_rm < 100:
                            quadrant = "🟡 Weakening"
                        else:
                            quadrant = "🔴 Lagging"
                        
                        results.append({
                            'Symbol': sym.replace('^', ''),
                            'RS-Ratio': round(latest_rr, 2),
                            'RS-Momentum': round(latest_rm, 2),
                            'Quadrant': quadrant,
                            'Price': round(sym_close.iloc[-1], 2)
                        })
            
            df = pd.DataFrame(results)
            st.session_state.df = df.sort_values('RS-Ratio', ascending=False)
            
        except Exception as e:
            st.error(f"Error: {e}")

# ==================== DISPLAY ====================
if 'df' in st.session_state:
    df = st.session_state.df
    
    col1, col2, col3 = st.columns([1, 3, 1])
    
    with col1:
        st.metric("Total", len(df))
        st.metric("🟢 Leading", len(df[df['Quadrant']=='🟢 Leading']))
        st.metric("🔵 Improving", len(df[df['Quadrant']=='🔵 Improving']))
    
    with col2:
        fig = go.Figure()
        
        # Quadrants
        fig.add_shape(type="rect", x0=100, y0=100, x1=110, y1=110, 
                     fillcolor="rgba(34,197,94,0.2)", line=dict(color="green"))
        fig.add_shape(type="rect", x0=90, y0=100, x1=100, y1=110, 
                     fillcolor="rgba(59,130,246,0.2)", line=dict(color="blue"))
        
        # Plot points
        for quad in df['Quadrant'].unique():
            df_quad = df[df['Quadrant']==quad]
            fig.add_trace(go.Scatter(x=df_quad['RS-Ratio'], y=df_quad['RS-Momentum'],
                                   mode='markers+text', text=df_quad['Symbol'],
                                   marker=dict(size=12, color='green' if 'Leading' in quad else 'blue'),
                                   textposition="top center", name=quad))
        
        fig.update_layout(title="Relative Rotation Graph", xaxis_title="RS-Ratio", yaxis_title="RS-Momentum",
                         height=500, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("Top 10")
        st.dataframe(df.head(10)[['Symbol', 'RS-Ratio', 'RS-Momentum', 'Quadrant', 'Price']])
    
    st.dataframe(df)

else:
    st.info("👆 Select parameters and click **LOAD DATA**")
    
    st.markdown("""
    ## Quick Start
    1. **Benchmark**: Nifty 50/200/500
    2. **Timeframe**: Daily/Weekly/Monthly  
    3. **Period**: 1Y/2Y/5Y
    4. Click **LOAD DATA**
    """)

st.markdown("---")
st.caption("📈 RRG Analysis | Data: Yahoo Finance | CSV: GitHub")

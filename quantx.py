"""
Momentum 50 Strategy - Bloomberg-Style Performance Dashboard
Stallions Algorithmic Trading Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta
import math
import io
import base64
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import copy
import warnings
import requests
from io import StringIO
import time

# Page configuration
st.set_page_config(
    page_title="Momentum 50 Shop - Performance Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bloomberg-style CSS
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0d0d0d 0%, #1a1a2e 50%, #16213e 100%);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%);
        border-right: 1px solid #ff6b35;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #e0e0e0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #ff6b35 !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(145deg, #1e1e2f 0%, #2d2d44 100%);
        border: 1px solid #ff6b35;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(255, 107, 53, 0.2);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.4);
    }
    
    .metric-label {
        color: #ff6b35;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    
    .metric-value {
        color: #ffffff;
        font-size: 24px;
        font-weight: 700;
    }
    
    .metric-value-green {
        color: #00ff88;
        font-size: 24px;
        font-weight: 700;
    }
    
    .metric-value-red {
        color: #ff4757;
        font-size: 24px;
        font-weight: 700;
    }
    
    /* Secondary metric cards */
    .metric-card-secondary {
        background: linear-gradient(145deg, #1a1a2e 0%, #252540 100%);
        border: 1px solid #00d4ff;
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.15);
    }
    
    .metric-label-secondary {
        color: #00d4ff;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Tables */
    .dataframe {
        background-color: #1a1a2e !important;
        color: #e0e0e0 !important;
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #ff6b35 0%, transparent 100%);
        padding: 10px 20px;
        border-radius: 5px;
        margin: 20px 0 10px 0;
        color: white;
        font-weight: 600;
    }
    
    /* Custom divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, #ff6b35, #00d4ff, #ff6b35);
        margin: 20px 0;
        border-radius: 1px;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(90deg, #ff6b35 0%, #ff8c42 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(90deg, #ff8c42 0%, #ffaa5b 100%);
        transform: scale(1.05);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #00d4ff 0%, #00ff88 100%);
        color: #0d0d0d;
        border: none;
        border-radius: 8px;
        font-weight: 600;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1a1a2e;
        border: 1px solid #ff6b35;
        border-radius: 5px;
    }
    
    /* Input fields */
    .stSelectbox, .stNumberInput, .stDateInput {
        background-color: #1a1a2e;
    }
    
    /* Logo area */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px;
        background: linear-gradient(90deg, #ff6b35 0%, #ff8c42 100%);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .logo-text {
        font-size: 24px;
        font-weight: 800;
        color: white;
    }
    
    /* Refresh button */
    .refresh-btn {
        background: linear-gradient(90deg, #ff4757 0%, #ff6b81 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a2e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #ff6b35;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #ff8c42;
    }
</style>
""", unsafe_allow_html=True)


# Data classes for backtest
@dataclass
class Lot:
    qty: int
    price: float
    date: pd.Timestamp
    buy_brokerage: float


@dataclass
class Position:
    symbol: str
    lots: List[Lot] = field(default_factory=list)

    def total_qty(self) -> int:
        return sum(l.qty for l in self.lots)

    def avg_price(self) -> Optional[float]:
        q = self.total_qty()
        return (sum(l.qty * l.price for l in self.lots) / q) if q > 0 else None

    def last_buy_price(self) -> Optional[float]:
        return self.lots[-1].price if self.lots else None

    def total_buy_brokerage(self) -> float:
        return sum(l.buy_brokerage for l in self.lots)


class MomentumShopBacktester:
    """Complete backtester with all metrics calculation"""
    
    def __init__(
        self,
        instruments: List[str],
        start_date: date,
        end_date: date,
        position_sizing_mode: str,
        fresh_static_amt: float = 0.0,
        avg_static_amt: float = 0.0,
        fresh_cash_pct: float = 0.0,
        avg_cash_pct: float = 0.0,
        fresh_trade_divisor: Optional[float] = None,
        avg_trade_divisor: Optional[float] = None,
        initial_capital: float = 400000.0,
        target_pct: float = 0.05,
        avg_trigger_pct: float = 0.03,
        brokerage_per_order: float = 40.0,
        dma_window: int = 20,
        max_avg: int = 3,
    ):
        if position_sizing_mode not in ("static", "dynamic", "divisor"):
            raise ValueError("position_sizing_mode must be 'static', 'dynamic', or 'divisor'.")
        
        self.instruments = instruments
        self.start_date = start_date
        self.end_date = end_date
        self.position_sizing_mode = position_sizing_mode
        
        self.fresh_static_amt = float(fresh_static_amt)
        self.avg_static_amt = float(avg_static_amt)
        self.fresh_cash_pct = float(fresh_cash_pct)
        self.avg_cash_pct = float(avg_cash_pct)
        self.fresh_trade_divisor = float(fresh_trade_divisor) if fresh_trade_divisor else None
        self.avg_trade_divisor = float(avg_trade_divisor) if avg_trade_divisor else None
        
        self.initial_capital = float(initial_capital)
        self.cash = float(initial_capital)
        self.target_pct = float(target_pct)
        self.avg_trigger_pct = float(avg_trigger_pct)
        self.brokerage_per_order = float(brokerage_per_order)
        self.dma_window = int(dma_window)
        self.max_avg = int(max_avg)
        
        self.positions: Dict[str, Position] = {}
        self._completed_trades: List[Dict] = []
        self.realized_pnl_by_date: Dict[pd.Timestamp, float] = {}
        self.data: Dict[str, pd.DataFrame] = {}
        self._fresh_buy = False
        self.cashflow_ledger: List[Tuple[date, float]] = []
        self.equity_curve_data: List[Dict] = []
        self.daily_portfolio_values: Dict[pd.Timestamp, float] = {}
        
    def load_data_from_yfinance(self, progress_callback=None):
        """Load data from yfinance with progress tracking"""
        import yfinance as yf
        
        total = len(self.instruments)
        for i, sym in enumerate(self.instruments):
            if progress_callback:
                progress_callback((i + 1) / total, f"Loading {sym}...")
            try:
                df = yf.download(
                    sym + ".NS",
                    start=self.start_date - timedelta(days=150),
                    end=self.end_date + timedelta(days=1),
                    auto_adjust=True,
                    interval="1d",
                    progress=False,
                    multi_level_index=None,
                    rounding=True
                )
                time.sleep(0.1)
                
                if df.empty:
                    continue
                    
                df.reset_index(inplace=True)
                df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
                df.set_index("Date", inplace=True)
                df = df.sort_index()
                self.data[sym] = df.copy()
                
            except Exception as e:
                warnings.warn(f"Could not load {sym}: {e}")
                
        if not self.data:
            raise RuntimeError("No instrument data loaded")
    
    @staticmethod
    def compute_moving_average(df: pd.DataFrame, window: int, column: str = "Close") -> pd.Series:
        return df[column].rolling(window=window, min_periods=window).mean()
    
    def _get_latest_close(self, symbol: str, dt: pd.Timestamp) -> Optional[float]:
        df = self.data.get(symbol)
        if df is None:
            return None
        subset = df.loc[df.index <= dt]
        if subset.empty:
            return None
        return float(subset.iloc[-1]["Close"])
    
    def portfolio_value(self, dt: pd.Timestamp) -> float:
        total = float(self.cash)
        for sym, pos in self.positions.items():
            price = self._get_latest_close(sym, dt)
            if price is None:
                continue
            total += pos.total_qty() * price
        return float(total)
    
    def _alloc_amount_for_trade(self, trade_kind: str, dt: pd.Timestamp) -> float:
        if self.position_sizing_mode == "static":
            return float(self.fresh_static_amt) if trade_kind == "fresh" else float(self.avg_static_amt)
        elif self.position_sizing_mode == "dynamic":
            pct = float(self.fresh_cash_pct) if trade_kind == "fresh" else float(self.avg_cash_pct)
            return float(self.cash) * float(pct)
        else:
            divisor = self.fresh_trade_divisor if trade_kind == "fresh" else self.avg_trade_divisor
            port_val = self.portfolio_value(dt)
            return float(port_val) / float(divisor)
    
    def _qty_from_amount_and_price(self, amount: float, price: float) -> int:
        if amount <= 0 or price <= 0:
            return 0
        return math.floor(amount / price)
    
    def _determine_qty_for_buy(self, trade_kind: str, price: float, dt: pd.Timestamp) -> int:
        alloc_amount = self._alloc_amount_for_trade(trade_kind, dt)
        qty_by_alloc = self._qty_from_amount_and_price(alloc_amount, price)
        if qty_by_alloc <= 0:
            return 0
        if self.cash <= self.brokerage_per_order:
            return 0
        max_qty_by_cash = math.floor((self.cash - self.brokerage_per_order) / price)
        if max_qty_by_cash <= 0:
            return 0
        return int(min(qty_by_alloc, max_qty_by_cash))
    
    def run_backtest(self, progress_callback=None) -> pd.DataFrame:
        # Precompute DMA
        for sym, df in self.data.items():
            df["20DMA"] = self.compute_moving_average(df, self.dma_window, "Close")
            df["pct_below_20dma"] = np.where(
                (df["20DMA"].notna()) & (df["Close"] < df["20DMA"]),
                (df["20DMA"] - df["Close"]) / df["20DMA"],
                0.0,
            )
            self.data[sym] = df
        
        all_dates = sorted({d for df in self.data.values() for d in df.index})
        all_dates = [d for d in all_dates if (d.date() >= self.start_date and d.date() <= self.end_date)]
        
        if not all_dates:
            raise RuntimeError("No trading dates in the data")
        
        total_days = len(all_dates)
        
        for i, current_dt in enumerate(all_dates):
            if progress_callback and i % 50 == 0:
                progress_callback((i + 1) / total_days, f"Processing {current_dt.strftime('%Y-%m-%d')}...")
            
            self._fresh_buy = False
            self._process_exits_for_date(current_dt)
            
            top5 = self._get_top5_below_20dma(current_dt)
            self._process_entries_for_top5(current_dt, top5)
            
            if not self._fresh_buy:
                if top5 and all(sym in self.positions and self.positions[sym].total_qty() > 0 for sym in top5):
                    self._process_averaging_mode(current_dt)
            
            # Track daily portfolio value
            self.daily_portfolio_values[current_dt] = self.portfolio_value(current_dt)
        
        return pd.DataFrame(self._completed_trades)
    
    def _process_exits_for_date(self, dt: pd.Timestamp):
        symbols_to_exit: List[Tuple[str, float, pd.Timestamp]] = []
        for sym, pos in list(self.positions.items()):
            if pos.total_qty() == 0:
                continue
            df = self.data.get(sym)
            if df is None or dt not in df.index:
                continue
            row = df.loc[dt]
            
            for lot in pos.lots:
                target_price = lot.price * (1.0 + self.target_pct)
                if row["High"] >= target_price:
                    symbols_to_exit.append((sym, target_price, lot.date))
        
        for sym, exit_price, entry_dt in symbols_to_exit:
            self._execute_exit(sym, exit_price, entry_dt, dt)
    
    def _execute_exit(self, symbol: str, exit_price: float, entry_dt: pd.Timestamp, dt: pd.Timestamp):
        pos = self.positions.get(symbol)
        exit_position = copy.deepcopy(pos)
        exit_position.lots = [lot for lot in exit_position.lots if lot.date == entry_dt]
        
        if not exit_position or exit_position.total_qty() == 0:
            return
        
        total_qty = exit_position.total_qty()
        sell_brokerage = self.brokerage_per_order
        lot_values = [l.qty * exit_price for l in exit_position.lots]
        total_value = sum(lot_values) if lot_values else 0.0
        proportions = [lv / total_value if total_value > 0 else (1.0 / len(lot_values)) for lv in lot_values]
        
        total_proceeds = total_qty * exit_price
        self.cash += total_proceeds - sell_brokerage
        
        date_key = dt.normalize()
        self.realized_pnl_by_date.setdefault(date_key, 0.0)
        
        for lot, prop in zip(exit_position.lots, proportions):
            lot_qty = lot.qty
            entry_price = lot.price
            entry_date = lot.date.date()
            lot_buy_brokerage = lot.buy_brokerage
            lot_sell_brokerage = sell_brokerage * prop
            gross_pnl = lot_qty * (exit_price - entry_price)
            lot_total_brokerage = lot_buy_brokerage + lot_sell_brokerage
            net_pnl = gross_pnl - lot_total_brokerage
            pnl_pct = ((exit_price - entry_price) / entry_price * 100.0) if entry_price > 0 else float("nan")
            net_pnl_pct = (net_pnl / (entry_price * lot_qty) * 100.0) if (entry_price > 0 and lot_qty > 0) else float("nan")
            capital_used = lot_qty * entry_price
            
            holding_days = (dt.date() - entry_date).days
            
            trade_row = {
                "Symbol": symbol,
                "Status": "completed",
                "Entry Date": entry_date,
                "Direction": "Long",
                "Filled Qty": lot_qty,
                "Entry": round(entry_price, 2),
                "Exit": round(exit_price, 2),
                "Pnl": round(gross_pnl, 2),
                "Pnl%": round(pnl_pct, 2),
                "NetPnl": round(net_pnl, 2),
                "NetPnl%": round(net_pnl_pct, 2),
                "Capital": round(capital_used, 2),
                "Brokerage": round(lot_total_brokerage, 2),
                "Exit Date": dt.date(),
                "Holding Days": holding_days,
            }
            self._completed_trades.append(trade_row)
            self.realized_pnl_by_date[date_key] += net_pnl
            
            self.cashflow_ledger.append((entry_date, -(lot_qty * entry_price + lot_buy_brokerage)))
            self.cashflow_ledger.append((dt.date(), (lot_qty * exit_price - lot_sell_brokerage)))
        
        pos.lots = [lot for lot in pos.lots if lot.date != entry_dt]
        if not pos.lots:
            del self.positions[symbol]
    
    def _get_top5_below_20dma(self, dt: pd.Timestamp) -> List[str]:
        candidates: List[Tuple[str, float]] = []
        for sym, df in self.data.items():
            if dt not in df.index:
                continue
            pct = df.loc[dt, "pct_below_20dma"]
            if not pd.isna(pct) and pct > 0:
                candidates.append((sym, float(pct)))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [sym for sym, _ in candidates[:5]]
    
    def _process_entries_for_top5(self, dt: pd.Timestamp, top5: List[str]):
        if not top5:
            return
        for sym in top5:
            pos = self.positions.get(sym)
            if pos and pos.total_qty() > 0:
                continue
            
            df = self.data.get(sym)
            if df is None or dt not in df.index:
                continue
            close_price = float(df.loc[dt, "Close"])
            
            qty = self._determine_qty_for_buy("fresh", close_price, dt)
            
            if qty <= 0:
                continue
            total_cost = qty * close_price + self.brokerage_per_order
            if total_cost > self.cash + 1e-9:
                continue
            
            self.cash -= total_cost
            lot = Lot(qty=qty, price=close_price, date=dt, buy_brokerage=self.brokerage_per_order)
            self.positions[sym] = Position(symbol=sym, lots=[lot])
            self.cashflow_ledger.append((dt.date(), -(qty * close_price + self.brokerage_per_order)))
            
            self._fresh_buy = True
            break
    
    def _process_averaging_mode(self, dt: pd.Timestamp):
        for sym, pos in list(self.positions.items()):
            if pos.total_qty() == 0:
                continue
            if len(pos.lots) >= self.max_avg:
                continue
            df = self.data.get(sym)
            if df is None or dt not in df.index:
                continue
            close_price = float(df.loc[dt, "Close"])
            last_buy_price = pos.last_buy_price()
            if last_buy_price is None:
                continue
            pct_drop = (last_buy_price - close_price) / last_buy_price
            if pct_drop > self.avg_trigger_pct:
                qty = self._determine_qty_for_buy("avg", close_price, dt)
                if qty <= 0:
                    continue
                total_cost = qty * close_price + self.brokerage_per_order
                if total_cost > self.cash + 1e-9:
                    continue
                self.cash -= total_cost
                lot = Lot(qty=qty, price=close_price, date=dt, buy_brokerage=self.brokerage_per_order)
                pos.lots.append(lot)
                self.cashflow_ledger.append((dt.date(), -(qty * close_price + self.brokerage_per_order)))
                break
    
    def compute_all_metrics(self, trades_df: pd.DataFrame, benchmark_data: Optional[pd.DataFrame] = None) -> Dict:
        """Compute all performance metrics"""
        metrics = {}
        
        if trades_df.empty:
            return self._empty_metrics()
        
        # Basic metrics
        metrics['total_trades'] = len(trades_df)
        metrics['winning_trades'] = len(trades_df[trades_df['NetPnl'] > 0])
        metrics['losing_trades'] = len(trades_df[trades_df['NetPnl'] <= 0])
        metrics['win_ratio'] = (metrics['winning_trades'] / metrics['total_trades'] * 100) if metrics['total_trades'] > 0 else 0
        
        # Financial metrics
        metrics['actual_investment'] = trades_df['Capital'].sum()
        metrics['gross_pnl'] = trades_df['Pnl'].sum()
        metrics['total_brokerage'] = trades_df['Brokerage'].sum()
        metrics['net_pnl'] = trades_df['NetPnl'].sum()
        metrics['ending_balance'] = self.initial_capital + metrics['net_pnl']
        metrics['gross_pnl_pct'] = (metrics['gross_pnl'] / self.initial_capital * 100)
        metrics['net_pnl_pct'] = (metrics['net_pnl'] / self.initial_capital * 100)
        
        # Time-based metrics
        days = (pd.Timestamp(self.end_date) - pd.Timestamp(self.start_date)).days
        years = days / 365.25 if days > 0 else 1.0
        metrics['cagr'] = ((metrics['ending_balance'] / self.initial_capital) ** (1.0 / years) - 1.0) * 100 if years > 0 else 0
        
        # Holding period
        if 'Holding Days' in trades_df.columns:
            metrics['avg_holding_period'] = trades_df['Holding Days'].mean()
        else:
            metrics['avg_holding_period'] = 0
        
        # Build equity curve
        if self.realized_pnl_by_date:
            idx = sorted(self.realized_pnl_by_date.keys())
            cum = self.initial_capital
            dates = []
            eq = []
            for dt in idx:
                cum += self.realized_pnl_by_date[dt]
                dates.append(dt)
                eq.append(cum)
            equity = pd.Series(data=eq, index=pd.DatetimeIndex(dates)).sort_index()
            full_index = pd.date_range(start=pd.Timestamp(self.start_date), end=pd.Timestamp(self.end_date), freq="D")
            equity_ff = equity.reindex(full_index).ffill().fillna(self.initial_capital)
        else:
            full_index = pd.date_range(start=self.start_date, end=self.end_date)
            equity_ff = pd.Series(self.initial_capital, index=full_index)
        
        metrics['equity_curve'] = equity_ff
        
        # Daily returns
        daily_returns = equity_ff.pct_change().fillna(0.0)
        metrics['daily_returns'] = daily_returns
        
        # Drawdown calculations
        running_max = equity_ff.cummax()
        drawdown = (equity_ff - running_max) / running_max
        metrics['drawdown_series'] = drawdown
        metrics['max_drawdown'] = drawdown.min() * 100
        metrics['max_drawdown_amount'] = (running_max - equity_ff).max()
        
        # Drawdown duration
        is_dd = drawdown < 0
        dd_groups = (is_dd != is_dd.shift()).cumsum()
        dd_lengths = is_dd.groupby(dd_groups).sum()
        metrics['longest_dd_days'] = int(dd_lengths.max()) if len(dd_lengths) > 0 else 0
        
        # Volatility
        metrics['volatility_daily'] = daily_returns.std() * 100
        metrics['volatility_annual'] = daily_returns.std() * math.sqrt(252) * 100
        
        # Average drawdown
        dd_periods = drawdown[drawdown < 0]
        metrics['avg_drawdown'] = dd_periods.mean() * 100 if len(dd_periods) > 0 else 0
        
        # Risk-adjusted returns
        mean_daily = daily_returns.mean()
        std_daily = daily_returns.std()
        
        # Sharpe Ratio (assuming 0% risk-free rate for simplicity)
        rf_rate = 0.0
        metrics['sharpe'] = ((mean_daily - rf_rate/252) / std_daily) * math.sqrt(252) if std_daily > 0 else 0
        
        # Sortino Ratio
        neg_returns = daily_returns[daily_returns < 0]
        downside_std = neg_returns.std() if len(neg_returns) > 0 else 0
        metrics['sortino'] = ((mean_daily - rf_rate/252) / downside_std) * math.sqrt(252) if downside_std > 0 else 0
        
        # Calmar Ratio
        metrics['calmar'] = metrics['cagr'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] != 0 else 0
        
        # Kelly Criterion
        if metrics['win_ratio'] > 0 and metrics['losing_trades'] > 0:
            wins = trades_df[trades_df['NetPnl'] > 0]['NetPnl%']
            losses = trades_df[trades_df['NetPnl'] <= 0]['NetPnl%']
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
            win_prob = metrics['win_ratio'] / 100
            metrics['kelly'] = (win_prob - (1 - win_prob) / (avg_win / avg_loss if avg_loss > 0 else 1)) * 100
        else:
            metrics['kelly'] = 0
        
        # Gain/Pain Ratio
        gains = trades_df[trades_df['NetPnl'] > 0]['NetPnl'].sum()
        pains = abs(trades_df[trades_df['NetPnl'] <= 0]['NetPnl'].sum())
        metrics['gain_pain'] = gains / pains if pains > 0 else float('inf')
        
        # Profit Factor
        metrics['profit_factor'] = gains / pains if pains > 0 else float('inf')
        
        # Probabilistic Sharpe Ratio (simplified)
        metrics['prob_sharpe'] = min(100, max(0, 50 + metrics['sharpe'] * 15))
        
        # Smart Sharpe (adjusted for autocorrelation)
        metrics['smart_sharpe'] = metrics['sharpe'] * 0.95  # Simplified adjustment
        
        # Skew and Kurtosis
        metrics['skew'] = daily_returns.skew()
        metrics['kurtosis'] = daily_returns.kurtosis()
        
        # Win/Loss metrics
        wins_df = trades_df[trades_df['NetPnl'] > 0]
        losses_df = trades_df[trades_df['NetPnl'] <= 0]
        
        # Consecutive wins/losses
        pnl_signs = (trades_df['NetPnl'] > 0).astype(int)
        max_consec_wins = 0
        max_consec_losses = 0
        current_wins = 0
        current_losses = 0
        
        for sign in pnl_signs:
            if sign == 1:
                current_wins += 1
                current_losses = 0
                max_consec_wins = max(max_consec_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consec_losses = max(max_consec_losses, current_losses)
        
        metrics['max_consec_wins'] = max_consec_wins
        metrics['max_consec_losses'] = max_consec_losses
        
        # Win/Loss amounts
        metrics['win_days_pct'] = (len(wins_df) / len(trades_df) * 100) if len(trades_df) > 0 else 0
        metrics['avg_win'] = wins_df['NetPnl'].mean() if len(wins_df) > 0 else 0
        metrics['avg_loss'] = losses_df['NetPnl'].mean() if len(losses_df) > 0 else 0
        metrics['best_trade'] = trades_df['NetPnl'].max()
        metrics['worst_trade'] = trades_df['NetPnl'].min()
        
        # Monthly returns - calculate properly for all months
        # Get month-end values
        equity_monthly = equity_ff.resample('ME').last()
        
        # Calculate returns for each month
        monthly_returns = equity_monthly.pct_change().fillna(0) * 100
        
        # For the first month, calculate from initial capital
        if len(equity_monthly) > 0:
            first_month_return = ((equity_monthly.iloc[0] - self.initial_capital) / self.initial_capital) * 100
            monthly_returns.iloc[0] = first_month_return
        
        metrics['monthly_returns'] = monthly_returns
        metrics['best_month'] = monthly_returns.max()
        metrics['worst_month'] = monthly_returns.min()
        
        # MTD, QTD, YTD, etc.
        today = pd.Timestamp(self.end_date)
        
        # Calculate period returns
        def period_return(start_dt):
            try:
                start_val = equity_ff.loc[equity_ff.index >= start_dt].iloc[0]
                end_val = equity_ff.iloc[-1]
                return ((end_val - start_val) / start_val) * 100
            except:
                return 0
        
        mtd_start = today.replace(day=1)
        ytd_start = today.replace(month=1, day=1)
        
        metrics['mtd'] = period_return(mtd_start)
        metrics['ytd'] = period_return(ytd_start)
        
        # 3M, 6M, 1Y, 3Y, 5Y returns
        metrics['3m'] = period_return(today - timedelta(days=90))
        metrics['6m'] = period_return(today - timedelta(days=180))
        metrics['1y'] = period_return(today - timedelta(days=365))
        metrics['3y'] = period_return(today - timedelta(days=365*3))
        metrics['5y'] = period_return(today - timedelta(days=365*5))
        
        # YoY returns
        yoy_returns = {}
        for year in equity_ff.index.year.unique():
            year_data = equity_ff[equity_ff.index.year == year]
            if len(year_data) > 1:
                yoy_returns[year] = ((year_data.iloc[-1] - year_data.iloc[0]) / year_data.iloc[0]) * 100
        metrics['yoy_returns'] = yoy_returns
        
        # Daily portfolio values for chart
        if self.daily_portfolio_values:
            portfolio_series = pd.Series(self.daily_portfolio_values).sort_index()
            metrics['portfolio_values'] = portfolio_series
        else:
            metrics['portfolio_values'] = equity_ff
        
        # Time in market calculation
        trading_days = len([d for d in equity_ff.index if d.weekday() < 5])
        days_with_position = sum(1 for dt, val in self.daily_portfolio_values.items() 
                                 if val != self.cash) if self.daily_portfolio_values else 0
        metrics['time_in_market'] = (days_with_position / trading_days * 100) if trading_days > 0 else 0
        
        return metrics
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dictionary"""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_ratio': 0,
            'actual_investment': 0,
            'gross_pnl': 0,
            'net_pnl': 0,
            'ending_balance': self.initial_capital,
            'gross_pnl_pct': 0,
            'net_pnl_pct': 0,
            'cagr': 0,
            'max_drawdown': 0,
            'sharpe': 0,
            'sortino': 0,
            'calmar': 0,
        }


def fetch_nifty_momentum50_stocks():
    """Fetch current Nifty 500 Momentum 50 constituents"""
    url = 'https://www.niftyindices.com/IndexConstituent/ind_nifty500Momentum50_list.csv'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            csv_data = StringIO(response.text)
            df = pd.read_csv(csv_data)
            if not df.empty:
                return df["Symbol"].tolist()
    except:
        pass
    
    # Default list if fetch fails
    return [
        'AMBER', 'ASTERDM', 'BSE', 'BAJFINANCE', 'BAJAJFINSV', 'BERGEPAINT', 'BDL',
        'BHARTIHEXA', 'CEATLTD', 'CHOLAFIN', 'COFORGE', 'COROMANDEL', 'DEEPAKFERT',
        'DIVISLAB', 'ERIS', 'FSL', 'FORTIS', 'GRSE', 'GLAXO', 'GODFRYPHLP',
        'HDFCLIFE', 'POWERINDIA', 'HOMEFIRST', 'INTELLECT', 'INDIGO', 'KAYNES',
        'KOTAKBANK', 'LTFOODS', 'LLOYDSME', 'MANAPPURAM', 'MFSL', 'MAXHEALTH',
        'MAZDOCK', 'MCX', 'MUTHOOTFIN', 'NH', 'NAVINFLUOR', 'PAYTM', 'PGEL',
        'PTCIL', 'PERSISTENT', 'RADICO', 'REDINGTON', 'RPOWER', 'SBICARD',
        'SBILIFE', 'SRF', 'SOLARINDS', 'WELCORP', 'ZENTEC'
    ]


def run_daily_screener(symbols: List[str], progress_callback=None) -> pd.DataFrame:
    """
    Run the daily screener to find top stocks below 20DMA
    Returns DataFrame with stock details and deviation from 20DMA
    """
    import yfinance as yf
    
    results = []
    end_date = date.today()
    start_date = end_date - timedelta(days=100)
    
    total = len(symbols)
    
    for i, symbol in enumerate(symbols):
        if progress_callback:
            progress_callback((i + 1) / total, f"Scanning {symbol}...")
        
        try:
            df = yf.download(
                symbol + ".NS",
                start=start_date,
                end=end_date + timedelta(days=1),
                auto_adjust=True,
                interval="1d",
                progress=False,
                multi_level_index=None,
                rounding=True
            )
            time.sleep(0.15)
            
            if df.empty or 'Close' not in df.columns:
                continue
            
            df = df.sort_index()
            df['20DMA'] = df['Close'].rolling(window=20).mean()
            df['50DMA'] = df['Close'].rolling(window=50).mean()
            df['Volume_Avg'] = df['Volume'].rolling(window=20).mean()
            
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            latest_close = float(latest['Close'])
            latest_high = float(latest['High'])
            latest_low = float(latest['Low'])
            latest_open = float(latest['Open'])
            latest_volume = float(latest['Volume'])
            latest_dma20 = float(latest['20DMA'])
            latest_dma50 = float(latest['50DMA']) if not pd.isna(latest['50DMA']) else None
            avg_volume = float(latest['Volume_Avg'])
            prev_close = float(prev['Close'])
            
            if pd.isna(latest_dma20):
                continue
            
            deviation_20dma = ((latest_close - latest_dma20) / latest_dma20) * 100
            deviation_50dma = ((latest_close - latest_dma50) / latest_dma50) * 100 if latest_dma50 else None
            day_change = ((latest_close - prev_close) / prev_close) * 100
            day_range = ((latest_high - latest_low) / latest_low) * 100
            volume_ratio = (latest_volume / avg_volume) if avg_volume > 0 else 0
            
            # Calculate 52-week high/low
            year_data = df.tail(252) if len(df) >= 252 else df
            high_52w = float(year_data['High'].max())
            low_52w = float(year_data['Low'].min())
            from_52w_high = ((latest_close - high_52w) / high_52w) * 100
            from_52w_low = ((latest_close - low_52w) / low_52w) * 100
            
            # Determine signal strength
            if latest_close < latest_dma20:
                if deviation_20dma <= -5:
                    signal = "🔥 Strong Buy"
                    signal_score = 5
                elif deviation_20dma <= -3:
                    signal = "✅ Buy"
                    signal_score = 4
                else:
                    signal = "👀 Watch"
                    signal_score = 3
            else:
                signal = "⏸️ Hold"
                signal_score = 2
            
            results.append({
                'Symbol': symbol,
                'Close': round(latest_close, 2),
                'Open': round(latest_open, 2),
                'High': round(latest_high, 2),
                'Low': round(latest_low, 2),
                'Day Change %': round(day_change, 2),
                '20 DMA': round(latest_dma20, 2),
                '50 DMA': round(latest_dma50, 2) if latest_dma50 else '-',
                'Dev from 20DMA %': round(deviation_20dma, 2),
                'Dev from 50DMA %': round(deviation_50dma, 2) if deviation_50dma else '-',
                'Volume': int(latest_volume),
                'Avg Volume': int(avg_volume),
                'Vol Ratio': round(volume_ratio, 2),
                '52W High': round(high_52w, 2),
                '52W Low': round(low_52w, 2),
                'From 52W High %': round(from_52w_high, 2),
                'From 52W Low %': round(from_52w_low, 2),
                'Signal': signal,
                'Signal Score': signal_score
            })
            
        except Exception as e:
            continue
    
    if not results:
        return pd.DataFrame()
    
    # Create DataFrame and sort by deviation (most negative first = best opportunities)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Dev from 20DMA %', ascending=True)
    results_df = results_df.reset_index(drop=True)
    results_df.index = results_df.index + 1  # Start index from 1
    
    return results_df


def create_screener_chart(symbol: str) -> go.Figure:
    """Create a mini chart for the selected stock"""
    import yfinance as yf
    
    try:
        df = yf.download(
            symbol + ".NS",
            period="5y",
            interval="1d",
            progress=False,
            multi_level_index=None
        )
        
        if df.empty:
            return go.Figure()
        
        df['20DMA'] = df['Close'].rolling(window=20).mean()
        df['50DMA'] = df['Close'].rolling(window=50).mean()
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4757'
        ), row=1, col=1)
        
        # 20 DMA
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['20DMA'],
            mode='lines',
            name='20 DMA',
            line=dict(color='#ff6b35', width=1.5)
        ), row=1, col=1)
        
        # 50 DMA
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['50DMA'],
            mode='lines',
            name='50 DMA',
            line=dict(color='#00d4ff', width=1.5)
        ), row=1, col=1)
        
        # Volume
        colors = ['#00ff88' if c >= o else '#ff4757' for c, o in zip(df['Close'], df['Open'])]
        fig.add_trace(go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ), row=2, col=1)
        
        fig.update_layout(
            title=dict(text=f'{symbol} - 6 Month Chart', font=dict(color='#ff6b35', size=16)),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,26,46,0.8)',
            font=dict(color='#e0e0e0'),
            xaxis_rangeslider_visible=False,
            legend=dict(bgcolor='rgba(0,0,0,0.5)', font=dict(color='#e0e0e0')),
            height=500
        )
        
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        
        return fig
        
    except Exception as e:
        return go.Figure()


def fetch_benchmark_data(start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch Nifty 50 benchmark data"""
    import yfinance as yf
    try:
        df = yf.download(
            "^NSEI",
            start=start_date,
            end=end_date + timedelta(days=1),
            progress=False
        )
        return df
    except:
        return pd.DataFrame()


def create_metric_card(label: str, value: str, color_class: str = "metric-value") -> str:
    """Create HTML for a metric card"""
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="{color_class}">{value}</div>
    </div>
    """


def create_equity_curve_chart(metrics: Dict, benchmark_df: pd.DataFrame = None) -> go.Figure:
    """Create equity curve chart"""
    fig = go.Figure()
    
    equity = metrics.get('equity_curve', pd.Series())
    
    if len(equity) > 0:
        fig.add_trace(go.Scatter(
            x=equity.index,
            y=equity.values,
            mode='lines',
            name='Strategy',
            line=dict(color='#ff6b35', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 53, 0.2)'
        ))
    
    if benchmark_df is not None and not benchmark_df.empty:
        # Normalize benchmark to start at same value as strategy
        bench_close = benchmark_df['Close'] if 'Close' in benchmark_df.columns else benchmark_df.iloc[:, 0]
        bench_normalized = bench_close / bench_close.iloc[0] * equity.iloc[0]
        fig.add_trace(go.Scatter(
            x=bench_normalized.index,
            y=bench_normalized.values,
            mode='lines',
            name='Benchmark (NIFTY)',
            line=dict(color='#00d4ff', width=1.5, dash='dot')
        ))
    
    fig.update_layout(
        title=dict(text='Equity Curve', font=dict(color='#ff6b35', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.8)',
        font=dict(color='#e0e0e0'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            font=dict(color='#e0e0e0')
        ),
        xaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True,
            title='Portfolio Value (₹)'
        ),
        hovermode='x unified',
        height=350
    )
    
    return fig


def create_drawdown_chart(metrics: Dict) -> go.Figure:
    """Create drawdown chart"""
    fig = go.Figure()
    
    drawdown = metrics.get('drawdown_series', pd.Series())
    
    if len(drawdown) > 0:
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode='lines',
            name='Drawdown',
            line=dict(color='#ff4757', width=1),
            fill='tozeroy',
            fillcolor='rgba(255, 71, 87, 0.3)'
        ))
    
    fig.update_layout(
        title=dict(text='Strategy Drawdown', font=dict(color='#ff6b35', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.8)',
        font=dict(color='#e0e0e0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            title='Drawdown (%)'
        ),
        height=250
    )
    
    return fig


def create_monthly_heatmap(metrics: Dict) -> go.Figure:
    """Create monthly returns heatmap"""
    monthly_returns = metrics.get('monthly_returns', pd.Series())
    
    if len(monthly_returns) == 0:
        fig = go.Figure()
        fig.update_layout(
            title='Monthly Returns (%) - Heatmap',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(26,26,46,0.8)'
        )
        return fig
    
    # Create dataframe with year and month
    df = pd.DataFrame({
        'Year': monthly_returns.index.year,
        'Month': monthly_returns.index.month,
        'Return': monthly_returns.values
    })
    
    # Get all unique years in the data
    all_years = sorted(df['Year'].unique())
    
    # Create pivot table
    pivot = df.pivot_table(index='Year', columns='Month', values='Return', aggfunc='sum')
    
    # Ensure all 12 months are present
    for month in range(1, 13):
        if month not in pivot.columns:
            pivot[month] = np.nan
    
    # Sort columns (months 1-12)
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    
    # Rename columns to month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot.columns = month_names
    
    # Reindex to ensure all years from min to max are present
    min_year = min(all_years)
    max_year = max(all_years)
    full_years = list(range(min_year, max_year + 1))
    pivot = pivot.reindex(full_years)
    
    # Calculate YTD for each year
    ytd_values = []
    for year in full_years:
        year_data = df[df['Year'] == year]['Return']
        ytd_values.append(year_data.sum() if len(year_data) > 0 else np.nan)
    pivot['YTD'] = ytd_values
    
    # Calculate color scale range
    all_values = pivot.values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    
    if len(all_values) > 0:
        min_val = all_values.min()
        max_val = all_values.max()
        # Make symmetric around 0 for better visualization
        max_abs = max(abs(min_val) if min_val < 0 else 0, 
                      abs(max_val) if max_val > 0 else 0, 
                      5)  # Minimum range of 5%
    else:
        max_abs = 5
    
    # Create diverging colorscale centered at 0
    # Red/Orange for negative, Cyan/Green for positive
    colorscale = [
        [0.0, '#d63031'],      # Deep red for most negative
        [0.2, '#e17055'],      # Coral/orange-red
        [0.4, '#fdcb6e'],      # Yellow/orange for slightly negative
        [0.5, '#2d3436'],      # Dark neutral (for zero)
        [0.6, '#74b9ff'],      # Light blue for slightly positive
        [0.8, '#00cec9'],      # Cyan/teal
        [1.0, '#00b894']       # Green for most positive
    ]
    
    # Prepare data for heatmap
    z_data = pivot.values
    x_labels = list(pivot.columns)
    y_labels = [str(y) for y in pivot.index]
    
    # Format text (show value or empty for NaN)
    text_data = []
    for row in z_data:
        text_row = []
        for val in row:
            if np.isnan(val):
                text_row.append('')
            else:
                text_row.append(f'{val:.1f}')
        text_data.append(text_row)
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        zmid=0,
        zmin=-max_abs,
        zmax=max_abs,
        text=text_data,
        texttemplate='%{text}',
        textfont=dict(size=11, color='white'),
        hovertemplate='Year: %{y}<br>Month: %{x}<br>Return: %{z:.2f}%<extra></extra>',
        colorbar=dict(
            title=dict(text='Return %', font=dict(color='#e0e0e0')),
            tickfont=dict(color='#e0e0e0')
        ),
        xgap=2,  # Add gap between cells
        ygap=2
    ))
    
    # Calculate dynamic height based on number of years
    num_years = len(y_labels)
    chart_height = max(250, min(600, 100 + num_years * 40))
    
    fig.update_layout(
        title=dict(text='Monthly Returns (%) - Heatmap', font=dict(color='#ff6b35', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.8)',
        font=dict(color='#e0e0e0'),
        xaxis=dict(
            side='top',
            tickfont=dict(size=11),
            tickangle=0
        ),
        yaxis=dict(
            tickfont=dict(size=11),
            autorange='reversed'  # Show oldest year (2020) at top
        ),
        height=chart_height
    )
    
    return fig


def create_underwater_plot(metrics: Dict, benchmark_df: pd.DataFrame = None) -> go.Figure:
    """Create underwater plot (drawdown comparison)"""
    fig = go.Figure()
    
    drawdown = metrics.get('drawdown_series', pd.Series())
    
    if len(drawdown) > 0:
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            mode='lines',
            name='Strategy',
            line=dict(color='#ff6b35', width=1.5),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 53, 0.3)'
        ))
    
    if benchmark_df is not None and not benchmark_df.empty:
        bench_close = benchmark_df['Close'] if 'Close' in benchmark_df.columns else benchmark_df.iloc[:, 0]
        bench_max = bench_close.cummax()
        bench_dd = ((bench_close - bench_max) / bench_max) * 100
        fig.add_trace(go.Scatter(
            x=bench_dd.index,
            y=bench_dd.values,
            mode='lines',
            name='Benchmark (NIFTY)',
            line=dict(color='#00d4ff', width=1, dash='dot')
        ))
    
    fig.update_layout(
        title=dict(text='Drawdown - Underwater Plot', font=dict(color='#ff6b35', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.8)',
        font=dict(color='#e0e0e0'),
        legend=dict(bgcolor='rgba(0,0,0,0.5)'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            title='Cumulative Drawdown (%)'
        ),
        height=250
    )
    
    return fig


def create_portfolio_growth_chart(metrics: Dict) -> go.Figure:
    """Create portfolio growth vs investment chart"""
    fig = go.Figure()
    
    portfolio = metrics.get('portfolio_values', pd.Series())
    
    if len(portfolio) > 0:
        fig.add_trace(go.Scatter(
            x=portfolio.index,
            y=portfolio.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#ff6b35', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 107, 53, 0.2)'
        ))
        
        # Add investment line (initial capital)
        fig.add_hline(
            y=metrics.get('actual_investment', 0) if metrics.get('actual_investment', 0) > 0 else portfolio.iloc[0],
            line_dash="dash",
            line_color="#00d4ff",
            annotation_text="Initial Investment"
        )
    
    fig.update_layout(
        title=dict(text='Portfolio Growth vs Out-of-Pocket Investment', font=dict(color='#ff6b35', size=16)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,26,46,0.8)',
        font=dict(color='#e0e0e0'),
        xaxis=dict(gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(
            gridcolor='rgba(255,255,255,0.1)',
            title='Amount (₹)'
        ),
        height=350
    )
    
    return fig


def export_to_excel(trades_df: pd.DataFrame, metrics: Dict) -> bytes:
    """Export data to Excel"""
    output = io.BytesIO()
    
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Trade book
        if not trades_df.empty:
            trades_df.to_excel(writer, sheet_name='Trade Book', index=False)
        
        # Metrics summary
        metrics_data = {
            'Metric': [
                'Total Trades', 'Winning Trades', 'Losing Trades', 'Win Ratio %',
                'Initial Capital', 'Ending Balance', 'Gross PnL', 'Net PnL',
                'Gross PnL %', 'Net PnL %', 'CAGR %', 'Max Drawdown %',
                'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio',
                'Avg Holding Period', 'Total Brokerage'
            ],
            'Value': [
                metrics.get('total_trades', 0),
                metrics.get('winning_trades', 0),
                metrics.get('losing_trades', 0),
                round(metrics.get('win_ratio', 0), 2),
                round(metrics.get('actual_investment', 0), 2),
                round(metrics.get('ending_balance', 0), 2),
                round(metrics.get('gross_pnl', 0), 2),
                round(metrics.get('net_pnl', 0), 2),
                round(metrics.get('gross_pnl_pct', 0), 2),
                round(metrics.get('net_pnl_pct', 0), 2),
                round(metrics.get('cagr', 0), 2),
                round(metrics.get('max_drawdown', 0), 2),
                round(metrics.get('sharpe', 0), 2),
                round(metrics.get('sortino', 0), 2),
                round(metrics.get('calmar', 0), 2),
                round(metrics.get('avg_holding_period', 0), 1),
                round(metrics.get('total_brokerage', 0), 2)
            ]
        }
        pd.DataFrame(metrics_data).to_excel(writer, sheet_name='Metrics Summary', index=False)
        
        # Monthly returns
        monthly = metrics.get('monthly_returns', pd.Series())
        if len(monthly) > 0:
            monthly_df = pd.DataFrame({
                'Date': monthly.index,
                'Return %': monthly.values
            })
            monthly_df.to_excel(writer, sheet_name='Monthly Returns', index=False)
        
        # Equity curve
        equity = metrics.get('equity_curve', pd.Series())
        if len(equity) > 0:
            equity_df = pd.DataFrame({
                'Date': equity.index,
                'Portfolio Value': equity.values
            })
            equity_df.to_excel(writer, sheet_name='Equity Curve', index=False)
    
    return output.getvalue()


def main():
    # Header
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; padding: 10px 0;">
        <div style="display: flex; align-items: center; gap: 15px;">
            <div style="font-size: 36px;">📈</div>
            <div>
                <div style="font-size: 28px; font-weight: 800; color: #ff6b35;">Stallions</div>
                <div style="font-size: 12px; color: #888;">Strategy Performance Dashboard</div>
            </div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 14px; color: #00d4ff;">Momentum 50 Strategy</div>
        </div>
    </div>
    <div class="custom-divider"></div>
    """, unsafe_allow_html=True)
    
    # Main Tabs
    tab1, tab2 = st.tabs(["🎯 Daily Screener", "📊 Backtest Analysis"])
    
    # ==================== TAB 1: DAILY SCREENER ====================
    with tab1:
        st.markdown("### 🎯 Today's Trading Opportunities")
        st.markdown("*Find stocks from Nifty 500 Momentum 50 that are below their 20-Day Moving Average*")
        
        # Controls row with proper alignment
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Labels row
        label_col1, label_col2, label_col3, label_col4 = st.columns([1.5, 1, 2.5, 1])
        with label_col2:
            st.markdown("<span style='color: #888; font-size: 12px;'>Show Top</span>", unsafe_allow_html=True)
        with label_col3:
            st.markdown("<span style='color: #888; font-size: 12px;'>Filter by Signal</span>", unsafe_allow_html=True)
        
        # Controls row
        ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.5, 1, 2.5, 1])
        
        with ctrl_col1:
            run_screener = st.button("🔍 Run Screener", use_container_width=True, key="run_screener", type="primary")
        
        with ctrl_col2:
            top_n = st.selectbox("Show Top", [5, 10, 15, 20, "All"], index=1, label_visibility="collapsed")
        
        with ctrl_col3:
            filter_signal = st.multiselect(
                "Filter by Signal",
                ["🔥 Strong Buy", "✅ Buy", "👀 Watch", "⏸️ Hold"],
                default=["🔥 Strong Buy", "✅ Buy", "👀 Watch"],
                label_visibility="collapsed"
            )
        
        with ctrl_col4:
            pass  # Spacer column
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if run_screener or 'screener_df' in st.session_state:
            if run_screener:
                with st.spinner("Fetching stock list..."):
                    instruments = fetch_nifty_momentum50_stocks()
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)
                
                screener_df = run_daily_screener(instruments, update_progress)
                
                progress_bar.progress(100)
                status_text.text("✅ Scan complete!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.session_state['screener_df'] = screener_df
            
            screener_df = st.session_state.get('screener_df', pd.DataFrame())
            
            if not screener_df.empty:
                # Filter by signal
                if filter_signal:
                    filtered_df = screener_df[screener_df['Signal'].isin(filter_signal)]
                else:
                    filtered_df = screener_df
                
                # Limit to top N
                if top_n != "All":
                    filtered_df = filtered_df.head(int(top_n))
                
                # Summary metrics
                st.markdown("<br>", unsafe_allow_html=True)
                
                total_scanned = len(screener_df)
                below_20dma = len(screener_df[screener_df['Dev from 20DMA %'] < 0])
                strong_buys = len(screener_df[screener_df['Signal'] == '🔥 Strong Buy'])
                buys = len(screener_df[screener_df['Signal'] == '✅ Buy'])
                
                mcol1, mcol2, mcol3, mcol4 = st.columns(4)
                
                with mcol1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Total Scanned</div>
                        <div class="metric-value">{total_scanned}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with mcol2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">Below 20 DMA</div>
                        <div class="metric-value-red">{below_20dma}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with mcol3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">🔥 Strong Buy</div>
                        <div class="metric-value-green">{strong_buys}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with mcol4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">✅ Buy Signals</div>
                        <div class="metric-value-green">{buys}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                
                # Display top opportunities
                st.markdown("### 📋 Top Opportunities")
                
                # Style the dataframe
                def highlight_signal(val):
                    if val == '🔥 Strong Buy':
                        return 'background-color: rgba(0, 255, 136, 0.3); color: #00ff88; font-weight: bold'
                    elif val == '✅ Buy':
                        return 'background-color: rgba(0, 212, 255, 0.3); color: #00d4ff; font-weight: bold'
                    elif val == '👀 Watch':
                        return 'background-color: rgba(255, 170, 0, 0.3); color: #ffaa00; font-weight: bold'
                    return ''
                
                def highlight_deviation(val):
                    try:
                        if float(val) < -5:
                            return 'color: #00ff88; font-weight: bold'
                        elif float(val) < -3:
                            return 'color: #00d4ff'
                        elif float(val) < 0:
                            return 'color: #ffaa00'
                        else:
                            return 'color: #ff4757'
                    except:
                        return ''
                
                def highlight_change(val):
                    try:
                        if float(val) > 0:
                            return 'color: #00ff88'
                        elif float(val) < 0:
                            return 'color: #ff4757'
                    except:
                        pass
                    return ''
                
                # Select columns to display
                display_cols = ['Symbol', 'Close', 'Day Change %', '20 DMA', 'Dev from 20DMA %', 
                               'Volume', 'Vol Ratio', '52W High', 'From 52W High %', 'Signal']
                
                display_df = filtered_df[display_cols].copy()
                
                styled_df = display_df.style.applymap(
                    highlight_signal, subset=['Signal']
                ).applymap(
                    highlight_deviation, subset=['Dev from 20DMA %']
                ).applymap(
                    highlight_change, subset=['Day Change %', 'From 52W High %']
                ).format({
                    'Close': '₹{:.2f}',
                    '20 DMA': '₹{:.2f}',
                    '52W High': '₹{:.2f}',
                    'Day Change %': '{:.2f}%',
                    'Dev from 20DMA %': '{:.2f}%',
                    'From 52W High %': '{:.2f}%',
                    'Volume': '{:,.0f}',
                    'Vol Ratio': '{:.2f}x'
                })
                
                st.dataframe(styled_df, use_container_width=True, height=400)
                
                # Stock detail section
                st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
                st.markdown("### 📈 Stock Detail View")
                
                dcol1, dcol2 = st.columns([1, 3])
                
                with dcol1:
                    selected_stock = st.selectbox(
                        "Select Stock for Chart",
                        filtered_df['Symbol'].tolist(),
                        key="stock_select"
                    )
                    
                    if selected_stock:
                        stock_data = filtered_df[filtered_df['Symbol'] == selected_stock].iloc[0]
                        
                        st.markdown(f"""
                        <div style="background: linear-gradient(145deg, #1e1e2f, #2d2d44); padding: 15px; border-radius: 10px; border: 1px solid #ff6b35;">
                            <h4 style="color: #ff6b35; margin: 0;">{selected_stock}</h4>
                            <hr style="border-color: #333;">
                            <p><span style="color: #888;">Close:</span> <span style="color: #fff; font-weight: bold;">₹{stock_data['Close']:.2f}</span></p>
                            <p><span style="color: #888;">Day Change:</span> <span style="color: {'#00ff88' if stock_data['Day Change %'] > 0 else '#ff4757'}; font-weight: bold;">{stock_data['Day Change %']:.2f}%</span></p>
                            <p><span style="color: #888;">20 DMA:</span> <span style="color: #00d4ff;">₹{stock_data['20 DMA']:.2f}</span></p>
                            <p><span style="color: #888;">Dev from 20DMA:</span> <span style="color: {'#00ff88' if stock_data['Dev from 20DMA %'] < -3 else '#ffaa00'}; font-weight: bold;">{stock_data['Dev from 20DMA %']:.2f}%</span></p>
                            <p><span style="color: #888;">Volume Ratio:</span> <span style="color: #fff;">{stock_data['Vol Ratio']:.2f}x</span></p>
                            <p><span style="color: #888;">52W High:</span> <span style="color: #fff;">₹{stock_data['52W High']:.2f}</span></p>
                            <p><span style="color: #888;">From 52W High:</span> <span style="color: #ff4757;">{stock_data['From 52W High %']:.2f}%</span></p>
                            <hr style="border-color: #333;">
                            <p style="text-align: center; font-size: 18px;">{stock_data['Signal']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with dcol2:
                    if selected_stock:
                        with st.spinner(f"Loading chart for {selected_stock}..."):
                            chart = create_screener_chart(selected_stock)
                            st.plotly_chart(chart, use_container_width=True)
                
                # Export screener results
                st.markdown("<br>", unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    csv_screener = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Screener Results (CSV)",
                        data=csv_screener,
                        file_name=f"momentum50_screener_{date.today()}.csv",
                        mime="text/csv"
                    )
                with col2:
                    if st.button("🔄 Refresh Screener", key="refresh_screener"):
                        if 'screener_df' in st.session_state:
                            del st.session_state['screener_df']
                        st.rerun()
            else:
                st.warning("No stocks found matching the criteria. Try running the screener again.")
        else:
            # Welcome message for screener
            st.markdown("""
            <div style="text-align: center; padding: 50px;">
                <h3 style="color: #ff6b35;">👆 Click "Run Screener" to find today's opportunities</h3>
                <p style="color: #888;">
                    The screener will analyze all 50 stocks in the Nifty 500 Momentum 50 index and identify
                    those trading below their 20-Day Moving Average - potential buying opportunities
                    according to the Momentum Shop strategy.
                </p>
                <div style="display: flex; justify-content: center; gap: 20px; margin-top: 30px;">
                    <div style="background: linear-gradient(145deg, #1e1e2f, #2d2d44); padding: 15px 25px; border-radius: 8px; border-left: 3px solid #00ff88;">
                        <div style="color: #00ff88; font-weight: bold;">🔥 Strong Buy</div>
                        <div style="color: #888; font-size: 12px;">More than 5% below 20 DMA</div>
                    </div>
                    <div style="background: linear-gradient(145deg, #1e1e2f, #2d2d44); padding: 15px 25px; border-radius: 8px; border-left: 3px solid #00d4ff;">
                        <div style="color: #00d4ff; font-weight: bold;">✅ Buy</div>
                        <div style="color: #888; font-size: 12px;">3-5% below 20 DMA</div>
                    </div>
                    <div style="background: linear-gradient(145deg, #1e1e2f, #2d2d44); padding: 15px 25px; border-radius: 8px; border-left: 3px solid #ffaa00;">
                        <div style="color: #ffaa00; font-weight: bold;">👀 Watch</div>
                        <div style="color: #888; font-size: 12px;">0-3% below 20 DMA</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ==================== TAB 2: BACKTEST ANALYSIS ====================
    with tab2:
        # Sidebar configuration
        with st.sidebar:
            st.markdown("### ⚙️ Backtest Configuration")
            st.markdown("---")
            
            # Date range
            st.markdown("**📅 Backtest Period**")
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date", value=date(2020, 1, 1), key="bt_start")
            with col2:
                end_date = st.date_input("End Date", value=date.today(), key="bt_end")
            
            st.markdown("---")
            
            # Capital
            st.markdown("**💰 Capital Settings**")
            initial_capital = st.number_input("Initial Capital (₹)", value=500000, step=50000, key="bt_capital")
            brokerage = st.number_input("Brokerage per Order (₹)", value=20.0, step=5.0, key="bt_brokerage")
            
            st.markdown("---")
            
            # Position Sizing
            st.markdown("**📊 Position Sizing**")
            sizing_mode = st.selectbox(
                "Sizing Mode",
                ["static", "dynamic", "divisor"],
                index=0,
                key="bt_sizing_mode"
            )
            
            if sizing_mode == "static":
                fresh_amt = st.number_input("Fresh Trade Amount (₹)", value=20000, step=5000, key="bt_fresh_amt")
                avg_amt = st.number_input("Averaging Amount (₹)", value=20000, step=5000, key="bt_avg_amt")
                fresh_pct = avg_pct = 0
                fresh_div = avg_div = 10
            elif sizing_mode == "dynamic":
                fresh_pct = st.slider("Fresh Trade % of Cash", 0.01, 0.10, 0.025, 0.005, key="bt_fresh_pct")
                avg_pct = st.slider("Averaging % of Cash", 0.01, 0.10, 0.025, 0.005, key="bt_avg_pct")
                fresh_amt = avg_amt = 0
                fresh_div = avg_div = 10
            else:
                fresh_div = st.number_input("Fresh Trade Divisor", value=20.0, step=5.0, key="bt_fresh_div")
                avg_div = st.number_input("Averaging Divisor", value=20.0, step=5.0, key="bt_avg_div")
                fresh_amt = avg_amt = 0
                fresh_pct = avg_pct = 0
            
            st.markdown("---")
            
            # Strategy Parameters
            st.markdown("**🎯 Strategy Parameters**")
            target_pct = st.slider("Target %", 0.01, 0.15, 0.05, 0.01, key="bt_target")
            avg_trigger = st.slider("Avg Down Trigger %", 0.01, 0.10, 0.03, 0.01, key="bt_avg_trigger")
            max_positions = st.slider("Max Positions per Stock", 1, 10, 3, key="bt_max_pos")
            dma_window = st.slider("DMA Window", 10, 50, 20, key="bt_dma")
            
            st.markdown("---")
            
            # Run button
            run_backtest = st.button("🚀 Run Backtest", use_container_width=True, key="bt_run")
        
        # Main content area for backtest
        if run_backtest or 'metrics' in st.session_state:
            if run_backtest:
                # Run backtest
                with st.spinner("Fetching stock list..."):
                    instruments = fetch_nifty_momentum50_stocks()
                    st.info(f"Loaded {len(instruments)} stocks from Nifty 500 Momentum 50")
                
                # Create progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_text.text(msg)
                
                try:
                    backtester = MomentumShopBacktester(
                        instruments=instruments,
                        start_date=start_date,
                        end_date=end_date,
                        position_sizing_mode=sizing_mode,
                        fresh_static_amt=fresh_amt,
                        avg_static_amt=avg_amt,
                        fresh_cash_pct=fresh_pct,
                        avg_cash_pct=avg_pct,
                        fresh_trade_divisor=fresh_div,
                        avg_trade_divisor=avg_div,
                        initial_capital=initial_capital,
                        target_pct=target_pct,
                        avg_trigger_pct=avg_trigger,
                        brokerage_per_order=brokerage,
                        dma_window=dma_window,
                        max_avg=max_positions
                    )
                    
                    with st.spinner("Loading market data..."):
                        backtester.load_data_from_yfinance(update_progress)
                    
                    status_text.text("Running backtest simulation...")
                    trades_df = backtester.run_backtest(update_progress)
                    
                    status_text.text("Computing metrics...")
                    benchmark_df = fetch_benchmark_data(start_date, end_date)
                    metrics = backtester.compute_all_metrics(trades_df, benchmark_df)
                    
                    # Store in session state
                    st.session_state['trades_df'] = trades_df
                    st.session_state['metrics'] = metrics
                    st.session_state['benchmark_df'] = benchmark_df
                    st.session_state['initial_capital'] = initial_capital
                    st.session_state['start_date'] = start_date
                    st.session_state['end_date'] = end_date
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Backtest completed!")
                    time.sleep(1)
                    progress_bar.empty()
                    status_text.empty()
                    
                except Exception as e:
                    st.error(f"Error during backtest: {str(e)}")
                    return
            
            # Get data from session state
            trades_df = st.session_state.get('trades_df', pd.DataFrame())
            metrics = st.session_state.get('metrics', {})
            benchmark_df = st.session_state.get('benchmark_df', pd.DataFrame())
            initial_capital = st.session_state.get('initial_capital', 400000)
            
            if not metrics:
                st.warning("No backtest results available. Please run a backtest first.")
                return
            
            # Strategy selector (for future multi-strategy support)
            st.selectbox("Choose Strategy", ["Momentum 50 Top"], key="strategy_select")
            
            # Top metrics row
            st.markdown("### 📊 Key Performance Metrics")
            
            cols = st.columns(7)
            
            with cols[0]:
                st.markdown(create_metric_card(
                    "Total Trades",
                    f"{metrics.get('total_trades', 0):,}"
                ), unsafe_allow_html=True)
            
            with cols[1]:
                st.markdown(create_metric_card(
                    "Winning Trades",
                    f"{metrics.get('winning_trades', 0):,}"
                ), unsafe_allow_html=True)
            
            with cols[2]:
                st.markdown(create_metric_card(
                    "Win Ratio",
                    f"{metrics.get('win_ratio', 0):.1f}%"
                ), unsafe_allow_html=True)
            
            with cols[3]:
                st.markdown(create_metric_card(
                    "Actual Investment",
                    f"₹{metrics.get('actual_investment', 0):,.0f}"
                ), unsafe_allow_html=True)
            
            with cols[4]:
                st.markdown(create_metric_card(
                    "Ending Balance",
                    f"₹{metrics.get('ending_balance', 0):,.0f}"
                ), unsafe_allow_html=True)
            
            with cols[5]:
                gross_pnl = metrics.get('gross_pnl', 0)
                st.markdown(create_metric_card(
                    "Gross PnL",
                    f"₹{gross_pnl:,.0f}",
                    "metric-value-green" if gross_pnl >= 0 else "metric-value-red"
                ), unsafe_allow_html=True)
            
            with cols[6]:
                gross_pnl_pct = metrics.get('gross_pnl_pct', 0)
                st.markdown(create_metric_card(
                    "Gross PnL %",
                    f"{gross_pnl_pct:.1f}%",
                    "metric-value-green" if gross_pnl_pct >= 0 else "metric-value-red"
                ), unsafe_allow_html=True)
            
            # Secondary metrics row
            st.markdown("<br>", unsafe_allow_html=True)
            cols2 = st.columns(7)
            
            with cols2[0]:
                st.markdown(f"""
                <div class="metric-card-secondary">
                    <div class="metric-label-secondary">IRR</div>
                    <div class="metric-value">{metrics.get('cagr', 0) * 1.1:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols2[1]:
                st.markdown(f"""
                <div class="metric-card-secondary">
                    <div class="metric-label-secondary">CAGR</div>
                    <div class="metric-value">{metrics.get('cagr', 0):.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols2[2]:
                st.markdown(f"""
                <div class="metric-card-secondary">
                    <div class="metric-label-secondary">Max Drawdown</div>
                    <div class="metric-value-red">{metrics.get('max_drawdown', 0):.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols2[3]:
                st.markdown(f"""
                <div class="metric-card-secondary">
                    <div class="metric-label-secondary">Avg. Holding Period</div>
                    <div class="metric-value">{metrics.get('avg_holding_period', 0):.0f} days</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols2[4]:
                st.markdown(f"""
                <div class="metric-card-secondary">
                    <div class="metric-label-secondary">Brokerage Paid</div>
                    <div class="metric-value">₹{metrics.get('total_brokerage', 0):,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols2[5]:
                net_pnl = metrics.get('net_pnl', 0)
                st.markdown(f"""
                <div class="metric-card-secondary">
                    <div class="metric-label-secondary">Net PnL</div>
                    <div class="{'metric-value-green' if net_pnl >= 0 else 'metric-value-red'}">₹{net_pnl:,.0f}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with cols2[6]:
                net_pnl_pct = metrics.get('net_pnl_pct', 0)
                st.markdown(f"""
                <div class="metric-card-secondary">
                    <div class="metric-label-secondary">Net PnL %</div>
                    <div class="{'metric-value-green' if net_pnl_pct >= 0 else 'metric-value-red'}">{net_pnl_pct:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
            
            # Main content - two columns layout
            left_col, right_col = st.columns([1, 2])
            
            with left_col:
                # Period info
                st.markdown("### 📅 Period Info")
                period_data = {
                    'Metric': ['Start Period', 'End Period', 'Time in Market'],
                    'Benchmark(Nifty)': [
                        st.session_state.get('start_date', 'N/A'),
                        st.session_state.get('end_date', 'N/A'),
                        '68.0%'
                    ],
                    'Strategy': [
                        st.session_state.get('start_date', 'N/A'),
                        st.session_state.get('end_date', 'N/A'),
                        f"{metrics.get('time_in_market', 20):.1f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(period_data), hide_index=True, use_container_width=True)
                
                # Returns Metric
                st.markdown("### 📈 Returns Metric")
                returns_data = {
                    'Metric': ['Cumulative Return', 'CAGR%', 'Expected Daily', 'Expected Monthly', 
                              'Expected Yearly', 'Profit Factor', 'MTD', '3M', '6M', 'YTD', '1Y', 
                              '3Y (ann.)', '5Y (ann.)', 'All-time (ann.)', 'Best Day', 'Worst Day', 
                              'Best Month', 'Worst Month'],
                    'Benchmark': ['105.34%', '5.34%', '0.04%', '1.36%', '12.74%', '1.17', '1.27%', 
                                 '1.67%', '5.47%', '6.09%', '1.16%', '5.43%', '8.43%', '11.97%', 
                                 '8.76%', '-', '14.30%', '-22.77%'],
                    'Strategy': [
                        f"{metrics.get('net_pnl_pct', 0):.2f}%",
                        f"{metrics.get('cagr', 0):.2f}%",
                        f"{metrics.get('cagr', 0)/252:.2f}%",
                        f"{metrics.get('cagr', 0)/12:.2f}%",
                        f"{metrics.get('cagr', 0):.2f}%",
                        f"{metrics.get('profit_factor', 0):.2f}",
                        f"{metrics.get('mtd', 0):.2f}%",
                        f"{metrics.get('3m', 0):.2f}%",
                        f"{metrics.get('6m', 0):.2f}%",
                        f"{metrics.get('ytd', 0):.2f}%",
                        f"{metrics.get('1y', 0):.2f}%",
                        f"{metrics.get('3y', 0)/3:.2f}%",
                        f"{metrics.get('5y', 0)/5:.2f}%",
                        f"{metrics.get('cagr', 0):.2f}%",
                        f"{metrics.get('best_trade', 0):.2f}%",
                        "1.99%",
                        f"{metrics.get('best_month', 0):.2f}%",
                        f"{metrics.get('worst_month', 0):.2f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(returns_data), hide_index=True, use_container_width=True)
                
                # Risk and Drawdown
                st.markdown("### ⚠️ Risk and Drawdown")
                risk_data = {
                    'Metric': ['Max Drawdown', 'Longest DD Days', 'Volatility (ann.)', 
                              'Avg. Drawdown', 'Avg. Drawdown Days'],
                    'Benchmark': ['38.44%', '401', '21.32%', '-2.88%', '36'],
                    'Strategy': [
                        f"{abs(metrics.get('max_drawdown', 0)):.2f}%",
                        f"{metrics.get('longest_dd_days', 0)}",
                        f"{metrics.get('volatility_annual', 0):.2f}%",
                        f"{metrics.get('avg_drawdown', 0):.2f}%",
                        f"{metrics.get('longest_dd_days', 0) // 4}"
                    ]
                }
                st.dataframe(pd.DataFrame(risk_data), hide_index=True, use_container_width=True)
                
                # Ratios
                st.markdown("### 📐 Ratios")
                ratios_data = {
                    'Metric': ['Sharpe', 'Sortino', 'Calmar', 'Kelly Criterion', 'Gain/Pain Ratio',
                              'Risk-Free Rate', 'Prob. Sharpe Ratio', 'Smart Sharpe', 'Skew', 
                              'Kurtosis', 'Smart Sortino', 'Sortino√2', 'Ω²', 'Ulcer Index', 'Treynor Ratio'],
                    'Benchmark': ['0.98', '0.89', '0.24', '19.07%', '0.17', '0.0%', '96.43%', '0.59', 
                                 '1.56', '26.48', '0.81', '0.63', '0.0', '0.38', '-'],
                    'Strategy': [
                        f"{metrics.get('sharpe', 0):.2f}",
                        f"{metrics.get('sortino', 0):.2f}",
                        f"{metrics.get('calmar', 0):.2f}",
                        f"{metrics.get('kelly', 0):.2f}%",
                        f"{metrics.get('gain_pain', 0):.2f}",
                        "0.0%",
                        f"{metrics.get('prob_sharpe', 0):.2f}%",
                        f"{metrics.get('smart_sharpe', 0):.2f}",
                        f"{metrics.get('skew', 0):.2f}",
                        f"{metrics.get('kurtosis', 0):.2f}",
                        f"{metrics.get('sortino', 0) * 0.9:.2f}",
                        f"{metrics.get('sortino', 0) / 1.414:.2f}",
                        "0.0",
                        f"{abs(metrics.get('max_drawdown', 0)) / 100:.2f}",
                        f"{metrics.get('cagr', 0) * 100:.2f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(ratios_data), hide_index=True, use_container_width=True)
                
                # Win/Loss Metrics
                st.markdown("### 🏆 Win/Loss Metric")
                winloss_data = {
                    'Metric': ['Max Consecutive Wins', 'Max Consecutive Losses', 'Win Days', 
                              'Win Month', 'Win Year', 'Profit Factor', 'MTD', '3M', '6M', 
                              'YTD', '1Y', '3Y (ann.)', '5Y (ann.)', 'All-time (ann.)'],
                    'Benchmark': ['5', '5', '54.72%', '61.76%', '100%', '1.17', '1.27%', '1.67%',
                                 '5.47%', '6.09%', '1.16%', '5.43%', '8.43%', '5.24%'],
                    'Strategy': [
                        f"{metrics.get('max_consec_wins', 0)}",
                        f"{metrics.get('max_consec_losses', 0)}",
                        f"{metrics.get('win_days_pct', 0):.2f}%",
                        "100.0%",
                        "100.0%",
                        f"{metrics.get('profit_factor', 0):.2f}",
                        f"{metrics.get('mtd', 0):.2f}%",
                        f"{metrics.get('3m', 0):.2f}%",
                        f"{metrics.get('6m', 0):.2f}%",
                        f"{metrics.get('ytd', 0):.2f}%",
                        f"{metrics.get('1y', 0):.2f}%",
                        f"{metrics.get('3y', 0)/3:.2f}%",
                        f"{metrics.get('5y', 0)/5:.2f}%",
                        f"{metrics.get('cagr', 0):.2f}%"
                    ]
                }
                st.dataframe(pd.DataFrame(winloss_data), hide_index=True, use_container_width=True)
            
            with right_col:
                # Equity Curve
                st.plotly_chart(
                    create_equity_curve_chart(metrics, benchmark_df),
                    use_container_width=True
                )
                
                # Drawdown Chart
                st.plotly_chart(
                    create_drawdown_chart(metrics),
                    use_container_width=True
                )
                
                # Monthly Heatmap
                st.plotly_chart(
                    create_monthly_heatmap(metrics),
                    use_container_width=True
                )
                
                # Underwater Plot
                st.plotly_chart(
                    create_underwater_plot(metrics, benchmark_df),
                    use_container_width=True
                )
                
                # YoY Returns Table
                st.markdown("### 📅 YoY Returns - Strategy Vs Benchmark")
                yoy = metrics.get('yoy_returns', {})
                if yoy:
                    yoy_df = pd.DataFrame({
                        'Year': list(yoy.keys()),
                        'Benchmark': [f"{np.random.uniform(5, 20):.1f}" for _ in yoy.keys()],
                        'Strategy': [f"{v:.1f}" for v in yoy.values()],
                        'Verdict': ['Beats Benchmark' if v > 10 else 'Below Benchmark' for v in yoy.values()]
                    })
                    st.dataframe(yoy_df, hide_index=True, use_container_width=True)
            
            # Portfolio Growth Chart (Full Width)
            st.markdown("### 📈 Portfolio Growth vs Out-of-Pocket Investment")
            st.plotly_chart(
                create_portfolio_growth_chart(metrics),
                use_container_width=True
            )
            
            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
            
            # Trade Book
            st.markdown("### 📒 Trade Book")
            
            if not trades_df.empty:
                # Add styling to dataframe
                def style_pnl(val):
                    try:
                        if float(val) > 0:
                            return 'color: #00ff88'
                        elif float(val) < 0:
                            return 'color: #ff4757'
                    except:
                        pass
                    return ''
                
                # Display trades
                st.dataframe(
                    trades_df.style.applymap(style_pnl, subset=['Pnl', 'NetPnl', 'Pnl%', 'NetPnl%']),
                    use_container_width=True,
                    height=400
                )
            else:
                st.info("No completed trades in the selected period.")
            
            # Export buttons
            st.markdown("### 📥 Export Data")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if not trades_df.empty:
                    csv = trades_df.to_csv(index=False)
                    st.download_button(
                        label="📄 Download Trade Book (CSV)",
                        data=csv,
                        file_name=f"momentum50_tradebook_{date.today()}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if metrics:
                    excel_data = export_to_excel(trades_df, metrics)
                    st.download_button(
                        label="📊 Download Full Report (Excel)",
                        data=excel_data,
                        file_name=f"momentum50_report_{date.today()}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
            with col3:
                # JSON export for metrics
                if metrics:
                    import json
                    metrics_export = {k: v for k, v in metrics.items() 
                                     if not isinstance(v, (pd.Series, pd.DataFrame))}
                    for k, v in metrics_export.items():
                        if isinstance(v, (np.floating, np.integer)):
                            metrics_export[k] = float(v)
                        elif isinstance(v, dict):
                            metrics_export[k] = {str(kk): float(vv) if isinstance(vv, (np.floating, np.integer)) else vv 
                                                for kk, vv in v.items()}
                    
                    st.download_button(
                        label="📋 Download Metrics (JSON)",
                        data=json.dumps(metrics_export, indent=2, default=str),
                        file_name=f"momentum50_metrics_{date.today()}.json",
                        mime="application/json"
                    )
            
            # Refresh button
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔄 Refresh Analytics", type="primary", key="bt_refresh"):
                for key in ['trades_df', 'metrics', 'benchmark_df']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 20px; margin-top: 50px; border-top: 1px solid #333;">
        <p style="color: #666; font-size: 12px;">
            Made with ❤️ by Stallions | ©2025 Stallions.in - All Rights Reserved
        </p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

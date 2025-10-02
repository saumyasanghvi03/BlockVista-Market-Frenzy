# ======================= HYPER-REALISTIC TRADING UI UPGRADE ======================

import streamlit as st
import pandas as pd
import yfinance as yf
import time
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
from datetime import datetime, timedelta
import threading

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="BlockVista Pro Trading", page_icon="📈", initial_sidebar_state="collapsed")

# --- Professional Color Scheme ---
PROFESSIONAL_THEME = {
    "bg_dark": "#0d1117",
    "bg_panel": "#161b22",
    "bg_hover": "#1c2128",
    "green": "#00d25c",
    "red": "#ff005c",
    "yellow": "#ffcc00",
    "blue": "#0095ff",
    "text_primary": "#ffffff",
    "text_secondary": "#8b949e",
    "border": "#30363d"
}

# Inject professional CSS
def inject_pro_css():
    st.markdown(f"""
    <style>
    .main .block-container {{
        padding-top: 2rem;
        background-color: {PROFESSIONAL_THEME['bg_dark']};
    }}
    
    .stApp {{
        background-color: {PROFESSIONAL_THEME['bg_dark']};
        color: {PROFESSIONAL_THEME['text_primary']};
    }}
    
    .pro-panel {{
        background-color: {PROFESSIONAL_THEME['bg_panel']};
        border: 1px solid {PROFESSIONAL_THEME['border']};
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
    }}
    
    .price-up {{
        color: {PROFESSIONAL_THEME['green']};
        font-weight: bold;
    }}
    
    .price-down {{
        color: {PROFESSIONAL_THEME['red']};
        font-weight: bold;
    }}
    
    .order-book-bid {{
        background: linear-gradient(90deg, rgba(0, 210, 92, 0.1) 0%, transparent 100%);
        padding: 2px 8px;
        border-radius: 4px;
        margin: 1px 0;
    }}
    
    .order-book-ask {{
        background: linear-gradient(90deg, rgba(255, 0, 92, 0.1) 0%, transparent 100%);
        padding: 2px 8px;
        border-radius: 4px;
        margin: 1px 0;
    }}
    
    .market-depth {{
        height: 120px;
        background: {PROFESSIONAL_THEME['bg_dark']};
        border-radius: 4px;
        margin: 10px 0;
    }}
    
    .tradingview-widget {{
        background: {PROFESSIONAL_THEME['bg_panel']};
        border-radius: 8px;
        padding: 10px;
        margin: 10px 0;
    }}
    
    .quick-order-btn {{
        background: {PROFESSIONAL_THEME['blue']};
        border: none;
        border-radius: 4px;
        padding: 8px 16px;
        color: white;
        font-weight: bold;
        margin: 2px;
        cursor: pointer;
        transition: all 0.2s;
    }}
    
    .quick-order-btn:hover {{
        opacity: 0.8;
    }}
    
    .position-pnl-positive {{
        color: {PROFESSIONAL_THEME['green']};
        background: rgba(0, 210, 92, 0.1);
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }}
    
    .position-pnl-negative {{
        color: {PROFESSIONAL_THEME['red']};
        background: rgba(255, 0, 92, 0.1);
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }}
    
    /* Custom metric styling */
    [data-testid="stMetric"] {{
        background: {PROFESSIONAL_THEME['bg_panel']};
        padding: 15px;
        border-radius: 8px;
        border: 1px solid {PROFESSIONAL_THEME['border']};
    }}
    
    /* Hide streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display:none;}}
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: {PROFESSIONAL_THEME['bg_panel']};
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        border: 1px solid {PROFESSIONAL_THEME['border']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {PROFESSIONAL_THEME['blue']} !important;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- Enhanced Market Data with Order Books ---
class OrderBook:
    def __init__(self, symbol):
        self.symbol = symbol
        self.bids = []  # (price, quantity)
        self.asks = []  # (price, quantity)
        self.last_update = time.time()
        
    def update(self, mid_price, spread_percent=0.001):
        """Generate realistic order book data"""
        self.bids = []
        self.asks = []
        
        spread = mid_price * spread_percent
        bid_price = mid_price - spread/2
        ask_price = mid_price + spread/2
        
        # Generate bids (descending prices)
        current_bid = bid_price
        for i in range(10):
            qty = random.randint(100, 10000)
            self.bids.append((current_bid, qty))
            current_bid -= random.uniform(0.001, 0.01) * mid_price
            
        # Generate asks (ascending prices)
        current_ask = ask_price
        for i in range(10):
            qty = random.randint(100, 10000)
            self.asks.append((current_ask, qty))
            current_ask += random.uniform(0.001, 0.01) * mid_price
            
        self.last_update = time.time()

# --- Enhanced Game State with Order Books ---
class EnhancedGameState:
    def __init__(self):
        self.players = {}
        self.game_status = "Stopped"
        self.game_start_time = 0
        self.round_duration_seconds = 20 * 60
        self.futures_expiry_time = 0
        self.futures_settled = False
        self.prices = {}
        self.base_real_prices = {}
        self.price_history = []
        self.transactions = {}
        self.market_sentiment = {s: 0 for s in ALL_SYMBOLS}
        self.liquidity = {s: random.uniform(0.5, 1.0) for s in ALL_SYMBOLS}
        self.event_active = False
        self.event_type = None
        self.event_target_symbol = None
        self.event_end = 0
        self.volatility_multiplier = 1.0
        self.news_feed = []
        self.auto_square_off_complete = False
        self.block_deal_offer = None
        self.closing_warning_triggered = False
        self.difficulty_level = 1
        self.current_margin_requirement = 0.2
        self.bid_ask_spread = 0.001
        self.slippage_threshold = 10
        self.base_slippage_rate = 0.005
        self.hft_rebate_window = 60
        self.hft_rebate_trades = 5
        self.hft_rebate_amount = 5000
        self.short_squeeze_threshold = 3
        self.order_books = {symbol: OrderBook(symbol) for symbol in ALL_SYMBOLS}
        self.market_depth = {}
        self.last_trades = []

@st.cache_resource
def get_enhanced_game_state():
    return EnhancedGameState()

# --- Professional Trading Components ---
def render_professional_header():
    """Professional trading platform header"""
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    
    with col1:
        st.markdown(f"<h1 style='color: {PROFESSIONAL_THEME['blue']}; margin: 0;'>BLOCKVISTA PRO</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {PROFESSIONAL_THEME['text_secondary']}; margin: 0;'>Professional Trading Terminal</p>", unsafe_allow_html=True)
    
    with col2:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.metric("🕒 Market Time", current_time)
    
    with col3:
        game_state = get_enhanced_game_state()
        if game_state.game_status == "Running":
            remaining = max(0, game_state.round_duration_seconds - int(time.time() - game_state.game_start_time))
            st.metric("⏰ Session", f"{remaining//60:02d}:{remaining%60:02d}")
        else:
            st.metric("⏰ Session", "CLOSED")
    
    with col4:
        st.metric("📊 Live Players", len(game_state.players))
    
    with col5:
        # Market status indicator
        if game_state.game_status == "Running":
            st.markdown(f"<div style='background: {PROFESSIONAL_THEME['green']}; color: black; padding: 5px; border-radius: 4px; text-align: center; font-weight: bold;'>LIVE</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background: {PROFESSIONAL_THEME['red']}; color: white; padding: 5px; border-radius: 4px; text-align: center; font-weight: bold;'>CLOSED</div>", unsafe_allow_html=True)

def render_advanced_chart(symbol, prices):
    """Professional trading chart with technical indicators"""
    try:
        # Get historical data for the chart
        hist_data = get_historical_data([symbol], period="1d")
        if hist_data.empty:
            return
        
        df = hist_data.iloc[-100:]  # Last 100 periods
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'] if 'Open' in df.columns else df.iloc[:,0],
            high=df['High'] if 'High' in df.columns else df.iloc[:,0],
            low=df['Low'] if 'Low' in df.columns else df.iloc[:,0],
            close=df['Close'] if 'Close' in df.columns else df.iloc[:,0],
            name="Price"
        ))
        
        # Add moving averages
        df['MA20'] = df.iloc[:,0].rolling(20).mean()
        df['MA50'] = df.iloc[:,0].rolling(50).mean()
        
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], line=dict(color='orange', width=1), name="MA20"))
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='blue', width=1), name="MA50"))
        
        fig.update_layout(
            height=400,
            plot_bgcolor=PROFESSIONAL_THEME['bg_panel'],
            paper_bgcolor=PROFESSIONAL_THEME['bg_panel'],
            font=dict(color=PROFESSIONAL_THEME['text_primary']),
            xaxis_rangeslider_visible=False,
            margin=dict(l=10, r=10, t=30, b=10),
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        fig.update_xaxes(gridcolor=PROFESSIONAL_THEME['border'])
        fig.update_yaxes(gridcolor=PROFESSIONAL_THEME['border'])
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Chart error: {e}")

def render_order_book(symbol, prices):
    """Professional order book visualization"""
    game_state = get_enhanced_game_state()
    order_book = game_state.order_books.get(symbol)
    
    if order_book:
        order_book.update(prices.get(symbol, 0))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**BID**")
            # Show top 5 bids
            for price, qty in sorted(order_book.bids, reverse=True)[:5]:
                depth_percent = (qty / 50000) * 100
                st.markdown(f"""
                <div class="order-book-bid">
                    <span style="float: left;">{format_indian_currency(price)}</span>
                    <span style="float: right;">{qty:,}</span>
                    <div style="clear: both;"></div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**ASK**")
            # Show top 5 asks
            for price, qty in sorted(order_book.asks)[:5]:
                depth_percent = (qty / 50000) * 100
                st.markdown(f"""
                <div class="order-book-ask">
                    <span style="float: left;">{format_indian_currency(price)}</span>
                    <span style="float: right;">{qty:,}</span>
                    <div style="clear: both;"></div>
                </div>
                """, unsafe_allow_html=True)

def render_market_depth(symbol, prices):
    """Market depth chart"""
    game_state = get_enhanced_game_state()
    order_book = game_state.order_books.get(symbol)
    
    if not order_book:
        return
    
    # Prepare data for depth chart
    bids = sorted(order_book.bids, reverse=True)
    asks = sorted(order_book.asks)
    
    bid_prices = [price for price, qty in bids]
    bid_volumes = [qty for price, qty in bids]
    ask_prices = [price for price, qty in asks]
    ask_volumes = [qty for price, qty in asks]
    
    if not bid_prices or not ask_prices:
        return
    
    fig = go.Figure()
    
    # Bid depth (left side)
    fig.add_trace(go.Bar(
        x=bid_volumes,
        y=bid_prices,
        orientation='h',
        name='Bid',
        marker_color=PROFESSIONAL_THEME['green'],
        opacity=0.7
    ))
    
    # Ask depth (right side)
    fig.add_trace(go.Bar(
        x=[-v for v in ask_volumes],  # Negative for right side
        y=ask_prices,
        orientation='h',
        name='Ask',
        marker_color=PROFESSIONAL_THEME['red'],
        opacity=0.7
    ))
    
    fig.update_layout(
        height=200,
        title="Market Depth",
        showlegend=False,
        plot_bgcolor=PROFESSIONAL_THEME['bg_panel'],
        paper_bgcolor=PROFESSIONAL_THEME['bg_panel'],
        font=dict(color=PROFESSIONAL_THEME['text_primary']),
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(showgrid=False, zeroline=True, zerolinewidth=2, zerolinecolor='white'),
        yaxis=dict(showgrid=True, gridcolor=PROFESSIONAL_THEME['border'])
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_pro_trading_interface(player_name, player, prices):
    """Professional trading interface with advanced features"""
    
    with st.container():
        # Top bar with key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        pnl = total_value - (INITIAL_CAPITAL * 5 if player['mode'] == 'HNI' else INITIAL_CAPITAL)
        
        with col1:
            st.metric("💰 Cash", format_indian_currency(player['capital']))
        with col2:
            st.metric("📈 Portfolio", format_indian_currency(total_value))
        with col3:
            delta_color = "normal" if pnl >= 0 else "inverse"
            st.metric("💸 P&L", format_indian_currency(pnl), delta=format_indian_currency(pnl), delta_color=delta_color)
        with col4:
            margin_used = abs(holdings_value) * get_enhanced_game_state().current_margin_requirement
            st.metric("⚡ Margin Used", format_indian_currency(margin_used))
        with col5:
            st.metric("🎯 Mode", player['mode'])
    
    # Main trading area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chart and trading area
        st.markdown("<div class='pro-panel'>", unsafe_allow_html=True)
        st.subheader("📊 Advanced Charting")
        
        # Symbol selector for chart
        chart_symbol = st.selectbox("Select Symbol", 
                                   [s.replace('.NS', '') for s in NIFTY50_SYMBOLS] + CRYPTO_SYMBOLS,
                                   key="chart_symbol")
        
        full_symbol = chart_symbol + '.NS' if chart_symbol in [s.replace('.NS', '') for s in NIFTY50_SYMBOLS] else chart_symbol
        render_advanced_chart(full_symbol, prices)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Order entry panel
        st.markdown("<div class='pro-panel'>", unsafe_allow_html=True)
        st.subheader("🚀 Quick Order")
        
        # Symbol selection
        symbol = st.selectbox("Symbol", 
                             [s.replace('.NS', '') for s in NIFTY50_SYMBOLS] + CRYPTO_SYMBOLS,
                             key="order_symbol")
        full_symbol = symbol + '.NS' if symbol in [s.replace('.NS', '') for s in NIFTY50_SYMBOLS] else symbol
        
        # Current price display
        current_price = prices.get(full_symbol, 0)
        price_change = 0
        if len(get_enhanced_game_state().price_history) >= 2:
            prev_price = get_enhanced_game_state().price_history[-2].get(full_symbol, current_price)
            price_change = ((current_price - prev_price) / prev_price) * 100
        
        price_color = PROFESSIONAL_THEME['green'] if price_change >= 0 else PROFESSIONAL_THEME['red']
        st.markdown(f"<h3 style='color: {price_color};'>{format_indian_currency(current_price)} ({price_change:+.2f}%)</h3>", unsafe_allow_html=True)
        
        # Order type
        order_type = st.radio("Order Type", ["Market", "Limit", "Stop"], horizontal=True)
        
        # Quantity with quick buttons
        qty = st.number_input("Quantity", min_value=1, value=100, step=1)
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        with col_q1:
            if st.button("100", use_container_width=True):
                st.session_state.quantity = 100
        with col_q2:
            if st.button("500", use_container_width=True):
                st.session_state.quantity = 500
        with col_q3:
            if st.button("1000", use_container_width=True):
                st.session_state.quantity = 1000
        with col_q4:
            if st.button("MAX", use_container_width=True):
                # Calculate max affordable quantity
                max_qty = int(player['capital'] / current_price)
                st.session_state.quantity = max(1, max_qty)
        
        if 'quantity' in st.session_state:
            qty = st.session_state.quantity
        
        # Price for limit/stop orders
        if order_type != "Market":
            limit_price = st.number_input("Price", min_value=0.01, value=current_price, step=0.01)
        
        # Order buttons
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            if st.button("BUY", type="primary", use_container_width=True, 
                        disabled=get_enhanced_game_state().game_status != "Running"):
                if execute_trade(player_name, player, "Buy", full_symbol, qty, prices):
                    play_sound('success')
                else:
                    play_sound('error')
                st.rerun()
        
        with col_b2:
            if st.button("SELL", use_container_width=True,
                        disabled=get_enhanced_game_state().game_status != "Running"):
                if execute_trade(player_name, player, "Sell", full_symbol, qty, prices):
                    play_sound('success')
                else:
                    play_sound('error')
                st.rerun()
        
        with col_b3:
            if st.button("SHORT", use_container_width=True,
                        disabled=get_enhanced_game_state().game_status != "Running"):
                if execute_trade(player_name, player, "Short", full_symbol, qty, prices):
                    play_sound('success')
                else:
                    play_sound('error')
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Order book
        st.markdown("<div class='pro-panel'>", unsafe_allow_html=True)
        st.subheader("📋 Order Book")
        render_order_book(full_symbol, prices)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Bottom panels
    col3, col4 = st.columns(2)
    
    with col3:
        # Positions
        st.markdown("<div class='pro-panel'>", unsafe_allow_html=True)
        st.subheader("💼 Positions")
        render_enhanced_positions(player, prices)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        # Market depth
        st.markdown("<div class='pro-panel'>", unsafe_allow_html=True)
        st.subheader("📈 Market Depth")
        render_market_depth(full_symbol, prices)
        st.markdown("</div>", unsafe_allow_html=True)

def render_enhanced_positions(player, prices):
    """Enhanced positions display with P&L and quick actions"""
    if not player['holdings']:
        st.info("No active positions")
        return
    
    positions_data = []
    for symbol, qty in player['holdings'].items():
        current_price = prices.get(symbol, 0)
        value = current_price * abs(qty)
        avg_price = current_price * 0.98  # Simplified for demo
        
        if qty > 0:  # Long position
            pnl = (current_price - avg_price) * qty
            pnl_percent = ((current_price - avg_price) / avg_price) * 100
        else:  # Short position
            pnl = (avg_price - current_price) * abs(qty)
            pnl_percent = ((avg_price - current_price) / avg_price) * 100
        
        positions_data.append({
            "Symbol": symbol,
            "Qty": qty,
            "Avg Price": avg_price,
            "Current": current_price,
            "P&L": pnl,
            "P&L %": pnl_percent,
            "Value": value
        })
    
    for pos in positions_data:
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.write(f"**{pos['Symbol']}**")
            st.write(f"{pos['Qty']} shares")
        
        with col2:
            pnl_color = "green" if pos['P&L'] >= 0 else "red"
            st.markdown(f"<span style='color: {pnl_color}; font-weight: bold;'>{format_indian_currency(pos['P&L'])} ({pos['P&L %']:+.2f}%)</span>", unsafe_allow_html=True)
        
        with col3:
            if st.button("✕", key=f"close_{pos['Symbol']}", help="Close Position"):
                # Execute closing trade
                action = "Sell" if pos['Qty'] > 0 else "Buy"
                execute_trade(player['name'], player, action, pos['Symbol'], abs(pos['Qty']), prices)
                st.rerun()
        
        st.divider()

def render_professional_news_feed():
    """Professional news feed with impact indicators"""
    game_state = get_enhanced_game_state()
    
    st.markdown("<div class='pro-panel'>", unsafe_allow_html=True)
    st.subheader("📰 Market News")
    
    if not game_state.news_feed:
        st.info("No market news at the moment")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    for news in game_state.news_feed[:10]:  # Show last 10 news items
        # Determine impact color
        if any(word in news.lower() for word in ['rally', 'bull', 'up', 'gain', 'profit']):
            impact_color = PROFESSIONAL_THEME['green']
            impact_icon = "📈"
        elif any(word in news.lower() for word in ['crash', 'bear', 'down', 'loss', 'fraud']):
            impact_color = PROFESSIONAL_THEME['red']
            impact_icon = "📉"
        else:
            impact_color = PROFESSIONAL_THEME['yellow']
            impact_icon = "📊"
        
        col1, col2 = st.columns([1, 20])
        with col1:
            st.markdown(f"<span style='color: {impact_color}; font-size: 1.2em;'>{impact_icon}</span>", unsafe_allow_html=True)
        with col2:
            st.write(news)
        
        st.divider()
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_advanced_analytics(player, prices):
    """Advanced analytics and risk metrics"""
    st.markdown("<div class='pro-panel'>", unsafe_allow_html=True)
    st.subheader("📊 Advanced Analytics")
    
    # Portfolio metrics
    holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
    total_value = player['capital'] + holdings_value
    
    # Calculate beta (simplified)
    portfolio_return = calculate_sharpe_ratio(player.get('value_history', [])) / 10  # Simplified
    market_return = 0.08  # Assumed market return
    
    beta = min(2.0, max(0.5, portfolio_return / market_return if market_return != 0 else 1.0))
    
    # Risk metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sharpe Ratio", f"{calculate_sharpe_ratio(player.get('value_history', [])):.2f}")
    
    with col2:
        st.metric("Portfolio Beta", f"{beta:.2f}")
    
    with col3:
        var_95 = total_value * 0.05  # Simplified VaR calculation
        st.metric("VaR (95%)", format_indian_currency(-var_95))
    
    with col4:
        max_drawdown = calculate_max_drawdown(player.get('value_history', []))
        st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
    
    st.markdown("</div>", unsafe_allow_html=True)

def calculate_max_drawdown(value_history):
    """Calculate maximum drawdown"""
    if len(value_history) < 2:
        return 0.0
    
    peak = value_history[0]
    max_dd = 0.0
    
    for value in value_history[1:]:
        if value > peak:
            peak = value
        dd = (peak - value) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    return max_dd

def render_professional_leaderboard(prices):
    """Enhanced leaderboard with professional styling"""
    game_state = get_enhanced_game_state()
    
    st.markdown("<div class='pro-panel'>", unsafe_allow_html=True)
    st.subheader("🏆 Leaderboard")
    
    lb = []
    for pname, pdata in game_state.players.items():
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in pdata['holdings'].items())
        total_value = pdata['capital'] + holdings_value
        sharpe_ratio = calculate_sharpe_ratio(pdata.get('value_history', []))
        
        lb.append({
            "Player": pname,
            "Mode": pdata['mode'],
            "Portfolio Value": total_value,
            "P&L": pdata['pnl'],
            "Sharpe Ratio": sharpe_ratio
        })
    
    if lb:
        lb_df = pd.DataFrame(lb).sort_values("Portfolio Value", ascending=False).reset_index(drop=True)
        
        # Add ranking
        lb_df['Rank'] = lb_df.index + 1
        
        # Style the dataframe
        def style_leaderboard(row):
            if row['Rank'] == 1:
                return ['background: rgba(255, 204, 0, 0.2)'] * len(row)
            elif row['Rank'] <= 3:
                return ['background: rgba(192, 192, 192, 0.2)'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = lb_df.style.format({
            "Portfolio Value": lambda x: format_indian_currency(x),
            "P&L": lambda x: format_indian_currency(x),
            "Sharpe Ratio": "{:.2f}"
        }).apply(style_leaderboard, axis=1)
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("No players yet")
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- Enhanced Main Interface ---
def render_enhanced_main_interface(prices):
    """Main interface with professional trading layout"""
    inject_pro_css()
    render_professional_header()
    
    game_state = get_enhanced_game_state()
    
    if st.session_state.get('role') == 'admin':
        render_admin_dashboard(prices)
    elif 'player' in st.query_params:
        player_name = st.query_params.get("player")
        if player_name in game_state.players:
            player = game_state.players[player_name]
            
            # Main trading layout
            render_pro_trading_interface(player_name, player, prices)
            
            # Bottom panels
            col1, col2 = st.columns([2, 1])
            
            with col1:
                render_professional_news_feed()
                render_advanced_analytics(player, prices)
            
            with col2:
                render_professional_leaderboard(prices)
        else:
            st.error("Player not found")
    else:
        # Welcome screen for new users
        st.markdown("<div class='pro-panel'>", unsafe_allow_html=True)
        st.title("🚀 Welcome to BlockVista Pro")
        st.markdown("""
        ### Professional Trading Platform
        
        Features:
        - 📊 **Advanced Charting** with technical indicators
        - 📋 **Real-time Order Book** with market depth
        - 🚀 **Quick Order Execution** with one-click trading
        - 📰 **Live Market News** with impact analysis
        - 📈 **Advanced Analytics** with risk metrics
        - 💼 **Portfolio Management** with real-time P&L
        
        *Join the game from the sidebar to start trading!*
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        
        render_professional_leaderboard(prices)

# --- Replace the existing main interface calls ---
# In your main() function, replace render_main_interface with:
# render_enhanced_main_interface(final_prices)

# Also update the game state initialization to use EnhancedGameState

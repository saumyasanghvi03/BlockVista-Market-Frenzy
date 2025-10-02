# ======================= BLOCKVISTA PRO TRADING PLATFORM ======================

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
st.set_page_config(layout="wide", page_title="BlockVista Pro Trading", page_icon="📈", initial_sidebar_state="expanded")

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

# --- API & Game Configuration ---
GAME_NAME = "BlockVista Pro Trading"
INITIAL_CAPITAL = 1000000  # ₹10L

# Define asset symbols
NIFTY50_SYMBOLS = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS',
                   'KOTAKBANK.NS', 'SBIN.NS', 'ITC.NS', 'ASIANPAINT.NS', 'AXISBANK.NS']
CRYPTO_SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'ADA-USD', 'XRP-USD']
GOLD_SYMBOL = 'GC=F'
NIFTY_INDEX_SYMBOL = '^NSEI'
BANKNIFTY_INDEX_SYMBOL = '^NSEBANK'
FUTURES_SYMBOLS = ['NIFTY-FUT', 'BANKNIFTY-FUT']
LEVERAGED_ETFS = ['NIFTY_BULL_3X', 'NIFTY_BEAR_3X']
OPTION_SYMBOLS = ['NIFTY_CALL', 'NIFTY_PUT']

ALL_SYMBOLS = NIFTY50_SYMBOLS + CRYPTO_SYMBOLS + [GOLD_SYMBOL] + OPTION_SYMBOLS + FUTURES_SYMBOLS + LEVERAGED_ETFS

# Game Mechanics Settings
ADMIN_PASSWORD = "100370"

# --- Pre-built News Headlines ---
PRE_BUILT_NEWS = [
    {"headline": "Breaking: RBI unexpectedly cuts repo rate by 25 basis points!", "impact": "Bull Rally"},
    {"headline": "Government announces major infrastructure spending package, boosting banking stocks.", "impact": "Banking Boost"},
    {"headline": "Shocking fraud uncovered at a major private bank, sending shockwaves through the financial sector.", "impact": "Flash Crash"},
    {"headline": "Indian tech firm announces breakthrough in AI, sparking a rally in tech stocks.", "impact": "Sector Rotation"},
    {"headline": "FIIs show renewed interest in Indian equities, leading to broad-based buying.", "impact": "Bull Rally"},
    {"headline": "SEBI announces stricter margin rules for derivatives, market turns cautious.", "impact": "Flash Crash"},
    {"headline": "Monsoon forecast revised upwards, boosting rural demand expectations.", "impact": "Bull Rally"},
    {"headline": "Global News: US inflation data comes in hotter than expected, spooking global markets.", "impact": "Flash Crash"},
    {"headline": "Global News: European Central Bank signals a dovish stance, boosting liquidity.", "impact": "Bull Rally"},
    {"headline": "Global News: Major supply chain disruption reported in Asia, affecting global trade.", "impact": "Volatility Spike"},
    {"headline": "{symbol} secures a massive government contract, sending its stock soaring!", "impact": "Symbol Bull Run"},
    {"headline": "Regulatory probe launched into {symbol} over accounting irregularities.", "impact": "Symbol Crash"},
    {"headline": "{symbol} announces a surprise stock split, shares to adjust at market open.", "impact": "Stock Split"},
    {"headline": "{symbol} declares a special dividend for all shareholders.", "impact": "Dividend"},
    {"headline": "High short interest in {symbol} triggers a potential short squeeze!", "impact": "Short Squeeze"},
]

# --- Professional CSS Injection ---
def inject_pro_css():
    st.markdown(f"""
    <style>
    .main .block-container {{
        padding-top: 1rem;
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
        padding: 4px 8px;
        border-radius: 4px;
        margin: 2px 0;
        font-family: 'Courier New', monospace;
    }}
    
    .order-book-ask {{
        background: linear-gradient(90deg, rgba(255, 0, 92, 0.1) 0%, transparent 100%);
        padding: 4px 8px;
        border-radius: 4px;
        margin: 2px 0;
        font-family: 'Courier New', monospace;
    }}
    
    .position-pnl-positive {{
        color: {PROFESSIONAL_THEME['green']};
        background: rgba(0, 210, 92, 0.1);
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.9em;
    }}
    
    .position-pnl-negative {{
        color: {PROFESSIONAL_THEME['red']};
        background: rgba(255, 0, 92, 0.1);
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 0.9em;
    }}
    
    /* Custom metric styling */
    [data-testid="stMetric"] {{
        background: {PROFESSIONAL_THEME['bg_panel']};
        padding: 15px;
        border-radius: 8px;
        border: 1px solid {PROFESSIONAL_THEME['border']};
    }}
    
    [data-testid="stMetricDelta"] {{
        font-weight: bold;
    }}
    
    /* Hide streamlit branding */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display:none;}}
    header {{visibility: hidden;}}
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2px;
        background-color: {PROFESSIONAL_THEME['bg_dark']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: {PROFESSIONAL_THEME['bg_panel']};
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
        border: 1px solid {PROFESSIONAL_THEME['border']};
        color: {PROFESSIONAL_THEME['text_secondary']};
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {PROFESSIONAL_THEME['blue']} !important;
        color: white !important;
    }}
    
    /* Button styling */
    .stButton button {{
        border: 1px solid {PROFESSIONAL_THEME['border']};
        border-radius: 4px;
        transition: all 0.2s;
    }}
    
    .stButton button:hover {{
        border-color: {PROFESSIONAL_THEME['blue']};
        transform: translateY(-1px);
    }}
    
    /* Dataframe styling */
    .dataframe {{
        background-color: {PROFESSIONAL_THEME['bg_panel']} !important;
        color: {PROFESSIONAL_THEME['text_primary']} !important;
    }}
    
    .dataframe th {{
        background-color: {PROFESSIONAL_THEME['bg_hover']} !important;
        color: {PROFESSIONAL_THEME['text_primary']} !important;
    }}
    
    .dataframe td {{
        background-color: {PROFESSIONAL_THEME['bg_panel']} !important;
        color: {PROFESSIONAL_THEME['text_primary']} !important;
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

# --- Enhanced Game State ---
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

    def reset(self):
        """Resets the game to its initial state"""
        base_prices = self.base_real_prices
        difficulty = self.difficulty_level
        self.__init__()
        self.base_real_prices = base_prices
        self.difficulty_level = difficulty

@st.cache_resource
def get_game_state():
    return EnhancedGameState()

# --- Sound Effects ---
def play_sound(sound_type):
    js = ""
    if sound_type == 'success':
        js = """
        <script>
            if (typeof Tone !== 'undefined') {
                const synth = new Tone.Synth().toDestination();
                synth.triggerAttackRelease("C5", "8n");
            }
        </script>
        """
    elif sound_type == 'error':
        js = """
        <script>
            if (typeof Tone !== 'undefined') {
                const synth = new Tone.Synth().toDestination();
                synth.triggerAttackRelease("C3", "8n");
            }
        </script>
        """
    st.components.v1.html(js, height=0)

def announce_news(headline):
    safe_headline = headline.replace("'", "\\'").replace("\n", " ")
    js = f"""
    <script>
        if ('speechSynthesis' in window) {{
            const utterance = new SpeechSynthesisUtterance('{safe_headline}');
            speechSynthesis.speak(utterance);
        }}
    </script>
    """
    st.components.v1.html(js, height=0)

# --- Data Fetching & Market Simulation ---
@st.cache_data(ttl=86400)
def get_daily_base_prices():
    prices = {}
    yf_symbols = NIFTY50_SYMBOLS + CRYPTO_SYMBOLS + [GOLD_SYMBOL, NIFTY_INDEX_SYMBOL, BANKNIFTY_INDEX_SYMBOL]
    try:
        data = yf.download(tickers=yf_symbols, period="1d", interval="1m", progress=False)
        for symbol in yf_symbols:
            if not data.empty and symbol in data['Close'] and not pd.isna(data['Close'][symbol].iloc[-1]):
                prices[symbol] = data['Close'][symbol].iloc[-1]
            else:
                prices[symbol] = random.uniform(10, 50000)
    except Exception:
        for symbol in yf_symbols:
            prices[symbol] = random.uniform(10, 50000)
    return prices

def simulate_tick_prices(last_prices):
    game_state = get_game_state()
    simulated_prices = last_prices.copy()
    volatility_multiplier = getattr(game_state, 'volatility_multiplier', 1.0)

    for symbol, price in simulated_prices.items():
        if symbol not in FUTURES_SYMBOLS + LEVERAGED_ETFS + OPTION_SYMBOLS:
            sentiment = game_state.market_sentiment.get(symbol, 0)
            noise = random.uniform(-0.0005, 0.0005) * volatility_multiplier
            price_multiplier = 1 + (sentiment * 0.001) + noise
            simulated_prices[symbol] = price * price_multiplier
            
    return simulated_prices

def calculate_derived_prices(base_prices):
    game_state = get_game_state()
    derived_prices = base_prices.copy()

    nifty_price = derived_prices.get(NIFTY_INDEX_SYMBOL, 20000)
    banknifty_price = derived_prices.get(BANKNIFTY_INDEX_SYMBOL, 45000)
    
    derived_prices['NIFTY_CALL'] = nifty_price * 1.02
    derived_prices['NIFTY_PUT'] = nifty_price * 0.98
    
    derived_prices['NIFTY-FUT'] = nifty_price * random.uniform(1.0, 1.005)
    derived_prices['BANKNIFTY-FUT'] = banknifty_price * random.uniform(1.0, 1.005)
    
    if len(game_state.price_history) >= 2:
        prev_nifty = game_state.price_history[-2].get(NIFTY_INDEX_SYMBOL, nifty_price)
        nifty_change = (nifty_price - prev_nifty) / prev_nifty
        
        current_bull = game_state.prices.get('NIFTY_BULL_3X', nifty_price/100)
        current_bear = game_state.prices.get('NIFTY_BEAR_3X', nifty_price/100)

        derived_prices['NIFTY_BULL_3X'] = current_bull * (1 + 3 * nifty_change)
        derived_prices['NIFTY_BEAR_3X'] = current_bear * (1 - 3 * nifty_change)
    else:
        derived_prices['NIFTY_BULL_3X'] = nifty_price / 100
        derived_prices['NIFTY_BEAR_3X'] = nifty_price / 100

    return derived_prices

@st.cache_data(ttl=3600)
def get_historical_data(symbols, period="6mo"):
    try:
        data = yf.download(tickers=symbols, period=period, progress=False)
        close_data = data['Close']
        if isinstance(close_data, pd.Series):
            return close_data.to_frame(name=symbols[0])
        return close_data
    except Exception:
        return pd.DataFrame()

# --- Professional UI Components ---
def render_professional_header():
    col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
    
    with col1:
        st.markdown(f"<h1 style='color: {PROFESSIONAL_THEME['blue']}; margin: 0; font-size: 2.5em;'>BLOCKVISTA PRO</h1>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: {PROFESSIONAL_THEME['text_secondary']}; margin: 0;'>Professional Trading Terminal</p>", unsafe_allow_html=True)
    
    with col2:
        current_time = datetime.now().strftime("%H:%M:%S")
        st.metric("🕒 Market Time", current_time)
    
    with col3:
        game_state = get_game_state()
        if game_state.game_status == "Running":
            remaining = max(0, game_state.round_duration_seconds - int(time.time() - game_state.game_start_time))
            st.metric("⏰ Session", f"{remaining//60:02d}:{remaining%60:02d}")
        else:
            st.metric("⏰ Session", "CLOSED")
    
    with col4:
        st.metric("📊 Live Players", len(game_state.players))
    
    with col5:
        if game_state.game_status == "Running":
            st.markdown(f"<div style='background: {PROFESSIONAL_THEME['green']}; color: black; padding: 8px; border-radius: 4px; text-align: center; font-weight: bold;'>LIVE</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background: {PROFESSIONAL_THEME['red']}; color: white; padding: 8px; border-radius: 4px; text-align: center; font-weight: bold;'>CLOSED</div>", unsafe_allow_html=True)

def render_advanced_chart(symbol, prices):
    try:
        # Create synthetic price data for demonstration
        dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
        base_price = prices.get(symbol, 100)
        
        # Generate realistic price movement
        returns = np.random.normal(0, 0.02, 100)
        price_series = base_price * (1 + returns).cumprod()
        
        # Create OHLC data
        df = pd.DataFrame({
            'Date': dates,
            'Open': price_series * 0.998,
            'High': price_series * 1.005,
            'Low': price_series * 0.995,
            'Close': price_series
        }).set_index('Date')
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name="Price"
        ))
        
        # Add moving averages
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        
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
        
        fig.update_xaxes(gridcolor=PROFESSIONAL_THEME['border'], showgrid=True)
        fig.update_yaxes(gridcolor=PROFESSIONAL_THEME['border'], showgrid=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Chart error: {e}")

def render_order_book(symbol, prices):
    game_state = get_game_state()
    order_book = game_state.order_books.get(symbol)
    
    if order_book:
        order_book.update(prices.get(symbol, 0))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**<span style='color: #00d25c;'>BID</span>**", unsafe_allow_html=True)
            for price, qty in sorted(order_book.bids, reverse=True)[:5]:
                st.markdown(f"""
                <div class="order-book-bid">
                    <span style="float: left; font-weight: bold;">{format_indian_currency(price)}</span>
                    <span style="float: right; color: #8b949e;">{qty:,}</span>
                    <div style="clear: both;"></div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("**<span style='color: #ff005c;'>ASK</span>**", unsafe_allow_html=True)
            for price, qty in sorted(order_book.asks)[:5]:
                st.markdown(f"""
                <div class="order-book-ask">
                    <span style="float: left; font-weight: bold;">{format_indian_currency(price)}</span>
                    <span style="float: right; color: #8b949e;">{qty:,}</span>
                    <div style="clear: both;"></div>
                </div>
                """, unsafe_allow_html=True)

def render_market_depth(symbol, prices):
    game_state = get_game_state()
    order_book = game_state.order_books.get(symbol)
    
    if not order_book:
        return
    
    bids = sorted(order_book.bids, reverse=True)[:8]
    asks = sorted(order_book.asks)[:8]
    
    if not bids or not asks:
        return
    
    bid_prices = [price for price, qty in bids]
    bid_volumes = [qty for price, qty in bids]
    ask_prices = [price for price, qty in asks]
    ask_volumes = [qty for price, qty in asks]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=bid_volumes,
        y=bid_prices,
        orientation='h',
        name='Bid',
        marker_color=PROFESSIONAL_THEME['green'],
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        x=[-v for v in ask_volumes],
        y=ask_prices,
        orientation='h',
        name='Ask',
        marker_color=PROFESSIONAL_THEME['red'],
        opacity=0.7
    ))
    
    fig.update_layout(
        height=200,
        showlegend=False,
        plot_bgcolor=PROFESSIONAL_THEME['bg_panel'],
        paper_bgcolor=PROFESSIONAL_THEME['bg_panel'],
        font=dict(color=PROFESSIONAL_THEME['text_primary']),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(showgrid=False, zeroline=True, zerolinewidth=2, zerolinecolor='white', showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor=PROFESSIONAL_THEME['border'])
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- Trading Functions ---
def format_indian_currency(n):
    if n is None: return "₹0.00"
    n = float(n)
    if abs(n) < 100000: return f"₹{n:,.2f}"
    elif abs(n) < 10000000: return f"₹{n/100000:.2f}L"
    else: return f"₹{n/10000000:.2f}Cr"

def calculate_slippage(player, symbol, qty, action):
    game_state = get_game_state()
    liquidity_level = game_state.liquidity.get(symbol, 1.0)
    if qty <= game_state.slippage_threshold: return 1.0
    
    slippage_multiplier = player.get('slippage_multiplier', 1.0)
    
    excess_qty = qty - game_state.slippage_threshold
    slippage_rate = (game_state.base_slippage_rate / max(0.1, liquidity_level)) * slippage_multiplier
    slippage_mult = 1 + (slippage_rate * excess_qty) * (1 if action == "Buy" else -1)
    return max(0.9, min(1.1, slippage_mult))

def execute_trade(player_name, player, action, symbol, qty, prices, is_algo=False, order_type="Market"):
    game_state = get_game_state()
    mid_price = prices.get(symbol, 0)
    if mid_price == 0: return False

    if action == "Buy":
        trade_price = mid_price * (1 + game_state.bid_ask_spread / 2)
    else:
        trade_price = mid_price * (1 - game_state.bid_ask_spread / 2)
    
    trade_price *= calculate_slippage(player, symbol, qty, action)
    cost = trade_price * qty
    
    trade_executed = False
    if action == "Buy" and player['capital'] >= cost:
        player['capital'] -= cost
        player['holdings'][symbol] = player['holdings'].get(symbol, 0) + qty
        trade_executed = True
    elif action == "Short" and player['capital'] >= cost * game_state.current_margin_requirement:
        player['capital'] += cost
        player['holdings'][symbol] = player['holdings'].get(symbol, 0) - qty
        trade_executed = True
    elif action == "Sell":
        current_qty = player['holdings'].get(symbol, 0)
        if current_qty > 0 and current_qty >= qty:
            player['capital'] += cost
            player['holdings'][symbol] -= qty
            trade_executed = True
        elif current_qty < 0 and abs(current_qty) >= qty:
            player['capital'] -= cost
            player['holdings'][symbol] += qty
            trade_executed = True
        if trade_executed and player['holdings'][symbol] == 0:
            del player['holdings'][symbol]
    
    if trade_executed: 
        game_state.market_sentiment[symbol] = game_state.market_sentiment.get(symbol, 0) + (qty / 50) * (1 if action in ["Buy", "Short"] else -1)
        log_transaction(player_name, f"{order_type} {action}", symbol, qty, trade_price, cost, is_algo)
    elif not is_algo: 
        st.error("Trade failed: Insufficient capital or holdings.")
    return trade_executed

def log_transaction(player_name, action, symbol, qty, price, total, is_algo=False):
    game_state = get_game_state()
    prefix = "🤖 Algo" if is_algo else ""
    game_state.transactions.setdefault(player_name, []).append([time.strftime("%H:%M:%S"), f"{prefix} {action}".strip(), symbol, qty, price, total])

# --- Professional Trading Interface ---
def render_pro_trading_interface(player_name, player, prices):
    with st.container():
        # Top metrics bar
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
            margin_used = abs(holdings_value) * get_game_state().current_margin_requirement
            st.metric("⚡ Margin Used", format_indian_currency(margin_used))
        with col5:
            st.metric("🎯 Mode", player['mode'])
    
    # Main trading area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("<div class='pro-panel'>", unsafe_allow_html=True)
        st.subheader("📊 Advanced Charting")
        
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
        
        symbol = st.selectbox("Symbol", 
                             [s.replace('.NS', '') for s in NIFTY50_SYMBOLS] + CRYPTO_SYMBOLS,
                             key="order_symbol")
        full_symbol = symbol + '.NS' if symbol in [s.replace('.NS', '') for s in NIFTY50_SYMBOLS] else symbol
        
        current_price = prices.get(full_symbol, 0)
        price_change = 0
        if len(get_game_state().price_history) >= 2:
            prev_price = get_game_state().price_history[-2].get(full_symbol, current_price)
            price_change = ((current_price - prev_price) / prev_price) * 100
        
        price_color = PROFESSIONAL_THEME['green'] if price_change >= 0 else PROFESSIONAL_THEME['red']
        st.markdown(f"<h3 style='color: {price_color}; text-align: center;'>{format_indian_currency(current_price)} <span style='font-size: 0.6em;'>({price_change:+.2f}%)</span></h3>", unsafe_allow_html=True)
        
        order_type = st.radio("Order Type", ["Market", "Limit", "Stop"], horizontal=True)
        
        qty = st.number_input("Quantity", min_value=1, value=100, step=1, key="order_qty")
        
        # Quick quantity buttons
        col_q1, col_q2, col_q3, col_q4 = st.columns(4)
        with col_q1:
            if st.button("100", use_container_width=True, key="qty_100"):
                st.session_state.order_qty = 100
        with col_q2:
            if st.button("500", use_container_width=True, key="qty_500"):
                st.session_state.order_qty = 500
        with col_q3:
            if st.button("1000", use_container_width=True, key="qty_1000"):
                st.session_state.order_qty = 1000
        with col_q4:
            if st.button("MAX", use_container_width=True, key="qty_max"):
                max_qty = int(player['capital'] / current_price) if current_price > 0 else 1
                st.session_state.order_qty = max(1, max_qty)
        
        if order_type != "Market":
            limit_price = st.number_input("Price", min_value=0.01, value=current_price, step=0.01, key="limit_price")
        
        # Order buttons
        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            if st.button("BUY", type="primary", use_container_width=True, 
                        disabled=get_game_state().game_status != "Running", key="buy_btn"):
                if execute_trade(player_name, player, "Buy", full_symbol, qty, prices):
                    play_sound('success')
                else:
                    play_sound('error')
                st.rerun()
        
        with col_b2:
            if st.button("SELL", use_container_width=True,
                        disabled=get_game_state().game_status != "Running", key="sell_btn"):
                if execute_trade(player_name, player, "Sell", full_symbol, qty, prices):
                    play_sound('success')
                else:
                    play_sound('error')
                st.rerun()
        
        with col_b3:
            if st.button("SHORT", use_container_width=True,
                        disabled=get_game_state().game_status != "Running", key="short_btn"):
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
        st.markdown("<div class='pro-panel'>", unsafe_allow_html=True)
        st.subheader("💼 Positions")
        render_enhanced_positions(player, prices)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='pro-panel'>", unsafe_allow_html=True)
        st.subheader("📈 Market Depth")
        render_market_depth(full_symbol, prices)
        st.markdown("</div>", unsafe_allow_html=True)

def render_enhanced_positions(player, prices):
    if not player['holdings']:
        st.info("No active positions")
        return
    
    for symbol, qty in player['holdings'].items():
        current_price = prices.get(symbol, 0)
        value = current_price * abs(qty)
        avg_price = current_price * random.uniform(0.95, 1.05)  # Simulated avg price
        
        if qty > 0:
            pnl = (current_price - avg_price) * qty
            pnl_percent = ((current_price - avg_price) / avg_price) * 100
        else:
            pnl = (avg_price - current_price) * abs(qty)
            pnl_percent = ((avg_price - current_price) / avg_price) * 100
        
        col1, col2, col3 = st.columns([3, 2, 1])
        
        with col1:
            st.write(f"**{symbol}**")
            st.write(f"{abs(qty):,} shares {'LONG' if qty > 0 else 'SHORT'}")
        
        with col2:
            pnl_color_class = "position-pnl-positive" if pnl >= 0 else "position-pnl-negative"
            st.markdown(f"<div class='{pnl_color_class}'>{format_indian_currency(pnl)} ({pnl_percent:+.2f}%)</div>", unsafe_allow_html=True)
        
        with col3:
            if st.button("✕", key=f"close_{symbol}", help="Close Position"):
                action = "Sell" if qty > 0 else "Buy"
                execute_trade(player['name'], player, action, symbol, abs(qty), prices)
                st.rerun()
        
        st.divider()

def render_professional_news_feed():
    game_state = get_game_state()
    
    st.markdown("<div class='pro-panel'>", unsafe_allow_html=True)
    st.subheader("📰 Market News")
    
    if not game_state.news_feed:
        st.info("No market news at the moment")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    
    for news in game_state.news_feed[:8]:
        if any(word in news.lower() for word in ['rally', 'bull', 'up', 'gain', 'profit', 'positive']):
            impact_color = PROFESSIONAL_THEME['green']
            impact_icon = "📈"
        elif any(word in news.lower() for word in ['crash', 'bear', 'down', 'loss', 'fraud', 'negative']):
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

def calculate_sharpe_ratio(value_history):
    if len(value_history) < 2: return 0.0
    returns = pd.Series(value_history).pct_change().dropna()
    if returns.std() == 0: return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(252)

def calculate_max_drawdown(value_history):
    if len(value_history) < 2: return 0.0
    peak = value_history[0]
    max_dd = 0.0
    for value in value_history[1:]:
        if value > peak: peak = value
        dd = (peak - value) / peak * 100
        if dd > max_dd: max_dd = dd
    return max_dd

def render_advanced_analytics(player, prices):
    st.markdown("<div class='pro-panel'>", unsafe_allow_html=True)
    st.subheader("📊 Advanced Analytics")
    
    holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
    total_value = player['capital'] + holdings_value
    
    portfolio_return = calculate_sharpe_ratio(player.get('value_history', [])) / 10
    market_return = 0.08
    beta = min(2.0, max(0.5, portfolio_return / market_return if market_return != 0 else 1.0))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sharpe Ratio", f"{calculate_sharpe_ratio(player.get('value_history', [])):.2f}")
    
    with col2:
        st.metric("Portfolio Beta", f"{beta:.2f}")
    
    with col3:
        var_95 = total_value * 0.05
        st.metric("VaR (95%)", format_indian_currency(-var_95))
    
    with col4:
        max_drawdown = calculate_max_drawdown(player.get('value_history', []))
        st.metric("Max Drawdown", f"{max_drawdown:.1f}%")
    
    st.markdown("</div>", unsafe_allow_html=True)

def render_professional_leaderboard(prices):
    game_state = get_game_state()
    
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
        lb_df['Rank'] = lb_df.index + 1
        
        def style_leaderboard(row):
            if row['Rank'] == 1:
                return ['background: rgba(255, 215, 0, 0.2)'] * len(row)
            elif row['Rank'] <= 3:
                return ['background: rgba(192, 192, 192, 0.2)'] * len(row)
            else:
                return [''] * len(row)
        
        styled_df = lb_df[['Rank', 'Player', 'Mode', 'Portfolio Value', 'P&L', 'Sharpe Ratio']].style.format({
            "Portfolio Value": lambda x: format_indian_currency(x),
            "P&L": lambda x: format_indian_currency(x),
            "Sharpe Ratio": "{:.2f}"
        }).apply(style_leaderboard, axis=1)
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
    else:
        st.info("No players yet")
    
    st.markdown("</div>", unsafe_allow_html=True)

# --- Admin Dashboard ---
def render_admin_dashboard(prices):
    game_state = get_game_state()
    
    st.title("👑 Admin Control Center")
    
    # Game status overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Game Status", game_state.game_status)
    with col2:
        if game_state.game_status == "Running":
            remaining = max(0, game_state.round_duration_seconds - int(time.time() - game_state.game_start_time))
            st.metric("Time Remaining", f"{remaining//60:02d}:{remaining%60:02d}")
        else:
            st.metric("Time Remaining", "00:00")
    with col3:
        st.metric("Active Players", len(game_state.players))
    with col4:
        st.metric("Difficulty", f"Level {game_state.difficulty_level}")
    
    st.markdown("---")
    
    # Admin controls in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎮 Game Controls", "👥 Player Management", "📰 News Control", "⚙️ Settings"])
    
    with tab1:
        st.subheader("Game Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            game_duration = st.number_input("Game Duration (minutes)", min_value=1, max_value=180, value=20, key="admin_duration")
            if st.button("▶️ Start Game", use_container_width=True, type="primary"):
                if game_state.players:
                    game_state.game_status = "Running"
                    game_state.game_start_time = time.time()
                    game_state.round_duration_seconds = game_duration * 60
                    st.success("Game started!")
                    st.rerun()
                else:
                    st.error("No players in the game!")
        
        with col2:
            if st.button("⏸️ Stop Game", use_container_width=True):
                game_state.game_status = "Stopped"
                st.success("Game stopped!")
                st.rerun()
            
            if st.button("🔄 Reset Game", use_container_width=True):
                game_state.reset()
                st.success("Game reset!")
                st.rerun()
        
        with col3:
            if st.button("🔔 Test Sound", use_container_width=True):
                play_sound('success')
            if st.button("📢 Test News", use_container_width=True):
                test_news = "Test news announcement for system check"
                game_state.news_feed.insert(0, f"📢 {time.strftime('%H:%M:%S')} - {test_news}")
                announce_news(test_news)
                st.success("News test completed!")
    
    with tab2:
        st.subheader("Player Management")
        
        if game_state.players:
            player_to_manage = st.selectbox("Select Player", list(game_state.players.keys()), key="admin_player_select")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Current Stats**")
                player = game_state.players[player_to_manage]
                holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
                total_value = player['capital'] + holdings_value
                
                st.metric("Capital", format_indian_currency(player['capital']))
                st.metric("Portfolio Value", format_indian_currency(total_value))
                st.metric("Holdings", len(player['holdings']))
            
            with col2:
                st.write("**Adjust Capital**")
                adjustment = st.number_input("Amount", value=10000, step=1000, key="admin_adjustment")
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("💸 Add Funds", use_container_width=True):
                        player['capital'] += adjustment
                        st.success(f"Added {format_indian_currency(adjustment)} to {player_to_manage}")
                        st.rerun()
                with col_b:
                    if st.button("⚡ Remove Funds", use_container_width=True):
                        player['capital'] = max(0, player['capital'] - adjustment)
                        st.success(f"Removed {format_indian_currency(adjustment)} from {player_to_manage}")
                        st.rerun()
                
                if st.button("🗑️ Clear Holdings", use_container_width=True):
                    player['holdings'] = {}
                    st.success(f"Cleared all holdings for {player_to_manage}")
                    st.rerun()
        else:
            st.info("No players to manage")
    
    with tab3:
        st.subheader("News Management")
        
        news_options = {news['headline']: news['impact'] for news in PRE_BUILT_NEWS}
        selected_news = st.selectbox("Select News Event", list(news_options.keys()), key="admin_news_select")
        
        target_symbol = None
        if selected_news and "{symbol}" in selected_news:
            target_symbol = st.selectbox("Target Symbol", [s.replace('.NS', '') for s in NIFTY50_SYMBOLS], key="admin_target_symbol") + ".NS"
        
        if selected_news:
            st.info(f"Impact Type: {news_options[selected_news]}")
            
            if st.button("📢 Publish News", type="primary", use_container_width=True):
                headline = selected_news.format(symbol=target_symbol.replace('.NS', '') if target_symbol else "")
                game_state.news_feed.insert(0, f"📢 {time.strftime('%H:%M:%S')} - {headline}")
                if len(game_state.news_feed) > 8:
                    game_state.news_feed.pop()
                
                game_state.event_type = news_options[selected_news]
                game_state.event_target_symbol = target_symbol
                game_state.event_active = True
                game_state.event_end = time.time() + 60
                
                announce_news(headline)
                st.success("News published successfully!")
                st.rerun()
    
    with tab4:
        st.subheader("Game Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            game_state.volatility_multiplier = st.slider("Market Volatility", 0.5, 5.0, game_state.volatility_multiplier, 0.1, key="admin_volatility")
            game_state.current_margin_requirement = st.slider("Margin Requirement %", 10, 50, int(game_state.current_margin_requirement * 100), 5, key="admin_margin") / 100.0
            game_state.difficulty_level = st.selectbox("Difficulty Level", [1, 2, 3], index=game_state.difficulty_level-1, key="admin_difficulty")
        
        with col2:
            game_state.bid_ask_spread = st.slider("Bid-Ask Spread %", 0.05, 1.0, game_state.bid_ask_spread * 100, 0.05, key="admin_spread") / 100.0
            game_state.slippage_threshold = st.slider("Slippage Threshold", 1, 100, game_state.slippage_threshold, key="admin_slippage")
        
        if st.button("💾 Save Settings", use_container_width=True):
            st.success("Game settings updated!")
    
    st.markdown("---")
    
    # Live market data
    st.subheader("📊 Live Market Overview")
    render_professional_leaderboard(prices)

# --- Sidebar ---
def render_sidebar():
    game_state = get_game_state()
    
    with st.sidebar:
        st.markdown("<div class='pro-panel'>", unsafe_allow_html=True)
        st.title("🎮 Player Access")
        
        if 'player' not in st.query_params:
            player_name = st.text_input("Trader Name", placeholder="Enter your name", key="name_input")
            mode = st.radio("Trading Mode", ["Trader", "HFT", "HNI"], key="mode_select")
            
            if st.button("🚀 Join Game", type="primary", use_container_width=True):
                if player_name and player_name.strip():
                    if player_name not in game_state.players:
                        starting_capital = INITIAL_CAPITAL * 5 if mode == "HNI" else INITIAL_CAPITAL
                        game_state.players[player_name] = {
                            "name": player_name, "mode": mode, "capital": starting_capital, 
                            "holdings": {}, "pnl": 0, "leverage": 1.0, "margin_calls": 0, 
                            "pending_orders": [], "algo": "Off", "custom_algos": {},
                            "slippage_multiplier": 0.5 if mode == "HFT" else 1.0,
                            "value_history": [starting_capital], "trade_timestamps": []
                        }
                        game_state.transactions[player_name] = []
                        st.query_params["player"] = player_name
                        st.rerun()
                    else:
                        st.error("Name already taken!")
                else:
                    st.error("Please enter a valid name!")
        else:
            st.success(f"**👤 {st.query_params['player']}**")
            player_data = game_state.players.get(st.query_params['player'], {})
            if player_data:
                st.write(f"**Mode:** {player_data.get('mode', 'Trader')}")
                st.write(f"**Capital:** {format_indian_currency(player_data.get('capital', 0))}")
            
            if st.button("🚪 Logout", use_container_width=True):
                st.query_params.clear()
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Admin section
        st.markdown("<div class='pro-panel'>", unsafe_allow_html=True)
        st.title("🔐 Admin Access")
        
        if st.session_state.get('role') == 'admin':
            st.success("✅ Admin Mode Active")
            
            if st.button("🚪 Exit Admin Mode", use_container_width=True):
                del st.session_state['role']
                st.rerun()
        else:
            password = st.text_input("Admin Password", type="password", placeholder="Enter admin password", key="admin_pass")
            if st.button("🔑 Login as Admin", use_container_width=True):
                if password == ADMIN_PASSWORD:
                    st.session_state.role = 'admin'
                    st.rerun()
                elif password:
                    st.error("❌ Incorrect password")
        
        st.markdown("</div>", unsafe_allow_html=True)

# --- Main Interface ---
def render_enhanced_main_interface(prices):
    inject_pro_css()
    render_professional_header()
    
    game_state = get_game_state()
    
    if st.session_state.get('role') == 'admin':
        render_admin_dashboard(prices)
    elif 'player' in st.query_params:
        player_name = st.query_params.get("player")
        if player_name in game_state.players:
            player = game_state.players[player_name]
            render_pro_trading_interface(player_name, player, prices)
            
            # Bottom panels
            col1, col2 = st.columns([2, 1])
            with col1:
                render_professional_news_feed()
                render_advanced_analytics(player, prices)
            with col2:
                render_professional_leaderboard(prices)
        else:
            st.error("Player not found - please rejoin the game")
            if st.button("Rejoin Game"):
                st.query_params.clear()
                st.rerun()
    else:
        # Welcome screen
        st.markdown("<div class='pro-panel'>", unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.title("🚀 Welcome to BlockVista Pro Trading")
            st.markdown("""
            ### Professional Trading Platform
            
            **Features:**
            - 📊 **Advanced Charting** with technical indicators
            - 📋 **Real-time Order Book** with market depth  
            - 🚀 **Quick Order Execution** with one-click trading
            - 📰 **Live Market News** with impact analysis
            - 📈 **Advanced Analytics** with risk metrics
            - 💼 **Portfolio Management** with real-time P&L
            
            *Join the game from the sidebar to start trading!*
            """)
            
            # Quick stats
            if game_state.players:
                st.info(f"**{len(game_state.players)}** traders currently active")
            if game_state.game_status == "Running":
                st.success("🎯 Trading session is LIVE")
            else:
                st.warning("⏸️ Trading session paused")
                
        with col2:
            st.markdown("""
            <div style='text-align: center;'>
                <h3>🎮 Get Started</h3>
                <p>1. Enter your name</p>
                <p>2. Choose trading mode</p>
                <p>3. Click 'Join Game'</p>
                <br>
                <p><strong>Or login as Admin to manage the game</strong></p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Show leaderboard on welcome page
        render_professional_leaderboard(prices)

# --- Game Logic ---
def run_game_tick(prices):
    game_state = get_game_state()
    if game_state.game_status != "Running": return prices
    
    # Sentiment decay
    for symbol in game_state.market_sentiment:
        game_state.market_sentiment[symbol] *= 0.95 

    # Random news events
    if not game_state.event_active and random.random() < 0.05:
        news_item = random.choice(PRE_BUILT_NEWS)
        headline = news_item['headline']
        impact = news_item['impact']
        target_symbol = None

        if "{symbol}" in headline:
            target_symbol = random.choice(NIFTY50_SYMBOLS)
            headline = headline.format(symbol=target_symbol.replace(".NS", ""))
        
        game_state.news_feed.insert(0, f"📢 {time.strftime('%H:%M:%S')} - {headline}")
        if len(game_state.news_feed) > 8: game_state.news_feed.pop()
        
        game_state.event_type = impact
        game_state.event_target_symbol = target_symbol
        game_state.event_active = True
        game_state.event_end = time.time() + random.randint(30, 60)
        announce_news(headline)
        
    if game_state.event_active and time.time() >= game_state.event_end:
        game_state.event_active = False
        
    # Update order books
    for symbol in game_state.order_books:
        if symbol in prices:
            game_state.order_books[symbol].update(prices[symbol])
    
    # Record portfolio values
    for player in game_state.players.values():
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        player['value_history'].append(total_value)
        if len(player['value_history']) > 100:
            player['value_history'].pop(0)
        
    return prices

def main():
    game_state = get_game_state()
    
    if 'role' not in st.session_state:
        st.session_state.role = 'player'
    
    render_sidebar()
    
    # Price simulation
    if not game_state.base_real_prices:
        game_state.base_real_prices = get_daily_base_prices()
    
    last_prices = game_state.prices if game_state.prices else game_state.base_real_prices
    current_prices = simulate_tick_prices(last_prices)
    prices_with_derivatives = calculate_derived_prices(current_prices)
    final_prices = run_game_tick(prices_with_derivatives)
    
    game_state.prices = final_prices
    
    if not isinstance(game_state.price_history, list): 
        game_state.price_history = []
    game_state.price_history.append(final_prices)
    if len(game_state.price_history) > 10: 
        game_state.price_history.pop(0)
    
    # Render main interface
    render_enhanced_main_interface(final_prices)
    
    # Auto-refresh
    if game_state.game_status == "Running": 
        time.sleep(1)
        st.rerun()
    else:
        time.sleep(3)
        st.rerun()

if __name__ == "__main__":
    main()

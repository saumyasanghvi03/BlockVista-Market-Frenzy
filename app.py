# ======================= Expo Game: BlockVista Market Frenzy ======================

import streamlit as st
import pandas as pd
import yfinance as yf
import time
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="BlockVista Market Frenzy", page_icon="📈")

# --- API & Game Configuration ---
GAME_NAME = "BlockVista Market Frenzy"
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
SLIPPAGE_THRESHOLD = 10
BASE_SLIPPAGE_RATE = 0.005
MARGIN_REQUIREMENT = 0.2
ADMIN_PASSWORD = "100370" # Set your admin password here

# --- Game State Management (Singleton for Live Sync) ---
class GameState:
    """A singleton class to hold the shared game state across all user sessions."""
    def __init__(self):
        self.players = {}
        self.game_status = "Stopped"
        self.game_start_time = 0
        self.round_duration_seconds = 20 * 60 # Default duration
        self.futures_expiry_time = 0
        self.futures_settled = False
        self.prices = {}
        self.price_history = []
        self.transactions = {}
        self.market_sentiment = {s: 0 for s in ALL_SYMBOLS}
        self.liquidity = {s: random.uniform(0.5, 1.0) for s in ALL_SYMBOLS}
        self.event_active = False
        self.event_type = None
        self.event_end = 0
        self.volatility_multiplier = 1.0

    def reset(self):
        """Resets the game to its initial state."""
        self.__init__() # Re-initialize all attributes

@st.cache_resource
def get_game_state():
    """Returns the singleton GameState object, ensuring all users share the same state."""
    return GameState()


# --- Quiz Questions ---
QUIZ_QUESTIONS = [
    {"question": "What does RSI stand for in technical analysis?", "options": ["Relative Strength Index", "Rapid Stock Increase", "Risk Sensitivity Indicator"], "answer": 0},
    {"question": "Which candlestick pattern signals a bullish reversal?", "options": ["Doji", "Hammer", "Shooting Star"], "answer": 1},
    {"question": "What is the primary role of SEBI in India?", "options": ["Regulate securities market", "Control inflation", "Manage foreign exchange"], "answer": 0},
]

# --- Data Fetching & Market Simulation ---
@st.cache_data(ttl=30) # Fetch base prices more frequently
def get_base_live_prices():
    """Fetches the latest real-world price from yfinance to use as a baseline."""
    prices = {}
    yf_symbols = NIFTY50_SYMBOLS + CRYPTO_SYMBOLS + [GOLD_SYMBOL, NIFTY_INDEX_SYMBOL, BANKNIFTY_INDEX_SYMBOL]
    try:
        data = yf.download(tickers=yf_symbols, period="1d", interval="1m", progress=False)
        for symbol in yf_symbols:
            if not data.empty and symbol in data['Close'] and not pd.isna(data['Close'][symbol].iloc[-1]):
                prices[symbol] = data['Close'][symbol].iloc[-1]
            else: # Fallback
                prices[symbol] = random.uniform(10, 50000)
    except Exception:
        for symbol in yf_symbols: # Full fallback on API error
            prices[symbol] = random.uniform(10, 50000)
    return prices

def simulate_market_prices(base_prices):
    """Adjusts base prices based on various game mechanics."""
    game_state = get_game_state()
    simulated_prices = base_prices.copy()
    
    # Safely get volatility_multiplier, defaulting to 1.0 if it doesn't exist
    volatility_multiplier = getattr(game_state, 'volatility_multiplier', 1.0)
    
    # Apply sentiment and volatility
    for symbol in simulated_prices:
        sentiment = game_state.market_sentiment.get(symbol, 0)
        volatility_noise = random.uniform(-0.001, 0.001) * volatility_multiplier
        price_multiplier = 1 + (sentiment * 0.005) + volatility_noise
        simulated_prices[symbol] *= price_multiplier
    
    # Calculate derived asset prices
    nifty_price = simulated_prices.get(NIFTY_INDEX_SYMBOL, 20000)
    banknifty_price = simulated_prices.get(BANKNIFTY_INDEX_SYMBOL, 45000)
    
    # Mock options
    simulated_prices['NIFTY_CALL'] = nifty_price * 1.02
    simulated_prices['NIFTY_PUT'] = nifty_price * 0.98
    
    # Futures with a random basis
    simulated_prices['NIFTY-FUT'] = nifty_price * random.uniform(1.0, 1.005)
    simulated_prices['BANKNIFTY-FUT'] = banknifty_price * random.uniform(1.0, 1.005)
    
    # Leveraged ETFs
    if len(game_state.price_history) >= 2:
        prev_nifty = game_state.price_history[-2].get(NIFTY_INDEX_SYMBOL, nifty_price)
        nifty_change = (nifty_price - prev_nifty) / prev_nifty
        
        # Get current ETF prices to apply change
        current_bull = game_state.prices.get('NIFTY_BULL_3X', nifty_price/100)
        current_bear = game_state.prices.get('NIFTY_BEAR_3X', nifty_price/100)

        simulated_prices['NIFTY_BULL_3X'] = current_bull * (1 + 3 * nifty_change)
        simulated_prices['NIFTY_BEAR_3X'] = current_bear * (1 - 3 * nifty_change)
    else:
        simulated_prices['NIFTY_BULL_3X'] = nifty_price / 100
        simulated_prices['NIFTY_BEAR_3X'] = nifty_price / 100

    return simulated_prices

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

# --- Game Logic Functions ---
def calculate_slippage(player, symbol, qty, action):
    game_state = get_game_state()
    liquidity_level = game_state.liquidity.get(symbol, 1.0)
    if qty <= SLIPPAGE_THRESHOLD: return 1.0
    
    slippage_multiplier = player.get('slippage_multiplier', 1.0)
    
    excess_qty = qty - SLIPPAGE_THRESHOLD
    slippage_rate = (BASE_SLIPPAGE_RATE / max(0.1, liquidity_level)) * slippage_multiplier
    slippage_mult = 1 + (slippage_rate * excess_qty) * (1 if action == "Buy" else -1)
    return max(0.9, min(1.1, slippage_mult))

def apply_event_adjustment(prices, event_type):
    adjusted_prices = prices.copy()
    if event_type == "Flash Crash":
        adjusted_prices = {k: v * random.uniform(0.88, 0.92) for k, v in adjusted_prices.items()}
    elif event_type == "Bull Rally":
        adjusted_prices = {k: v * random.uniform(1.08, 1.12) for k, v in adjusted_prices.items()}
    elif event_type == "Banking Boost":
        for sym in ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'AXISBANK.NS']:
            if sym in adjusted_prices: adjusted_prices[sym] *= 1.07
    elif event_type == "Sector Rotation":
        for sym in ['HDFCBANK.NS', 'ICICIBANK.NS']: # Out of Banking
            if sym in adjusted_prices: adjusted_prices[sym] *= 0.95
        for sym in ['INFY.NS', 'TCS.NS']: # Into Tech
            if sym in adjusted_prices: adjusted_prices[sym] *= 1.10
    return adjusted_prices

def format_indian_currency(n):
    if n is None: return "₹0.00"
    n = float(n)
    if abs(n) < 100000: return f"₹{n:,.2f}"
    elif abs(n) < 10000000: return f"₹{n/100000:.2f}L"
    else: return f"₹{n/10000000:.2f}Cr"

def optimize_portfolio(player_holdings):
    symbols = [s for s in player_holdings.keys() if s in NIFTY50_SYMBOLS + CRYPTO_SYMBOLS]
    if len(symbols) < 2: return None, "Need at least 2 assets to optimize."
    try:
        hist_data = get_historical_data(symbols)
        if hist_data.empty or hist_data.isnull().values.any(): return None, "Could not fetch sufficient historical data."
        mu = expected_returns.mean_historical_return(hist_data)
        S = risk_models.sample_cov(hist_data)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        return ef.clean_weights(), ef.portfolio_performance(verbose=False)
    except Exception as e: return None, f"Optimization failed: {e}"

def calculate_indicator(indicator, symbol):
    hist = get_historical_data([symbol], period="2mo")
    if hist.empty or len(hist) < 30: return None
    if indicator == "Price Change % (5-day)":
        if len(hist) < 6: return None
        return ((hist[symbol].iloc[-1] - hist[symbol].iloc[-6]) / hist[symbol].iloc[-6]) * 100
    elif indicator == "SMA Crossover (10/20)":
        if len(hist) < 20: return None
        sma_10 = hist[symbol].rolling(window=10).mean().iloc[-1]
        sma_20 = hist[symbol].rolling(window=20).mean().iloc[-1]
        return sma_10 - sma_20
    elif indicator == "Price Change % (30-day)":
        if len(hist) < 31: return None
        return ((hist[symbol].iloc[-1] - hist[symbol].iloc[-31]) / hist[symbol].iloc[-31]) * 100
    return None

# --- UI Functions ---
def render_sidebar():
    game_state = get_game_state()
    st.sidebar.title("📝 Game Entry")
    player_name = st.sidebar.text_input("Enter Name", key="name_input")
    mode = st.sidebar.radio("Select Mode", ["Trader", "HFT", "HNI"], key="mode_select")
    
    if st.sidebar.button("Join Game", disabled=(game_state.game_status == "Running")):
        if player_name and player_name.strip() and player_name not in game_state.players:
            starting_capital = INITIAL_CAPITAL * 5 if mode == "HNI" else INITIAL_CAPITAL
            game_state.players[player_name] = {
                "mode": mode, 
                "capital": starting_capital, 
                "holdings": {}, "pnl": 0, "leverage": 1.0, 
                "margin_calls": 0, "pending_orders": [], "algo": "Off", "custom_algos": {},
                "slippage_multiplier": 0.5 if mode == "HFT" else 1.0
            }
            game_state.transactions[player_name] = []
            st.sidebar.success(f"{player_name} joined as {mode}!")
            st.rerun()
        else: st.sidebar.error("Name is invalid or already taken!")
    
    st.sidebar.title("🔐 Admin Login")
    password = st.sidebar.text_input("Enter Password", type="password")

    if password == ADMIN_PASSWORD:
        st.sidebar.success("Admin Access Granted")
        st.sidebar.title("⚙️ Admin Controls")
        
        # Game Duration - Safely access attribute
        default_duration_minutes = int(getattr(game_state, 'round_duration_seconds', 1200) / 60)
        game_duration_minutes = st.sidebar.number_input("Game Duration (minutes)", min_value=1, value=default_duration_minutes, disabled=(game_state.game_status == "Running"))

        # Volatility Control
        game_state.volatility_multiplier = st.sidebar.slider("Market Volatility", 0.5, 5.0, getattr(game_state, 'volatility_multiplier', 1.0), 0.5)

        # Manual Event Trigger
        event_to_trigger = st.sidebar.selectbox("Trigger Market Event", ["None", "Flash Crash", "Bull Rally", "Banking Boost", "Sector Rotation"])
        if event_to_trigger != "None":
            game_state.event_type = event_to_trigger
            game_state.event_active = True
            game_state.event_end = time.time() + 60
            st.toast(f"Admin triggered: {event_to_trigger}!", icon="⚡")
            st.rerun()
            
        # Player Cash Adjustment
        st.sidebar.markdown("---")
        st.sidebar.subheader("Adjust Player Capital")
        if game_state.players:
            player_to_adjust = st.sidebar.selectbox("Select Player", list(game_state.players.keys()))
            amount = st.sidebar.number_input("Amount", value=10000, step=1000)
            c1, c2 = st.sidebar.columns(2)
            if c1.button("Give Bonus"):
                if player_to_adjust in game_state.players:
                    game_state.players[player_to_adjust]['capital'] += amount
                    st.toast(f"Gave {format_indian_currency(amount)} bonus to {player_to_adjust}", icon="💰")
            if c2.button("Apply Penalty"):
                if player_to_adjust in game_state.players:
                    game_state.players[player_to_adjust]['capital'] -= amount
                    st.toast(f"Applied {format_indian_currency(amount)} penalty to {player_to_adjust}", icon="💸")
        else:
            st.sidebar.info("No players to adjust.")

        st.sidebar.markdown("---")
        # Game Controls
        if st.sidebar.button("▶️ Start Game", type="primary"):
            if game_state.players:
                game_state.game_status = "Running"; game_state.game_start_time = time.time()
                game_state.round_duration_seconds = game_duration_minutes * 60
                game_state.futures_expiry_time = time.time() + (game_state.round_duration_seconds / 2) # Expiry halfway through
                st.toast("Game Started!", icon="🎉"); st.rerun()
            else: st.sidebar.warning("Add at least one player to start.")
        if st.sidebar.button("⏸️ Stop Game"):
            game_state.game_status = "Stopped"; st.toast("Game Paused!", icon="⏸️"); st.rerun()
        if st.sidebar.button("🔄 Reset Game"):
            game_state.reset(); st.toast("Game has been reset.", icon="🔄"); st.rerun()
            
    elif password: st.sidebar.error("Incorrect Password")

def render_main_interface(prices):
    game_state = get_game_state()
    st.title(f"📈 {GAME_NAME}")
    
    if game_state.game_status == "Running":
        remaining_time = max(0, game_state.round_duration_seconds - int(time.time() - game_state.game_start_time))
        if remaining_time == 0: game_state.game_status = "Finished"
        st.markdown(f"**Time Remaining: {remaining_time // 60:02d}:{remaining_time % 60:02d}**")
    elif game_state.game_status == "Stopped": st.info("Game is paused. Press 'Start Game' to begin.")
    elif game_state.game_status == "Finished": st.success("Game has finished! See the final leaderboard below.")

    col1, col2 = st.columns([1, 1]); 
    with col1: render_trade_execution_panel(prices)
    with col2: render_global_views(prices)

def render_global_views(prices):
    st.subheader("Live Player Standings")
    render_leaderboard(prices)
    st.subheader("Live Market Feed")
    render_live_market_table(prices)

def render_trade_execution_panel(prices):
    game_state = get_game_state()
    st.subheader("Trade Execution Panel")
    player_list = list(game_state.players.keys())
    if not player_list: st.warning("No players have joined the game yet."); return
    acting_player = st.selectbox("Select Your Player to Trade", player_list)
    
    if acting_player and acting_player in game_state.players:
        player = game_state.players[acting_player]
        st.markdown(f"**{acting_player}'s Terminal (Mode: {player['mode']})**")
        
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        pnl = total_value - (INITIAL_CAPITAL * 5 if player['mode'] == 'HNI' else INITIAL_CAPITAL)
        player['pnl'] = pnl
        pnl_arrow = "🔼" if pnl >= 0 else "🔽"

        c1, c2, c3 = st.columns(3)
        c1.metric("Cash", format_indian_currency(player['capital']))
        c2.metric("Portfolio Value", format_indian_currency(total_value))
        c3.metric("P&L", format_indian_currency(pnl), f"{pnl_arrow}")

        tabs = ["👨‍💻 Trade Terminal", "🤖 Algo Trading", "📂 Transaction History", "📊 Strategy & Insights"]
        tab1, tab2, tab3, tab4 = st.tabs(tabs)
        is_game_running = game_state.game_status == "Running"

        with tab1: render_trade_interface(acting_player, player, prices, is_game_running)
        with tab2: render_algo_trading_tab(acting_player, player, is_game_running)
        with tab3: render_transaction_history(acting_player)
        with tab4: render_strategy_tab(player)

def render_trade_interface(player_name, player, prices, disabled_status):
    with st.container(border=True):
        asset_types = ["Stock", "Crypto", "Gold", "Futures", "Leveraged ETF", "Option"]
        asset_type = st.radio("Asset Type", asset_types, horizontal=True, key=f"asset_{player_name}", disabled=disabled_status)

        if asset_type == "Futures" and getattr(get_game_state(), 'futures_expiry_time', 0) > 0:
            expiry_remaining = max(0, get_game_state().futures_expiry_time - time.time())
            st.warning(f"Futures contracts expire and will be cash-settled in **{int(expiry_remaining // 60)}m {int(expiry_remaining % 60)}s**")

        col1, col2 = st.columns([2, 1])
        with col1:
            if asset_type == "Stock": symbol_choice = st.selectbox("Stock", [s.replace('.NS', '') for s in NIFTY50_SYMBOLS], key=f"stock_{player_name}", disabled=disabled_status) + '.NS'
            elif asset_type == "Crypto": symbol_choice = st.selectbox("Cryptocurrency", CRYPTO_SYMBOLS, key=f"crypto_{player_name}", disabled=disabled_status)
            elif asset_type == "Gold": symbol_choice = GOLD_SYMBOL
            elif asset_type == "Futures": symbol_choice = st.selectbox("Futures", FUTURES_SYMBOLS, key=f"futures_{player_name}", disabled=disabled_status)
            elif asset_type == "Leveraged ETF": symbol_choice = st.selectbox("Leveraged ETF", LEVERAGED_ETFS, key=f"letf_{player_name}", disabled=disabled_status)
            else: symbol_choice = st.selectbox("Option", OPTION_SYMBOLS, key=f"option_{player_name}", disabled=disabled_status)
        
        with col2: qty = st.number_input("Quantity", min_value=1, step=1, value=1, key=f"qty_{player_name}", disabled=disabled_status)

        current_price = prices.get(symbol_choice, 0)
        estimated_cost = current_price * qty
        st.info(f"Current Price: {format_indian_currency(current_price)} | Estimated Cost: {format_indian_currency(estimated_cost)}")
        
        b1, b2, b3 = st.columns(3)
        if b1.button(f"Buy {qty} {symbol_choice.split('-')[0].split('.')[0]}", key=f"buy_{player_name}", use_container_width=True, disabled=disabled_status, type="primary"): execute_trade(player_name, player, "Buy", symbol_choice, qty, prices); st.rerun()
        if b2.button(f"Sell {qty} {symbol_choice.split('-')[0].split('.')[0]}", key=f"sell_{player_name}", use_container_width=True, disabled=disabled_status): execute_trade(player_name, player, "Sell", symbol_choice, qty, prices); st.rerun()
        if b3.button(f"Short {qty} {symbol_choice.split('-')[0].split('.')[0]}", key=f"short_{player_name}", use_container_width=True, disabled=disabled_status): execute_trade(player_name, player, "Short", symbol_choice, qty, prices); st.rerun()
            
    st.markdown("---"); render_current_holdings(player, prices)

def render_algo_trading_tab(player_name, player, disabled_status):
    st.subheader("Automated Trading Strategies")
    default_strats = ["Off", "Momentum Trader", "Mean Reversion", "Volatility Breakout", "Value Investor"]
    custom_strats = list(player.get('custom_algos', {}).keys()); all_strats = default_strats + custom_strats
    active_algo = player.get('algo', 'Off')
    player['algo'] = st.selectbox("Choose Strategy", all_strats, index=all_strats.index(active_algo) if active_algo in all_strats else 0, disabled=disabled_status, key=f"algo_{player_name}")
    
    # Descriptions for default strats
    if player['algo'] in default_strats and player['algo'] != 'Off': st.info("Default strategy description...")

    with st.expander("Create Custom Strategy"):
        st.markdown("##### Define Your Own Algorithm")
        c1, c2 = st.columns(2)
        with c1:
            algo_name = st.text_input("Strategy Name", key=f"algo_name_{player_name}")
            indicator = st.selectbox("Indicator", ["Price Change % (5-day)", "SMA Crossover (10/20)", "Price Change % (30-day)"], key=f"indicator_{player_name}")
            condition = st.selectbox("Condition", ["Greater Than", "Less Than"], key=f"condition_{player_name}")
        with c2:
            threshold = st.number_input("Threshold Value", value=0.0, step=0.1, key=f"threshold_{player_name}")
            action = st.radio("Action if True", ["Buy", "Sell"], key=f"algo_action_{player_name}")
        if st.button("Save Strategy", key=f"save_algo_{player_name}"):
            if algo_name.strip():
                player['custom_algos'][algo_name] = {"indicator": indicator, "condition": condition, "threshold": threshold, "action": action}
                st.success(f"Custom strategy '{algo_name}' saved!"); st.rerun()
            else: st.error("Strategy name cannot be empty.")

def render_current_holdings(player, prices):
    st.subheader("💼 Current Holdings")
    if player['holdings']:
        holdings_data = [{"Symbol": sym, "Quantity": qty, "Value": prices.get(sym, 0) * qty} for sym, qty in player['holdings'].items()]
        st.dataframe(pd.DataFrame(holdings_data).style.format(formatter={"Value": format_indian_currency}), use_container_width=True)
    else: st.info("No holdings yet.")
        
def render_transaction_history(player_name):
    game_state = get_game_state()
    st.subheader("Transaction History")
    if game_state.transactions.get(player_name):
        trans_df = pd.DataFrame(game_state.transactions[player_name], columns=["Time", "Action", "Symbol", "Qty", "Price", "Total"])
        st.dataframe(trans_df.style.format(formatter={"Price": format_indian_currency, "Total": format_indian_currency}), use_container_width=True)
    else: st.info("No transactions recorded.")

def render_strategy_tab(player):
    st.subheader("📊 Strategy & Insights")
    tab1, tab2 = st.tabs(["Technical Analysis (SMA)", "Portfolio Optimizer"])
    with tab1: render_sma_chart(player['holdings'])
    with tab2: render_optimizer(player['holdings'])

def render_sma_chart(holdings):
    st.markdown("##### Simple Moving Average (SMA) Chart")
    chartable_assets = [s for s in holdings.keys() if s not in OPTION_SYMBOLS + FUTURES_SYMBOLS + LEVERAGED_ETFS]
    if not chartable_assets: st.info("No chartable assets in portfolio to analyze."); return
    chart_symbol = st.selectbox("Select Asset to Chart", chartable_assets)
    hist_data = get_historical_data([chart_symbol], period="6mo")
    if not hist_data.empty:
        df = hist_data.rename(columns={chart_symbol: 'Close'})
        df['SMA_20'] = df['Close'].rolling(window=20).mean(); df['SMA_50'] = df['Close'].rolling(window=50).mean()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='20-Day SMA'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='50-Day SMA'))
        st.plotly_chart(fig, use_container_width=True)
    else: st.warning(f"Could not load historical data for {chart_symbol}.")

def render_optimizer(holdings):
    st.subheader("Portfolio Optimization (Max Sharpe Ratio)")
    if st.button("Optimize My Portfolio"):
        weights, performance = optimize_portfolio(holdings)
        if weights:
            st.success("Optimal weights for max risk-adjusted return:"); st.json({k: f"{v:.2%}" for k, v in weights.items()})
            if performance: st.write(f"Expected Return: {performance[0]:.2%}, Volatility: {performance[1]:.2%}, Sharpe Ratio: {performance[2]:.2f}")
        else: st.error(performance)

def execute_trade(player_name, player, action, symbol, qty, prices, is_algo=False):
    game_state = get_game_state()
    base_price = prices.get(symbol, 0)
    if base_price == 0: return False
    
    game_state.market_sentiment[symbol] = game_state.market_sentiment.get(symbol, 0) + (qty / 50) * (1 if action in ["Buy", "Short"] else -1)
    trade_price = base_price * calculate_slippage(player, symbol, qty, action); cost = trade_price * qty
    
    trade_executed = False
    if action == "Buy" and player['capital'] >= cost:
        player['capital'] -= cost; player['holdings'][symbol] = player['holdings'].get(symbol, 0) + qty; trade_executed = True
    elif action == "Short" and player['capital'] >= cost * MARGIN_REQUIREMENT:
        player['capital'] += cost; player['holdings'][symbol] = player['holdings'].get(symbol, 0) - qty; trade_executed = True
    elif action == "Sell":
        current_qty = player['holdings'].get(symbol, 0)
        if current_qty > 0 and current_qty >= qty: # Closing a long
            player['capital'] += cost; player['holdings'][symbol] -= qty; trade_executed = True
        elif current_qty < 0 and abs(current_qty) >= qty: # Covering a short
            player['capital'] -= cost; player['holdings'][symbol] += qty; trade_executed = True
        if trade_executed and player['holdings'][symbol] == 0: del player['holdings'][symbol]

    if trade_executed: log_transaction(player_name, action, symbol, qty, trade_price, cost, is_algo)
    elif not is_algo: st.error("Trade failed: Insufficient capital or holdings.")
    return trade_executed

def log_transaction(player_name, action, symbol, qty, price, total, is_algo=False):
    game_state = get_game_state()
    prefix = "🤖 Algo" if is_algo else ""
    game_state.transactions.setdefault(player_name, []).append([time.strftime("%H:%M:%S"), f"{prefix} {action}".strip(), symbol, qty, price, total])
    if not is_algo: st.success(f"Trade Executed: {action} {qty} {symbol} @ {format_indian_currency(price)}")
    else: st.toast(f"Algo Trade: {action} {qty} {symbol}", icon="🤖")

def render_leaderboard(prices):
    game_state = get_game_state()
    lb = []
    for pname, pdata in game_state.players.items():
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in pdata['holdings'].items())
        total_value = pdata['capital'] + holdings_value
        lb.append((pname, pdata['mode'], total_value, pdata['pnl']))
    
    if lb:
        lb_df = pd.DataFrame(lb, columns=["Player", "Mode", "Portfolio Value", "P&L"]).sort_values("Portfolio Value", ascending=False).reset_index(drop=True)
        st.dataframe(lb_df.style.format(formatter={"Portfolio Value": format_indian_currency, "P&L": format_indian_currency}), use_container_width=True)
        
        if game_state.game_status == "Finished":
            st.balloons(); winner = lb_df.iloc[0]
            st.success(f"🎉 The winner is {winner['Player']}! 🎉")
            c1, c2 = st.columns(2)
            c1.metric("🏆 Final Portfolio Value", format_indian_currency(winner['Portfolio Value']))
            c2.metric("💰 Total P&L", format_indian_currency(winner['P&L']))
            st.subheader("Final Top 3 Standings:"); st.table(lb_df.head(3))

def render_live_market_table(prices):
    game_state = get_game_state(); prices_df = pd.DataFrame(prices.items(), columns=['Symbol', 'Price'])
    if len(game_state.price_history) >= 2:
        prev_prices = game_state.price_history[-2]
        prices_df['prev_price'] = prices_df['Symbol'].map(prev_prices).fillna(prices_df['Price'])
        prices_df['Change'] = prices_df['Price'] - prices_df['prev_price']
    else: prices_df['Change'] = 0.0
    prices_df.drop(columns=['prev_price'], inplace=True, errors='ignore')

    all_trades = [[player] + t for player, transactions in game_state.transactions.items() for t in transactions]
    if all_trades:
        feed_df = pd.DataFrame(all_trades, columns=["Player", "Time", "Action", "Symbol", "Qty", "Trade Price", "Total"])
        last_trades = feed_df.sort_values('Time').groupby('Symbol').last()
        last_trades['Last Order'] = last_trades.apply(lambda r: f"{r['Player']} {r['Action']} {r['Qty']} @ {format_indian_currency(r['Trade Price'])}", axis=1)
        prices_df = pd.merge(prices_df, last_trades[['Last Order']], on='Symbol', how='left')
    else: prices_df['Last Order'] = '-'
    prices_df.fillna({'Last Order': '-'}, inplace=True)
    
    st.dataframe(prices_df.style.apply(lambda x: ['color: green' if v > 0 else 'color: red' if v < 0 else '' for v in x], subset=['Change']).format({'Price': format_indian_currency, 'Change': lambda v: f"{format_indian_currency(v) if v != 0 else '-'}"}), use_container_width=True, hide_index=True)

# --- Main Game Loop Functions ---
def run_game_tick(prices):
    game_state = get_game_state()
    if game_state.game_status != "Running": return prices
    for symbol in game_state.market_sentiment: game_state.market_sentiment[symbol] *= 0.95
    if not game_state.event_active and random.random() < 0.01:
        events = ["Flash Crash", "Bull Rally", "Banking Boost", "Sector Rotation"]
        game_state.event_type = random.choice(events); game_state.event_active = True
        game_state.event_end = time.time() + random.randint(30, 60); st.toast(f"⚡ Market Event: {game_state.event_type}!", icon="🎉")
    if game_state.event_active and time.time() >= game_state.event_end:
        game_state.event_active = False; st.info("Market event has ended.")
    if game_state.event_active: prices = apply_event_adjustment(prices, game_state.event_type)
    handle_futures_expiry(prices)
    run_algo_strategies(prices)
    return prices

def handle_futures_expiry(prices):
    game_state = get_game_state()
    if not game_state.futures_settled and getattr(game_state, 'futures_expiry_time', 0) > 0 and time.time() > game_state.futures_expiry_time:
        st.warning("FUTURES EXPIRED! All open futures positions are being cash-settled.")
        settlement_prices = { 'NIFTY-FUT': prices.get(NIFTY_INDEX_SYMBOL), 'BANKNIFTY-FUT': prices.get(BANKNIFTY_INDEX_SYMBOL) }
        for name, player in game_state.players.items():
            for symbol in FUTURES_SYMBOLS:
                if symbol in player['holdings']:
                    qty = player['holdings'][symbol]
                    settlement_price = settlement_prices.get(symbol, 0)
                    pnl = (settlement_price - prices.get(symbol, 0)) * qty # Simplified PnL
                    player['capital'] += pnl
                    log_transaction(name, "Futures Settlement", symbol, qty, settlement_price, pnl)
                    del player['holdings'][symbol]
        game_state.futures_settled = True

def run_algo_strategies(prices):
    game_state = get_game_state()
    if len(game_state.price_history) < 2: return
    prev_prices = game_state.price_history[-2]
    
    for name, player in game_state.players.items():
        active_algo = player.get('algo', 'Off')
        if active_algo == 'Off': continue
        trade_symbol = random.choice(NIFTY50_SYMBOLS + CRYPTO_SYMBOLS)
        if active_algo in player.get('custom_algos', {}):
            strategy = player['custom_algos'][active_algo]
            indicator_val = calculate_indicator(strategy['indicator'], trade_symbol)
            if indicator_val is None: continue
            condition_met = (strategy['condition'] == 'Greater Than' and indicator_val > strategy['threshold']) or \
                            (strategy['condition'] == 'Less Than' and indicator_val < strategy['threshold'])
            if condition_met: execute_trade(name, player, strategy['action'], trade_symbol, 1, prices, is_algo=True)
        else: # Default Algos
            price_change = prices[trade_symbol] - prev_prices.get(trade_symbol, prices[trade_symbol])
            if active_algo == "Momentum Trader" and abs(price_change / prices[trade_symbol]) > 0.001:
                execute_trade(name, player, "Buy" if price_change > 0 else "Sell", trade_symbol, 1, prices, is_algo=True)
            elif active_algo == "Mean Reversion" and abs(price_change / prices[trade_symbol]) > 0.001:
                execute_trade(name, player, "Sell" if price_change > 0 else "Buy", trade_symbol, 1, prices, is_algo=True)
            elif active_algo == "Volatility Breakout" and abs((price_change) / prices[trade_symbol]) * 100 > 0.1:
                execute_trade(name, player, "Buy", trade_symbol, 1, prices, is_algo=True)
            elif active_algo == "Value Investor":
                change_30_day = calculate_indicator("Price Change % (30-day)", trade_symbol)
                if change_30_day is not None and change_30_day < -10: execute_trade(name, player, "Buy", trade_symbol, 1, prices, is_algo=True)

def main():
    game_state = get_game_state(); render_sidebar()
    base_prices = get_base_live_prices()
    prices = simulate_market_prices(base_prices)
    prices = run_game_tick(prices)
    game_state.prices = prices
    if not isinstance(game_state.price_history, list): game_state.price_history = []
    game_state.price_history.append(prices)
    if len(game_state.price_history) > 10: game_state.price_history.pop(0)
    render_main_interface(prices)
    if game_state.game_status == "Running": time.sleep(2); st.rerun()

if __name__ == "__main__":
    main()


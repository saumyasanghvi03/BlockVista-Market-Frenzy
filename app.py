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
INITIAL_CAPITAL = 1000000  # ₹10 lakh
ROUND_DURATION = 20 * 60   # 20 minutes in seconds

# Define asset symbols
NIFTY50_SYMBOLS = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS',
                   'KOTAKBANK.NS', 'SBIN.NS', 'ITC.NS', 'ASIANPAINT.NS', 'AXISBANK.NS']
CRYPTO_SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD']
GOLD_SYMBOL = 'GC=F'
NIFTY_INDEX_SYMBOL = '^NSEI' # yfinance symbol for Nifty 50 Index
OPTION_SYMBOLS = ['NIFTY_CALL', 'NIFTY_PUT']

ALL_SYMBOLS = NIFTY50_SYMBOLS + CRYPTO_SYMBOLS + [GOLD_SYMBOL] + OPTION_SYMBOLS

# Game Mechanics Settings
SLIPPAGE_THRESHOLD = 10
BASE_SLIPPAGE_RATE = 0.005
MARGIN_REQUIREMENT = 0.2

# --- Quiz Questions ---
QUIZ_QUESTIONS = [
    {"question": "What does RSI stand for in technical analysis?", "options": ["Relative Strength Index", "Rapid Stock Increase", "Risk Sensitivity Indicator"], "answer": 0},
    {"question": "Which candlestick pattern signals a bullish reversal?", "options": ["Doji", "Hammer", "Shooting Star"], "answer": 1},
    {"question": "What is the primary role of SEBI in India?", "options": ["Regulate securities market", "Control inflation", "Manage foreign exchange"], "answer": 0},
    {"question": "In crypto, what is a 'halving' event for Bitcoin?", "options": ["A 50% market crash", "Splitting the blockchain", "Reducing block rewards by half"], "answer": 2},
    {"question": "What is short-selling?", "options": ["Buying low, selling high", "Selling borrowed shares", "Holding for dividends"], "answer": 1},
    {"question": "What does 'DeFi' stand for?", "options": ["Decentralized Finance", "Defined Finance", "Default Finality"], "answer": 0}
]

# --- Data Fetching & Market Simulation ---
@st.cache_data(ttl=30) # Fetch base prices more frequently
def get_base_live_prices():
    """Fetches the latest real-world price from yfinance to use as a baseline."""
    prices = {}
    yf_symbols = NIFTY50_SYMBOLS + CRYPTO_SYMBOLS + [GOLD_SYMBOL, NIFTY_INDEX_SYMBOL]
    try:
        data = yf.download(tickers=yf_symbols, period="1d", interval="1m", progress=False)
        for symbol in yf_symbols:
            if not data.empty and symbol in data['Close'] and not pd.isna(data['Close'][symbol].iloc[-1]):
                prices[symbol] = data['Close'][symbol].iloc[-1]
            else: # Fallback
                prices[symbol] = random.uniform(10, 5000)
    except Exception:
        for symbol in yf_symbols: # Full fallback on API error
            prices[symbol] = random.uniform(10, 5000)
    return prices

def simulate_market_prices(base_prices):
    """Adjusts base prices based on simulated demand/supply (market sentiment)."""
    simulated_prices = base_prices.copy()
    for symbol in simulated_prices:
        sentiment = st.session_state.market_sentiment.get(symbol, 0)
        # Price moves based on sentiment, with some random noise
        price_multiplier = 1 + (sentiment * 0.005) + random.uniform(-0.001, 0.001)
        simulated_prices[symbol] *= price_multiplier
    
    # Mock options based on simulated Nifty
    nifty_price = simulated_prices.get(NIFTY_INDEX_SYMBOL, 20000)
    simulated_prices['NIFTY_CALL'] = nifty_price * 1.02
    simulated_prices['NIFTY_PUT'] = nifty_price * 0.98
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
def calculate_slippage(symbol, qty, action):
    liquidity_level = st.session_state.liquidity.get(symbol, 1.0)
    if qty <= SLIPPAGE_THRESHOLD: return 1.0
    excess_qty = qty - SLIPPAGE_THRESHOLD
    slippage_rate = BASE_SLIPPAGE_RATE / max(0.1, liquidity_level)
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

# --- Portfolio Optimization & Algo Strategy ---
def optimize_portfolio(player_holdings):
    symbols = [s for s in player_holdings.keys() if s in NIFTY50_SYMBOLS + CRYPTO_SYMBOLS]
    if len(symbols) < 2: return None, "Need at least 2 assets to optimize."
    try:
        hist_data = get_historical_data(symbols)
        if hist_data.empty or hist_data.isnull().values.any():
            return None, "Could not fetch sufficient historical data."
        mu = expected_returns.mean_historical_return(hist_data)
        S = risk_models.sample_cov(hist_data)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        return ef.clean_weights(), ef.portfolio_performance(verbose=False)
    except Exception as e:
        return None, f"Optimization failed: {e}"

# --- Game State Initialization ---
def init_game_state():
    if "players" not in st.session_state: st.session_state.players = {}
    if "game_status" not in st.session_state: st.session_state.game_status = "Stopped"
    if "game_start_time" not in st.session_state: st.session_state.game_start_time = 0
    if "prices" not in st.session_state: st.session_state.prices = {}
    if "price_history" not in st.session_state: st.session_state.price_history = {}
    if "transactions" not in st.session_state: st.session_state.transactions = {}
    if "market_sentiment" not in st.session_state: 
        st.session_state.market_sentiment = {s: 0 for s in ALL_SYMBOLS}
    if "liquidity" not in st.session_state:
        st.session_state.liquidity = {s: random.uniform(0.5, 1.0) for s in ALL_SYMBOLS}

# --- UI Functions ---
def render_sidebar():
    st.sidebar.title("📝 Game Entry")
    # ... (sidebar code remains the same as previous version)
    if "game_status" not in st.session_state: st.session_state.game_status = "Stopped"

    player_name = st.sidebar.text_input("Enter Name", key="name_input")
    mode = st.sidebar.radio("Select Mode", ["VIP Guest", "Student"], key="mode_select")
    
    if st.sidebar.button("Join Game", disabled=(st.session_state.game_status == "Running")):
        if player_name and player_name.strip() and player_name not in st.session_state.players:
            st.session_state.players[player_name] = {
                "mode": mode, "capital": INITIAL_CAPITAL, "holdings": {}, "pnl": 0,
                "leverage": 1.0, "margin_calls": 0, "pending_orders": [], "algo": "Off"
            }
            st.session_state.transactions[player_name] = []
            st.sidebar.success(f"{player_name} joined as {mode}!")
        else:
            st.sidebar.error("Name is invalid or already taken!")

    st.sidebar.title("⚙️ Admin Controls")
    if st.sidebar.button("▶️ Start Game", type="primary"):
        if st.session_state.players:
            st.session_state.game_status = "Running"
            st.session_state.game_start_time = time.time()
            st.toast("Game Started!", icon="🎉")
        else:
            st.sidebar.warning("Add at least one player to start.")
            
    if st.sidebar.button("⏸️ Stop Game"):
        st.session_state.game_status = "Stopped"
        st.toast("Game Paused!", icon="⏸️")
        
    if st.sidebar.button("🔄 Reset Game"):
        for player in st.session_state.players.values():
            player.update({"capital": INITIAL_CAPITAL, "holdings": {}, "pnl": 0, "pending_orders": [], "algo": "Off"})
        st.session_state.game_status = "Stopped"
        st.session_state.game_start_time = 0
        st.toast("Game has been reset.", icon="🔄")
        st.rerun()

def render_main_interface(prices):
    st.title(f"📈 {GAME_NAME}")
    # ... (timer and status logic remains the same)
    game_status = st.session_state.game_status
    if game_status == "Running":
        remaining_time = max(0, ROUND_DURATION - int(time.time() - st.session_state.game_start_time))
        if remaining_time == 0:
            st.session_state.game_status = "Finished"
        st.markdown(f"<div class='timer'>Time Remaining: {remaining_time // 60:02d}:{remaining_time % 60:02d}</div>", unsafe_allow_html=True)
    elif game_status == "Stopped":
        st.info("Game is paused. Press 'Start Game' in the sidebar to begin.")
    elif game_status == "Finished":
        st.success("Game has finished! See the final leaderboard below.")

    st.markdown("<div class='section'>Live Market Prices (Simulated)</div>", unsafe_allow_html=True)
    price_df = pd.DataFrame.from_dict(prices, orient='index', columns=['Price']).sort_index()
    price_df['Sentiment'] = pd.Series(st.session_state.market_sentiment)
    st.dataframe(price_df.style.format({"Price": "₹{:,.2f}", "Sentiment": "{:,.2f}"}), height=300)

    player_list = list(st.session_state.players.keys())
    if player_list:
        selected_player = st.selectbox("Choose Player", player_list, key="player_select")
        render_player_dashboard(selected_player, prices)

def render_player_dashboard(player_name, prices):
    player = st.session_state.players[player_name]
    # ... (metrics calculation remains the same)
    st.markdown(f"### {player_name}'s Terminal (Mode: {player['mode']})")
    holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
    total_value = player['capital'] + holdings_value
    pnl = total_value - INITIAL_CAPITAL
    player['pnl'] = pnl
    pnl_arrow = "🔼" if pnl >= 0 else "🔽"

    col1, col2, col3 = st.columns(3)
    col1.metric("Cash", f"₹{player['capital']:,.2f}")
    col2.metric("Portfolio Value", f"₹{total_value:,.2f}")
    col3.metric("P&L", f"₹{pnl:,.2f}", f"{pnl_arrow} {pnl/INITIAL_CAPITAL:.2%}")

    tabs = ["👨‍💻 Trade Terminal", "🤖 Algo Trading", "📂 Holdings & History", "📊 Strategy & Insights"]
    tab1, tab2, tab3, tab4 = st.tabs(tabs)

    is_game_running = st.session_state.game_status == "Running"

    with tab1:
        render_trade_interface(player_name, player, prices, is_game_running)
    with tab2:
        render_algo_trading_tab(player_name, player, is_game_running)
    with tab3:
        render_holdings_and_history(player_name, player, prices)
    with tab4:
        render_strategy_tab(player)

def render_trade_interface(player_name, player, prices, disabled_status):
    st.subheader("Market Order")
    # ... (UI is improved with columns)
    col1, col2 = st.columns([2,1])
    with col1:
        asset_type = st.radio("Asset Type", ["Stock", "Crypto", "Gold", "Option"], horizontal=True, key=f"asset_{player_name}", disabled=not disabled_status)
        if asset_type == "Stock":
            symbols = [s.replace('.NS', '') for s in NIFTY50_SYMBOLS]
            symbol_choice = st.selectbox("Stock", symbols, key=f"stock_{player_name}", disabled=not disabled_status) + '.NS'
        elif asset_type == "Crypto":
            symbol_choice = st.selectbox("Cryptocurrency", CRYPTO_SYMBOLS, key=f"crypto_{player_name}", disabled=not disabled_status)
        elif asset_type == "Gold": symbol_choice = GOLD_SYMBOL
        else: symbol_choice = st.selectbox("Option", OPTION_SYMBOLS, key=f"option_{player_name}", disabled=not disabled_status)
        
        current_price = prices.get(symbol_choice, 0)
        st.info(f"Current Price of {symbol_choice}: ₹{current_price:,.2f}")
        
    with col2:
        qty = st.number_input("Quantity", min_value=1, step=1, value=1, key=f"qty_{player_name}", disabled=not disabled_status)
        action = st.radio("Action", ["Buy", "Sell", "Short"], key=f"action_{player_name}", disabled=not disabled_status)
    
    if st.button("Place Trade", key=f"trade_btn_{player_name}", type="primary", disabled=not disabled_status, use_container_width=True):
        execute_trade(player_name, player, action, symbol_choice, qty, prices)
        st.rerun()

def render_algo_trading_tab(player_name, player, disabled_status):
    st.subheader("Automated Trading Strategies")
    st.write("Activate an algorithm to trade automatically on your behalf.")
    
    algo_choice = st.selectbox(
        "Choose Strategy",
        ["Off", "Momentum Trader", "Mean Reversion"],
        index=["Off", "Momentum Trader", "Mean Reversion"].index(player['algo']),
        disabled=not disabled_status
    )
    player['algo'] = algo_choice

    if player['algo'] == "Momentum Trader":
        st.info("This bot buys assets that have risen in price over the last minute and sells those that have fallen, betting that trends will continue.")
    elif player['algo'] == "Mean Reversion":
        st.info("This bot buys assets that have fallen, expecting them to 'revert' back to their average price, and sells assets that have risen.")
    
    if player['algo'] != "Off":
        st.success(f"'{player['algo']}' is now active! It will execute trades periodically.")

def render_holdings_and_history(player_name, player, prices):
    # ... (no changes to this function)
    st.subheader("Current Holdings")
    if player['holdings']:
        holdings_data = []
        for sym, qty in player['holdings'].items():
            value = prices.get(sym, 0) * qty
            holdings_data.append({"Symbol": sym, "Quantity": qty, "Current Value": f"₹{value:,.2f}"})
        st.dataframe(pd.DataFrame(holdings_data))
    else:
        st.info("No holdings yet.")
        
    st.subheader("Transaction History")
    if st.session_state.transactions.get(player_name):
        trans_df = pd.DataFrame(st.session_state.transactions[player_name],
                                columns=["Time", "Action", "Symbol", "Qty", "Price", "Total"])
        st.dataframe(trans_df.style.format({"Price": "{:,.2f}", "Total": "{:,.2f}"}))
    else:
        st.info("No transactions recorded.")

def render_strategy_tab(player):
    # ... (no changes to this function)
    st.subheader("📊 Strategy & Insights")
    sub_tab1, sub_tab2 = st.tabs(["Technical Analysis (SMA)", "Portfolio Optimizer"])
    
    with sub_tab1:
        render_sma_chart(player['holdings'])
    with sub_tab2:
        render_optimizer(player['holdings'])

def render_sma_chart(holdings):
    # ... (no changes to this function)
    st.markdown("##### Simple Moving Average (SMA) Chart")
    st.write("SMAs smooth out price data to identify trend direction. A shorter-term average crossing above a longer-term one (e.g., 20-day over 50-day) can signal bullish momentum.")
    
    chartable_assets = [s for s in holdings.keys() if s not in OPTION_SYMBOLS]
    if not chartable_assets:
        st.info("No chartable assets in portfolio (stocks, crypto, gold).")
        return
        
    chart_symbol = st.selectbox("Select Asset to Chart", chartable_assets)
    hist_data = get_historical_data([chart_symbol], period="6mo")

    if not hist_data.empty:
        df = hist_data.rename(columns={chart_symbol: 'Close'})
        
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='20-Day SMA'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='50-Day SMA'))
        
        fig.update_layout(title=f"Price and Moving Averages for {chart_symbol}", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Could not load historical data for {chart_symbol}.")

def render_optimizer(holdings):
    # ... (no changes to this function)
    st.subheader("Portfolio Optimization (Max Sharpe Ratio)")
    if st.button("Optimize My Portfolio"):
        weights, performance = optimize_portfolio(holdings)
        if weights:
            st.success("Optimal weights for maximum risk-adjusted return:")
            st.json({k: f"{v:.2%}" for k, v in weights.items()})
            if performance:
                st.write(f"Expected annual return: {performance[0]:.2%}")
                st.write(f"Annual volatility: {performance[1]:.2%}")
                st.write(f"Sharpe Ratio: {performance[2]:.2f}")
        else:
            st.error(performance) # Display error message

def execute_trade(player_name, player, action, symbol, qty, prices, is_algo=False):
    base_price = prices.get(symbol, 0)
    if base_price == 0: return False
    
    # Update market sentiment based on trade
    sentiment_impact = (qty / 50) * (1 if action == "Buy" else -1)
    st.session_state.market_sentiment[symbol] = st.session_state.market_sentiment.get(symbol, 0) + sentiment_impact

    trade_price = base_price * calculate_slippage(symbol, qty, action)
    cost = trade_price * qty
    
    if action == "Buy" and player['capital'] >= cost:
        player['capital'] -= cost
        player['holdings'][symbol] = player['holdings'].get(symbol, 0) + qty
    elif action == "Short" and player['capital'] >= cost * MARGIN_REQUIREMENT:
        player['capital'] += cost
        player['holdings'][symbol] = player['holdings'].get(symbol, 0) - qty
    elif action == "Sell" and player['holdings'].get(symbol, 0) >= qty:
        player['capital'] += cost
        player['holdings'][symbol] -= qty
        if player['holdings'][symbol] == 0: del player['holdings'][symbol]
    else:
        if not is_algo: st.error("Trade failed: Insufficient capital or holdings.")
        return False
        
    log_transaction(player_name, action, symbol, qty, trade_price, cost, is_algo)
    return True

def log_transaction(player_name, action, symbol, qty, price, total, is_algo=False):
    prefix = "🤖 Algo" if is_algo else ""
    st.session_state.transactions[player_name].append([time.strftime("%H:%M:%S"), f"{prefix} {action}".strip(), symbol, qty, price, total])
    if not is_algo: st.success(f"Trade Executed: {action} {qty} {symbol} @ ₹{price:,.2f}")
    else: st.toast(f"Algo Trade: {action} {qty} {symbol}", icon="🤖")

def render_leaderboard(prices):
    # ... (no changes to this function)
    st.markdown("<div class='section'>Leaderboard</div>", unsafe_allow_html=True)
    lb = []
    for pname, pdata in st.session_state.players.items():
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in pdata['holdings'].items())
        total_value = pdata['capital'] + holdings_value
        leveraged_value = (total_value - INITIAL_CAPITAL) * pdata.get('leverage', 1.0) + INITIAL_CAPITAL
        lb.append((pname, pdata['mode'], leveraged_value, pdata['pnl']))
    
    if lb:
        lb_df = pd.DataFrame(lb, columns=["Player", "Mode", "Portfolio Value", "P&L"]).sort_values("Portfolio Value", ascending=False)
        st.dataframe(lb_df.style.format({"Portfolio Value": "₹{:,.2f}", "P&L": "₹{:,.2f}"}))
        
        if st.session_state.game_status == "Finished":
            st.balloons()
            st.subheader("🏆 Round Over! Final Standings:")
            st.table(lb_df.head(3))

# --- Main Game Loop Functions ---
def run_game_tick(prices):
    if st.session_state.game_status != "Running": return prices
    
    # 1. Decay market sentiment towards neutral
    for symbol, sent in st.session_state.market_sentiment.items():
        st.session_state.market_sentiment[symbol] *= 0.95 # Decay by 5% each tick

    # 2. Handle Market Events
    if not st.session_state.get('event_active') and random.random() < 0.01:
        events = ["Flash Crash", "Bull Rally", "Banking Boost", "Sector Rotation"]
        st.session_state.event_type = random.choice(events)
        st.session_state.event_active = True
        st.session_state.event_end = time.time() + random.randint(30, 60)
        st.toast(f"⚡ Market Event: {st.session_state.event_type}!", icon="🎉")
    
    if st.session_state.get('event_active'):
        if time.time() < st.session_state.event_end:
            prices = apply_event_adjustment(prices, st.session_state.event_type)
        else:
            st.session_state.event_active = False
            st.info("Market event has ended.")
            
    # 3. Run Algo Strategies
    run_algo_strategies(prices)
    return prices

def run_algo_strategies(prices):
    history = st.session_state.price_history
    if len(history) < 2: return # Not enough data to make a decision

    prev_prices = history[-2] # Prices from the previous tick
    
    for name, player in st.session_state.players.items():
        if player['algo'] == 'Off': continue
        
        # Decide which asset to trade
        trade_symbol = random.choice(NIFTY50_SYMBOLS + CRYPTO_SYMBOLS)
        price_change = prices[trade_symbol] - prev_prices.get(trade_symbol, prices[trade_symbol])
        
        if player['algo'] == "Momentum Trader":
            if price_change > 0: # If price went up, buy
                execute_trade(name, player, "Buy", trade_symbol, 1, prices, is_algo=True)
            elif price_change < 0: # If price went down, sell
                execute_trade(name, player, "Sell", trade_symbol, 1, prices, is_algo=True)
                
        elif player['algo'] == "Mean Reversion":
            if price_change > 0: # If price went up, sell (expect it to go down)
                execute_trade(name, player, "Sell", trade_symbol, 1, prices, is_algo=True)
            elif price_change < 0: # If price went down, buy (expect it to go up)
                execute_trade(name, player, "Buy", trade_symbol, 1, prices, is_algo=True)

# --- Main App ---
def main():
    init_game_state()
    
    render_sidebar()
    
    base_prices = get_base_live_prices()
    
    # Store price history for algos
    if 'prices' in st.session_state:
        st.session_state.price_history.append(st.session_state.prices)
        if len(st.session_state.price_history) > 10: # Keep last 10 ticks
            st.session_state.price_history.pop(0)

    prices = simulate_market_prices(base_prices)
    prices = run_game_tick(prices)
    st.session_state.prices = prices # Update global prices
    
    render_main_interface(prices)
    render_leaderboard(prices)

if __name__ == "__main__":
    main()


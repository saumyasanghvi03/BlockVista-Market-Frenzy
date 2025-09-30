# ======================= Expo Game: BlockVista Market Frenzy ======================

import streamlit as st
import pandas as pd
import yfinance as yf
import time
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from alpha_vantage.timeseries import TimeSeries
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- API & Game Configuration ---
GAME_NAME = "BlockVista Market Frenzy"
INITIAL_CAPITAL = 1000000  # ₹10 lakh
ROUND_DURATION = 20 * 60   # 20 minutes in seconds

# Store API key in Streamlit secrets
ALPHA_VANTAGE_API_KEY = st.secrets.get("ALPHA_VANTAGE_API_KEY", "YOUR_API_KEY")

# Define asset symbols
NIFTY50_SYMBOLS = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS',
                   'KOTAKBANK.NS', 'SBIN.NS', 'ITC.NS', 'ASIANPAINT.NS', 'AXISBANK.NS']
CRYPTO_SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD']
GOLD_SYMBOL = 'GC=F'
NIFTY_INDEX_SYMBOL = 'NSEI' # Adjusted for Alpha Vantage query
OPTION_SYMBOLS = ['NIFTY_CALL', 'NIFTY_PUT']

ALL_SYMBOLS = NIFTY50_SYMBOLS + CRYPTO_SYMBOLS + [GOLD_SYMBOL] + OPTION_SYMBOLS

# Game Mechanics Settings
LIQUIDITY_LEVELS = {sym: random.uniform(0.5, 1.0) for sym in ALL_SYMBOLS}
SLIPPAGE_THRESHOLD = 10
BASE_SLIPPAGE_RATE = 0.005
MARGIN_REQUIREMENT = 0.2
VOLATILITY_BASE = {sym: random.uniform(0.1, 0.3) for sym in ALL_SYMBOLS}

# --- Quiz Questions ---
QUIZ_QUESTIONS = [
    {"question": "What does RSI stand for in technical analysis?", "options": ["Relative Strength Index", "Rapid Stock Increase", "Risk Sensitivity Indicator"], "answer": 0},
    {"question": "Which candlestick pattern signals a bullish reversal?", "options": ["Doji", "Hammer", "Shooting Star"], "answer": 1},
    {"question": "What is the primary role of SEBI in India?", "options": ["Regulate securities market", "Control inflation", "Manage foreign exchange"], "answer": 0},
    {"question": "In crypto, what is a 'halving' event for Bitcoin?", "options": ["A 50% market crash", "Splitting the blockchain", "Reducing block rewards by half"], "answer": 2},
    {"question": "What is short-selling?", "options": ["Buying low, selling high", "Selling borrowed shares", "Holding for dividends"], "answer": 1},
    {"question": "What does 'DeFi' stand for?", "options": ["Decentralized Finance", "Defined Finance", "Default Finality"], "answer": 0}
]

# --- Data Fetching Functions ---
@st.cache_data(ttl=60)
def get_live_prices():
    prices = {}
    
    # Fetch Nifty50, Crypto, and Gold using yfinance
    yf_symbols = NIFTY50_SYMBOLS + CRYPTO_SYMBOLS + [GOLD_SYMBOL]
    try:
        data = yf.download(tickers=yf_symbols, period="1d", interval="1m", progress=False)
        for symbol in yf_symbols:
            if not data.empty and symbol in data['Close'] and not pd.isna(data['Close'][symbol].iloc[-1]):
                prices[symbol] = data['Close'][symbol].iloc[-1]
            else: # Fallback for failed fetch
                prices[symbol] = random.uniform(10, 5000)
    except Exception as e:
        st.warning(f"yfinance fetch failed: {e}. Using random data for stocks/crypto.")
        for symbol in yf_symbols:
            prices[symbol] = random.uniform(10, 5000)

    # Fetch Nifty Index using Alpha Vantage
    try:
        ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
        data, _ = ts.get_quote_endpoint(symbol=NIFTY_INDEX_SYMBOL)
        prices['^NSEI'] = float(data['05. price'].iloc[0])
    except Exception as e:
        st.warning(f"Alpha Vantage API error: {e}. Using fallback for Nifty index.")
        prices['^NSEI'] = random.uniform(18000, 22000)
        
    # Mock options based on Nifty
    nifty_price = prices.get('^NSEI', 20000)
    prices['NIFTY_CALL'] = nifty_price * 1.02
    prices['NIFTY_PUT'] = nifty_price * 0.98
    
    return prices

@st.cache_data(ttl=3600)
def get_historical_data(symbols, period="3mo"):
    try:
        data = yf.download(tickers=symbols, period=period, progress=False)
        return data['Close']
    except Exception:
        return pd.DataFrame()

# --- Game Logic Functions ---
def calculate_slippage(symbol, qty, action, liquidity_level):
    if qty <= SLIPPAGE_THRESHOLD:
        return 1.0
    excess_qty = qty - SLIPPAGE_THRESHOLD
    slippage_rate = BASE_SLIPPAGE_RATE / max(0.1, liquidity_level)
    slippage_mult = 1 + (slippage_rate * excess_qty) * (1 if action == "Buy" else -1)
    return max(0.9, min(1.1, slippage_mult))

def apply_event_adjustment(prices, event_type):
    adjusted_prices = prices.copy()
    event_duration = random.randint(30, 90)
    if event_type == "Flash Crash":
        adjusted_prices = {k: v * random.uniform(0.88, 0.92) for k, v in adjusted_prices.items()}
    elif event_type == "Bull Rally":
        adjusted_prices = {k: v * random.uniform(1.08, 1.12) for k, v in adjusted_prices.items()}
    elif event_type == "Crypto Pump":
        for sym in CRYPTO_SYMBOLS:
            if sym in adjusted_prices:
                adjusted_prices[sym] *= random.uniform(1.15, 1.25)
        st.info("🚀 Crypto Pump! Digital assets are soaring.")
    elif event_type == "Regulatory Crackdown":
        for sym in CRYPTO_SYMBOLS + OPTION_SYMBOLS:
             if sym in adjusted_prices:
                adjusted_prices[sym] *= random.uniform(0.80, 0.90)
        st.warning("🚨 Regulatory Crackdown on Crypto & Options!")
    return adjusted_prices, event_duration

# --- Portfolio Optimization ---
def optimize_portfolio(player_holdings):
    if not player_holdings or len(player_holdings) < 2:
        return None, "Need at least 2 assets to optimize."

    symbols = [s for s in player_holdings.keys() if s in NIFTY50_SYMBOLS + CRYPTO_SYMBOLS]
    if len(symbols) < 2:
        return None, "Not enough optimizable assets in portfolio."
        
    try:
        hist_data = get_historical_data(symbols)
        if hist_data.empty or hist_data.isnull().values.any():
            return None, "Could not fetch sufficient historical data for optimization."

        mu = expected_returns.mean_historical_return(hist_data)
        S = risk_models.sample_cov(hist_data)
        
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        
        return cleaned_weights, ef.portfolio_performance(verbose=True)
    except Exception as e:
        return None, f"Optimization failed: {e}"

# --- Game State Initialization ---
def init_game_state():
    if "players" not in st.session_state:
        st.session_state.players = {}
    if "game_start" not in st.session_state:
        st.session_state.game_start = time.time()
    if "prices" not in st.session_state:
        st.session_state.prices = get_live_prices()
    if "event_active" not in st.session_state:
        st.session_state.event_active = False
    if "event_end" not in st.session_state:
        st.session_state.event_end = 0
    if "quiz_triggered" not in st.session_state:
        st.session_state.quiz_triggered = False
    if "transactions" not in st.session_state:
        st.session_state.transactions = {}

# --- UI Functions ---
def load_css():
    try:
        with open("styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def render_sidebar():
    st.sidebar.title("📝 Game Entry")
    player_name = st.sidebar.text_input("Enter Name", key="name_input")
    mode = st.sidebar.radio("Select Mode", ["VIP Guest", "Student"], key="mode_select")
    
    if st.sidebar.button("Join Game"):
        if player_name and player_name.strip() and player_name not in st.session_state.players:
            st.session_state.players[player_name] = {
                "mode": mode, "capital": INITIAL_CAPITAL, "holdings": {}, "pnl": 0,
                "leverage": 1.0, "margin_calls": 0
            }
            st.session_state.transactions[player_name] = []
            st.sidebar.success(f"{player_name} joined as {mode}!")
        else:
            st.sidebar.error("Name is invalid or already taken!")

    st.sidebar.title("⚙️ Admin Controls")
    if st.sidebar.button("Reset Game"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.sidebar.success("Game reset!")
        st.rerun()

def render_main_interface(prices):
    st.title(GAME_NAME)
    remaining_time = max(0, ROUND_DURATION - int(time.time() - st.session_state.game_start))
    st.markdown(f"<div class='timer'>Time Remaining: {remaining_time // 60:02d}:{remaining_time % 60:02d}</div>", unsafe_allow_html=True)

    st.markdown("<div class='section'>Live Market Prices</div>", unsafe_allow_html=True)
    price_df = pd.DataFrame.from_dict(prices, orient='index', columns=['Price']).sort_index()
    st.dataframe(price_df.style.format({"Price": "{:,.2f}"}), height=300)

    player_list = list(st.session_state.players.keys())
    if player_list:
        selected_player = st.selectbox("Choose Player", player_list, key="player_select")
        render_player_dashboard(selected_player, prices)

def render_player_dashboard(player_name, prices):
    player = st.session_state.players[player_name]
    
    # Portfolio Metrics
    st.markdown(f"<div class='section'>{player_name}'s Portfolio (Mode: {player['mode']})</div>", unsafe_allow_html=True)
    holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
    total_value = player['capital'] + holdings_value
    leverage = player.get('leverage', 1.0)
    leveraged_value = (total_value - INITIAL_CAPITAL) * leverage + INITIAL_CAPITAL
    pnl = leveraged_value - INITIAL_CAPITAL
    player['pnl'] = pnl
    pnl_color = "green" if pnl >= 0 else "red"

    col1, col2, col3 = st.columns(3)
    col1.metric("Cash", f"₹{player['capital']:,.2f}")
    col2.metric("Portfolio Value", f"₹{leveraged_value:,.2f}")
    col3.metric("P&L", f"₹{pnl:,.2f}")

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Trade", "Holdings & History", "Charts", "Portfolio Optimizer"])

    with tab1:
        render_trade_interface(player_name, player, prices)
    with tab2:
        render_holdings_and_history(player_name, player, prices)
    with tab3:
        render_charts(player['holdings'])
    with tab4:
        render_optimizer(player['holdings'])

def render_trade_interface(player_name, player, prices):
    action = st.radio("Action", ["Buy", "Sell", "Short"], horizontal=True, key=f"action_{player_name}")
    asset_type = st.radio("Asset Type", ["Stock", "Crypto", "Gold", "Option"], horizontal=True, key=f"asset_{player_name}")

    if asset_type == "Stock":
        symbols = [s.replace('.NS', '') for s in NIFTY50_SYMBOLS]
        symbol_choice = st.selectbox("Stock", symbols, key=f"stock_{player_name}") + '.NS'
    elif asset_type == "Crypto":
        symbol_choice = st.selectbox("Cryptocurrency", CRYPTO_SYMBOLS, key=f"crypto_{player_name}")
    elif asset_type == "Gold":
        symbol_choice = GOLD_SYMBOL
    else:
        symbol_choice = st.selectbox("Option", OPTION_SYMBOLS, key=f"option_{player_name}")

    qty = st.number_input("Quantity", min_value=1, step=1, value=1, key=f"qty_{player_name}")
    
    if st.button("Place Trade", key=f"trade_btn_{player_name}"):
        execute_trade(player_name, player, action, symbol_choice, qty, prices)
        st.rerun()

def render_holdings_and_history(player_name, player, prices):
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

def render_charts(holdings):
    if not holdings:
        st.info("No assets in portfolio to chart.")
        return
    
    chart_symbol = st.selectbox("Select Asset to Chart", list(holdings.keys()))
    hist_data = get_historical_data([chart_symbol], period="1y")

    if not hist_data.empty:
        fig = go.Figure(data=[go.Candlestick(x=hist_data.index,
                                             open=hist_data[chart_symbol],
                                             high=hist_data[chart_symbol],
                                             low=hist_data[chart_symbol],
                                             close=hist_data[chart_symbol])])
        fig.update_layout(title=f"Historical Price Chart for {chart_symbol}", xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Could not load historical data for {chart_symbol}.")

def render_optimizer(holdings):
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

def execute_trade(player_name, player, action, symbol, qty, prices):
    base_price = prices.get(symbol, 0)
    if base_price == 0:
        st.error(f"Could not get price for {symbol}. Trade cancelled.")
        return

    liq_level = LIQUIDITY_LEVELS.get(symbol, 1.0)
    slippage_mult = calculate_slippage(symbol, qty, action, liq_level)
    trade_price = base_price * slippage_mult
    cost = trade_price * qty
    
    if action == "Buy":
        if player['capital'] >= cost:
            player['capital'] -= cost
            player['holdings'][symbol] = player['holdings'].get(symbol, 0) + qty
            log_transaction(player_name, action, symbol, qty, trade_price, cost)
        else:
            st.error("Insufficient capital!")
    elif action == "Short":
        margin_needed = cost * MARGIN_REQUIREMENT
        if player['capital'] >= margin_needed:
            player['capital'] += cost # Add proceeds to capital
            player['holdings'][symbol] = player['holdings'].get(symbol, 0) - qty
            log_transaction(player_name, action, symbol, qty, trade_price, -cost)
        else:
            st.error("Insufficient capital for margin!")
    elif action == "Sell":
        current_qty = player['holdings'].get(symbol, 0)
        if abs(current_qty) >= qty:
            player['capital'] += cost
            player['holdings'][symbol] -= qty
            if player['holdings'][symbol] == 0:
                del player['holdings'][symbol]
            log_transaction(player_name, action, symbol, qty, trade_price, cost)
        else:
            st.error("Insufficient holdings to sell!")

def log_transaction(player_name, action, symbol, qty, price, total):
    st.session_state.transactions[player_name].append([
        time.strftime("%H:%M:%S"), action, symbol, qty, price, total
    ])
    st.success(f"Trade Executed: {action} {qty} {symbol} @ ₹{price:,.2f}")

def render_leaderboard(prices):
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
        
        remaining_time = max(0, ROUND_DURATION - int(time.time() - st.session_state.game_start))
        if remaining_time <= 0:
            st.balloons()
            st.subheader("🏆 Round Over! Final Standings:")
            st.table(lb_df.head(3))

def handle_market_events(prices):
    if not st.session_state.event_active and random.random() < 0.015:
        events = ["Flash Crash", "Bull Rally", "Crypto Pump", "Regulatory Crackdown"]
        event_type = random.choice(events)
        st.session_state.event_active = True
        st.session_state.event_type = event_type
        _, event_duration = apply_event_adjustment(prices, event_type)
        st.session_state.event_end = time.time() + event_duration
        st.toast(f"⚡ Market Event: {event_type}!", icon="🎉")
        st.rerun()

    if st.session_state.event_active:
        if time.time() < st.session_state.event_end:
            st.warning(f"⚡ Market Event Active: {st.session_state.event_type}! Prices are affected.")
            prices, _ = apply_event_adjustment(prices, st.session_state.event_type)
        else:
            st.session_state.event_active = False
            st.info("Market event has ended. Prices are normalizing.")
            st.rerun()
    return prices
            
def handle_quiz_bonus():
    if not st.session_state.quiz_triggered and random.random() < 0.005 and st.session_state.players:
        st.session_state.quiz_triggered = True
        player_name = random.choice(list(st.session_state.players.keys()))
        quiz = random.choice(QUIZ_QUESTIONS)
        st.session_state.quiz_data = {"player": player_name, "quiz": quiz}

    if st.session_state.get('quiz_data'):
        data = st.session_state.quiz_data
        player_name = data["player"]
        quiz = data["quiz"]
        
        with st.sidebar.expander(f"📚 Quiz for {player_name}!", expanded=True):
            st.write(quiz["question"])
            answer = st.radio("Options", quiz["options"], key=f"quiz_{player_name}")
            if st.button("Submit Answer", key=f"quiz_btn_{player_name}"):
                player = st.session_state.players[player_name]
                if quiz["options"].index(answer) == quiz["answer"]:
                    bonus = random.randint(25000, 75000)
                    player['capital'] += bonus
                    st.success(f"Correct! {player_name} gets a ₹{bonus:,.2f} bonus.")
                    log_transaction(player_name, "Quiz Bonus", "N/A", 1, bonus, bonus)
                else:
                    st.error("Incorrect. Better luck next time!")
                del st.session_state.quiz_data
                st.rerun()

# --- Main Game Loop ---
def main():
    load_css()
    init_game_state()
    
    render_sidebar()
    
    prices = get_live_prices()
    prices = handle_market_events(prices)
    
    render_main_interface(prices)
    render_leaderboard(prices)
    
    handle_quiz_bonus()

if __name__ == "__main__":
    main()


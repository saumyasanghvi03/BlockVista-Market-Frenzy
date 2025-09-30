# ======================= Expo Game: BlockVista Market Frenzy ======================

import streamlit as st
import pandas as pd
import yfinance as yf
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import io
import requests  # Added for BlockVista API integration
from typing import Dict, Any

# --- BlockVista API Configuration ---
# Replace with your actual BlockVista API details
BLOCKVISTA_API_URL = "https://api.blockvista.com/v1"  # Example base URL
BLOCKVISTA_API_KEY = st.secrets.get("BLOCKVISTA_API_KEY", "your_api_key_here")  # Store in Streamlit secrets
SYMBOLS_FOR_API = ['RELIANCE', 'HDFCBANK', 'ICICIBANK', 'INFY', 'TCS',
                   'KOTAKBANK', 'SBIN', 'ITC', 'ASIANPAINT', 'AXISBANK']  # Without .NS for API

# --- Game Config ---
GAME_NAME = "BlockVista Market Frenzy"
INITIAL_CAPITAL = 1000000  # ₹10 lakh
ROUND_DURATION = 20 * 60   # 20 minutes in seconds
NIFTY50_SYMBOLS = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS',
                   'KOTAKBANK.NS', 'SBIN.NS', 'ITC.NS', 'ASIANPAINT.NS', 'AXISBANK.NS']
GOLD_SYMBOL = 'GC=F'  # Gold futures (fallback to yf if not in BlockVista)
OPTION_SYMBOLS = ['NIFTY_CALL', 'NIFTY_PUT']  # Mock options (extend API if supported)
LIQUIDITY_LEVELS = {sym: random.uniform(0.5, 1.0) for sym in NIFTY50_SYMBOLS + [GOLD_SYMBOL] + OPTION_SYMBOLS}
SLIPPAGE_THRESHOLD = 10  # Qty threshold for slippage
BASE_SLIPPAGE_RATE = 0.005  # 0.5% per extra unit
MARGIN_REQUIREMENT = 0.2  # 20% margin for shorts/leverage
VOLATILITY_BASE = {sym: random.uniform(0.1, 0.3) for sym in NIFTY50_SYMBOLS + [GOLD_SYMBOL] + OPTION_SYMBOLS}

# --- Quiz Questions ---
QUIZ_QUESTIONS = [
    {"question": "What does RSI stand for in technical analysis?", "options": ["Relative Strength Index", "Rapid Stock Increase", "Risk Sensitivity Indicator"], "answer": 0},
    {"question": "Which candlestick pattern signals a bullish reversal?", "options": ["Doji", "Hammer", "Shooting Star"], "answer": 1},
    {"question": "What is the primary role of SEBI in India?", "options": ["Regulate securities market", "Control inflation", "Manage foreign exchange"], "answer": 0},
    {"question": "In crypto trading, what causes 'slippage'?", "options": ["Low liquidity", "High fees", "Network congestion"], "answer": 0},
    {"question": "What is short-selling?", "options": ["Buying low, selling high", "Selling borrowed shares", "Holding for dividends"], "answer": 1},
]

# --- Fetch Live Prices from BlockVista API ---
@st.cache_data(ttl=60)
def get_live_prices():
    prices = {}
    try:
        # BlockVista API Call for Nifty50
        headers = {"Authorization": f"Bearer {BLOCKVISTA_API_KEY}"}
        params = {"symbols": ",".join(SYMBOLS_FOR_API), "exchange": "NSE"}
        response = requests.get(f"{BLOCKVISTA_API_URL}/prices", headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            api_data = response.json()
            # Assume API returns {'data': {'RELIANCE': {'price': 2500.0}, ...}}
            for symbol, data in api_data.get('data', {}).items():
                prices[f"{symbol}.NS"] = data.get('price', random.uniform(1000, 3500))
            st.success("Fetched live prices from BlockVista API!")  # Optional notification
        else:
            st.warning(f"BlockVista API error ({response.status_code}). Falling back to yfinance.")
    except Exception as e:
        st.warning(f"BlockVista API unavailable: {e}. Falling back to yfinance.")
    
    # Fallback to yfinance for missing symbols (e.g., Gold, Options, Nifty Index)
    fallback_symbols = [s for s in [GOLD_SYMBOL, '^NSEI'] if s not in prices]
    for symbol in fallback_symbols:
        try:
            data = yf.Ticker(symbol).info
            prices[symbol] = data.get('regularMarketPrice') or data.get('previousClose') or random.uniform(1000, 3500)
        except:
            prices[symbol] = random.uniform(1000, 3500)
    
    # Mock options based on Nifty (extend with BlockVista options endpoint if available)
    nifty_price = prices.get('^NSEI', 100)
    prices['NIFTY_CALL'] = nifty_price * 1.02
    prices['NIFTY_PUT'] = nifty_price * 0.98
    
    return prices

# --- Calculate Slippage ---
def calculate_slippage(symbol, qty, action, liquidity_level):
    if qty <= SLIPPAGE_THRESHOLD:
        return 1.0
    excess_qty = qty - SLIPPAGE_THRESHOLD
    slippage_rate = BASE_SLIPPAGE_RATE / max(0.1, liquidity_level)
    slippage_mult = 1 + (slippage_rate * excess_qty) * (1 if action == "Buy" else -1)
    return max(0.9, min(1.1, slippage_mult))  # Cap slippage at ±10%

# --- Apply Volatility Spike ---
def apply_volatility_spike(prices, symbol, multiplier):
    prices[symbol] *= multiplier
    VOLATILITY_BASE[symbol] *= 1.5  # Increase volatility temporarily
    return prices

# --- Apply Special Event Adjustments ---
def apply_event_adjustment(prices, event_type, liquidity_adjust=False):
    adjusted_prices = prices.copy()
    event_duration = random.randint(30, 90)  # Varied duration
    if event_type == "Flash Crash":
        adjusted_prices = {k: v * 0.90 for k, v in adjusted_prices.items()}
        if liquidity_adjust:
            for sym in LIQUIDITY_LEVELS:
                LIQUIDITY_LEVELS[sym] *= 0.65
    elif event_type == "Bull Rally":
        adjusted_prices = {k: v * 1.15 for k, v in adjusted_prices.items()}
        if liquidity_adjust:
            for sym in LIQUIDITY_LEVELS:
                LIQUIDITY_LEVELS[sym] *= 1.3
    elif event_type == "Banking Boost":
        for sym in ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'AXISBANK.NS']:
            adjusted_prices[sym] *= 1.07
    elif event_type == "Earnings Surprise":
        surprise_sym = random.choice(list(adjusted_prices.keys()))
        multiplier = random.choice([1.2, 0.8])
        adjusted_prices = apply_volatility_spike(adjusted_prices, surprise_sym, multiplier)
        st.info(f"🚨 Earnings Surprise on {surprise_sym}: {multiplier > 1 and 'Beat!' or 'Miss!'}")
    elif event_type == "Geopolitical Tension":
        adjusted_prices = {k: v * 0.93 for k, v in adjusted_prices.items()}
        if liquidity_adjust:
            for sym in LIQUIDITY_LEVELS:
                LIQUIDITY_LEVELS[sym] *= 0.75
    elif event_type == "Meme Stock Pump":
        for sym in ['INFY.NS', 'TCS.NS']:
            adjusted_prices = apply_volatility_spike(adjusted_prices, sym, 1.25)
        st.warning("🔥 Meme frenzy in Tech stocks!")
    elif event_type == "Halt Trading":
        st.error("⏸️ Trading Halted for 30s! Prices frozen.")
        return prices, 30  # Freeze prices
    elif event_type == "Regulatory Crackdown":  # New: Options hit hard
        for sym in OPTION_SYMBOLS:
            adjusted_prices[sym] *= 0.85
            LIQUIDITY_LEVELS[sym] *= 0.6
        st.warning("🚨 Regulatory Crackdown on Options!")
    elif event_type == "Sector Rotation":  # New: Shift between sectors
        old_sector = ['HDFCBANK.NS', 'ICICIBANK.NS']
        new_sector = ['INFY.NS', 'TCS.NS']
        for sym in old_sector:
            adjusted_prices[sym] *= 0.95
        for sym in new_sector:
            adjusted_prices[sym] *= 1.10
        st.info("🔄 Sector Rotation: Out of Banking, into Tech!")
    return adjusted_prices, event_duration

# --- Game State ---
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
if "liquidity_shown" not in st.session_state:
    st.session_state.liquidity_shown = False
if "transactions" not in st.session_state:
    st.session_state.transactions = {}

# --- CSS Styling ---
try:
    with open("styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    pass

# --- Game Registration ---
st.sidebar.title("📝 Game Entry")
player_name = st.sidebar.text_input("Enter Name", key="name_input")
mode = st.sidebar.radio("Select Mode", ["VIP Guest", "Student"], key="mode_select")
if st.sidebar.button("Join Game"):
    if player_name and player_name not in st.session_state.players and len(player_name.strip()) > 0:
        st.session_state.players[player_name] = {
            "mode": mode,
            "capital": INITIAL_CAPITAL,
            "holdings": {},  # Positive for long, negative for short
            "pnl": 0,
            "leverage": 1.0 if mode == "Student" else 2.0,  # VIPs get 2x leverage option
            "margin_calls": 0
        }
        st.session_state.transactions[player_name] = []
        st.sidebar.success(f"{player_name} joined as {mode}!")
    else:
        st.sidebar.error("Name taken, empty, or invalid!")

# --- Admin Controls ---
st.sidebar.title("⚙️ Admin Controls")
if st.sidebar.button("Reset Game"):
    st.session_state.players = {}
    st.session_state.transactions = {}
    st.session_state.game_start = time.time()
    st.session_state.event_active = False
    st.session_state.quiz_triggered = False
    global LIQUIDITY_LEVELS, VOLATILITY_BASE
    LIQUIDITY_LEVELS = {sym: random.uniform(0.5, 1.0) for sym in NIFTY50_SYMBOLS + [GOLD_SYMBOL] + OPTION_SYMBOLS}
    VOLATILITY_BASE = {sym: random.uniform(0.1, 0.3) for sym in NIFTY50_SYMBOLS + [GOLD_SYMBOL] + OPTION_SYMBOLS}
    st.sidebar.success("Game reset!")

# --- Trading Interface ---
st.title(GAME_NAME)
remaining_time = max(0, ROUND_DURATION - int(time.time() - st.session_state.game_start))
st.markdown(f"<div class='timer'>Time Remaining: {remaining_time // 60} min {remaining_time % 60} sec</div>", unsafe_allow_html=True)

# Update prices
prices = get_live_prices()
if st.session_state.event_active and time.time() < st.session_state.event_end:
    prices, _ = apply_event_adjustment(prices, st.session_state.event_type, liquidity_adjust=True)
st.session_state.prices = prices

st.markdown("<div class='section'>Live Market Prices (via BlockVista API)</div>", unsafe_allow_html=True)
price_df = pd.DataFrame.from_dict(prices, orient='index', columns=['Price']).sort_index()
price_df['Liquidity'] = pd.Series(LIQUIDITY_LEVELS).reindex(price_df.index)
price_df['Volatility'] = pd.Series(VOLATILITY_BASE).reindex(price_df.index)
st.dataframe(price_df.style.format({"Price": "{:.2f}", "Liquidity": "{:.2f}", "Volatility": "{:.2f}"}))

# Liquidity Explanation
if not st.session_state.liquidity_shown:
    with st.expander("💡 Liquidity & Volatility Explained"):
        st.write("""
        - **Liquidity (0.5-1.0):** Low values mean higher slippage on large trades (>10 units).
        - **Slippage:** Prices worsen by ~0.5% per extra unit, worse for low liquidity.
        - **Volatility (0.1-0.3):** Higher values mean bigger price swings during events.
        - **Short-Selling:** Sell before owning (negative holdings), but maintain 20% margin.
        - **Leverage (VIP Only):** 2x multiplier on gains/losses, with margin calls if capital drops.
        """)
    st.session_state.liquidity_shown = True

# Player Interface
player_list = list(st.session_state.players.keys())
if player_list:
    selected_player = st.selectbox("Choose Player", player_list, key="player_select")
    player = st.session_state.players[selected_player]
    
    # Holdings Chart
    if player['holdings']:
        fig, ax = plt.subplots(figsize=(6, 4))
        holdings_df = pd.DataFrame(list(player['holdings'].items()), columns=['Symbol', 'Qty'])
        holdings_df['Symbol'] = holdings_df['Symbol'].str.replace('.NS', '')
        holdings_df.plot(kind='bar', x='Symbol', y='Qty', ax=ax, color='#1f77b4')
        ax.set_title("Portfolio Holdings (Positive = Long, Negative = Short)")
        st.pyplot(fig)
    
    # Portfolio Snapshot Download
    holdings_data = []
    for sym, qty in player['holdings'].items():
        holdings_data.append({"Symbol": sym.replace('.NS', ''), "Quantity": qty, "Value": prices.get(sym, 0) * qty})
    holdings_df = pd.DataFrame(holdings_data)
    csv = holdings_df.to_csv(index=False)
    st.download_button(
        label="Download Portfolio Snapshot",
        data=csv,
        file_name=f"{selected_player}_portfolio.csv",
        mime="text/csv"
    )
    
    # Transaction History
    if st.session_state.transactions.get(selected_player):
        st.markdown("<div class='section'>Transaction History</div>", unsafe_allow_html=True)
        trans_df = pd.DataFrame(st.session_state.transactions[selected_player],
                                columns=["Time", "Action", "Symbol", "Qty", "Price", "Total"])
        st.dataframe(trans_df.style.format({"Price": "{:.2f}", "Total": "{:.2f}"}))

    # Portfolio Metrics
    st.markdown(f"<div class='section'>{selected_player}'s Portfolio (Mode: {player['mode']})</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'>Cash: ₹{player['capital']:.2f}</div>", unsafe_allow_html=True)
    holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
    total_value = player['capital'] + holdings_value
    leverage = player['leverage']
    leveraged_value = total_value * leverage
    pnl = (total_value - INITIAL_CAPITAL) * leverage
    player['pnl'] = pnl
    color = "green" if pnl >= 0 else "red"
    st.markdown(f"<div class='metric'>Portfolio Value: ₹{leveraged_value:.2f} (Leverage: {leverage}x) | P&L: <span style='color:{color}'>₹{pnl:.2f}</span></div>", unsafe_allow_html=True)
    
    # Margin Check
    margin_needed = abs(holdings_value) * MARGIN_REQUIREMENT
    if total_value < margin_needed and player['holdings']:
        player['margin_calls'] += 1
        st.error(f"⚠️ Margin Call! Portfolio value (₹{total_value:.2f}) below required margin (₹{margin_needed:.2f}). Liquidate positions! (Call #{player['margin_calls']})")
    
    # Trade Entry
    action = st.radio("Action", ["Buy", "Sell", "Short"], key="action_select", horizontal=True)
    asset_type = st.radio("Asset Type", ["Stock", "Gold", "Option"], key="asset_select", horizontal=True)
    leverage_toggle = st.checkbox("Use 2x Leverage (VIP Only)", value=player['leverage'] == 2.0, disabled=player['mode'] != "VIP Guest", key="leverage_toggle")
    if leverage_toggle and player['mode'] == "VIP Guest":
        player['leverage'] = 2.0
    else:
        player['leverage'] = 1.0
    
    if asset_type == "Stock":
        symbols = [s.replace('.NS', '') for s in NIFTY50_SYMBOLS]
        symbol = st.selectbox("Stock", symbols, key="stock_select")
        symbol = symbol + '.NS'
    elif asset_type == "Gold":
        symbol = GOLD_SYMBOL
    else:
        symbol = st.selectbox("Option", OPTION_SYMBOLS, key="option_select")
    
    liq_level = LIQUIDITY_LEVELS.get(symbol, 1.0)
    vol_level = VOLATILITY_BASE.get(symbol, 0.2)
    if liq_level < 0.7:
        st.warning(f"⚠️ Low Liquidity ({liq_level:.2f}) on {symbol} - Expect slippage!")
    if vol_level > 0.25:
        st.warning(f"🔥 High Volatility ({vol_level:.2f}) on {symbol} - Prices may swing!")
    
    qty = st.number_input("Quantity", min_value=1, step=1, value=1, key="qty_input")
    if st.button("Place Trade"):
        base_price = prices.get(symbol, 0)
        slippage_mult = calculate_slippage(symbol, qty, action, liq_level)
        trade_price = base_price * slippage_mult
        cost = trade_price * qty * leverage
        margin_needed = cost * MARGIN_REQUIREMENT
        if action in ["Buy", "Short"]:
            if player['capital'] >= cost + margin_needed:
                player['capital'] -= cost
                qty_adjusted = qty if action == "Buy" else -qty  # Negative for short
                player['holdings'][symbol] = player['holdings'].get(symbol, 0) + qty_adjusted
                slippage_note = f" (Slippage: {slippage_mult:.3f}x)" if abs(slippage_mult - 1) > 0.001 else ""
                st.success(f"{action} {qty} of {symbol} at ₹{trade_price:.2f}{slippage_note} (Leverage: {leverage}x)!")
                st.session_state.transactions[selected_player].append([
                    time.strftime("%H:%M:%S"), action, symbol.replace('.NS', ''), qty, trade_price, cost
                ])
            else:
                st.error("Insufficient capital or margin!")
        elif action == "Sell":
            current_qty = player['holdings'].get(symbol, 0)
            if abs(current_qty) >= qty and (current_qty > 0 or (current_qty < 0 and action == "Sell")):
                player['capital'] += cost
                player['holdings'][symbol] -= qty
                if player['holdings'][symbol] == 0:
                    del player['holdings'][symbol]
                slippage_note = f" (Slippage: {slippage_mult:.3f}x)" if abs(slippage_mult - 1) > 0.001 else ""
                st.success(f"Sold {qty} of {symbol} at ₹{trade_price:.2f}{slippage_note} (Leverage: {leverage}x)!")
                st.session_state.transactions[selected_player].append([
                    time.strftime("%H:%M:%S"), action, symbol.replace('.NS', ''), qty, trade_price, cost
                ])
            else:
                st.error("Insufficient holdings or invalid sell for short position!")

# --- Leaderboard ---
lb = []
for pname, pdata in st.session_state.players.items():
    holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in pdata['holdings'].items())
    total_value = pdata['capital'] + holdings_value
    leveraged_value = total_value * pdata['leverage']
    lb.append((pname, pdata['mode'], leveraged_value, pdata['pnl']))
lb_df = pd.DataFrame(lb, columns=["Player", "Mode", "Portfolio Value", "P&L"]).sort_values("Portfolio Value", ascending=False)
st.markdown("<div class='section'>Leaderboard</div>", unsafe_allow_html=True)
st.dataframe(lb_df.style.format({"Portfolio Value": "{:.2f}", "P&L": "{:.2f}"}))

# --- End of Round ---
if remaining_time <= 0:
    st.balloons()
    st.markdown("<div class='section'>Round Over! Winners:</div>", unsafe_allow_html=True)
    # Reset index to start from 1 for ranking
    winner_df = lb_df.head(3).reset_index(drop=True)
    winner_df.index += 1
    for i, row in winner_df.iterrows():
        st.markdown(f"<div class='winner'>{i}. {row['Player']} ({row['Mode']}): ₹{row['Portfolio Value']:.2f}</div>", unsafe_allow_html=True)

# --- Special Market Events ---
if random.random() < 0.02 and not st.session_state.event_active:
    events = ["Flash Crash", "Bull Rally", "Banking Boost", "Earnings Surprise",
              "Geopolitical Tension", "Meme Stock Pump", "Halt Trading",
              "Regulatory Crackdown", "Sector Rotation"]
    event_type = random.choice(events)
    st.warning(f"⚡ Market Event: {event_type}! Prices & liquidity adjusted!")
    st.session_state.event_active = True
    st.session_state.event_type = event_type
    prices, event_duration = apply_event_adjustment(prices, event_type, liquidity_adjust=True)
    st.session_state.prices = prices
    st.session_state.event_end = time.time() + event_duration
elif st.session_state.event_active and time.time() >= st.session_state.event_end:
    st.info("Market Event Ended. Prices & liquidity normalized.")
    st.session_state.event_active = False
    for sym in VOLATILITY_BASE:  # Reset volatility
        VOLATILITY_BASE[sym] = random.uniform(0.1, 0.3)

# --- Quiz Bonus ---
if random.random() < 0.005 and not st.session_state.quiz_triggered and 'selected_player' in locals():
    quiz = random.choice(QUIZ_QUESTIONS)
    with st.expander("📚 Quick Quiz! Answer for Bonus Capital", expanded=True):
        st.markdown(f"<div class='quiz'>{quiz['question']}</div>", unsafe_allow_html=True)
        answer = st.radio("Options:", quiz["options"], key="quiz_answer", index=None)
        if st.button("Submit Answer"):
            if answer and quiz["options"].index(answer) == quiz["answer"]:
                bonus = random.randint(10000, 50000)
                player['capital'] += bonus
                st.success(f"Correct! +₹{bonus} bonus.")
                st.session_state.transactions[selected_player].append([
                    time.strftime("%H:%M:%S"), "Quiz Bonus", "N/A", 1, bonus, bonus
                ])
            else:
                st.error("Incorrect. Try next time!")
            st.session_state.quiz_triggered = True # Prevent immediate re-trigger
# Reset trigger for next round
elif st.session_state.quiz_triggered and random.random() > 0.95:
     st.session_state.quiz_triggered = False

# Auto-refresh
time.sleep(1)
st.rerun()


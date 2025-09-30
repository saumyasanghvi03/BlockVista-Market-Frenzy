import streamlit as st
import pandas as pd
import yfinance as yf
import time
import random
import matplotlib.pyplot as plt

# --- Game Config ---
INITIAL_CAPITAL = 1000000  # ₹10 lakh
ROUND_DURATION = 20 * 60   # 20 minutes in seconds
NIFTY50_SYMBOLS = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS', 
                   'KOTAKBANK.NS', 'SBIN.NS', 'ITC.NS', 'ASIANPAINT.NS', 'AXISBANK.NS']
GOLD_SYMBOL = 'GC=F'  # Gold futures
OPTION_SYMBOLS = ['NIFTY_CALL', 'NIFTY_PUT']  # Mock options

# --- Quiz Questions ---
QUIZ_QUESTIONS = [
    {"question": "What does RSI stand for in technical analysis?", "options": ["Relative Strength Index", "Rapid Stock Increase", "Risk Sensitivity Indicator"], "answer": 0},
    {"question": "Which candlestick pattern signals a bullish reversal?", "options": ["Doji", "Hammer", "Shooting Star"], "answer": 1},
    {"question": "What is the primary role of SEBI in India?", "options": ["Regulate securities market", "Control inflation", "Manage foreign exchange"], "answer": 0},
]

# --- Fetch Live Prices ---
@st.cache_data(ttl=60)
def get_live_prices():
    prices = {}
    for symbol in NIFTY50_SYMBOLS + [GOLD_SYMBOL, '^NSEI']:
        try:
            data = yf.Ticker(symbol).info
            prices[symbol] = data.get('regularMarketPrice') or data.get('previousClose') or random.randint(1000, 3500)
        except:
            prices[symbol] = random.randint(1000, 3500)
    prices['NIFTY_CALL'] = prices['^NSEI'] * 1.01 if '^NSEI' in prices else 100
    prices['NIFTY_PUT'] = prices['^NSEI'] * 0.99 if '^NSEI' in prices else 100
    return prices

# --- Apply Special Event Adjustments ---
def apply_event_adjustment(prices, event_type):
    if event_type == "Flash Crash":
        return {k: v * 0.95 for k, v in prices.items()}
    elif event_type == "Bull Rally":
        return {k: v * 1.07 for k, v in prices.items()}
    elif event_type == "Banking Boost":
        for sym in ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'AXISBANK.NS']:
            prices[sym] *= 1.04
        return prices
    return prices

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

# --- CSS Styling ---
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Game Registration ---
st.sidebar.title("📝 Expo Game Entry")
player_name = st.sidebar.text_input("Enter Name")
mode = st.sidebar.radio("Select Mode", ["VIP Guest", "Student"])
if st.sidebar.button("Join Game"):
    if player_name and player_name not in st.session_state.players:
        st.session_state.players[player_name] = {
            "mode": mode,
            "capital": INITIAL_CAPITAL,
            "holdings": {},
            "pnl": 0
        }
        st.sidebar.success(f"{player_name} joined as {mode}!")
    else:
        st.sidebar.error("Name taken or empty!")

# --- Admin Controls ---
st.sidebar.title("⚙️ Admin Controls")
if st.sidebar.button("Reset Game"):
    st.session_state.players = {}
    st.session_state.game_start = time.time()
    st.session_state.event_active = False
    st.session_state.quiz_triggered = False
    st.sidebar.success("Game reset!")

# --- Trading Interface ---
st.title("BlockVista Bulls & Bears Challenge")
remaining_time = max(0, ROUND_DURATION - int(time.time() - st.session_state.game_start))
st.markdown(f"<div class='timer'>Time Remaining: {remaining_time // 60} min {remaining_time % 60} sec</div>", unsafe_allow_html=True)

# Update prices
prices = get_live_prices()
if st.session_state.event_active and time.time() < st.session_state.event_end:
    prices = apply_event_adjustment(prices, st.session_state.event_type)
st.session_state.prices = prices

st.markdown("<div class='section'>Live Market Prices</div>", unsafe_allow_html=True)
price_df = pd.DataFrame.from_dict(prices, orient='index', columns=['Price']).sort_index()
st.dataframe(price_df.style.format("{:.2f}"))

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
        ax.set_title("Portfolio Holdings")
        st.pyplot(fig)
    
    # Trade Entry
    st.markdown(f"<div class='section'>{selected_player}'s Portfolio (Mode: {player['mode']})</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'>Cash: ₹{player['capital']:.2f}</div>", unsafe_allow_html=True)
    holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
    total_value = player['capital'] + holdings_value
    pnl = total_value - INITIAL_CAPITAL
    player['pnl'] = pnl
    color = "green" if pnl >= 0 else "red"
    st.markdown(f"<div class='metric'>Portfolio Value: ₹{total_value:.2f} | P&L: <span style='color:{color}'>₹{pnl:.2f}</span></div>", unsafe_allow_html=True)
    
    action = st.radio("Action", ["Buy", "Sell"], key="action_select")
    asset_type = st.radio("Asset Type", ["Stock", "Gold", "Option"], key="asset_select")
    if asset_type == "Stock":
        symbols = [s.replace('.NS', '') for s in NIFTY50_SYMBOLS]
        symbol = st.selectbox("Stock", symbols, key="stock_select")
        symbol = symbol + '.NS'
    elif asset_type == "Gold":
        symbol = GOLD_SYMBOL
    else:
        symbol = st.selectbox("Option", OPTION_SYMBOLS, key="option_select")
    qty = st.number_input("Quantity", min_value=1, step=1, value=1, key="qty_input")
    if st.button("Place Trade"):
        price = prices.get(symbol, 0)
        cost = price * qty
        if action == "Buy":
            if player['capital'] >= cost:
                player['capital'] -= cost
                player['holdings'][symbol] = player['holdings'].get(symbol, 0) + qty
                st.success(f"Bought {qty} of {symbol} at ₹{price:.2f}!")
            else:
                st.error("Insufficient capital!")
        elif action == "Sell":
            if player['holdings'].get(symbol, 0) >= qty:
                player['capital'] += cost
                player['holdings'][symbol] -= qty
                if player['holdings'][symbol] == 0:
                    del player['holdings'][symbol]
                st.success(f"Sold {qty} of {symbol} at ₹{price:.2f}!")
            else:
                st.error("Insufficient holdings!")

# --- Leaderboard ---
lb = []
for pname, pdata in st.session_state.players.items():
    holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in pdata['holdings'].items())
    total_value = pdata['capital'] + holdings_value
    lb.append((pname, pdata['mode'], total_value, pdata['pnl']))
lb_df = pd.DataFrame(lb, columns=["Player", "Mode", "Portfolio Value", "P&L"]).sort_values("Portfolio Value", ascending=False)
st.markdown("<div class='section'>Leaderboard</div>", unsafe_allow_html=True)
st.dataframe(lb_df.style.format({"Portfolio Value": "{:.2f}", "P&L": "{:.2f}"}))

# --- End of Round ---
if remaining_time <= 0:
    st.balloons()
    st.markdown("<div class='section'>Round Over! Winners:</div>", unsafe_allow_html=True)
    for i, row in lb_df.head(3).iterrows():
        st.markdown(f"<div class='winner'>{i+1}. {row['Player']} ({row['Mode']}): ₹{row['Portfolio Value']:.2f}</div>", unsafe_allow_html=True)

# --- Special Market Events ---
if random.random() < 0.01 and not st.session_state.event_active:
    events = ["Flash Crash", "Bull Rally", "Banking Boost"]
    event_type = random.choice(events)
    st.warning(f"⚡ Market Event: {event_type}! Prices adjusted for 60 seconds!")
    st.session_state.event_active = True
    st.session_state.event_type = event_type
    st.session_state.event_end = time.time() + 60
elif st.session_state.event_active and time.time() >= st.session_state.event_end:
    st.info("Market Event Ended. Prices normalized.")
    st.session_state.event_active = False

# --- Quiz Bonus ---
if random.random() < 0.005 and not st.session_state.quiz_triggered:
    quiz = random.choice(QUIZ_QUESTIONS)
    with st.expander("📚 Quick Quiz! Answer for Bonus Capital", expanded=True):
        st.markdown(f"<div class='quiz'>{quiz['question']}</div>", unsafe_allow_html=True)
        answer = st.radio("Options:", quiz["options"], key="quiz_answer")
        if st.button("Submit Answer"):
            if quiz["options"].index(answer) == quiz["answer"]:
                bonus = random.randint(10000, 50000)
                player['capital'] += bonus
                st.success(f"Correct! +₹{bonus} bonus.")
            else:
                st.error("Incorrect. Try next time!")
            st.session_state.quiz_triggered = True
else:
    st.session_state.quiz_triggered = False

# Auto-refresh
time.sleep(1)
st.rerun()

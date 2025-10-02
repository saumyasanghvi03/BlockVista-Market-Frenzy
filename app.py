# BlockVista Market Frenzy
import streamlit as st
import pandas as pd
import yfinance as yf
import time
import random
import plotly.graph_objects as go
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
import uuid
from threading import Lock
import logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="BlockVista Market Frenzy", page_icon="📈")

# --- Constants ---
GAME_NAME = "BlockVista Market Frenzy"
INITIAL_CAPITAL = 1000000
ADMIN_PASSWORD = "100370"
NIFTY50 = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS']
CRYPTO = ['BTC-USD', 'ETH-USD', 'SOL-USD']
GOLD = 'GC=F'
NIFTY_INDEX = '^NSEI'
FUTURES = ['NIFTY-FUT']
ALL_SYMBOLS = NIFTY50 + CRYPTO + [GOLD] + FUTURES
PRE_BUILT_NEWS = [
    {"headline": "RBI cuts repo rate!", "impact": "Bull Rally"},
    {"headline": "Bank fraud shakes markets.", "impact": "Flash Crash"},
    {"headline": "{symbol} wins contract!", "impact": "Symbol Bull Run"}
]
QUIZ_QUESTIONS = [
    {"question": "What does RSI stand for?", "options": ["Relative Strength Index", "Rapid Stock Increase"], "answer": 0}
]

# --- Game State ---
@st.cache_resource
def get_game_state():
    class GameState:
        def __init__(self):
            self.players = {}
            self.game_status = "Stopped"
            self.game_start_time = 0
            self.round_duration_seconds = 10 * 60  # Reduced for faster testing
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
            self.closing_warning_triggered = False
            self.difficulty_level = 1
            self.current_margin_requirement = 0.2
            self.bid_ask_spread = 0.001
            self.slippage_threshold = 10
            self.base_slippage_rate = 0.005
            self.lock = Lock()
        def reset(self):
            with self.lock:
                base_prices = self.base_real_prices
                difficulty = self.difficulty_level
                self.__init__()
                self.base_real_prices = base_prices
                self.difficulty_level = difficulty
    return GameState()

# --- Sound Effects (Simplified) ---
def play_sound(sound_type):
    sounds = {'success': 'synth.triggerAttackRelease("C5", "8n");', 'error': 'synth.triggerAttackRelease("C3", "8n");'}
    if sound_type in sounds:
        st.components.v1.html(f'<script>if (typeof Tone !== "undefined") {{const synth = new Tone.Synth().toDestination(); {sounds[sound_type]}}}</script>', height=0)

# --- Data Fetching & Market Simulation ---
@st.cache_data(ttl=86400)
def get_daily_base_prices():
    prices = {}
    yf_symbols = NIFTY50 + CRYPTO + [GOLD, NIFTY_INDEX]
    try:
        data = yf.download(tickers=yf_symbols, period="1d", interval="1m", progress=False, threads=True)
        for symbol in yf_symbols:
            prices[symbol] = data['Close'][symbol].iloc[-1] if not data.empty and symbol in data['Close'] and not pd.isna(data['Close'][symbol].iloc[-1]) else random.uniform(10, 50000)
    except Exception as e:
        logger.error(f"Price fetch failed: {e}")
        prices.update({s: random.uniform(10, 50000) for s in yf_symbols})
    return prices

def simulate_tick_prices(last_prices):
    game_state = get_game_state()
    prices = last_prices.copy()
    volatility = game_state.volatility_multiplier
    for symbol in prices:
        if symbol not in FUTURES:
            sentiment = game_state.market_sentiment.get(symbol, 0)
            prices[symbol] = max(0.01, prices[symbol] * (1 + sentiment * 0.001 + random.uniform(-0.0005, 0.0005) * volatility))
    return prices

def calculate_derived_prices(base_prices):
    prices = base_prices.copy()
    nifty = prices.get(NIFTY_INDEX, 20000)
    prices['NIFTY-FUT'] = nifty * random.uniform(1.0, 1.005)
    return prices

@st.cache_data(ttl=3600)
def get_historical_data(symbols, period="6mo"):
    try:
        data = yf.download(tickers=symbols, period=period, progress=False, threads=True)
        return data['Close'] if not data.empty else pd.DataFrame()
    except Exception as e:
        logger.error(f"Historical data fetch failed: {e}")
        return pd.DataFrame()

# --- Game Logic ---
def calculate_slippage(player, symbol, qty, action):
    game_state = get_game_state()
    if qty <= game_state.slippage_threshold: return 1.0
    liquidity = game_state.liquidity.get(symbol, 1.0)
    slippage_mult = player.get('slippage_multiplier', 1.0)
    excess_qty = qty - game_state.slippage_threshold
    slippage_rate = (game_state.base_slippage_rate / max(0.1, liquidity)) * slippage_mult
    return max(0.9, min(1.1, 1 + (slippage_rate * excess_qty) * (1 if action == "Buy" else -1)))

def apply_event_adjustment(prices, event_type, target_symbol=None):
    game_state = get_game_state()
    prices = prices.copy()
    event_messages = {
        "Flash Crash": ("⚡ Flash Crash!", lambda p: {k: v * 0.95 for k, v in p.items()}),
        "Bull Rally": ("📈 Bull Rally!", lambda p: {k: v * 1.05 for k, v in p.items()}),
        "Symbol Bull Run": (f"🚀 {target_symbol} Bull Run!", lambda p: {k: v * (1.15 if k == target_symbol else 1.0) for k, v in p.items()})
    }
    if event_type in event_messages:
        message, price_func = event_messages[event_type]
        st.toast(message, icon="📢")
        game_state.news_feed.insert(0, f"📢 {time.strftime('%H:%M:%S')} - {message}")
        if len(game_state.news_feed) > 3: game_state.news_feed.pop()
        prices = price_func(prices)
    return prices

def format_indian_currency(n):
    if n is None: return "₹0.00"
    n = float(n)
    return f"₹{n/10000000:.2f}Cr" if abs(n) >= 10000000 else f"₹{n/100000:.2f}L" if abs(n) >= 100000 else f"₹{n:,.2f}"

def optimize_portfolio(holdings):
    symbols = [s for s in holdings.keys() if s in NIFTY50 + CRYPTO]
    if len(symbols) < 2: return None, "Need at least 2 assets."
    try:
        hist_data = get_historical_data(symbols)
        if hist_data.empty or hist_data.isnull().values.any(): return None, "Insufficient data."
        mu = expected_returns.mean_historical_return(hist_data)
        S = risk_models.sample_cov(hist_data)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        return ef.clean_weights(), ef.portfolio_performance(verbose=False)
    except Exception as e:
        logger.error(f"Portfolio optimization failed: {e}")
        return None, f"Optimization failed: {e}"

# --- UI Functions ---
def render_left_sidebar():
    game_state = get_game_state()
    with st.sidebar:
        st.title("📝 Game Entry")
        if 'player' not in st.session_state:
            player_name = st.text_input("Enter Name", key="name_input")
            mode = st.radio("Select Mode", ["Trader", "HFT"], key="mode_select")
            if st.button("Join Game"):
                if player_name.strip() and player_name not in game_state.players:
                    with game_state.lock:
                        game_state.players[player_name] = {
                            "name": player_name, "mode": mode, "capital": INITIAL_CAPITAL, "holdings": {},
                            "pnl": 0, "pending_orders": [], "algo": "Off", "custom_algos": {},
                            "slippage_multiplier": 0.5 if mode == "HFT" else 1.0, "value_history": [], "trade_timestamps": [], "hft_trade_count": 0
                        }
                        game_state.transactions[player_name] = []
                    st.session_state.player = player_name
                    st.session_state.session_id = str(uuid.uuid4())
                    st.rerun()
                else:
                    st.error("Name invalid or taken!")
        else:
            st.success(f"Logged in as {st.session_state.player}")
            if st.button("Logout"):
                del st.session_state.player
                del st.session_state.session_id
                st.rerun()
        st.title("🔐 Admin Login")
        password = st.text_input("Enter Password", type="password")
        if password == ADMIN_PASSWORD:
            st.session_state.role = 'admin'
            st.success("Admin Access Granted")
        elif password:
            st.error("Incorrect Password")
        if st.session_state.get('role') == 'admin':
            st.title("⚙️ Admin Controls")
            status_color = {"Running": "green", "Stopped": "red", "Finished": "blue"}
            status_text = f"Game Status: {game_state.game_status}"
            if game_state.game_status == "Running":
                remaining = max(0, game_state.round_duration_seconds - int(time.time() - game_state.game_start_time))
                status_text += f" | Time Left: {remaining // 60:02d}:{remaining % 60:02d}"
            st.markdown(f"<p style='color: {status_color.get(game_state.game_status, 'black')}'>{status_text}</p>", unsafe_allow_html=True)
            game_duration = st.number_input("Game Duration (min)", min_value=1, value=10, disabled=(game_state.game_status == "Running"))
            game_state.volatility_multiplier = st.slider("Volatility", 0.5, 3.0, 1.0, 0.5)
            if st.button("▶️ Start Game", type="primary"):
                if game_state.players:
                    with game_state.lock:
                        game_state.game_status = "Running"
                        game_state.game_start_time = time.time()
                        game_state.round_duration_seconds = game_duration * 60
                        game_state.futures_expiry_time = time.time() + (game_state.round_duration_seconds / 2)
                        game_state.closing_warning_triggered = False
                    st.toast("Game Started!", icon="🎉")
                    st.rerun()
                else:
                    st.warning("Add at least one player.")
            if st.button("⏸️ Stop Game"):
                with game_state.lock:
                    game_state.game_status = "Stopped"
                st.toast("Game Paused!", icon="⏸️")
                st.rerun()
            if st.button("🔄 Reset Game"):
                with game_state.lock:
                    game_state.reset()
                st.session_state.clear()
                st.toast("Game Reset.", icon="🔄")
                st.rerun()

def render_right_sidebar(prices):
    game_state = get_game_state()
    with st.sidebar:
        if 'player' in st.session_state:
            player = game_state.players.get(st.session_state.player, {})
            st.title(f"{st.session_state.player}'s Stats")
            holdings_value = sum(prices.get(s, 0) * q for s, q in player.get('holdings', {}).items())
            total_value = player.get('capital', 0) + holdings_value
            player['pnl'] = total_value - INITIAL_CAPITAL
            c1, c2 = st.columns(2)
            c1.metric("Cash", format_indian_currency(player.get('capital', 0)))
            c2.metric("P&L", format_indian_currency(player['pnl']))

def render_main_interface(prices):
    game_state = get_game_state()
    st.title(f"📈 {GAME_NAME}")
    if game_state.game_status == "Running":
        remaining = max(0, game_state.round_duration_seconds - int(time.time() - game_state.game_start_time))
        if remaining <= 30 and not game_state.closing_warning_triggered:
            with game_state.lock:
                game_state.closing_warning_triggered = True
        if remaining == 0:
            with game_state.lock:
                game_state.game_status = "Finished"
        st.markdown(f"**Time Remaining: {remaining // 60:02d}:{remaining % 60:02d}**")
    elif game_state.game_status == "Stopped":
        st.info("Game paused. Press 'Start Game' to begin.")
    elif game_state.game_status == "Finished":
        st.success("Game finished! See final leaderboard.")
    if st.session_state.get('role') == 'admin':
        render_global_views(prices, is_admin=True)
    elif 'player' in st.session_state:
        col1, col2 = st.columns([1.5, 1])
        with col1:
            render_trade_execution_panel(prices)
        with col2:
            render_global_views(prices)
    else:
        st.info("Welcome to BlockVista! Join from the left sidebar.")
        render_global_views(prices)

def render_global_views(prices, is_admin=False):
    game_state = get_game_state()
    with st.container(border=True):
        st.subheader("Global Market View")
        sentiments = [s for s in game_state.market_sentiment.values() if s != 0]
        overall_sentiment = np.clip((np.mean(sentiments) + 5) * 10, 0, 100) if sentiments else 0
        st.markdown("##### Market Sentiment")
        st.progress(int(overall_sentiment))
        st.subheader("📰 Live News Feed")
        news_feed = game_state.news_feed
        if news_feed:
            for news in news_feed: st.info(news)
        else:
            st.info("No market news.")
        st.subheader("Live Player Standings")
        render_leaderboard(prices)
        st.subheader("Live Market Feed")
        prices_df = pd.DataFrame(prices.items(), columns=['Symbol', 'Price'])
        prices_df['Change'] = 0.0 if len(game_state.price_history) < 2 else prices_df['Symbol'].map(lambda s: prices[s] - game_state.price_history[-2].get(s, prices[s]))
        st.dataframe(prices_df.style.apply(lambda x: ['color: green' if v > 0 else 'color: red' if v < 0 else '' for v in x], subset=['Change']).format({'Price': format_indian_currency, 'Change': lambda v: f"{format_indian_currency(v) if v != 0 else '-'}"}), use_container_width=True, hide_index=True)

def render_trade_execution_panel(prices):
    game_state = get_game_state()
    with st.container(border=True):
        st.subheader("Trade Execution Panel")
        player_name = st.session_state.get('player')
        if not player_name or player_name not in game_state.players:
            st.warning("Join the game to access your trading terminal.")
            return
        player = game_state.players[player_name]
        st.markdown(f"**{player_name}'s Terminal (Mode: {player['mode']})**")
        holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        player['pnl'] = total_value - INITIAL_CAPITAL
        c1, c2 = st.columns(2)
        c1.metric("Cash", format_indian_currency(player['capital']))
        c2.metric("P&L", format_indian_currency(player['pnl']))
        tabs = ["👨‍💻 Trade", "📂 History"]
        tab1, tab2 = st.tabs(tabs)
        disabled = game_state.game_status != "Running"
        with tab1:
            render_market_order_ui(player_name, player, prices, disabled)
            st.subheader("💼 Portfolio")
            if player['holdings']:
                holdings_data = [{"Symbol": s, "Qty": q, "Value": prices.get(s, 0) * q} for s, q in player['holdings'].items()]
                st.dataframe(pd.DataFrame(holdings_data).style.format({"Value": format_indian_currency}), use_container_width=True)
            else:
                st.info("No holdings.")
        with tab2:
            st.subheader("Transaction History")
            if game_state.transactions.get(player_name):
                trans_df = pd.DataFrame(game_state.transactions[player_name], columns=["Time", "Action", "Symbol", "Qty", "Price", "Total"])
                st.dataframe(trans_df.style.format(formatter={"Price": format_indian_currency, "Total": format_indian_currency}), use_container_width=True)
            else:
                st.info("No transactions.")

def render_market_order_ui(player_name, player, prices, disabled):
    asset_types = ["Stock", "Crypto", "Gold", "Futures"]
    asset_type = st.radio("Asset Type", asset_types, horizontal=True, key=f"market_asset_{player_name}", disabled=disabled)
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = {
            "Stock": st.selectbox("Stock", [s.replace('.NS', '') for s in NIFTY50], key=f"market_stock_{player_name}", disabled=disabled) + '.NS',
            "Crypto": st.selectbox("Cryptocurrency", CRYPTO, key=f"market_crypto_{player_name}", disabled=disabled),
            "Gold": GOLD,
            "Futures": st.selectbox("Futures", FUTURES, key=f"market_futures_{player_name}", disabled=disabled)
        }.get(asset_type, GOLD)
    with col2:
        qty = st.number_input("Quantity", min_value=1, step=1, value=1, key=f"market_qty_{player_name}", disabled=disabled)
    mid_price = prices.get(symbol, 0)
    ask_price = mid_price * (1 + game_state.bid_ask_spread / 2)
    bid_price = mid_price * (1 - game_state.bid_ask_spread / 2)
    st.info(f"Bid: {format_indian_currency(bid_price)} | Ask: {format_indian_currency(ask_price)}")
    c1, c2 = st.columns(2)
    for action, key, type_ in [("Buy", f"buy_{player_name}", "primary"), ("Sell", f"sell_{player_name}", "secondary")]:
        if st.button(f"{action} {qty}", key=key, disabled=disabled, type=type_):
            if execute_trade(player_name, player, action, symbol, qty, prices):
                play_sound('success')
            else:
                play_sound('error')
            st.rerun()

def execute_trade(player_name, player, action, symbol, qty, prices, order_type="Market"):
    game_state = get_game_state()
    mid_price = prices.get(symbol, 0)
    if mid_price == 0: return False
    trade_price = mid_price * (1 + game_state.bid_ask_spread / 2 if action == "Buy" else 1 - game_state.bid_ask_spread / 2) * calculate_slippage(player, symbol, qty, action)
    cost = trade_price * qty
    trade_executed = False
    with game_state.lock:
        if action == "Buy" and player['capital'] >= cost:
            player['capital'] -= cost
            player['holdings'][symbol] = player['holdings'].get(symbol, 0) + qty
            trade_executed = True
        elif action == "Sell":
            current_qty = player['holdings'].get(symbol, 0)
            if current_qty >= qty:
                player['capital'] += cost
                player['holdings'][symbol] -= qty
                if player['holdings'].get(symbol, 0) == 0:
                    del player['holdings'][symbol]
                trade_executed = True
        if trade_executed:
            game_state.market_sentiment[symbol] = game_state.market_sentiment.get(symbol, 0) + (qty / 50) * (1 if action == "Buy" else -1)
            log_transaction(player_name, f"{order_type} {action}", symbol, qty, trade_price, cost)
    return trade_executed

def log_transaction(player_name, action, symbol, qty, price, total):
    game_state = get_game_state()
    with game_state.lock:
        game_state.transactions.setdefault(player_name, []).append([time.strftime("%H:%M:%S"), action, symbol, qty, price, total])
        if len(game_state.transactions[player_name]) > 100:
            game_state.transactions[player_name] = game_state.transactions[player_name][-100:]

def render_leaderboard(prices):
    game_state = get_game_state()
    lb = []
    for name, p in game_state.players.items():
        portfolio_value = p['capital'] + sum(prices.get(s, 0) * q for s, q in p['holdings'].items())
        lb.append((name, p['mode'], portfolio_value, p['pnl']))
    if lb:
        lb_df = pd.DataFrame(lb, columns=["Player", "Mode", "Portfolio Value", "P&L"]).sort_values("Portfolio Value", ascending=False).reset_index(drop=True)
        st.dataframe(lb_df.style.format(formatter={"Portfolio Value": format_indian_currency, "P&L": format_indian_currency}), use_container_width=True, hide_index=True)
        if game_state.game_status == "Finished" and not game_state.auto_square_off_complete:
            auto_square_off_positions(prices)
            with game_state.lock:
                game_state.auto_square_off_complete = True
            st.rerun()
        if game_state.game_status == "Finished":
            st.balloons()
            winner = lb_df.iloc[0]
            st.success(f"🎉 Winner: {winner['Player']}!")
            st.metric("🏆 Final Portfolio", format_indian_currency(winner['Portfolio Value']))

def auto_square_off_positions(prices):
    game_state = get_game_state()
    st.info("End of game: Auto-squaring positions...")
    with game_state.lock:
        for name, player in game_state.players.items():
            for symbol, qty in list(player['holdings'].items()):
                closing_price = prices.get(symbol, 0)
                value = closing_price * qty
                action = "Sell" if qty > 0 else "Buy"
                player['capital'] += value if qty > 0 else -value
                log_transaction(name, f"Auto-Squareoff ({action})", symbol, abs(qty), closing_price, value)
                del player['holdings'][symbol]

def handle_futures_expiry(prices):
    game_state = get_game_state()
    if not game_state.futures_settled and game_state.futures_expiry_time > 0 and time.time() > game_state.futures_expiry_time:
        st.warning("FUTURES EXPIRED!")
        with game_state.lock:
            for name, player in game_state.players.items():
                for symbol in FUTURES:
                    if symbol in player['holdings']:
                        qty = player['holdings'][symbol]
                        settlement_price = prices.get(NIFTY_INDEX, 0)
                        pnl = (settlement_price - prices.get(symbol, 0)) * qty
                        player['capital'] += pnl
                        log_transaction(name, "Futures Settlement", symbol, qty, settlement_price, pnl)
                        del player['holdings'][symbol]
            game_state.futures_settled = True

def run_game_tick(prices):
    game_state = get_game_state()
    if game_state.game_status != "Running": return prices
    with game_state.lock:
        for symbol in game_state.market_sentiment:
            game_state.market_sentiment[symbol] *= 0.95
        if not game_state.event_active and random.random() < 0.05:
            news = random.choice(PRE_BUILT_NEWS)
            headline = news['headline']
            target_symbol = random.choice(NIFTY50) if "{symbol}" in headline else None
            if target_symbol:
                headline = headline.format(symbol=target_symbol.replace(".NS", ""))
            game_state.news_feed.insert(0, f"📢 {time.strftime('%H:%M:%S')} - {headline}")
            if len(game_state.news_feed) > 3: game_state.news_feed.pop()
            game_state.event_type = news['impact']
            game_state.event_target_symbol = target_symbol
            game_state.event_active = True
            game_state.event_end = time.time() + 30
            st.toast(f"⚡ Market Event!", icon="🎉")
        if game_state.event_active and time.time() >= game_state.event_end:
            game_state.event_active = False
        if game_state.event_active:
            prices = apply_event_adjustment(prices, game_state.event_type, game_state.event_target_symbol)
        handle_futures_expiry(prices)
        for player in game_state.players.values():
            holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
            player['value_history'].append(player['capital'] + holdings_value)
            if len(player['value_history']) > 100:
                player['value_history'] = player['value_history'][-100:]
        game_state.prices = prices
        game_state.price_history.append(prices.copy())
        if len(game_state.price_history) > 50:
            game_state.price_history.pop(0)
    return prices

def main():
    game_state = get_game_state()
    if not game_state.base_real_prices:
        with game_state.lock:
            game_state.base_real_prices = get_daily_base_prices()
            game_state.prices = game_state.base_real_prices.copy()
    st.markdown("<style>.stApp {max-width: 100%; padding: 1rem;}</style>", unsafe_allow_html=True)
    if 'role' not in st.session_state:
        st.session_state.role = 'player'
    render_left_sidebar()
    render_right_sidebar(game_state.prices)
    render_main_interface(game_state.prices)
    if game_state.game_status == "Running":
        game_state.prices = simulate_tick_prices(game_state.prices)
        game_state.prices = calculate_derived_prices(game_state.prices)
        game_state.prices = run_game_tick(game_state.prices)
        with game_state.lock:
            game_state.price_history.append(game_state.prices.copy())
            if len(game_state.price_history) > 50:
                game_state.price_history.pop(0)

if __name__ == "__main__":
    main()


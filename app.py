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
ADMIN_PASSWORD = "100370" # Set your admin password here


# --- Pre-built News Headlines ---
PRE_BUILT_NEWS = [
    # India Specific
    {"headline": "Breaking: RBI unexpectedly cuts repo rate by 25 basis points!", "impact": "Bull Rally"},
    {"headline": "Government announces major infrastructure spending package, boosting banking stocks.", "impact": "Banking Boost"},
    {"headline": "Shocking fraud uncovered at a major private bank, sending shockwaves through the financial sector.", "impact": "Flash Crash"},
    {"headline": "Indian tech firm announces breakthrough in AI, sparking a rally in tech stocks.", "impact": "Sector Rotation"},
    {"headline": "FIIs show renewed interest in Indian equities, leading to broad-based buying.", "impact": "Bull Rally"},
    {"headline": "SEBI announces stricter margin rules for derivatives, market turns cautious.", "impact": "Flash Crash"},
    {"headline": "Monsoon forecast revised upwards, boosting rural demand expectations.", "impact": "Bull Rally"},
    
    # Global News
    {"headline": "Global News: US inflation data comes in hotter than expected, spooking global markets.", "impact": "Flash Crash"},
    {"headline": "Global News: European Central Bank signals a dovish stance, boosting liquidity.", "impact": "Bull Rally"},
    {"headline": "Global News: Major supply chain disruption reported in Asia, affecting global trade.", "impact": "Volatility Spike"},
    {"headline": "Global News: President Trump announces new trade tariffs, increasing market uncertainty.", "impact": "Volatility Spike"},
    {"headline": "Global News: President Trump tweets about 'tremendous' economic growth, boosting investor confidence.", "impact": "Bull Rally"},

    # Symbol Specific
    {"headline": "{symbol} secures a massive government contract, sending its stock soaring!", "impact": "Symbol Bull Run"},
    {"headline": "Regulatory probe launched into {symbol} over accounting irregularities.", "impact": "Symbol Crash"},
    {"headline": "{symbol} announces a surprise stock split, shares to adjust at market open.", "impact": "Stock Split"},
    {"headline": "{symbol} declares a special dividend for all shareholders.", "impact": "Dividend"},
    {"headline": "High short interest in {symbol} triggers a potential short squeeze!", "impact": "Short Squeeze"},
]

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
        self.base_real_prices = {} # Stores prices fetched once per day
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

    def reset(self):
        """Resets the game to its initial state, but keeps the daily base prices and difficulty."""
        base_prices = self.base_real_prices
        difficulty = self.difficulty_level
        self.__init__()
        self.base_real_prices = base_prices
        self.difficulty_level = difficulty


@st.cache_resource
def get_game_state():
    """Returns the singleton GameState object, ensuring all users share the same state."""
    return GameState()


# --- Sound Effects ---
def play_sound(sound_type):
    """Embeds HTML to play a sound effect using JavaScript and Tone.js."""
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
    elif sound_type == 'opening_bell':
        js = """
        <script>
            if (typeof Tone !== 'undefined') {
                const synth = new Tone.Synth().toDestination();
                const now = Tone.now();
                synth.triggerAttackRelease("C4", "8n", now);
                synth.triggerAttackRelease("E4", "8n", now + 0.2);
                synth.triggerAttackRelease("G4", "8n", now + 0.4);
            }
        </script>
        """
    elif sound_type == 'closing_warning':
        js = """
        <script>
            if (typeof Tone !== 'undefined') {
                const synth = new Tone.Synth().toDestination();
                const now = Tone.now();
                synth.triggerAttackRelease("G5", "16n", now);
                synth.triggerAttackRelease("G5", "16n", now + 0.3);
            }
        </script>
        """
    elif sound_type == 'final_bell':
        js = """
        <script>
            if (typeof Tone !== 'undefined') {
                const synth = new Tone.Synth().toDestination();
                synth.triggerAttackRelease("C4", "2n");
            }
        </script>
        """
    st.components.v1.html(js, height=0)


def announce_news(headline):
    """Embeds HTML to play a TTS announcement of the news headline."""
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


# --- Quiz Questions ---
QUIZ_QUESTIONS = [
    {"question": "What does RSI stand for in technical analysis?", "options": ["Relative Strength Index", "Rapid Stock Increase", "Risk Sensitivity Indicator"], "answer": 0},
    {"question": "Which candlestick pattern signals a bullish reversal?", "options": ["Doji", "Hammer", "Shooting Star"], "answer": 1},
    {"question": "What is the primary role of SEBI in India?", "options": ["Regulate securities market", "Control inflation", "Manage foreign exchange"], "answer": 0},
]

# --- Data Fetching & Market Simulation ---
@st.cache_data(ttl=86400) # Cache for 24 hours
def get_daily_base_prices():
    """Fetches real-world prices from yfinance ONCE per day to use as a baseline."""
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

def simulate_tick_prices(last_prices):
    """Applies small, random fluctuations and market sentiment to prices."""
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
    """Calculates prices for simulated assets like Futures, Options, and ETFs based on the current simulated index prices."""
    game_state = get_game_state()
    derived_prices = base_prices.copy()

    nifty_price = derived_prices.get(NIFTY_INDEX_SYMBOL, 20000)
    banknifty_price = derived_prices.get(BANKNIFTY_INDEX_SYMBOL, 45000)
    
    # Mock options
    derived_prices['NIFTY_CALL'] = nifty_price * 1.02
    derived_prices['NIFTY_PUT'] = nifty_price * 0.98
    
    # Futures with a random basis
    derived_prices['NIFTY-FUT'] = nifty_price * random.uniform(1.0, 1.005)
    derived_prices['BANKNIFTY-FUT'] = banknifty_price * random.uniform(1.0, 1.005)
    
    # Leveraged ETFs
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

# --- Game Logic Functions ---
def calculate_slippage(player, symbol, qty, action):
    game_state = get_game_state()
    liquidity_level = game_state.liquidity.get(symbol, 1.0)
    if qty <= game_state.slippage_threshold: return 1.0
    
    slippage_multiplier = player.get('slippage_multiplier', 1.0)
    
    excess_qty = qty - game_state.slippage_threshold
    slippage_rate = (game_state.base_slippage_rate / max(0.1, liquidity_level)) * slippage_multiplier
    slippage_mult = 1 + (slippage_rate * excess_qty) * (1 if action == "Buy" else -1)
    return max(0.9, min(1.1, slippage_mult))

def apply_event_adjustment(prices, event_type, target_symbol=None):
    adjusted_prices = prices.copy()
    game_state = get_game_state()
    if event_type == "Flash Crash":
        adjusted_prices = {k: v * random.uniform(0.95, 0.98) for k, v in adjusted_prices.items()}
    elif event_type == "Bull Rally":
        adjusted_prices = {k: v * random.uniform(1.02, 1.05) for k, v in adjusted_prices.items()}
    elif event_type == "Banking Boost":
        for sym in ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'AXISBANK.NS']:
            if sym in adjusted_prices: adjusted_prices[sym] *= 1.07
    elif event_type == "Sector Rotation":
        for sym in ['HDFCBANK.NS', 'ICICIBANK.NS']:
            if sym in adjusted_prices: adjusted_prices[sym] *= 0.95
        for sym in ['INFY.NS', 'TCS.NS']:
            if sym in adjusted_prices: adjusted_prices[sym] *= 1.10
    elif event_type == "Symbol Bull Run" and target_symbol:
        adjusted_prices[target_symbol] *= 1.15
    elif event_type == "Symbol Crash" and target_symbol:
        adjusted_prices[target_symbol] *= 0.85
    elif event_type == "Short Squeeze" and target_symbol:
        adjusted_prices[target_symbol] *= 1.25
    elif event_type == "Volatility Spike":
         # This event is now handled by the volatility_multiplier in the main tick
        pass
    elif event_type == "Stock Split" and target_symbol:
        adjusted_prices[target_symbol] /= 2
        for player in game_state.players.values():
            if target_symbol in player['holdings']:
                player['holdings'][target_symbol] *= 2
    elif event_type == "Dividend" and target_symbol:
        dividend_per_share = adjusted_prices[target_symbol] * 0.01 # 1% dividend
        for player in game_state.players.values():
            if target_symbol in player['holdings'] and player['holdings'][target_symbol] > 0:
                dividend_received = dividend_per_share * player['holdings'][target_symbol]
                player['capital'] += dividend_received
                log_transaction(player['name'], "Dividend", target_symbol, player['holdings'][target_symbol], dividend_per_share, dividend_received)
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
    
    # Robustly select the first column, regardless of its name
    price_series = hist.iloc[:, 0]

    if indicator == "Price Change % (5-day)":
        if len(hist) < 6: return None
        price_now = price_series.iloc[-1]
        price_then = price_series.iloc[-6]
        return ((price_now - price_then) / price_then) * 100
    
    elif indicator == "SMA Crossover (10/20)":
        if len(hist) < 20: return None
        sma_10 = price_series.rolling(window=10).mean().iloc[-1]
        sma_20 = price_series.rolling(window=20).mean().iloc[-1]
        return sma_10 - sma_20
        
    elif indicator == "Price Change % (30-day)":
        if len(hist) < 31: return None
        price_now = price_series.iloc[-1]
        price_then = price_series.iloc[-31]
        return ((price_now - price_then) / price_then) * 100
    return None

# --- UI Functions ---
def render_sidebar():
    game_state = get_game_state()
    
    if 'player' not in st.query_params:
        st.sidebar.title("📝 Game Entry")
        player_name = st.sidebar.text_input("Enter Name", key="name_input")
        mode = st.sidebar.radio("Select Mode", ["Trader", "HFT", "HNI"], key="mode_select")
        
        if st.sidebar.button("Join Game"):
            if player_name and player_name.strip() and player_name not in game_state.players:
                starting_capital = INITIAL_CAPITAL * 5 if mode == "HNI" else INITIAL_CAPITAL
                game_state.players[player_name] = {
                    "name": player_name, "mode": mode, "capital": starting_capital, 
                    "holdings": {}, "pnl": 0, "leverage": 1.0, "margin_calls": 0, 
                    "pending_orders": [], "algo": "Off", "custom_algos": {},
                    "slippage_multiplier": 0.5 if mode == "HFT" else 1.0,
                    "value_history": [], "trade_timestamps": []
                }
                game_state.transactions[player_name] = []
                st.query_params["player"] = player_name
                st.rerun()
            else: st.sidebar.error("Name is invalid or already taken!")
    else:
        st.sidebar.success(f"Logged in as {st.query_params['player']}")
        if st.sidebar.button("Logout"):
            st.query_params.clear()
            st.rerun()

    st.sidebar.title("🔐 Admin Login")
    password = st.sidebar.text_input("Enter Password", type="password")

    if password == ADMIN_PASSWORD:
        st.session_state.role = 'admin'
        st.sidebar.success("Admin Access Granted")

    if st.session_state.get('role') == 'admin':
        if st.sidebar.button("Logout Admin"):
            del st.session_state['role']
            st.rerun()

        st.sidebar.title("⚙️ Admin Controls")
        
        default_duration_minutes = int(getattr(game_state, 'round_duration_seconds', 1200) / 60)
        game_duration_minutes = st.sidebar.number_input("Game Duration (minutes)", min_value=1, value=default_duration_minutes, disabled=(game_state.game_status == "Running"))

        game_state.volatility_multiplier = st.sidebar.slider("Market Volatility", 0.5, 5.0, getattr(game_state, 'volatility_multiplier', 1.0), 0.5)
        
        difficulty_index = getattr(game_state, 'difficulty_level', 1) - 1
        game_state.difficulty_level = st.sidebar.selectbox("Game Difficulty", [1, 2, 3], index=difficulty_index, format_func=lambda x: f"Level {x}", disabled=(game_state.game_status == "Running"))

        game_state.current_margin_requirement = st.sidebar.slider("Margin Requirement (%)", 10, 50, int(getattr(game_state, 'current_margin_requirement', 0.2) * 100), 5) / 100.0


        st.sidebar.markdown("---")
        st.sidebar.subheader("Broadcast News")
        
        news_options = {news['headline']: news['impact'] for news in PRE_BUILT_NEWS}
        news_to_trigger = st.sidebar.selectbox("Select News to Publish", ["None"] + list(news_options.keys()))
        
        target_symbol = None
        if news_to_trigger and "{symbol}" in news_to_trigger:
            target_symbol = st.sidebar.selectbox("Target Symbol", [s.replace(".NS", "") for s in NIFTY50_SYMBOLS]) + ".NS"

        if news_to_trigger != "None":
            st.sidebar.info(f"Impact: {news_options[news_to_trigger]}")

        if st.sidebar.button("Publish News"):
            if news_to_trigger != "None":
                selected_news = next((news for news in PRE_BUILT_NEWS if news["headline"] == news_to_trigger), None)
                if selected_news:
                    headline = selected_news['headline'].format(symbol=target_symbol.replace(".NS","") if target_symbol else "")
                    game_state.news_feed.insert(0, f"📢 {time.strftime('%H:%M:%S')} - {headline}")
                    if len(game_state.news_feed) > 5: game_state.news_feed.pop()
                    game_state.event_type = selected_news['impact']
                    game_state.event_target_symbol = target_symbol
                    game_state.event_active = True
                    game_state.event_end = time.time() + 60
                    st.toast(f"News Published!", icon="📰"); announce_news(headline)
                    st.rerun()

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
        if st.sidebar.button("▶️ Start Game", type="primary"):
            if game_state.players:
                game_state.game_status = "Running"; game_state.game_start_time = time.time()
                game_state.round_duration_seconds = game_duration_minutes * 60
                game_state.futures_expiry_time = time.time() + (game_state.round_duration_seconds / 2)
                st.toast("Game Started!", icon="🎉"); st.rerun()
            else: st.sidebar.warning("Add at least one player to start.")
        if st.sidebar.button("⏸️ Stop Game"):
            game_state.game_status = "Stopped"; st.toast("Game Paused!", icon="⏸️"); st.rerun()
        if st.sidebar.button("🔄 Reset Game"):
            game_state.reset(); st.toast("Game has been reset.", icon="🔄"); st.rerun()
            
    elif password: st.sidebar.error("Incorrect Password")

def render_main_interface(prices):
    game_state = get_game_state()
    
    # Use a container to wrap the entire interface
    with st.container():
        st.title(f"📈 {GAME_NAME}")
        
        # Inject Tone.js script - only once
        st.components.v1.html('<script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.7.77/Tone.js"></script>', height=0)

        if game_state.game_status == "Running":
            remaining_time = max(0, game_state.round_duration_seconds - int(time.time() - game_state.game_start_time))
            if remaining_time == 0: 
                if game_state.game_status != "Finished": 
                    play_sound('final_bell')
                game_state.game_status = "Finished"
            
            if remaining_time <= 30 and not getattr(game_state, 'closing_warning_triggered', False):
                play_sound('closing_warning')
                game_state.closing_warning_triggered = True
            st.markdown(f"**Time Remaining: {remaining_time // 60:02d}:{remaining_time % 60:02d}** | **Difficulty: Level {getattr(game_state, 'difficulty_level', 1)}**")
        elif game_state.game_status == "Stopped": 
            st.info("Game is paused. Press 'Start Game' to begin.")
        elif game_state.game_status == "Finished": 
            st.success("Game has finished! See the final leaderboard below.")

        # Only render one view based on user state
        if st.session_state.get('role') == 'admin':
            render_global_views(prices, is_admin=True)
        elif 'player' in st.query_params:
            col1, col2 = st.columns([1, 1])
            with col1: 
                render_trade_execution_panel(prices)
            with col2: 
                render_global_views(prices)
        else:
            st.info("Welcome to the BlockVista Market Frenzy! Please join the game from the sidebar to start trading.")
            render_global_views(prices)


def render_global_views(prices, is_admin=False):
    # Use a container to prevent duplication
    with st.container():
        st.subheader("Global Market View")
        render_market_sentiment_meter()
        
        st.markdown("---")
        st.subheader("📰 Live News Feed")
        game_state = get_game_state()
        news_feed = getattr(game_state, 'news_feed', [])
        if news_feed:
            for news in news_feed:
                st.info(news)
        else:
            st.info("No market news at the moment.")

        st.markdown("---")
        st.subheader("Live Player Standings")
        render_leaderboard(prices)
        
        if is_admin:
            st.markdown("---")
            st.subheader("Live Player Performance")
            render_admin_performance_chart()

        st.markdown("---")
        st.subheader("Live Market Feed")
        render_live_market_table(prices)

def render_market_sentiment_meter():
    game_state = get_game_state()
    sentiments = [s for s in game_state.market_sentiment.values() if s != 0]
    if not sentiments:
        overall_sentiment = 0
    else:
        overall_sentiment = np.mean(sentiments)
    
    # Normalize sentiment to 0-100 scale
    normalized_sentiment = np.clip((overall_sentiment + 5) * 10, 0, 100)
    
    st.markdown("##### Market Sentiment")
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        st.write("<p style='text-align: right; margin-top: -5px;'>Fear</p>", unsafe_allow_html=True)
    with col2:
        st.progress(int(normalized_sentiment))
    with col3:
        st.write("<p style='margin-top: -5px;'>Greed</p>", unsafe_allow_html=True)

def render_admin_performance_chart():
    game_state = get_game_state()
    if not game_state.players:
        st.info("No players have joined yet.")
        return
        
    chart_data = {}
    for name, player_data in game_state.players.items():
        if player_data.get('value_history'):
            chart_data[name] = player_data['value_history']
            
    if chart_data:
        df = pd.DataFrame(chart_data)
        st.line_chart(df)
    else:
        st.info("No trading activity yet to display.")


def render_trade_execution_panel(prices):
    game_state = get_game_state()
    
    with st.container(border=True):
        st.subheader("Trade Execution Panel")
        acting_player = st.query_params.get("player")
        if not acting_player or acting_player not in game_state.players:
            st.warning("Please join the game to access your trading terminal.")
            return
        
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
        is_trade_disabled = game_state.game_status != "Running"

        with tab1: render_trade_interface(acting_player, player, prices, is_trade_disabled)
        with tab2: render_algo_trading_tab(acting_player, player, is_trade_disabled)
        with tab3: render_transaction_history(acting_player)
        with tab4: render_strategy_tab(player)

def render_trade_interface(player_name, player, prices, disabled):
    order_type_tabs = ["Market", "Limit", "Stop-Loss"]
    market_tab, limit_tab, stop_loss_tab = st.tabs(order_type_tabs)

    with market_tab:
        render_market_order_ui(player_name, player, prices, disabled)

    with limit_tab:
        render_limit_order_ui(player_name, player, prices, disabled)
    
    with stop_loss_tab:
        render_stop_loss_order_ui(player_name, player, prices, disabled)

    st.markdown("---")
    render_current_holdings(player, prices)
    render_pending_orders(player)

def render_market_order_ui(player_name, player, prices, disabled):
    asset_types = ["Stock", "Crypto", "Gold", "Futures", "Leveraged ETF", "Option"]
    asset_type = st.radio("Asset Type", asset_types, horizontal=True, key=f"market_asset_{player_name}", disabled=disabled)
    
    if asset_type == "Futures" and getattr(get_game_state(), 'futures_expiry_time', 0) > 0:
        expiry_remaining = max(0, get_game_state().futures_expiry_time - time.time())
        st.warning(f"Futures expire in **{int(expiry_remaining // 60)}m {int(expiry_remaining % 60)}s**")

    col1, col2 = st.columns([2, 1])
    with col1:
        if asset_type == "Stock": symbol_choice = st.selectbox("Stock", [s.replace('.NS', '') for s in NIFTY50_SYMBOLS], key=f"market_stock_{player_name}", disabled=disabled) + '.NS'
        elif asset_type == "Crypto": symbol_choice = st.selectbox("Cryptocurrency", CRYPTO_SYMBOLS, key=f"market_crypto_{player_name}", disabled=disabled)
        elif asset_type == "Gold": symbol_choice = GOLD_SYMBOL
        elif asset_type == "Futures": symbol_choice = st.selectbox("Futures", FUTURES_SYMBOLS, key=f"market_futures_{player_name}", disabled=disabled)
        elif asset_type == "Leveraged ETF": symbol_choice = st.selectbox("Leveraged ETF", LEVERAGED_ETFS, key=f"market_letf_{player_name}", disabled=disabled)
        else: symbol_choice = st.selectbox("Option", OPTION_SYMBOLS, key=f"market_option_{player_name}", disabled=disabled)
    with col2:
        qty = st.number_input("Quantity", min_value=1, step=1, value=1, key=f"market_qty_{player_name}", disabled=disabled)
    
    mid_price = prices.get(symbol_choice, 0)
    ask_price = mid_price * (1 + get_game_state().bid_ask_spread / 2)
    bid_price = mid_price * (1 - get_game_state().bid_ask_spread / 2)
    st.info(f"Bid: {format_indian_currency(bid_price)} | Ask: {format_indian_currency(ask_price)}")

    b1, b2, b3 = st.columns(3)
    if b1.button(f"Buy {qty} at Ask", key=f"buy_{player_name}", use_container_width=True, disabled=disabled, type="primary"): 
        if execute_trade(player_name, player, "Buy", symbol_choice, qty, prices): play_sound('success')
        else: play_sound('error')
        st.rerun()
    if b2.button(f"Sell {qty} at Bid", key=f"sell_{player_name}", use_container_width=True, disabled=disabled): 
        if execute_trade(player_name, player, "Sell", symbol_choice, qty, prices): play_sound('success')
        else: play_sound('error')
        st.rerun()
    if b3.button(f"Short {qty} at Bid", key=f"short_{player_name}", use_container_width=True, disabled=disabled): 
        if execute_trade(player_name, player, "Short", symbol_choice, qty, prices): play_sound('success')
        else: play_sound('error')
        st.rerun()

def render_limit_order_ui(player_name, player, prices, disabled):
    st.write("Set a price to automatically buy or sell an asset.")
    # UI for Limit order
    pass # Placeholder for Limit Order UI

def render_stop_loss_order_ui(player_name, player, prices, disabled):
    st.write("Set a price to automatically sell an asset if it drops, to limit losses.")
    # UI for Stop-Loss order
    pass # Placeholder for Stop-Loss Order UI

def render_pending_orders(player):
    st.subheader("🕒 Pending Orders")
    if player['pending_orders']:
        orders_df = pd.DataFrame(player['pending_orders'])
        st.dataframe(orders_df, use_container_width=True)
    else:
        st.info("No pending orders.")

def render_algo_trading_tab(player_name, player, disabled):
    st.subheader("Automated Trading Strategies")
    default_strats = ["Off", "Momentum Trader", "Mean Reversion", "Volatility Breakout", "Value Investor"]
    custom_strats = list(player.get('custom_algos', {}).keys()); all_strats = default_strats + custom_strats
    active_algo = player.get('algo', 'Off')
    player['algo'] = st.selectbox("Choose Strategy", all_strats, index=all_strats.index(active_algo) if active_algo in all_strats else 0, disabled=disabled, key=f"algo_{player_name}")
    
    if player['algo'] in default_strats and player['algo'] != 'Off': 
        if player['algo'] == "Momentum Trader": st.info("This bot buys assets that have risen in price and sells those that have fallen, betting that recent trends will continue.")
        elif player['algo'] == "Mean Reversion": st.info("This bot buys assets that have recently fallen and sells those that have risen, betting on a return to their average price.")
        elif player['algo'] == "Volatility Breakout": st.info("This bot identifies assets making a significant daily price move (up or down) and trades in the same direction, aiming to capitalize on strong momentum.")
        elif player['algo'] == "Value Investor": st.info("This bot looks for assets that have dropped significantly over the past month and buys them, operating on the principle of buying undervalued assets for a potential long-term recovery.")

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
    st.subheader("💼 Portfolio Allocation")
    if player['holdings']:
        holdings_data = [{"Symbol": sym, "Value": prices.get(sym, 0) * qty} for sym, qty in player['holdings'].items()]
        holdings_df = pd.DataFrame(holdings_data)

        fig = go.Figure(data=[go.Pie(labels=holdings_df['Symbol'], values=holdings_df['Value'], hole=.3)])
        fig.update_layout(showlegend=False, height=200, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)
    else: 
        st.info("No holdings yet.")
        
def render_transaction_history(player_name):
    game_state = get_game_state()
    st.subheader("Transaction History")
    if game_state.transactions.get(player_name):
        trans_df = pd.DataFrame(game_state.transactions[player_name], columns=["Time", "Action", "Symbol", "Qty", "Price", "Total"])
        st.dataframe(trans_df.style.format(formatter={"Price": format_indian_currency, "Total": format_indian_currency}), use_container_width=True)
    else: st.info("No transactions recorded.")

def render_strategy_tab(player):
    st.subheader("📊 Strategy & Insights")
    tab1, tab2, tab3 = st.tabs(["Performance Chart", "Technical Analysis (SMA)", "Portfolio Optimizer"])
    with tab1:
        st.markdown("##### Portfolio Value Over Time")
        if len(player.get('value_history', [])) > 1:
            st.line_chart(player['value_history'])
        else:
            st.info("Trade more to see your performance chart.")
    with tab2: render_sma_chart(player['holdings'])
    with tab3: render_optimizer(player['holdings'])

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

def execute_trade(player_name, player, action, symbol, qty, prices, is_algo=False, order_type="Market"):
    game_state = get_game_state()
    mid_price = prices.get(symbol, 0)
    if mid_price == 0: return False

    if action == "Buy":
        trade_price = mid_price * (1 + game_state.bid_ask_spread / 2)
    else: # Sell or Short
        trade_price = mid_price * (1 - game_state.bid_ask_spread / 2)
    
    trade_price *= calculate_slippage(player, symbol, qty, action)
    cost = trade_price * qty
    
    trade_executed = False
    if action == "Buy" and player['capital'] >= cost:
        player['capital'] -= cost; player['holdings'][symbol] = player['holdings'].get(symbol, 0) + qty; trade_executed = True
    elif action == "Short" and player['capital'] >= cost * game_state.current_margin_requirement:
        player['capital'] += cost; player['holdings'][symbol] = player['holdings'].get(symbol, 0) - qty; trade_executed = True
    elif action == "Sell":
        current_qty = player['holdings'].get(symbol, 0)
        if current_qty > 0 and current_qty >= qty: # Closing a long
            player['capital'] += cost; player['holdings'][symbol] -= qty; trade_executed = True
        elif current_qty < 0 and abs(current_qty) >= qty: # Covering a short
            player['capital'] -= cost; player['holdings'][symbol] += qty; trade_executed = True
        if trade_executed and player['holdings'][symbol] == 0: del player['holdings'][symbol]
    
    if trade_executed: 
        get_game_state().market_sentiment[symbol] = get_game_state().market_sentiment.get(symbol, 0) + (qty / 50) * (1 if action in ["Buy", "Short"] else -1)
        log_transaction(player_name, f"{order_type} {action}", symbol, qty, trade_price, cost, is_algo)
    elif not is_algo: 
        st.error("Trade failed: Insufficient capital or holdings.")
    return trade_executed

def log_transaction(player_name, action, symbol, qty, price, total, is_algo=False):
    game_state = get_game_state()
    prefix = "🤖 Algo" if is_algo else ""
    game_state.transactions.setdefault(player_name, []).append([time.strftime("%H:%M:%S"), f"{prefix} {action}".strip(), symbol, qty, price, total])
    if "Auto-Liquidation" in action or "Settlement" in action:
        st.toast(f"{action}: {qty} {symbol}", icon="⚠️")
    elif not is_algo: 
        st.success(f"Trade Executed: {action} {qty} {symbol} @ {format_indian_currency(price)}")
    else: 
        st.toast(f"Algo Trade: {action} {qty} {symbol}", icon="🤖")

def calculate_sharpe_ratio(value_history):
    if len(value_history) < 2: return 0.0
    returns = pd.Series(value_history).pct_change().dropna()
    if returns.std() == 0: return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(252) # Annualized

def render_leaderboard(prices):
    game_state = get_game_state()
    lb = []
    for pname, pdata in game_state.players.items():
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in pdata['holdings'].items())
        total_value = pdata['capital'] + holdings_value
        sharpe_ratio = calculate_sharpe_ratio(pdata.get('value_history', []))
        lb.append((pname, pdata['mode'], total_value, pdata['pnl'], sharpe_ratio))
    
    if lb:
        lb_df = pd.DataFrame(lb, columns=["Player", "Mode", "Portfolio Value", "P&L", "Sharpe Ratio"]).sort_values("Portfolio Value", ascending=False).reset_index(drop=True)
        st.dataframe(lb_df.style.format(formatter={"Portfolio Value": format_indian_currency, "P&L": format_indian_currency, "Sharpe Ratio": "{:.2f}"}), use_container_width=True)
        
        if game_state.game_status == "Finished":
            if not getattr(game_state, 'auto_square_off_complete', False):
                auto_square_off_positions(prices)
                game_state.auto_square_off_complete = True
                st.rerun() # Rerun to update the leaderboard with final values

            st.balloons(); winner = lb_df.iloc[0]
            st.success(f"🎉 The winner is {winner['Player']}! 🎉")
            c1, c2 = st.columns(2)
            c1.metric("🏆 Final Portfolio Value", format_indian_currency(winner['Portfolio Value']))
            c2.metric("💰 Total P&L", format_indian_currency(winner['P&L']))

            prudent_winner = lb_df.sort_values("Sharpe Ratio", ascending=False).iloc[0]
            st.info(f"🧐 The Prudent Investor Award goes to {prudent_winner['Player']} with a Sharpe Ratio of {prudent_winner['Sharpe Ratio']:.2f}!")

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
    
    # Sentiment Decay
    for symbol in game_state.market_sentiment:
        game_state.market_sentiment[symbol] *= 0.95 

    # Random News Event Trigger
    if not game_state.event_active and random.random() < 0.07: # Increased frequency for 20 min session
        news_item = random.choice(PRE_BUILT_NEWS)
        headline = news_item['headline']
        impact = news_item['impact']
        target_symbol = None

        if "{symbol}" in headline:
            target_symbol = random.choice(NIFTY50_SYMBOLS)
            headline = headline.format(symbol=target_symbol.replace(".NS", ""))
        
        game_state.news_feed.insert(0, f"📢 {time.strftime('%H:%M:%S')} - {headline}")
        if len(game_state.news_feed) > 5: game_state.news_feed.pop()
        
        game_state.event_type = impact
        game_state.event_target_symbol = target_symbol # Store target for apply_event_adjustment
        game_state.event_active = True
        game_state.event_end = time.time() + random.randint(30, 60)
        st.toast(f"⚡ Market Event!", icon="🎉"); announce_news(headline)
        
    if game_state.event_active and time.time() >= game_state.event_end:
        game_state.event_active = False; st.info("Market event has ended.")
    if game_state.event_active: 
        if game_state.event_type == 'Volatility Spike':
            prices = {k: v * (1 + random.uniform(-0.01, 0.01) * 2) for k, v in prices.items()}
        else:
            prices = apply_event_adjustment(prices, game_state.event_type, getattr(game_state, 'event_target_symbol', None))
    
    handle_futures_expiry(prices)
    check_margin_calls_and_orders(prices)
    run_algo_strategies(prices)

    # Record portfolio value history for Sharpe Ratio calculation
    for player in game_state.players.values():
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        player['value_history'].append(total_value)
        
    return prices

def auto_square_off_positions(prices):
    """Automatically closes all intraday positions at the end of the game."""
    game_state = get_game_state()
    st.info("End of game: Auto-squaring off all intraday, futures, and options positions...")
    square_off_assets = NIFTY50_SYMBOLS + FUTURES_SYMBOLS + OPTION_SYMBOLS + LEVERAGED_ETFS
    
    for name, player in game_state.players.items():
        for symbol, qty in list(player['holdings'].items()): # Use list to avoid modification issues
            if symbol in square_off_assets:
                closing_price = prices.get(symbol, 0)
                value = closing_price * qty
                if qty > 0: # Long position
                    player['capital'] += value
                    log_transaction(name, "Auto-Squareoff (Sell)", symbol, qty, closing_price, value)
                else: # Short position
                    player['capital'] -= value # Cost to buy back
                    log_transaction(name, "Auto-Squareoff (Buy)", abs(qty), closing_price, value)
                del player['holdings'][symbol]

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
                    pnl = (settlement_price - prices.get(symbol, 0)) * qty
                    player['capital'] += pnl
                    log_transaction(name, "Futures Settlement", symbol, qty, settlement_price, pnl)
                    del player['holdings'][symbol]
        game_state.futures_settled = True

def check_margin_calls_and_orders(prices):
    game_state = get_game_state()
    for name, player in game_state.players.items():
        # Margin Call Logic
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        margin_needed = abs(holdings_value) * game_state.current_margin_requirement

        if total_value < margin_needed and player['holdings']:
            player['margin_calls'] += 1
            st.warning(f"MARGIN CALL for {name}! Liquidating largest position.")
            
            # Find largest position by absolute value to liquidate
            largest_position = max(player['holdings'].items(), key=lambda item: abs(item[1] * prices.get(item[0], 0)), default=(None, 0))
            if largest_position[0]:
                symbol_to_liquidate, qty_to_liquidate = largest_position
                action = "Sell" if qty_to_liquidate > 0 else "Buy"
                execute_trade(name, player, action, symbol_to_liquidate, abs(qty_to_liquidate), prices, order_type="Auto-Liquidation")

        # Pending Orders Logic
        orders_to_remove = []
        for i, order in enumerate(player['pending_orders']):
            current_price = prices.get(order['symbol'], 0)
            if current_price == 0: continue
            
            order_executed = False
            if order['type'] == 'Limit' and order['action'] == 'Buy' and current_price <= order['price']:
                order_executed = execute_trade(name, player, 'Buy', order['symbol'], order['qty'], prices, order_type="Limit")
            elif order['type'] == 'Limit' and order['action'] == 'Sell' and current_price >= order['price']:
                order_executed = execute_trade(name, player, 'Sell', order['symbol'], order['qty'], prices, order_type="Limit")
            elif order['type'] == 'Stop-Loss' and current_price <= order['price']:
                order_executed = execute_trade(name, player, 'Sell', order['symbol'], order['qty'], prices, order_type="Stop-Loss")
            
            if order_executed:
                orders_to_remove.append(i)
        
        # Remove executed orders
        for i in sorted(orders_to_remove, reverse=True):
            del player['pending_orders'][i]


def run_algo_strategies(prices):
    game_state = get_game_state()
    if len(game_state.price_history) < 2: return
    prev_prices = game_state.price_history[-2]
    
    for name, player in game_state.players.items():
        active_algo = player.get('algo', 'Off')
        if active_algo == 'Off': continue
        
        # Scan multiple symbols for faster execution
        for _ in range(3):
            trade_symbol = random.choice(NIFTY50_SYMBOLS + CRYPTO_SYMBOLS)
            if active_algo in player.get('custom_algos', {}):
                strategy = player['custom_algos'][active_algo]
                indicator_val = calculate_indicator(strategy['indicator'], trade_symbol)
                if indicator_val is None: continue
                condition_met = (strategy['condition'] == 'Greater Than' and indicator_val > strategy['threshold']) or \
                                (strategy['condition'] == 'Less Than' and indicator_val < strategy['threshold'])
                if condition_met: 
                    execute_trade(name, player, strategy['action'], trade_symbol, 1, prices, is_algo=True)
                    break # Execute one trade per tick
            else: # Default Algos
                price_change = prices.get(trade_symbol, 0) - prev_prices.get(trade_symbol, prices.get(trade_symbol, 0))
                if prices.get(trade_symbol, 0) == 0: continue

                if active_algo == "Momentum Trader" and abs(price_change / prices[trade_symbol]) > 0.001:
                    if execute_trade(name, player, "Buy" if price_change > 0 else "Sell", trade_symbol, 1, prices, is_algo=True): break
                elif active_algo == "Mean Reversion" and abs(price_change / prices[trade_symbol]) > 0.001:
                    if execute_trade(name, player, "Sell" if price_change > 0 else "Buy", trade_symbol, 1, prices, is_algo=True): break
                elif active_algo == "Volatility Breakout" and abs(price_change / prices[trade_symbol]) * 100 > 0.1:
                    if execute_trade(name, player, "Buy", trade_symbol, 1, prices, is_algo=True): break
                elif active_algo == "Value Investor":
                    change_30_day = calculate_indicator("Price Change % (30-day)", trade_symbol)
                    if change_30_day is not None and change_30_day < -10:
                        if execute_trade(name, player, "Buy", trade_symbol, 1, prices, is_algo=True): break

def main():
    game_state = get_game_state()
    if 'role' not in st.session_state:
        st.session_state.role = 'player'
    
    render_sidebar()
    
    # --- Main Price Flow ---
    # Fetch base prices only if they haven't been fetched for the day
    if not game_state.base_real_prices:
        game_state.base_real_prices = get_daily_base_prices()
        st.toast("Fetched daily base market prices.")

    # Use the last known price or the daily base price to start the tick simulation
    last_prices = game_state.prices if game_state.prices else game_state.base_real_prices
    
    # 1. Simulate small price fluctuations for the current tick
    current_prices = simulate_tick_prices(last_prices)
    
    # 2. Calculate prices for simulated assets (Futures, ETFs, Options) based on the new simulated prices
    prices_with_derivatives = calculate_derived_prices(current_prices)
    
    # 3. Apply temporary simulation effects like market events
    final_prices = run_game_tick(prices_with_derivatives)
    
    game_state.prices = final_prices
    
    if not isinstance(game_state.price_history, list): 
        game_state.price_history = []
    game_state.price_history.append(final_prices)
    if len(game_state.price_history) > 10: 
        game_state.price_history.pop(0)
    
    # Use a main container to prevent ghost UI
    main_container = st.container()
    
    with main_container:
        if st.session_state.role == 'admin':
            st.title(f"👑 {GAME_NAME} - Admin Dashboard")
            st.components.v1.html('<script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.7.77/Tone.js"></script>', height=0)
            
            if game_state.game_status == "Running":
                remaining_time = max(0, game_state.round_duration_seconds - int(time.time() - game_state.game_start_time))
                st.markdown(f"**Time Remaining: {remaining_time // 60:02d}:{remaining_time % 60:02d}**")
            elif game_state.game_status == "Stopped":
                st.info("Game is paused.")
            elif game_state.game_status == "Finished":
                st.success("Game has finished!")

            render_global_views(final_prices, is_admin=True)
        else:
            render_main_interface(final_prices)
    
    # Control refresh rate more carefully
    if game_state.game_status == "Running": 
        time.sleep(1)
        st.rerun()
    else: # Slower refresh for lobby/stopped state
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()

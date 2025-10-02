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
from concurrent.futures import ThreadPoolExecutor
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
NIFTY50 = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'ITC.NS', 'ASIANPAINT.NS', 'AXISBANK.NS']
CRYPTO = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'ADA-USD', 'XRP-USD']
GOLD = 'GC=F'
NIFTY_INDEX = '^NSEI'
BANKNIFTY_INDEX = '^NSEBANK'
FUTURES = ['NIFTY-FUT', 'BANKNIFTY-FUT']
LEVERAGED_ETFS = ['NIFTY_BULL_3X', 'NIFTY_BEAR_3X']
OPTIONS = ['NIFTY_CALL', 'NIFTY_PUT']
ALL_SYMBOLS = NIFTY50 + CRYPTO + [GOLD] + OPTIONS + FUTURES + LEVERAGED_ETFS
PRE_BUILT_NEWS = [
    {"headline": "RBI cuts repo rate by 25bps!", "impact": "Bull Rally"},
    {"headline": "Govt announces infrastructure package.", "impact": "Banking Boost"},
    {"headline": "Bank fraud shakes markets.", "impact": "Flash Crash"},
    {"headline": "Tech firm unveils AI breakthrough.", "impact": "Sector Rotation"},
    {"headline": "FIIs pour into equities.", "impact": "Bull Rally"},
    {"headline": "SEBI tightens margin rules.", "impact": "Flash Crash"},
    {"headline": "Monsoon forecast upgraded.", "impact": "Bull Rally"},
    {"headline": "US inflation spikes.", "impact": "Flash Crash"},
    {"headline": "ECB signals dovish policy.", "impact": "Bull Rally"},
    {"headline": "Supply chain disruption.", "impact": "Volatility Spike"},
    {"headline": "Trump announces tariffs.", "impact": "Volatility Spike"},
    {"headline": "Trump tweets growth.", "impact": "Bull Rally"},
    {"headline": "{symbol} wins contract!", "impact": "Symbol Bull Run"},
    {"headline": "{symbol} faces probe.", "impact": "Symbol Crash"},
    {"headline": "{symbol} stock split.", "impact": "Stock Split"},
    {"headline": "{symbol} declares dividend.", "impact": "Dividend"},
    {"headline": "{symbol} short squeeze!", "impact": "Short Squeeze"}
]
QUIZ_QUESTIONS = [
    {"question": "What does RSI stand for?", "options": ["Relative Strength Index", "Rapid Stock Increase", "Risk Sensitivity Indicator"], "answer": 0},
    {"question": "Bullish reversal pattern?", "options": ["Doji", "Hammer", "Shooting Star"], "answer": 1},
    {"question": "SEBI's primary role?", "options": ["Regulate securities market", "Control inflation", "Manage forex"], "answer": 0}
]

# --- Game State ---
@st.cache_resource
def get_game_state():
    class GameState:
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
            self.teams = {'A': [], 'B': []}
        def reset(self):
            base_prices = self.base_real_prices
            difficulty = self.difficulty_level
            self.__init__()
            self.base_real_prices = base_prices
            self.difficulty_level = difficulty
    return GameState()

# --- Sound Effects ---
def play_sound(sound_type):
    sounds = {
        'success': 'synth.triggerAttackRelease("C5", "8n");',
        'error': 'synth.triggerAttackRelease("C3", "8n");',
        'opening_bell': 'const now = Tone.now(); synth.triggerAttackRelease("G4", "8n", now); synth.triggerAttackRelease("C5", "8n", now + 0.3); synth.triggerAttackRelease("E5", "8n", now + 0.6);',
        'closing_warning': 'const now = Tone.now(); synth.triggerAttackRelease("G5", "16n", now); synth.triggerAttackRelease("G5", "16n", now + 0.3);',
        'final_bell': 'synth.triggerAttackRelease("C4", "2n");'
    }
    if sound_type in sounds:
        st.components.v1.html(f'<script>if (typeof Tone !== "undefined") {{const synth = new Tone.Synth().toDestination(); {sounds[sound_type]}}}</script>', height=0)

def announce_news(headline):
    safe_headline = headline.replace("'", "\\'").replace("\n", " ")
    st.components.v1.html(f'<script>if ("speechSynthesis" in window) {{const u = new SpeechSynthesisUtterance("{safe_headline}"); u.rate = 1.2; speechSynthesis.speak(u);}}</script>', height=0)

# --- Data Fetching & Market Simulation ---
@st.cache_data(ttl=86400)
def get_daily_base_prices():
    prices = {}
    yf_symbols = NIFTY50 + CRYPTO + [GOLD, NIFTY_INDEX, BANKNIFTY_INDEX]
    try:
        with ThreadPoolExecutor() as executor:
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
    volatility = game_state.volatility_multiplier * (1 + 0.2 * (game_state.difficulty_level - 1))
    for symbol in prices:
        if symbol not in FUTURES + LEVERAGED_ETFS + OPTIONS:
            sentiment = game_state.market_sentiment.get(symbol, 0)
            noise = random.uniform(-0.0005, 0.0005) * volatility
            prices[symbol] = max(0.01, prices[symbol] * (1 + sentiment * 0.001 + noise))
    return prices

def calculate_derived_prices(base_prices):
    game_state = get_game_state()
    prices = base_prices.copy()
    nifty = prices.get(NIFTY_INDEX, 20000)
    banknifty = prices.get(BANKNIFTY_INDEX, 45000)
    prices.update({
        'NIFTY_CALL': nifty * 1.02,
        'NIFTY_PUT': nifty * 0.98,
        'NIFTY-FUT': nifty * random.uniform(1.0, 1.005),
        'BANKNIFTY-FUT': banknifty * random.uniform(1.0, 1.005)
    })
    if len(game_state.price_history) >= 2:
        prev_nifty = game_state.price_history[-2].get(NIFTY_INDEX, nifty)
        nifty_change = (nifty - prev_nifty) / prev_nifty
        current_bull = game_state.prices.get('NIFTY_BULL_3X', nifty / 100)
        current_bear = game_state.prices.get('NIFTY_BEAR_3X', nifty / 100)
        prices['NIFTY_BULL_3X'] = current_bull * (1 + 3 * nifty_change)
        prices['NIFTY_BEAR_3X'] = current_bear * (1 - 3 * nifty_change)
    else:
        prices['NIFTY_BULL_3X'] = nifty / 100
        prices['NIFTY_BEAR_3X'] = nifty / 100
    return prices

@st.cache_data(ttl=3600)
def get_historical_data(symbols, period="6mo"):
    try:
        with ThreadPoolExecutor() as executor:
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
        "Flash Crash": ("⚡ Flash Crash! Prices dropping.", lambda p: {k: v * random.uniform(0.95, 0.98) for k, v in p.items()}),
        "Bull Rally": ("📈 Bull Rally! Prices surging.", lambda p: {k: v * random.uniform(1.02, 1.05) for k, v in p.items()}),
        "Banking Boost": ("🏦 Banking Boost!", lambda p: {k: v * 1.07 if k in ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'AXISBANK.NS'] else v for k, v in p.items()}),
        "Sector Rotation": ("🔄 Sector Rotation!", lambda p: {k: v * (0.95 if k in ['HDFCBANK.NS', 'ICICIBANK.NS'] else 1.10 if k in ['INFY.NS', 'TCS.NS'] else 1.0) for k, v in p.items()}),
        "Symbol Bull Run": (f"🚀 {target_symbol} Bull Run!", lambda p: {k: v * (1.15 if k == target_symbol else 1.0) for k, v in p.items()}),
        "Symbol Crash": (f"💥 {target_symbol} Crash!", lambda p: {k: v * (0.85 if k == target_symbol else 1.0) for k, v in p.items()}),
        "Short Squeeze": (f"🌀 Short Squeeze on {target_symbol}!", lambda p: {k: v * (1.25 if k == target_symbol else 1.0) for k, v in p.items()}),
        "Volatility Spike": ("🌪️ Volatility Spike!", lambda p: p),
        "Stock Split": (f"🔀 Stock Split for {target_symbol}!", lambda p: {k: v / 2 if k == target_symbol else v for k, v in p.items()}),
        "Dividend": (f"💰 Dividend for {target_symbol}!", lambda p: p)
    }
    if event_type in event_messages:
        message, price_func = event_messages[event_type]
        st.toast(message, icon="📢")
        summary = f"{message} (Type: {event_type})"
        if summary not in game_state.news_feed:
            game_state.news_feed.insert(0, summary)
            if len(game_state.news_feed) > 5: game_state.news_feed.pop()
        prices = price_func(prices)
        if event_type == "Stock Split" and target_symbol:
            for player in game_state.players.values():
                if target_symbol in player['holdings']:
                    player['holdings'][target_symbol] *= 2
        elif event_type == "Dividend" and target_symbol:
            dividend = prices[target_symbol] * 0.01
            for player in game_state.players.values():
                if target_symbol in player['holdings'] and player['holdings'][target_symbol] > 0:
                    payout = dividend * player['holdings'][target_symbol]
                    player['capital'] += payout
                    log_transaction(player['name'], "Dividend", target_symbol, player['holdings'][target_symbol], dividend, payout)
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

def calculate_indicator(indicator, symbol):
    hist = get_historical_data([symbol], period="2mo")
    if hist.empty or len(hist) < 30: return None
    prices = hist.iloc[:, 0]
    if indicator == "Price Change % (5-day)":
        if len(hist) < 6: return None
        return ((prices.iloc[-1] - prices.iloc[-6]) / prices.iloc[-6]) * 100
    elif indicator == "SMA Crossover (10/20)":
        if len(hist) < 20: return None
        return prices.rolling(window=10).mean().iloc[-1] - prices.rolling(window=20).mean().iloc[-1]
    elif indicator == "Price Change % (30-day)":
        if len(hist) < 31: return None
        return ((prices.iloc[-1] - prices.iloc[-31]) / prices.iloc[-31]) * 100
    return None

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=period, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs.replace([np.inf, -np.inf], np.nan)))
    return rsi.iloc[-1] if not rsi.empty else None

def calculate_macd(prices, short=12, long=26, signal=9):
    short_ema = prices.ewm(span=short, adjust=False).mean()
    long_ema = prices.ewm(span=long, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line.iloc[-1], signal_line.iloc[-1]

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    return sma.iloc[-1] + std.iloc[-1] * std_dev, sma.iloc[-1], sma.iloc[-1] - std.iloc[-1] * std_dev

# --- UI Functions ---
def render_left_sidebar():
    game_state = get_game_state()
    with st.sidebar:
        st.title("📝 Game Entry")
        if 'player' not in st.session_state:
            player_name = st.text_input("Enter Name", key="name_input")
            mode = st.radio("Select Mode", ["Trader", "HFT", "HNI"], key="mode_select")
            if st.button("Join Game"):
                if player_name.strip() and player_name not in game_state.players:
                    starting_capital = INITIAL_CAPITAL * 5 if mode == "HNI" else INITIAL_CAPITAL
                    game_state.players[player_name] = {
                        "name": player_name, "mode": mode, "capital": starting_capital, "holdings": {},
                        "pnl": 0, "leverage": 1.0, "margin_calls": 0, "pending_orders": [],
                        "algo": "Off", "custom_algos": {}, "slippage_multiplier": 0.5 if mode == "HFT" else 1.0,
                        "value_history": [], "trade_timestamps": [], "hft_trade_count": 0
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
            default_duration = int(game_state.round_duration_seconds / 60)
            game_duration = st.number_input("Game Duration (min)", min_value=1, value=default_duration, disabled=(game_state.game_status == "Running"))
            game_state.volatility_multiplier = st.slider("Volatility", 0.5, 5.0, game_state.volatility_multiplier, 0.5)
            difficulty_index = game_state.difficulty_level - 1
            game_state.difficulty_level = st.selectbox("Difficulty", [1, 2, 3], index=difficulty_index, format_func=lambda x: f"Level {x}", disabled=(game_state.game_status == "Running"))
            game_state.current_margin_requirement = st.slider("Margin (%)", 10, 50, int(game_state.current_margin_requirement * 100), 5) / 100.0
            st.subheader("Broadcast News")
            news_options = {news['headline']: news['impact'] for news in PRE_BUILT_NEWS}
            news = st.selectbox("Select News", ["None"] + list(news_options.keys()))
            target_symbol = None
            if news and "{symbol}" in news:
                target_symbol = st.selectbox("Target Symbol", [s.replace(".NS", "") for s in NIFTY50]) + ".NS"
            if news != "None":
                st.info(f"Impact: {news_options[news]}")
            if st.button("Publish News"):
                if news != "None":
                    selected_news = next(news for news in PRE_BUILT_NEWS if news["headline"] == news)
                    headline = selected_news['headline'].format(symbol=target_symbol.replace(".NS", "") if target_symbol else "")
                    game_state.news_feed.insert(0, f"📢 {time.strftime('%H:%M:%S')} - {headline}")
                    if len(game_state.news_feed) > 5: game_state.news_feed.pop()
                    game_state.event_type = selected_news['impact']
                    game_state.event_target_symbol = target_symbol
                    game_state.event_active = True
                    game_state.event_end = time.time() + 60
                    st.toast(f"News Published!", icon="📰"); announce_news(headline)
                    st.rerun()
            if st.button("▶️ Start Game", type="primary"):
                if game_state.players:
                    game_state.game_status = "Running"
                    game_state.game_start_time = time.time()
                    game_state.round_duration_seconds = game_duration * 60
                    game_state.futures_expiry_time = time.time() + (game_state.round_duration_seconds / 2)
                    game_state.closing_warning_triggered = False
                    play_sound('opening_bell')
                    st.toast("Game Started!", icon="🎉")
                    st.rerun()
                else:
                    st.warning("Add at least one player.")
            if st.button("⏸️ Stop Game"):
                game_state.game_status = "Stopped"
                st.toast("Game Paused!", icon="⏸️")
                st.rerun()
            if st.button("🔄 Reset Game"):
                game_state.reset()
                st.session_state.clear()
                st.toast("Game Reset.", icon="🔄")
                st.rerun()
            if st.button("Assign Teams"):
                assign_teams(game_state)
                st.success("Teams Assigned!")
            if st.button("Halt Trading"):
                game_state.game_status = "Halted"
                st.toast("Trading Halted!", icon="🚫")
                st.rerun()

def render_right_sidebar(prices):
    game_state = get_game_state()
    with st.sidebar:
        if st.session_state.get('role') == 'admin':
            st.title("Advanced Admin Controls")
            st.subheader("Create Block Deal")
            block_symbol = st.selectbox("Symbol", [s.replace(".NS", "") for s in NIFTY50] + CRYPTO)
            block_qty = st.number_input("Quantity", min_value=100, step=100, value=100)
            block_discount = st.slider("Discount (%)", 0, 20, 5)
            if st.button("Offer Block Deal"):
                game_state.block_deal_offer = {'symbol': block_symbol + (".NS" if block_symbol in [s.replace(".NS", "") for s in NIFTY50] else ""), 'qty': block_qty, 'discount': block_discount / 100, 'expires': time.time() + 120}
                st.success(f"Block deal: {block_qty} {block_symbol} at {block_discount}% discount!")
            st.subheader("Live Algo Bot Analyzer")
            algo_data = {name: sum(prices.get(s, 0) * q for s, q in p['holdings'].items()) + p['capital'] - (INITIAL_CAPITAL * 5 if p['mode'] == 'HNI' else INITIAL_CAPITAL) for name, p in game_state.players.items() if p['algo'] != 'Off'}
            if algo_data:
                st.dataframe(pd.DataFrame.from_dict(algo_data, orient='index', columns=['Algo P&L']).style.format({"Algo P&L": format_indian_currency}), width='stretch')
            else:
                st.info("No active algo bots.")
        elif 'player' in st.session_state:
            player = game_state.players.get(st.session_state.player, {})
            st.title(f"{st.session_state.player}'s Stats")
            holdings_value = sum(prices.get(s, 0) * q for s, q in player.get('holdings', {}).items())
            total_value = player.get('capital', 0) + holdings_value
            player['pnl'] = total_value - (INITIAL_CAPITAL * 5 if player.get('mode') == 'HNI' else INITIAL_CAPITAL)
            c1, c2, c3 = st.columns(3)
            c1.metric("Cash", format_indian_currency(player.get('capital', 0)))
            c2.metric("Portfolio", format_indian_currency(total_value))
            c3.metric("P&L", format_indian_currency(player['pnl']))
            if player.get('mode') == 'HNI' and game_state.block_deal_offer and time.time() < game_state.block_deal_offer['expires']:
                deal = game_state.block_deal_offer
                st.subheader("Exclusive Block Deal")
                st.info(f"Buy {deal['qty']} {deal['symbol']} at {deal['discount']*100:.1f}% discount!")
                if st.button("Accept Block Deal"):
                    execute_block_deal(st.session_state.player, player, deal, prices)
                    st.rerun()

def render_main_interface(prices):
    game_state = get_game_state()
    st.title(f"📈 {GAME_NAME}")
    st.components.v1.html('<script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.7.77/Tone.js"></script>', height=0)
    if game_state.game_status == "Running":
        remaining = max(0, game_state.round_duration_seconds - int(time.time() - game_state.game_start_time))
        if remaining <= 30 and not game_state.closing_warning_triggered:
            play_sound('closing_warning')
            game_state.closing_warning_triggered = True
        if remaining == 0:
            game_state.game_status = "Finished"
            play_sound('final_bell')
        st.markdown(f"**Time Remaining: {remaining // 60:02d}:{remaining % 60:02d}** | **Difficulty: Level {game_state.difficulty_level}**")
    elif game_state.game_status == "Stopped":
        st.info("Game paused. Press 'Start Game' to begin.")
    elif game_state.game_status == "Halted":
        st.warning("Trading halted by admin!")
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
    with st.container(border=True):
        st.subheader("Global Market View")
        sentiments = [s for s in get_game_state().market_sentiment.values() if s != 0]
        overall_sentiment = np.clip((np.mean(sentiments) + 5) * 10, 0, 100) if sentiments else 0
        st.markdown("##### Market Sentiment")
        c1, c2, c3 = st.columns([1, 4, 1])
        c1.write("<p style='text-align: right; margin-top: -5px;'>Fear</p>", unsafe_allow_html=True)
        c2.progress(int(overall_sentiment))
        c3.write("<p style='margin-top: -5px;'>Greed</p>", unsafe_allow_html=True)
        st.subheader("📰 Live News Feed")
        news_feed = get_game_state().news_feed
        if news_feed:
            for news in news_feed: st.info(news)
        else:
            st.info("No market news.")
        st.subheader("Live Player Standings")
        render_leaderboard(prices)
        if is_admin:
            st.subheader("Live Player Performance")
            chart_data = {name: p['value_history'] for name, p in get_game_state().players.items() if p.get('value_history')}
            if chart_data:
                # Align value_history lengths by padding with last value
                max_length = max(len(v) for v in chart_data.values())
                aligned_data = {name: v + [v[-1]] * (max_length - len(v)) if v else [0] * max_length for name, v in chart_data.items()}
                st.line_chart(pd.DataFrame(aligned_data))
            else:
                st.info("No trading activity.")
        st.subheader("Live Market Feed")
        prices_df = pd.DataFrame(prices.items(), columns=['Symbol', 'Price'])
        game_state = get_game_state()
        if len(game_state.price_history) >= 2:
            prev_prices = game_state.price_history[-2]
            prices_df['Change'] = prices_df['Symbol'].map(lambda s: prices[s] - prev_prices.get(s, prices[s]))
        else:
            prices_df['Change'] = 0.0
        all_trades = [[p] + t for p, ts in game_state.transactions.items() for t in ts]
        if all_trades:
            feed_df = pd.DataFrame(all_trades, columns=["Player", "Time", "Action", "Symbol", "Qty", "Trade Price", "Total"])
            last_trades = feed_df.sort_values('Time').groupby('Symbol').last()
            prices_df['Last Order'] = last_trades.apply(lambda r: f"{r['Player']} {r['Action']} {r['Qty']} @ {format_indian_currency(r['Trade Price'])}", axis=1).reindex(prices_df['Symbol']).fillna('-')
        else:
            prices_df['Last Order'] = '-'
        st.dataframe(prices_df.style.apply(lambda x: ['color: green' if v > 0 else 'color: red' if v < 0 else '' for v in x], subset=['Change']).format({'Price': format_indian_currency, 'Change': lambda v: f"{format_indian_currency(v) if v != 0 else '-'}"}), width='stretch', hide_index=True)

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
        player['pnl'] = total_value - (INITIAL_CAPITAL * 5 if player['mode'] == 'HNI' else INITIAL_CAPITAL)
        c1, c2, c3 = st.columns(3)
        c1.metric("Cash", format_indian_currency(player['capital']))
        c2.metric("Portfolio", format_indian_currency(total_value))
        c3.metric("P&L", format_indian_currency(player['pnl']))
        tabs = ["👨‍💻 Trade", "🤖 Algo", "📂 History", "📊 Insights"]
        tab1, tab2, tab3, tab4 = st.tabs(tabs)
        disabled = game_state.game_status != "Running"
        with tab1:
            order_types = ["Market", "Limit", "Stop-Loss"]
            t1, t2, t3 = st.tabs(order_types)
            with t1:
                render_market_order_ui(player_name, player, prices, disabled)
            with t2:
                render_limit_order_ui(player_name, player, prices, disabled)
            with t3:
                render_stop_loss_order_ui(player_name, player, prices, disabled)
            st.markdown("---")
            st.subheader("💼 Portfolio Allocation")
            if player['holdings']:
                holdings_data = [{"Symbol": s, "Value": prices.get(s, 0) * q} for s, q in player['holdings'].items()]
                fig = go.Figure(data=[go.Pie(labels=[d['Symbol'] for d in holdings_data], values=[d['Value'] for d in holdings_data], hole=.3)])
                fig.update_layout(showlegend=False, height=200, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, width='stretch')
            else:
                st.info("No holdings.")
            st.subheader("🕒 Pending Orders")
            if player['pending_orders']:
                st.dataframe(pd.DataFrame(player['pending_orders']), width='stretch')
            else:
                st.info("No pending orders.")
        with tab2:
            st.subheader("Automated Trading")
            default_strats = ["Off", "Momentum Trader", "Mean Reversion", "Volatility Breakout", "Value Investor"]
            custom_strats = list(player.get('custom_algos', {}).keys())
            all_strats = default_strats + custom_strats
            active_algo = player.get('algo', 'Off')
            player['algo'] = st.selectbox("Choose Strategy", all_strats, index=all_strats.index(active_algo) if active_algo in all_strats else 0, disabled=disabled, key=f"algo_{player_name}")
            if player['algo'] in default_strats and player['algo'] != 'Off':
                st.info({
                    "Momentum Trader": "Buys rising assets, sells falling.",
                    "Mean Reversion": "Buys fallen assets, sells risen.",
                    "Volatility Breakout": "Trades assets with big moves.",
                    "Value Investor": "Buys undervalued assets."
                }.get(player['algo'], ""))
            with st.expander("Create Custom Strategy"):
                c1, c2 = st.columns(2)
                with c1:
                    algo_name = st.text_input("Strategy Name", key=f"algo_name_{player_name}")
                    indicator = st.selectbox("Indicator", ["Price Change % (5-day)", "SMA Crossover (10/20)", "Price Change % (30-day)"], key=f"indicator_{player_name}")
                    condition = st.selectbox("Condition", ["Greater Than", "Less Than"], key=f"condition_{player_name}")
                with c2:
                    threshold = st.number_input("Threshold", value=0.0, step=0.1, key=f"threshold_{player_name}")
                    action = st.radio("Action", ["Buy", "Sell"], key=f"algo_action_{player_name}")
                if st.button("Save Strategy", key=f"save_algo_{player_name}"):
                    if algo_name.strip():
                        player['custom_algos'][algo_name] = {"indicator": indicator, "condition": condition, "threshold": threshold, "action": action}
                        st.success(f"Strategy '{algo_name}' saved!")
                        st.rerun()
                    else:
                        st.error("Strategy name cannot be empty.")
        with tab3:
            st.subheader("Transaction History")
            if game_state.transactions.get(player_name):
                trans_df = pd.DataFrame(game_state.transactions[player_name], columns=["Time", "Action", "Symbol", "Qty", "Price", "Total"])
                st.dataframe(trans_df.style.format(formatter={"Price": format_indian_currency, "Total": format_indian_currency}), width='stretch')
            else:
                st.info("No transactions.")
        with tab4:
            st.subheader("📊 Strategy & Insights")
            tabs = ["Performance", "SMA", "RSI", "MACD", "Bollinger", "Optimizer"]
            t1, t2, t3, t4, t5, t6 = st.tabs(tabs)
            chartable_assets = [s for s in player['holdings'].keys() if s not in OPTIONS + FUTURES + LEVERAGED_ETFS]
            with t1:
                st.markdown("##### Portfolio Value")
                if len(player.get('value_history', [])) > 1:
                    value_series = pd.Series(player['value_history'])
                    st.line_chart(value_series)
                    sharpe = calculate_sharpe_ratio(value_series)
                    peak = value_series.cummax()
                    drawdown = (value_series - peak) / peak
                    max_drawdown = drawdown.min() * peak.max()
                    returns = value_series.pct_change().dropna()
                    downside_std = returns[returns < 0].std() if not returns[returns < 0].empty else 0
                    sortino = (returns.mean() / downside_std) * np.sqrt(252) if downside_std != 0 else 0
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    c2.metric("Max Drawdown", format_indian_currency(max_drawdown))
                    c3.metric("Sortino Ratio", f"{sortino:.2f}")
                else:
                    st.info("Trade more to see performance.")
            for tab, indicator, key in [(t2, "SMA", "sma_symbol"), (t3, "RSI", "rsi_symbol"), (t4, "MACD", "macd_symbol"), (t5, "Bollinger", "bb_symbol")]:
                with tab:
                    st.markdown(f"##### {indicator}")
                    if not chartable_assets:
                        st.info("No chartable assets.")
                        continue
                    symbol = st.selectbox("Select Asset", chartable_assets, key=key)
                    hist_data = get_historical_data([symbol], period="6mo")
                    if not hist_data.empty:
                        df = hist_data.rename(columns={symbol: 'Close'})
                        fig = go.Figure()
                        if indicator == "SMA":
                            df['SMA_20'] = df['Close'].rolling(window=20).mean()
                            df['SMA_50'] = df['Close'].rolling(window=50).mean()
                            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price'))
                            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='20-Day SMA'))
                            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='50-Day SMA'))
                        elif indicator == "RSI":
                            df['RSI'] = calculate_rsi(df['Close'])
                            fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], mode='lines', name='RSI'))
                            fig.add_hline(y=70, line_dash="dash", line_color="red")
                            fig.add_hline(y=30, line_dash="dash", line_color="green")
                        elif indicator == "MACD":
                            df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
                            df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
                            fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD'))
                            fig.add_trace(go.Scatter(x=df.index, y=df['Signal'], mode='lines', name='Signal'))
                        elif indicator == "Bollinger":
                            df['Upper'] = df['Close'].rolling(window=20).mean() + (df['Close'].rolling(window=20).std() * 2)
                            df['Middle'] = df['Close'].rolling(window=20).mean()
                            df['Lower'] = df['Close'].rolling(window=20).mean() - (df['Close'].rolling(window=20).std() * 2)
                            fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price'))
                            fig.add_trace(go.Scatter(x=df.index, y=df['Upper'], mode='lines', name='Upper Band'))
                            fig.add_trace(go.Scatter(x=df.index, y=df['Middle'], mode='lines', name='SMA'))
                            fig.add_trace(go.Scatter(x=df.index, y=df['Lower'], mode='lines', name='Lower Band'))
                        st.plotly_chart(fig, width='stretch')
                    else:
                        st.warning(f"No data for {symbol}.")
            with t6:
                st.subheader("Portfolio Optimization")
                if st.button("Optimize Portfolio"):
                    weights, performance = optimize_portfolio(player['holdings'])
                    if weights:
                        st.success("Optimal weights:")
                        st.json({k: f"{v:.2%}" for k, v in weights.items()})
                        if performance:
                            st.write(f"Return: {performance[0]:.2%}, Volatility: {performance[1]:.2%}, Sharpe: {performance[2]:.2f}")
                    else:
                        st.error(performance)

def render_market_order_ui(player_name, player, prices, disabled):
    asset_types = ["Stock", "Crypto", "Gold", "Futures", "Leveraged ETF", "Option"]
    asset_type = st.radio("Asset Type", asset_types, horizontal=True, key=f"market_asset_{player_name}", disabled=disabled)
    if asset_type == "Futures" and get_game_state().futures_expiry_time > 0:
        expiry_remaining = max(0, get_game_state().futures_expiry_time - time.time())
        st.warning(f"Futures expire in {int(expiry_remaining // 60)}m {int(expiry_remaining % 60)}s")
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = {
            "Stock": st.selectbox("Stock", [s.replace('.NS', '') for s in NIFTY50], key=f"market_stock_{player_name}", disabled=disabled) + '.NS',
            "Crypto": st.selectbox("Cryptocurrency", CRYPTO, key=f"market_crypto_{player_name}", disabled=disabled),
            "Gold": GOLD,
            "Futures": st.selectbox("Futures", FUTURES, key=f"market_futures_{player_name}", disabled=disabled),
            "Leveraged ETF": st.selectbox("Leveraged ETF", LEVERAGED_ETFS, key=f"market_letf_{player_name}", disabled=disabled),
            "Option": st.selectbox("Option", OPTIONS, key=f"market_option_{player_name}", disabled=disabled)
        }.get(asset_type, GOLD)
    with col2:
        qty = st.number_input("Quantity", min_value=1, step=1, value=1, key=f"market_qty_{player_name}", disabled=disabled)
    mid_price = prices.get(symbol, 0)
    ask_price = mid_price * (1 + get_game_state().bid_ask_spread / 2)
    bid_price = mid_price * (1 - get_game_state().bid_ask_spread / 2)
    st.info(f"Bid: {format_indian_currency(bid_price)} | Ask: {format_indian_currency(ask_price)}")
    c1, c2, c3 = st.columns(3)
    for action, key, type_ in [("Buy", f"buy_{player_name}", "primary"), ("Sell", f"sell_{player_name}", "secondary"), ("Short", f"short_{player_name}", "secondary")]:
        if st.button(f"{action} {qty}", key=key, disabled=disabled, type=type_):
            if execute_trade(player_name, player, action, symbol, qty, prices):
                check_hft_rebate(player_name, player)
                play_sound('success')
            else:
                play_sound('error')
            st.rerun()

def render_limit_order_ui(player_name, player, prices, disabled):
    st.write("Set price for auto buy/sell.")
    asset_types = ["Stock", "Crypto", "Gold", "Futures", "Leveraged ETF", "Option"]
    asset_type = st.radio("Asset Type", asset_types, horizontal=True, key=f"limit_asset_{player_name}", disabled=disabled)
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = {
            "Stock": st.selectbox("Stock", [s.replace('.NS', '') for s in NIFTY50], key=f"limit_stock_{player_name}", disabled=disabled) + '.NS',
            "Crypto": st.selectbox("Cryptocurrency", CRYPTO, key=f"limit_crypto_{player_name}", disabled=disabled),
            "Gold": GOLD,
            "Futures": st.selectbox("Futures", FUTURES, key=f"limit_futures_{player_name}", disabled=disabled),
            "Leveraged ETF": st.selectbox("Leveraged ETF", LEVERAGED_ETFS, key=f"limit_letf_{player_name}", disabled=disabled),
            "Option": st.selectbox("Option", OPTIONS, key=f"limit_option_{player_name}", disabled=disabled)
        }.get(asset_type, GOLD)
    with col2:
        qty = st.number_input("Quantity", min_value=1, step=1, value=1, key=f"limit_qty_{player_name}", disabled=disabled)
    limit_price = st.number_input("Limit Price", min_value=0.01, value=prices.get(symbol, 0), step=0.01, key=f"limit_price_{player_name}", disabled=disabled)
    c1, c2, c3 = st.columns(3)
    for action, key, type_ in [("Buy Limit", f"buy_limit_{player_name}", "primary"), ("Sell Limit", f"sell_limit_{player_name}", "secondary"), ("Short Limit", f"short_limit_{player_name}", "secondary")]:
        if st.button(action, key=key, disabled=disabled, type=type_):
            player['pending_orders'].append({'type': 'Limit', 'action': action.split()[0], 'symbol': symbol, 'qty': qty, 'price': limit_price})
            st.success(f"Limit {action.split()[0]} order placed!")
            st.rerun()

def render_stop_loss_order_ui(player_name, player, prices, disabled):
    st.write("Set price to sell if asset drops.")
    asset_types = ["Stock", "Crypto", "Gold", "Futures", "Leveraged ETF", "Option"]
    asset_type = st.radio("Asset Type", asset_types, horizontal=True, key=f"stop_asset_{player_name}", disabled=disabled)
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = {
            "Stock": st.selectbox("Stock", [s.replace('.NS', '') for s in NIFTY50], key=f"stop_stock_{player_name}", disabled=disabled) + '.NS',
            "Crypto": st.selectbox("Cryptocurrency", CRYPTO, key=f"stop_crypto_{player_name}", disabled=disabled),
            "Gold": GOLD,
            "Futures": st.selectbox("Futures", FUTURES, key=f"stop_futures_{player_name}", disabled=disabled),
            "Leveraged ETF": st.selectbox("Leveraged ETF", LEVERAGED_ETFS, key=f"stop_letf_{player_name}", disabled=disabled),
            "Option": st.selectbox("Option", OPTIONS, key=f"stop_option_{player_name}", disabled=disabled)
        }.get(asset_type, GOLD)
    with col2:
        qty = st.number_input("Quantity", min_value=1, step=1, value=1, key=f"stop_qty_{player_name}", disabled=disabled)
    stop_price = st.number_input("Stop Price", min_value=0.01, value=prices.get(symbol, 0) * 0.95, step=0.01, key=f"stop_price_{player_name}", disabled=disabled)
    if st.button("Set Stop-Loss", key=f"set_stop_{player_name}", disabled=disabled):
        player['pending_orders'].append({'type': 'Stop-Loss', 'action': 'Sell', 'symbol': symbol, 'qty': qty, 'price': stop_price})
        st.success("Stop-Loss order placed!")
        st.rerun()

def execute_trade(player_name, player, action, symbol, qty, prices, is_algo=False, order_type="Market"):
    game_state = get_game_state()
    mid_price = prices.get(symbol, 0)
    if mid_price == 0: return False
    trade_price = mid_price * (1 + game_state.bid_ask_spread / 2 if action == "Buy" else 1 - game_state.bid_ask_spread / 2) * calculate_slippage(player, symbol, qty, action)
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
        if (current_qty > 0 and current_qty >= qty) or (current_qty < 0 and abs(current_qty) >= qty):
            player['capital'] += cost if current_qty > 0 else -cost
            player['holdings'][symbol] += -qty if current_qty > 0 else qty
            trade_executed = True
        if trade_executed and player['holdings'].get(symbol, 0) == 0:
            del player['holdings'][symbol]
    if trade_executed:
        game_state.market_sentiment[symbol] = game_state.market_sentiment.get(symbol, 0) + (qty / 50) * (1 if action in ["Buy", "Short"] else -1)
        log_transaction(player_name, f"{order_type} {action}", symbol, qty, trade_price, cost, is_algo)
        player['trade_timestamps'].append(time.time())
    elif not is_algo:
        st.error("Trade failed: Insufficient capital or holdings.")
    return trade_executed

def execute_block_deal(player_name, player, deal, prices):
    if player['mode'] != 'HNI':
        st.error("Only HNI players can accept block deals.")
        return
    symbol, qty, discount = deal['symbol'], deal['qty'], deal['discount']
    mid_price = prices.get(symbol, 0)
    if mid_price == 0: return
    trade_price = mid_price * (1 - discount)
    cost = trade_price * qty
    if player['capital'] >= cost:
        player['capital'] -= cost
        player['holdings'][symbol] = player['holdings'].get(symbol, 0) + qty
        log_transaction(player_name, "Block Deal Buy", symbol, qty, trade_price, cost)
        st.success(f"Block deal: {qty} {symbol} at {format_indian_currency(trade_price)}")
        get_game_state().block_deal_offer = None
    else:
        st.error("Insufficient capital.")

def check_hft_rebate(player_name, player):
    game_state = get_game_state()
    if player['mode'] != 'HFT': return
    player['hft_trade_count'] = player.get('hft_trade_count', 0) + 1
    recent_trades = [t for t in player['trade_timestamps'] if time.time() - t <= game_state.hft_rebate_window]
    player['trade_timestamps'] = recent_trades
    if len(recent_trades) >= game_state.hft_rebate_trades:
        player['capital'] += game_state.hft_rebate_amount
        log_transaction(player_name, "HFT Rebate", "CASH", 1, game_state.hft_rebate_amount, game_state.hft_rebate_amount)
        st.toast(f"HFT Rebate: {format_indian_currency(game_state.hft_rebate_amount)}", icon="💰")
        player['trade_timestamps'] = []
        player['hft_trade_count'] = 0

def log_transaction(player_name, action, symbol, qty, price, total, is_algo=False):
    game_state = get_game_state()
    prefix = "🤖 Algo" if is_algo else ""
    game_state.transactions.setdefault(player_name, []).append([time.strftime("%H:%M:%S"), f"{prefix} {action}".strip(), symbol, qty, price, total])
    if len(game_state.transactions[player_name]) > 1000:  # Limit transaction history
        game_state.transactions[player_name] = game_state.transactions[player_name][-1000:]
    if "Auto-Liquidation" in action or "Settlement" in action:
        st.toast(f"{action}: {qty} {symbol}", icon="⚠️")
    elif not is_algo:
        st.success(f"Trade: {action} {qty} {symbol} @ {format_indian_currency(price)}")
    else:
        st.toast(f"Algo Trade: {action} {qty} {symbol}", icon="🤖")

def calculate_sharpe_ratio(values):
    if len(values) < 2: return 0.0
    returns = pd.Series(values).pct_change().dropna()
    return (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0.0

def render_leaderboard(prices):
    game_state = get_game_state()
    def calculate_player_metrics(name, player):
        portfolio_value = player['capital'] + sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
        return name, player['mode'], portfolio_value, player['pnl'], calculate_sharpe_ratio(player.get('value_history', []))
    
    with ThreadPoolExecutor() as executor:
        lb = list(executor.map(lambda x: calculate_player_metrics(*x), game_state.players.items()))
    
    if lb:
        lb_df = pd.DataFrame(lb, columns=["Player", "Mode", "Portfolio Value", "P&L", "Sharpe Ratio"]).sort_values("Portfolio Value", ascending=False).reset_index(drop=True)
        st.dataframe(lb_df.style.format(formatter={"Portfolio Value": format_indian_currency, "P&L": format_indian_currency, "Sharpe Ratio": "{:.2f}"}), width='stretch', hide_index=True)
        if game_state.teams['A'] or game_state.teams['B']:
            st.subheader("Team Standings")
            team_a_pnl, team_a_port = sum(p['pnl'] for p in game_state.players.values() if p['name'] in game_state.teams['A']), sum(p['capital'] + sum(prices.get(s, 0) * q for s, q in p['holdings'].items()) for p in game_state.players.values() if p['name'] in game_state.teams['A'])
            team_b_pnl, team_b_port = sum(p['pnl'] for p in game_state.players.values() if p['name'] in game_state.teams['B']), sum(p['capital'] + sum(prices.get(s, 0) * q for s, q in p['holdings'].items()) for p in game_state.players.values() if p['name'] in game_state.teams['B'])
            team_df = pd.DataFrame({"Team": ["Team A", "Team B"], "Total Portfolio": [team_a_port, team_b_port], "Total P&L": [team_a_pnl, team_b_pnl]})
            st.dataframe(team_df.style.format(formatter={"Total Portfolio": format_indian_currency, "Total P&L": format_indian_currency}), width='stretch', hide_index=True)
        if game_state.game_status == "Finished" and not game_state.auto_square_off_complete:
            auto_square_off_positions(prices)
            game_state.auto_square_off_complete = True
            st.rerun()
        if game_state.game_status == "Finished":
            st.balloons()
            winner = lb_df.iloc[0]
            st.success(f"🎉 Winner: {winner['Player']}!")
            c1, c2 = st.columns(2)
            c1.metric("🏆 Final Portfolio", format_indian_currency(winner['Portfolio Value']))
            c2.metric("💰 Total P&L", format_indian_currency(winner['P&L']))
            prudent = lb_df.sort_values("Sharpe Ratio", ascending=False).iloc[0]
            st.info(f"🧐 Prudent Investor: {prudent['Player']} (Sharpe: {prudent['Sharpe Ratio']:.2f})")

def auto_square_off_positions(prices):
    game_state = get_game_state()
    st.info("End of game: Auto-squaring positions...")
    for name, player in game_state.players.items():
        for symbol, qty in list(player['holdings'].items()):
            if symbol in NIFTY50 + FUTURES + OPTIONS + LEVERAGED_ETFS:
                closing_price = prices.get(symbol, 0)
                value = closing_price * qty
                action = "Sell" if qty > 0 else "Buy"
                player['capital'] += value if qty > 0 else -value
                log_transaction(name, f"Auto-Squareoff ({action})", symbol, abs(qty), closing_price, value)
                del player['holdings'][symbol]

def handle_futures_expiry(prices):
    game_state = get_game_state()
    if not game_state.futures_settled and game_state.futures_expiry_time > 0 and time.time() > game_state.futures_expiry_time:
        st.warning("FUTURES EXPIRED! Cash-settling.")
        settlement_prices = {'NIFTY-FUT': prices.get(NIFTY_INDEX), 'BANKNIFTY-FUT': prices.get(BANKNIFTY_INDEX)}
        for name, player in game_state.players.items():
            for symbol in FUTURES:
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
    def process_player(name, player):
        holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        margin_needed = abs(holdings_value) * game_state.current_margin_requirement
        if total_value < margin_needed and player['holdings']:
            player['margin_calls'] += 1
            st.warning(f"MARGIN CALL for {name}! Liquidating.")
            largest = max(player['holdings'].items(), key=lambda x: abs(x[1] * prices.get(x[0], 0)), default=(None, 0))
            if largest[0]:
                symbol, qty = largest
                action = "Sell" if qty > 0 else "Buy"
                execute_trade(name, player, action, symbol, abs(qty), prices, order_type="Auto-Liquidation")
    
    with ThreadPoolExecutor() as executor:
        executor.map(lambda x: process_player(*x), game_state.players.items())

def process_pending_orders(prices):
    game_state = get_game_state()
    def process_player_orders(name, player):
        orders_to_remove = []
        for i, order in enumerate(player['pending_orders']):
            current_price = prices.get(order['symbol'], 0)
            if current_price == 0: continue
            order_executed = False
            if order['type'] == 'Limit' and order['action'] == 'Buy' and current_price <= order['price']:
                order_executed = execute_trade(name, player, 'Buy', order['symbol'], order['qty'], prices, order_type="Limit")
            elif order['type'] == 'Limit' and order['action'] in ['Sell', 'Short'] and current_price >= order['price']:
                order_executed = execute_trade(name, player, order['action'], order['symbol'], order['qty'], prices, order_type="Limit")
            elif order['type'] == 'Stop-Loss' and current_price <= order['price']:
                order_executed = execute_trade(name, player, 'Sell', order['symbol'], order['qty'], prices, order_type="Stop-Loss")
            if order_executed:
                orders_to_remove.append(i)
        for i in sorted(orders_to_remove, reverse=True):
            del player['pending_orders'][i]
    
    with ThreadPoolExecutor() as executor:
        executor.map(lambda x: process_player_orders(*x), game_state.players.items())

def assign_teams(game_state):
    players = list(game_state.players.keys())
    random.shuffle(players)
    half = len(players) // 2
    game_state.teams['A'], game_state.teams['B'] = players[:half], players[half:]

def apply_difficulty_mechanics(prices):
    game_state = get_game_state()
    def process_difficulty(name, player):
        holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        initial_capital = INITIAL_CAPITAL * 5 if player['mode'] == 'HNI' else INITIAL_CAPITAL
        if game_state.difficulty_level >= 2:
            performance = (total_value - initial_capital) / initial_capital
            if performance > 0.1:
                bonus = initial_capital * 0.05
                player['capital'] += bonus
                log_transaction(name, "Performance Bonus", "CASH", 1, bonus, bonus)
            elif performance < -0.1:
                penalty = initial_capital * 0.05
                player['capital'] = max(0, player['capital'] - penalty)
                log_transaction(name, "Performance Penalty", "CASH", 1, -penalty, -penalty)
        if game_state.difficulty_level == 3 and abs(holdings_value) > player['capital'] * 2:
            st.warning(f"Risk alert for {name}! Reducing positions.")
            for symbol, qty in list(player['holdings'].items()):
                if abs(qty) > 10:
                    action = "Sell" if qty > 0 else "Buy"
                    execute_trade(name, player, action, symbol, abs(qty) // 2, prices, order_type="Risk Management")
    
    with ThreadPoolExecutor() as executor:
        executor.map(lambda x: process_difficulty(*x), game_state.players.items())

def run_algo_strategies(prices):
    game_state = get_game_state()
    if len(game_state.price_history) < 2: return
    prev_prices = game_state.price_history[-2]
    def process_algo(name, player):
        algo = player.get('algo', 'Off')
        if algo == 'Off': return
        for _ in range(3):
            symbol = random.choice(NIFTY50 + CRYPTO)
            if algo in player.get('custom_algos', {}):
                strategy = player['custom_algos'][algo]
                indicator_val = calculate_indicator(strategy['indicator'], symbol)
                if indicator_val is None: continue
                if (strategy['condition'] == 'Greater Than' and indicator_val > strategy['threshold']) or \
                   (strategy['condition'] == 'Less Than' and indicator_val < strategy['threshold']):
                    execute_trade(name, player, strategy['action'], symbol, 1, prices, is_algo=True)
                    break
            else:
                price_change = prices.get(symbol, 0) - prev_prices.get(symbol, prices.get(symbol, 0))
                if prices.get(symbol, 0) == 0: continue
                if algo == "Momentum Trader" and abs(price_change / prices[symbol]) > 0.001:
                    if execute_trade(name, player, "Buy" if price_change > 0 else "Sell", symbol, 1, prices, is_algo=True):
                        break
                elif algo == "Mean Reversion" and abs(price_change / prices[symbol]) > 0.005:
                    if execute_trade(name, player, "Buy" if price_change < 0 else "Sell", symbol, 1, prices, is_algo=True):
                        break
                elif algo == "Volatility Breakout" and abs(price_change / prices[symbol]) > 0.01:
                    if execute_trade(name, player, "Buy" if price_change > 0 else "Sell", symbol, 1, prices, is_algo=True):
                        break
                elif algo == "Value Investor" and abs(price_change / prices[symbol]) > 0.02 and price_change < 0:
                    if execute_trade(name, player, "Buy", symbol, 1, prices, is_algo=True):
                        break
    
    with ThreadPoolExecutor() as executor:
        executor.map(lambda x: process_algo(*x), game_state.players.items())

def main():
    game_state = get_game_state()
    if not game_state.base_real_prices:
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
        game_state.price_history.append(game_state.prices.copy())
        if len(game_state.price_history) > 100:
            game_state.price_history.pop(0)
        if random.random() < 0.02 and 'player' in st.session_state:
            player = game_state.players.get(st.session_state.player)
            if player:
                with st.container():
                    st.subheader("📝 Market Quiz Bonus")
                    quiz = random.choice(QUIZ_QUESTIONS)
                    st.write(quiz['question'])
                    answer = st.radio("Select Answer", quiz['options'], key=f"quiz_{st.session_state.player}_{time.time()}")
                    if st.button("Submit Answer"):
                        if quiz['options'].index(answer) == quiz['answer']:
                            bonus = player['capital'] * 0.05
                            player['capital'] += bonus
                            log_transaction(st.session_state.player, "Quiz Bonus", "CASH", 1, bonus, bonus)
                            st.success(f"Correct! Bonus: {format_indian_currency(bonus)}")
                            play_sound('success')
                        else:
                            st.error("Incorrect answer.")
                            play_sound('error')
                        st.rerun()

def run_game_tick(prices):
    game_state = get_game_state()
    if game_state.game_status != "Running": return prices
    process_pending_orders(prices)
    for symbol in game_state.market_sentiment:
        game_state.market_sentiment[symbol] *= 0.95
    if not game_state.event_active and random.random() < 0.07:
        news = random.choice(PRE_BUILT_NEWS)
        headline = news['headline']
        target_symbol = random.choice(NIFTY50) if "{symbol}" in headline else None
        if target_symbol:
            headline = headline.format(symbol=target_symbol.replace(".NS", ""))
        game_state.news_feed.insert(0, f"📢 {time.strftime('%H:%M:%S')} - {headline}")
        if len(game_state.news_feed) > 5: game_state.news_feed.pop()
        game_state.event_type = news['impact']
        game_state.event_target_symbol = target_symbol
        game_state.event_active = True
        game_state.event_end = time.time() + random.randint(30, 60)
        st.toast(f"⚡ Market Event!", icon="🎉")
        announce_news(headline)
    if game_state.event_active and time.time() >= game_state.event_end:
        game_state.event_active = False
        st.info("Market event ended.")
    if game_state.event_active:
        prices = apply_event_adjustment(prices, game_state.event_type, game_state.event_target_symbol) if game_state.event_type != 'Volatility Spike' else {k: v * (1 + random.uniform(-0.01, 0.01) * 2) for k, v in prices.items()}
    handle_futures_expiry(prices)
    check_margin_calls_and_orders(prices)
    run_algo_strategies(prices)
    apply_difficulty_mechanics(prices)
    def update_player_value(player):
        holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
        player['value_history'].append(player['capital'] + holdings_value)
        if len(player['value_history']) > 1000:  # Limit history length
            player['value_history'] = player['value_history'][-1000:]
    
    with ThreadPoolExecutor() as executor:
        executor.map(update_player_value, game_state.players.values())
    return prices

if __name__ == "__main__":
    main()

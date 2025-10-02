import streamlit as st
import pandas as pd
import yfinance as yf
import time
import random
import plotly.graph_objects as go
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
import uuid
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="BlockVista Market Frenzy", page_icon="📈")

# --- Constants ---
GAME_NAME = "BlockVista Market Frenzy"
INITIAL_CAPITAL = 1000000
ADMIN_PASSWORD = "100370"
NIFTY50_SYMBOLS = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'ITC.NS', 'ASIANPAINT.NS', 'AXISBANK.NS']
CRYPTO_SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'ADA-USD', 'XRP-USD']
GOLD_SYMBOL = 'GC=F'
NIFTY_INDEX_SYMBOL = '^NSEI'
BANKNIFTY_INDEX_SYMBOL = '^NSEBANK'
FUTURES_SYMBOLS = ['NIFTY-FUT', 'BANKNIFTY-FUT']
LEVERAGED_ETFS = ['NIFTY_BULL_3X', 'NIFTY_BEAR_3X']
OPTION_SYMBOLS = ['NIFTY_CALL', 'NIFTY_PUT']
ALL_SYMBOLS = NIFTY50_SYMBOLS + CRYPTO_SYMBOLS + [GOLD_SYMBOL] + OPTION_SYMBOLS + FUTURES_SYMBOLS + LEVERAGED_ETFS + [NIFTY_INDEX_SYMBOL, BANKNIFTY_INDEX_SYMBOL]
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
    {"headline": "Global News: President Trump announces new trade tariffs, increasing market uncertainty.", "impact": "Volatility Spike"},
    {"headline": "Global News: President Trump tweets about 'tremendous' economic growth, boosting investor confidence.", "impact": "Bull Rally"},
    {"headline": "{symbol} secures a massive government contract, sending its stock soaring!", "impact": "Symbol Bull Run"},
    {"headline": "Regulatory probe launched into {symbol} over accounting irregularities.", "impact": "Symbol Crash"},
    {"headline": "{symbol} announces a surprise stock split, shares to adjust at market open.", "impact": "Stock Split"},
    {"headline": "{symbol} declares a special dividend for all shareholders.", "impact": "Dividend"},
    {"headline": "High short interest in {symbol} triggers a potential short squeeze!", "impact": "Short Squeeze"},
]
QUIZ_QUESTIONS = [
    {"question": "What does RSI stand for in technical analysis?", "options": ["Relative Strength Index", "Rapid Stock Increase", "Risk Sensitivity Indicator"], "answer": 0},
    {"question": "Which candlestick pattern signals a bullish reversal?", "options": ["Doji", "Hammer", "Shooting Star"], "answer": 1},
    {"question": "What is the primary role of SEBI in India?", "options": ["Regulate securities market", "Control inflation", "Manage foreign exchange"], "answer": 0},
]

# --- Game State Management (Singleton for Live Sync) ---
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
            self.algo_pnl = {}
            self.circuit_breaker_active = False
            self.circuit_breaker_end = 0
            self.algo_trades = {}
            self.team_scores = {'A': 0, 'B': 0}
            
        def reset(self):
            base_prices = self.base_real_prices
            difficulty = self.difficulty_level
            self.__init__()
            self.base_real_prices = base_prices
            self.difficulty_level = difficulty
            
    return GameState()

# --- Sound Effects & Notifications ---
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
    yf_symbols = NIFTY50_SYMBOLS + CRYPTO_SYMBOLS + [GOLD_SYMBOL, NIFTY_INDEX_SYMBOL, BANKNIFTY_INDEX_SYMBOL]
    try:
        data = yf.download(tickers=yf_symbols, period="1d", interval="1m", progress=False, threads=True)
        for symbol in yf_symbols:
            prices[symbol] = data['Close'][symbol].iloc[-1] if not data.empty and symbol in data['Close'] and not pd.isna(data['Close'][symbol].iloc[-1]) else random.uniform(10, 50000)
    except Exception:
        prices.update({s: random.uniform(10, 50000) for s in yf_symbols})
    return prices

def simulate_tick_prices(last_prices):
    game_state = get_game_state()
    prices = last_prices.copy()
    volatility = game_state.volatility_multiplier * (1 + 0.2 * (game_state.difficulty_level - 1))
    for symbol in prices:
        if symbol not in FUTURES_SYMBOLS + LEVERAGED_ETFS + OPTION_SYMBOLS:
            sentiment = game_state.market_sentiment.get(symbol, 0)
            noise = random.uniform(-0.0005, 0.0005) * volatility
            prices[symbol] = max(0.01, prices[symbol] * (1 + sentiment * 0.001 + noise))
    return prices

def calculate_derived_prices(base_prices):
    game_state = get_game_state()
    prices = base_prices.copy()
    nifty = prices.get(NIFTY_INDEX_SYMBOL, 20000)
    banknifty = prices.get(BANKNIFTY_INDEX_SYMBOL, 45000)
    prices.update({
        'NIFTY_CALL': nifty * 1.02,
        'NIFTY_PUT': nifty * 0.98,
        'NIFTY-FUT': nifty * random.uniform(1.0, 1.005),
        'BANKNIFTY-FUT': banknifty * random.uniform(1.0, 1.005)
    })
    if len(game_state.price_history) >= 2:
        prev_nifty = game_state.price_history[-2].get(NIFTY_INDEX_SYMBOL, nifty)
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
        data = yf.download(tickers=symbols, period=period, progress=False, threads=True)
        return data['Close'] if not data.empty else pd.DataFrame()
    except Exception:
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
    symbols = [s for s in holdings.keys() if s in NIFTY50_SYMBOLS + CRYPTO_SYMBOLS]
    if len(symbols) < 2: return None, "Need at least 2 assets."
    try:
        hist_data = get_historical_data(symbols)
        if hist_data.empty or hist_data.isnull().values.any(): return None, "Insufficient data."
        mu = expected_returns.mean_historical_return(hist_data)
        S = risk_models.sample_cov(hist_data)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        return ef.clean_weights(), ef.portfolio_performance(verbose=False)
    except Exception as e: return None, f"Optimization failed: {e}"

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

def calculate_max_drawdown(value_history):
    if len(value_history) < 2: return 0.0
    peak = value_history[0]
    max_dd = 0.0
    for value in value_history:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd * 100

def calculate_sortino_ratio(value_history):
    if len(value_history) < 2: return 0.0
    returns = pd.Series(value_history).pct_change().dropna()
    target_return = 0
    downside_returns = returns[returns < target_return]
    if len(downside_returns) == 0: return float('inf')
    downside_deviation = np.std(downside_returns)
    if downside_deviation == 0: return float('inf')
    excess_return = returns.mean() - target_return
    return excess_return / downside_deviation * np.sqrt(252)

def run_game_tick(prices):
    game_state = get_game_state()
    if game_state.game_status != "Running": return prices
    process_pending_orders(prices)
    for symbol in game_state.market_sentiment:
        game_state.market_sentiment[symbol] *= 0.95
    if not game_state.event_active and random.random() < 0.07:
        news = random.choice(PRE_BUILT_NEWS)
        headline = news['headline']
        target_symbol = random.choice(NIFTY50_SYMBOLS) if "{symbol}" in headline else None
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
    for player in game_state.players.values():
        holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
        player['value_history'].append(player['capital'] + holdings_value)
    return prices

def execute_trade(player_name, player, action, symbol, qty, prices, is_algo=False, order_type="Market"):
    game_state = get_game_state()
    
    # Check if circuit breaker is active
    if game_state.circuit_breaker_active and time.time() < game_state.circuit_breaker_end:
        if not is_algo:
            st.error("🚫 Trading halted due to circuit breaker!")
        return False
        
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
        
        # Track algo P&L
        if is_algo:
            if player_name not in game_state.algo_pnl:
                game_state.algo_pnl[player_name] = 0
            game_state.algo_pnl[player_name] += -cost if action == "Buy" else cost
            
        # Check for HFT rebate
        check_hft_rebate(player_name, player)
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

def process_pending_orders(prices):
    game_state = get_game_state()
    for player_name, player in game_state.players.items():
        orders_to_remove = []
        for i, order in enumerate(player['pending_orders']):
            symbol = order['symbol']
            current_price = prices.get(symbol, 0)
            
            if order['type'] == 'Limit':
                if order['action'] == 'Buy' and current_price <= order['price']:
                    if execute_trade(player_name, player, "Buy", symbol, order['qty'], prices, order_type="Limit"):
                        orders_to_remove.append(i)
                elif order['action'] in ['Sell', 'Short'] and current_price >= order['price']:
                    if execute_trade(player_name, player, order['action'], symbol, order['qty'], prices, order_type="Limit"):
                        orders_to_remove.append(i)
                        
            elif order['type'] == 'Stop-Loss':
                if order['action'] == 'Sell' and current_price <= order['price']:
                    if execute_trade(player_name, player, "Sell", symbol, order['qty'], prices, order_type="Stop-Loss"):
                        orders_to_remove.append(i)
        
        # Remove executed orders (in reverse to avoid index issues)
        for i in sorted(orders_to_remove, reverse=True):
            player['pending_orders'].pop(i)

def handle_futures_expiry(prices):
    game_state = get_game_state()
    if game_state.futures_expiry_time > 0 and time.time() >= game_state.futures_expiry_time and not game_state.futures_settled:
        for player_name, player in game_state.players.items():
            for symbol in FUTURES_SYMBOLS:
                if symbol in player['holdings']:
                    qty = player['holdings'][symbol]
                    settlement_price = prices.get(symbol, 0)
                    settlement_value = qty * settlement_price
                    player['capital'] += settlement_value
                    log_transaction(player_name, "Futures Settlement", symbol, qty, settlement_price, settlement_value)
                    del player['holdings'][symbol]
        game_state.futures_settled = True
        st.toast("Futures contracts settled!", icon="⚖️")

def check_margin_calls_and_orders(prices):
    game_state = get_game_state()
    for player_name, player in game_state.players.items():
        # Calculate portfolio value and margin requirements
        holdings_value = sum(prices.get(s, 0) * abs(q) for s, q in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        margin_required = sum(prices.get(s, 0) * abs(q) * game_state.current_margin_requirement 
                             for s, q in player['holdings'].items() if q < 0)  # Only for short positions
        
        # Check if margin call is needed
        if player['capital'] < margin_required * 0.5:  # If capital is less than 50% of required margin
            # Auto-liquidate some short positions
            for symbol, qty in list(player['holdings'].items()):
                if qty < 0:  # Short position
                    liquidate_qty = min(abs(qty), int(abs(qty) * 0.5))  # Liquidate 50% of position
                    if execute_trade(player_name, player, "Sell", symbol, liquidate_qty, prices, order_type="Auto-Liquidation"):
                        st.toast(f"Margin call: {player_name} liquidated {liquidate_qty} {symbol}", icon="⚠️")
                        break

def run_algo_strategies(prices):
    game_state = get_game_state()
    for player_name, player in game_state.players.items():
        if player['algo'] != 'Off' and game_state.game_status == "Running":
            # Track algo trades
            if player_name not in game_state.algo_trades:
                game_state.algo_trades[player_name] = []
                
            if player['algo'] == "Momentum Trader":
                # Buy assets that have risen, sell those that have fallen
                for symbol in NIFTY50_SYMBOLS[:3]:  # Limit to first 3 symbols for performance
                    if len(game_state.price_history) >= 2:
                        prev_price = game_state.price_history[-2].get(symbol, prices.get(symbol, 0))
                        curr_price = prices.get(symbol, 0)
                        if prev_price > 0:
                            change_pct = (curr_price - prev_price) / prev_price
                            if change_pct > 0.01:  # Up more than 1%
                                if random.random() < 0.3:  # 30% chance to trigger
                                    qty = max(1, int(player['capital'] * 0.1 / curr_price))
                                    execute_trade(player_name, player, "Buy", symbol, qty, prices, is_algo=True)
                            elif change_pct < -0.01:  # Down more than 1%
                                if random.random() < 0.3 and symbol in player['holdings'] and player['holdings'][symbol] > 0:
                                    qty = min(player['holdings'][symbol], max(1, int(player['holdings'][symbol] * 0.5)))
                                    execute_trade(player_name, player, "Sell", symbol, qty, prices, is_algo=True)
                                    
            elif player['algo'] == "Mean Reversion":
                # Buy assets that have fallen, sell those that have risen
                for symbol in NIFTY50_SYMBOLS[:3]:
                    hist_data = get_historical_data([symbol], period="1mo")
                    if not hist_data.empty:
                        current_price = prices.get(symbol, 0)
                        avg_price = hist_data.iloc[:, 0].mean()
                        if current_price > 0 and avg_price > 0:
                            deviation = (current_price - avg_price) / avg_price
                            if deviation < -0.05:  # 5% below average
                                if random.random() < 0.3:
                                    qty = max(1, int(player['capital'] * 0.1 / current_price))
                                    execute_trade(player_name, player, "Buy", symbol, qty, prices, is_algo=True)
                            elif deviation > 0.05 and symbol in player['holdings'] and player['holdings'][symbol] > 0:
                                if random.random() < 0.3:
                                    qty = min(player['holdings'][symbol], max(1, int(player['holdings'][symbol] * 0.5)))
                                    execute_trade(player_name, player, "Sell", symbol, qty, prices, is_algo=True)

def apply_difficulty_mechanics(prices):
    game_state = get_game_state()
    # Increase volatility and reduce liquidity at higher difficulty levels
    if game_state.difficulty_level > 1:
        for symbol in game_state.liquidity:
            game_state.liquidity[symbol] *= max(0.3, 1 - (game_state.difficulty_level - 1) * 0.2)

def auto_square_off_positions(prices):
    game_state = get_game_state()
    for player_name, player in game_state.players.items():
        for symbol, qty in list(player['holdings'].items()):
            if qty > 0:  # Long position
                execute_trade(player_name, player, "Sell", symbol, qty, prices, order_type="Auto-SquareOff")
            elif qty < 0:  # Short position  
                execute_trade(player_name, player, "Sell", symbol, abs(qty), prices, order_type="Auto-SquareOff")

def assign_teams(game_state):
    players = list(game_state.players.keys())
    random.shuffle(players)
    mid = len(players) // 2
    game_state.teams['A'] = players[:mid]
    game_state.teams['B'] = players[mid:]
    
    # Initialize team scores
    game_state.team_scores = {'A': 0, 'B': 0}

def calculate_team_scores(prices):
    game_state = get_game_state()
    team_scores = {'A': 0, 'B': 0}
    
    for team, players in game_state.teams.items():
        for player_name in players:
            if player_name in game_state.players:
                player = game_state.players[player_name]
                holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
                total_value = player['capital'] + holdings_value
                team_scores[team] += total_value
                
    game_state.team_scores = team_scores
    return team_scores

# --- UI Functions ---
def render_left_sidebar():
    game_state = get_game_state()
    
    with st.sidebar:
        st.title("🎮 Game Controls")
        
        # Player Registration
        if 'player' not in st.query_params:
            st.subheader("📝 Player Entry")
            player_name = st.text_input("Enter Your Name", key="name_input")
            mode = st.radio("Select Trading Mode", ["Trader", "HFT", "HNI"], 
                           help="Trader: Standard trading | HFT: High-frequency trading with rebates | HNI: High net worth with block deals")
            
            if st.button("Join Game", type="primary", use_container_width=True):
                if player_name and player_name.strip() and player_name not in game_state.players:
                    starting_capital = INITIAL_CAPITAL * 5 if mode == "HNI" else INITIAL_CAPITAL
                    game_state.players[player_name] = {
                        "name": player_name, "mode": mode, "capital": starting_capital, 
                        "holdings": {}, "pnl": 0, "leverage": 1.0, "margin_calls": 0, 
                        "pending_orders": [], "algo": "Off", "custom_algos": {},
                        "slippage_multiplier": 0.5 if mode == "HFT" else 1.0,
                        "value_history": [starting_capital], "trade_timestamps": [],
                        "hft_trade_count": 0
                    }
                    game_state.transactions[player_name] = []
                    st.query_params["player"] = player_name
                    st.rerun()
                else: 
                    st.error("Name is invalid or already taken!")
        else:
            player_name = st.query_params.get("player")
            st.success(f"👤 Logged in as **{player_name}**")
            if st.button("Logout", use_container_width=True):
                st.query_params.clear()
                st.rerun()

        st.markdown("---")
        
        # Admin Controls
        st.subheader("🔐 Admin Panel")
        password = st.text_input("Admin Password", type="password", key="admin_pass")
        
        if password == ADMIN_PASSWORD:
            st.session_state.role = 'admin'
            st.success("✅ Admin Access Granted")
            
        if st.session_state.get('role') == 'admin':
            if st.button("Logout Admin", use_container_width=True):
                del st.session_state['role']
                st.rerun()
                
            st.subheader("⚙️ Game Settings")
            
            # Game duration
            default_duration = int(getattr(game_state, 'round_duration_seconds', 1200) / 60)
            game_duration = st.number_input("Game Duration (minutes)", min_value=1, value=default_duration, 
                                          disabled=(game_state.game_status == "Running"))
            
            # Difficulty level
            difficulty_index = getattr(game_state, 'difficulty_level', 1) - 1
            game_state.difficulty_level = st.selectbox("Difficulty Level", [1, 2, 3], 
                                                     index=difficulty_index, 
                                                     format_func=lambda x: f"Level {x}",
                                                     disabled=(game_state.game_status == "Running"))
            
            # Game control buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Start Game", type="primary", use_container_width=True):
                    if game_state.players:
                        game_state.game_status = "Running"
                        game_state.game_start_time = time.time()
                        game_state.round_duration_seconds = game_duration * 60
                        game_state.futures_expiry_time = time.time() + (game_state.round_duration_seconds / 2)
                        st.toast("Game Started!", icon="🎉")
                        play_sound('opening_bell')
                        st.rerun()
                    else:
                        st.warning("Add at least one player to start.")
            
            with col2:
                if st.button("⏸️ Stop Game", use_container_width=True):
                    game_state.game_status = "Stopped"
                    st.toast("Game Paused!", icon="⏸️")
                    st.rerun()
                    
            if st.button("🔄 Reset Game", use_container_width=True):
                game_state.reset()
                st.toast("Game has been reset.", icon="🔄")
                st.rerun()
                
            if st.button("🎯 Assign Teams", use_container_width=True):
                assign_teams(game_state)
                st.success("Players randomly assigned to Team A and Team B!")
                
            # Circuit Breaker Control
            st.markdown("---")
            st.subheader("🚫 Circuit Breaker")
            if st.button("Halt Trading (1 min)", use_container_width=True):
                game_state.circuit_breaker_active = True
                game_state.circuit_breaker_end = time.time() + 60
                st.toast("Trading halted for 1 minute!", icon="🚫")
                
        elif password:
            st.error("❌ Incorrect Password")

def render_right_sidebar(prices):
    game_state = get_game_state()
    
    with st.sidebar:
        if st.session_state.get('role') == 'admin':
            st.title("👑 Admin Dashboard")
            
            # Algo Bot Analyzer
            st.subheader("🤖 Algo Bot Analyzer")
            if game_state.algo_pnl:
                algo_data = []
                for player_name, pnl in game_state.algo_pnl.items():
                    algo_data.append({
                        "Player": player_name,
                        "Algo P&L": format_indian_currency(pnl),
                        "Strategy": game_state.players[player_name].get('algo', 'Off')
                    })
                algo_df = pd.DataFrame(algo_data)
                st.dataframe(algo_df, use_container_width=True)
            else:
                st.info("No algo trading activity yet.")
                
            # Team Tournament View
            st.subheader("🏆 Team Tournament")
            if game_state.teams['A'] or game_state.teams['B']:
                team_scores = calculate_team_scores(prices)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Team A Score", format_indian_currency(team_scores['A']))
                    if game_state.teams['A']:
                        st.write("Players:", ", ".join(game_state.teams['A']))
                    
                with col2:
                    st.metric("Team B Score", format_indian_currency(team_scores['B']))
                    if game_state.teams['B']:
                        st.write("Players:", ", ".join(game_state.teams['B']))
                        
                # Determine leader
                if team_scores['A'] > team_scores['B']:
                    st.success("🏅 Team A is leading!")
                elif team_scores['B'] > team_scores['A']:
                    st.success("🏅 Team B is leading!")
                else:
                    st.info("Teams are tied!")
            else:
                st.info("No teams assigned yet. Use 'Assign Teams' in admin controls.")
                
        else:
            # Player right sidebar - Live Stats
            st.title("📊 Live Stats")
            
            if 'player' in st.query_params:
                player_name = st.query_params.get("player")
                if player_name in game_state.players:
                    player = game_state.players[player_name]
                    
                    # Advanced metrics
                    if len(player.get('value_history', [])) > 1:
                        sharpe = calculate_sharpe_ratio(player['value_history'])
                        max_dd = calculate_max_drawdown(player['value_history'])
                        sortino = calculate_sortino_ratio(player['value_history'])
                        
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                        st.metric("Max Drawdown", f"{max_dd:.1f}%")
                        st.metric("Sortino Ratio", f"{sortino:.2f}")
                    
                    # Market overview
                    st.subheader("🌡️ Market Health")
                    positive_changes = sum(1 for symbol in prices if 
                                         len(game_state.price_history) > 1 and 
                                         prices.get(symbol, 0) > game_state.price_history[-2].get(symbol, 0))
                    total_assets = len([s for s in prices if s not in FUTURES_SYMBOLS + LEVERAGED_ETFS + OPTION_SYMBOLS])
                    
                    st.metric("Advance/Decline", f"{positive_changes}/{total_assets}")
                    
                    # Circuit breaker status
                    if game_state.circuit_breaker_active:
                        remaining = max(0, game_state.circuit_breaker_end - time.time())
                        st.error(f"🚫 Trading Halted: {int(remaining)}s")
            
            # Global market sentiment
            st.subheader("📈 Market Sentiment")
            render_market_sentiment_meter()

def render_market_sentiment_meter():
    game_state = get_game_state()
    sentiments = [s for s in game_state.market_sentiment.values() if s != 0]
    if not sentiments:
        overall_sentiment = 0
    else:
        overall_sentiment = np.mean(sentiments)
    
    # Normalize sentiment to 0-100 scale
    normalized_sentiment = np.clip((overall_sentiment + 5) * 10, 0, 100)
    
    col1, col2, col3 = st.columns([1, 4, 1])
    with col1:
        st.write("<p style='text-align: right; margin-top: -5px; color: red;'>Fear</p>", unsafe_allow_html=True)
    with col2:
        st.progress(int(normalized_sentiment))
    with col3:
        st.write("<p style='margin-top: -5px; color: green;'>Greed</p>", unsafe_allow_html=True)
    
    st.caption(f"Sentiment: {normalized_sentiment:.1f}/100")

def render_main_interface(prices):
    game_state = get_game_state()
    
    # Inject Tone.js script for sounds
    st.components.v1.html('<script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.7.77/Tone.js"></script>', height=0)
    
    # Header with game status
    st.title(f"📈 {GAME_NAME}")
    
    # Game status and timer
    if game_state.game_status == "Running":
        remaining_time = max(0, game_state.round_duration_seconds - int(time.time() - game_state.game_start_time))
        if remaining_time == 0: 
            if game_state.game_status != "Finished": 
                play_sound('final_bell')
            game_state.game_status = "Finished"
        
        if remaining_time <= 30 and not getattr(game_state, 'closing_warning_triggered', False):
            play_sound('closing_warning')
            game_state.closing_warning_triggered = True
            
        # Display timer with color coding
        if remaining_time <= 60:
            st.error(f"⏰ **Time Remaining: {remaining_time // 60:02d}:{remaining_time % 60:02d}**")
        else:
            st.warning(f"⏰ **Time Remaining: {remaining_time // 60:02d}:{remaining_time % 60:02d}**")
            
        st.write(f"**Difficulty: Level {getattr(game_state, 'difficulty_level', 1)}**")
        
    elif game_state.game_status == "Stopped":
        st.info("⏸️ Game is paused. Press 'Start Game' to begin.")
    elif game_state.game_status == "Finished":
        st.success("🎉 Game has finished! See the final leaderboard below.")
    
    # Circuit breaker warning
    if game_state.circuit_breaker_active:
        remaining = max(0, game_state.circuit_breaker_end - time.time())
        st.error(f"🚫 **CIRCUIT BREAKER ACTIVE**: Trading halted for {int(remaining)} seconds")
    
    # Main content based on user role
    if st.session_state.get('role') == 'admin':
        render_global_views(prices, is_admin=True)
    elif 'player' in st.query_params:
        # Two-column layout for players
        col1, col2 = st.columns([1, 1])
        with col1:
            render_trade_execution_panel(prices)
        with col2:
            render_global_views(prices)
    else:
        # Welcome screen for spectators
        st.info("🎯 Welcome to BlockVista Market Frenzy! Please join the game from the sidebar to start trading.")
        render_global_views(prices)

def render_global_views(prices, is_admin=False):
    with st.container(border=True):
        st.subheader("🌐 Global Market View")
        
        # News feed
        st.markdown("---")
        st.subheader("📰 Live News Feed")
        game_state = get_game_state()
        news_feed = getattr(game_state, 'news_feed', [])
        if news_feed:
            for news in news_feed[:5]:  # Show only latest 5 news items
                st.info(news)
        else:
            st.info("No market news at the moment.")

        # Leaderboard
        st.markdown("---")
        st.subheader("🏆 Live Player Standings")
        render_leaderboard(prices)
        
        # Admin-only views
        if is_admin:
            st.markdown("---")
            st.subheader("📊 Live Player Performance")
            render_admin_performance_chart()

        # Market data table
        st.markdown("---")
        st.subheader("📊 Live Market Data")
        render_live_market_table(prices)

def render_admin_performance_chart():
    game_state = get_game_state()
    if not game_state.players:
        st.info("No players have joined yet.")
        return
        
    chart_data = {}
    for name, player_data in game_state.players.items():
        if player_data.get('value_history'):
            # Ensure all histories have the same length by padding with last value
            history = player_data['value_history']
            if history:
                chart_data[name] = history
    
    if chart_data:
        # Create DataFrame with proper alignment
        max_len = max(len(history) for history in chart_data.values())
        padded_data = {}
        for name, history in chart_data.items():
            if len(history) < max_len:
                # Pad with the last value
                padded_data[name] = history + [history[-1]] * (max_len - len(history))
            else:
                padded_data[name] = history
                
        df = pd.DataFrame(padded_data)
        st.line_chart(df)
    else:
        st.info("No trading activity yet to display.")

def render_trade_execution_panel(prices):
    game_state = get_game_state()
    
    with st.container(border=True):
        st.subheader("💻 Trade Execution Panel")
        acting_player = st.query_params.get("player")
        if not acting_player or acting_player not in game_state.players:
            st.warning("Please join the game to access your trading terminal.")
            return
        
        player = game_state.players[acting_player]
        st.markdown(f"**{acting_player}'s Terminal** · Mode: **{player['mode']}**")
        
        # Portfolio summary
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        starting_capital = INITIAL_CAPITAL * 5 if player['mode'] == 'HNI' else INITIAL_CAPITAL
        pnl = total_value - starting_capital
        player['pnl'] = pnl
        pnl_color = "green" if pnl >= 0 else "red"
        pnl_arrow = "🔼" if pnl >= 0 else "🔽"

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Cash", format_indian_currency(player['capital']))
        col2.metric("📊 Portfolio Value", format_indian_currency(total_value))
        col3.metric("📈 P&L", format_indian_currency(pnl), delta=format_indian_currency(pnl))
        col4.metric("🎯 Sharpe Ratio", f"{calculate_sharpe_ratio(player.get('value_history', [])):.2f}")

        # Trading interface tabs
        tabs = ["⚡ Trade Terminal", "🤖 Algo Trading", "📋 Transaction History", "📈 Strategy & Insights"]
        tab1, tab2, tab3, tab4 = st.tabs(tabs)
        
        is_trade_disabled = game_state.game_status != "Running" or game_state.circuit_breaker_active

        with tab1:
            render_trade_interface(acting_player, player, prices, is_trade_disabled)
        with tab2:
            render_algo_trading_tab(acting_player, player, is_trade_disabled)
        with tab3:
            render_transaction_history(acting_player)
        with tab4:
            render_strategy_tab(player, prices)

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
    
    # Holdings and pending orders
    col1, col2 = st.columns(2)
    with col1:
        render_current_holdings(player, prices)
    with col2:
        render_pending_orders(player)

def render_market_order_ui(player_name, player, prices, disabled):
    asset_types = ["Stock", "Crypto", "Gold", "Futures", "Leveraged ETF", "Option"]
    asset_type = st.radio("Asset Type", asset_types, horizontal=True, key=f"market_asset_{player_name}", disabled=disabled)
    
    # Futures expiry warning
    if asset_type == "Futures" and getattr(get_game_state(), 'futures_expiry_time', 0) > 0:
        expiry_remaining = max(0, get_game_state().futures_expiry_time - time.time())
        st.warning(f"⏳ Futures expire in **{int(expiry_remaining // 60)}m {int(expiry_remaining % 60)}s**")

    col1, col2 = st.columns([2, 1])
    with col1:
        if asset_type == "Stock": 
            symbol_choice = st.selectbox("Stock", [s.replace('.NS', '') for s in NIFTY50_SYMBOLS], 
                                       key=f"market_stock_{player_name}", disabled=disabled) + '.NS'
        elif asset_type == "Crypto": 
            symbol_choice = st.selectbox("Cryptocurrency", CRYPTO_SYMBOLS, 
                                       key=f"market_crypto_{player_name}", disabled=disabled)
        elif asset_type == "Gold": 
            symbol_choice = GOLD_SYMBOL
        elif asset_type == "Futures": 
            symbol_choice = st.selectbox("Futures", FUTURES_SYMBOLS, 
                                       key=f"market_futures_{player_name}", disabled=disabled)
        elif asset_type == "Leveraged ETF": 
            symbol_choice = st.selectbox("Leveraged ETF", LEVERAGED_ETFS, 
                                       key=f"market_letf_{player_name}", disabled=disabled)
        else: 
            symbol_choice = st.selectbox("Option", OPTION_SYMBOLS, 
                                       key=f"market_option_{player_name}", disabled=disabled)
    with col2:
        qty = st.number_input("Quantity", min_value=1, step=1, value=1, 
                            key=f"market_qty_{player_name}", disabled=disabled)
    
    # Display bid-ask spread
    mid_price = prices.get(symbol_choice, 0)
    ask_price = mid_price * (1 + get_game_state().bid_ask_spread / 2)
    bid_price = mid_price * (1 - get_game_state().bid_ask_spread / 2)
    st.info(f"💱 Bid: {format_indian_currency(bid_price)} | Ask: {format_indian_currency(ask_price)}")

    # Trade buttons
    col1, col2, col3 = st.columns(3)
    if col1.button(f"🟢 Buy {qty}", key=f"buy_{player_name}", use_container_width=True, 
                  disabled=disabled, type="primary"): 
        if execute_trade(player_name, player, "Buy", symbol_choice, qty, prices): 
            play_sound('success')
        else: 
            play_sound('error')
        st.rerun()
        
    if col2.button(f"🔴 Sell {qty}", key=f"sell_{player_name}", use_container_width=True, disabled=disabled): 
        if execute_trade(player_name, player, "Sell", symbol_choice, qty, prices): 
            play_sound('success')
        else: 
            play_sound('error')
        st.rerun()
        
    if col3.button(f"⚫ Short {qty}", key=f"short_{player_name}", use_container_width=True, disabled=disabled): 
        if execute_trade(player_name, player, "Short", symbol_choice, qty, prices): 
            play_sound('success')
        else: 
            play_sound('error')
        st.rerun()

def render_limit_order_ui(player_name, player, prices, disabled):
    st.write("Set a price to automatically buy or sell an asset.")
    asset_types = ["Stock", "Crypto", "Gold", "Futures", "Leveraged ETF", "Option"]
    asset_type = st.radio("Asset Type", asset_types, horizontal=True, key=f"limit_asset_{player_name}", disabled=disabled)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if asset_type == "Stock": 
            symbol_choice = st.selectbox("Stock", [s.replace('.NS', '') for s in NIFTY50_SYMBOLS], 
                                       key=f"limit_stock_{player_name}", disabled=disabled) + '.NS'
        elif asset_type == "Crypto": 
            symbol_choice = st.selectbox("Cryptocurrency", CRYPTO_SYMBOLS, 
                                       key=f"limit_crypto_{player_name}", disabled=disabled)
        elif asset_type == "Gold": 
            symbol_choice = GOLD_SYMBOL
        elif asset_type == "Futures": 
            symbol_choice = st.selectbox("Futures", FUTURES_SYMBOLS, 
                                       key=f"limit_futures_{player_name}", disabled=disabled)
        elif asset_type == "Leveraged ETF": 
            symbol_choice = st.selectbox("Leveraged ETF", LEVERAGED_ETFS, 
                                       key=f"limit_letf_{player_name}", disabled=disabled)
        else: 
            symbol_choice = st.selectbox("Option", OPTION_SYMBOLS, 
                                       key=f"limit_option_{player_name}", disabled=disabled)
    with col2:
        qty = st.number_input("Quantity", min_value=1, step=1, value=1, 
                            key=f"limit_qty_{player_name}", disabled=disabled)
    
    current_price = prices.get(symbol_choice, 0)
    limit_price = st.number_input("Limit Price", min_value=0.01, value=current_price, step=0.01, 
                                key=f"limit_price_{player_name}", disabled=disabled)
    
    st.info(f"Current Price: {format_indian_currency(current_price)}")
    
    col1, col2, col3 = st.columns(3)
    if col1.button("🟢 Buy Limit", key=f"buy_limit_{player_name}", use_container_width=True, disabled=disabled, type="primary"):
        player['pending_orders'].append({'type': 'Limit', 'action': 'Buy', 'symbol': symbol_choice, 'qty': qty, 'price': limit_price})
        st.success("✅ Limit Buy order placed!")
        st.rerun()
        
    if col2.button("🔴 Sell Limit", key=f"sell_limit_{player_name}", use_container_width=True, disabled=disabled):
        player['pending_orders'].append({'type': 'Limit', 'action': 'Sell', 'symbol': symbol_choice, 'qty': qty, 'price': limit_price})
        st.success("✅ Limit Sell order placed!")
        st.rerun()
        
    if col3.button("⚫ Short Limit", key=f"short_limit_{player_name}", use_container_width=True, disabled=disabled):
        player['pending_orders'].append({'type': 'Limit', 'action': 'Short', 'symbol': symbol_choice, 'qty': qty, 'price': limit_price})
        st.success("✅ Limit Short order placed!")
        st.rerun()

def render_stop_loss_order_ui(player_name, player, prices, disabled):
    st.write("Set a price to automatically sell an asset if it drops, to limit losses.")
    asset_types = ["Stock", "Crypto", "Gold", "Futures", "Leveraged ETF", "Option"]
    asset_type = st.radio("Asset Type", asset_types, horizontal=True, key=f"stop_asset_{player_name}", disabled=disabled)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        if asset_type == "Stock": 
            symbol_choice = st.selectbox("Stock", [s.replace('.NS', '') for s in NIFTY50_SYMBOLS], 
                                       key=f"stop_stock_{player_name}", disabled=disabled) + '.NS'
        elif asset_type == "Crypto": 
            symbol_choice = st.selectbox("Cryptocurrency", CRYPTO_SYMBOLS, 
                                       key=f"stop_crypto_{player_name}", disabled=disabled)
        elif asset_type == "Gold": 
            symbol_choice = GOLD_SYMBOL
        elif asset_type == "Futures": 
            symbol_choice = st.selectbox("Futures", FUTURES_SYMBOLS, 
                                       key=f"stop_futures_{player_name}", disabled=disabled)
        elif asset_type == "Leveraged ETF": 
            symbol_choice = st.selectbox("Leveraged ETF", LEVERAGED_ETFS, 
                                       key=f"stop_letf_{player_name}", disabled=disabled)
        else: 
            symbol_choice = st.selectbox("Option", OPTION_SYMBOLS, 
                                       key=f"stop_option_{player_name}", disabled=disabled)
    with col2:
        qty = st.number_input("Quantity", min_value=1, step=1, value=1, 
                            key=f"stop_qty_{player_name}", disabled=disabled)
    
    current_price = prices.get(symbol_choice, 0)
    stop_price = st.number_input("Stop Price", min_value=0.01, value=current_price * 0.95, step=0.01, 
                               key=f"stop_price_{player_name}", disabled=disabled)
    
    st.info(f"Current Price: {format_indian_currency(current_price)} | Stop at: {format_indian_currency(stop_price)}")
    
    if st.button("🛑 Set Stop-Loss", key=f"set_stop_{player_name}", disabled=disabled, use_container_width=True):
        player['pending_orders'].append({'type': 'Stop-Loss', 'action': 'Sell', 'symbol': symbol_choice, 'qty': qty, 'price': stop_price})
        st.success("✅ Stop-Loss order placed!")
        st.rerun()

def render_pending_orders(player):
    st.subheader("🕒 Pending Orders")
    if player['pending_orders']:
        orders_df = pd.DataFrame(player['pending_orders'])
        st.dataframe(orders_df, use_container_width=True)
        
        if st.button("Clear All Orders", type="secondary"):
            player['pending_orders'] = []
            st.rerun()
    else:
        st.info("No pending orders.")

def render_algo_trading_tab(player_name, player, disabled):
    st.subheader("🤖 Automated Trading Strategies")
    
    default_strats = ["Off", "Momentum Trader", "Mean Reversion", "Volatility Breakout", "Value Investor"]
    custom_strats = list(player.get('custom_algos', {}).keys())
    all_strats = default_strats + custom_strats
    
    active_algo = player.get('algo', 'Off')
    player['algo'] = st.selectbox("Choose Strategy", all_strats, 
                                index=all_strats.index(active_algo) if active_algo in all_strats else 0, 
                                disabled=disabled, key=f"algo_{player_name}")
    
    # Strategy descriptions
    if player['algo'] in default_strats and player['algo'] != 'Off': 
        if player['algo'] == "Momentum Trader": 
            st.info("📈 This bot buys assets that have risen in price and sells those that have fallen, betting that recent trends will continue.")
        elif player['algo'] == "Mean Reversion": 
            st.info("🔄 This bot buys assets that have recently fallen and sells those that have risen, betting on a return to their average price.")
        elif player['algo'] == "Volatility Breakout": 
            st.info("🌪️ This bot identifies assets making significant price moves and trades in the same direction, aiming to capitalize on strong momentum.")
        elif player['algo'] == "Value Investor": 
            st.info("💰 This bot looks for assets that have dropped significantly and buys them, operating on the principle of buying undervalued assets.")

    # Custom strategy creation
    with st.expander("🛠️ Create Custom Strategy"):
        st.markdown("##### Define Your Own Algorithm")
        col1, col2 = st.columns(2)
        with col1:
            algo_name = st.text_input("Strategy Name", key=f"algo_name_{player_name}")
            indicator = st.selectbox("Indicator", ["Price Change % (5-day)", "SMA Crossover (10/20)", "Price Change % (30-day)"], 
                                   key=f"indicator_{player_name}")
            condition = st.selectbox("Condition", ["Greater Than", "Less Than"], 
                                   key=f"condition_{player_name}")
        with col2:
            threshold = st.number_input("Threshold Value", value=0.0, step=0.1, 
                                      key=f"threshold_{player_name}")
            action = st.radio("Action if True", ["Buy", "Sell"], 
                            key=f"algo_action_{player_name}")
            
        if st.button("💾 Save Strategy", key=f"save_algo_{player_name}"):
            if algo_name.strip():
                player['custom_algos'][algo_name] = {
                    "indicator": indicator, 
                    "condition": condition, 
                    "threshold": threshold, 
                    "action": action
                }
                st.success(f"✅ Custom strategy '{algo_name}' saved!")
                st.rerun()
            else: 
                st.error("❌ Strategy name cannot be empty.")

def render_current_holdings(player, prices):
    st.subheader("💼 Portfolio Allocation")
    if player['holdings']:
        # Create holdings data
        holdings_data = []
        for sym, qty in player['holdings'].items():
            if qty != 0:  # Only include non-zero holdings
                price = prices.get(sym, 0)
                value = price * abs(qty)
                holdings_data.append({
                    "Symbol": sym, 
                    "Quantity": qty,
                    "Price": price,
                    "Value": value
                })
        
        if holdings_data:
            holdings_df = pd.DataFrame(holdings_data)
            
            # Display as table
            st.dataframe(holdings_df.style.format({
                "Price": format_indian_currency,
                "Value": format_indian_currency
            }), use_container_width=True)
            
            # Pie chart
            fig = go.Figure(data=[go.Pie(
                labels=holdings_df['Symbol'], 
                values=holdings_df['Value'], 
                hole=.3,
                textinfo='label+percent'
            )])
            fig.update_layout(showlegend=False, height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No holdings yet.")
    else: 
        st.info("No holdings yet.")

def render_transaction_history(player_name):
    game_state = get_game_state()
    st.subheader("📋 Transaction History")
    if game_state.transactions.get(player_name):
        trans_df = pd.DataFrame(game_state.transactions[player_name], 
                              columns=["Time", "Action", "Symbol", "Qty", "Price", "Total"])
        st.dataframe(trans_df.style.format({
            "Price": format_indian_currency, 
            "Total": format_indian_currency
        }), use_container_width=True, height=400)
    else: 
        st.info("No transactions recorded.")

def render_strategy_tab(player, prices):
    st.subheader("📊 Strategy & Insights")
    tab1, tab2, tab3, tab4 = st.tabs(["Performance Chart", "Technical Analysis", "Portfolio Optimizer", "Risk Metrics"])
    
    with tab1:
        st.markdown("##### Portfolio Value Over Time")
        if len(player.get('value_history', [])) > 1:
            # Create time indices for x-axis
            time_points = list(range(len(player['value_history'])))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=time_points, 
                y=player['value_history'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color='#00D4AA', width=3)
            ))
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Portfolio Value",
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Trade more to see your performance chart.")
            
    with tab2: 
        render_sma_chart(player['holdings'])
    with tab3: 
        render_optimizer(player['holdings'])
    with tab4:
        render_risk_metrics(player)

def render_sma_chart(holdings):
    st.markdown("##### Simple Moving Average (SMA) Chart")
    chartable_assets = [s for s in holdings.keys() if s not in OPTION_SYMBOLS + FUTURES_SYMBOLS + LEVERAGED_ETFS]
    if not chartable_assets: 
        st.info("No chartable assets in portfolio to analyze.")
        return
        
    chart_symbol = st.selectbox("Select Asset to Chart", chartable_assets)
    hist_data = get_historical_data([chart_symbol], period="6mo")
    if not hist_data.empty:
        df = hist_data.rename(columns={chart_symbol: 'Close'})
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price', line=dict(color='#1f77b4')))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='20-Day SMA', line=dict(color='#ff7f0e')))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='#2ca02c')))
        
        fig.update_layout(
            title=f"{chart_symbol} Price with Moving Averages",
            xaxis_title="Date",
            yaxis_title="Price",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    else: 
        st.warning(f"Could not load historical data for {chart_symbol}.")

def render_optimizer(holdings):
    st.subheader("Portfolio Optimization (Max Sharpe Ratio)")
    if st.button("Optimize My Portfolio"):
        weights, performance = optimize_portfolio(holdings)
        if weights:
            st.success("✅ Optimal weights for max risk-adjusted return:")
            
            # Display weights in a nice format
            weight_data = []
            for symbol, weight in weights.items():
                if weight > 0.01:  # Only show significant weights
                    weight_data.append({"Symbol": symbol, "Weight": f"{weight:.2%}"})
            
            if weight_data:
                weight_df = pd.DataFrame(weight_data)
                st.dataframe(weight_df, use_container_width=True)
            
            if performance:
                col1, col2, col3 = st.columns(3)
                col1.metric("Expected Return", f"{performance[0]:.2%}")
                col2.metric("Expected Volatility", f"{performance[1]:.2%}")
                col3.metric("Sharpe Ratio", f"{performance[2]:.2f}")
        else:
            st.error(performance)

def render_risk_metrics(player):
    st.subheader("📉 Risk Analysis")
    
    if len(player.get('value_history', [])) > 1:
        sharpe = calculate_sharpe_ratio(player['value_history'])
        max_dd = calculate_max_drawdown(player['value_history'])
        sortino = calculate_sortino_ratio(player['value_history'])
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col2.metric("Max Drawdown", f"{max_dd:.1f}%")
        col3.metric("Sortino Ratio", f"{sortino:.2f}")
        
        # Drawdown chart
        st.markdown("##### Drawdown Analysis")
        values = player['value_history']
        peaks = []
        drawdowns = []
        peak = values[0]
        
        for value in values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            peaks.append(peak)
            drawdowns.append(drawdown)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=drawdowns,
            mode='lines',
            name='Drawdown %',
            line=dict(color='red', width=2),
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.1)'
        ))
        fig.update_layout(
            xaxis_title="Time",
            yaxis_title="Drawdown %",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need more trading history to calculate risk metrics.")

def render_leaderboard(prices):
    game_state = get_game_state()
    lb = []
    for pname, pdata in game_state.players.items():
        holdings_value = sum(prices.get(s, 0) * q for s, q in pdata['holdings'].items())
        total_value = pdata['capital'] + holdings_value
        starting_capital = INITIAL_CAPITAL * 5 if pdata['mode'] == 'HNI' else INITIAL_CAPITAL
        pnl = total_value - starting_capital
        sharpe_ratio = calculate_sharpe_ratio(pdata.get('value_history', []))
        max_dd = calculate_max_drawdown(pdata.get('value_history', []))
        
        lb.append((pname, pdata['mode'], total_value, pnl, sharpe_ratio, max_dd))
    
    if lb:
        lb_df = pd.DataFrame(lb, columns=["Player", "Mode", "Portfolio Value", "P&L", "Sharpe Ratio", "Max DD %"])
        lb_df = lb_df.sort_values("Portfolio Value", ascending=False).reset_index(drop=True)
        
        # Apply styling
        styled_df = lb_df.style.format({
            "Portfolio Value": format_indian_currency,
            "P&L": format_indian_currency,
            "Sharpe Ratio": "{:.2f}",
            "Max DD %": "{:.1f}%"
        }).apply(lambda x: ['background-color: #e6f7ff' if x.name == 0 else '' for i in x], axis=1)
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Game finished logic
        if game_state.game_status == "Finished":
            if not getattr(game_state, 'auto_square_off_complete', False):
                auto_square_off_positions(prices)
                game_state.auto_square_off_complete = True
                st.rerun()

            # Celebration and winners
            st.balloons()
            winner = lb_df.iloc[0]
            st.success(f"🎉 The winner is **{winner['Player']}**! 🎉")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🏆 Final Portfolio Value", format_indian_currency(winner['Portfolio Value']))
            col2.metric("💰 Total P&L", format_indian_currency(winner['P&L']))
            col3.metric("⚡ Sharpe Ratio", f"{winner['Sharpe Ratio']:.2f}")

            # Prudent investor award (best Sharpe ratio)
            prudent_winner = lb_df.sort_values("Sharpe Ratio", ascending=False).iloc[0]
            st.info(f"🧐 The Prudent Investor Award goes to **{prudent_winner['Player']}** with a Sharpe Ratio of **{prudent_winner['Sharpe Ratio']:.2f}**!")

def render_live_market_table(prices):
    game_state = get_game_state()
    prices_df = pd.DataFrame(prices.items(), columns=['Symbol', 'Price'])
    
    # Calculate price changes
    if len(game_state.price_history) >= 2:
        prev_prices = game_state.price_history[-2]
        prices_df['prev_price'] = prices_df['Symbol'].map(prev_prices).fillna(prices_df['Price'])
        prices_df['Change'] = prices_df['Price'] - prices_df['prev_price']
        prices_df['Change %'] = (prices_df['Change'] / prices_df['prev_price']) * 100
    else: 
        prices_df['Change'] = 0.0
        prices_df['Change %'] = 0.0
    
    prices_df.drop(columns=['prev_price'], inplace=True, errors='ignore')

    # Get latest trades for each symbol
    all_trades = [[player] + t for player, transactions in game_state.transactions.items() for t in transactions]
    if all_trades:
        feed_df = pd.DataFrame(all_trades, columns=["Player", "Time", "Action", "Symbol", "Qty", "Trade Price", "Total"])
        last_trades = feed_df.sort_values('Time').groupby('Symbol').last()
        last_trades['Last Order'] = last_trades.apply(
            lambda r: f"{r['Player']} {r['Action']} {r['Qty']} @ {format_indian_currency(r['Trade Price'])}", 
            axis=1
        )
        prices_df = pd.merge(prices_df, last_trades[['Last Order']], on='Symbol', how='left')
    else: 
        prices_df['Last Order'] = '-'
    
    prices_df.fillna({'Last Order': '-'}, inplace=True)
    
    # Apply styling
    def color_change(val):
        if val > 0: return 'color: green'
        elif val < 0: return 'color: red'
        return ''
    
    styled_df = prices_df.style.applymap(color_change, subset=['Change', 'Change %']).format({
        'Price': format_indian_currency,
        'Change': lambda v: f"{format_indian_currency(v) if v != 0 else '-'}",
        'Change %': lambda v: f"{v:+.2f}%" if v != 0 else "-"
    })
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

# --- Main Application Loop ---
def main():
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.role = 'player'
    
    # Initialize game state
    game_state = get_game_state()
    
    # Three-column layout: left sidebar, main content, right sidebar
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        render_left_sidebar()
    
    with col2:
        # --- Main Price Flow ---
        # Fetch base prices only if they haven't been fetched for the day
        if not game_state.base_real_prices:
            game_state.base_real_prices = get_daily_base_prices()
            st.toast("📊 Fetched daily base market prices.")

        # Use the last known price or the daily base price to start the tick simulation
        last_prices = game_state.prices if game_state.prices else game_state.base_real_prices
        
        # 1. Simulate small price fluctuations for the current tick
        current_prices = simulate_tick_prices(last_prices)
        
        # 2. Calculate prices for simulated assets (Futures, ETFs, Options) based on the new simulated prices
        prices_with_derivatives = calculate_derived_prices(current_prices)
        
        # 3. Apply temporary simulation effects like market events
        final_prices = run_game_tick(prices_with_derivatives)
        
        game_state.prices = final_prices
        
        # Update price history
        if not isinstance(game_state.price_history, list): 
            game_state.price_history = []
        game_state.price_history.append(final_prices)
        if len(game_state.price_history) > 10: 
            game_state.price_history.pop(0)
        
        # Check circuit breaker expiry
        if game_state.circuit_breaker_active and time.time() >= game_state.circuit_breaker_end:
            game_state.circuit_breaker_active = False
            st.toast("✅ Circuit breaker lifted - Trading resumed!", icon="✅")
        
        # Render main interface
        render_main_interface(final_prices)
    
    with col3:
        render_right_sidebar(final_prices)
    
    # Auto-refresh logic
    if game_state.game_status == "Running":
        time.sleep(1)  # Fast refresh during active game
        st.rerun()
    else:
        time.sleep(3)  # Slower refresh for lobby/stopped state
        st.rerun()

if __name__ == "__main__":
    main()

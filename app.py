import streamlit as st
import pandas as pd
import yfinance as yf
import time
import random
import numpy as np
import plotly.graph_objects as go
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="BlockVista Market Frenzy", page_icon="📈")

# --- Global Game Configuration ---
GAME_NAME = "BlockVista Market Frenzy"
INITIAL_CAPITAL = 1000000  # ₹10L
ADMIN_PASSWORD = "100370"

# Asset Definitions
NIFTY50_SYMBOLS = ['RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS',
                   'KOTAKBANK.NS', 'SBIN.NS', 'ITC.NS', 'ASIANPAINT.NS', 'AXISBANK.NS']
CRYPTO_SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'ADA-USD', 'XRP-USD']
GOLD_SYMBOL = 'GC=F'
NIFTY_INDEX_SYMBOL = '^NSEI'
BANKNIFTY_INDEX_SYMBOL = '^NSEBANK'
FUTURES_SYMBOLS = ['NIFTY-FUT', 'BANKNIFTY-FUT']
LEVERAGED_ETFS = ['NIFTY_BULL_3X', 'NIFTY_BEAR_3X']
OPTION_SYMBOLS = ['NIFTY_CALL', 'NIFTY_PUT']

ALL_SYMBOLS = NIFTY50_SYMBOLS + CRYPTO_SYMBOLS + [GOLD_SYMBOL] + OPTION_SYMBOLS + FUTURES_SYMBOLS + LEVERAGED_ETFS + [NIFTY_INDEX_SYMBOL, BANKNIFTY_INDEX_SYMBOL]

# Pre-built News Headlines
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
    {"headline": "Regulatory probe launched into {symbol} over accounting irregularities.", "impact": "Symbol Crash"},
    {"headline": "{symbol} announces a surprise stock split, shares to adjust at market open.", "impact": "Stock Split"},
]

# --- Game State Management (Singleton for Live Sync) ---
class GameState:
    def __init__(self):
        self.players = {}
        self.game_status = "Stopped"
        self.game_start_time = 0
        self.round_duration_seconds = 20 * 60
        self.futures_expiry_time = 0
        self.prices = {}
        self.base_real_prices = {}
        self.price_history = []
        self.transactions = {}
        self.market_sentiment = {s: 0 for s in ALL_SYMBOLS}
        self.liquidity = {s: random.uniform(0.5, 1.0) for s in ALL_SYMBOLS}
        self.event_active = False
        self.event_end = 0
        self.news_feed = []
        self.auto_square_off_complete = False
        self.closing_warning_triggered = False
        self.difficulty_level = 1
        self.current_margin_requirement = 0.2
        self.bid_ask_spread = 0.001
        self.slippage_threshold = 10
        self.base_slippage_rate = 0.005
        self.hft_rebate_window = 60
        self.hft_rebate_trades = 5
        self.hft_rebate_amount = 5000
        self.teams = {'A': [], 'B': []}
        self.admin_log_messages = []
        self.is_sound_enabled = True
        self.opening_bell_rung = False

    def reset(self):
        base_prices = self.base_real_prices
        difficulty = self.difficulty_level
        self.__init__()
        self.base_real_prices = base_prices
        self.difficulty_level = difficulty
        self.is_sound_enabled = True

@st.cache_resource
def get_game_state():
    return GameState()

# --- Sound Effects & Notifications ---
def play_sound(sound_type):
    js = ""
    if sound_type == 'opening_bell':
        js = """
        <script>
            if (typeof Tone !== 'undefined') {
                const synth = new Tone.Synth().toDestination();
                const now = Tone.now();
                synth.triggerAttackRelease("G4", "8n", now);
                synth.triggerAttackRelease("C5", "8n", now + 0.2);
                synth.triggerAttackRelease("E5", "8n", now + 0.4);
                synth.triggerAttackRelease("G5", "8n", now + 0.6);
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
    elif sound_type == 'success':
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
    if js:
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

# --- Data Fetching & Simulation ---
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
        current_bull = game_state.prices.get('NIFTY_BULL_3X', nifty_price / 100)
        current_bear = game_state.prices.get('NIFTY_BEAR_3X', nifty_price / 100)
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

# --- Core Game Logic Functions ---
def apply_event_adjustment(prices, event_type, target_symbol=None):
    adjusted_prices = prices.copy()
    game_state = get_game_state()
    event_messages = {
        "Flash Crash": "⚡ Flash Crash! Prices dropping sharply.",
        "Bull Rally": "📈 Bull Rally! All prices surging.",
        "Banking Boost": "🏦 Banking Boost! Banking stocks rallying.",
        "Sector Rotation": "🔄 Sector Rotation! Tech up, Banks down.",
        "Symbol Bull Run": f"🚀 {target_symbol} Bull Run! Price soaring.",
        "Symbol Crash": f"💥 {target_symbol} Crash! Price plummeting.",
        "Volatility Spike": "🌪️ Volatility Spike! Prices fluctuating wildly.",
        "Stock Split": f"🔀 Stock Split for {target_symbol}!",
        "Dividend": f"💰 Dividend declared for {target_symbol}!"
    }
    st.toast(event_messages[event_type], icon="📢")
    game_state.news_feed.insert(0, f"📢 {time.strftime('%H:%M:%S')} - {event_messages[event_type]}")
    if len(game_state.news_feed) > 5: game_state.news_feed.pop()
    if event_type == "Flash Crash": adjusted_prices = {k: v * random.uniform(0.95, 0.98) for k, v in adjusted_prices.items()}
    elif event_type == "Bull Rally": adjusted_prices = {k: v * random.uniform(1.02, 1.05) for k, v in adjusted_prices.items()}
    elif event_type == "Banking Boost":
        for sym in ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'AXISBANK.NS']:
            if sym in adjusted_prices: adjusted_prices[sym] *= 1.07
    elif event_type == "Sector Rotation":
        for sym in ['HDFCBANK.NS', 'ICICIBANK.NS']:
            if sym in adjusted_prices: adjusted_prices[sym] *= 0.95
        for sym in ['INFY.NS', 'TCS.NS']:
            if sym in adjusted_prices: adjusted_prices[sym] *= 1.10
    elif event_type == "Symbol Bull Run" and target_symbol: adjusted_prices[target_symbol] *= 1.15
    elif event_type == "Symbol Crash" and target_symbol: adjusted_prices[target_symbol] *= 0.85
    elif event_type == "Stock Split" and target_symbol:
        adjusted_prices[target_symbol] /= 2
        for player in game_state.players.values():
            if target_symbol in player['holdings']: player['holdings'][target_symbol] *= 2
    elif event_type == "Dividend" and target_symbol:
        dividend_per_share = adjusted_prices[target_symbol] * 0.01
        for player in game_state.players.values():
            if target_symbol in player['holdings'] and player['holdings'][target_symbol] > 0:
                dividend_received = dividend_per_share * player['holdings'][target_symbol]
                player['capital'] += dividend_received
                log_transaction(player['name'], "Dividend", target_symbol, player['holdings'][target_symbol], dividend_per_share, dividend_received)
    return adjusted_prices

def execute_trade(player_name, player, action, symbol, qty, prices, is_algo=False, order_type="Market"):
    game_state = get_game_state()
    mid_price = prices.get(symbol, 0)
    if mid_price == 0: return False
    
    trade_price = mid_price * (1 + game_state.bid_ask_spread / 2) if action == "Buy" else mid_price * (1 - game_state.bid_ask_spread / 2)
    cost = trade_price * qty
    
    if player['mode'] == "HFT":
        player['trade_timestamps'].append(time.time())
        player['trade_timestamps'] = [t for t in player['trade_timestamps'] if time.time() - t <= game_state.hft_rebate_window]
        if len(player['trade_timestamps']) >= game_state.hft_rebate_trades:
            player['capital'] += game_state.hft_rebate_amount
            st.toast(f"HFT Rebate: {format_indian_currency(game_state.hft_rebate_amount)} added to your capital!", icon="💸")
            player['trade_timestamps'] = []
    
    trade_executed = False
    if action == "Buy" and player['capital'] >= cost:
        player['capital'] -= cost; player['holdings'][symbol] = player['holdings'].get(symbol, 0) + qty; trade_executed = True
    elif action == "Short" and player['capital'] >= cost * game_state.current_margin_requirement:
        player['capital'] += cost; player['holdings'][symbol] = player['holdings'].get(symbol, 0) - qty; trade_executed = True
    elif action == "Sell":
        current_qty = player['holdings'].get(symbol, 0)
        if current_qty > 0 and current_qty >= qty:
            player['capital'] += cost; player['holdings'][symbol] -= qty; trade_executed = True
        elif current_qty < 0 and abs(current_qty) >= qty:
            player['capital'] -= cost; player['holdings'][symbol] += qty; trade_executed = True
        if trade_executed and player['holdings'][symbol] == 0: del player['holdings'][symbol]
    
    if trade_executed:
        game_state.market_sentiment[symbol] = game_state.market_sentiment.get(symbol, 0) + (qty / 50) * (1 if action in ["Buy", "Short"] else -1)
        log_transaction(player_name, f"{order_type} {action}", symbol, qty, trade_price, cost, is_algo)
    elif not is_algo:
        st.error("Trade failed: Insufficient capital or holdings.")
    return trade_executed

def run_game_tick(prices):
    game_state = get_game_state()
    if game_state.game_status != "Running":
        return prices
    
    # Auto-trade execution and checks
    process_pending_orders(prices)
    check_margin_calls_and_orders(prices)
    run_algo_strategies(prices)
    handle_futures_expiry(prices)
    handle_hni_block_deals()
    
    # Market mechanics
    for symbol in game_state.market_sentiment:
        game_state.market_sentiment[symbol] *= 0.95

    # Random News Event Trigger
    if not game_state.event_active and random.random() < 0.05:
        news_item = random.choice(PRE_BUILT_NEWS)
        headline = news_item['headline']
        impact = news_item['impact']
        target_symbol = None
        if "{symbol}" in headline:
            target_symbol = random.choice(NIFTY50_SYMBOLS)
            headline = headline.format(symbol=target_symbol.replace(".NS", ""))
        game_state.event_type = impact
        game_state.event_target_symbol = target_symbol
        game_state.event_active = True
        game_state.event_end = time.time() + random.randint(30, 60)
        announce_news(headline)
    
    if game_state.event_active and time.time() >= game_state.event_end:
        game_state.event_active = False; st.info("Market event has ended.")
    if game_state.event_active:
        prices = apply_event_adjustment(prices, game_state.event_type, getattr(game_state, 'event_target_symbol', None))

    # Record portfolio value history for all players
    for player in game_state.players.values():
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        player['value_history'].append(total_value)

    return prices

def process_pending_orders(prices):
    game_state = get_game_state()
    for name, player in game_state.players.items():
        orders_to_remove = []
        for i, order in enumerate(player['pending_orders']):
            current_price = prices.get(order['symbol'], 0)
            if current_price == 0: continue
            order_executed = False
            if order['type'] == 'Limit':
                if order['action'] == 'Buy' and current_price <= order['price']:
                    order_executed = execute_trade(name, player, 'Buy', order['symbol'], order['qty'], prices, order_type="Limit")
                elif (order['action'] == 'Sell' or order['action'] == 'Short') and current_price >= order['price']:
                    order_executed = execute_trade(name, player, order['action'], order['symbol'], order['qty'], prices, order_type="Limit")
            elif order['type'] == 'Stop-Loss' and current_price <= order['price']:
                order_executed = execute_trade(name, player, 'Sell', order['symbol'], order['qty'], prices, order_type="Stop-Loss")
            if order_executed: orders_to_remove.append(i)
        for i in sorted(orders_to_remove, reverse=True): del player['pending_orders'][i]

def check_margin_calls_and_orders(prices):
    game_state = get_game_state()
    if game_state.difficulty_level >= 3:
        for name, player in game_state.players.items():
            holdings_value = sum(prices.get(symbol, 0) * abs(qty) for symbol, qty in player['holdings'].items())
            if holdings_value == 0: continue
            total_value = player['capital'] + sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
            margin_needed = holdings_value * game_state.current_margin_requirement
            if total_value < margin_needed:
                player['margin_calls'] += 1
                st.warning(f"⚠️ MARGIN CALL for {name}! Liquidating largest position.", icon="⚠️")
                largest_position = max(player['holdings'].items(), key=lambda item: abs(item[1] * prices.get(item[0], 0)), default=(None, 0))
                if largest_position[0]:
                    symbol_to_liquidate, qty_to_liquidate = largest_position
                    action = "Sell" if qty_to_liquidate > 0 else "Buy"
                    execute_trade(name, player, action, symbol_to_liquidate, abs(qty_to_liquidate), prices, order_type="Auto-Liquidation")
    
def handle_hni_block_deals():
    game_state = get_game_state()
    if not game_state.block_deal_offer and random.random() < 0.02:
        symbol = random.choice(NIFTY50_SYMBOLS)
        price_modifier = random.uniform(0.98, 1.02)
        qty = random.randint(500, 2000)
        game_state.block_deal_offer = {'symbol': symbol, 'price_modifier': price_modifier, 'qty': qty, 'timestamp': time.time()}
        st.toast(f"HNI Alert: Exclusive block deal for {symbol} is now available!", icon="⭐")

def handle_futures_expiry(prices):
    game_state = get_game_state()
    if game_state.game_status == "Running" and not getattr(game_state, 'futures_settled', False) and time.time() > game_state.futures_expiry_time:
        st.warning("FUTURES EXPIRED! All open futures positions are being cash-settled.")
        settlement_prices = {'NIFTY-FUT': prices.get(NIFTY_INDEX_SYMBOL), 'BANKNIFTY-FUT': prices.get(BANKNIFTY_INDEX_SYMBOL)}
        for name, player in game_state.players.items():
            for symbol in FUTURES_SYMBOLS:
                if symbol in player['holdings']:
                    qty = player['holdings'][symbol]
                    pnl = (settlement_prices.get(symbol, 0) - prices.get(symbol, 0)) * qty
                    player['capital'] += pnl
                    log_transaction(name, "Futures Settlement", symbol, qty, settlement_prices.get(symbol, 0), pnl)
                    del player['holdings'][symbol]
        game_state.futures_settled = True

def run_algo_strategies(prices):
    game_state = get_game_state()
    if len(game_state.price_history) < 2: return
    prev_prices = game_state.price_history[-2]
    for name, player in game_state.players.items():
        if player['algo'] == 'Off': continue
        if player['algo'] in player.get('custom_algos', {}):
            strategy = player['custom_algos'][player['algo']]
            trade_symbol = random.choice(NIFTY50_SYMBOLS + CRYPTO_SYMBOLS)
            indicator_val = calculate_indicator(strategy['indicator'], trade_symbol)
            if indicator_val is None: continue
            condition_met = (strategy['condition'] == 'Greater Than' and indicator_val > strategy['threshold']) or \
                            (strategy['condition'] == 'Less Than' and indicator_val < strategy['threshold'])
            if condition_met: execute_trade(name, player, strategy['action'], trade_symbol, 1, prices, is_algo=True)
        else:
            trade_symbol = random.choice(NIFTY50_SYMBOLS + CRYPTO_SYMBOLS)
            if prices.get(trade_symbol, 0) == 0: continue
            price_change = prices.get(trade_symbol, 0) - prev_prices.get(trade_symbol, prices.get(trade_symbol, 0))
            if player['algo'] == "Momentum Trader" and abs(price_change / prices[trade_symbol]) > 0.001:
                execute_trade(name, player, "Buy" if price_change > 0 else "Sell", trade_symbol, 1, prices, is_algo=True)
            elif player['algo'] == "Mean Reversion" and abs(price_change / prices[trade_symbol]) > 0.001:
                execute_trade(name, player, "Sell" if price_change > 0 else "Buy", trade_symbol, 1, prices, is_algo=True)

# --- Utility Functions ---
def format_indian_currency(n):
    if n is None: return "₹0.00"
    n = float(n)
    if abs(n) < 100000: return f"₹{n:,.2f}"
    elif abs(n) < 10000000: return f"₹{n/100000:.2f}L"
    else: return f"₹{n/10000000:.2f}Cr"

def log_transaction(player_name, action, symbol, qty, price, total, is_algo=False):
    game_state = get_game_state()
    prefix = "🤖 Algo" if is_algo else "HFT" if "HFT" in player_name else ""
    game_state.transactions.setdefault(player_name, []).append([time.strftime("%H:%M:%S"), f"{prefix} {action}".strip(), symbol, qty, price, total])
    if "Auto-Liquidation" in action or "Settlement" in action:
        st.toast(f"{action}: {qty} {symbol}", icon="⚠️")
    elif not is_algo:
        st.toast(f"Trade Executed: {action} {qty} {symbol} @ {format_indian_currency(price)}", icon="✅")
    else:
        st.toast(f"Algo Trade: {action} {qty} {symbol}", icon="🤖")

def calculate_indicator(indicator, symbol):
    hist = get_historical_data([symbol], period="2mo")
    if hist.empty or len(hist) < 30: return None
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

def calculate_sharpe_ratio(value_history):
    if len(value_history) < 2: return 0.0
    returns = pd.Series(value_history).pct_change().dropna()
    if returns.std() == 0: return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(252)

# --- UI Components ---
def render_main_page(prices):
    game_state = get_game_state()
    st.title(f"📈 {GAME_NAME}")
    st.markdown("---")
    st.info("Welcome to the BlockVista Market Frenzy! Please join the game from the sidebar to start trading.")
    st.markdown("---")
    render_global_views(prices)

def render_global_views(prices, is_admin=False):
    game_state = get_game_state()
    
    # Live Clock & Status
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        st.metric("Game Status", game_state.game_status)
    with col2:
        if game_state.game_status == "Running":
            remaining_time = max(0, game_state.round_duration_seconds - int(time.time() - game_state.game_start_time))
            st.metric("Time Remaining", f"{remaining_time // 60:02d}:{remaining_time % 60:02d}")
        else:
            st.metric("Time Remaining", "--:--")
    with col3:
        st.metric("Difficulty", f"Level {game_state.difficulty_level}")

    st.markdown("---")
    
    if is_admin:
        with st.container(border=True):
            st.subheader("Live Player Performance")
            render_admin_performance_chart()
            st.subheader("Live Algo Bot Analyzer")
            render_admin_algo_analyzer()
            st.subheader("Admin Log")
            st.text_area("Event Log", value="\n".join(game_state.admin_log_messages[-10:]), height=150)

    st.subheader("📰 Live News Feed")
    if game_state.news_feed:
        for news in game_state.news_feed:
            st.info(news)
    else: st.info("No market news at the moment.")
    
    st.markdown("---")
    st.subheader("Live Player Standings")
    render_leaderboard(prices)

    st.markdown("---")
    st.subheader("Live Market Feed")
    render_live_market_table(prices)

def render_admin_performance_chart():
    game_state = get_game_state()
    chart_data = {name: player_data['value_history'] for name, player_data in game_state.players.items() if player_data.get('value_history')}
    if chart_data:
        df = pd.DataFrame(chart_data)
        st.line_chart(df)
    else:
        st.info("No trading activity yet to display.")

def render_admin_algo_analyzer():
    game_state = get_game_state()
    algo_pnl = {pname: 0 for pname in game_state.players.keys()}
    for pname, transactions in game_state.transactions.items():
        for t in transactions:
            if "Algo" in t[1]:
                # Simplified PnL calculation for display
                # Need to track buys/sells to get a real PnL
                pass
    st.info("Algo Bot Analyzer is not fully implemented in this version.")

def render_player_terminal(prices):
    game_state = get_game_state()
    acting_player = st.query_params.get("player")
    if not acting_player or acting_player not in game_state.players:
        st.error("Please log in to a player account.")
        return

    player = game_state.players[acting_player]
    
    # Dual sidebar approach - Left sidebar is the st.sidebar
    # Right sidebar is simulated with a column
    main_col, right_sidebar_col = st.columns([0.8, 0.2])

    with right_sidebar_col:
        st.subheader("Live Stats")
        with st.container(border=True):
            holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
            total_value = player['capital'] + holdings_value
            pnl = total_value - (INITIAL_CAPITAL * 5 if player['mode'] == 'HNI' else INITIAL_CAPITAL)
            player['pnl'] = pnl
            st.metric("Current P&L", format_indian_currency(pnl), f"{format_indian_currency(pnl)}")
            st.metric("Total Value", format_indian_currency(total_value))
            st.metric("Cash", format_indian_currency(player['capital']))

    with main_col:
        st.title(f"{acting_player}'s Trading Terminal")
        st.markdown(f"**Mode:** {player['mode']} | **Difficulty:** Level {game_state.difficulty_level}")
        
        # Check for HNI Block Deal
        if player['mode'] == 'HNI' and game_state.block_deal_offer:
            offer = game_state.block_deal_offer
            st.info(f"💰 HNI EXCLUSIVE: Block deal for {offer['qty']} shares of {offer['symbol'].replace('.NS','')} at a special price!")
            if st.button("Accept Block Deal"):
                trade_price = prices.get(offer['symbol'], 0) * offer['price_modifier']
                if execute_trade(acting_player, player, "Buy", offer['symbol'], offer['qty'], prices):
                    game_state.block_deal_offer = None
                    st.rerun()

        tabs = ["👨‍💻 Trade Terminal", "🤖 Algo Trading", "📂 Transaction History", "📊 Strategy & Insights"]
        tab1, tab2, tab3, tab4 = st.tabs(tabs)
        is_trade_disabled = game_state.game_status != "Running"

        with tab1: render_trade_interface(acting_player, player, prices, is_trade_disabled)
        with tab2: render_algo_trading_tab(acting_player, player, is_trade_disabled)
        with tab3: render_transaction_history(acting_player)
        with tab4: render_strategy_tab(player)
        
        st.markdown("---")
        render_current_holdings(player, prices)
        render_pending_orders(player)

def render_trade_interface(player_name, player, prices, disabled):
    order_type_tabs = ["Market", "Limit", "Stop-Loss"]
    market_tab, limit_tab, stop_loss_tab = st.tabs(order_type_tabs)
    with market_tab: render_market_order_ui(player_name, player, prices, disabled)
    with limit_tab: render_limit_order_ui(player_name, player, prices, disabled)
    with stop_loss_tab: render_stop_loss_order_ui(player_name, player, prices, disabled)

def render_market_order_ui(player_name, player, prices, disabled):
    asset_types = ["Stock", "Crypto", "Gold", "Futures", "Leveraged ETF", "Option"]
    asset_type = st.radio("Asset Type", asset_types, horizontal=True, key=f"market_asset_{player_name}", disabled=disabled)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbols_map = {
            "Stock": NIFTY50_SYMBOLS, "Crypto": CRYPTO_SYMBOLS, "Gold": [GOLD_SYMBOL],
            "Futures": FUTURES_SYMBOLS, "Leveraged ETF": LEVERAGED_ETFS, "Option": OPTION_SYMBOLS
        }
        symbol_options = symbols_map.get(asset_type, [])
        if asset_type == "Stock": symbol_options = [s.replace('.NS', '') for s in symbol_options]
        symbol_choice = st.selectbox(f"Select {asset_type}", symbol_options, key=f"market_symbol_{player_name}", disabled=disabled)
        if asset_type == "Stock": symbol_choice += '.NS'
    with col2:
        qty = st.number_input("Quantity", min_value=1, step=1, value=1, key=f"market_qty_{player_name}", disabled=disabled)
    
    mid_price = prices.get(symbol_choice, 0)
    ask_price = mid_price * (1 + get_game_state().bid_ask_spread / 2)
    bid_price = mid_price * (1 - get_game_state().bid_ask_spread / 2)
    st.info(f"Bid: {format_indian_currency(bid_price)} | Ask: {format_indian_currency(ask_price)}")

    b1, b2, b3 = st.columns(3)
    if b1.button(f"Buy {qty}", key=f"buy_{player_name}", use_container_width=True, disabled=disabled, type="primary"):
        execute_trade(player_name, player, "Buy", symbol_choice, qty, prices); st.rerun()
    if b2.button(f"Sell {qty}", key=f"sell_{player_name}", use_container_width=True, disabled=disabled):
        execute_trade(player_name, player, "Sell", symbol_choice, qty, prices); st.rerun()
    if b3.button(f"Short {qty}", key=f"short_{player_name}", use_container_width=True, disabled=disabled):
        execute_trade(player_name, player, "Short", symbol_choice, qty, prices); st.rerun()

def render_limit_order_ui(player_name, player, prices, disabled):
    st.write("Set a price to automatically buy or sell an asset.")
    asset_types = ["Stock", "Crypto", "Gold", "Futures", "Leveraged ETF", "Option"]
    asset_type = st.radio("Asset Type", asset_types, horizontal=True, key=f"limit_asset_{player_name}", disabled=disabled)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbols_map = {
            "Stock": NIFTY50_SYMBOLS, "Crypto": CRYPTO_SYMBOLS, "Gold": [GOLD_SYMBOL],
            "Futures": FUTURES_SYMBOLS, "Leveraged ETF": LEVERAGED_ETFS, "Option": OPTION_SYMBOLS
        }
        symbol_options = symbols_map.get(asset_type, [])
        if asset_type == "Stock": symbol_options = [s.replace('.NS', '') for s in symbol_options]
        symbol_choice = st.selectbox(f"Select {asset_type}", symbol_options, key=f"limit_symbol_{player_name}", disabled=disabled)
        if asset_type == "Stock": symbol_choice += '.NS'
    with col2:
        qty = st.number_input("Quantity", min_value=1, step=1, value=1, key=f"limit_qty_{player_name}", disabled=disabled)
    
    limit_price = st.number_input("Limit Price", min_value=0.01, value=prices.get(symbol_choice, 0), step=0.01, key=f"limit_price_{player_name}", disabled=disabled)
    
    b1, b2, b3 = st.columns(3)
    if b1.button("Buy Limit", key=f"buy_limit_{player_name}", use_container_width=True, disabled=disabled, type="primary"):
        player['pending_orders'].append({'type': 'Limit', 'action': 'Buy', 'symbol': symbol_choice, 'qty': qty, 'price': limit_price})
        st.success("Limit Buy order placed!"); st.rerun()
    if b2.button("Sell Limit", key=f"sell_limit_{player_name}", use_container_width=True, disabled=disabled):
        player['pending_orders'].append({'type': 'Limit', 'action': 'Sell', 'symbol': symbol_choice, 'qty': qty, 'price': limit_price})
        st.success("Limit Sell order placed!"); st.rerun()
    if b3.button("Short Limit", key=f"short_limit_{player_name}", use_container_width=True, disabled=disabled):
        player['pending_orders'].append({'type': 'Limit', 'action': 'Short', 'symbol': symbol_choice, 'qty': qty, 'price': limit_price})
        st.success("Limit Short order placed!"); st.rerun()

def render_stop_loss_order_ui(player_name, player, prices, disabled):
    st.write("Set a price to automatically sell an asset if it drops, to limit losses.")
    asset_types = ["Stock", "Crypto", "Gold", "Futures", "Leveraged ETF", "Option"]
    asset_type = st.radio("Asset Type", asset_types, horizontal=True, key=f"stop_asset_{player_name}", disabled=disabled)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbols_map = {
            "Stock": NIFTY50_SYMBOLS, "Crypto": CRYPTO_SYMBOLS, "Gold": [GOLD_SYMBOL],
            "Futures": FUTURES_SYMBOLS, "Leveraged ETF": LEVERAGED_ETFS, "Option": OPTION_SYMBOLS
        }
        symbol_options = symbols_map.get(asset_type, [])
        if asset_type == "Stock": symbol_options = [s.replace('.NS', '') for s in symbol_options]
        symbol_choice = st.selectbox(f"Select {asset_type}", symbol_options, key=f"stop_symbol_{player_name}", disabled=disabled)
        if asset_type == "Stock": symbol_choice += '.NS'
    with col2:
        qty = st.number_input("Quantity", min_value=1, step=1, value=1, key=f"stop_qty_{player_name}", disabled=disabled)
    
    stop_price = st.number_input("Stop Price", min_value=0.01, value=prices.get(symbol_choice, 0) * 0.95, step=0.01, key=f"stop_price_{player_name}", disabled=disabled)
    
    if st.button("Set Stop-Loss", key=f"set_stop_{player_name}", disabled=disabled):
        player['pending_orders'].append({'type': 'Stop-Loss', 'action': 'Sell', 'symbol': symbol_choice, 'qty': qty, 'price': stop_price})
        st.success("Stop-Loss order placed!"); st.rerun()

def render_algo_trading_tab(player_name, player, disabled):
    st.subheader("Automated Trading Strategies")
    default_strats = ["Off", "Momentum Trader", "Mean Reversion"]
    if get_game_state().difficulty_level >= 2:
        default_strats.extend(["Volatility Breakout", "Value Investor"])
    custom_strats = list(player.get('custom_algos', {}).keys()); all_strats = default_strats + custom_strats
    active_algo = player.get('algo', 'Off')
    player['algo'] = st.selectbox("Choose Strategy", all_strats, index=all_strats.index(active_algo) if active_algo in all_strats else 0, disabled=disabled, key=f"algo_{player_name}")
    st.info(f"Active Algo: **{player['algo']}**")

    if player['algo'] in default_strats and player['algo'] != 'Off':
        descriptions = {
            "Momentum Trader": "Buys assets rising in price, sells those falling.",
            "Mean Reversion": "Buys assets that have recently fallen, sells those that have risen.",
            "Volatility Breakout": "Trades in the direction of significant daily price moves.",
            "Value Investor": "Looks for assets that have dropped significantly and buys them."
        }
        st.info(descriptions.get(player['algo'], ""))

    with st.expander("Create Custom Strategy"):
        st.markdown("##### Define Your Own Algorithm")
        c1, c2 = st.columns(2)
        with c1:
            algo_name = st.text_input("Strategy Name", key=f"algo_name_{player_name}")
            indicator = st.selectbox("Indicator", ["Price Change % (5-day)", "SMA Crossover (10/20)", "Price Change % (30-day)"], key=f"indicator_{player_name}")
        with c2:
            condition = st.selectbox("Condition", ["Greater Than", "Less Than"], key=f"condition_{player_name}")
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
    else: st.info("No holdings yet.")

def render_pending_orders(player):
    st.subheader("🕒 Pending Orders")
    if player['pending_orders']:
        orders_df = pd.DataFrame(player['pending_orders'])
        st.dataframe(orders_df, use_container_width=True)
    else: st.info("No pending orders.")

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
            value_series = pd.Series(player['value_history'])
            sharpe = calculate_sharpe_ratio(player['value_history'])
            peak = value_series.cummax()
            drawdown = (value_series - peak) / peak
            max_drawdown = drawdown.min() * peak.max()
            returns = value_series.pct_change().dropna()
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() if not downside_returns.empty else 0
            sortino = (returns.mean() / downside_std) * np.sqrt(252) if downside_std != 0 else 0
            col1, col2, col3 = st.columns(3)
            col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
            col2.metric("Max Drawdown", format_indian_currency(max_drawdown))
            col3.metric("Sortino Ratio", f"{sortino:.2f}")
        else: st.info("Trade more to see your performance chart.")
    with tab2:
        st.markdown("##### Simple Moving Average (SMA) Chart")
        chartable_assets = [s for s in player['holdings'].keys() if s not in OPTION_SYMBOLS + FUTURES_SYMBOLS + LEVERAGED_ETFS]
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
    with tab3:
        st.subheader("Portfolio Optimization (Max Sharpe Ratio)")
        if st.button("Optimize My Portfolio"):
            weights, performance = optimize_portfolio(player['holdings'])
            if weights:
                st.success("Optimal weights for max risk-adjusted return:"); st.json({k: f"{v:.2%}" for k, v in weights.items()})
                if performance: st.write(f"Expected Return: {performance[0]:.2%}, Volatility: {performance[1]:.2%}, Sharpe Ratio: {performance[2]:.2f}")
            else: st.error(performance)
            
def render_leaderboard(prices):
    game_state = get_game_state()
    lb = []
    for pname, pdata in game_state.players.items():
        holdings_value = sum(prices.get(s, 0) * q for s, q in pdata['holdings'].items())
        total_value = pdata['capital'] + holdings_value
        sharpe_ratio = calculate_sharpe_ratio(pdata.get('value_history', []))
        lb.append((pname, pdata['mode'], total_value, pdata['pnl'], sharpe_ratio))
    if lb:
        lb_df = pd.DataFrame(lb, columns=["Player", "Mode", "Portfolio Value", "P&L", "Sharpe Ratio"]).sort_values("Portfolio Value", ascending=False).reset_index(drop=True)
        st.dataframe(lb_df.style.format(formatter={"Portfolio Value": format_indian_currency, "P&L": format_indian_currency, "Sharpe Ratio": "{:.2f}"}), use_container_width=True)
        if game_state.teams['A'] or game_state.teams['B']:
            st.subheader("Team Tournament Standings")
            team_a_pnl, team_a_port = calculate_team_metrics(game_state.teams['A'], game_state, prices)
            team_b_pnl, team_b_port = calculate_team_metrics(game_state.teams['B'], game_state, prices)
            team_df = pd.DataFrame({
                "Team": ["Team A", "Team B"],
                "Total Portfolio": [team_a_port, team_b_port],
                "Total P&L": [team_a_pnl, team_b_pnl]
            })
            st.dataframe(team_df.style.format(formatter={"Total Portfolio": format_indian_currency, "Total P&L": format_indian_currency}), use_container_width=True, hide_index=True)
        if game_state.game_status == "Finished":
            if not getattr(game_state, 'auto_square_off_complete', False):
                auto_square_off_positions(prices)
                game_state.auto_square_off_complete = True
                st.rerun()
            st.balloons(); winner = lb_df.iloc[0]
            st.success(f"🎉 The winner is {winner['Player']}! 🎉")
            c1, c2 = st.columns(2)
            c1.metric("🏆 Final Portfolio Value", format_indian_currency(winner['Portfolio Value']))
            c2.metric("💰 Total P&L", format_indian_currency(winner['P&L']))
            prudent_winner = lb_df.sort_values("Sharpe Ratio", ascending=False).iloc[0]
            st.info(f"🧐 The Prudent Investor Award goes to {prudent_winner['Player']} with a Sharpe Ratio of {prudent_winner['Sharpe Ratio']:.2f}!")

def render_live_market_table(prices):
    game_state = get_game_state()
    prices_df = pd.DataFrame(prices.items(), columns=['Symbol', 'Price'])
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

# --- Main Application Flow ---
def main():
    game_state = get_game_state()
    
    # Inject Tone.js script once
    st.components.v1.html('<script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.7.77/Tone.js"></script>', height=0)

    # Sidebar for login and controls
    render_sidebar()

    # Fetch base prices only if they haven't been fetched for the day
    if not game_state.base_real_prices:
        game_state.base_real_prices = get_daily_base_prices()
        st.toast("Fetched daily base market prices.")

    # Main game loop logic
    last_prices = game_state.prices if game_state.prices else game_state.base_real_prices
    current_prices = simulate_tick_prices(last_prices)
    prices_with_derivatives = calculate_derived_prices(current_prices)
    final_prices = run_game_tick(prices_with_derivatives)
    
    game_state.prices = final_prices
    game_state.price_history.append(final_prices)
    if len(game_state.price_history) > 10: game_state.price_history.pop(0)

    if st.session_state.get('role') == 'admin':
        render_admin_dashboard(final_prices)
    elif 'player' in st.query_params:
        render_player_terminal(final_prices)
    else:
        render_main_page(final_prices)

    if game_state.game_status == "Running":
        if not game_state.opening_bell_rung:
            play_sound('opening_bell')
            game_state.opening_bell_rung = True
        remaining_time = max(0, game_state.round_duration_seconds - int(time.time() - game_state.game_start_time))
        if remaining_time <= 30 and not game_state.closing_warning_triggered:
            play_sound('closing_warning')
            game_state.closing_warning_triggered = True
        if remaining_time == 0:
            play_sound('final_bell')
            game_state.game_status = "Finished"
            st.rerun()
        time.sleep(1)
        st.rerun()
    elif game_state.game_status == "Finished":
        st.balloons()
        render_global_views(final_prices)
    else:
        # Slower refresh for lobby/stopped state
        time.sleep(5)
        st.rerun()

def render_sidebar():
    game_state = get_game_state()
    st.sidebar.title("📝 Game Entry")
    if 'player' not in st.query_params:
        player_name = st.sidebar.text_input("Enter Name", key="name_input")
        mode = st.sidebar.radio("Select Mode", ["Trader", "HFT", "HNI"], key="mode_select")
        if st.sidebar.button("Join Game"):
            if player_name and player_name.strip() and player_name not in game_state.players:
                starting_capital = INITIAL_CAPITAL * 5 if mode == "HNI" else INITIAL_CAPITAL
                game_state.players[player_name] = {
                    "name": player_name, "mode": mode, "capital": starting_capital,
                    "holdings": {}, "pnl": 0, "leverage": 1.0, "margin_calls": 0,
                    "pending_orders": [], "algo": "Off", "custom_algos": {},
                    "value_history": [starting_capital], "trade_timestamps": []
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
    elif password: st.sidebar.error("Incorrect Password")

    if st.session_state.get('role') == 'admin':
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚙️ Admin Controls")
        if st.sidebar.button("▶️ Start Game", type="primary"):
            if game_state.players:
                game_state.game_status = "Running"
                game_state.game_start_time = time.time()
                game_state.futures_expiry_time = time.time() + (game_state.round_duration_seconds / 2)
                st.toast("Game Started!", icon="🎉"); st.rerun()
            else: st.sidebar.warning("Add at least one player to start.")
        if st.sidebar.button("⏸️ Stop Game"):
            game_state.game_status = "Stopped"; st.toast("Game Paused!", icon="⏸️"); st.rerun()
        if st.sidebar.button("🔄 Reset Game"):
            game_state.reset(); st.toast("Game has been reset.", icon="🔄"); st.rerun()
        if st.sidebar.button("Assign Teams"):
            assign_teams(game_state); st.sidebar.success("Players randomly assigned to Team A and Team B!")
        game_state.difficulty_level = st.sidebar.selectbox("Game Difficulty", [1, 2, 3], index=game_state.difficulty_level - 1, format_func=lambda x: f"Level {x}")
        game_state.volatility_multiplier = st.sidebar.slider("Market Volatility", 0.5, 5.0, game_state.volatility_multiplier, 0.5)

def render_admin_dashboard(prices):
    st.title(f"👑 {GAME_NAME} - Admin Dashboard")
    render_global_views(prices, is_admin=True)

def assign_teams(game_state):
    players = list(game_state.players.keys())
    random.shuffle(players)
    half = len(players) // 2
    game_state.teams['A'] = players[:half]
    game_state.teams['B'] = players[half:]

def calculate_team_metrics(team_players, game_state, prices):
    total_pnl = 0; total_portfolio = 0
    for p in team_players:
        player = game_state.players[p]
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
        portfolio_value = player['capital'] + holdings_value
        total_portfolio += portfolio_value
        total_pnl += player['pnl']
    return total_pnl, total_portfolio

def auto_square_off_positions(prices):
    game_state = get_game_state()
    st.info("End of game: Auto-squaring off all intraday, futures, and options positions...")
    square_off_assets = NIFTY50_SYMBOLS + FUTURES_SYMBOLS + OPTION_SYMBOLS + LEVERAGED_ETFS
    for name, player in game_state.players.items():
        for symbol, qty in list(player['holdings'].items()):
            if symbol in square_off_assets:
                closing_price = prices.get(symbol, 0)
                value = closing_price * qty
                if qty > 0: player['capital'] += value
                else: player['capital'] -= value
                log_transaction(name, "Auto-Squareoff", symbol, qty, closing_price, value)
                del player['holdings'][symbol]
    
if __name__ == "__main__":
    main()

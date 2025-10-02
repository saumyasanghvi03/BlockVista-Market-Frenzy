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
GAME_NAME = "BlockVista Market Frenzy - Tournament Edition"
INITIAL_CAPITAL = 1000000
ADMIN_PASSWORD = "100370"
MAX_PLAYERS = 50

# Enhanced symbol lists
NIFTY50_SYMBOLS = [
    'RELIANCE.NS', 'HDFCBANK.NS', 'ICICIBANK.NS', 'INFY.NS', 'TCS.NS', 
    'KOTAKBANK.NS', 'SBIN.NS', 'ITC.NS', 'ASIANPAINT.NS', 'AXISBANK.NS',
    'LT.NS', 'HINDUNILVR.NS', 'BHARTIARTL.NS', 'HCLTECH.NS', 'MARUTI.NS'
]

CRYPTO_SYMBOLS = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'DOGE-USD', 'ADA-USD', 'XRP-USD']
GOLD_SYMBOL = 'GC=F'
NIFTY_INDEX_SYMBOL = '^NSEI'
BANKNIFTY_INDEX_SYMBOL = '^NSEBANK'

FUTURES_SYMBOLS = ['NIFTY-FUT', 'BANKNIFTY-FUT']
LEVERAGED_ETFS = ['NIFTY_BULL_3X', 'NIFTY_BEAR_3X']
OPTION_SYMBOLS = ['NIFTY_CALL', 'NIFTY_PUT']

ALL_SYMBOLS = NIFTY50_SYMBOLS + CRYPTO_SYMBOLS + [GOLD_SYMBOL] + OPTION_SYMBOLS + FUTURES_SYMBOLS + LEVERAGED_ETFS + [NIFTY_INDEX_SYMBOL, BANKNIFTY_INDEX_SYMBOL]

# Enhanced news events
PRE_BUILT_NEWS = [
    {"headline": "Breaking: RBI unexpectedly cuts repo rate by 25 basis points!", "impact": "Bull Rally"},
    {"headline": "Government announces major infrastructure spending package!", "impact": "Bull Rally"},
    {"headline": "Shocking fraud uncovered at a major private bank!", "impact": "Flash Crash"},
    {"headline": "Indian tech firm announces breakthrough in AI!", "impact": "Sector Rotation"},
    {"headline": "FIIs show renewed interest in Indian equities!", "impact": "Bull Rally"},
    {"headline": "SEBI announces stricter margin rules for derivatives!", "impact": "Flash Crash"},
    {"headline": "Monsoon forecast revised upwards!", "impact": "Bull Rally"},
    {"headline": "Global News: US inflation data comes in hotter than expected!", "impact": "Flash Crash"},
    {"headline": "Global News: European Central Bank signals dovish stance!", "impact": "Bull Rally"},
    {"headline": "Global News: Major supply chain disruption in Asia!", "impact": "Volatility Spike"},
    {"headline": "{symbol} secures massive government contract!", "impact": "Symbol Bull Run"},
    {"headline": "Regulatory probe launched into {symbol}!", "impact": "Symbol Crash"},
    {"headline": "{symbol} announces surprise stock split!", "impact": "Stock Split"},
    {"headline": "{symbol} declares special dividend!", "impact": "Dividend"},
    {"headline": "High short interest in {symbol} triggers short squeeze!", "impact": "Short Squeeze"},
]

# --- Enhanced Game State Management ---
@st.cache_resource
def get_game_state():
    class GameState:
        def __init__(self):
            self.players = {}
            self.game_status = "Stopped"  # Stopped, Running, Break, Finished
            self.game_start_time = 0
            self.break_start_time = 0
            self.break_duration = 30  # Default 30 seconds break
            self.current_level = 1
            self.total_levels = 3
            self.level_durations = [10 * 60, 8 * 60, 6 * 60]  # 10, 8, 6 minutes per level
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
            self.price_stability_counter = {s: 0 for s in ALL_SYMBOLS}
            self.admin_trading_halt = False  # New: Admin can halt trading instantly
            self.manual_event_pending = None  # New: Admin can trigger events manually
            
        def reset(self):
            base_prices = self.base_real_prices.copy() if self.base_real_prices else {}
            self.__init__()
            self.base_real_prices = base_prices
            
    return GameState()

# --- Sound Effects ---
def play_sound(sound_type):
    sounds = {
        'success': 'synth.triggerAttackRelease("C5", "8n");',
        'error': 'synth.triggerAttackRelease("C3", "8n");',
        'opening_bell': 'const now = Tone.now(); synth.triggerAttackRelease("G4", "8n", now); synth.triggerAttackRelease("C5", "8n", now + 0.3); synth.triggerAttackRelease("E5", "8n", now + 0.6);',
        'closing_warning': 'const now = Tone.now(); synth.triggerAttackRelease("G5", "16n", now); synth.triggerAttackRelease("G5", "16n", now + 0.3);',
        'final_bell': 'synth.triggerAttackRelease("C4", "2n");',
        'level_complete': 'const now = Tone.now(); synth.triggerAttackRelease("C5", "8n", now); synth.triggerAttackRelease("E5", "8n", now + 0.2); synth.triggerAttackRelease("G5", "8n", now + 0.4);',
        'break_start': 'synth.triggerAttackRelease("A4", "4n");',
    }
    if sound_type in sounds:
        st.components.v1.html(f'<script>if (typeof Tone !== "undefined") {{const synth = new Tone.Synth().toDestination(); {sounds[sound_type]}}}</script>', height=0)

def announce_news(headline):
    safe_headline = headline.replace("'", "\\'").replace("\n", " ")
    st.components.v1.html(f'<script>if ("speechSynthesis" in window) {{const u = new SpeechSynthesisUtterance("{safe_headline}"); u.rate = 1.2; speechSynthesis.speak(u);}}</script>', height=0)

# --- Enhanced Game Flow Management ---
def update_game_state():
    """Handle level transitions, breaks, and game timing"""
    game_state = get_game_state()
    
    if game_state.game_status == "Running":
        current_time = time.time()
        level_duration = game_state.level_durations[game_state.current_level - 1]
        level_end_time = game_state.game_start_time + level_duration
        
        if current_time >= level_end_time:
            # Level completed
            if game_state.current_level < game_state.total_levels:
                # Start break before next level
                game_state.game_status = "Break"
                game_state.break_start_time = current_time
                play_sound('level_complete')
                st.toast(f"🎉 Level {game_state.current_level} Complete! Break time...", icon="⏸️")
            else:
                # Game finished
                game_state.game_status = "Finished"
                play_sound('final_bell')
                st.toast("🏆 Tournament Complete! Final results below.", icon="🎯")
                
    elif game_state.game_status == "Break":
        current_time = time.time()
        break_end_time = game_state.break_start_time + game_state.break_duration
        
        if current_time >= break_end_time:
            # Break over, start next level
            game_state.current_level += 1
            game_state.game_status = "Running"
            game_state.game_start_time = current_time
            game_state.difficulty_level = game_state.current_level  # Increase difficulty
            play_sound('opening_bell')
            st.toast(f"🚀 Level {game_state.current_level} Starting! Good luck!", icon="🎮")

def get_remaining_time():
    """Calculate remaining time for current level or break"""
    game_state = get_game_state()
    current_time = time.time()
    
    if game_state.game_status == "Running":
        level_duration = game_state.level_durations[game_state.current_level - 1]
        remaining = max(0, level_duration - (current_time - game_state.game_start_time))
        return remaining, "level"
    elif game_state.game_status == "Break":
        remaining = max(0, game_state.break_duration - (current_time - game_state.break_start_time))
        return remaining, "break"
    else:
        return 0, "stopped"

# --- Enhanced Market Simulation ---
@st.cache_data(ttl=3600)
def get_daily_base_prices():
    prices = {}
    yf_symbols = NIFTY50_SYMBOLS + CRYPTO_SYMBOLS + [GOLD_SYMBOL, NIFTY_INDEX_SYMBOL, BANKNIFTY_INDEX_SYMBOL]
    try:
        data = yf.download(tickers=yf_symbols, period="1d", interval="1m", progress=False, threads=True)
        for symbol in yf_symbols:
            prices[symbol] = data['Close'][symbol].iloc[-1] if not data.empty and symbol in data['Close'] and not pd.isna(data['Close'][symbol].iloc[-1]) else random.uniform(100, 5000)
    except Exception:
        for symbol in yf_symbols:
            prices[symbol] = random.uniform(100, 5000)
    return prices

def simulate_tick_prices(last_prices):
    game_state = get_game_state()
    prices = last_prices.copy()
    volatility = game_state.volatility_multiplier * (1 + 0.3 * (game_state.difficulty_level - 1))
    
    for symbol in prices:
        if symbol not in FUTURES_SYMBOLS + LEVERAGED_ETFS + OPTION_SYMBOLS:
            sentiment = game_state.market_sentiment.get(symbol, 0)
            noise = random.uniform(-0.001, 0.001) * volatility
            
            new_price = max(0.01, prices[symbol] * (1 + sentiment * 0.002 + noise))
            
            # Price stability
            base_price = game_state.base_real_prices.get(symbol, prices[symbol])
            if symbol in game_state.price_stability_counter:
                if abs(new_price - base_price) / base_price > 1.5:
                    game_state.price_stability_counter[symbol] += 1
                    if game_state.price_stability_counter[symbol] > 3:
                        correction = (base_price - new_price) / base_price * 0.3
                        new_price *= (1 + correction)
                        game_state.price_stability_counter[symbol] = 0
                else:
                    game_state.price_stability_counter[symbol] = max(0, game_state.price_stability_counter[symbol] - 0.1)
            
            prices[symbol] = round(new_price, 2)
            
    return prices

def calculate_derived_prices(base_prices):
    game_state = get_game_state()
    prices = base_prices.copy()
    nifty = prices.get(NIFTY_INDEX_SYMBOL, 20000)
    banknifty = prices.get(BANKNIFTY_INDEX_SYMBOL, 45000)
    
    prices.update({
        'NIFTY_CALL': max(0.01, nifty * random.uniform(1.02, 1.05)),
        'NIFTY_PUT': max(0.01, nifty * random.uniform(0.95, 0.98)),
        'NIFTY-FUT': max(0.01, nifty * random.uniform(0.99, 1.01)),
        'BANKNIFTY-FUT': max(0.01, banknifty * random.uniform(0.99, 1.01))
    })
    
    if len(game_state.price_history) >= 2:
        prev_nifty = game_state.price_history[-2].get(NIFTY_INDEX_SYMBOL, nifty)
        if prev_nifty > 0:
            nifty_change = (nifty - prev_nifty) / prev_nifty
            current_bull = game_state.prices.get('NIFTY_BULL_3X', nifty / 100)
            current_bear = game_state.prices.get('NIFTY_BEAR_3X', nifty / 100)
            prices['NIFTY_BULL_3X'] = max(0.01, current_bull * (1 + 3 * nifty_change))
            prices['NIFTY_BEAR_3X'] = max(0.01, current_bear * (1 - 3 * nifty_change))
    
    return prices

# --- Enhanced Event System with Admin Controls ---
def apply_event_adjustment(prices, event_type, target_symbol=None):
    game_state = get_game_state()
    prices = prices.copy()
    
    event_effects = {
        "Bull Rally": lambda p: {k: max(0.01, v * random.uniform(1.05, 1.15)) for k, v in p.items()},
        "Flash Crash": lambda p: {k: max(0.01, v * random.uniform(0.85, 0.95)) for k, v in p.items()},
        "Symbol Bull Run": lambda p: {k: max(0.01, v * (1.25 if k == target_symbol else 1.0)) for k, v in p.items()},
        "Symbol Crash": lambda p: {k: max(0.01, v * (0.75 if k == target_symbol else 1.0)) for k, v in p.items()},
        "Short Squeeze": lambda p: {k: max(0.01, v * (1.35 if k == target_symbol else 1.0)) for k, v in p.items()},
        "Volatility Spike": lambda p: {k: max(0.01, v * (1 + random.uniform(-0.03, 0.03) * 2)) for k, v in p.items()},
    }
    
    if event_type in event_effects:
        message = f"⚡ {event_type}!"
        if target_symbol:
            message = f"⚡ {event_type} on {target_symbol.replace('.NS', '')}!"
        
        st.toast(message, icon="🎪")
        game_state.news_feed.insert(0, f"📢 {time.strftime('%H:%M:%S')} - {message}")
        if len(game_state.news_feed) > 5: 
            game_state.news_feed.pop()
            
        prices = event_effects[event_type](prices)
        announce_news(message)
        
    return prices

def trigger_manual_event(event_type, target_symbol=None):
    """Admin function to manually trigger market events"""
    game_state = get_game_state()
    game_state.manual_event_pending = {
        'type': event_type,
        'target_symbol': target_symbol,
        'timestamp': time.time()
    }
    st.toast(f"🎪 Manual {event_type} triggered!", icon="🎯")

# --- Enhanced Trading with Admin Halt ---
def execute_trade(player_name, player, action, symbol, qty, prices, is_algo=False, order_type="Market"):
    game_state = get_game_state()
    
    # Check if trading is halted (circuit breaker or admin halt)
    if (game_state.circuit_breaker_active and time.time() < game_state.circuit_breaker_end) or game_state.admin_trading_halt:
        if not is_algo:
            st.error("🚫 Trading halted by admin!")
        return False
        
    mid_price = prices.get(symbol, 0)
    if mid_price <= 0:
        if not is_algo:
            st.error(f"Invalid price for {symbol}")
        return False
        
    trade_price = mid_price * (1 + game_state.bid_ask_spread / 2 if action == "Buy" else 1 - game_state.bid_ask_spread / 2)
    trade_price *= calculate_slippage(player, symbol, qty, action)
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
            if player['holdings'].get(symbol, 0) == 0:
                del player['holdings'][symbol]
                
    if trade_executed:
        game_state.market_sentiment[symbol] = game_state.market_sentiment.get(symbol, 0) + (qty / 50)
        log_transaction(player_name, f"{order_type} {action}", symbol, qty, trade_price, cost, is_algo)
        player['trade_timestamps'].append(time.time())
        
        if is_algo:
            if player_name not in game_state.algo_pnl:
                game_state.algo_pnl[player_name] = 0
            game_state.algo_pnl[player_name] += -cost if action == "Buy" else cost
            
        check_hft_rebate(player_name, player)
    elif not is_algo:
        st.error("Trade failed: Insufficient capital or holdings.")
        
    return trade_executed

def calculate_slippage(player, symbol, qty, action):
    game_state = get_game_state()
    if qty <= game_state.slippage_threshold: 
        return 1.0
    liquidity = game_state.liquidity.get(symbol, 1.0)
    slippage_mult = player.get('slippage_multiplier', 1.0)
    excess_qty = qty - game_state.slippage_threshold
    slippage_rate = (game_state.base_slippage_rate / max(0.1, liquidity)) * slippage_mult
    return max(0.9, min(1.1, 1 + (slippage_rate * excess_qty) * (1 if action == "Buy" else -1)))

def check_hft_rebate(player_name, player):
    game_state = get_game_state()
    if player['mode'] != 'HFT': 
        return
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
    game_state.transactions.setdefault(player_name, []).append([
        time.strftime("%H:%M:%S"), 
        f"{prefix} {action}".strip(), 
        symbol, qty, price, total
    ])

# --- Enhanced Game Logic ---
def run_game_tick(prices):
    game_state = get_game_state()
    if game_state.game_status != "Running": 
        return prices
        
    # Process manual events first
    if game_state.manual_event_pending:
        event = game_state.manual_event_pending
        prices = apply_event_adjustment(prices, event['type'], event['target_symbol'])
        game_state.manual_event_pending = None
    
    # Process other game mechanics
    process_pending_orders(prices)
    
    # Market dynamics
    for symbol in game_state.market_sentiment:
        game_state.market_sentiment[symbol] *= 0.95
        
    # Random events (only when no manual event)
    if not game_state.event_active and random.random() < 0.06:
        news = random.choice(PRE_BUILT_NEWS)
        headline = news['headline']
        target_symbol = random.choice(NIFTY50_SYMBOLS) if "{symbol}" in headline else None
        if target_symbol:
            headline = headline.format(symbol=target_symbol.replace(".NS", ""))
        game_state.news_feed.insert(0, f"📢 {time.strftime('%H:%M:%S')} - {headline}")
        if len(game_state.news_feed) > 5: 
            game_state.news_feed.pop()
        game_state.event_type = news['impact']
        game_state.event_target_symbol = target_symbol
        game_state.event_active = True
        game_state.event_end = time.time() + random.randint(30, 60)
        st.toast(f"⚡ Market Event!", icon="🎉")
        announce_news(headline)
        
    if game_state.event_active and time.time() >= game_state.event_end:
        game_state.event_active = False
        
    if game_state.event_active:
        prices = apply_event_adjustment(prices, game_state.event_type, game_state.event_target_symbol)
            
    # Run game systems
    handle_futures_expiry(prices)
    check_margin_calls_and_orders(prices)
    run_algo_strategies(prices)
    apply_difficulty_mechanics(prices)
    
    # Update player metrics
    for player in game_state.players.values():
        holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        player['value_history'].append(total_value)
        
    return prices

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
        margin_required = sum(prices.get(s, 0) * abs(q) * game_state.current_margin_requirement 
                             for s, q in player['holdings'].items() if q < 0)
        if player['capital'] < margin_required * 0.5:
            for symbol, qty in list(player['holdings'].items()):
                if qty < 0:
                    liquidate_qty = min(abs(qty), max(1, int(abs(qty) * 0.3)))
                    if execute_trade(player_name, player, "Sell", symbol, liquidate_qty, prices, order_type="Auto-Liquidation"):
                        st.toast(f"Margin call: {player_name} liquidated {liquidate_qty} {symbol}", icon="⚠️")
                        break

def run_algo_strategies(prices):
    game_state = get_game_state()
    for player_name, player in game_state.players.items():
        if player['algo'] != 'Off' and game_state.game_status == "Running":
            if player_name not in game_state.algo_trades:
                game_state.algo_trades[player_name] = []
                
            if player['algo'] == "Momentum Trader":
                for symbol in NIFTY50_SYMBOLS[:3]:
                    if len(game_state.price_history) >= 2:
                        prev_price = game_state.price_history[-2].get(symbol, prices.get(symbol, 0))
                        curr_price = prices.get(symbol, 0)
                        if prev_price > 0 and curr_price > 0:
                            change_pct = (curr_price - prev_price) / prev_price
                            if change_pct > 0.01 and random.random() < 0.2:
                                qty = max(1, int(player['capital'] * 0.05 / curr_price))
                                execute_trade(player_name, player, "Buy", symbol, qty, prices, is_algo=True)
                            elif change_pct < -0.01 and symbol in player['holdings'] and player['holdings'][symbol] > 0 and random.random() < 0.2:
                                qty = min(player['holdings'][symbol], max(1, int(player['holdings'][symbol] * 0.3)))
                                execute_trade(player_name, player, "Sell", symbol, qty, prices, is_algo=True)

def apply_difficulty_mechanics(prices):
    game_state = get_game_state()
    if game_state.difficulty_level > 1:
        for symbol in game_state.liquidity:
            game_state.liquidity[symbol] *= max(0.3, 1 - (game_state.difficulty_level - 1) * 0.2)

def auto_square_off_positions(prices):
    game_state = get_game_state()
    for player_name, player in game_state.players.items():
        for symbol, qty in list(player['holdings'].items()):
            if qty > 0:
                execute_trade(player_name, player, "Sell", symbol, qty, prices, order_type="Auto-SquareOff")
            elif qty < 0:
                execute_trade(player_name, player, "Sell", symbol, abs(qty), prices, order_type="Auto-SquareOff")

# --- UI Helper Functions ---
def format_indian_currency(n):
    if n is None: return "₹0.00"
    n = float(n)
    if abs(n) >= 10000000:
        return f"₹{n/10000000:.2f}Cr"
    elif abs(n) >= 100000:
        return f"₹{n/100000:.2f}L"
    elif abs(n) >= 1000:
        return f"₹{n/1000:.1f}K"
    else:
        return f"₹{n:,.2f}"

def calculate_sharpe_ratio(values):
    if len(values) < 2: return 0.0
    returns = pd.Series(values).pct_change().dropna()
    if returns.std() == 0: return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(252)

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

# --- Enhanced UI Components ---
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
                
            st.subheader("⚙️ Tournament Settings")
            
            # Break duration control
            game_state.break_duration = st.number_input("Break Duration (seconds)", 
                                                      min_value=10, max_value=120, 
                                                      value=game_state.break_duration)
            
            # Level durations
            st.write("Level Durations (minutes):")
            col1, col2, col3 = st.columns(3)
            with col1:
                game_state.level_durations[0] = st.number_input("Level 1", min_value=1, value=10) * 60
            with col2:
                game_state.level_durations[1] = st.number_input("Level 2", min_value=1, value=8) * 60
            with col3:
                game_state.level_durations[2] = st.number_input("Level 3", min_value=1, value=6) * 60
            
            # Game control buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Start Tournament", type="primary", use_container_width=True):
                    if game_state.players:
                        game_state.game_status = "Running"
                        game_state.game_start_time = time.time()
                        game_state.current_level = 1
                        game_state.difficulty_level = 1
                        game_state.auto_square_off_complete = False
                        game_state.closing_warning_triggered = False
                        st.toast("Tournament Started! Level 1 begins.", icon="🎉")
                        play_sound('opening_bell')
                        st.rerun()
                    else:
                        st.warning("Add at least one player to start.")
            
            with col2:
                if st.button("⏸️ Stop Game", use_container_width=True):
                    game_state.game_status = "Stopped"
                    st.toast("Game Stopped!", icon="⏸️")
                    st.rerun()
                    
            if st.button("🔄 Reset Game", use_container_width=True):
                game_state.reset()
                st.toast("Game has been reset.", icon="🔄")
                st.rerun()
                
            # Enhanced Admin Controls
            st.markdown("---")
            st.subheader("🎪 Market Controls")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚫 Halt Trading", use_container_width=True):
                    game_state.admin_trading_halt = True
                    st.toast("Trading halted by admin!", icon="🚫")
                    
                if st.button("✅ Resume Trading", use_container_width=True):
                    game_state.admin_trading_halt = False
                    st.toast("Trading resumed!", icon="✅")
                    
            with col2:
                if st.button("📈 Trigger Bull Run", use_container_width=True):
                    trigger_manual_event("Bull Rally")
                    
                if st.button("📉 Trigger Flash Crash", use_container_width=True):
                    trigger_manual_event("Flash Crash")
            
            # Individual stock events
            st.subheader("🎯 Stock-Specific Events")
            target_stock = st.selectbox("Select Stock", [s.replace('.NS', '') for s in NIFTY50_SYMBOLS])
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚀 Stock Bull Run", use_container_width=True):
                    trigger_manual_event("Symbol Bull Run", f"{target_stock}.NS")
            with col2:
                if st.button("💥 Stock Crash", use_container_width=True):
                    trigger_manual_event("Symbol Crash", f"{target_stock}.NS")
                    
        elif password:
            st.error("❌ Incorrect Password")

def render_right_sidebar(prices):
    game_state = get_game_state()
    
    with st.sidebar:
        if st.session_state.get('role') == 'admin':
            st.title("👑 Admin Dashboard")
            
            # Quick stats
            st.metric("Active Players", len(game_state.players))
            st.metric("Current Level", game_state.current_level)
            st.metric("Trading Status", "HALTED" if game_state.admin_trading_halt else "ACTIVE")
            
        else:
            # Player stats
            st.title("📊 Live Stats")
            
            if 'player' in st.query_params:
                player_name = st.query_params.get("player")
                if player_name in game_state.players:
                    player = game_state.players[player_name]
                    
                    # Real-time holdings value
                    holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
                    total_value = player['capital'] + holdings_value
                    
                    st.metric("💰 Available Cash", format_indian_currency(player['capital']))
                    st.metric("💎 Holdings Value", format_indian_currency(holdings_value))
                    st.metric("📈 Total Portfolio", format_indian_currency(total_value))
                    
                    # Quick holdings overview
                    if player['holdings']:
                        st.subheader("📦 Your Holdings")
                        for symbol, qty in list(player['holdings'].items())[:3]:  # Show top 3
                            if qty != 0:
                                current_price = prices.get(symbol, 0)
                                st.write(f"{symbol}: {qty} @ {format_indian_currency(current_price)}")

def render_main_interface(prices):
    game_state = get_game_state()
    
    # Inject sound system
    st.components.v1.html('<script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.7.77/Tone.js"></script>', height=0)
    
    # Enhanced header with tournament info
    st.title(f"🎮 {GAME_NAME}")
    
    # Tournament status display
    remaining_time, time_type = get_remaining_time()
    
    if game_state.game_status == "Running":
        minutes, seconds = divmod(int(remaining_time), 60)
        
        if time_type == "level":
            st.subheader(f"🎯 Level {game_state.current_level} | Time Remaining: {minutes:02d}:{seconds:02d}")
            st.write(f"**Difficulty: Level {game_state.difficulty_level}** | "
                    f"**Players: {len(game_state.players)}**")
            
            if remaining_time <= 60:
                st.error("⏰ FINAL MINUTE!")
                
        elif time_type == "break":
            st.subheader(f"⏸️ Break Time | Next: Level {game_state.current_level + 1}")
            st.warning(f"Break ends in: {int(remaining_time)} seconds")
            st.info("Use this time to analyze your strategy for the next level!")
            
    elif game_state.game_status == "Break":
        st.subheader(f"⏸️ Break Time | Next: Level {game_state.current_level}")
        st.warning(f"Break ends in: {int(remaining_time)} seconds")
        
    elif game_state.game_status == "Stopped":
        st.info("⏸️ Game is paused. Press 'Start Tournament' to begin!")
        
    elif game_state.game_status == "Finished":
        st.success("🎉 Tournament Complete! Final results below.")
    
    # Trading halt warning
    if game_state.admin_trading_halt:
        st.error("🚫 **TRADING HALTED BY ADMIN**")
    
    # Main content
    if st.session_state.get('role') == 'admin':
        render_global_views(prices, is_admin=True)
    elif 'player' in st.query_params:
        col1, col2 = st.columns([2, 1])
        with col1:
            render_trade_execution_panel(prices)
        with col2:
            render_global_views(prices)
    else:
        st.info("🎯 Welcome to BlockVista Market Frenzy Tournament Edition! Join the game from the sidebar.")
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
            for news in news_feed[:5]:
                st.info(news)
        else:
            st.info("No market news at the moment.")

        # Leaderboard
        st.markdown("---")
        st.subheader("🏆 Live Leaderboard")
        render_leaderboard(prices)
        
        if is_admin:
            st.markdown("---")
            st.subheader("📊 Player Performance")
            render_admin_performance_chart()

        # Market data
        st.markdown("---")
        st.subheader("📈 Live Market Data")
        render_live_market_table(prices)

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
        max_len = max(len(history) for history in chart_data.values())
        padded_data = {}
        for name, history in chart_data.items():
            padded_data[name] = history + [history[-1]] * (max_len - len(history)) if len(history) < max_len else history
                
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

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("💰 Cash", format_indian_currency(player['capital']))
        col2.metric("📊 Portfolio Value", format_indian_currency(total_value))
        col3.metric("📈 P&L", format_indian_currency(pnl), delta=format_indian_currency(pnl))
        col4.metric("🎯 Sharpe Ratio", f"{calculate_sharpe_ratio(player.get('value_history', [])):.2f}")

        # Trading interface
        tabs = ["⚡ Trade Terminal", "🤖 Algo Trading", "📋 Transaction History", "📈 Strategy & Insights"]
        tab1, tab2, tab3, tab4 = st.tabs(tabs)
        
        is_trade_disabled = (game_state.game_status != "Running" or 
                           game_state.admin_trading_halt or
                           game_state.circuit_breaker_active)

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
    render_current_holdings(player, prices)

def render_market_order_ui(player_name, player, prices, disabled):
    asset_types = ["Stock", "Crypto", "Gold", "Futures", "Leveraged ETF", "Option"]
    asset_type = st.radio("Asset Type", asset_types, horizontal=True, key=f"market_asset_{player_name}", disabled=disabled)
    
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
    
    mid_price = prices.get(symbol_choice, 0)
    ask_price = mid_price * (1 + get_game_state().bid_ask_spread / 2)
    bid_price = mid_price * (1 - get_game_state().bid_ask_spread / 2)
    st.info(f"💱 Bid: {format_indian_currency(bid_price)} | Ask: {format_indian_currency(ask_price)}")

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
    safe_current_price = max(0.01, current_price)
    limit_price = st.number_input("Limit Price", min_value=0.01, value=safe_current_price, step=0.01, 
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
    safe_current_price = max(0.01, current_price)
    stop_price_default = safe_current_price * 0.95
    stop_price = st.number_input("Stop Price", min_value=0.01, value=stop_price_default, step=0.01, 
                               key=f"stop_price_{player_name}", disabled=disabled)
    
    st.info(f"Current Price: {format_indian_currency(current_price)} | Stop at: {format_indian_currency(stop_price)}")
    
    if st.button("🛑 Set Stop-Loss", key=f"set_stop_{player_name}", disabled=disabled, use_container_width=True):
        player['pending_orders'].append({'type': 'Stop-Loss', 'action': 'Sell', 'symbol': symbol_choice, 'qty': qty, 'price': stop_price})
        st.success("✅ Stop-Loss order placed!")
        st.rerun()

def render_current_holdings(player, prices):
    st.subheader("💼 Portfolio Allocation")
    if player['holdings']:
        holdings_data = []
        for sym, qty in player['holdings'].items():
            if qty != 0:
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
            st.dataframe(holdings_df.style.format({
                "Price": format_indian_currency,
                "Value": format_indian_currency
            }), use_container_width=True)
    else: 
        st.info("No holdings yet.")

def render_algo_trading_tab(player_name, player, disabled):
    st.subheader("🤖 Automated Trading Strategies")
    
    default_strats = ["Off", "Momentum Trader", "Mean Reversion", "Volatility Breakout", "Value Investor"]
    custom_strats = list(player.get('custom_algos', {}).keys())
    all_strats = default_strats + custom_strats
    
    active_algo = player.get('algo', 'Off')
    player['algo'] = st.selectbox("Choose Strategy", all_strats, 
                                index=all_strats.index(active_algo) if active_algo in all_strats else 0, 
                                disabled=disabled, key=f"algo_{player_name}")

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
    tab1, tab2, tab3 = st.tabs(["Performance Chart", "Technical Analysis", "Risk Metrics"])
    
    with tab1:
        st.markdown("##### Portfolio Value Over Time")
        if len(player.get('value_history', [])) > 1:
            st.line_chart(player['value_history'])
        else:
            st.info("Trade more to see your performance chart.")
            
    with tab2: 
        render_sma_chart(player['holdings'])
    with tab3:
        render_risk_metrics(player)

def render_sma_chart(holdings):
    st.markdown("##### Simple Moving Average (SMA) Chart")
    chartable_assets = [s for s in holdings.keys() if s not in OPTION_SYMBOLS + FUTURES_SYMBOLS + LEVERAGED_ETFS]
    if not chartable_assets: 
        st.info("No chartable assets in portfolio to analyze.")
        return
        
    chart_symbol = st.selectbox("Select Asset to Chart", chartable_assets)
    # Implementation would go here...

def render_risk_metrics(player):
    st.subheader("📉 Risk Analysis")
    
    if len(player.get('value_history', [])) > 1:
        sharpe = calculate_sharpe_ratio(player['value_history'])
        max_dd = calculate_max_drawdown(player['value_history'])
        
        col1, col2 = st.columns(2)
        col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
        col2.metric("Max Drawdown", f"{max_dd:.1f}%")

def render_leaderboard(prices):
    game_state = get_game_state()
    lb = []
    for pname, pdata in game_state.players.items():
        holdings_value = sum(prices.get(s, 0) * q for s, q in pdata['holdings'].items())
        total_value = pdata['capital'] + holdings_value
        starting_capital = INITIAL_CAPITAL * 5 if pdata['mode'] == 'HNI' else INITIAL_CAPITAL
        pnl = total_value - starting_capital
        sharpe_ratio = calculate_sharpe_ratio(pdata.get('value_history', []))
        
        lb.append((pname, pdata['mode'], total_value, pnl, sharpe_ratio))
    
    if lb:
        lb_df = pd.DataFrame(lb, columns=["Player", "Mode", "Portfolio Value", "P&L", "Sharpe Ratio"])
        lb_df = lb_df.sort_values("Portfolio Value", ascending=False).reset_index(drop=True)
        
        styled_df = lb_df.style.format({
            "Portfolio Value": format_indian_currency,
            "P&L": format_indian_currency,
            "Sharpe Ratio": "{:.2f}"
        })
        
        st.dataframe(styled_df, use_container_width=True)
        
        if game_state.game_status == "Finished":
            if not getattr(game_state, 'auto_square_off_complete', False):
                auto_square_off_positions(prices)
                game_state.auto_square_off_complete = True
                st.rerun()

            st.balloons()
            winner = lb_df.iloc[0]
            st.success(f"🎉 Tournament Winner: **{winner['Player']}**! 🎉")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("🏆 Final Portfolio", format_indian_currency(winner['Portfolio Value']))
            col2.metric("💰 Total P&L", format_indian_currency(winner['P&L']))
            col3.metric("⚡ Sharpe Ratio", f"{winner['Sharpe Ratio']:.2f}")

def render_live_market_table(prices):
    game_state = get_game_state()
    prices_df = pd.DataFrame(prices.items(), columns=['Symbol', 'Price'])
    
    if len(game_state.price_history) >= 2:
        prev_prices = game_state.price_history[-2]
        prices_df['prev_price'] = prices_df['Symbol'].map(prev_prices).fillna(prices_df['Price'])
        prices_df['Change'] = prices_df['Price'] - prices_df['prev_price']
        prices_df['Change %'] = (prices_df['Change'] / prices_df['prev_price']) * 100
    else: 
        prices_df['Change'] = 0.0
        prices_df['Change %'] = 0.0
    
    prices_df.drop(columns=['prev_price'], inplace=True, errors='ignore')

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
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.role = 'player'
        st.session_state.last_refresh = time.time()
    
    game_state = get_game_state()
    
    # Update game state (level transitions, breaks, etc.)
    update_game_state()
    
    # Three-column layout
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        render_left_sidebar()
    
    with col2:
        # Price simulation pipeline
        if not game_state.base_real_prices:
            game_state.base_real_prices = get_daily_base_prices()
            
        last_prices = game_state.prices if game_state.prices else game_state.base_real_prices
        current_prices = simulate_tick_prices(last_prices)
        prices_with_derivatives = calculate_derived_prices(current_prices)
        final_prices = run_game_tick(prices_with_derivatives)
        
        game_state.prices = final_prices
        
        if not isinstance(game_state.price_history, list): 
            game_state.price_history = []
        game_state.price_history.append(final_prices)
        if len(game_state.price_history) > 8: 
            game_state.price_history.pop(0)
        
        # Circuit breaker check
        if game_state.circuit_breaker_active and time.time() >= game_state.circuit_breaker_end:
            game_state.circuit_breaker_active = False
            
        render_main_interface(final_prices)
    
    with col3:
        render_right_sidebar(final_prices)
    
    # Optimized auto-refresh
    current_time = time.time()
    refresh_interval = 1.5 if game_state.game_status == "Running" else 3
    if current_time - st.session_state.last_refresh > refresh_interval:
        st.session_state.last_refresh = current_time
        st.rerun()

if __name__ == "__main__":
    main()

# BlockVista Market Frenzy - Tournament Edition with Auto-Qualification

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

# Auto-qualification criteria
QUALIFICATION_CRITERIA = {
    1: {"min_portfolio_value": 1100000, "description": "Reach ₹11L portfolio"},  # 10% gain
    2: {"min_portfolio_value": 1250000, "description": "Reach ₹12.5L portfolio"}  # 25% gain
}

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
            self.last_event_time = 0
            self.event_cooldown = 60
            self.manual_event_pending = None
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
            self.admin_trading_halt = False
            self.qualified_players = set()  # Track players who qualify for next level
            self.eliminated_players = set()  # Track eliminated players
            self.level_results = {}  # Store level completion results
            
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
        'qualification': 'const now = Tone.now(); synth.triggerAttackRelease("C6", "8n", now); synth.triggerAttackRelease("G5", "8n", now + 0.2);',
        'elimination': 'synth.triggerAttackRelease("C3", "2n");'
    }
    if sound_type in sounds:
        st.components.v1.html(f'<script>if (typeof Tone !== "undefined") {{const synth = new Tone.Synth().toDestination(); {sounds[sound_type]}}}</script>', height=0)

def announce_news(headline):
    safe_headline = headline.replace("'", "\\'").replace("\n", " ")
    st.components.v1.html(f'<script>if ("speechSynthesis" in window) {{const u = new SpeechSynthesisUtterance("{safe_headline}"); u.rate = 1.2; speechSynthesis.speak(u);}}</script>', height=0)

# --- MISSING FUNCTION: run_game_tick ---
def run_game_tick(prices):
    """Handle manual events, random events, and market dynamics for each tick"""
    game_state = get_game_state()
    current_time = time.time()
    
    # Process manual events first (admin triggered)
    if game_state.manual_event_pending:
        event = game_state.manual_event_pending
        prices = apply_event_adjustment(prices, event['type'], event.get('target_symbol'))
        game_state.manual_event_pending = None
        game_state.last_event_time = current_time
    
    # Process random events (only during running game)
    elif (game_state.game_status == "Running" and 
          current_time - game_state.last_event_time > game_state.event_cooldown and
          random.random() < 0.02):  # 2% chance per tick
        
        event_types = ["Bull Rally", "Flash Crash", "Volatility Spike", "Sector Rotation"]
        event_type = random.choice(event_types)
        
        # Symbol-specific events
        if event_type in ["Symbol Bull Run", "Symbol Crash", "Short Squeeze"]:
            target_symbol = random.choice(ALL_SYMBOLS)
            prices = apply_event_adjustment(prices, event_type, target_symbol)
        else:
            prices = apply_event_adjustment(prices, event_type)
            
        game_state.last_event_time = current_time
    
    # Update player portfolio history
    for player_name, player in game_state.players.items():
        if player_name not in game_state.eliminated_players:
            holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
            total_value = player['capital'] + holdings_value
            player['value_history'].append(total_value)
            if len(player['value_history']) > 100:  # Keep last 100 values
                player['value_history'].pop(0)
    
    return prices

def apply_event_adjustment(prices, event_type, target_symbol=None):
    """Apply market event adjustments to prices"""
    game_state = get_game_state()
    prices = prices.copy()
    
    event_effects = {
        "Bull Rally": lambda p: {k: max(0.01, v * random.uniform(1.05, 1.15)) for k, v in p.items()},
        "Flash Crash": lambda p: {k: max(0.01, v * random.uniform(0.85, 0.95)) for k, v in p.items()},
        "Symbol Bull Run": lambda p: {k: max(0.01, v * (1.25 if k == target_symbol else 1.0)) for k, v in p.items()},
        "Symbol Crash": lambda p: {k: max(0.01, v * (0.75 if k == target_symbol else 1.0)) for k, v in p.items()},
        "Short Squeeze": lambda p: {k: max(0.01, v * (1.35 if k == target_symbol else 1.0)) for k, v in p.items()},
        "Volatility Spike": lambda p: {k: max(0.01, v * (1 + random.uniform(-0.03, 0.03) * 2)) for k, v in p.items()},
        "Sector Rotation": lambda p: apply_sector_rotation(p)
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

def apply_sector_rotation(prices):
    """Apply sector rotation - some sectors up, some down"""
    sectors = {
        'banks': [s for s in NIFTY50_SYMBOLS if 'BANK' in s],
        'tech': [s for s in NIFTY50_SYMBOLS if any(x in s for x in ['TECH', 'INFY', 'TCS', 'HCL'])],
        'consumers': [s for s in NIFTY50_SYMBOLS if any(x in s for x in ['ITC', 'HINDUNILVR', 'ASIANPAINT'])]
    }
    
    # Randomly choose winning and losing sectors
    sector_keys = list(sectors.keys())
    random.shuffle(sector_keys)
    winning_sectors = sector_keys[:1]
    losing_sectors = sector_keys[1:2]
    
    for sector in winning_sectors:
        for symbol in sectors[sector]:
            if symbol in prices:
                prices[symbol] *= random.uniform(1.08, 1.15)
                
    for sector in losing_sectors:
        for symbol in sectors[sector]:
            if symbol in prices:
                prices[symbol] *= random.uniform(0.85, 0.95)
                
    return prices

# --- Enhanced Game Flow Management with Qualification System ---
def update_game_state():
    """Handle level transitions, breaks, qualification checks, and game timing"""
    game_state = get_game_state()
    
    if game_state.game_status == "Running":
        current_time = time.time()
        level_duration = game_state.level_durations[game_state.current_level - 1]
        level_end_time = game_state.game_start_time + level_duration
        
        if current_time >= level_end_time:
            # Level completed - check qualifications
            check_level_qualifications()
            
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
            # Break over, start next level with qualified players only
            game_state.current_level += 1
            game_state.game_status = "Running"
            game_state.game_start_time = current_time
            game_state.difficulty_level = game_state.current_level
            
            # Reset qualification for next level
            if game_state.current_level < game_state.total_levels:
                game_state.qualified_players = set()
            
            play_sound('opening_bell')
            st.toast(f"🚀 Level {game_state.current_level} Starting! {len(game_state.qualified_players)} players qualified!", icon="🎮")

def check_level_qualifications():
    """Check which players qualify for the next level"""
    game_state = get_game_state()
    current_level = game_state.current_level
    
    if current_level not in QUALIFICATION_CRITERIA:
        return
        
    criteria = QUALIFICATION_CRITERIA[current_level]
    min_value = criteria["min_portfolio_value"]
    
    qualified_count = 0
    eliminated_count = 0
    
    for player_name, player in game_state.players.items():
        if player_name in game_state.eliminated_players:
            continue
            
        holdings_value = sum(game_state.prices.get(s, 0) * q for s, q in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        
        if total_value >= min_value:
            game_state.qualified_players.add(player_name)
            qualified_count += 1
        else:
            game_state.eliminated_players.add(player_name)
            eliminated_count += 1
    
    # Store level results
    game_state.level_results[current_level] = {
        'qualified': qualified_count,
        'eliminated': eliminated_count,
        'min_required': min_value
    }
    
    # Announce results
    if qualified_count > 0:
        st.toast(f"🎯 Level {current_level} Results: {qualified_count} players qualified!", icon="✅")
        play_sound('qualification')
    if eliminated_count > 0:
        st.toast(f"😔 {eliminated_count} players eliminated from Level {current_level}", icon="💔")
        play_sound('elimination')

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
def trigger_manual_event(event_type, target_symbol=None):
    """Admin function to manually trigger market events"""
    game_state = get_game_state()
    game_state.manual_event_pending = {
        'type': event_type,
        'target_symbol': target_symbol,
        'timestamp': time.time()
    }
    st.toast(f"🎪 Manual {event_type} triggered!", icon="🎯")

# --- Enhanced Trading with Admin Halt and Qualification Checks ---
def execute_trade(player_name, player, action, symbol, qty, prices, is_algo=False, order_type="Market"):
    game_state = get_game_state()
    
    # Check if player is eliminated
    if player_name in game_state.eliminated_players:
        if not is_algo:
            st.error("❌ You have been eliminated from the current level!")
        return False
    
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

def auto_square_off_positions(prices):
    """Auto square off all positions at game end"""
    game_state = get_game_state()
    for player_name, player in game_state.players.items():
        for symbol, qty in list(player['holdings'].items()):
            if qty != 0:
                action = "Sell" if qty > 0 else "Buy"
                execute_trade(player_name, player, action, symbol, abs(qty), prices)
        player['value_history'].append(player['capital'])

# --- UI Components ---
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

def render_trade_interface(player_name, player, prices, is_trade_disabled):
    st.subheader("⚡ Quick Trade")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        symbol = st.selectbox("Symbol", ALL_SYMBOLS, key="trade_symbol")
    with col2:
        action = st.selectbox("Action", ["Buy", "Sell", "Short"], key="trade_action")
    with col3:
        qty = st.number_input("Quantity", min_value=1, max_value=10000, value=10, key="trade_qty")
    
    if st.button("Execute Trade", type="primary", disabled=is_trade_disabled, use_container_width=True):
        if execute_trade(player_name, player, action, symbol, qty, prices):
            st.success("Trade executed successfully!")
            play_sound('success')
        else:
            play_sound('error')
    
    # Display current holdings
    st.subheader("📦 Your Holdings")
    if player['holdings']:
        holdings_data = []
        for symbol, qty in player['holdings'].items():
            if qty != 0:
                current_price = prices.get(symbol, 0)
                value = current_price * qty
                holdings_data.append({
                    'Symbol': symbol,
                    'Quantity': qty,
                    'Avg Price': 'N/A',  # Simplified for this example
                    'Current Price': current_price,
                    'Value': value
                })
        
        if holdings_data:
            holdings_df = pd.DataFrame(holdings_data)
            st.dataframe(holdings_df, use_container_width=True)
    else:
        st.info("No holdings yet. Start trading!")

def render_algo_trading_tab(player_name, player, is_trade_disabled):
    st.subheader("🤖 Algorithmic Trading")
    
    algo_options = ["Off", "Momentum", "Mean Reversion", "Market Making"]
    selected_algo = st.selectbox("Select Algorithm", algo_options, index=0)
    
    if selected_algo != "Off":
        st.info(f"🔄 {selected_algo} algorithm is now active")
        player['algo'] = selected_algo
        
        if st.button("Run Single Algo Cycle", disabled=is_trade_disabled):
            st.info("Algo cycle completed (simulated)")
    else:
        player['algo'] = "Off"

def render_transaction_history(player_name):
    game_state = get_game_state()
    st.subheader("📋 Transaction History")
    
    if player_name in game_state.transactions and game_state.transactions[player_name]:
        trans_df = pd.DataFrame(
            game_state.transactions[player_name],
            columns=["Time", "Action", "Symbol", "Qty", "Price", "Total"]
        )
        st.dataframe(trans_df, use_container_width=True)
    else:
        st.info("No transactions yet.")

def render_strategy_tab(player, prices):
    st.subheader("📈 Portfolio Analysis")
    
    # Simple portfolio metrics
    holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
    total_value = player['capital'] + holdings_value
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cash Allocation", f"{(player['capital'] / total_value * 100):.1f}%")
        st.metric("Stock Allocation", f"{(holdings_value / total_value * 100):.1f}%")
    with col2:
        st.metric("Number of Holdings", len(player['holdings']))
        st.metric("Total Trades", len(player.get('trade_timestamps', [])))
    
    # Simple chart of portfolio value history
    if len(player['value_history']) > 1:
        st.subheader("📊 Portfolio Value Trend")
        chart_data = pd.DataFrame({
            'Portfolio Value': player['value_history']
        })
        st.line_chart(chart_data)

def render_player_qualification_status(player_name, prices):
    """Show qualification status and progress for current player"""
    game_state = get_game_state()
    
    if player_name in game_state.eliminated_players:
        st.error("## ❌ Better Luck Next Time!")
        st.info("""
        You have been eliminated from the current level. 
        
        **What happened?**
        - You didn't meet the minimum portfolio value requirement
        - Watch the remaining players compete in the next levels
        - Join again in the next tournament!
        
        *Tip for next time: Focus on risk management and consistent gains.*
        """)
        return False
        
    elif game_state.current_level < game_state.total_levels and player_name not in game_state.qualified_players:
        # Show qualification progress
        current_criteria = QUALIFICATION_CRITERIA.get(game_state.current_level, {})
        if current_criteria:
            min_required = current_criteria["min_portfolio_value"]
            holdings_value = sum(prices.get(s, 0) * q for s, q in game_state.players[player_name]['holdings'].items())
            total_value = game_state.players[player_name]['capital'] + holdings_value
            
            progress = min(100, (total_value / min_required) * 100)
            
            st.subheader("🎯 Qualification Progress")
            st.write(f"**Target for Level {game_state.current_level}:** {format_indian_currency(min_required)}")
            st.write(f"**Your Portfolio:** {format_indian_currency(total_value)}")
            
            # Progress bar
            st.progress(progress / 100)
            st.write(f"Progress: {progress:.1f}%")
            
            if total_value >= min_required:
                st.success("✅ You have qualified for the next level!")
            else:
                remaining = max(0, min_required - total_value)
                st.warning(f"💰 Need {format_indian_currency(remaining)} more to qualify")
    
    return True

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
            
            # Qualification criteria
            st.subheader("🎯 Qualification Criteria")
            col1, col2 = st.columns(2)
            with col1:
                QUALIFICATION_CRITERIA[1]["min_portfolio_value"] = st.number_input(
                    "Level 1 Min (₹)", 
                    value=QUALIFICATION_CRITERIA[1]["min_portfolio_value"],
                    min_value=INITIAL_CAPITAL,
                    step=100000
                )
            with col2:
                QUALIFICATION_CRITERIA[2]["min_portfolio_value"] = st.number_input(
                    "Level 2 Min (₹)", 
                    value=QUALIFICATION_CRITERIA[2]["min_portfolio_value"],
                    min_value=INITIAL_CAPITAL,
                    step=100000
                )
            
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
                        game_state.qualified_players = set()
                        game_state.eliminated_players = set()
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

def render_global_views(prices, is_admin=False):
    with st.container(border=True):
        st.subheader("🌐 Global Market View")
        
        # Show qualification status for all players (admin view)
        if is_admin:
            game_state = get_game_state()
            st.subheader("🎯 Player Qualification Status")
            qualified_count = len(game_state.qualified_players)
            eliminated_count = len(game_state.eliminated_players)
            active_count = len(game_state.players) - eliminated_count
            
            col1, col2, col3 = st.columns(3)
            col1.metric("✅ Qualified", qualified_count)
            col2.metric("🎯 Active", active_count)
            col3.metric("❌ Eliminated", eliminated_count)
        
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

        # Market data
        st.markdown("---")
        st.subheader("📈 Live Market Data")
        render_live_market_table(prices)

def render_leaderboard(prices):
    game_state = get_game_state()
    lb = []
    for pname, pdata in game_state.players.items():
        holdings_value = sum(prices.get(s, 0) * q for s, q in pdata['holdings'].items())
        total_value = pdata['capital'] + holdings_value
        starting_capital = INITIAL_CAPITAL * 5 if pdata['mode'] == 'HNI' else INITIAL_CAPITAL
        pnl = total_value - starting_capital
        sharpe_ratio = calculate_sharpe_ratio(pdata.get('value_history', []))
        
        # Add qualification status
        status = "✅ Qualified" if pname in game_state.qualified_players else "❌ Eliminated" if pname in game_state.eliminated_players else "🎯 Active"
        
        lb.append((pname, pdata['mode'], total_value, pnl, sharpe_ratio, status))
    
    if lb:
        lb_df = pd.DataFrame(lb, columns=["Player", "Mode", "Portfolio Value", "P&L", "Sharpe Ratio", "Status"])
        lb_df = lb_df.sort_values("Portfolio Value", ascending=False).reset_index(drop=True)
        
        # Color coding based on status
        def color_status(val):
            if val == "✅ Qualified": return 'color: green'
            elif val == "❌ Eliminated": return 'color: red'
            return 'color: orange'
        
        styled_df = lb_df.style.format({
            "Portfolio Value": format_indian_currency,
            "P&L": format_indian_currency,
            "Sharpe Ratio": "{:.2f}"
        }).applymap(color_status, subset=['Status'])
        
        st.dataframe(styled_df, use_container_width=True)
        
        if game_state.game_status == "Finished":
            if not getattr(game_state, 'auto_square_off_complete', False):
                auto_square_off_positions(prices)
                game_state.auto_square_off_complete = True
                st.rerun()

            st.balloons()
            # Find the winner (highest among qualified players)
            qualified_winners = lb_df[lb_df['Status'] == '✅ Qualified']
            if not qualified_winners.empty:
                winner = qualified_winners.iloc[0]
                st.success(f"🎉 Tournament Winner: **{winner['Player']}**! 🎉")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("🏆 Final Portfolio", format_indian_currency(winner['Portfolio Value']))
                col2.metric("💰 Total P&L", format_indian_currency(winner['P&L']))
                col3.metric("⚡ Sharpe Ratio", f"{winner['Sharpe Ratio']:.2f}")

def render_live_market_table(prices):
    """Display current market prices"""
    market_data = []
    for symbol in ALL_SYMBOLS[:15]:  # Show first 15 symbols to avoid clutter
        price = prices.get(symbol, 0)
        if price > 0:
            market_data.append({
                'Symbol': symbol.replace('.NS', ''),
                'Price': format_indian_currency(price),
                'Raw Price': price
            })
    
    if market_data:
        market_df = pd.DataFrame(market_data)
        market_df = market_df.sort_values('Raw Price', ascending=False)
        st.dataframe(market_df[['Symbol', 'Price']], use_container_width=True, hide_index=True)

def render_right_sidebar(prices):
    with st.sidebar:
        st.title("📊 Quick Stats")
        
        game_state = get_game_state()
        
        # Game status
        st.subheader("🎮 Game Status")
        status_color = {
            "Running": "🟢",
            "Stopped": "🔴", 
            "Break": "🟡",
            "Finished": "🔵"
        }
        st.write(f"{status_color.get(game_state.game_status, '⚪')} {game_state.game_status}")
        
        if game_state.game_status == "Running":
            remaining, time_type = get_remaining_time()
            if time_type == "level":
                minutes, seconds = divmod(int(remaining), 60)
                st.write(f"⏰ Level {game_state.current_level} ends in: {minutes:02d}:{seconds:02d}")
            elif time_type == "break":
                st.write(f"⏸️ Break ends in: {int(remaining)}s")
        
        # Player count
        active_players = len([p for p in game_state.players if p not in game_state.eliminated_players])
        st.write(f"👥 Active Players: {active_players}/{len(game_state.players)}")
        
        # Market sentiment
        st.subheader("📈 Market Sentiment")
        avg_sentiment = np.mean(list(game_state.market_sentiment.values())) if game_state.market_sentiment else 0
        sentiment_label = "🐂 Bullish" if avg_sentiment > 0.1 else "🐻 Bearish" if avg_sentiment < -0.1 else "➡️ Neutral"
        st.write(f"{sentiment_label} ({avg_sentiment:.2f})")

def render_main_interface(prices):
    game_state = get_game_state()
    
    # Initialize session state for refresh tracking
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = time.time()
    
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
            
            # Show qualification criteria
            current_criteria = QUALIFICATION_CRITERIA.get(game_state.current_level, {})
            if current_criteria:
                st.write(f"**Qualification Target:** {format_indian_currency(current_criteria['min_portfolio_value'])}")
            
            st.write(f"**Difficulty: Level {game_state.difficulty_level}** | "
                    f"**Active Players: {len([p for p in game_state.players if p not in game_state.eliminated_players])}**")
            
            if remaining_time <= 60:
                st.error("⏰ FINAL MINUTE!")
                
        elif time_type == "break":
            st.subheader(f"⏸️ Break Time | Next: Level {game_state.current_level + 1}")
            st.warning(f"Break ends in: {int(remaining_time)} seconds")
            
            # Show level results
            if game_state.current_level in game_state.level_results:
                results = game_state.level_results[game_state.current_level]
                st.info(f"Level {game_state.current_level} Results: {results['qualified']} qualified, {results['eliminated']} eliminated")
            
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
        player_name = st.query_params.get("player")
        
        # Check if player can participate
        if render_player_qualification_status(player_name, prices):
            col1, col2 = st.columns([2, 1])
            with col1:
                render_trade_execution_panel(prices)
            with col2:
                render_global_views(prices)
    else:
        st.info("🎯 Welcome to BlockVista Market Frenzy Tournament Edition! Join the game from the sidebar.")
        render_global_views(prices)

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
                           game_state.circuit_breaker_active or
                           acting_player in game_state.eliminated_players)

        with tab1:
            render_trade_interface(acting_player, player, prices, is_trade_disabled)
        with tab2:
            render_algo_trading_tab(acting_player, player, is_trade_disabled)
        with tab3:
            render_transaction_history(acting_player)
        with tab4:
            render_strategy_tab(player, prices)

# --- Main Application Loop with Fixed Session State ---
def main():
    # Initialize all required session state variables
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
        final_prices = run_game_tick(prices_with_derivatives)  # This was the missing function!
        
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
    
    # Fixed auto-refresh with proper session state handling
    current_time = time.time()
    refresh_interval = 1.5 if game_state.game_status == "Running" else 3
    if current_time - st.session_state.last_refresh > refresh_interval:
        st.session_state.last_refresh = current_time
        st.rerun()

if __name__ == "__main__":
    main()

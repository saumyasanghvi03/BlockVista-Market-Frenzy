# ======================= Expo Game: BlockVista Market Frenzy - Multi-Level Tournament ======================

import streamlit as st
import pandas as pd
import yfinance as yf
import time
import random
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
from pypfopt import EfficientFrontier, risk_models, expected_returns
import threading
from collections import defaultdict, deque
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor

# --- Page Configuration ---
st.set_page_config(layout="wide", page_title="BlockVista Market Frenzy", page_icon="📈")

# --- API & Game Configuration ---
GAME_NAME = "BlockVista Market Frenzy"
INITIAL_CAPITAL = 1000000  # ₹10L
MAX_PLAYERS = 30
PRICE_HISTORY_LIMIT = 30
PLAYER_HISTORY_LIMIT = 50

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

# --- Level Configuration ---
LEVEL_CONFIG = {
    1: {
        "name": "Beginner Arena",
        "duration_minutes": 5,
        "volatility": 0.8,
        "margin_requirement": 0.15,
        "winning_criteria": {
            "min_return": 0.03,  # 3% minimum return
            "max_drawdown": -0.10,  # Max 10% loss
            "diversification": 2,  # Min 2 different assets
        },
        "description": "Learn the basics with stable markets"
    },
    2: {
        "name": "Professional Challenge", 
        "duration_minutes": 7,
        "volatility": 1.5,
        "margin_requirement": 0.25,
        "winning_criteria": {
            "min_return": 0.08,  # 8% minimum return
            "max_drawdown": -0.15,  # Max 15% loss  
            "sharpe_ratio": 0.5,  # Minimum Sharpe ratio
            "diversification": 4,  # Min 4 different assets
        },
        "description": "Increased volatility and stricter requirements"
    },
    3: {
        "name": "Expert Gauntlet",
        "duration_minutes": 10,
        "volatility": 2.5,
        "margin_requirement": 0.35,
        "winning_criteria": {
            "min_return": 0.15,  # 15% minimum return
            "max_drawdown": -0.20,  # Max 20% loss
            "sharpe_ratio": 1.0,  # Good risk-adjusted returns
            "diversification": 6,  # Min 6 different assets
            "min_trades": 3,  # Minimum activity
        },
        "description": "High volatility with professional standards"
    }
}

# --- Player Type Configuration ---
PLAYER_TYPES = {
    "HFT": {
        "initial_capital_multiplier": 1.0,
        "slippage_multiplier": 0.3,
        "winning_bonus": 1.2,
        "description": "High-Frequency Trader - Low slippage, speed focus"
    },
    "HNI": {
        "initial_capital_multiplier": 5.0,
        "slippage_multiplier": 1.0, 
        "winning_bonus": 1.1,
        "description": "High Net-worth Individual - Large capital advantage"
    },
    "Trader": {
        "initial_capital_multiplier": 1.0,
        "slippage_multiplier": 0.8,
        "winning_bonus": 1.15,
        "description": "Professional Trader - Balanced approach"
    },
    "MF Manager": {
        "initial_capital_multiplier": 2.0,
        "slippage_multiplier": 1.2,
        "winning_bonus": 1.05,
        "description": "Mutual Fund Manager - Diversification focus, higher slippage"
    }
}

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

# --- Enhanced Game State Management with Performance Optimizations ---
class GameState:
    """Enhanced game state with multi-level support, thread safety, and performance optimizations"""
    def __init__(self):
        self.players = {}
        self.game_status = "Stopped"
        self.current_level = 1
        self.level_start_time = 0
        self.level_duration_seconds = LEVEL_CONFIG[1]["duration_minutes"] * 60
        self.futures_expiry_time = 0
        self.futures_settled = False
        self.prices = {}
        self.base_real_prices = {}
        self.price_history = deque(maxlen=PRICE_HISTORY_LIMIT)
        self.transactions = defaultdict(lambda: deque(maxlen=100))  # Limited transaction history
        self.market_sentiment = {s: 0 for s in ALL_SYMBOLS}
        self.liquidity = {s: random.uniform(0.5, 1.0) for s in ALL_SYMBOLS}
        self.event_active = False
        self.event_type = None
        self.event_target_symbol = None
        self.event_end = 0
        self.volatility_multiplier = LEVEL_CONFIG[1]["volatility"]
        self.news_feed = deque(maxlen=10)  # Limited news feed
        self.auto_square_off_complete = False
        self.block_deal_offer = None
        self.closing_warning_triggered = False
        self.current_margin_requirement = LEVEL_CONFIG[1]["margin_requirement"]
        self.bid_ask_spread = 0.001
        self.slippage_threshold = 10
        self.base_slippage_rate = 0.005
        self.hft_rebate_window = 60
        self.hft_rebate_trades = 5
        self.hft_rebate_amount = 5000
        self.short_squeeze_threshold = 3
        self.level_winners = {1: [], 2: [], 3: []}
        self.performance_boost_active = {}
        self._lock = threading.RLock()
        self._last_cleanup = time.time()
        self._batch_update_cache = {}

    def acquire_lock(self):
        """Acquire thread lock"""
        return self._lock

    def cleanup_old_data(self):
        """Periodic cleanup to prevent memory bloat"""
        current_time = time.time()
        if current_time - self._last_cleanup < 30:  # Cleanup every 30 seconds
            return
            
        with self._lock:
            # Clean up player histories
            for player in self.players.values():
                if 'value_history' in player and len(player['value_history']) > PLAYER_HISTORY_LIMIT:
                    player['value_history'] = deque(list(player['value_history'])[-PLAYER_HISTORY_LIMIT:], 
                                                   maxlen=PLAYER_HISTORY_LIMIT)
            
            # Clear batch cache periodically
            self._batch_update_cache.clear()
            
            self._last_cleanup = current_time

    def reset(self):
        """Reset game but preserve level progress"""
        with self._lock:
            # Store current level before reset
            current_level = self.current_level
            level_winners = self.level_winners.copy()
            
            # Re-initialize with optimizations
            self.__init__()
            
            # Restore level progress
            self.current_level = current_level
            self.level_winners = level_winners
            
            # Update level-specific settings
            level_config = LEVEL_CONFIG[self.current_level]
            self.volatility_multiplier = level_config["volatility"]
            self.current_margin_requirement = level_config["margin_requirement"]
            self.level_duration_seconds = level_config["duration_minutes"] * 60

    def add_player(self, player_name, player_data):
        """Thread-safe player addition with capacity check"""
        with self._lock:
            if len(self.players) >= MAX_PLAYERS:
                return False, "Game is full! Maximum 30 players allowed."
                
            if player_name not in self.players:
                # Convert lists to deques for performance
                player_data['value_history'] = deque([player_data['value_history'][0]], 
                                                   maxlen=PLAYER_HISTORY_LIMIT)
                player_data['trade_timestamps'] = deque(maxlen=100)
                player_data['pending_orders'] = deque(maxlen=50)
                
                self.players[player_name] = player_data
                self.transactions[player_name] = deque(maxlen=100)
                return True, "Player added successfully."
            return False, "Name is already taken!"

    def update_player(self, player_name, updates):
        """Thread-safe player update"""
        with self._lock:
            if player_name in self.players:
                self.players[player_name].update(updates)
                return True
            return False

    def get_player(self, player_name):
        """Thread-safe player retrieval"""
        with self._lock:
            return self.players.get(player_name)

    def batch_update_portfolio_values(self, prices):
        """Batch update all player portfolio values - optimized for performance"""
        with self._lock:
            for player in self.players.values():
                holdings_value = 0
                for symbol, qty in player['holdings'].items():
                    holdings_value += prices.get(symbol, 0) * qty
                
                total_value = player['capital'] + holdings_value
                player['value_history'].append(total_value)

    def get_leaderboard_data(self, prices):
        """Optimized leaderboard data calculation"""
        leaderboard_data = []
        level_criteria = LEVEL_CONFIG.get(self.current_level, LEVEL_CONFIG[1])["winning_criteria"]
        
        for name, player in self.players.items():
            # Calculate portfolio value
            holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
            total_value = player['capital'] + holdings_value
            
            # Calculate level performance
            level_start_value = player.get('level_start_value', 
                PLAYER_TYPES[player['mode']]['initial_capital_multiplier'] * INITIAL_CAPITAL)
            level_return = (total_value - level_start_value) / level_start_value if level_start_value > 0 else 0
            
            # Check qualification (cached for performance)
            qualifies = self.check_player_qualifies(player, level_criteria)
            
            leaderboard_data.append({
                'Player': name,
                'Type': player['mode'],
                'Portfolio Value': total_value,
                'Level Return': level_return,
                'Qualifies': qualifies
            })
        
        return leaderboard_data

    def advance_to_next_level(self):
        """Advance qualified players to the next level"""
        with self._lock:
            if self.current_level >= 3:
                return False  # Game complete
                
            next_level = self.current_level + 1
            current_winners = self.evaluate_level_winners()
            
            if not current_winners:
                return False
                
            # Store winners for current level
            self.level_winners[self.current_level] = current_winners
            
            # Store current players before clearing
            old_players = self.players.copy()
            
            # Clear and recreate players for next level
            self.players = {}
            for player_name in current_winners:
                old_player_data = old_players.get(player_name)
                if old_player_data:
                    # Apply performance-based capital adjustment
                    performance_multiplier = self.calculate_performance_multiplier(old_player_data)
                    player_type_config = PLAYER_TYPES[old_player_data['mode']]
                    
                    holdings_value = sum(self.prices.get(s, 0) * q for s, q in old_player_data['holdings'].items())
                    new_capital = old_player_data['capital'] * performance_multiplier * player_type_config['winning_bonus']
                    total_start_value = new_capital + holdings_value
                    
                    self.players[player_name] = {
                        "name": player_name,
                        "mode": old_player_data['mode'],
                        "capital": new_capital,
                        "holdings": old_player_data['holdings'].copy(),
                        "pnl": 0,
                        "leverage": 1.0,
                        "margin_calls": 0,
                        "pending_orders": deque(maxlen=50),
                        "algo": old_player_data.get('algo', 'Off'),
                        "custom_algos": old_player_data.get('custom_algos', {}).copy(),
                        "slippage_multiplier": player_type_config['slippage_multiplier'],
                        "value_history": deque([total_start_value], maxlen=PLAYER_HISTORY_LIMIT),
                        "trade_timestamps": deque(maxlen=100),
                        "level_start_value": total_start_value
                    }
            
            self.current_level = next_level
            level_config = LEVEL_CONFIG[next_level]
            
            # Update game parameters for new level
            self.volatility_multiplier = level_config["volatility"]
            self.current_margin_requirement = level_config["margin_requirement"]
            self.level_duration_seconds = level_config["duration_minutes"] * 60
            
            # Reset game state for new level
            self.game_status = "Running"
            self.level_start_time = time.time()
            self.futures_expiry_time = time.time() + (self.level_duration_seconds / 2)
            self.auto_square_off_complete = False
            self.closing_warning_triggered = False
            self.futures_settled = False
            
            return True

    def evaluate_level_winners(self):
        """Evaluate which players qualify for the next level"""
        with self._lock:
            winners = []
            level_criteria = LEVEL_CONFIG[self.current_level]["winning_criteria"]
            
            for player_name, player_data in self.players.items():
                if self.check_player_qualifies(player_data, level_criteria):
                    winners.append(player_name)
                    
            return winners

    def check_player_qualifies(self, player_data, criteria):
        """Check if player meets level completion criteria"""
        try:
            # Calculate current portfolio value
            holdings_value = sum(self.prices.get(symbol, 0) * qty for symbol, qty in player_data['holdings'].items())
            total_value = player_data['capital'] + holdings_value
            
            # Calculate return for this level
            level_start_value = player_data.get('level_start_value', 
                PLAYER_TYPES[player_data['mode']]['initial_capital_multiplier'] * INITIAL_CAPITAL)
            level_return = (total_value - level_start_value) / level_start_value
            
            # Check minimum return
            if level_return < criteria.get("min_return", -1.0):
                return False
                
            # Check maximum drawdown
            if 'value_history' in player_data and len(player_data['value_history']) > 1:
                peak = max(player_data['value_history'])
                current = player_data['value_history'][-1]
                drawdown = (current - peak) / peak
                if drawdown < criteria.get("max_drawdown", -1.0):
                    return False
            
            # Check Sharpe ratio
            if "sharpe_ratio" in criteria:
                sharpe = calculate_sharpe_ratio(list(player_data.get('value_history', [])))
                if sharpe < criteria["sharpe_ratio"]:
                    return False
            
            # Check diversification
            if "diversification" in criteria:
                unique_assets = len([s for s in player_data['holdings'].keys() if player_data['holdings'].get(s, 0) != 0])
                if unique_assets < criteria["diversification"]:
                    return False
            
            # Check minimum trades
            if "min_trades" in criteria:
                trade_count = len(self.transactions.get(player_data['name'], []))
                if trade_count < criteria["min_trades"]:
                    return False
                    
            return True
            
        except Exception as e:
            return False

    def calculate_performance_multiplier(self, player_data):
        """Calculate capital adjustment based on performance"""
        try:
            holdings_value = sum(self.prices.get(symbol, 0) * qty for symbol, qty in player_data['holdings'].items())
            total_value = player_data['capital'] + holdings_value
            
            level_start_value = player_data.get('level_start_value', 
                PLAYER_TYPES[player_data['mode']]['initial_capital_multiplier'] * INITIAL_CAPITAL)
            level_return = (total_value - level_start_value) / level_start_value
            
            # Base multiplier with diminishing returns
            if level_return <= 0:
                return 0.8  # Penalty for negative returns
            elif level_return <= 0.1:
                return 1.0 + level_return  # 1x to 1.1x
            elif level_return <= 0.25:
                return 1.1 + (level_return - 0.1) * 0.5  # 1.1x to 1.175x
            else:
                return 1.2  # Cap at 1.2x for very high returns
                
        except:
            return 1.0

@st.cache_resource
def get_game_state():
    """Returns the singleton GameState object with performance optimizations"""
    return GameState()

# --- Performance Optimized Sound Effects ---
def play_sound(sound_type):
    """Lightweight sound effects to reduce UI lag"""
    js = ""
    if sound_type == 'success':
        js = """<script>if(typeof Tone!=='undefined'){new Tone.Synth().toDestination().triggerAttackRelease("C5","8n");}</script>"""
    elif sound_type == 'error':
        js = """<script>if(typeof Tone!=='undefined'){new Tone.Synth().toDestination().triggerAttackRelease("C3","8n");}</script>"""
    elif sound_type == 'level_up':
        js = """<script>if(typeof Tone!=='undefined'){const s=new Tone.Synth().toDestination(),n=Tone.now();s.triggerAttackRelease("E5","8n",n);s.triggerAttackRelease("G5","8n",n+0.1);s.triggerAttackRelease("C6","8n",n+0.2);}</script>"""
    
    if js:
        st.components.v1.html(js, height=0)

# --- Optimized Data Fetching & Market Simulation ---
@st.cache_data(ttl=86400)
def get_daily_base_prices():
    """Fetches real-world prices from yfinance with performance optimizations"""
    prices = {}
    yf_symbols = NIFTY50_SYMBOLS + CRYPTO_SYMBOLS + [GOLD_SYMBOL, NIFTY_INDEX_SYMBOL, BANKNIFTY_INDEX_SYMBOL]
    
    # Progress bar for better UX
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        batch_size = 6  # Optimal batch size for 30 players
        total_batches = (len(yf_symbols) + batch_size - 1) // batch_size
        
        for i in range(0, len(yf_symbols), batch_size):
            batch = yf_symbols[i:i + batch_size]
            status_text.text(f"Fetching prices... Batch {i//batch_size + 1}/{total_batches}")
            
            data = yf.download(tickers=batch, period="1d", interval="1m", progress=False, timeout=10)
            for symbol in batch:
                if not data.empty and symbol in data['Close'] and not pd.isna(data['Close'][symbol].iloc[-1]):
                    prices[symbol] = data['Close'][symbol].iloc[-1]
                else: # Fallback
                    prices[symbol] = random.uniform(10, 50000)
            
            progress_bar.progress(min((i + batch_size) / len(yf_symbols), 1.0))
            time.sleep(0.3)  # Rate limiting
            
        status_text.text("Price data loaded successfully!")
        time.sleep(1)
        status_text.empty()
        progress_bar.empty()
        
    except Exception as e:
        st.warning(f"Using fallback prices due to API error: {e}")
        for symbol in yf_symbols:
            prices[symbol] = random.uniform(10, 50000)
        progress_bar.empty()
        status_text.empty()
        
    return prices

def simulate_tick_prices(last_prices):
    """Optimized price simulation"""
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
    """Optimized derived price calculation"""
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

@st.cache_data(ttl=3600, show_spinner=False)
def get_historical_data(symbols, period="1mo"):
    """Optimized historical data fetching"""
    try:
        if isinstance(symbols, str):
            symbols = [symbols]
            
        data = yf.download(tickers=symbols, period=period, progress=False, timeout=10)
        close_data = data['Close']
        if isinstance(close_data, pd.Series):
            return close_data.to_frame(name=symbols[0])
        return close_data
    except Exception:
        return pd.DataFrame()

# --- Optimized Game Logic Functions ---
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
    """Optimized event adjustment"""
    adjusted_prices = prices.copy()
    game_state = get_game_state()
    
    event_effects = {
        "Flash Crash": lambda: {k: v * random.uniform(0.95, 0.98) for k, v in adjusted_prices.items()},
        "Bull Rally": lambda: {k: v * random.uniform(1.02, 1.05) for k, v in adjusted_prices.items()},
        "Banking Boost": lambda: {**adjusted_prices, **{sym: adjusted_prices.get(sym, 0) * 1.07 
                      for sym in ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'AXISBANK.NS'] 
                      if sym in adjusted_prices}},
        "Symbol Bull Run": lambda: {**adjusted_prices, target_symbol: adjusted_prices.get(target_symbol, 0) * 1.15} if target_symbol else adjusted_prices,
        "Symbol Crash": lambda: {**adjusted_prices, target_symbol: adjusted_prices.get(target_symbol, 0) * 0.85} if target_symbol else adjusted_prices,
    }
    
    if event_type in event_effects:
        adjusted_prices = event_effects[event_type]()
    
    return adjusted_prices

def format_indian_currency(n):
    """Optimized currency formatting"""
    if n is None: return "₹0.00"
    n = float(n)
    if abs(n) < 100000: return f"₹{n:,.2f}"
    elif abs(n) < 10000000: return f"₹{n/100000:.2f}L"
    else: return f"₹{n/10000000:.2f}Cr"

def optimize_portfolio(player_holdings):
    """Optimized portfolio optimization"""
    symbols = [s for s in player_holdings.keys() if s in NIFTY50_SYMBOLS + CRYPTO_SYMBOLS]
    if len(symbols) < 2: return None, "Need at least 2 assets to optimize."
    try:
        hist_data = get_historical_data(symbols, period="1mo")  # Shorter period for performance
        if hist_data.empty or hist_data.isnull().values.any(): return None, "Could not fetch sufficient historical data."
        mu = expected_returns.mean_historical_return(hist_data)
        S = risk_models.sample_cov(hist_data)
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        return ef.clean_weights(), ef.portfolio_performance(verbose=False)
    except Exception as e: return None, f"Optimization failed: {e}"

def calculate_sharpe_ratio(value_history):
    """Optimized Sharpe ratio calculation"""
    if len(value_history) < 2: return 0.0
    returns = pd.Series(value_history).pct_change().dropna()
    if returns.std() == 0: return 0.0
    return (returns.mean() / returns.std()) * np.sqrt(252)

# --- Performance Optimized UI Functions ---
def should_rerun():
    """Smart rerun logic to reduce unnecessary updates"""
    game_state = get_game_state()
    
    # Always rerun if game is running
    if game_state.game_status == "Running":
        return True
        
    # Rerun if player count changed
    if 'last_player_count' not in st.session_state:
        st.session_state.last_player_count = len(game_state.players)
        return True
        
    if st.session_state.last_player_count != len(game_state.players):
        st.session_state.last_player_count = len(game_state.players)
        return True
        
    return False

def render_sidebar():
    """Optimized sidebar rendering"""
    game_state = get_game_state()
    current_level_config = LEVEL_CONFIG.get(game_state.current_level, LEVEL_CONFIG[1])
    
    if 'player' not in st.query_params:
        st.sidebar.title("📝 Game Entry")
        
        # Player limit check
        player_count = len(game_state.players)
        if player_count >= MAX_PLAYERS:
            st.sidebar.error(f"❌ Game is full! {MAX_PLAYERS}/{MAX_PLAYERS} players")
            st.sidebar.progress(1.0)
            return
        else:
            st.sidebar.write(f"Players: {player_count}/{MAX_PLAYERS}")
            st.sidebar.progress(player_count / MAX_PLAYERS)
            
        player_name = st.sidebar.text_input("Enter Name", key="name_input")
        
        # Enhanced player type selection
        player_types = list(PLAYER_TYPES.keys())
        mode = st.sidebar.radio("Select Player Type", player_types, 
                               format_func=lambda x: f"{x} - {PLAYER_TYPES[x]['description']}", 
                               key="mode_select")
        
        # Show current level information
        with st.sidebar.expander("Level Info", expanded=True):
            st.write(f"**{current_level_config['name']}**")
            st.write(f"⏱️ {current_level_config['duration_minutes']} min")
            st.write(f"⚡ {current_level_config['volatility']}x volatility")
            st.write(f"📊 {current_level_config['margin_requirement']*100}% margin")
        
        if st.sidebar.button("Join Game", type="primary", use_container_width=True):
            if player_name and player_name.strip():
                success, message = game_state.add_player(player_name, {
                    "name": player_name, 
                    "mode": mode, 
                    "capital": INITIAL_CAPITAL * PLAYER_TYPES[mode]['initial_capital_multiplier'], 
                    "holdings": {}, 
                    "pnl": 0, 
                    "leverage": 1.0, 
                    "margin_calls": 0, 
                    "pending_orders": deque(maxlen=50), 
                    "algo": "Off", 
                    "custom_algos": {},
                    "slippage_multiplier": PLAYER_TYPES[mode]['slippage_multiplier'],
                    "value_history": deque([INITIAL_CAPITAL * PLAYER_TYPES[mode]['initial_capital_multiplier']], 
                                         maxlen=PLAYER_HISTORY_LIMIT),
                    "trade_timestamps": deque(maxlen=100),
                    "level_start_value": INITIAL_CAPITAL * PLAYER_TYPES[mode]['initial_capital_multiplier']
                })
                if success:
                    st.query_params["player"] = player_name
                    st.rerun()
                else: 
                    st.sidebar.error(message)
            else: 
                st.sidebar.error("Please enter a valid name!")
    else:
        current_player = st.query_params['player']
        player_data = game_state.get_player(current_player)
        if player_data:
            st.sidebar.success(f"Logged in as {current_player}")
            st.sidebar.info(f"Type: {player_data.get('mode', 'N/A')}\nLevel: {game_state.current_level}")
        else:
            st.sidebar.error("Player data not found!")
            
        if st.sidebar.button("Logout", use_container_width=True):
            st.query_params.clear()
            st.rerun()

    # Admin controls
    st.sidebar.markdown("---")
    st.sidebar.title("🔐 Admin Controls")
    
    if st.session_state.get('role') != 'admin':
        password = st.sidebar.text_input("Enter Password", type="password", key="admin_pw")
        if password == ADMIN_PASSWORD:
            st.session_state.role = 'admin'
            st.rerun()
        elif password:
            st.sidebar.error("Incorrect Password")
    else:
        st.sidebar.success("🔓 Admin Mode")
        if st.sidebar.button("Logout Admin", use_container_width=True):
            del st.session_state['role']
            st.rerun()
        
        render_admin_controls()

def render_admin_controls():
    """Optimized admin controls"""
    game_state = get_game_state()
    
    # Player management
    st.sidebar.subheader("👥 Player Management")
    st.sidebar.write(f"Active: {len(game_state.players)}/{MAX_PLAYERS}")
    
    if game_state.players:
        player_to_remove = st.sidebar.selectbox("Select Player", list(game_state.players.keys()))
        col1, col2 = st.sidebar.columns(2)
        if col1.button("Remove", use_container_width=True):
            with game_state.acquire_lock():
                if player_to_remove in game_state.players:
                    del game_state.players[player_to_remove]
                    if player_to_remove in game_state.transactions:
                        del game_state.transactions[player_to_remove]
                    st.sidebar.success(f"Removed {player_to_remove}")
                    st.rerun()
        
        # Quick actions
        if col2.button("Remove All", use_container_width=True):
            game_state.players.clear()
            game_state.transactions.clear()
            st.rerun()

    # Game controls
    st.sidebar.subheader("🎮 Game Controls")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("▶️ Start", use_container_width=True, type="primary"):
            if game_state.players:
                game_state.game_status = "Running"
                game_state.level_start_time = time.time()
                game_state.futures_expiry_time = time.time() + (game_state.level_duration_seconds / 2)
                st.rerun()
            else: 
                st.sidebar.warning("Add players first!")
    with col2:
        if st.button("⏸️ Stop", use_container_width=True):
            game_state.game_status = "Stopped"
            st.rerun()
    
    if st.button("🔄 Reset Game", use_container_width=True):
        game_state.reset()
        st.rerun()
        
    # Level management
    st.sidebar.subheader("📊 Level Management")
    if game_state.game_status != "Running":
        new_level = st.sidebar.selectbox("Set Level", [1, 2, 3], 
                                       index=game_state.current_level-1,
                                       format_func=lambda x: f"Level {x}")
        if new_level != game_state.current_level:
            game_state.current_level = new_level
            level_config = LEVEL_CONFIG[new_level]
            game_state.volatility_multiplier = level_config["volatility"]
            game_state.current_margin_requirement = level_config["margin_requirement"]
            game_state.level_duration_seconds = level_config["duration_minutes"] * 60
            st.rerun()
    
    # Quick actions
    st.sidebar.subheader("⚡ Quick Actions")
    if st.button("Trigger News Event", use_container_width=True):
        news_item = random.choice(PRE_BUILT_NEWS)
        headline = news_item['headline']
        if "{symbol}" in headline:
            target_symbol = random.choice(NIFTY50_SYMBOLS)
            headline = headline.format(symbol=target_symbol.replace(".NS", ""))
        
        game_state.news_feed.appendleft(f"📢 {time.strftime('%H:%M:%S')} - {headline}")
        game_state.event_type = news_item['impact']
        game_state.event_target_symbol = target_symbol if "{symbol}" in news_item['headline'] else None
        game_state.event_active = True
        game_state.event_end = time.time() + 60
        st.sidebar.success("Event triggered!")

def render_main_interface(prices):
    """Optimized main interface"""
    game_state = get_game_state()
    current_level_config = LEVEL_CONFIG.get(game_state.current_level, LEVEL_CONFIG[1])
    
    st.title(f"📈 {GAME_NAME}")
    st.components.v1.html('<script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.7.77/Tone.js"></script>', height=0)

    # Performance optimized header
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.subheader(f"Level {game_state.current_level}")
        st.caption(current_level_config['name'])
    with col2:
        if game_state.game_status == "Running":
            remaining_time = max(0, game_state.level_duration_seconds - int(time.time() - game_state.level_start_time))
            st.metric("Time Remaining", f"{remaining_time // 60:02d}:{remaining_time % 60:02d}")
            st.progress(1 - (remaining_time / game_state.level_duration_seconds))
        else:
            st.metric("Status", game_state.game_status)
    with col3:
        st.metric("Volatility", f"{current_level_config['volatility']}x")
    with col4:
        st.metric("Players", f"{len(game_state.players)}/{MAX_PLAYERS}")

    if st.session_state.get('role') == 'admin':
        render_global_views(prices, is_admin=True)
    elif 'player' in st.query_params:
        render_player_interface(prices)
    else:
        st.info("🎯 Welcome to BlockVista Market Frenzy! Join the game from the sidebar.")
        render_global_views(prices)

def render_player_interface(prices):
    """Optimized player interface"""
    col1, col2 = st.columns([1, 1])
    with col1: 
        render_trade_execution_panel(prices)
    with col2: 
        render_global_views(prices)

def render_trade_execution_panel(prices):
    """Optimized trade panel"""
    game_state = get_game_state()
    
    with st.container(border=True):
        st.subheader("🎯 Trade Execution")
        acting_player = st.query_params.get("player")
        if not acting_player:
            st.warning("Please join the game to trade.")
            return
        
        player = game_state.get_player(acting_player)
        if not player:
            st.warning("Player data not found. Please rejoin.")
            return
            
        player_config = PLAYER_TYPES[player['mode']]
        
        st.markdown(f"**{acting_player}'s Terminal**")
        st.caption(f"Type: {player['mode']} | {player_config['description']}")
        
        # Performance metrics
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        level_start_value = player.get('level_start_value', total_value)
        level_return = (total_value - level_start_value) / level_start_value if level_start_value > 0 else 0
        
        # Check qualification status
        current_level_config = LEVEL_CONFIG.get(game_state.current_level, LEVEL_CONFIG[1])
        qualifies = game_state.check_player_qualifies(player, current_level_config["winning_criteria"])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cash", format_indian_currency(player['capital']))
        col2.metric("Portfolio", format_indian_currency(total_value))
        col3.metric("Return", f"{level_return:.2%}")
        col4.metric("Qualifies", "✅" if qualifies else "❌")
        
        if not qualifies and game_state.game_status == "Running":
            st.warning(f"⚠️ You don't meet Level {game_state.current_level} criteria!")
        
        # Quick trade interface
        render_quick_trade_interface(acting_player, player, prices)

def render_quick_trade_interface(player_name, player, prices):
    """Optimized quick trade interface"""
    st.markdown("---")
    st.subheader("⚡ Quick Trade")
    
    is_trade_disabled = get_game_state().game_status != "Running"
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        symbol = st.selectbox("Asset", 
                            [s.replace('.NS', '') for s in NIFTY50_SYMBOLS] + CRYPTO_SYMBOLS,
                            key=f"quick_asset_{player_name}")
        if symbol in CRYPTO_SYMBOLS:
            symbol_choice = symbol
        else:
            symbol_choice = symbol + '.NS'
    with col2:
        qty = st.number_input("Qty", min_value=1, value=10, key=f"quick_qty_{player_name}")
    with col3:
        action = st.selectbox("Action", ["Buy", "Sell", "Short"], key=f"quick_action_{player_name}")
    
    mid_price = prices.get(symbol_choice, 0)
    if mid_price > 0:
        ask_price = mid_price * (1 + get_game_state().bid_ask_spread / 2)
        bid_price = mid_price * (1 - get_game_state().bid_ask_spread / 2)
        st.info(f"Bid: {format_indian_currency(bid_price)} | Ask: {format_indian_currency(ask_price)}")
        
        if st.button(f"{action} {qty} {symbol}", use_container_width=True, disabled=is_trade_disabled, type="primary"):
            if execute_trade(player_name, player, action, symbol_choice, qty, prices): 
                play_sound('success')
            else: 
                play_sound('error')
            st.rerun()

def render_global_views(prices, is_admin=False):
    """Optimized global views"""
    with st.container(border=True):
        # Market overview
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("📊 Market Overview")
            render_optimized_market_table(prices)
        with col2:
            st.subheader("📰 News Feed")
            render_news_feed()
        
        st.markdown("---")
        st.subheader("🏆 Leaderboard")
        render_optimized_leaderboard(prices)
        
        if is_admin:
            st.markdown("---")
            st.subheader("📈 Performance Overview")
            render_admin_performance_chart()

def render_optimized_market_table(prices):
    """Highly optimized market table"""
    game_state = get_game_state()
    
    # Create efficient dataframe
    price_data = []
    for symbol, price in prices.items():
        if symbol in NIFTY50_SYMBOLS[:15] + CRYPTO_SYMBOLS[:3]:  # Limit displayed symbols
            change = 0.0
            if len(game_state.price_history) >= 2:
                prev_price = game_state.price_history[-2].get(symbol, price)
                change = ((price - prev_price) / prev_price) * 100
                
            price_data.append({
                'Symbol': symbol.replace('.NS', ''),
                'Price': price,
                'Change %': change
            })
    
    if price_data:
        df = pd.DataFrame(price_data)
        st.dataframe(
            df.style.format({
                'Price': lambda x: format_indian_currency(x),
                'Change %': lambda x: f"{x:.2f}%"
            }).apply(
                lambda x: ['color: green' if x['Change %'] > 0 else 'color: red' if x['Change %'] < 0 else '' for _ in x], 
                axis=1
            ), 
            use_container_width=True, 
            hide_index=True
        )

def render_news_feed():
    """Optimized news feed"""
    game_state = get_game_state()
    news_feed = list(game_state.news_feed)[:5]  # Only show recent news
    
    if news_feed:
        for news in news_feed:
            st.info(news)
    else:
        st.info("No market news at the moment.")

def render_optimized_leaderboard(prices):
    """Highly optimized leaderboard"""
    game_state = get_game_state()
    leaderboard_data = game_state.get_leaderboard_data(prices)
    
    if leaderboard_data:
        df = pd.DataFrame(leaderboard_data)
        df = df.sort_values('Portfolio Value', ascending=False)
        
        st.dataframe(
            df.style.format({
                'Portfolio Value': format_indian_currency,
                'Level Return': '{:.2%}'
            }).apply(
                lambda x: ['background: lightgreen' if x['Qualifies'] else '' for _ in x], 
                axis=1
            ), 
            use_container_width=True
        )
        
        # Show winners if game finished
        if game_state.game_status == "Finished":
            winners = [p for p in leaderboard_data if p['Qualifies']]
            if winners:
                st.success(f"🎉 Level {game_state.current_level} Winners: {', '.join([w['Player'] for w in winners])}")

def render_admin_performance_chart():
    """Optimized admin performance chart"""
    game_state = get_game_state()
    if not game_state.players:
        st.info("No players have joined yet.")
        return
        
    # Sample a few players for the chart to avoid overload
    sample_players = list(game_state.players.items())[:8]  # Limit to 8 players
    
    chart_data = {}
    for name, player_data in sample_players:
        if player_data.get('value_history'):
            chart_data[name] = list(player_data['value_history'])[-20:]  # Last 20 points
            
    if chart_data:
        df = pd.DataFrame(chart_data)
        st.line_chart(df)
    else:
        st.info("No trading activity to display.")

def execute_trade(player_name, player, action, symbol, qty, prices, is_algo=False, order_type="Market"):
    """Optimized trade execution"""
    game_state = get_game_state()
    mid_price = prices.get(symbol, 0)
    if mid_price == 0: 
        return False

    # Calculate trade price with slippage
    if action == "Buy":
        trade_price = mid_price * (1 + game_state.bid_ask_spread / 2)
    else: # Sell or Short
        trade_price = mid_price * (1 - game_state.bid_ask_spread / 2)
    
    trade_price *= calculate_slippage(player, symbol, qty, action)
    cost = trade_price * qty
    
    # Execute trade
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
        if current_qty > 0 and current_qty >= qty: # Closing a long
            player['capital'] += cost
            player['holdings'][symbol] -= qty
            trade_executed = True
        elif current_qty < 0 and abs(current_qty) >= qty: # Covering a short
            player['capital'] -= cost
            player['holdings'][symbol] += qty
            trade_executed = True
        
        if trade_executed and player['holdings'][symbol] == 0:
            del player['holdings'][symbol]
    
    if trade_executed:
        # Update market sentiment
        with game_state.acquire_lock():
            game_state.market_sentiment[symbol] = game_state.market_sentiment.get(symbol, 0) + (qty / 50) * (1 if action in ["Buy", "Short"] else -1)
        
        # Log transaction
        log_transaction(player_name, f"{order_type} {action}", symbol, qty, trade_price, cost, is_algo)
        
        # Record trade timestamp for HFT bonus
        player['trade_timestamps'].append(time.time())
        
    elif not is_algo:
        st.error("Trade failed: Insufficient capital or holdings.")
        
    return trade_executed

def log_transaction(player_name, action, symbol, qty, price, total, is_algo=False):
    """Optimized transaction logging"""
    game_state = get_game_state()
    prefix = "🤖" if is_algo else ""
    
    with game_state.acquire_lock():
        game_state.transactions[player_name].append([
            time.strftime("%H:%M:%S"), 
            f"{prefix} {action}".strip(), 
            symbol, qty, price, total
        ])
    
    if not is_algo:
        st.success(f"✅ {action} {qty} {symbol} @ {format_indian_currency(price)}")

# --- Optimized Main Game Loop ---
def run_game_tick(prices):
    """Optimized game tick processing"""
    game_state = get_game_state()
    if game_state.game_status != "Running": 
        return prices
    
    # Check level completion
    current_time = time.time()
    if current_time - game_state.level_start_time >= game_state.level_duration_seconds:
        if game_state.game_status != "Finished":
            play_sound('level_up')
            game_state.game_status = "Finished"
            auto_square_off_positions(prices)
            game_state.auto_square_off_complete = True
            
            # Evaluate and advance level
            level_winners = game_state.evaluate_level_winners()
            if level_winners and game_state.current_level < 3:
                if game_state.advance_to_next_level():
                    st.toast(f"🚀 Advanced to Level {game_state.current_level}!", icon="🎉")
    
    # Random events
    if not game_state.event_active and random.random() < 0.05:
        trigger_random_event()
    
    # Handle game mechanics
    if game_state.event_active and current_time >= game_state.event_end:
        game_state.event_active = False
        
    if game_state.event_active:
        prices = apply_event_adjustment(prices, game_state.event_type, game_state.event_target_symbol)
    
    # Batch updates for performance
    game_state.batch_update_portfolio_values(prices)
    handle_futures_expiry(prices)
    check_margin_calls_and_orders(prices)
    run_algo_strategies(prices)
    
    # Periodic cleanup
    game_state.cleanup_old_data()
    
    return prices

def trigger_random_event():
    """Optimized event triggering"""
    game_state = get_game_state()
    news_item = random.choice(PRE_BUILT_NEWS)
    headline = news_item['headline']
    
    target_symbol = None
    if "{symbol}" in headline:
        target_symbol = random.choice(NIFTY50_SYMBOLS)
        headline = headline.format(symbol=target_symbol.replace(".NS", ""))
    
    game_state.news_feed.appendleft(f"📢 {time.strftime('%H:%M:%S')} - {headline}")
    game_state.event_type = news_item['impact']
    game_state.event_target_symbol = target_symbol
    game_state.event_active = True
    game_state.event_end = time.time() + random.randint(30, 60)

def auto_square_off_positions(prices):
    """Optimized position squaring"""
    game_state = get_game_state()
    square_off_assets = NIFTY50_SYMBOLS + FUTURES_SYMBOLS + OPTION_SYMBOLS + LEVERAGED_ETFS
    
    for name, player in game_state.players.items():
        for symbol, qty in list(player['holdings'].items()):
            if symbol in square_off_assets:
                closing_price = prices.get(symbol, 0)
                value = closing_price * abs(qty)
                
                if qty > 0:  # Long
                    player['capital'] += value
                else:  # Short
                    player['capital'] -= value
                    
                del player['holdings'][symbol]

def handle_futures_expiry(prices):
    """Optimized futures handling"""
    game_state = get_game_state()
    if not game_state.futures_settled and game_state.futures_expiry_time > 0 and time.time() > game_state.futures_expiry_time:
        settlement_prices = {
            'NIFTY-FUT': prices.get(NIFTY_INDEX_SYMBOL, 0),
            'BANKNIFTY-FUT': prices.get(BANKNIFTY_INDEX_SYMBOL, 0)
        }
        
        for name, player in game_state.players.items():
            for symbol in FUTURES_SYMBOLS:
                if symbol in player['holdings']:
                    qty = player['holdings'][symbol]
                    settlement_price = settlement_prices.get(symbol, 0)
                    pnl = (settlement_price - prices.get(symbol, 0)) * qty
                    player['capital'] += pnl
                    del player['holdings'][symbol]
        
        game_state.futures_settled = True

def check_margin_calls_and_orders(prices):
    """Optimized margin checking"""
    game_state = get_game_state()
    
    for name, player in game_state.players.items():
        # Quick margin check
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        margin_needed = abs(holdings_value) * game_state.current_margin_requirement

        if total_value < margin_needed and player['holdings']:
            player['margin_calls'] += 1
            # Liquidate largest position
            largest_pos = max(player['holdings'].items(), 
                            key=lambda item: abs(item[1] * prices.get(item[0], 0)), 
                            default=(None, 0))
            if largest_pos[0]:
                symbol, qty = largest_pos
                action = "Sell" if qty > 0 else "Buy"
                execute_trade(name, player, action, symbol, abs(qty), prices, order_type="Margin Call")

def run_algo_strategies(prices):
    """Optimized algo strategies"""
    game_state = get_game_state()
    if len(game_state.price_history) < 2:
        return
        
    prev_prices = game_state.price_history[-2]
    
    for name, player in game_state.players.items():
        if player.get('algo', 'Off') == 'Off':
            continue
            
        # Sample a few symbols for performance
        sample_symbols = random.sample(NIFTY50_SYMBOLS + CRYPTO_SYMBOLS, min(5, len(NIFTY50_SYMBOLS + CRYPTO_SYMBOLS)))
        
        for symbol in sample_symbols:
            if player['algo'] == "Momentum Trader":
                price_change = (prices.get(symbol, 0) - prev_prices.get(symbol, prices.get(symbol, 0))) / prices.get(symbol, 1)
                if abs(price_change) > 0.001:
                    action = "Buy" if price_change > 0 else "Sell"
                    if execute_trade(name, player, action, symbol, 1, prices, is_algo=True):
                        break

def main():
    """Optimized main function with performance controls"""
    # Performance optimization - control rerun frequency
    if 'last_rerun' not in st.session_state:
        st.session_state.last_rerun = 0
        
    current_time = time.time()
    min_rerun_interval = 0.5  # Maximum 2 reruns per second
    
    if current_time - st.session_state.last_rerun < min_rerun_interval:
        time.sleep(min_rerun_interval - (current_time - st.session_state.last_rerun))
        
    st.session_state.last_rerun = current_time
    
    # Initialize game state
    game_state = get_game_state()
    if 'role' not in st.session_state:
        st.session_state.role = 'player'
    
    # Render UI
    render_sidebar()
    
    # Price simulation pipeline
    if not game_state.base_real_prices:
        game_state.base_real_prices = get_daily_base_prices()
    
    last_prices = game_state.prices if game_state.prices else game_state.base_real_prices
    
    # Efficient price simulation
    current_prices = simulate_tick_prices(last_prices)
    prices_with_derivatives = calculate_derived_prices(current_prices)
    final_prices = run_game_tick(prices_with_derivatives)
    
    game_state.prices = final_prices
    game_state.price_history.append(final_prices)
    
    # Render main interface
    render_main_interface(final_prices)
    
    # Smart rerun logic
    if should_rerun():
        st.rerun()
    else:
        time.sleep(1)  # Conservative refresh rate

if __name__ == "__main__":
    main()

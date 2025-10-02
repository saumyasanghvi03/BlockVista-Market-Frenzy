# BlockVista Market Frenzy - Complete High-Speed Trading Simulation
# Professional Inter-Collegiate Trading Championship

import streamlit as st
import pandas as pd
import yfinance as yf
import time
import random
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# --- Professional Page Configuration ---
st.set_page_config(
    layout="wide", 
    page_title="BlockVista Market Frenzy",
    page_icon="📈",
    initial_sidebar_state="expanded"
)

# --- Enhanced Professional Constants ---
GAME_NAME = "BLOCKVISTA MARKET FRENZY"
INITIAL_CAPITAL = 1000000
ADMIN_PASSWORD = "100370"
MAX_PLAYERS = 50

# Enhanced competitive qualification criteria
QUALIFICATION_CRITERIA = {
    1: {"min_gain_percent": 30, "description": "Achieve 30% portfolio growth in 8 minutes", "volatility_multiplier": 1.0},
    2: {"min_gain_percent": 60, "description": "Achieve 60% portfolio growth in 6 minutes", "volatility_multiplier": 1.5}, 
    3: {"min_gain_percent": 100, "description": "Achieve 100% portfolio growth in 4 minutes", "volatility_multiplier": 2.0}
}

# Enhanced asset universe with 6 categories (15 each)
STOCKS = [
    'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
    'KOTAKBANK.NS', 'HINDUNILVR.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'ITC.NS',
    'ASIANPAINT.NS', 'DMART.NS', 'BAJFINANCE.NS', 'WIPRO.NS', 'HCLTECH.NS'
]

FUTURES = [
    'NIFTY-FUT', 'BANKNIFTY-FUT', 'RELIANCE-FUT', 'TCS-FUT', 'HDFC-FUT',
    'INFY-FUT', 'ICICI-FUT', 'SBIN-FUT', 'HINDUNILVR-FUT', 'BAJFINANCE-FUT',
    'ITC-FUT', 'ASIANPAINT-FUT', 'DMART-FUT', 'WIPRO-FUT', 'HCLTECH-FUT'
]

OPTIONS = [
    'NIFTY-CALL-18000', 'NIFTY-PUT-18000', 'BANKNIFTY-CALL-45000', 'BANKNIFTY-PUT-45000',
    'RELIANCE-CALL-2500', 'RELIANCE-PUT-2500', 'TCS-CALL-3500', 'TCS-PUT-3500',
    'HDFC-CALL-1600', 'HDFC-PUT-1600', 'INFY-CALL-1800', 'INFY-PUT-1800',
    'ICICI-CALL-1000', 'ICICI-PUT-1000', 'SBIN-CALL-600', 'SBIN-PUT-600'
]

COMMODITIES = [
    'GOLD-1G', 'GOLD-10G', 'GOLD-100G', 'SILVER-1KG', 'SILVER-100G',
    'CRUDEOIL', 'NATURALGAS', 'COPPER', 'ALUMINIUM', 'ZINC',
    'NICKEL', 'LEAD', 'COTTON', 'SOYBEAN', 'SUGAR'
]

CRYPTO = [
    'BTC-INR', 'ETH-INR', 'SOL-INR', 'ADA-INR', 'DOT-INR',
    'MATIC-INR', 'AVAX-INR', 'LINK-INR', 'ATOM-INR', 'XRP-INR',
    'DOGE-INR', 'SHIB-INR', 'BNB-INR', 'TRX-INR', 'LTC-INR'
]

LEVERAGED_ETFS = [
    'NIFTY-2XL', 'NIFTY-2XS', 'BANKNIFTY-2XL', 'BANKNIFTY-2XS',
    'NIFTY-3XL', 'NIFTY-3XS', 'MIDCAP-2XL', 'MIDCAP-2XS',
    'SENSEX-2XL', 'SENSEX-2XS', 'SECTOR-2XL', 'SECTOR-2XS',
    'VOLATILITY-2X', 'GOLD-2XL', 'GOLD-2XS'
]

ALL_SYMBOLS = STOCKS + FUTURES + OPTIONS + COMMODITIES + CRYPTO + LEVERAGED_ETFS

# Enhanced Algo Trader Configurations with advanced strategies
ALGO_TRADERS = {
    "Momentum Master": {
        "description": "Chases trending stocks with high momentum using ML pattern recognition",
        "aggressiveness": 0.8,
        "risk_tolerance": 0.7,
        "speed": "High",
        "active": False,
        "capital": INITIAL_CAPITAL * 3,
        "strategy_type": "momentum",
        "max_position_size": 0.15
    },
    "Mean Reversion Pro": {
        "description": "Buys dips and sells rallies using statistical arbitrage models",
        "aggressiveness": 0.6,
        "risk_tolerance": 0.5,
        "speed": "Medium",
        "active": False,
        "capital": INITIAL_CAPITAL * 2,
        "strategy_type": "mean_reversion", 
        "max_position_size": 0.10
    },
    "Arbitrage Hunter": {
        "description": "Exploits price differences using high-frequency triangular arbitrage",
        "aggressiveness": 0.9,
        "risk_tolerance": 0.3,
        "speed": "Very High",
        "active": False,
        "capital": INITIAL_CAPITAL * 4,
        "strategy_type": "arbitrage",
        "max_position_size": 0.08
    },
    "Volatility Rider": {
        "description": "Thrives in high volatility using GARCH models and volatility forecasting",
        "aggressiveness": 0.7,
        "risk_tolerance": 0.8,
        "speed": "High",
        "active": False,
        "capital": INITIAL_CAPITAL * 3,
        "strategy_type": "volatility",
        "max_position_size": 0.12
    },
    "Market Maker Pro": {
        "description": "Provides liquidity and captures bid-ask spreads using market making algorithms",
        "aggressiveness": 0.5,
        "risk_tolerance": 0.4,
        "speed": "Ultra High",
        "active": False,
        "capital": INITIAL_CAPITAL * 5,
        "strategy_type": "market_making",
        "max_position_size": 0.06
    }
}

# Enhanced News Events for dynamic market impact
ENHANCED_NEWS_EVENTS = [
    {"headline": "RBI announces surprise 50bps rate cut", "impact": "Bull Rally", "duration": 120, "volatility_boost": 2.0},
    {"headline": "Major corporate fraud uncovered at large-cap company", "impact": "Flash Crash", "duration": 90, "volatility_boost": 3.0},
    {"headline": "Government announces $50B infrastructure package", "impact": "Sector Rotation", "duration": 150, "volatility_boost": 1.5},
    {"headline": "FIIs pour $1B into Indian equities", "impact": "Bull Rally", "duration": 100, "volatility_boost": 1.8},
    {"headline": "Global crude prices surge 20% overnight", "impact": "Commodity Boom", "duration": 120, "volatility_boost": 2.2},
    {"headline": "Cryptocurrency regulation framework announced", "impact": "Crypto Volatility", "duration": 110, "volatility_boost": 2.5},
    {"headline": "Circuit breaker triggered - market halt for 15 minutes", "impact": "Trading Halt", "duration": 15, "volatility_boost": 0.1},
    {"headline": "Major tech company announces breakthrough AI product", "impact": "Tech Rally", "duration": 130, "volatility_boost": 1.7},
    {"headline": "Banking sector faces liquidity crunch", "impact": "Banking Crisis", "duration": 140, "volatility_boost": 2.8},
    {"headline": "Retail trading frenzy grips markets", "impact": "Meme Stock Mania", "duration": 160, "volatility_boost": 3.2}
]

# --- Enhanced Game State Management ---
@st.cache_resource
def get_game_state():
    class ProfessionalGameState:
        def __init__(self):
            # Core game state
            self.players = {}
            self.game_status = "Stopped"
            self.game_start_time = 0
            self.break_start_time = 0
            self.break_duration = 30
            self.current_level = 1
            self.total_levels = 3
            self.level_durations = [8 * 60, 6 * 60, 4 * 60]  # 8, 6, 4 minutes
            
            # Enhanced market data
            self.prices = {}
            self.base_prices = {}
            self.price_history = []
            self.volume_data = {s: random.randint(1000, 10000) for s in ALL_SYMBOLS}
            self.volatility = {s: random.uniform(0.1, 0.4) for s in ALL_SYMBOLS}
            self.correlation_matrix = self._generate_correlation_matrix()
            
            # Enhanced player tracking
            self.transactions = {}
            self.market_sentiment = {s: 0 for s in ALL_SYMBOLS}
            self.qualified_players = set()
            self.eliminated_players = set()
            self.level_results = {}
            self.final_rankings = []
            self.player_analytics = {}  # Track player behavior analytics
            
            # Enhanced algo traders
            self.algo_traders = ALGO_TRADERS.copy()
            self.algo_performance = {}
            self.algo_activity_log = []
            
            # Enhanced market controls
            self.circuit_breaker_active = False
            self.circuit_breaker_end = 0
            self.admin_trading_halt = False
            self.active_news_events = []
            self.market_regime = "Normal"  # Normal, Volatile, Crisis, Bull, Bear
            self.regime_start_time = 0
            
            # Advanced trading features
            self.short_interest = {s: random.uniform(0.1, 0.3) for s in ALL_SYMBOLS}
            self.implied_volatility = {s: random.uniform(0.15, 0.35) for s in ALL_SYMBOLS}
            self.bid_ask_spreads = {s: random.uniform(0.001, 0.01) for s in ALL_SYMBOLS}
            
        def _generate_correlation_matrix(self):
            """Generate realistic correlation matrix between assets"""
            correlations = {}
            asset_groups = {
                'stocks': STOCKS,
                'futures': FUTURES,
                'options': OPTIONS,
                'commodities': COMMODITIES,
                'crypto': CRYPTO,
                'etfs': LEVERAGED_ETFS
            }
            
            for sym1 in ALL_SYMBOLS:
                correlations[sym1] = {}
                for sym2 in ALL_SYMBOLS:
                    if sym1 == sym2:
                        correlations[sym1][sym2] = 1.0
                    else:
                        # Same asset class has higher correlation
                        group1 = next((k for k, v in asset_groups.items() if sym1 in v), None)
                        group2 = next((k for k, v in asset_groups.items() if sym2 in v), None)
                        
                        if group1 == group2:
                            correlations[sym1][sym2] = random.uniform(0.6, 0.9)
                        else:
                            correlations[sym1][sym2] = random.uniform(-0.3, 0.5)
            
            return correlations
            
        def reset(self):
            self.__init__()
            
    return ProfessionalGameState()

# --- Enhanced Game State Updates ---
def update_game_state():
    """Handle level transitions, qualification checks, and market regime changes"""
    game_state = get_game_state()
    
    # Update market regime based on volatility and time
    current_time = time.time()
    if current_time - getattr(game_state, 'regime_start_time', current_time) > 120:  # Change regime every 2 minutes
        update_market_regime()
    
    # Trigger random news events
    if game_state.game_status == "Running" and random.random() < 0.02:  # 2% chance per update
        trigger_random_news_event()
    
    if game_state.game_status == "Running":
        current_time = time.time()
        level_duration = game_state.level_durations[game_state.current_level - 1]
        level_end_time = game_state.game_start_time + level_duration
        
        if current_time >= level_end_time:
            # Level completed - check qualifications
            check_level_qualifications()
            
            if game_state.current_level < game_state.total_levels:
                # Move to break
                game_state.game_status = "Break"
                game_state.break_start_time = current_time
            else:
                # Tournament finished
                game_state.game_status = "Finished"
                calculate_final_rankings()
    
    elif game_state.game_status == "Break":
        current_time = time.time()
        break_end_time = game_state.break_start_time + game_state.break_duration
        
        if current_time >= break_end_time:
            # Start next level
            game_state.current_level += 1
            game_state.game_status = "Running"
            game_state.game_start_time = current_time
            # Update volatility for new level
            update_level_volatility()

def update_market_regime():
    """Dynamically change market regime based on current conditions"""
    game_state = get_game_state()
    regimes = ["Normal", "Volatile", "Crisis", "Bull", "Bear"]
    weights = [0.4, 0.25, 0.1, 0.15, 0.1]  # Probability weights
    
    new_regime = random.choices(regimes, weights=weights)[0]
    game_state.market_regime = new_regime
    game_state.regime_start_time = time.time()
    
    # Log regime change
    if hasattr(game_state, 'algo_activity_log'):
        game_state.algo_activity_log.append({
            'timestamp': time.strftime("%H:%M:%S"),
            'type': 'REGIME_CHANGE',
            'message': f"Market regime changed to {new_regime}"
        })

def trigger_random_news_event():
    """Trigger random news events that affect market dynamics"""
    game_state = get_game_state()
    if random.random() < 0.3:  # 30% chance when called
        event = random.choice(ENHANCED_NEWS_EVENTS)
        game_state.active_news_events.append({
            **event,
            'start_time': time.time(),
            'end_time': time.time() + event['duration']
        })
        
        # Log news event
        game_state.algo_activity_log.append({
            'timestamp': time.strftime("%H:%M:%S"),
            'type': 'NEWS_EVENT',
            'message': f"News: {event['headline']}",
            'impact': event['impact']
        })

def update_level_volatility():
    """Update volatility based on current level"""
    game_state = get_game_state()
    current_level = game_state.current_level
    
    if current_level in QUALIFICATION_CRITERIA:
        volatility_multiplier = QUALIFICATION_CRITERIA[current_level]["volatility_multiplier"]
        for symbol in game_state.volatility:
            base_vol = game_state.volatility[symbol]
            game_state.volatility[symbol] = base_vol * volatility_multiplier

# --- Enhanced Price Simulation with Correlations ---
def simulate_high_speed_prices(last_prices):
    """Realistic high-frequency price simulation with correlations and news effects"""
    game_state = get_game_state()
    prices = last_prices.copy()
    
    # Get current market regime multiplier
    regime_multipliers = {
        "Normal": 1.0,
        "Volatile": 1.8,
        "Crisis": 3.0,
        "Bull": 1.2,
        "Bear": 1.5
    }
    regime_multiplier = regime_multipliers.get(game_state.market_regime, 1.0)
    
    # Apply news event effects
    news_volatility_boost = 1.0
    current_time = time.time()
    active_events = [e for e in getattr(game_state, 'active_news_events', []) 
                    if e['end_time'] > current_time]
    
    for event in active_events:
        news_volatility_boost *= event['volatility_boost']
    
    # Increase volatility based on level and regime
    level_volatility = 1.0 + (game_state.current_level - 1) * 0.5
    total_vol_multiplier = level_volatility * regime_multiplier * news_volatility_boost
    
    # Group symbols for correlated movement
    asset_groups = {
        'banking': ['HDFCBANK.NS', 'ICICIBANK.NS', 'KOTAKBANK.NS', 'SBIN.NS', 'AXISBANK.NS'],
        'tech': ['TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS'],
        'consumption': ['HINDUNILVR.NS', 'ITC.NS', 'ASIANPAINT.NS', 'DMART.NS']
    }
    
    # Calculate group momentum first
    group_momentum = {}
    for group_name, symbols in asset_groups.items():
        valid_symbols = [s for s in symbols if s in prices]
        if valid_symbols:
            group_returns = [(prices[s] - game_state.base_prices.get(s, prices[s])) / 
                           game_state.base_prices.get(s, prices[s]) for s in valid_symbols]
            group_momentum[group_name] = np.mean(group_returns) if group_returns else 0
    
    for symbol in prices:
        if symbol in game_state.base_prices:
            base_price = game_state.base_prices[symbol]
            volatility = game_state.volatility[symbol] * total_vol_multiplier
            
            # Base random walk
            momentum = random.uniform(-0.02, 0.02)
            noise = random.normalvariate(0, volatility * 0.01)
            sentiment_effect = game_state.market_sentiment.get(symbol, 0) * 0.001
            
            # Group momentum effect
            group_effect = 0
            for group_name, symbols in asset_groups.items():
                if symbol in symbols and group_name in group_momentum:
                    group_effect = group_momentum[group_name] * 0.1  # 10% of group momentum
            
            # Short squeeze effect
            short_squeeze_effect = 0
            if game_state.short_interest.get(symbol, 0) > 0.25 and random.random() < 0.05:
                short_squeeze_effect = random.uniform(0.02, 0.05)
            
            new_price = prices[symbol] * (1 + momentum + noise + sentiment_effect + 
                                        group_effect + short_squeeze_effect)
            
            # Ensure price doesn't go too far from base (circuit breaker effect)
            max_move = 0.15  # 15% maximum move per tick in volatile conditions
            price_change = abs(new_price - base_price) / base_price
            if price_change > max_move:
                correction = (base_price - new_price) / base_price * 0.3
                new_price *= (1 + correction)
            
            prices[symbol] = max(0.01, round(new_price, 2))
            
            # Update volume with regime effects
            volume_change = random.randint(-50, 100) * regime_multiplier
            game_state.volume_data[symbol] = max(100, 
                game_state.volume_data[symbol] + volume_change)
    
    # Clean up expired news events
    game_state.active_news_events = [e for e in game_state.active_news_events 
                                   if e['end_time'] > current_time]
    
    return prices

# --- Enhanced Algorithmic Traders ---
def run_algo_trader(trader_name, config, prices):
    """Execute enhanced algorithmic trading strategies"""
    game_state = get_game_state()
    
    if not config["active"]:
        return
    
    # Only trade during active game
    if game_state.game_status != "Running":
        return
    
    # Enhanced trader logic with regime awareness
    current_regime = game_state.market_regime
    
    # Adjust strategy based on market regime
    if current_regime == "Crisis" and config["strategy_type"] in ["momentum", "volatility"]:
        # Increase activity in crisis for certain strategies
        config["aggressiveness"] = min(1.0, config["aggressiveness"] * 1.3)
    elif current_regime == "Normal" and config["strategy_type"] == "arbitrage":
        # Arbitrage works better in normal markets
        config["aggressiveness"] = min(1.0, config["aggressiveness"] * 1.2)
    
    # Trader-specific enhanced logic
    if trader_name == "Momentum Master":
        enhanced_momentum_trader(trader_name, config, prices)
    elif trader_name == "Mean Reversion Pro":
        enhanced_mean_reversion_trader(trader_name, config, prices)
    elif trader_name == "Arbitrage Hunter":
        enhanced_arbitrage_trader(trader_name, config, prices)
    elif trader_name == "Volatility Rider":
        enhanced_volatility_trader(trader_name, config, prices)
    elif trader_name == "Market Maker Pro":
        market_maker_trader(trader_name, config, prices)

def enhanced_momentum_trader(trader_name, config, prices):
    """Enhanced momentum-based trading with regime awareness"""
    game_state = get_game_state()
    
    if len(game_state.price_history) < 10:
        return
    
    # Look across multiple timeframes
    for symbol in random.sample(ALL_SYMBOLS, 8):  # Check more symbols
        if symbol not in game_state.price_history[-1]:
            continue
            
        # Multi-timeframe momentum
        timeframes = [5, 10, 15]  # Lookback periods
        momentum_scores = []
        
        for tf in timeframes:
            if len(game_state.price_history) >= tf:
                recent_prices = [ph.get(symbol, prices[symbol]) for ph in game_state.price_history[-tf:]]
                if len(recent_prices) == tf:
                    returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] for i in range(1, tf)]
                    momentum_score = sum(returns) / len(returns)
                    momentum_scores.append(momentum_score)
        
        if momentum_scores:
            avg_momentum = sum(momentum_scores) / len(momentum_scores)
            
            # Volume confirmation
            current_volume = game_state.volume_data.get(symbol, 1000)
            avg_volume = np.mean(list(game_state.volume_data.values()))
            volume_boost = min(2.0, current_volume / max(avg_volume, 1000))
            
            if avg_momentum > 0.005 * volume_boost:  # Strong positive momentum with volume
                if random.random() < config["aggressiveness"]:
                    max_qty = int((config["capital"] * config["max_position_size"]) / prices[symbol])
                    qty = random.randint(5, min(50, max_qty))
                    execute_algo_trade(trader_name, "Buy", symbol, qty, prices)

def enhanced_mean_reversion_trader(trader_name, config, prices):
    """Enhanced mean reversion with volatility-adjusted bands"""
    game_state = get_game_state()
    
    for symbol in random.sample(ALL_SYMBOLS, 6):
        if symbol not in game_state.base_prices:
            continue
            
        current_price = prices[symbol]
        base_price = game_state.base_prices[symbol]
        volatility = game_state.volatility[symbol]
        
        # Dynamic bands based on volatility
        deviation_bands = 0.05 + (volatility * 0.5)  # Wider bands in high vol
        deviation = (current_price - base_price) / base_price
        
        if abs(deviation) > deviation_bands:
            # Confidence based on how far from mean
            confidence = min(1.0, abs(deviation) / (deviation_bands * 2))
            adjusted_aggressiveness = config["aggressiveness"] * confidence
            
            if deviation > 0 and random.random() < adjusted_aggressiveness:
                # Overbought - sell
                max_qty = int((config["capital"] * config["max_position_size"]) / prices[symbol])
                qty = random.randint(3, min(25, max_qty))
                execute_algo_trade(trader_name, "Sell", symbol, qty, prices)
            elif deviation < 0 and random.random() < adjusted_aggressiveness:
                # Oversold - buy
                max_qty = int((config["capital"] * config["max_position_size"]) / prices[symbol])
                qty = random.randint(3, min(25, max_qty))
                execute_algo_trade(trader_name, "Buy", symbol, qty, prices)

def enhanced_arbitrage_trader(trader_name, config, prices):
    """Enhanced arbitrage with triangular opportunities"""
    game_state = get_game_state()
    
    # Traditional pairs
    pairs = [
        ('RELIANCE.NS', 'RELIANCE-FUT'),
        ('TCS.NS', 'TCS-FUT'),
        ('HDFCBANK.NS', 'HDFC-FUT'),
        ('NIFTY-2XL', 'NIFTY-2XS')
    ]
    
    for asset1, asset2 in pairs:
        if asset1 in prices and asset2 in prices:
            price1 = prices[asset1]
            price2 = prices[asset2]
            spread = abs(price1 - price2) / min(price1, price2)
            
            if spread > 0.015 and random.random() < config["aggressiveness"]:  # 1.5% spread
                if price1 > price2:
                    execute_algo_trade(trader_name, "Buy", asset2, random.randint(8, 30), prices)
                    execute_algo_trade(trader_name, "Sell", asset1, random.randint(8, 30), prices)
                else:
                    execute_algo_trade(trader_name, "Buy", asset1, random.randint(8, 30), prices)
                    execute_algo_trade(trader_name, "Sell", asset2, random.randint(8, 30), prices)
    
    # Triangular arbitrage opportunities (simplified)
    if random.random() < 0.1:
        crypto_triplet = ['BTC-INR', 'ETH-INR', 'SOL-INR']
        if all(asset in prices for asset in crypto_triplet):
            # Simplified triangular arbitrage logic
            execute_algo_trade(trader_name, "Buy", crypto_triplet[0], random.randint(1, 5), prices)
            execute_algo_trade(trader_name, "Sell", crypto_triplet[1], random.randint(10, 20), prices)

def enhanced_volatility_trader(trader_name, config, prices):
    """Enhanced volatility trading with regime adaptation"""
    game_state = get_game_state()
    
    # Find assets with abnormal volatility
    high_vol_assets = []
    for symbol in ALL_SYMBOLS:
        current_vol = game_state.volatility[symbol]
        avg_vol = np.mean(list(game_state.volatility.values()))
        
        if current_vol > avg_vol * 1.5:  # 50% more volatile than average
            high_vol_assets.append(symbol)
    
    for symbol in random.sample(high_vol_assets, min(4, len(high_vol_assets))):
        if random.random() < config["aggressiveness"]:
            # In high volatility, trade both directions
            action = random.choice(["Buy", "Sell"])
            max_qty = int((config["capital"] * config["max_position_size"]) / prices[symbol])
            qty = random.randint(10, min(40, max_qty))
            execute_algo_trade(trader_name, action, symbol, qty, prices)

def market_maker_trader(trader_name, config, prices):
    """Market making strategy providing liquidity"""
    game_state = get_game_state()
    
    # Focus on high-volume assets
    high_volume_assets = sorted(game_state.volume_data.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for symbol, volume in high_volume_assets:
        if random.random() < config["aggressiveness"] * 0.5:  # Less frequent but larger trades
            # Market making: place both buy and sell orders around current price
            spread = game_state.bid_ask_spreads.get(symbol, 0.005)
            current_price = prices[symbol]
            
            # Place limit orders (simulated as market orders for simplicity)
            buy_price = current_price * (1 - spread/2)
            sell_price = current_price * (1 + spread/2)
            
            if random.random() < 0.5:
                # Execute buy at bid
                max_qty = int((config["capital"] * config["max_position_size"]) / buy_price)
                qty = random.randint(5, min(20, max_qty))
                execute_algo_trade(trader_name, "Buy", symbol, qty, prices)
            else:
                # Execute sell at ask
                max_qty = int((config["capital"] * config["max_position_size"]) / sell_price)
                qty = random.randint(5, min(20, max_qty))
                execute_algo_trade(trader_name, "Sell", symbol, qty, prices)

def execute_algo_trade(trader_name, action, symbol, qty, prices):
    """Enhanced trade execution for algorithmic traders with analytics"""
    game_state = get_game_state()
    
    # Use a dedicated player for algo trader
    algo_player_name = f"ALGO_{trader_name}"
    
    if algo_player_name not in game_state.players:
        # Initialize algo trader as player
        game_state.players[algo_player_name] = {
            "name": algo_player_name,
            "mode": "Algorithmic",
            "capital": ALGO_TRADERS[trader_name]["capital"],
            "holdings": {},
            "value_history": [ALGO_TRADERS[trader_name]["capital"]],
            "trade_timestamps": [],
            "strategy": ALGO_TRADERS[trader_name]["strategy_type"]
        }
        game_state.transactions[algo_player_name] = []
        game_state.algo_performance[algo_player_name] = {
            "trades": 0,
            "winning_trades": 0,
            "total_pnl": 0
        }
    
    player = game_state.players[algo_player_name]
    
    # Enhanced trade execution with analytics
    current_price = prices.get(symbol, 0)
    if current_price <= 0:
        return False
    
    cost = current_price * qty
    
    if action == "Buy" and player['capital'] >= cost:
        player['capital'] -= cost
        player['holdings'][symbol] = player['holdings'].get(symbol, 0) + qty
        log_enhanced_transaction(algo_player_name, f"ALGO {action}", symbol, qty, current_price, cost, trader_name)
        
        # Update algo performance
        game_state.algo_performance[algo_player_name]["trades"] += 1
        return True
        
    elif action == "Sell":
        current_qty = player['holdings'].get(symbol, 0)
        if current_qty >= qty:
            player['capital'] += cost
            player['holdings'][symbol] -= qty
            if player['holdings'][symbol] == 0:
                del player['holdings'][symbol]
            log_enhanced_transaction(algo_player_name, f"ALGO {action}", symbol, qty, current_price, cost, trader_name)
            
            # Update algo performance
            game_state.algo_performance[algo_player_name]["trades"] += 1
            return True
    
    return False

def log_enhanced_transaction(player_name, action, symbol, qty, price, total, algo_name=None):
    """Enhanced transaction logging with analytics"""
    game_state = get_game_state()
    timestamp = time.strftime("%H:%M:%S")
    
    transaction_data = [timestamp, action, symbol, qty, price, total]
    if algo_name:
        transaction_data.append(algo_name)
    
    game_state.transactions.setdefault(player_name, []).append(transaction_data)
    
    # Log algo activity
    if algo_name and hasattr(game_state, 'algo_activity_log'):
        game_state.algo_activity_log.append({
            'timestamp': timestamp,
            'algo': algo_name,
            'action': action,
            'symbol': symbol,
            'qty': qty,
            'price': price
        })

# --- Enhanced Trading Execution ---
def execute_trade(player_name, player, action, symbol, qty, prices):
    """Enhanced trade execution for human players with advanced features"""
    game_state = get_game_state()
    
    # Check if player is eliminated
    if player_name in game_state.eliminated_players:
        st.error("❌ You have been eliminated from the current level!")
        return False
    
    # Check trading halt
    if game_state.admin_trading_halt or game_state.circuit_breaker_active:
        st.error("🚫 Trading is currently halted!")
        return False
    
    current_price = prices.get(symbol, 0)
    if current_price <= 0:
        st.error("Invalid price for selected asset")
        return False
    
    # Apply bid-ask spread
    spread = game_state.bid_ask_spreads.get(symbol, 0.005)
    if action == "Buy":
        execution_price = current_price * (1 + spread/2)
    else:  # Sell
        execution_price = current_price * (1 - spread/2)
    
    cost = execution_price * qty
    trade_executed = False
    
    if action == "Buy" and player['capital'] >= cost:
        player['capital'] -= cost
        player['holdings'][symbol] = player['holdings'].get(symbol, 0) + qty
        trade_executed = True
    elif action == "Sell":
        current_qty = player['holdings'].get(symbol, 0)
        if current_qty >= qty:
            player['capital'] += cost
            player['holdings'][symbol] -= qty
            if player['holdings'][symbol] == 0:
                del player['holdings'][symbol]
            trade_executed = True
    
    if trade_executed:
        log_enhanced_transaction(player_name, action, symbol, qty, execution_price, cost)
        # Enhanced market sentiment update
        sentiment_impact = (qty / 1000) * (1 if action == "Buy" else -1)
        game_state.market_sentiment[symbol] = game_state.market_sentiment.get(symbol, 0) + sentiment_impact
        
        # Update volume
        game_state.volume_data[symbol] = game_state.volume_data.get(symbol, 1000) + qty
        
        player['trade_timestamps'].append(time.time())
        return True
    else:
        st.error("Trade failed: Insufficient capital or holdings")
        return False

# --- Enhanced UI Components ---
def render_vip_sidebar():
    """Enhanced VIP/admin sidebar with advanced controls"""
    game_state = get_game_state()
    
    with st.sidebar:
        st.title("🎮 Tournament Control")
        
        # Enhanced player registration
        if 'player' not in st.query_params:
            st.subheader("👤 Player Registration")
            player_name = st.text_input("Enter Your Name")
            trading_mode = st.selectbox("Trading Style", 
                                      ["Aggressive", "Balanced", "Conservative", "HFT", "Quant"])
            
            if st.button("Join Tournament", use_container_width=True):
                if player_name and player_name.strip() and player_name not in game_state.players:
                    # Enhanced player initialization based on trading style
                    style_multipliers = {
                        "Aggressive": 1.0,
                        "Balanced": 1.0,
                        "Conservative": 1.0,
                        "HFT": 1.0,
                        "Quant": 1.0
                    }
                    
                    game_state.players[player_name] = {
                        "name": player_name,
                        "mode": trading_mode,
                        "capital": INITIAL_CAPITAL,
                        "holdings": {},
                        "value_history": [INITIAL_CAPITAL],
                        "trade_timestamps": [],
                        "trading_style": trading_mode,
                        "risk_score": 0.5
                    }
                    game_state.transactions[player_name] = []
                    game_state.player_analytics[player_name] = {
                        "total_trades": 0,
                        "avg_trade_size": 0,
                        "preferred_assets": [],
                        "trade_frequency": 0
                    }
                    st.query_params["player"] = player_name
                    st.rerun()
        else:
            player_name = st.query_params.get("player")
            st.success(f"Welcome, **{player_name}**!")
            if st.button("Leave Game", use_container_width=True):
                st.query_params.clear()
                st.rerun()
        
        st.markdown("---")
        
        # Enhanced admin controls
        st.subheader("🔐 Admin Panel")
        admin_pass = st.text_input("Admin Password", type="password")
        
        if admin_pass == ADMIN_PASSWORD:
            st.session_state.admin_access = True
            st.success("✅ Admin Access Granted")
        
        if st.session_state.get('admin_access'):
            # Enhanced game controls
            st.subheader("⚙️ Enhanced Game Controls")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Start Tournament", use_container_width=True):
                    if game_state.players:
                        game_state.game_status = "Running"
                        game_state.game_start_time = time.time()
                        game_state.current_level = 1
                        initialize_base_prices()
                        st.rerun()
            with col2:
                if st.button("⏸️ Stop Game", use_container_width=True):
                    game_state.game_status = "Stopped"
                    st.rerun()
            
            if st.button("🔄 Reset Game", use_container_width=True):
                game_state.reset()
                st.rerun()
            
            # Enhanced market controls
            st.subheader("📊 Market Controls")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚫 Circuit Breaker", use_container_width=True):
                    game_state.circuit_breaker_active = True
                    game_state.circuit_breaker_end = time.time() + 60
                if st.button("📈 Bull Market", use_container_width=True):
                    game_state.market_regime = "Bull"
                    game_state.regime_start_time = time.time()
            with col2:
                if st.button("✅ Resume Trading", use_container_width=True):
                    game_state.admin_trading_halt = False
                    game_state.circuit_breaker_active = False
                if st.button("📉 Bear Market", use_container_width=True):
                    game_state.market_regime = "Bear"
                    game_state.regime_start_time = time.time()
            
            # Enhanced algo trader controls
            st.subheader("🤖 Enhanced Algo Traders")
            for algo_name, algo_config in game_state.algo_traders.items():
                col1, col2 = st.columns([3, 1])
                with col1:
                    status = "✅ Active" if algo_config['active'] else "❌ Inactive"
                    st.write(f"{status} **{algo_name}**")
                    st.caption(algo_config['description'])
                with col2:
                    if st.button(f"{'🔴' if algo_config['active'] else '🟢'} {algo_name}", 
                               key=f"algo_{algo_name}"):
                        algo_config['active'] = not algo_config['active']
                        st.rerun()
            
            # Market event triggers
            st.subheader("🎯 Market Events")
            event_options = [event['headline'] for event in ENHANCED_NEWS_EVENTS]
            selected_event = st.selectbox("Trigger Event", event_options)
            if st.button("Trigger Selected Event"):
                event = next(e for e in ENHANCED_NEWS_EVENTS if e['headline'] == selected_event)
                game_state.active_news_events.append({
                    **event,
                    'start_time': time.time(),
                    'end_time': time.time() + event['duration']
                })
                st.success(f"Event triggered: {event['headline']}")

def render_trading_interface(prices):
    """Enhanced main trading interface"""
    game_state = get_game_state()
    player_name = st.query_params.get("player")
    
    if not player_name or player_name not in game_state.players:
        st.info("👆 Please register from the sidebar to start trading")
        return
    
    player = game_state.players[player_name]
    
    # Enhanced portfolio overview
    holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
    total_value = player['capital'] + holdings_value
    gain_percent = ((total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    # Calculate additional metrics
    trade_count = len(player.get('trade_timestamps', []))
    avg_trade_size = total_value / max(1, trade_count)
    
    st.subheader("💰 Enhanced Portfolio Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cash", f"₹{player['capital']:,.0f}")
        st.metric("Trades", trade_count)
    with col2:
        st.metric("Holdings", f"₹{holdings_value:,.0f}")
        st.metric("Avg Trade", f"₹{avg_trade_size:,.0f}")
    with col3:
        st.metric("Total", f"₹{total_value:,.0f}")
        st.metric("Market Regime", game_state.market_regime)
    with col4:
        st.metric("P&L", f"{gain_percent:+.1f}%")
        st.metric("Level", game_state.current_level)
    
    # Enhanced trading panel
    st.subheader("⚡ Enhanced Trade Execution")
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        asset_category = st.selectbox("Asset Category", 
                                    ["Stocks", "Futures", "Options", "Commodities", "Crypto", "Leveraged ETFs"])
        symbol_map = {
            "Stocks": STOCKS,
            "Futures": FUTURES, 
            "Options": OPTIONS,
            "Commodities": COMMODITIES,
            "Crypto": CRYPTO,
            "Leveraged ETFs": LEVERAGED_ETFS
        }
        symbol = st.selectbox("Select Asset", symbol_map[asset_category])
        current_price = prices.get(symbol, 0)
        
        # Enhanced asset information
        vol = game_state.volatility.get(symbol, 0)
        volume = game_state.volume_data.get(symbol, 0)
        sentiment = game_state.market_sentiment.get(symbol, 0)
        
        st.info(f"Price: ₹{current_price:,.2f} | Vol: {vol:.1%} | Volume: {volume:,} | Sentiment: {sentiment:+.2f}")
    
    with col2:
        action = st.radio("Action", ["Buy", "Sell"])
        qty = st.number_input("Quantity", min_value=1, max_value=1000, value=10)
        
        # Quick quantity buttons
        col_q1, col_q2, col_q3 = st.columns(3)
        with col_q1:
            if st.button("10", use_container_width=True):
                qty = 10
        with col_q2:
            if st.button("50", use_container_width=True):
                qty = 50
        with col_q3:
            if st.button("100", use_container_width=True):
                qty = 100
    
    with col3:
        if st.button("🚀 Execute Trade", type="primary", use_container_width=True):
            if execute_trade(player_name, player, action, symbol, qty, prices):
                st.success("✅ Trade executed!")
        
        # Advanced order types (placeholder for future enhancement)
        if st.button("📊 Advanced Orders", use_container_width=True):
            st.info("Limit orders, stop-loss, and bracket orders coming soon!")
    
    # Enhanced holdings display
    st.subheader("📦 Enhanced Portfolio Analysis")
    if player['holdings']:
        holdings_data = []
        for symbol, qty in player['holdings'].items():
            if qty > 0:
                current_price = prices.get(symbol, 0)
                value = current_price * qty
                cost_basis = sum([t[5] for t in game_state.transactions.get(player_name, []) 
                                if t[2] == symbol and t[1] == "Buy"])
                unrealized_pnl = value - cost_basis
                
                holdings_data.append({
                    'Asset': symbol,
                    'Quantity': qty,
                    'Current Price': f"₹{current_price:,.2f}",
                    'Value': f"₹{value:,.0f}",
                    'Unrealized P&L': f"₹{unrealized_pnl:,.0f}"
                })
        
        if holdings_data:
            st.dataframe(pd.DataFrame(holdings_data), use_container_width=True)
            
            # Portfolio allocation chart
            holdings_df = pd.DataFrame(holdings_data)
            fig = px.pie(holdings_df, values='Value', names='Asset', 
                        title="Portfolio Allocation", 
                        hover_data=['Quantity'])
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No holdings yet. Start trading!")

def render_market_overview(prices):
    """Enhanced market data overview"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Enhanced Market Overview")
    
    game_state = get_game_state()
    
    # Enhanced market status
    remaining_time, time_type = get_remaining_time()
    if game_state.game_status == "Running":
        minutes, seconds = divmod(int(remaining_time), 60)
        st.sidebar.write(f"⏰ Level {game_state.current_level}: {minutes:02d}:{seconds:02d}")
        st.sidebar.write(f"🎯 Regime: {game_state.market_regime}")
    
    # Enhanced qualification status
    if game_state.current_level in QUALIFICATION_CRITERIA:
        criteria = QUALIFICATION_CRITERIA[game_state.current_level]
        st.sidebar.write(f"🎯 Target: +{criteria['min_gain_percent']}%")
        st.sidebar.write(f"⚡ Volatility: {criteria['volatility_multiplier']}x")
    
    # Enhanced market data with sentiment
    st.sidebar.subheader("💹 Enhanced Live Prices")
    sample_symbols = random.sample(ALL_SYMBOLS, 6)
    for symbol in sample_symbols:
        price = prices.get(symbol, 0)
        change = random.uniform(-3, 3)  # Wider range for enhanced volatility
        sentiment = game_state.market_sentiment.get(symbol, 0)
        sentiment_icon = "📈" if sentiment > 0 else "📉" if sentiment < 0 else "➡️"
        
        st.sidebar.write(f"{symbol}: ₹{price:,.0f} ({change:+.1f}%) {sentiment_icon}")

def render_leaderboard(prices):
    """Enhanced live leaderboard"""
    game_state = get_game_state()
    
    st.subheader("🏆 Enhanced Live Leaderboard")
    
    leaderboard_data = []
    for player_name, player in game_state.players.items():
        if "ALGO_" in player_name:  # Mark algo traders separately
            holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
            total_value = player['capital'] + holdings_value
            gain_percent = ((total_value - ALGO_TRADERS[player_name.replace("ALGO_", "")]["capital"]) / 
                          ALGO_TRADERS[player_name.replace("ALGO_", "")]["capital"]) * 100
            
            leaderboard_data.append({
                'Player': f"🤖 {player_name}",
                'Portfolio': f"₹{total_value:,.0f}",
                'Gain %': f"{gain_percent:+.1f}%",
                'Status': "Algorithm",
                'Strategy': player.get('strategy', 'N/A')
            })
            continue
            
        holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        gain_percent = ((total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        
        status = "✅ Qualified" if player_name in game_state.qualified_players else \
                 "❌ Eliminated" if player_name in game_state.eliminated_players else "🎯 Active"
        
        leaderboard_data.append({
            'Player': player_name,
            'Portfolio': f"₹{total_value:,.0f}",
            'Gain %': f"{gain_percent:+.1f}%",
            'Status': status,
            'Strategy': player.get('trading_style', 'Trader')
        })
    
    if leaderboard_data:
        df = pd.DataFrame(leaderboard_data)
        # Sort by portfolio value, handling both human and algo traders
        df['SortValue'] = df['Portfolio'].str.replace('₹', '').str.replace(',', '').astype(float)
        df = df.sort_values('SortValue', ascending=False).drop('SortValue', axis=1)
        st.dataframe(df, use_container_width=True)
        
        # Show algo performance if available
        if game_state.algo_performance:
            st.subheader("🤖 Algorithmic Trader Performance")
            algo_data = []
            for algo_name, perf in game_state.algo_performance.items():
                if perf['trades'] > 0:
                    win_rate = (perf['winning_trades'] / perf['trades']) * 100
                    algo_data.append({
                        'Algorithm': algo_name.replace('ALGO_', ''),
                        'Trades': perf['trades'],
                        'Win Rate': f"{win_rate:.1f}%",
                        'Total P&L': f"₹{perf['total_pnl']:,.0f}"
                    })
            
            if algo_data:
                st.dataframe(pd.DataFrame(algo_data), use_container_width=True)

def render_qualification_status(prices):
    """Enhanced qualification progress"""
    game_state = get_game_state()
    player_name = st.query_params.get("player")
    
    if not player_name or player_name not in game_state.players:
        return
    
    if player_name in game_state.eliminated_players:
        st.error("## ❌ Eliminated")
        st.info("You didn't meet the qualification criteria for this level. Better luck next time!")
        
        # Show what went wrong
        player = game_state.players[player_name]
        holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        gain_percent = ((total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        
        if game_state.current_level in QUALIFICATION_CRITERIA:
            criteria = QUALIFICATION_CRITERIA[game_state.current_level]
            needed = max(0, criteria['min_gain_percent'] - gain_percent)
            st.write(f"You achieved {gain_percent:.1f}% but needed {criteria['min_gain_percent']}%")
        return
    
    if game_state.current_level in QUALIFICATION_CRITERIA:
        criteria = QUALIFICATION_CRITERIA[game_state.current_level]
        min_gain = criteria["min_gain_percent"]
        
        player = game_state.players[player_name]
        holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        gain_percent = ((total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        
        progress = min(100, (gain_percent / min_gain) * 100)
        
        st.subheader("🎯 Enhanced Qualification Progress")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Target:** +{min_gain}% gain")
            st.write(f"**Your Gain:** {gain_percent:+.1f}%")
            st.write(f"**Level Volatility:** {criteria['volatility_multiplier']}x")
        
        with col2:
            st.write(f"**Time Remaining:** {get_remaining_time()[0]:.0f}s")
            st.write(f"**Market Regime:** {game_state.market_regime}")
            st.write(f"**Active Events:** {len(game_state.active_news_events)}")
        
        st.progress(progress / 100)
        
        if gain_percent >= min_gain:
            st.success("## ✅ You are QUALIFIED for the next level!")
            st.balloons()
        else:
            needed = max(0, min_gain - gain_percent)
            st.warning(f"💰 Need +{needed:.1f}% more to qualify")
            
            # Trading suggestions
            if gain_percent < 0:
                st.info("💡 Tip: Consider reducing risk and focusing on preserving capital")
            elif gain_percent < min_gain/2:
                st.info("💡 Tip: Look for momentum opportunities in high-volume assets")
            else:
                st.info("💡 Tip: You're close! Consider taking calculated risks to reach the target")

# --- Enhanced Helper Functions ---
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

def initialize_base_prices():
    """Initialize realistic base prices for all symbols"""
    game_state = get_game_state()
    
    if not game_state.base_prices:
        # Enhanced realistic price ranges for different asset classes
        price_ranges = {
            'STOCKS': (100, 5000),
            'FUTURES': (5000, 30000),
            'OPTIONS': (10, 500),
            'COMMODITIES': (50, 10000),
            'CRYPTO': (10, 500000),
            'LEVERAGED_ETFS': (100, 2000)
        }
        
        for symbol in ALL_SYMBOLS:
            if symbol in STOCKS:
                price_range = price_ranges['STOCKS']
            elif symbol in FUTURES:
                price_range = price_ranges['FUTURES']
            elif symbol in OPTIONS:
                price_range = price_ranges['OPTIONS']
            elif symbol in COMMODITIES:
                price_range = price_ranges['COMMODITIES']
            elif symbol in CRYPTO:
                price_range = price_ranges['CRYPTO']
            else:  # LEVERAGED_ETFS
                price_range = price_ranges['LEVERAGED_ETFS']
            
            game_state.base_prices[symbol] = random.uniform(price_range[0], price_range[1])
            game_state.prices[symbol] = game_state.base_prices[symbol]

# --- Enhanced Main Application ---
def main():
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.last_refresh = time.time()
    
    game_state = get_game_state()
    
    # Update enhanced game state
    update_game_state()
    
    # Initialize prices if not done
    if not game_state.base_prices:
        initialize_base_prices()
    
    # Run enhanced algo traders
    for algo_name, algo_config in game_state.algo_traders.items():
        run_algo_trader(algo_name, algo_config, game_state.prices)
    
    # Simulate enhanced price changes
    if game_state.game_status == "Running":
        game_state.prices = simulate_high_speed_prices(game_state.prices)
        game_state.price_history.append(game_state.prices.copy())
        if len(game_state.price_history) > 100:
            game_state.price_history.pop(0)
    
    # Render Enhanced UI
    render_vip_sidebar()
    
    # Main content
    st.title("🎯 BlockVista Market Frenzy")
    st.subheader("Professional Trading Championship - Enhanced Edition")
    
    # Enhanced game status display
    if game_state.game_status == "Finished" and game_state.final_rankings:
        st.balloons()
        st.success("## 🏆 Tournament Complete!")
        
        # Enhanced winner announcement
        winner = game_state.final_rankings[0]
        st.success(f"**Champion:** {winner['name']} with ₹{winner['portfolio_value']:,.0f} (+{winner['gain_percent']:.1f}%)")
        
        # Show top 3 performers
        st.subheader("🎖️ Top Performers")
        for i, player in enumerate(game_state.final_rankings[:3]):
            medal = ["🥇", "🥈", "🥉"][i]
            st.write(f"{medal} **{player['name']}**: ₹{player['portfolio_value']:,.0f} (+{player['gain_percent']:.1f}%)")
    
    # Enhanced main columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_trading_interface(game_state.prices)
        render_leaderboard(game_state.prices)
    
    with col2:
        render_qualification_status(game_state.prices)
        
        # Enhanced algo trader status
        st.subheader("🤖 Enhanced Algorithmic Traders")
        for algo_name, algo_config in game_state.algo_traders.items():
            status = "🟢 Active" if algo_config["active"] else "🔴 Inactive"
            performance = game_state.algo_performance.get(f"ALGO_{algo_name}", {})
            trades = performance.get('trades', 0)
            
            st.write(f"{status} **{algo_name}**")
            st.caption(f"{algo_config['description']} | Trades: {trades}")
        
        # Market regime and events
        st.subheader("🌡️ Market Conditions")
        st.write(f"**Current Regime:** {game_state.market_regime}")
        st.write(f"**Active Events:** {len(game_state.active_news_events)}")
        
        if game_state.active_news_events:
            st.write("**Active News:**")
            for event in game_state.active_news_events[:2]:  # Show latest 2
                time_left = event['end_time'] - time.time()
                if time_left > 0:
                    st.caption(f"• {event['headline']} ({int(time_left)}s)")
    
    # Enhanced auto-refresh with dynamic timing
    current_time = time.time()
    base_refresh_interval = 1.0  # 1 second base refresh
    
    # Adjust refresh rate based on game status and level
    if game_state.game_status == "Running":
        level_speed_boost = 1.0 - (game_state.current_level - 1) * 0.2  # Faster at higher levels
        refresh_interval = max(0.3, base_refresh_interval * level_speed_boost)
    else:
        refresh_interval = 2.0  # Slower when not running
    
    if current_time - st.session_state.last_refresh > refresh_interval:
        st.session_state.last_refresh = current_time
        st.rerun()

if __name__ == "__main__":
    main()

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

# --- Professional Constants ---
GAME_NAME = "BLOCKVISTA MARKET FRENZY"
INITIAL_CAPITAL = 1000000
ADMIN_PASSWORD = "100370"
MAX_PLAYERS = 50

# Competitive qualification criteria - HARD from beginning
QUALIFICATION_CRITERIA = {
    1: {"min_gain_percent": 30, "description": "Achieve 30% portfolio growth in 8 minutes"},
    2: {"min_gain_percent": 60, "description": "Achieve 60% portfolio growth in 6 minutes"}, 
    3: {"min_gain_percent": 100, "description": "Achieve 100% portfolio growth in 4 minutes"}
}

# Realistic asset universe with 6 categories (15 each)
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

# Algo Trader Configurations
ALGO_TRADERS = {
    "Momentum Master": {
        "description": "Chases trending stocks with high momentum",
        "aggressiveness": 0.8,
        "risk_tolerance": 0.7,
        "speed": "High",
        "active": False,
        "capital": INITIAL_CAPITAL * 3
    },
    "Mean Reversion Pro": {
        "description": "Buys dips and sells rallies in range-bound markets",
        "aggressiveness": 0.6,
        "risk_tolerance": 0.5,
        "speed": "Medium",
        "active": False,
        "capital": INITIAL_CAPITAL * 2
    },
    "Arbitrage Hunter": {
        "description": "Exploits price differences between related assets",
        "aggressiveness": 0.9,
        "risk_tolerance": 0.3,
        "speed": "Very High",
        "active": False,
        "capital": INITIAL_CAPITAL * 4
    },
    "Volatility Rider": {
        "description": "Thrives in high volatility environments",
        "aggressiveness": 0.7,
        "risk_tolerance": 0.8,
        "speed": "High",
        "active": False,
        "capital": INITIAL_CAPITAL * 3
    }
}

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
            
            # Market data
            self.prices = {}
            self.base_prices = {}
            self.price_history = []
            self.volume_data = {s: random.randint(1000, 10000) for s in ALL_SYMBOLS}
            self.volatility = {s: random.uniform(0.1, 0.4) for s in ALL_SYMBOLS}
            
            # Player tracking
            self.transactions = {}
            self.market_sentiment = {s: 0 for s in ALL_SYMBOLS}
            self.qualified_players = set()
            self.eliminated_players = set()
            self.level_results = {}
            self.final_rankings = []
            
            # Algo traders
            self.algo_traders = ALGO_TRADERS.copy()
            self.algo_performance = {}
            
            # Market controls
            self.circuit_breaker_active = False
            self.circuit_breaker_end = 0
            self.admin_trading_halt = False
            
        def reset(self):
            self.__init__()
            
    return ProfessionalGameState()

# --- MISSING FUNCTION: update_game_state ---
def update_game_state():
    """Handle level transitions and qualification checks"""
    game_state = get_game_state()
    
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

# --- MISSING FUNCTION: check_level_qualifications ---
def check_level_qualifications():
    """Check player qualifications based on percentage gains"""
    game_state = get_game_state()
    current_level = game_state.current_level
    
    if current_level not in QUALIFICATION_CRITERIA:
        return
    
    criteria = QUALIFICATION_CRITERIA[current_level]
    min_gain_percent = criteria["min_gain_percent"]
    
    qualified_count = 0
    eliminated_count = 0
    
    for player_name, player in game_state.players.items():
        if "ALGO_" in player_name:  # Skip algo traders for qualification
            continue
            
        if player_name in game_state.eliminated_players:
            continue
        
        starting_capital = INITIAL_CAPITAL
        holdings_value = sum(game_state.prices.get(s, 0) * q for s, q in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        gain_percent = ((total_value - starting_capital) / starting_capital) * 100
        
        if gain_percent >= min_gain_percent:
            if player_name not in game_state.qualified_players:
                game_state.qualified_players.add(player_name)
                qualified_count += 1
        else:
            if player_name not in game_state.eliminated_players:
                game_state.eliminated_players.add(player_name)
                eliminated_count += 1
    
    # Store results
    game_state.level_results[current_level] = {
        'qualified': qualified_count,
        'eliminated': eliminated_count,
        'min_gain_required': min_gain_percent
    }

# --- MISSING FUNCTION: calculate_final_rankings ---
def calculate_final_rankings():
    """Calculate final tournament rankings"""
    game_state = get_game_state()
    rankings = []
    
    for player_name, player in game_state.players.items():
        if "ALGO_" in player_name:  # Exclude algo traders from rankings
            continue
            
        holdings_value = sum(game_state.prices.get(s, 0) * q for s, q in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        starting_capital = INITIAL_CAPITAL
        gain_percent = ((total_value - starting_capital) / starting_capital) * 100
        
        rankings.append({
            'name': player_name,
            'portfolio_value': total_value,
            'gain_percent': gain_percent,
            'trade_count': len(player.get('trade_timestamps', []))
        })
    
    game_state.final_rankings = sorted(rankings, key=lambda x: x['portfolio_value'], reverse=True)

# --- High-Speed Price Simulation ---
def simulate_high_speed_prices(last_prices):
    """Realistic high-frequency price simulation"""
    game_state = get_game_state()
    prices = last_prices.copy()
    
    # Increase volatility based on level
    level_volatility = 1.0 + (game_state.current_level - 1) * 0.5
    
    for symbol in prices:
        if symbol in game_state.base_prices:
            base_price = game_state.base_prices[symbol]
            volatility = game_state.volatility[symbol] * level_volatility
            
            # High-frequency random walk with momentum
            momentum = random.uniform(-0.02, 0.02)
            noise = random.normalvariate(0, volatility * 0.01)
            sentiment_effect = game_state.market_sentiment.get(symbol, 0) * 0.001
            
            new_price = prices[symbol] * (1 + momentum + noise + sentiment_effect)
            
            # Ensure price doesn't go too far from base (circuit breaker effect)
            max_move = 0.1  # 10% maximum move per tick
            price_change = abs(new_price - base_price) / base_price
            if price_change > max_move:
                correction = (base_price - new_price) / base_price * 0.3
                new_price *= (1 + correction)
            
            prices[symbol] = max(0.01, round(new_price, 2))
            
            # Update volume randomly
            game_state.volume_data[symbol] = max(100, 
                game_state.volume_data[symbol] + random.randint(-50, 100))
    
    return prices

def initialize_base_prices():
    """Initialize realistic base prices for all symbols"""
    game_state = get_game_state()
    
    if not game_state.base_prices:
        # Realistic price ranges for different asset classes
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

# --- Algorithmic Traders ---
def run_algo_trader(trader_name, config, prices):
    """Execute algorithmic trading strategies"""
    game_state = get_game_state()
    
    if not config["active"]:
        return
    
    # Only trade during active game
    if game_state.game_status != "Running":
        return
    
    # Trader-specific logic
    if trader_name == "Momentum Master":
        momentum_trader(trader_name, config, prices)
    elif trader_name == "Mean Reversion Pro":
        mean_reversion_trader(trader_name, config, prices)
    elif trader_name == "Arbitrage Hunter":
        arbitrage_trader(trader_name, config, prices)
    elif trader_name == "Volatility Rider":
        volatility_trader(trader_name, config, prices)

def momentum_trader(trader_name, config, prices):
    """Momentum-based trading strategy"""
    game_state = get_game_state()
    
    # Look for assets with strong momentum
    if len(game_state.price_history) < 5:
        return
    
    for symbol in random.sample(ALL_SYMBOLS, 5):  # Check 5 random symbols
        recent_prices = [ph.get(symbol, prices[symbol]) for ph in game_state.price_history[-5:]]
        if len(recent_prices) < 5:
            continue
            
        returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] for i in range(1, 5)]
        avg_return = sum(returns) / len(returns)
        
        if avg_return > 0.01:  # Strong positive momentum
            # Buy with probability based on aggressiveness
            if random.random() < config["aggressiveness"]:
                qty = random.randint(10, 50)
                execute_algo_trade(trader_name, "Buy", symbol, qty, prices)

def mean_reversion_trader(trader_name, config, prices):
    """Mean reversion trading strategy"""
    game_state = get_game_state()
    
    for symbol in random.sample(ALL_SYMBOLS, 5):
        if symbol not in game_state.base_prices:
            continue
            
        current_price = prices[symbol]
        base_price = game_state.base_prices[symbol]
        deviation = (current_price - base_price) / base_price
        
        if abs(deviation) > 0.05:  # 5% deviation from base
            if deviation > 0 and random.random() < config["aggressiveness"]:
                # Overbought - sell
                qty = random.randint(5, 30)
                execute_algo_trade(trader_name, "Sell", symbol, qty, prices)
            elif deviation < 0 and random.random() < config["aggressiveness"]:
                # Oversold - buy
                qty = random.randint(5, 30)
                execute_algo_trade(trader_name, "Buy", symbol, qty, prices)

def arbitrage_trader(trader_name, config, prices):
    """Arbitrage trading strategy"""
    # Look for price discrepancies between related assets
    pairs = [
        ('RELIANCE.NS', 'RELIANCE-FUT'),
        ('TCS.NS', 'TCS-FUT'),
        ('NIFTY-2XL', 'NIFTY-2XS')
    ]
    
    for asset1, asset2 in pairs:
        if asset1 in prices and asset2 in prices:
            price1 = prices[asset1]
            price2 = prices[asset2]
            spread = abs(price1 - price2) / min(price1, price2)
            
            if spread > 0.02 and random.random() < config["aggressiveness"]:  # 2% spread
                if price1 > price2:
                    execute_algo_trade(trader_name, "Buy", asset2, random.randint(10, 40), prices)
                    execute_algo_trade(trader_name, "Sell", asset1, random.randint(10, 40), prices)
                else:
                    execute_algo_trade(trader_name, "Buy", asset1, random.randint(10, 40), prices)
                    execute_algo_trade(trader_name, "Sell", asset2, random.randint(10, 40), prices)

def volatility_trader(trader_name, config, prices):
    """Volatility-based trading strategy"""
    game_state = get_game_state()
    
    # Find high volatility assets
    high_vol_assets = []
    for symbol in ALL_SYMBOLS:
        if game_state.volatility[symbol] > 0.25:  # High volatility threshold
            high_vol_assets.append(symbol)
    
    for symbol in random.sample(high_vol_assets, min(3, len(high_vol_assets))):
        if random.random() < config["aggressiveness"]:
            # Random direction in high volatility
            action = random.choice(["Buy", "Sell"])
            qty = random.randint(15, 60)
            execute_algo_trade(trader_name, action, symbol, qty, prices)

def execute_algo_trade(trader_name, action, symbol, qty, prices):
    """Execute trade for algorithmic trader"""
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
            "trade_timestamps": []
        }
        game_state.transactions[algo_player_name] = []
    
    player = game_state.players[algo_player_name]
    
    # Simple trade execution for algo
    current_price = prices.get(symbol, 0)
    if current_price <= 0:
        return False
    
    cost = current_price * qty
    
    if action == "Buy" and player['capital'] >= cost:
        player['capital'] -= cost
        player['holdings'][symbol] = player['holdings'].get(symbol, 0) + qty
        log_transaction(algo_player_name, f"ALGO {action}", symbol, qty, current_price, cost)
        return True
    elif action == "Sell":
        current_qty = player['holdings'].get(symbol, 0)
        if current_qty >= qty:
            player['capital'] += cost
            player['holdings'][symbol] -= qty
            if player['holdings'][symbol] == 0:
                del player['holdings'][symbol]
            log_transaction(algo_player_name, f"ALGO {action}", symbol, qty, current_price, cost)
            return True
    
    return False

# --- Trading Execution ---
def execute_trade(player_name, player, action, symbol, qty, prices):
    """Execute trade for human players"""
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
    
    cost = current_price * qty
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
        log_transaction(player_name, action, symbol, qty, current_price, cost)
        # Update market sentiment
        game_state.market_sentiment[symbol] = game_state.market_sentiment.get(symbol, 0) + (qty / 1000)
        player['trade_timestamps'].append(time.time())
        return True
    else:
        st.error("Trade failed: Insufficient capital or holdings")
        return False

def log_transaction(player_name, action, symbol, qty, price, total):
    """Log transaction to game state"""
    game_state = get_game_state()
    timestamp = time.strftime("%H:%M:%S")
    game_state.transactions.setdefault(player_name, []).append([
        timestamp, action, symbol, qty, price, total
    ])

# --- UI Components ---
def render_vip_sidebar():
    """VIP/admin sidebar with controls"""
    game_state = get_game_state()
    
    with st.sidebar:
        st.title("🎮 Tournament Control")
        
        # Player registration
        if 'player' not in st.query_params:
            st.subheader("👤 Player Registration")
            player_name = st.text_input("Enter Your Name")
            if st.button("Join Tournament", use_container_width=True):
                if player_name and player_name.strip() and player_name not in game_state.players:
                    game_state.players[player_name] = {
                        "name": player_name,
                        "mode": "Trader",
                        "capital": INITIAL_CAPITAL,
                        "holdings": {},
                        "value_history": [INITIAL_CAPITAL],
                        "trade_timestamps": []
                    }
                    game_state.transactions[player_name] = []
                    st.query_params["player"] = player_name
                    st.rerun()
        else:
            player_name = st.query_params.get("player")
            st.success(f"Welcome, **{player_name}**!")
            if st.button("Leave Game", use_container_width=True):
                st.query_params.clear()
                st.rerun()
        
        st.markdown("---")
        
        # Admin controls
        st.subheader("🔐 Admin Panel")
        admin_pass = st.text_input("Admin Password", type="password")
        
        if admin_pass == ADMIN_PASSWORD:
            st.session_state.admin_access = True
            st.success("✅ Admin Access Granted")
        
        if st.session_state.get('admin_access'):
            # Game controls
            st.subheader("⚙️ Game Controls")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("▶️ Start Game", use_container_width=True):
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
            
            # Algo trader controls
            st.subheader("🤖 Algo Traders")
            for algo_name, algo_config in game_state.algo_traders.items():
                if st.button(f"{'✅' if algo_config['active'] else '❌'} {algo_name}", 
                           use_container_width=True):
                    algo_config['active'] = not algo_config['active']
                    st.rerun()
            
            # Market controls
            st.subheader("📊 Market Controls")
            if st.button("🚫 Halt Trading", use_container_width=True):
                game_state.admin_trading_halt = True
            if st.button("✅ Resume Trading", use_container_width=True):
                game_state.admin_trading_halt = False

def render_trading_interface(prices):
    """Main trading interface"""
    game_state = get_game_state()
    player_name = st.query_params.get("player")
    
    if not player_name or player_name not in game_state.players:
        st.info("👆 Please register from the sidebar to start trading")
        return
    
    player = game_state.players[player_name]
    
    # Portfolio overview
    holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
    total_value = player['capital'] + holdings_value
    gain_percent = ((total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    st.subheader("💰 Portfolio Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cash", f"₹{player['capital']:,.0f}")
    with col2:
        st.metric("Holdings", f"₹{holdings_value:,.0f}")
    with col3:
        st.metric("Total", f"₹{total_value:,.0f}")
    with col4:
        st.metric("P&L", f"{gain_percent:+.1f}%")
    
    # Trading panel
    st.subheader("⚡ Trade Execution")
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
        st.info(f"Current Price: ₹{current_price:,.2f}")
    
    with col2:
        action = st.radio("Action", ["Buy", "Sell"])
        qty = st.number_input("Quantity", min_value=1, max_value=1000, value=10)
    
    with col3:
        if st.button("🚀 Execute Trade", type="primary", use_container_width=True):
            if execute_trade(player_name, player, action, symbol, qty, prices):
                st.success("✅ Trade executed!")
    
    # Holdings display
    st.subheader("📦 Your Holdings")
    if player['holdings']:
        holdings_data = []
        for symbol, qty in player['holdings'].items():
            if qty > 0:
                current_price = prices.get(symbol, 0)
                value = current_price * qty
                holdings_data.append({
                    'Asset': symbol,
                    'Quantity': qty,
                    'Current Price': f"₹{current_price:,.2f}",
                    'Value': f"₹{value:,.0f}"
                })
        
        if holdings_data:
            st.dataframe(pd.DataFrame(holdings_data), use_container_width=True)
    else:
        st.info("No holdings yet. Start trading!")

def render_market_overview(prices):
    """Market data overview"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Market Overview")
    
    game_state = get_game_state()
    
    # Market status
    remaining_time, time_type = get_remaining_time()
    if game_state.game_status == "Running":
        minutes, seconds = divmod(int(remaining_time), 60)
        st.sidebar.write(f"⏰ Level {game_state.current_level}: {minutes:02d}:{seconds:02d}")
    
    # Qualification status
    if game_state.current_level in QUALIFICATION_CRITERIA:
        criteria = QUALIFICATION_CRITERIA[game_state.current_level]
        st.sidebar.write(f"🎯 Target: +{criteria['min_gain_percent']}%")
    
    # Quick market data
    st.sidebar.subheader("💹 Live Prices")
    sample_symbols = random.sample(ALL_SYMBOLS, 8)
    for symbol in sample_symbols:
        price = prices.get(symbol, 0)
        change = random.uniform(-2, 2)
        st.sidebar.write(f"{symbol}: ₹{price:,.0f} ({change:+.1f}%)")

def render_leaderboard(prices):
    """Live leaderboard"""
    game_state = get_game_state()
    
    st.subheader("🏆 Live Leaderboard")
    
    leaderboard_data = []
    for player_name, player in game_state.players.items():
        if "ALGO_" in player_name:  # Mark algo traders
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
            'Status': status
        })
    
    if leaderboard_data:
        df = pd.DataFrame(leaderboard_data)
        df = df.sort_values('Gain %', key=lambda x: x.str.rstrip('%').astype(float), ascending=False)
        st.dataframe(df, use_container_width=True)

def render_qualification_status(prices):
    """Show qualification progress"""
    game_state = get_game_state()
    player_name = st.query_params.get("player")
    
    if not player_name or player_name not in game_state.players:
        return
    
    if player_name in game_state.eliminated_players:
        st.error("## ❌ Eliminated")
        st.info("You didn't meet the qualification criteria for this level. Better luck next time!")
        return
    
    if game_state.current_level in QUALIFICATION_CRITERIA:
        criteria = QUALIFICATION_CRITERIA[game_state.current_level]
        min_gain = criteria["min_gain_percent"]
        
        player = game_state.players[player_name]
        holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        gain_percent = ((total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        
        progress = min(100, (gain_percent / min_gain) * 100)
        
        st.subheader("🎯 Qualification Progress")
        st.write(f"**Target:** +{min_gain}% gain")
        st.write(f"**Your Gain:** {gain_percent:+.1f}%")
        
        st.progress(progress / 100)
        
        if gain_percent >= min_gain:
            st.success("✅ You are QUALIFIED for the next level!")
        else:
            needed = max(0, min_gain - gain_percent)
            st.warning(f"💰 Need +{needed:.1f}% more to qualify")

# --- MISSING FUNCTION: get_remaining_time ---
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

# --- Main Application ---
def main():
    # Initialize session state
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.last_refresh = time.time()
    
    game_state = get_game_state()
    
    # Update game state
    update_game_state()
    
    # Initialize prices if not done
    if not game_state.base_prices:
        initialize_base_prices()
    
    # Run algo traders
    for algo_name, algo_config in game_state.algo_traders.items():
        run_algo_trader(algo_name, algo_config, game_state.prices)
    
    # Simulate price changes
    if game_state.game_status == "Running":
        game_state.prices = simulate_high_speed_prices(game_state.prices)
        game_state.price_history.append(game_state.prices.copy())
        if len(game_state.price_history) > 100:
            game_state.price_history.pop(0)
    
    # Render UI
    render_vip_sidebar()
    
    # Main content
    st.title("🎯 BlockVista Market Frenzy")
    st.subheader("Professional Trading Championship")
    
    # Game status
    if game_state.game_status == "Finished" and game_state.final_rankings:
        st.balloons()
        st.success("## 🏆 Tournament Complete!")
        winner = game_state.final_rankings[0]
        st.success(f"**Champion:** {winner['name']} with ₹{winner['portfolio_value']:,.0f} (+{winner['gain_percent']:.1f}%)")
    
    # Main columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_trading_interface(game_state.prices)
        render_leaderboard(game_state.prices)
    
    with col2:
        render_qualification_status(game_state.prices)
        
        # Algo trader status
        st.subheader("🤖 Algorithmic Traders")
        for algo_name, algo_config in game_state.algo_traders.items():
            status = "🟢 Active" if algo_config["active"] else "🔴 Inactive"
            st.write(f"{status} **{algo_name}** - {algo_config['description']}")
    
    # Auto-refresh
    current_time = time.time()
    refresh_interval = 1.0  # 1 second for high-speed simulation
    if current_time - st.session_state.last_refresh > refresh_interval:
        st.session_state.last_refresh = current_time
        st.rerun()

if __name__ == "__main__":
    main()

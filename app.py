# BlockVista Market Frenzy - Elite Trading Championship
# Manual Level Progression with Extreme Difficulty

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
    page_title="BlockVista Market Frenzy - Elite Championship",
    page_icon="🏆",
    initial_sidebar_state="expanded"
)

# --- Elite Constants ---
GAME_NAME = "BLOCKVISTA ELITE TRADING CHAMPIONSHIP"
INITIAL_CAPITAL = 10000000  # ₹1 Crore for elite traders
ADMIN_PASSWORD = "100370"
MAX_PLAYERS = 30

# EXTREME Qualification Criteria for Elite Traders
QUALIFICATION_CRITERIA = {
    1: {"min_gain_percent": 50, "description": "Achieve 50% portfolio growth (₹1.5Cr)", "duration": "10 minutes"},
    2: {"min_gain_percent": 120, "description": "Achieve 120% portfolio growth (₹2.2Cr)", "duration": "8 minutes"}, 
    3: {"min_gain_percent": 200, "description": "Achieve 200% portfolio growth (₹3Cr)", "duration": "6 minutes"}
}

# Elite Asset Universe
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
    'GOLD-1KG', 'SILVER-10KG', 'CRUDEOIL', 'NATURALGAS', 'COPPER-1TON',
    'ALUMINIUM', 'ZINC', 'NICKEL', 'LEAD', 'COTTON-10BALES',
    'SOYBEAN', 'SUGAR', 'COFFEE', 'COCOA', 'PALMOLINE'
]

CRYPTO = [
    'BTC-INR', 'ETH-INR', 'SOL-INR', 'ADA-INR', 'DOT-INR',
    'MATIC-INR', 'AVAX-INR', 'LINK-INR', 'ATOM-INR', 'XRP-INR',
    'DOGE-INR', 'SHIB-INR', 'BNB-INR', 'TRX-INR', 'LTC-INR'
]

LEVERAGED_ETFS = [
    'NIFTY-3XL', 'NIFTY-3XS', 'BANKNIFTY-3XL', 'BANKNIFTY-3XS',
    'MIDCAP-3XL', 'MIDCAP-3XS', 'SENSEX-3XL', 'SENSEX-3XS',
    'USDINR-3XL', 'USDINR-3XS', 'GOLD-3XL', 'GOLD-3XS',
    'OIL-3XL', 'OIL-3XS', 'VIX-3X'
]

ALL_SYMBOLS = STOCKS + FUTURES + OPTIONS + COMMODITIES + CRYPTO + LEVERAGED_ETFS

# Professional Algorithmic Traders
ALGO_TRADERS = {
    "Quantum Momentum": {
        "description": "AI-powered momentum with quantum computing simulation",
        "aggressiveness": 0.95,
        "risk_tolerance": 0.8,
        "speed": "Ultra High",
        "active": False,
        "capital": INITIAL_CAPITAL * 5
    },
    "Neural Arbitrage": {
        "description": "Deep learning based cross-asset arbitrage",
        "aggressiveness": 0.9,
        "risk_tolerance": 0.4,
        "speed": "Lightning",
        "active": False,
        "capital": INITIAL_CAPITAL * 6
    },
    "Volatility AI": {
        "description": "Reinforcement learning for volatility exploitation",
        "aggressiveness": 0.85,
        "risk_tolerance": 0.9,
        "speed": "High Frequency",
        "active": False,
        "capital": INITIAL_CAPITAL * 4
    },
    "HFT Master": {
        "description": "Institutional-grade high frequency trading",
        "aggressiveness": 0.98,
        "risk_tolerance": 0.7,
        "speed": "Nano-second",
        "active": False,
        "capital": INITIAL_CAPITAL * 8
    }
}

# --- Elite Game State Management ---
@st.cache_resource
def get_game_state():
    class EliteGameState:
        def __init__(self):
            # Core game state
            self.players = {}
            self.game_status = "Stopped"  # Stopped, Level1, Level2, Level3, Finished
            self.level_start_time = 0
            self.current_level = 0  # 0 = not started, 1-3 = active levels
            self.level_durations = [10 * 60, 8 * 60, 6 * 60]  # 10, 8, 6 minutes
            
            # Market data
            self.prices = {}
            self.base_prices = {}
            self.price_history = []
            self.volume_data = {s: random.randint(10000, 100000) for s in ALL_SYMBOLS}
            self.volatility = {s: random.uniform(0.2, 0.6) for s in ALL_SYMBOLS}
            
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
            
            # Manual level control
            self.level_completed = False
            self.qualification_checked = False
            
        def reset(self):
            self.__init__()
            
    return EliteGameState()

# --- High-Frequency Price Simulation ---
def simulate_elite_prices(last_prices):
    """Extreme volatility price simulation for elite traders"""
    game_state = get_game_state()
    prices = last_prices.copy()
    
    # Extreme volatility based on level
    level_multiplier = 1.0 + (game_state.current_level - 1) * 0.8
    
    for symbol in prices:
        if symbol in game_state.base_prices:
            base_price = game_state.base_prices[symbol]
            volatility = game_state.volatility[symbol] * level_multiplier
            
            # High-frequency factors
            momentum = random.uniform(-0.05, 0.05)  # Increased range
            noise = random.normalvariate(0, volatility * 0.02)  # More noise
            sentiment_effect = game_state.market_sentiment.get(symbol, 0) * 0.002
            volume_effect = (game_state.volume_data.get(symbol, 1000) - 50000) / 50000 * 0.01
            
            price_change = momentum + noise + sentiment_effect + volume_effect
            new_price = prices[symbol] * (1 + price_change)
            
            # Circuit breaker with wider range for elite trading
            max_move = 0.15  # 15% maximum move per tick
            price_deviation = abs(new_price - base_price) / base_price
            if price_deviation > max_move:
                correction = (base_price - new_price) / base_price * 0.2
                new_price *= (1 + correction)
            
            prices[symbol] = max(0.01, round(new_price, 2))
            
            # Update volume with larger swings
            game_state.volume_data[symbol] = max(1000, 
                game_state.volume_data[symbol] + random.randint(-500, 1000))
    
    return prices

def initialize_elite_prices():
    """Initialize realistic prices for elite trading"""
    game_state = get_game_state()
    
    if not game_state.base_prices:
        # Realistic price ranges for elite markets
        price_ranges = {
            'STOCKS': (1000, 10000),
            'FUTURES': (10000, 100000),
            'OPTIONS': (50, 2000),
            'COMMODITIES': (5000, 500000),
            'CRYPTO': (1000, 10000000),
            'LEVERAGED_ETFS': (500, 5000)
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

# --- Elite Algorithmic Traders ---
def run_elite_algo_traders(prices):
    """Run elite algorithmic trading strategies"""
    game_state = get_game_state()
    
    for algo_name, algo_config in game_state.algo_traders.items():
        if algo_config["active"] and game_state.game_status.startswith("Level"):
            if algo_name == "Quantum Momentum":
                quantum_momentum_strategy(algo_name, algo_config, prices)
            elif algo_name == "Neural Arbitrage":
                neural_arbitrage_strategy(algo_name, algo_config, prices)
            elif algo_name == "Volatility AI":
                volatility_ai_strategy(algo_name, algo_config, prices)
            elif algo_name == "HFT Master":
                hft_master_strategy(algo_name, algo_config, prices)

def quantum_momentum_strategy(algo_name, config, prices):
    """Quantum-inspired momentum strategy"""
    game_state = get_game_state()
    
    if len(game_state.price_history) < 10:
        return
    
    for symbol in random.sample(ALL_SYMBOLS, 10):
        recent_prices = [ph.get(symbol, prices[symbol]) for ph in game_state.price_history[-10:]]
        if len(recent_prices) < 10:
            continue
            
        # Advanced momentum calculation
        returns = [(recent_prices[i] - recent_prices[i-1]) / recent_prices[i-1] for i in range(1, 10)]
        momentum = sum(returns) / len(returns)
        volatility = np.std(returns) if len(returns) > 1 else 0
        
        # Quantum-inspired decision making
        if momentum > 0.01 and volatility > 0.005 and random.random() < config["aggressiveness"]:
            qty = random.randint(100, 500)  # Larger quantities
            execute_elite_algo_trade(algo_name, "Buy", symbol, qty, prices)
        elif momentum < -0.01 and volatility > 0.005 and random.random() < config["aggressiveness"]:
            qty = random.randint(100, 500)
            execute_elite_algo_trade(algo_name, "Sell", symbol, qty, prices)

def neural_arbitrage_strategy(algo_name, config, prices):
    """Neural network inspired arbitrage"""
    arbitrage_pairs = [
        ('RELIANCE.NS', 'RELIANCE-FUT'),
        ('TCS.NS', 'TCS-FUT'),
        ('HDFCBANK.NS', 'HDFC-FUT'),
        ('NIFTY-3XL', 'NIFTY-3XS'),
        ('BTC-INR', 'GOLD-1KG')
    ]
    
    for asset1, asset2 in arbitrage_pairs:
        if asset1 in prices and asset2 in prices:
            price1 = prices[asset1]
            price2 = prices[asset2]
            spread = abs(price1 - price2) / min(price1, price2)
            
            if spread > 0.03 and random.random() < config["aggressiveness"]:  # 3% spread
                if price1 > price2:
                    execute_elite_algo_trade(algo_name, "Buy", asset2, random.randint(200, 800), prices)
                    execute_elite_algo_trade(algo_name, "Sell", asset1, random.randint(200, 800), prices)
                else:
                    execute_elite_algo_trade(algo_name, "Buy", asset1, random.randint(200, 800), prices)
                    execute_elite_algo_trade(algo_name, "Sell", asset2, random.randint(200, 800), prices)

def volatility_ai_strategy(algo_name, config, prices):
    """AI-powered volatility trading"""
    game_state = get_game_state()
    
    high_vol_assets = [s for s in ALL_SYMBOLS if game_state.volatility[s] > 0.4]
    
    for symbol in random.sample(high_vol_assets, min(8, len(high_vol_assets))):
        if len(game_state.price_history) >= 5:
            recent_prices = [ph.get(symbol, prices[symbol]) for ph in game_state.price_history[-5:]]
            price_range = max(recent_prices) - min(recent_prices)
            avg_price = np.mean(recent_prices)
            volatility_ratio = price_range / avg_price
            
            if volatility_ratio > 0.06 and random.random() < config["aggressiveness"]:
                # Volatility breakout with AI direction
                current_price = prices[symbol]
                momentum = sum([1 for i in range(1, 5) if recent_prices[i] > recent_prices[i-1]])
                
                if momentum >= 3:
                    execute_elite_algo_trade(algo_name, "Buy", symbol, random.randint(150, 600), prices)
                else:
                    execute_elite_algo_trade(algo_name, "Sell", symbol, random.randint(150, 600), prices)

def hft_master_strategy(algo_name, config, prices):
    """High-frequency trading strategy"""
    # HFT trades very frequently on small movements
    for symbol in random.sample(ALL_SYMBOLS, 15):
        if random.random() < 0.3:  # 30% chance per symbol
            action = random.choice(["Buy", "Sell"])
            qty = random.randint(50, 200)  # Smaller, faster trades
            execute_elite_algo_trade(algo_name, action, symbol, qty, prices)

def execute_elite_algo_trade(trader_name, action, symbol, qty, prices):
    """Execute trade for elite algorithmic trader"""
    game_state = get_game_state()
    algo_player_name = f"ALGO_{trader_name}"
    
    if algo_player_name not in game_state.players:
        game_state.players[algo_player_name] = {
            "name": algo_player_name,
            "mode": "Algorithmic",
            "capital": ALGO_TRADERS[trader_name]["capital"],
            "holdings": {},
            "value_history": [ALGO_TRADERS[trader_name]["capital"]],
            "trade_timestamps": [],
            "strategy": trader_name
        }
        game_state.transactions[algo_player_name] = []
    
    player = game_state.players[algo_player_name]
    current_price = prices.get(symbol, 0)
    
    if current_price <= 0:
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
        timestamp = time.strftime("%H:%M:%S.%f")[:-3]
        game_state.transactions[algo_player_name].append([
            timestamp, f"ALGO {action}", symbol, qty, current_price, cost
        ])
        
        # Update market sentiment more aggressively
        game_state.market_sentiment[symbol] = game_state.market_sentiment.get(symbol, 0) + (qty / 100)
        player['trade_timestamps'].append(time.time())
        
        return True
    
    return False

# --- Trading Execution ---
def execute_elite_trade(player_name, player, action, symbol, qty, prices):
    """Execute trade for elite human traders"""
    game_state = get_game_state()
    
    # Check if player is eliminated
    if player_name in game_state.eliminated_players:
        st.error("❌ ELIMINATED - You didn't qualify for this level!")
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
        log_elite_transaction(player_name, action, symbol, qty, current_price, cost)
        # Update market sentiment
        game_state.market_sentiment[symbol] = game_state.market_sentiment.get(symbol, 0) + (qty / 500)
        player['trade_timestamps'].append(time.time())
        return True
    else:
        st.error("Trade failed: Insufficient capital or holdings")
        return False

def log_elite_transaction(player_name, action, symbol, qty, price, total):
    """Log transaction for elite trading"""
    game_state = get_game_state()
    timestamp = time.strftime("%H:%M:%S")
    game_state.transactions.setdefault(player_name, []).append([
        timestamp, action, symbol, qty, price, total
    ])

# --- Manual Level Management ---
def update_game_state():
    """Handle manual level progression"""
    game_state = get_game_state()
    
    # Check if current level time is up
    if game_state.game_status.startswith("Level") and not game_state.level_completed:
        current_time = time.time()
        level_duration = game_state.level_durations[game_state.current_level - 1]
        
        if current_time - game_state.level_start_time >= level_duration:
            game_state.level_completed = True
            # Auto-check qualifications when time expires
            if not game_state.qualification_checked:
                check_elite_qualifications()

def check_elite_qualifications():
    """Check qualifications with extreme criteria"""
    game_state = get_game_state()
    current_level = game_state.current_level
    
    if current_level not in QUALIFICATION_CRITERIA:
        return
    
    criteria = QUALIFICATION_CRITERIA[current_level]
    min_gain_percent = criteria["min_gain_percent"]
    
    qualified_count = 0
    eliminated_count = 0
    
    for player_name, player in game_state.players.items():
        if "ALGO_" in player_name:  # Skip algo traders
            continue
            
        if player_name in game_state.eliminated_players:
            continue
        
        holdings_value = sum(game_state.prices.get(s, 0) * q for s, q in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        gain_percent = ((total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        
        if gain_percent >= min_gain_percent:
            if player_name not in game_state.qualified_players:
                game_state.qualified_players.add(player_name)
                qualified_count += 1
        else:
            if player_name not in game_state.eliminated_players:
                game_state.eliminated_players.add(player_name)
                eliminated_count += 1
    
    game_state.qualification_checked = True
    game_state.level_results[current_level] = {
        'qualified': qualified_count,
        'eliminated': eliminated_count,
        'min_gain_required': min_gain_percent
    }

def start_level(level_number):
    """Start a specific level manually"""
    game_state = get_game_state()
    game_state.current_level = level_number
    game_state.game_status = f"Level{level_number}"
    game_state.level_start_time = time.time()
    game_state.level_completed = False
    game_state.qualification_checked = False

def calculate_final_rankings():
    """Calculate final elite rankings"""
    game_state = get_game_state()
    rankings = []
    
    for player_name, player in game_state.players.items():
        if "ALGO_" in player_name:
            continue
            
        holdings_value = sum(game_state.prices.get(s, 0) * q for s, q in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        gain_percent = ((total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        
        rankings.append({
            'name': player_name,
            'portfolio_value': total_value,
            'gain_percent': gain_percent,
            'trade_count': len(player.get('trade_timestamps', []))
        })
    
    game_state.final_rankings = sorted(rankings, key=lambda x: x['portfolio_value'], reverse=True)

def get_remaining_time():
    """Calculate remaining time for current level"""
    game_state = get_game_state()
    
    if game_state.game_status.startswith("Level"):
        level_duration = game_state.level_durations[game_state.current_level - 1]
        remaining = max(0, level_duration - (time.time() - game_state.level_start_time))
        return remaining
    else:
        return 0

# --- Elite UI Components ---
def render_elite_header():
    """Render elite championship header"""
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #1e3c72, #2a5298);
            padding: 2rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 15px 35px rgba(0,0,0,0.3);
            border: 3px solid #f59e0b;
        ">
            <h1 style="margin:0; font-size: 3.5rem; font-weight: 900; text-align: center;">
                {GAME_NAME}
            </h1>
            <p style="margin:0; font-size: 1.4rem; text-align: center; opacity: 0.9;">
                Elite Trading Championship • ₹1 Crore Starting Capital
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_elite_sidebar():
    """Elite sidebar with advanced controls"""
    game_state = get_game_state()
    
    with st.sidebar:
        st.title("🎯 Elite Control Center")
        
        # Player registration for elite traders
        if 'player' not in st.query_params:
            st.subheader("👑 Elite Registration")
            player_name = st.text_input("Trader Name")
            trader_type = st.selectbox("Trader Type", 
                                     ["HNI Investor", "HFT Professional", "Institutional", "Proprietary Trader"])
            
            if st.button("Join Elite Championship", type="primary", use_container_width=True):
                if player_name and player_name.strip() and player_name not in game_state.players:
                    game_state.players[player_name] = {
                        "name": player_name,
                        "type": trader_type,
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
            player_type = game_state.players[player_name]["type"] if player_name in game_state.players else "Trader"
            st.success(f"👑 **{player_name}** - {player_type}")
            if st.button("Leave Championship", use_container_width=True):
                st.query_params.clear()
                st.rerun()
        
        st.markdown("---")
        
        # Elite admin controls
        st.subheader("🔐 Elite Admin")
        admin_pass = st.text_input("Master Password", type="password")
        
        if admin_pass == ADMIN_PASSWORD:
            st.session_state.elite_admin = True
            st.success("✅ Elite Admin Access")
        
        if st.session_state.get('elite_admin'):
            # Manual level controls
            st.subheader("🎮 Level Management")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🚀 Level 1", use_container_width=True, 
                           disabled=game_state.game_status != "Stopped" and game_state.current_level != 0):
                    start_level(1)
                    st.rerun()
            with col2:
                if st.button("⚡ Level 2", use_container_width=True,
                           disabled=game_state.current_level != 1 or not game_state.level_completed):
                    start_level(2)
                    st.rerun()
            with col3:
                if st.button("🏆 Level 3", use_container_width=True,
                           disabled=game_state.current_level != 2 or not game_state.level_completed):
                    start_level(3)
                    st.rerun()
            
            # Qualification check
            if game_state.game_status.startswith("Level") and not game_state.qualification_checked:
                if st.button("📊 Check Qualifications", use_container_width=True):
                    check_elite_qualifications()
                    st.rerun()
            
            # Game controls
            st.subheader("⚙️ Master Controls")
            if st.button("⏹️ Stop Level", use_container_width=True,
                       disabled=not game_state.game_status.startswith("Level")):
                game_state.game_status = "Stopped"
                st.rerun()
            
            if st.button("🔄 Reset Championship", use_container_width=True):
                game_state.reset()
                st.rerun()
            
            # Algo trader controls
            st.subheader("🤖 Elite Algorithms")
            for algo_name, algo_config in game_state.algo_traders.items():
                status = "🟢" if algo_config['active'] else "🔴"
                if st.button(f"{status} {algo_name}", use_container_width=True):
                    algo_config['active'] = not algo_config['active']
                    st.rerun()
            
            # Market controls
            st.subheader("📈 Market Controls")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🚫 Halt All Trading", use_container_width=True):
                    game_state.admin_trading_halt = True
            with col2:
                if st.button("✅ Resume Trading", use_container_width=True):
                    game_state.admin_trading_halt = False

def render_elite_trading_interface(prices):
    """Elite trading interface"""
    game_state = get_game_state()
    player_name = st.query_params.get("player")
    
    if not player_name or player_name not in game_state.players:
        render_elite_welcome()
        return
    
    player = game_state.players[player_name]
    
    # Elite portfolio header
    render_elite_portfolio_header(player, prices)
    
    # Trading interface
    st.subheader("⚡ Elite Trading Terminal")
    
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        asset_category = st.selectbox("Asset Class", 
                                    ["Stocks", "Futures", "Options", "Commodities", "Crypto", "Leveraged ETFs"])
        symbol_map = {
            "Stocks": STOCKS, "Futures": FUTURES, "Options": OPTIONS,
            "Commodities": COMMODITIES, "Crypto": CRYPTO, "Leveraged ETFs": LEVERAGED_ETFS
        }
        symbol = st.selectbox("Select Instrument", symbol_map[asset_category])
        current_price = prices.get(symbol, 0)
        
        # Elite price display
        if len(game_state.price_history) >= 2:
            prev_price = game_state.price_history[-2].get(symbol, current_price)
            change = ((current_price - prev_price) / prev_price) * 100
            color = "#10b981" if change >= 0 else "#ef4444"
            st.markdown(f"<h3 style='color: {color};'>₹{current_price:,.2f} ({change:+.2f}%)</h3>", 
                       unsafe_allow_html=True)
    
    with col2:
        action = st.radio("Order", ["Buy", "Sell"], horizontal=True)
        order_type = st.selectbox("Type", ["Market", "Limit"])
    
    with col3:
        qty = st.number_input("Quantity", min_value=1, max_value=10000, value=100)
        if order_type == "Limit":
            limit_price = st.number_input("Limit Price", min_value=0.01, value=current_price, step=0.01)
    
    with col4:
        st.write("")
        st.write("")
        if st.button("🎯 EXECUTE", type="primary", use_container_width=True, 
                   disabled=not game_state.game_status.startswith("Level")):
            if execute_elite_trade(player_name, player, action, symbol, qty, prices):
                st.success("✅ Elite Trade Executed!")
    
    # Holdings display
    st.subheader("📊 Elite Holdings")
    if player['holdings']:
        holdings_data = []
        for symbol, qty in player['holdings'].items():
            if qty > 0:
                current_price = prices.get(symbol, 0)
                value = current_price * qty
                holdings_data.append({
                    'Instrument': symbol,
                    'Quantity': f"{qty:,}",
                    'Price': f"₹{current_price:,.2f}",
                    'Value': f"₹{value:,.0f}"
                })
        
        if holdings_data:
            st.dataframe(pd.DataFrame(holdings_data), use_container_width=True)
    else:
        st.info("No positions yet. Start trading to build your elite portfolio!")

def render_elite_portfolio_header(player, prices):
    """Elite portfolio header"""
    holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
    total_value = player['capital'] + holdings_value
    gain_percent = ((total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    gain_color = "#10b981" if gain_percent >= 0 else "#ef4444"
    
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #0f172a, #1e293b);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            margin-bottom: 1rem;
            border-left: 5px solid #f59e0b;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <h2 style="margin:0; color: #f59e0b;">{player['name']}</h2>
                    <p style="margin:0; opacity: 0.8;">{player.get('type', 'Elite Trader')}</p>
                </div>
                <div style="text-align: right;">
                    <h3 style="margin:0; color: {gain_color};">+{gain_percent:.1f}%</h3>
                    <p style="margin:0; font-size: 1.2rem;">₹{total_value:,.0f}</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Portfolio metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("💰 Cash", f"₹{player['capital']:,.0f}")
    with col2:
        st.metric("📈 Holdings", f"₹{holdings_value:,.0f}")
    with col3:
        st.metric("🏦 Net Worth", f"₹{total_value:,.0f}")
    with col4:
        st.metric("📊 P&L", f"₹{total_value - INITIAL_CAPITAL:,.0f}")
    with col5:
        st.metric("⚡ Trades", len(player.get('trade_timestamps', [])))

def render_elite_leaderboard(prices):
    """Elite leaderboard"""
    game_state = get_game_state()
    
    st.subheader("🏆 Elite Leaderboard")
    
    leaderboard_data = []
    for player_name, player in game_state.players.items():
        if "ALGO_" in player_name:
            continue
            
        holdings_value = sum(prices.get(s, 0) * q for s, q in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        gain_percent = ((total_value - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
        
        status = "✅ QUALIFIED" if player_name in game_state.qualified_players else \
                 "❌ ELIMINATED" if player_name in game_state.eliminated_players else "🎯 IN PLAY"
        
        leaderboard_data.append({
            'Rank': 0,
            'Trader': player_name,
            'Type': player.get('type', 'Trader'),
            'Portfolio (₹Cr)': f"{total_value/10000000:.2f}",
            'Gain %': f"{gain_percent:+.1f}%",
            'Status': status
        })
    
    if leaderboard_data:
        # Sort and rank
        leaderboard_data.sort(key=lambda x: float(x['Portfolio (₹Cr)']), reverse=True)
        for i, player in enumerate(leaderboard_data):
            player['Rank'] = i + 1
        
        df = pd.DataFrame(leaderboard_data)
        st.dataframe(df[['Rank', 'Trader', 'Type', 'Portfolio (₹Cr)', 'Gain %', 'Status']], 
                    use_container_width=True, hide_index=True)

def render_level_status():
    """Display current level status"""
    game_state = get_game_state()
    
    if game_state.game_status.startswith("Level"):
        remaining = get_remaining_time()
        minutes, seconds = divmod(int(remaining), 60)
        
        level_info = QUALIFICATION_CRITERIA[game_state.current_level]
        
        st.subheader(f"🎯 Level {game_state.current_level} - IN PROGRESS")
        st.write(f"**Time Remaining:** {minutes:02d}:{seconds:02d}")
        st.write(f"**Target:** {level_info['description']}")
        st.write(f"**Duration:** {level_info['duration']}")
        
        # Progress bar for time
        level_duration = game_state.level_durations[game_state.current_level - 1]
        time_progress = 1 - (remaining / level_duration)
        st.progress(time_progress)
        
    elif game_state.level_completed:
        st.subheader(f"✅ Level {game_state.current_level} - COMPLETED")
        if game_state.qualification_checked:
            results = game_state.level_results.get(game_state.current_level, {})
            st.success(f"**Qualified:** {results.get('qualified', 0)} traders")
            st.error(f"**Eliminated:** {results.get('eliminated', 0)} traders")
        
    elif game_state.game_status == "Finished":
        st.balloons()
        st.success("## 🏆 CHAMPIONSHIP COMPLETE!")
        if game_state.final_rankings:
            winner = game_state.final_rankings[0]
            st.success(f"**ELITE CHAMPION:** {winner['name']} with ₹{winner['portfolio_value']/10000000:.2f}Cr (+{winner['gain_percent']:.1f}%)")

def render_elite_welcome():
    """Elite welcome screen"""
    st.markdown("""
        <div style="text-align: center; padding: 4rem;">
            <h1 style="color: #f59e0b; font-size: 4rem;">👑</h1>
            <h2>Welcome to Elite Trading Championship</h2>
            <p style="font-size: 1.2rem; opacity: 0.8;">
                ₹1 Crore Starting Capital • Extreme Volatility • Professional Algorithms
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("🎯 Championship Structure")
        st.write("""
        **Level 1:** 50% gain in 10 minutes  
        **Level 2:** 120% gain in 8 minutes  
        **Level 3:** 200% gain in 6 minutes
        """)
    
    with col2:
        st.subheader("💎 Elite Features")
        st.write("""
        • 90+ Trading Instruments  
        • 4 AI Trading Algorithms  
        • High-Frequency Simulation  
        • Real-time Market Dynamics
        """)
    
    with col3:
        st.subheader("🏆 Prize Tiers")
        st.write("""
        **Champion:** Elite Recognition  
        **Top 3:** Professional Credibility  
        **Qualifiers:** Certificate of Excellence
        """)

# --- Main Application ---
def main():
    # Initialize elite session state
    if 'elite_initialized' not in st.session_state:
        st.session_state.elite_initialized = True
        st.session_state.last_refresh = time.time()
        st.session_state.elite_admin = False
    
    game_state = get_game_state()
    
    # Update game state
    update_game_state()
    
    # Initialize prices
    if not game_state.base_prices:
        initialize_elite_prices()
    
    # Run algo traders
    run_elite_algo_traders(game_state.prices)
    
    # High-frequency price simulation
    if game_state.game_status.startswith("Level"):
        game_state.prices = simulate_elite_prices(game_state.prices)
        game_state.price_history.append(game_state.prices.copy())
        if len(game_state.price_history) > 150:
            game_state.price_history.pop(0)
    
    # Render elite interface
    render_elite_header()
    render_elite_sidebar()
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_elite_trading_interface(game_state.prices)
        render_elite_leaderboard(game_state.prices)
    
    with col2:
        render_level_status()
        
        # Algorithm status
        st.subheader("🤖 Elite Algorithms")
        for algo_name, algo_config in game_state.algo_traders.items():
            status = "🟢 ACTIVE" if algo_config['active'] else "🔴 INACTIVE"
            st.write(f"**{algo_name}** - {status}")
            st.caption(algo_config['description'])
    
    # Ultra high-speed refresh
    current_time = time.time()
    refresh_interval = 0.5  # 500ms for elite trading
    if current_time - st.session_state.last_refresh > refresh_interval:
        st.session_state.last_refresh = current_time
        st.rerun()

if __name__ == "__main__":
    main()

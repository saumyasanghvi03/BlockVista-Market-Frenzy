# ======================= Expo Game: BlockVista Market Frenzy - Complete Enhanced Version ======================

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
ADMIN_PASSWORD = "100370"

# --- Level Configuration ---
LEVEL_CONFIG = {
    1: {
        "name": "Beginner Arena",
        "duration_minutes": 5,
        "volatility": 0.8,
        "margin_requirement": 0.15,
        "winning_criteria": {
            "min_return": 0.03,
            "max_drawdown": -0.10,
            "diversification": 2,
        },
        "description": "Learn the basics with stable markets"
    },
    2: {
        "name": "Professional Challenge", 
        "duration_minutes": 7,
        "volatility": 1.5,
        "margin_requirement": 0.25,
        "winning_criteria": {
            "min_return": 0.08,
            "max_drawdown": -0.15,
            "sharpe_ratio": 0.5,
            "diversification": 4,
        },
        "description": "Increased volatility and stricter requirements"
    },
    3: {
        "name": "Expert Gauntlet",
        "duration_minutes": 10,
        "volatility": 2.5,
        "margin_requirement": 0.35,
        "winning_criteria": {
            "min_return": 0.15,
            "max_drawdown": -0.20,
            "sharpe_ratio": 1.0,
            "diversification": 6,
            "min_trades": 3,
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

# --- Enhanced Game State Management ---
class GameState:
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
        self.price_history = []
        self.transactions = {}
        self.market_sentiment = {s: 0 for s in ALL_SYMBOLS}
        self.liquidity = {s: random.uniform(0.5, 1.0) for s in ALL_SYMBOLS}
        self.event_active = False
        self.event_type = None
        self.event_target_symbol = None
        self.event_end = 0
        self.volatility_multiplier = LEVEL_CONFIG[1]["volatility"]
        self.news_feed = []
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
        self.animation_state = {
            "sentiment_pulse": 0,
            "leaderboard_highlight": 0,
            "market_pulse": 0
        }

    def reset(self):
        current_level = self.current_level
        level_winners = self.level_winners.copy()
        self.__init__()
        self.current_level = current_level
        self.level_winners = level_winners
        level_config = LEVEL_CONFIG[self.current_level]
        self.volatility_multiplier = level_config["volatility"]
        self.current_margin_requirement = level_config["margin_requirement"]
        self.level_duration_seconds = level_config["duration_minutes"] * 60

    def advance_to_next_level(self):
        if self.current_level >= 3:
            return False
            
        next_level = self.current_level + 1
        current_winners = self.evaluate_level_winners()
        
        if not current_winners:
            st.error("No winners qualified for next level! Game cannot continue.")
            return False
            
        self.level_winners[self.current_level] = current_winners
        old_players = self.players.copy()
        self.players = {}
        
        for player_name in current_winners:
            old_player_data = old_players.get(player_name)
            if old_player_data:
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
                    "pending_orders": [],
                    "algo": old_player_data.get('algo', 'Off'),
                    "custom_algos": old_player_data.get('custom_algos', {}).copy(),
                    "slippage_multiplier": player_type_config['slippage_multiplier'],
                    "value_history": [total_start_value],
                    "trade_timestamps": [],
                    "level_start_value": total_start_value
                }
        
        self.current_level = next_level
        level_config = LEVEL_CONFIG[next_level]
        self.volatility_multiplier = level_config["volatility"]
        self.current_margin_requirement = level_config["margin_requirement"]
        self.level_duration_seconds = level_config["duration_minutes"] * 60
        
        self.game_status = "Running"
        self.level_start_time = time.time()
        self.futures_expiry_time = time.time() + (self.level_duration_seconds / 2)
        self.auto_square_off_complete = False
        self.closing_warning_triggered = False
        self.futures_settled = False
        
        return True

    def evaluate_level_winners(self):
        winners = []
        level_criteria = LEVEL_CONFIG[self.current_level]["winning_criteria"]
        
        for player_name, player_data in self.players.items():
            if self.check_player_qualifies(player_data, level_criteria):
                winners.append(player_name)
                
        return winners

    def check_player_qualifies(self, player_data, criteria):
        try:
            holdings_value = sum(self.prices.get(symbol, 0) * qty for symbol, qty in player_data['holdings'].items())
            total_value = player_data['capital'] + holdings_value
            
            level_start_value = player_data.get('level_start_value', 
                PLAYER_TYPES[player_data['mode']]['initial_capital_multiplier'] * INITIAL_CAPITAL)
            level_return = (total_value - level_start_value) / level_start_value
            
            if level_return < criteria.get("min_return", -1.0):
                return False
                
            if 'value_history' in player_data and len(player_data['value_history']) > 1:
                peak = max(player_data['value_history'])
                current = player_data['value_history'][-1]
                drawdown = (current - peak) / peak
                if drawdown < criteria.get("max_drawdown", -1.0):
                    return False
            
            if "sharpe_ratio" in criteria:
                sharpe = calculate_sharpe_ratio(player_data.get('value_history', []))
                if sharpe < criteria["sharpe_ratio"]:
                    return False
            
            if "diversification" in criteria:
                unique_assets = len([s for s in player_data['holdings'].keys() if player_data['holdings'].get(s, 0) != 0])
                if unique_assets < criteria["diversification"]:
                    return False
            
            if "min_trades" in criteria:
                trade_count = len(self.transactions.get(player_data['name'], []))
                if trade_count < criteria["min_trades"]:
                    return False
                    
            return True
            
        except Exception:
            return False

    def calculate_performance_multiplier(self, player_data):
        try:
            holdings_value = sum(self.prices.get(symbol, 0) * qty for symbol, qty in player_data['holdings'].items())
            total_value = player_data['capital'] + holdings_value
            
            level_start_value = player_data.get('level_start_value', 
                PLAYER_TYPES[player_data['mode']]['initial_capital_multiplier'] * INITIAL_CAPITAL)
            level_return = (total_value - level_start_value) / level_start_value
            
            if level_return <= 0:
                return 0.8
            elif level_return <= 0.1:
                return 1.0 + level_return
            elif level_return <= 0.25:
                return 1.1 + (level_return - 0.1) * 0.5
            else:
                return 1.2
        except:
            return 1.0

@st.cache_resource
def get_game_state():
    return GameState()

# --- Sound Effects ---
def play_sound(sound_type):
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
    elif sound_type == 'level_up':
        js = """
        <script>
            if (typeof Tone !== 'undefined') {
                const synth = new Tone.Synth().toDestination();
                const now = Tone.now();
                synth.triggerAttackRelease("E5", "8n", now);
                synth.triggerAttackRelease("G5", "8n", now + 0.1);
                synth.triggerAttackRelease("C6", "8n", now + 0.2);
            }
        </script>
        """
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

# --- Data Fetching & Market Simulation ---
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
        pass
    elif event_type == "Stock Split" and target_symbol:
        adjusted_prices[target_symbol] /= 2
        for player in game_state.players.values():
            if target_symbol in player['holdings']:
                player['holdings'][target_symbol] *= 2
    elif event_type == "Dividend" and target_symbol:
        dividend_per_share = adjusted_prices[target_symbol] * 0.01
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

# --- Enhanced Animation Functions ---
def render_animated_sentiment_meter():
    game_state = get_game_state()
    sentiments = [s for s in game_state.market_sentiment.values() if s != 0]
    if not sentiments:
        overall_sentiment = 0
    else:
        overall_sentiment = np.mean(sentiments)
    
    normalized_sentiment = np.clip((overall_sentiment + 5) * 10, 0, 100)
    
    game_state.animation_state["sentiment_pulse"] = (game_state.animation_state["sentiment_pulse"] + 1) % 100
    
    st.markdown("##### 📊 Market Sentiment Analysis")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = normalized_sentiment,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Market Pulse", 'font': {'size': 20}},
            delta = {'reference': 50, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'red'},
                    {'range': [30, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'green'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': normalized_sentiment}
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            font={'color': "darkblue", 'family': "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        pulse_effect = "🔴" if normalized_sentiment < 30 else "⚫"
        st.markdown(f"<h3 style='text-align: center; color: {'red' if normalized_sentiment < 30 else 'gray'};'>{pulse_effect} FEAR</h3>", 
                   unsafe_allow_html=True)
        if normalized_sentiment < 30:
            st.markdown("<p style='text-align: center; color: red; font-size: 12px;'>Market in panic mode</p>", 
                       unsafe_allow_html=True)
    
    with col2:
        neutral_effect = "🟡" if 30 <= normalized_sentiment <= 70 else "⚫"
        st.markdown(f"<h3 style='text-align: center; color: {'orange' if 30 <= normalized_sentiment <= 70 else 'gray'};'>{neutral_effect} NEUTRAL</h3>", 
                   unsafe_allow_html=True)
        if 30 <= normalized_sentiment <= 70:
            st.markdown("<p style='text-align: center; color: orange; font-size: 12px;'>Market balanced</p>", 
                       unsafe_allow_html=True)
    
    with col3:
        greed_effect = "🟢" if normalized_sentiment > 70 else "⚫"
        st.markdown(f"<h3 style='text-align: center; color: {'green' if normalized_sentiment > 70 else 'gray'};'>{greed_effect} GREED</h3>", 
                   unsafe_allow_html=True)
        if normalized_sentiment > 70:
            st.markdown("<p style='text-align: center; color: green; font-size: 12px;'>Market optimistic</p>", 
                       unsafe_allow_html=True)

def render_animated_leaderboard(prices):
    game_state = get_game_state()
    lb = []
    
    for pname, pdata in game_state.players.items():
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in pdata['holdings'].items())
        total_value = pdata['capital'] + holdings_value
        
        level_start_value = pdata.get('level_start_value', 
            PLAYER_TYPES[pdata['mode']]['initial_capital_multiplier'] * INITIAL_CAPITAL)
        level_return = (total_value - level_start_value) / level_start_value if level_start_value > 0 else 0
        
        sharpe_ratio = calculate_sharpe_ratio(pdata.get('value_history', []))
        
        current_level_config = LEVEL_CONFIG.get(game_state.current_level, LEVEL_CONFIG[1])
        qualifies = game_state.check_player_qualifies(pdata, 
            current_level_config["winning_criteria"])
        
        lb.append((pname, pdata['mode'], total_value, pdata['pnl'], 
                 level_return, sharpe_ratio, qualifies))
    
    if lb:
        lb_df = pd.DataFrame(lb, columns=["Player", "Type", "Portfolio Value", "Total P&L", 
                                         "Level Return", "Sharpe Ratio", "Qualifies"])
        lb_df = lb_df.sort_values("Portfolio Value", ascending=False).reset_index(drop=True)
        
        game_state.animation_state["leaderboard_highlight"] = (game_state.animation_state["leaderboard_highlight"] + 1) % len(lb_df)
        
        def style_leaderboard(row):
            styles = []
            highlight_idx = game_state.animation_state["leaderboard_highlight"]
            
            for i, val in enumerate(row):
                if row.name == highlight_idx and game_state.game_status == "Running":
                    styles.append('background: linear-gradient(45deg, #ffd700, #ffed4e); font-weight: bold;')
                elif row['Qualifies']:
                    styles.append('background: linear-gradient(45deg, #90EE90, #C1FFC1);')
                elif row.name == 0:
                    styles.append('background: linear-gradient(45deg, #FFD700, #FFF8DC); font-weight: bold;')
                else:
                    styles.append('')
            return styles
        
        st.markdown("##### 🏆 Live Player Standings")
        
        if game_state.game_status == "Running":
            emojis = ["🚀", "💹", "📈", "💰", "🎯"]
            current_emoji = emojis[game_state.animation_state["leaderboard_highlight"] % len(emojis)]
            st.markdown(f"<p style='text-align: center; font-size: 20px;'>{current_emoji} Live Trading Active {current_emoji}</p>", 
                       unsafe_allow_html=True)
        
        styled_df = lb_df.style.format({
            "Portfolio Value": format_indian_currency,
            "Total P&L": format_indian_currency, 
            "Level Return": "{:.2%}",
            "Sharpe Ratio": "{:.2f}"
        }).apply(style_leaderboard, axis=1)
        
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        if game_state.game_status == "Finished":
            winners = [row['Player'] for _, row in lb_df.iterrows() if row['Qualifies']]
            if winners:
                st.success(f"🎉 Level {game_state.current_level} Complete! 🎉")
                st.balloons()
                
                if game_state.current_level == 3:
                    overall_winner = lb_df.iloc[0]
                    st.success(f"🏆 **TOURNAMENT CHAMPION: {overall_winner['Player']}** 🏆")
                    
                    cols = st.columns(3)
                    cols[0].metric("Final Portfolio", format_indian_currency(overall_winner['Portfolio Value']))
                    cols[1].metric("Total Return", f"{overall_winner['Level Return']:.2%}")
                    cols[2].metric("Sharpe Ratio", f"{overall_winner['Sharpe Ratio']:.2f}")
            else:
                st.error("❌ No players qualified for the next level!")

def render_animated_market_overview():
    game_state = get_game_state()
    
    st.markdown("##### 🌐 Live Market Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if len(game_state.price_history) > 1:
        current_prices = game_state.prices
        prev_prices = game_state.price_history[-2] if len(game_state.price_history) > 1 else current_prices
        
        price_changes = []
        for symbol in NIFTY50_SYMBOLS[:5]:
            if symbol in current_prices and symbol in prev_prices:
                change = (current_prices[symbol] - prev_prices[symbol]) / prev_prices[symbol]
                price_changes.append(change)
        
        if price_changes:
            avg_change = np.mean(price_changes) * 100
            rising_stocks = len([c for c in price_changes if c > 0])
            market_health = min(100, max(0, 50 + (avg_change * 10)))
        else:
            avg_change = 0
            rising_stocks = 0
            market_health = 50
    else:
        avg_change = 0
        rising_stocks = 0
        market_health = 50
    
    with col1:
        st.metric(
            "Market Health", 
            f"{market_health:.0f}%",
            delta=f"{avg_change:+.2f}%",
            delta_color="normal" if avg_change >= 0 else "inverse"
        )
    
    with col2:
        st.metric("Rising Stocks", f"{rising_stocks}/5", "Bullish" if rising_stocks >= 3 else "Bearish")
    
    with col3:
        volatility_color = "red" if game_state.volatility_multiplier > 2 else "orange" if game_state.volatility_multiplier > 1 else "green"
        st.markdown(f"<h3 style='color: {volatility_color};'>Volatility: {game_state.volatility_multiplier:.1f}x</h3>", 
                   unsafe_allow_html=True)
    
    with col4:
        active_players = len(game_state.players)
        st.metric("Active Traders", active_players, "Trading" if active_players > 0 else "Waiting")

def inject_css_animations():
    st.markdown("""
    <style>
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.02); }
        100% { transform: scale(1); }
    }
    
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(255, 215, 0, 0.5); }
        50% { box-shadow: 0 0 20px rgba(255, 215, 0, 0.8); }
        100% { box-shadow: 0 0 5px rgba(255, 215, 0, 0.5); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    .glow-animation {
        animation: glow 2s infinite;
    }
    
    .market-up {
        background: linear-gradient(45deg, #90EE90, #C1FFC1) !important;
        transition: all 0.3s ease;
    }
    
    .market-down {
        background: linear-gradient(45deg, #FFB6C1, #FFCCCB) !important;
        transition: all 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)

# --- Enhanced Global Views with Animations ---
def render_global_views(prices, is_admin=False):
    with st.container(border=True):
        st.markdown("### 🌍 Global Market View")
        
        render_animated_market_overview()
        
        st.markdown("---")
        
        render_animated_sentiment_meter()
        
        st.markdown("---")
        
        st.subheader("📰 Live News Feed")
        game_state = get_game_state()
        news_feed = getattr(game_state, 'news_feed', [])
        
        if news_feed:
            for i, news in enumerate(news_feed):
                if i == 0 and "📢" in news:
                    st.markdown(f"""
                    <div style="background: linear-gradient(45deg, #FF6B6B, #FFE66D); 
                                padding: 15px; 
                                border-radius: 10px; 
                                margin: 10px 0;
                                border-left: 5px solid #FF6B6B;
                                animation: pulse 2s infinite;">
                        <strong>🚨 BREAKING:</strong> {news.split(' - ')[1] if ' - ' in news else news}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info(news)
        else:
            st.info("📡 Scanning market news...")
        
        st.markdown("---")
        
        render_animated_leaderboard(prices)
        
        if is_admin:
            st.markdown("---")
            st.subheader("📊 Live Player Performance")
            render_admin_performance_chart()

        st.markdown("---")
        st.subheader("📈 Live Market Feed")
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
        df = pd.DataFrame(chart_data)
        st.line_chart(df)
    else:
        st.info("No trading activity yet to display.")

def render_live_market_table(prices):
    game_state = get_game_state()
    prices_df = pd.DataFrame(prices.items(), columns=['Symbol', 'Price'])
    
    if len(game_state.price_history) >= 2:
        prev_prices = game_state.price_history[-2]
        prices_df['prev_price'] = prices_df['Symbol'].map(prev_prices).fillna(prices_df['Price'])
        prices_df['Change'] = prices_df['Price'] - prices_df['prev_price']
    else:
        prices_df['Change'] = 0.0
    prices_df.drop(columns=['prev_price'], inplace=True, errors='ignore')

    all_trades = [[player] + t for player, transactions in game_state.transactions.items() for t in transactions]
    if all_trades:
        feed_df = pd.DataFrame(all_trades, columns=["Player", "Time", "Action", "Symbol", "Qty", "Trade Price", "Total"])
        last_trades = feed_df.sort_values('Time').groupby('Symbol').last()
        last_trades['Last Order'] = last_trades.apply(lambda r: f"{r['Player']} {r['Action']} {r['Qty']} @ {format_indian_currency(r['Trade Price'])}", axis=1)
        prices_df = pd.merge(prices_df, last_trades[['Last Order']], on='Symbol', how='left')
    else:
        prices_df['Last Order'] = '-'
    prices_df.fillna({'Last Order': '-'}, inplace=True)
    
    st.dataframe(prices_df.style.apply(lambda x: ['color: green' if v > 0 else 'color: red' if v < 0 else '' for v in x], subset=['Change']).format({'Price': format_indian_currency, 'Change': lambda v: f"{format_indian_currency(v) if v != 0 else '-'}"}), use_container_width=True, hide_index=True)

# --- UI Functions ---
def render_sidebar():
    game_state = get_game_state()
    
    if not hasattr(game_state, 'current_level'):
        game_state.current_level = 1
        
    current_level_config = LEVEL_CONFIG.get(game_state.current_level, LEVEL_CONFIG[1])
    
    if 'player' not in st.query_params:
        st.sidebar.title("📝 Game Entry")
        player_name = st.sidebar.text_input("Enter Name", key="name_input")
        
        player_types = list(PLAYER_TYPES.keys())
        mode = st.sidebar.radio("Select Player Type", player_types, 
                               format_func=lambda x: f"{x} - {PLAYER_TYPES[x]['description']}", 
                               key="mode_select")
        
        st.sidebar.info(f"**Current Level: {current_level_config['name']}**\n\n"
                       f"Duration: {current_level_config['duration_minutes']} min\n"
                       f"Volatility: {current_level_config['volatility']}x\n"
                       f"Margin: {current_level_config['margin_requirement']*100}%")
        
        if st.sidebar.button("Join Game", type="primary"):
            if player_name and player_name.strip() and player_name not in game_state.players:
                player_config = PLAYER_TYPES[mode]
                starting_capital = INITIAL_CAPITAL * player_config['initial_capital_multiplier']
                
                game_state.players[player_name] = {
                    "name": player_name, 
                    "mode": mode, 
                    "capital": starting_capital, 
                    "holdings": {}, 
                    "pnl": 0, 
                    "leverage": 1.0, 
                    "margin_calls": 0, 
                    "pending_orders": [], 
                    "algo": "Off", 
                    "custom_algos": {},
                    "slippage_multiplier": player_config['slippage_multiplier'],
                    "value_history": [starting_capital],
                    "trade_timestamps": [],
                    "level_start_value": starting_capital
                }
                game_state.transactions[player_name] = []
                st.query_params["player"] = player_name
                st.rerun()
            else: 
                st.sidebar.error("Name is invalid or already taken!")
    else:
        current_player = st.query_params['player']
        player_data = game_state.players.get(current_player, {})
        st.sidebar.success(f"Logged in as {current_player}")
        st.sidebar.info(f"Type: {player_data.get('mode', 'N/A')}\n"
                       f"Level: {game_state.current_level}")
        
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
        
        st.sidebar.subheader("Level Management")
        current_level = game_state.current_level
        
        if game_state.game_status != "Running":
            new_level = st.sidebar.selectbox("Set Level", [1, 2, 3], 
                                           index=current_level-1,
                                           format_func=lambda x: f"Level {x}: {LEVEL_CONFIG[x]['name']}")
            if new_level != current_level:
                game_state.current_level = new_level
                level_config = LEVEL_CONFIG[new_level]
                game_state.volatility_multiplier = level_config["volatility"]
                game_state.current_margin_requirement = level_config["margin_requirement"]
                game_state.level_duration_seconds = level_config["duration_minutes"] * 60
                st.sidebar.success(f"Level set to {level_config['name']}")
        
        level_criteria = current_level_config["winning_criteria"]
        st.sidebar.markdown("**Winning Criteria:**")
        for criterion, value in level_criteria.items():
            if criterion == "min_return":
                st.sidebar.write(f"• Min Return: {value:.1%}")
            elif criterion == "max_drawdown":
                st.sidebar.write(f"• Max Drawdown: {value:.1%}")
            elif criterion == "sharpe_ratio":
                st.sidebar.write(f"• Min Sharpe: {value}")
            elif criterion == "diversification":
                st.sidebar.write(f"• Min Assets: {value}")
            elif criterion == "min_trades":
                st.sidebar.write(f"• Min Trades: {value}")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            if st.button("▶️ Start Level", type="primary"):
                if game_state.players:
                    game_state.game_status = "Running"
                    game_state.level_start_time = time.time()
                    game_state.futures_expiry_time = time.time() + (game_state.level_duration_seconds / 2)
                    st.toast(f"Level {game_state.current_level} Started!", icon="🎉")
                    st.rerun()
                else: 
                    st.sidebar.warning("Add at least one player to start.")
        
        with col2:
            if st.button("⏸️ Stop Game"):
                game_state.game_status = "Stopped"
                st.toast("Game Paused!", icon="⏸️")
                st.rerun()
        
        if st.button("🔄 Reset Game"):
            game_state.reset()
            st.toast("Game has been reset.", icon="🔄")
            st.rerun()
            
        if game_state.game_status == "Finished" and game_state.current_level < 3:
            if st.button("🚀 Advance to Next Level"):
                if game_state.advance_to_next_level():
                    st.toast(f"Advanced to Level {game_state.current_level}!", icon="🚀")
                    st.rerun()
                else:
                    st.error("Cannot advance - no qualified players!")
        
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
                    st.toast(f"News Published!", icon="📰")
                    announce_news(headline)
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
            
    elif password: 
        st.sidebar.error("Incorrect Password")

def render_main_interface(prices):
    game_state = get_game_state()
    
    if not hasattr(game_state, 'current_level'):
        game_state.current_level = 1
        
    current_level_config = LEVEL_CONFIG.get(game_state.current_level, LEVEL_CONFIG[1])
    
    inject_css_animations()
    
    st.title(f"📈 {GAME_NAME}")
    st.components.v1.html('<script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.7.77/Tone.js"></script>', height=0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader(f"🎮 Level {game_state.current_level}: {current_level_config['name']}")
        st.caption(current_level_config['description'])
    
    with col2:
        if game_state.game_status == "Running":
            remaining_time = max(0, game_state.level_duration_seconds - int(time.time() - game_state.level_start_time))
            st.metric("⏰ Time Remaining", f"{remaining_time // 60:02d}:{remaining_time % 60:02d}")
            progress = 1 - (remaining_time / game_state.level_duration_seconds)
            st.progress(progress)
            
            if remaining_time <= 60:
                st.markdown('<div class="pulse-animation">🕒 HURRY UP!</div>', unsafe_allow_html=True)
        else:
            status_color = "green" if game_state.game_status == "Finished" else "orange"
            st.markdown(f"<h2 style='color: {status_color}; text-align: center;'>{game_state.game_status}</h2>", 
                       unsafe_allow_html=True)
    
    with col3:
        volatility_emoji = "🌪️" if current_level_config["volatility"] > 2 else "💨" if current_level_config["volatility"] > 1 else "🌊"
        st.metric(f"{volatility_emoji} Volatility", f"{current_level_config['volatility']}x")
        st.metric("📊 Margin Req", f"{current_level_config['margin_requirement']*100}%")

    if 'player' in st.query_params:
        col1, col2 = st.columns([1, 1])
        with col1: 
            render_trade_execution_panel(prices)
        with col2: 
            render_global_views(prices)
    else:
        st.info("🎯 Welcome to BlockVista Market Frenzy! Please join the game from the sidebar to start trading.")
        render_global_views(prices)

def render_trade_execution_panel(prices):
    game_state = get_game_state()
    
    with st.container(border=True):
        st.subheader("Trade Execution Panel")
        acting_player = st.query_params.get("player")
        if not acting_player or acting_player not in game_state.players:
            st.warning("Please join the game to access your trading terminal.")
            return
        
        player = game_state.players[acting_player]
        player_config = PLAYER_TYPES[player['mode']]
        
        st.markdown(f"**{acting_player}'s Terminal**")
        st.caption(f"Type: {player['mode']} | {player_config['description']}")
        
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        level_start_value = player.get('level_start_value', total_value)
        level_return = (total_value - level_start_value) / level_start_value if level_start_value > 0 else 0
        
        current_level_config = LEVEL_CONFIG.get(game_state.current_level, LEVEL_CONFIG[1])
        qualifies = game_state.check_player_qualifies(player, 
            current_level_config["winning_criteria"])
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Cash", format_indian_currency(player['capital']))
        col2.metric("Portfolio Value", format_indian_currency(total_value))
        col3.metric("Level Return", f"{level_return:.2%}", 
                   delta=f"{level_return:.2%}", delta_color="normal")
        col4.metric("Qualifies", "✅" if qualifies else "❌")
        
        if not qualifies and game_state.game_status == "Running":
            st.warning(f"⚠️ You don't currently meet Level {game_state.current_level} criteria!")
        
        tabs = ["👨‍💻 Trade Terminal", "🤖 Algo Trading", "📂 Transaction History", "📊 Strategy & Insights"]
        tab1, tab2, tab3, tab4 = st.tabs(tabs)
        is_trade_disabled = game_state.game_status != "Running"

        with tab1: 
            render_trade_interface(acting_player, player, prices, is_trade_disabled)
        with tab2: 
            render_algo_trading_tab(acting_player, player, is_trade_disabled)
        with tab3: 
            render_transaction_history(acting_player)
        with tab4: 
            render_strategy_tab(player)

def render_trade_interface(player_name, player, prices, disabled):
    order_type_tabs = ["Market", "Limit", "Stop-Loss"]
    market_tab, limit_tab, stop_loss_tab = st.tabs(order_type_tabs)

    with market_tab:
        render_market_order_ui(player_name, player, prices, disabled)

    with limit_tab:
        st.write("Set a price to automatically buy or sell an asset.")

    with stop_loss_tab:
        st.write("Set a price to automatically sell an asset if it drops, to limit losses.")

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
        if asset_type == "Stock": 
            symbol_choice = st.selectbox("Stock", [s.replace('.NS', '') for s in NIFTY50_SYMBOLS], key=f"market_stock_{player_name}", disabled=disabled) + '.NS'
        elif asset_type == "Crypto": 
            symbol_choice = st.selectbox("Cryptocurrency", CRYPTO_SYMBOLS, key=f"market_crypto_{player_name}", disabled=disabled)
        elif asset_type == "Gold": 
            symbol_choice = GOLD_SYMBOL
        elif asset_type == "Futures": 
            symbol_choice = st.selectbox("Futures", FUTURES_SYMBOLS, key=f"market_futures_{player_name}", disabled=disabled)
        elif asset_type == "Leveraged ETF": 
            symbol_choice = st.selectbox("Leveraged ETF", LEVERAGED_ETFS, key=f"market_letf_{player_name}", disabled=disabled)
        else: 
            symbol_choice = st.selectbox("Option", OPTION_SYMBOLS, key=f"market_option_{player_name}", disabled=disabled)
    with col2:
        qty = st.number_input("Quantity", min_value=1, step=1, value=1, key=f"market_qty_{player_name}", disabled=disabled)
    
    mid_price = prices.get(symbol_choice, 0)
    ask_price = mid_price * (1 + get_game_state().bid_ask_spread / 2)
    bid_price = mid_price * (1 - get_game_state().bid_ask_spread / 2)
    st.info(f"Bid: {format_indian_currency(bid_price)} | Ask: {format_indian_currency(ask_price)}")

    b1, b2, b3 = st.columns(3)
    if b1.button(f"Buy {qty} at Ask", key=f"buy_{player_name}", use_container_width=True, disabled=disabled, type="primary"): 
        if execute_trade(player_name, player, "Buy", symbol_choice, qty, prices): 
            play_sound('success')
        else: 
            play_sound('error')
        st.rerun()
    if b2.button(f"Sell {qty} at Bid", key=f"sell_{player_name}", use_container_width=True, disabled=disabled): 
        if execute_trade(player_name, player, "Sell", symbol_choice, qty, prices): 
            play_sound('success')
        else: 
            play_sound('error')
        st.rerun()
    if b3.button(f"Short {qty} at Bid", key=f"short_{player_name}", use_container_width=True, disabled=disabled): 
        if execute_trade(player_name, player, "Short", symbol_choice, qty, prices): 
            play_sound('success')
        else: 
            play_sound('error')
        st.rerun()

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
    custom_strats = list(player.get('custom_algos', {}).keys())
    all_strats = default_strats + custom_strats
    active_algo = player.get('algo', 'Off')
    player['algo'] = st.selectbox("Choose Strategy", all_strats, index=all_strats.index(active_algo) if active_algo in all_strats else 0, disabled=disabled, key=f"algo_{player_name}")
    
    if player['algo'] in default_strats and player['algo'] != 'Off': 
        if player['algo'] == "Momentum Trader": 
            st.info("This bot buys assets that have risen in price and sells those that have fallen, betting that recent trends will continue.")
        elif player['algo'] == "Mean Reversion": 
            st.info("This bot buys assets that have recently fallen and sells those that have risen, betting on a return to their average price.")
        elif player['algo'] == "Volatility Breakout": 
            st.info("This bot identifies assets making a significant daily price move (up or down) and trades in the same direction, aiming to capitalize on strong momentum.")
        elif player['algo'] == "Value Investor": 
            st.info("This bot looks for assets that have dropped significantly over the past month and buys them, operating on the principle of buying undervalued assets for a potential long-term recovery.")

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
                st.success(f"Custom strategy '{algo_name}' saved!")
                st.rerun()
            else: 
                st.error("Strategy name cannot be empty.")

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
    else: 
        st.info("No transactions recorded.")

def render_strategy_tab(player):
    st.subheader("📊 Strategy & Insights")
    tab1, tab2, tab3 = st.tabs(["Performance Chart", "Technical Analysis (SMA)", "Portfolio Optimizer"])
    with tab1:
        st.markdown("##### Portfolio Value Over Time")
        if len(player.get('value_history', [])) > 1:
            st.line_chart(player['value_history'])
        else:
            st.info("Trade more to see your performance chart.")
    with tab2: 
        render_sma_chart(player['holdings'])
    with tab3: 
        render_optimizer(player['holdings'])

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
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Price'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='20-Day SMA'))
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], mode='lines', name='50-Day SMA'))
        st.plotly_chart(fig, use_container_width=True)
    else: 
        st.warning(f"Could not load historical data for {chart_symbol}.")

def render_optimizer(holdings):
    st.subheader("Portfolio Optimization (Max Sharpe Ratio)")
    if st.button("Optimize My Portfolio"):
        weights, performance = optimize_portfolio(holdings)
        if weights:
            st.success("Optimal weights for max risk-adjusted return:")
            st.json({k: f"{v:.2%}" for k, v in weights.items()})
            if performance: 
                st.write(f"Expected Return: {performance[0]:.2%}, Volatility: {performance[1]:.2%}, Sharpe Ratio: {performance[2]:.2f}")
        else: 
            st.error(performance)

def execute_trade(player_name, player, action, symbol, qty, prices, is_algo=False, order_type="Market"):
    game_state = get_game_state()
    mid_price = prices.get(symbol, 0)
    if mid_price == 0: 
        return False

    if action == "Buy":
        trade_price = mid_price * (1 + game_state.bid_ask_spread / 2)
    else:
        trade_price = mid_price * (1 - game_state.bid_ask_spread / 2)
    
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
        if current_qty > 0 and current_qty >= qty:
            player['capital'] += cost
            player['holdings'][symbol] -= qty
            trade_executed = True
        elif current_qty < 0 and abs(current_qty) >= qty:
            player['capital'] -= cost
            player['holdings'][symbol] += qty
            trade_executed = True
        if trade_executed and player['holdings'][symbol] == 0: 
            del player['holdings'][symbol]
    
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

# --- Main Game Loop Functions ---
def run_game_tick(prices):
    game_state = get_game_state()
    if game_state.game_status != "Running": 
        return prices
    
    if time.time() - game_state.level_start_time >= game_state.level_duration_seconds:
        if game_state.game_status != "Finished":
            play_sound('final_bell')
            game_state.game_status = "Finished"
            auto_square_off_positions(prices)
            game_state.auto_square_off_complete = True
            
            level_winners = game_state.evaluate_level_winners()
            if level_winners:
                st.toast(f"Level {game_state.current_level} complete! {len(level_winners)} players qualified.")
                if level_winners and game_state.current_level < 3:
                    time.sleep(3)
                    if game_state.advance_to_next_level():
                        play_sound('level_up')
                        st.toast(f"Auto-advanced to Level {game_state.current_level}!", icon="🚀")
                        st.rerun()
            else:
                st.toast(f"Level {game_state.current_level} complete! No players qualified for next level.")
    
    for symbol in game_state.market_sentiment:
        game_state.market_sentiment[symbol] *= 0.95 

    if not game_state.event_active and random.random() < 0.07:
        news_item = random.choice(PRE_BUILT_NEWS)
        headline = news_item['headline']
        impact = news_item['impact']
        target_symbol = None

        if "{symbol}" in headline:
            target_symbol = random.choice(NIFTY50_SYMBOLS)
            headline = headline.format(symbol=target_symbol.replace(".NS", ""))
        
        game_state.news_feed.insert(0, f"📢 {time.strftime('%H:%M:%S')} - {headline}")
        if len(game_state.news_feed) > 5: 
            game_state.news_feed.pop()
        
        game_state.event_type = impact
        game_state.event_target_symbol = target_symbol
        game_state.event_active = True
        game_state.event_end = time.time() + random.randint(30, 60)
        st.toast(f"⚡ Market Event!", icon="🎉")
        announce_news(headline)
        
    if game_state.event_active and time.time() >= game_state.event_end:
        game_state.event_active = False
        st.info("Market event has ended.")
        
    if game_state.event_active: 
        if game_state.event_type == 'Volatility Spike':
            prices = {k: v * (1 + random.uniform(-0.01, 0.01) * 2) for k, v in prices.items()}
        else:
            prices = apply_event_adjustment(prices, game_state.event_type, getattr(game_state, 'event_target_symbol', None))
    
    handle_futures_expiry(prices)
    check_margin_calls_and_orders(prices)
    run_algo_strategies(prices)

    for player in game_state.players.values():
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        player['value_history'].append(total_value)
        
    return prices

def auto_square_off_positions(prices):
    game_state = get_game_state()
    st.info("End of level: Auto-squaring off all intraday, futures, and options positions...")
    square_off_assets = NIFTY50_SYMBOLS + FUTURES_SYMBOLS + OPTION_SYMBOLS + LEVERAGED_ETFS
    
    for name, player in game_state.players.items():
        for symbol, qty in list(player['holdings'].items()):
            if symbol in square_off_assets:
                closing_price = prices.get(symbol, 0)
                value = closing_price * qty
                if qty > 0:
                    player['capital'] += value
                    log_transaction(name, "Auto-Squareoff (Sell)", symbol, qty, closing_price, value)
                else:
                    player['capital'] -= value
                    log_transaction(name, "Auto-Squareoff (Buy)", symbol, abs(qty), closing_price, value)
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
        holdings_value = sum(prices.get(symbol, 0) * qty for symbol, qty in player['holdings'].items())
        total_value = player['capital'] + holdings_value
        margin_needed = abs(holdings_value) * game_state.current_margin_requirement

        if total_value < margin_needed and player['holdings']:
            player['margin_calls'] += 1
            st.warning(f"MARGIN CALL for {name}! Liquidating largest position.")
            
            largest_position = max(player['holdings'].items(), key=lambda item: abs(item[1] * prices.get(item[0], 0)), default=(None, 0))
            if largest_position[0]:
                symbol_to_liquidate, qty_to_liquidate = largest_position
                action = "Sell" if qty_to_liquidate > 0 else "Buy"
                execute_trade(name, player, action, symbol_to_liquidate, abs(qty_to_liquidate), prices, order_type="Auto-Liquidation")

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
        
        for i in sorted(orders_to_remove, reverse=True):
            del player['pending_orders'][i]

def run_algo_strategies(prices):
    game_state = get_game_state()
    if len(game_state.price_history) < 2: return
    prev_prices = game_state.price_history[-2]
    
    for name, player in game_state.players.items():
        active_algo = player.get('algo', 'Off')
        if active_algo == 'Off': continue
        
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
                    break
            else:
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
    
    if not game_state.base_real_prices:
        game_state.base_real_prices = get_daily_base_prices()
        st.toast("📊 Fetched daily base market prices.")

    last_prices = game_state.prices if game_state.prices else game_state.base_real_prices
    current_prices = simulate_tick_prices(last_prices)
    prices_with_derivatives = calculate_derived_prices(current_prices)
    final_prices = run_game_tick(prices_with_derivatives)
    
    game_state.prices = final_prices
    
    if not isinstance(game_state.price_history, list): 
        game_state.price_history = []
    game_state.price_history.append(final_prices)
    if len(game_state.price_history) > 10: 
        game_state.price_history.pop(0)
    
    if st.session_state.get('role') == 'admin' and 'player' not in st.query_params:
        st.title(f"👑 {GAME_NAME} - Admin Dashboard")
        current_level_config = LEVEL_CONFIG.get(game_state.current_level, LEVEL_CONFIG[1])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader(f"Level {game_state.current_level}: {current_level_config['name']}")
        with col2:
            if game_state.game_status == "Running":
                remaining_time = max(0, game_state.level_duration_seconds - int(time.time() - game_state.level_start_time))
                st.metric("Time Remaining", f"{remaining_time // 60:02d}:{remaining_time % 60:02d}")
            else:
                st.metric("Status", game_state.game_status)
        with col3:
            st.metric("Active Players", len(game_state.players))

        render_global_views(final_prices, is_admin=True)
    else:
        render_main_interface(final_prices)
    
    if game_state.game_status == "Running": 
        time.sleep(1)
        st.rerun()
    else:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()

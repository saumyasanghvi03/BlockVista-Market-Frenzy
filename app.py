# ======================= Expo Game: BlockVista Market Frenzy - Enhanced with Animations ======================

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

# --- Enhanced Animation Functions ---
def render_animated_sentiment_meter():
    game_state = get_game_state()
    sentiments = [s for s in game_state.market_sentiment.values() if s != 0]
    if not sentiments:
        overall_sentiment = 0
    else:
        overall_sentiment = np.mean(sentiments)
    
    normalized_sentiment = np.clip((overall_sentiment + 5) * 10, 0, 100)
    
    # Update animation state
    game_state.animation_state["sentiment_pulse"] = (game_state.animation_state["sentiment_pulse"] + 1) % 100
    
    st.markdown("##### 📊 Market Sentiment Analysis")
    
    # Create animated sentiment gauge
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Animated sentiment gauge
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
    
    # Sentiment indicators with animations
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
        
        # Add animation highlight
        game_state.animation_state["leaderboard_highlight"] = (game_state.animation_state["leaderboard_highlight"] + 1) % len(lb_df)
        
        # Enhanced styling with animations
        def style_leaderboard(row):
            styles = []
            highlight_idx = game_state.animation_state["leaderboard_highlight"]
            
            for i, val in enumerate(row):
                if row.name == highlight_idx and game_state.game_status == "Running":
                    styles.append('background: linear-gradient(45deg, #ffd700, #ffed4e); font-weight: bold;')
                elif row['Qualifies']:
                    styles.append('background: linear-gradient(45deg, #90EE90, #C1FFC1);')
                elif row.name == 0:  # Top player
                    styles.append('background: linear-gradient(45deg, #FFD700, #FFF8DC); font-weight: bold;')
                else:
                    styles.append('')
            return styles
        
        st.markdown("##### 🏆 Live Player Standings")
        
        # Add some animated emojis
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
        
        # Level completion message with animations
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
    
    # Market health indicators
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate market health metrics
    if len(game_state.price_history) > 1:
        current_prices = game_state.prices
        prev_prices = game_state.price_history[-2] if len(game_state.price_history) > 1 else current_prices
        
        price_changes = []
        for symbol in NIFTY50_SYMBOLS[:5]:  # Sample of major stocks
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

# --- Enhanced Global Views with Animations ---
def render_global_views(prices, is_admin=False):
    with st.container(border=True):
        st.markdown("### 🌍 Global Market View")
        
        # Animated Market Overview
        render_animated_market_overview()
        
        st.markdown("---")
        
        # Enhanced Market Sentiment with Animation
        render_animated_sentiment_meter()
        
        st.markdown("---")
        
        # Live News Feed with animations
        st.subheader("📰 Live News Feed")
        game_state = get_game_state()
        news_feed = getattr(game_state, 'news_feed', [])
        
        if news_feed:
            for i, news in enumerate(news_feed):
                # Add pulse animation to latest news
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
        
        # Enhanced Live Player Standings with Animation
        render_animated_leaderboard(prices)
        
        if is_admin:
            st.markdown("---")
            st.subheader("📊 Live Player Performance")
            render_admin_performance_chart()

        st.markdown("---")
        st.subheader("📈 Live Market Feed")
        render_live_market_table(prices)

# --- CSS Animations Injection ---
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
    
    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    .glow-animation {
        animation: glow 2s infinite;
    }
    
    .slide-in {
        animation: slideIn 0.5s ease-out;
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

# --- Modified Main Interface (Removed duplicate admin dashboard) ---
def render_main_interface(prices):
    game_state = get_game_state()
    
    if not hasattr(game_state, 'current_level'):
        game_state.current_level = 1
        
    current_level_config = LEVEL_CONFIG.get(game_state.current_level, LEVEL_CONFIG[1])
    
    # Inject CSS animations
    inject_css_animations()
    
    st.title(f"📈 {GAME_NAME}")
    st.components.v1.html('<script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.7.77/Tone.js"></script>', height=0)

    # Level information header with animations
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
            
            # Add pulse effect when time is running low
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

    # Only show player interface for players, not admins
    if 'player' in st.query_params:
        col1, col2 = st.columns([1, 1])
        with col1: 
            render_trade_execution_panel(prices)
        with col2: 
            render_global_views(prices)
    else:
        st.info("🎯 Welcome to BlockVista Market Frenzy! Please join the game from the sidebar to start trading.")
        render_global_views(prices)

# --- Rest of the functions remain the same but use the enhanced versions above ---
# [Previous functions for data fetching, game logic, trading interface, etc. remain unchanged]
# Only replacing the render_global_views, render_market_sentiment_meter, and render_leaderboard with enhanced versions

# Note: The following functions would remain from the previous implementation:
# - get_daily_base_prices()
# - simulate_tick_prices()
# - calculate_derived_prices()
# - get_historical_data()
# - calculate_slippage()
# - apply_event_adjustment()
# - format_indian_currency()
# - optimize_portfolio()
# - calculate_indicator()
# - calculate_sharpe_ratio()
# - render_sidebar()
# - render_trade_execution_panel()
# - render_trade_interface()
# - render_market_order_ui()
# - render_limit_order_ui()
# - render_stop_loss_order_ui()
# - render_pending_orders()
# - render_algo_trading_tab()
# - render_current_holdings()
# - render_transaction_history()
# - render_strategy_tab()
# - render_sma_chart()
# - render_optimizer()
# - execute_trade()
# - log_transaction()
# - render_live_market_table()
# - run_game_tick()
# - auto_square_off_positions()
# - handle_futures_expiry()
# - check_margin_calls_and_orders()
# - run_algo_strategies()
# - main()

# Due to length constraints, I'm showing only the enhanced parts. The rest of the functions remain as in the previous implementation.

def main():
    game_state = get_game_state()
    if 'role' not in st.session_state:
        st.session_state.role = 'player'
    
    render_sidebar()
    
    # --- Main Price Flow ---
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
    
    # Only show admin dashboard if user is admin AND not already viewing as player
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

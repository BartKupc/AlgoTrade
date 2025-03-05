import json
import ta
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import requests
import time

# Add this to properly import from parent directory
import sys
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from utilities.bitget_futures import BitgetFutures

# Ensure directories exist
(base_dir / 'trade_logs').mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(base_dir / 'trade_logs' / f"momentum_live_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)


# Telegram Bot Setup
TELEGRAM_BOT_TOKEN = "7292159640:AAEZnq6jOkze_PBBWMiBHnyCW_dtiHVv6xo"
TELEGRAM_CHAT_ID = "7393611077"  # Make sure this is the correct chat ID

def send_telegram_message(message):
    """Sends a message to the Telegram bot"""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            logging.info("Telegram alert sent successfully.")
        else:
            # Remove emoji from logging to avoid encoding issues
            logging.warning(f"Telegram message failed: {response.text}")
    except Exception as e:
        # Remove emoji from logging to avoid encoding issues
        logging.error(f"Telegram API error: {str(e)}")


# Configuration and setup
params = {
    'symbol': 'ETH/USDT:USDT',
    'timeframe': '1h',
    'limit': 100,
    'ema_period': 9,
    'ema_threshold': 0.006,
    'stop_loss_pct': 0.01,      # 1% stop loss
    'take_profit_pct': 0.015,    # 2% take profit
    'leverage': 3,
    'max_short_duration': 24,
    'short_underwater_threshold': -0.02,
    'strong_momentum_threshold': 0.015,  # 1.5% price move
    'strong_volume_multiplier': 2.0,     # 2x average volume
    'cooldown_period': 15,      # minutes to wait after a loss
    'price_trend_bars': 3,      # number of bars to check price trend
    'min_price_trend': 0.001,   # minimum price trend (0.1%)
    'require_price_alignment': True,  # price must align with MACD
}

# Load API keys
try:
    key_path = base_dir / 'config' / 'config.json'
    with open(key_path, "r") as f:
        api_setup = json.load(f)['bitget']
    bitget = BitgetFutures(api_setup)
except Exception as e:
    logging.error(f"Failed to initialize: {str(e)}")
    sys.exit(1)

def fetch_data():
    """Fetch and prepare data with all necessary indicators"""
    try:
        # Fetch 1h OHLCV data
        data = bitget.fetch_recent_ohlcv(
            params['symbol'], 
            params['timeframe'], 
            params['limit']
        ).iloc[:-1]  # Get completed candles
        
        # Calculate indicators for 1h timeframe
        data['macd'] = ta.trend.macd(data['close'])
        data['macd_signal'] = ta.trend.macd_signal(data['close'])
        data['ema9'] = ta.trend.ema_indicator(data['close'], window=params['ema_period'])
        
        # Get real-time market data
        ticker = bitget.fetch_ticker(params['symbol'])
        current_price = ticker['last']
        orderbook = bitget.fetch_order_book(params['symbol'])
        
        # Calculate price movement from last candle's close
        last_candle_close = data.iloc[-1]['close']
        price_change_pct = (current_price - last_candle_close) / last_candle_close
        
        # Determine trend (using 0.2% threshold)
        if price_change_pct > 0.005:  # More than 0.2% up
            price_momentum = "up"
        elif price_change_pct < -0.005:  # More than 0.2% down
            price_momentum = "down"
        else:
            price_momentum = "sideways"
        
        # Get volume data
        current_minute_volume = ticker['quoteVolume']  # Current volume
        bid_volume = sum(bid[1] for bid in orderbook['bids'][:5])
        ask_volume = sum(ask[1] for ask in orderbook['asks'][:5])
        
        return data, current_price, current_minute_volume, bid_volume, ask_volume, price_momentum, price_change_pct
    except Exception as e:
        logging.error(f"Error fetching data: {str(e)}")
        raise

def check_trend(data, direction='long'):
    """Check if trend conditions are met"""
    current_close = data.iloc[-1]['close']
    y_bars_ago = data.iloc[-params['trend_bars']]['close']
    half_y_bars_ago = data.iloc[-params['trend_bars']//2]['close']
    
    if direction == 'long':
        return current_close > y_bars_ago and current_close > half_y_bars_ago
    else:  # short
        return current_close < y_bars_ago and current_close < half_y_bars_ago

def check_volume(data, direction='long'):
    """Check if volume conditions are met"""
    current_volume = data.iloc[-1]['volume']
    avg_volume = data.iloc[-1]['volume_ma']
    
    if direction == 'long':
        return True
    else:  # short
        return current_volume > (avg_volume * params['volume_threshold'])

def check_volatility(data):
    """Check if volatility conditions are met using ATR"""
    current_atr = data.iloc[-1]['atr']
    avg_atr = data.iloc[-1]['atr_ma']
    return current_atr > (avg_atr * params['atr_threshold'])

def calculate_position_size(close_price):
    """Calculate position size based on account balance"""
    try:
        balance = bitget.fetch_balance()
        usdt_balance = float(balance['USDT']['free'])
        
        # Use only a portion of the balance (e.g., 95% to account for fees)
        safe_balance = usdt_balance * 0.95
        
        # Calculate position value with leverage
        position_value = safe_balance * params['leverage']
        
        # Calculate quantity in contracts
        quantity = position_value / close_price
        
        # Round down to avoid exceeding available margin
        quantity = float(bitget.amount_to_precision(params['symbol'], quantity))
        
        logging.info(f"Position Size Calculation:")
        logging.info(f"Available USDT: ${usdt_balance:.2f}")
        logging.info(f"Safe Balance (95%): ${safe_balance:.2f}")
        logging.info(f"Position Value (with {params['leverage']}x leverage): ${position_value:.2f}")
        logging.info(f"Calculated Quantity: {quantity} contracts")
        
        return quantity
    except Exception as e:
        logging.error(f"Error calculating position size: {str(e)}")
        return None

def check_entry_conditions(data, current_price, current_volume, bid_volume, ask_volume, price_momentum, price_change_pct, direction='long'):
    """Check entry conditions with time-weighted momentum override"""
    last_row = data.iloc[-1]
    prev_row = data.iloc[-2]
    prev2_row = data.iloc[-3]
    macd, macd_signal, ema9 = last_row[['macd', 'macd_signal', 'ema9']]
    
    # Volume analysis
    avg_hourly_volume = data['volume'].tail(4).mean()
    avg_minute_volume = avg_hourly_volume / 60
    volume_ratio = current_volume / avg_minute_volume
    volume_acceptable = volume_ratio > 0.7

    # Check MACD trend over last 3 bars - only check if it's increasing/decreasing
    macd_values = [
        last_row['macd'],
        prev_row['macd'],
        prev2_row['macd']
    ]
    macd_increasing = all(macd_values[i] > macd_values[i+1] for i in range(2))
    macd_decreasing = all(macd_values[i] < macd_values[i+1] for i in range(2))

    # Check price trend over last 3 bars
    price_trend = (current_price - data['close'].iloc[-3]) / data['close'].iloc[-3]

    if direction == 'long':
        # Standard conditions
        macd_condition = macd_increasing  # Only check if MACD is increasing
        price_condition = (current_price/ema9 - 1) > params['ema_threshold']  # 0.6%
        volume_condition = volume_acceptable
        pressure_condition = bid_volume > ask_volume * 1.2
        trend_condition = price_trend > params['min_price_trend']  # 0.1%
        
        standard_conditions = (
            macd_condition and
            price_condition and
            volume_condition and
            pressure_condition and
            trend_condition
        )
        
        # Strong momentum conditions
        momentum_price = price_change_pct > params['strong_momentum_threshold']  # 1.5%
        momentum_volume = volume_ratio > params['strong_volume_multiplier']  # 2x
        momentum_pressure = bid_volume > ask_volume * 1.5
        momentum_price_level = current_price > ema9
        momentum_macd = macd_increasing  # Changed to only check direction
        
        strong_momentum = (
            momentum_price and
            momentum_volume and
            momentum_pressure and
            momentum_price_level and
            momentum_macd
        )
        
    else:  # short conditions
        # Standard conditions with debug logging
        macd_condition = macd_decreasing
        logging.info(f"SHORT - MACD decreasing: {macd_condition}")
        
        price_condition = (ema9/current_price - 1) > params['ema_threshold']
        logging.info(f"SHORT - Price condition: {price_condition} ({(ema9/current_price - 1)*100:.2f}%)")
        
        volume_condition = volume_acceptable
        logging.info(f"SHORT - Volume condition: {volume_condition} ({volume_ratio:.2f}x)")
        
        pressure_condition = ask_volume > bid_volume * 1.2
        logging.info(f"SHORT - Pressure condition: {pressure_condition} (ask/bid: {ask_volume/bid_volume:.2f})")
        
        trend_condition = price_trend < -params['min_price_trend']
        logging.info(f"SHORT - Trend condition: {trend_condition} ({price_trend*100:.2f}%)")
        
        standard_conditions = (
            macd_condition and
            price_condition and
            volume_condition and
            pressure_condition and
            trend_condition
        )
        logging.info(f"SHORT - Final standard conditions: {standard_conditions}")
        
        # Strong momentum conditions
        momentum_price = price_change_pct < -params['strong_momentum_threshold']  # -1.5%
        momentum_volume = volume_ratio > params['strong_volume_multiplier']  # 2x
        momentum_pressure = ask_volume > bid_volume * 1.5
        momentum_price_level = current_price < ema9
        momentum_macd = macd_decreasing  # Changed to only check direction
        
        strong_momentum = (
            momentum_price and
            momentum_volume and
            momentum_pressure and
            momentum_price_level and
            momentum_macd
        )

    return standard_conditions, strong_momentum, price_trend, macd_values, macd_increasing, macd_decreasing

def check_exit_conditions(data, current_price, position_side):
    """Check exit conditions using 1h OHLC and current price"""
    last_row = data.iloc[-1]
    prev_row = data.iloc[-2]
    macd_diff = last_row['macd'] - last_row['macd_signal']
    prev_macd_diff = prev_row['macd'] - prev_row['macd_signal']
    
    if position_side == 'long':
        return (macd_diff < prev_macd_diff and 
                current_price < last_row['ema9'])
    else:  # short
        return (macd_diff > prev_macd_diff and 
                current_price > last_row['ema9'])

# Add after the existing logging setup
def setup_trade_logger():
    trade_logger = logging.getLogger('trade_logger')
    trade_logger.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s,%(message)s')
    
    # Create file handler with current date
    trade_log_file = base_dir / 'trade_logs' / f"trades_{datetime.now().strftime('%Y%m%d')}.csv"
    
    # Add header if file doesn't exist
    if not trade_log_file.exists():
        with open(trade_log_file, 'w') as f:
            f.write("Timestamp,Action,Side,Entry Price,Exit Price,Contracts,PnL,Duration\n")
    
    file_handler = logging.FileHandler(trade_log_file)
    file_handler.setFormatter(formatter)
    
    trade_logger.addHandler(file_handler)
    return trade_logger

trade_logger = setup_trade_logger()

def log_trade(action, side, entry_price, exit_price=None, contracts=None, pnl=None, duration=None):
    """Log trade details to CSV file"""
    trade_details = f"{action},{side},{entry_price},{exit_price},{contracts},{pnl},{duration}"
    trade_logger.info(trade_details)

def check_recent_trades():
    """Check recent trade history for losses"""
    try:
        # Use fetch_positions to get recent position history
        positions = bitget.fetch_positions([params['symbol']])
        if not positions:
            return True
        
        # Get the most recent closed position
        closed_positions = [pos for pos in positions if float(pos['info'].get('holdSide', 0)) == 0]
        if not closed_positions:
            return True
        
        # Check last position
        last_position = closed_positions[-1]
        last_position_time = datetime.fromtimestamp(float(last_position['timestamp']) / 1000)
        time_since_last_position = (datetime.now() - last_position_time).total_seconds() / 60
        
        # If last position was a loss and within cooldown period, don't trade
        unrealized_pnl = float(last_position['info'].get('unrealisedPnl', 0))
        if unrealized_pnl < 0 and time_since_last_position < params['cooldown_period']:
            logging.info(f"In cooldown period after loss. {params['cooldown_period'] - time_since_last_position:.1f} minutes remaining")
            return False
            
        return True
    except Exception as e:
        logging.error(f"Error checking recent trades: {str(e)}")
        # Return True in case of error to allow trading to continue
        return True

def check_price_trend(data, current_price, direction='long'):
    """Check if price trend aligns with intended direction"""
    recent_prices = data['close'].tail(params['price_trend_bars']).values
    price_change = (current_price - recent_prices[0]) / recent_prices[0]
    
    if direction == 'long':
        return price_change > params['min_price_trend']
    else:
        return price_change < -params['min_price_trend']

def trade_logic():
    logging.info("Fetching and processing data...")
    data, current_price, current_volume, bid_volume, ask_volume, price_momentum, price_change_pct = fetch_data()
    
    # Calculate volume metrics early
    avg_hourly_volume = data['volume'].tail(4).mean()
    avg_minute_volume = avg_hourly_volume / 60
    volume_ratio = current_volume / avg_minute_volume
    bid_ask_ratio = (bid_volume/ask_volume if ask_volume > 0 else 0)
    
    # Check for open positions and orders
    position = bitget.fetch_open_positions(params['symbol'])
    has_position = len(position) > 0
    
    # Check cooldown period
    can_trade = check_recent_trades()
    
    # Check entry conditions
    long_standard, long_momentum, price_trend, macd_values, macd_increasing, macd_decreasing = check_entry_conditions(
        data, current_price, current_volume, bid_volume, ask_volume, 
        price_momentum, price_change_pct, 'long'
    )
    short_standard, short_momentum, _, _, _, _ = check_entry_conditions(
        data, current_price, current_volume, bid_volume, ask_volume, 
        price_momentum, price_change_pct, 'short'
    )
    
    # Add cooldown check to entry signals
    long_standard = long_standard and can_trade
    long_momentum = long_momentum and can_trade
    short_standard = short_standard and can_trade
    short_momentum = short_momentum and can_trade
    
    # If no position is open, cancel all trigger orders
    if not has_position:
        try:
            trigger_orders = bitget.fetch_open_trigger_orders(params['symbol'])
            if trigger_orders:
                logging.info("Cancelling orphaned trigger orders:")
                for order in trigger_orders:
                    bitget.cancel_trigger_order(order['id'], params['symbol'])
                    logging.info(f"[*] Cancelled order {order['id']}")
        except Exception as e:
            logging.error(f"Error cancelling trigger orders: {str(e)}")
    
    # Log current market state with trading implications
    logging.info(f"Current Price: ${current_price:.2f}")
    logging.info(f"Current Volume: {current_volume:.2f}")
    logging.info(f"Volume vs Last Hour: {volume_ratio:.2f}x")
    logging.info(f"Bid/Ask Volume Ratio: {bid_ask_ratio:.2f} " + 
                f"({'Bullish' if bid_ask_ratio > 1 else 'Bearish'} pressure)")
    
    # Get key indicators for logging
    last_row = data.iloc[-1]
    prev_row = data.iloc[-2]
    prev2_row = data.iloc[-3]
    
    macd = last_row['macd']
    macd_signal = last_row['macd_signal']
    ema9 = last_row['ema9']
    
    # Log all relevant indicators with trading implications
    logging.info("\n=== Current Indicators ===")
    logging.info(f"MACD: {macd:.6f}, Signal: {macd_signal:.6f}")
    logging.info(f"MACD Position: {'Above Signal (Bullish)' if macd > macd_signal else 'Below Signal (Bearish)'}")
    
    ema_ratio = (current_price/ema9-1)*100
    logging.info(f"EMA9: {ema9:.2f} (Price/EMA Ratio: {ema_ratio:.2f}%)")
    logging.info(f"EMA Status: {'Price Above EMA (Bullish)' if current_price > ema9 else 'Price Below EMA (Bearish)'}")
    
    volume_change = ((last_row['volume'] - prev_row['volume'])/prev_row['volume'])*100
    logging.info(f"Volume: {last_row['volume']:.2f} (Previous: {prev_row['volume']:.2f})")
    logging.info(f"Volume Change: {volume_change:.2f}% ({'Increasing' if volume_change > 0 else 'Decreasing'} momentum)")
    
    # MACD trend analysis
    macd_values = [
        macd - macd_signal,
        data.iloc[-2]['macd'] - data.iloc[-2]['macd_signal'],
        data.iloc[-3]['macd'] - data.iloc[-3]['macd_signal']
    ]
    
    # Check if MACD differences are increasing/decreasing
    macd_increasing = all(macd_values[i] > macd_values[i+1] for i in range(2))
    macd_decreasing = all(macd_values[i] < macd_values[i+1] for i in range(2))
    
    # Determine MACD trend status
    if macd_increasing:
        macd_trend = "Strong Bullish (MACD increasing for 3 bars)"
        macd_status = "BULLISH MOMENTUM"
    elif macd_decreasing:
        macd_trend = "Strong Bearish (MACD decreasing for 3 bars)"
        macd_status = "BEARISH MOMENTUM"
    elif macd > macd_signal:
        macd_trend = "MACD above signal but no momentum"
        macd_status = "MACD POSITIVE (no momentum)"
    else:
        macd_trend = "MACD below signal but no momentum"
        macd_status = "MACD NEGATIVE (no momentum)"

    # Log MACD analysis
    logging.info("\n=== MACD Analysis ===")
    logging.info(f"Current MACD-Signal: {macd_values[0]:.6f}")
    logging.info(f"Previous MACD-Signal: {macd_values[1]:.6f}")
    logging.info(f"2 Candles Ago MACD-Signal: {macd_values[2]:.6f}")
    logging.info(f"MACD Trend: {macd_trend}")
    logging.info(f"MACD Status: {macd_status}")
    
    # Add real-time price movement analysis
    logging.info("\n=== Real-time Price Analysis ===")
    logging.info(f"Last Candle Close: ${data.iloc[-1]['close']:.2f}")
    logging.info(f"Current Price: ${current_price:.2f}")
    logging.info(f"Price Change: {price_change_pct*100:.2f}%")
    logging.info(f"Price Movement: {price_momentum.upper()}")
    logging.info(f"Aligned with MACD Trend: {'YES' if (macd_increasing and price_momentum != 'down') or (macd_decreasing and price_momentum != 'up') else 'NO'}")
    
    # Initialize entry variables early
    long_entry = False
    short_entry = False
    long_standard = False
    short_standard = False
    long_momentum = False
    short_momentum = False
    
    # Only proceed with new orders if no position exists
    if not has_position:
        # Check if we should be trading (MOVED THIS CHECK BEFORE ANY ENTRY LOGIC)
        if not check_recent_trades():
            logging.info("Skipping trade: In cooldown period after recent loss")
            return
        
        # Determine if we should enter
        long_entry = long_standard or long_momentum
        short_entry = short_standard or short_momentum
        
        if long_entry or short_entry:
            side = 'buy' if long_entry else 'sell'
            is_momentum_entry = long_momentum if long_entry else short_momentum
            
            logging.info(f"\n=== Entry Analysis ===")
            logging.info(f"Entry Type: {'Strong Momentum' if is_momentum_entry else 'Standard'}")
            logging.info(f"Direction: {side.upper()}")
            logging.info(f"Price Movement: {price_momentum}")
            logging.info(f"Volume Ratio: {volume_ratio:.2f}x")
            logging.info(f"Bid/Ask Ratio: {bid_ask_ratio:.2f}")
            
            # Calculate position size
            quantity = calculate_position_size(current_price)
            if quantity is None:
                logging.warning("Failed to calculate position size")
                return
            
            try:
                # Place entry order
                if is_momentum_entry:
                    entry_order = bitget.place_market_order(
                        symbol=params['symbol'],
                        side=side,
                        amount=quantity
                    )
                    entry_price = current_price
                else:
                    entry_price = current_price * (1.001 if side == 'buy' else 0.999)
                    entry_order = bitget.place_limit_order(
                        symbol=params['symbol'],
                        side=side,
                        amount=quantity,
                        price=entry_price
                    )
                
                logging.info(f"\n=== Entry Order Placed ===")
                logging.info(f"Side: {side.upper()}")
                logging.info(f"Entry Price: ${entry_price:.2f}")
                logging.info(f"Quantity: {quantity}")
                
                # Simple SL/TP logic
                if side == 'buy':  # LONG position
                    stop_loss = entry_price * (1 - params['stop_loss_pct'])
                    take_profit = entry_price * (1 + params['take_profit_pct'])
                    close_side = 'sell'
                else:  # SHORT position
                    stop_loss = entry_price * (1 + params['stop_loss_pct'])
                    take_profit = entry_price * (1 - params['take_profit_pct'])
                    close_side = 'buy'
                
                # Place Stop Loss - using quantity instead of size
                sl_order = bitget.place_trigger_market_order(
                    symbol=params['symbol'],
                    side=close_side,
                    amount=quantity,  # Changed from size to quantity
                    trigger_price=stop_loss,
                    reduce=True
                )
                
                # Place Take Profit - using quantity instead of size
                tp_order = bitget.place_trigger_market_order(
                    symbol=params['symbol'],
                    side=close_side,
                    amount=quantity,  # Changed from size to quantity
                    trigger_price=take_profit,
                    reduce=True
                )
                
                # Log orders
                logging.info(f"\n=== Stop Loss Order ===")
                logging.info(f"Type: {'LONG' if side == 'buy' else 'SHORT'} Stop Loss")
                logging.info(f"Close Side: {close_side.upper()}")
                logging.info(f"Trigger Price: ${stop_loss:.2f}")
                
                logging.info(f"\n=== Take Profit Order ===")
                logging.info(f"Type: {'LONG' if side == 'buy' else 'SHORT'} Take Profit")
                logging.info(f"Close Side: {close_side.upper()}")
                logging.info(f"Trigger Price: ${take_profit:.2f}")
                
                # Send Telegram message
                entry_message = (
                    f"🚨 NEW {side.upper()} POSITION OPENED 🚨\n"
                    f"Entry Type: {'Strong Momentum' if is_momentum_entry else 'Standard Entry'}\n"
                    f"Price: ${entry_price:.2f}\n"
                    f"Size: {quantity} contracts\n"
                    f"Stop Loss: ${stop_loss:.2f} ({'-' if side == 'buy' else '+'}{params['stop_loss_pct']*100}%)\n"
                    f"Take Profit: ${take_profit:.2f} ({'+' if side == 'buy' else '-'}{params['take_profit_pct']*100}%)"
                )
                send_telegram_message(entry_message)

            except Exception as e:
                logging.error(f"Error placing orders: {str(e)}")
                send_telegram_message(f"⚠️ Error placing orders: {str(e)}")

    # Get position details if any exists
    position_info = "No Position"
    position_entry = "N/A"
    position_pnl = "N/A"
    if has_position:
        try:
            pos = position[0]
            position_side = pos['side']
            entry_price = float(pos['info']['openPriceAvg'])
            size = pos['contracts']
            
            try:
                unrealized_pnl = float(pos['info'].get('unrealisedPnl') or 
                                     pos['info'].get('unrealizedPnl') or 
                                     pos.get('unrealizedPnl') or 
                                     0.0)
            except (KeyError, ValueError):
                unrealized_pnl = 0.0
                logging.warning("Could not get unrealized PnL value")
            
            position_info = f"{position_side.upper()} {size} contracts"
            position_entry = f"${entry_price:.2f}"
            position_pnl = f"${unrealized_pnl:.2f}"
        except Exception as e:
            logging.error(f"Error processing position info: {str(e)}")
            position_info = "Error getting position details"

    # Enhanced volume logging
    logging.info("\n=== Volume Analysis ===")
    logging.info(f"Current Minute Volume: {current_volume:.2f}")
    logging.info(f"Average Minute Volume (4h): {avg_minute_volume:.2f}")
    logging.info(f"Volume Ratio: {volume_ratio:.2f}x average")
    logging.info(f"Volume Acceptable: {volume_ratio > 0.7}")
    logging.info(f"Bid/Ask Ratio: {(bid_volume/ask_volume):.2f}")
    logging.info(f"Order Book Pressure: {'Buying' if bid_volume > ask_volume else 'Selling'}")
    
    # Add strong momentum analysis to logging
    logging.info("\n=== Strong Momentum Analysis ===")
    logging.info(f"Price Change: {price_change_pct*100:.2f}% (Threshold: {params['strong_momentum_threshold']*100:.1f}%)")
    logging.info(f"Volume Ratio: {volume_ratio:.2f}x (Threshold: {params['strong_volume_multiplier']:.1f}x)")
    logging.info(f"Strong Momentum Conditions:")
    logging.info(f"[*] Large Price Move: {'YES' if abs(price_change_pct) > params['strong_momentum_threshold'] else 'NO'}")
    logging.info(f"[*] High Volume: {'YES' if volume_ratio > params['strong_volume_multiplier'] else 'NO'}")
    logging.info(f"[*] Strong Order Book Pressure: {'YES' if max(bid_volume/ask_volume, ask_volume/bid_volume) > 1.5 else 'NO'}")
    
    # Position management
    if has_position:
        logging.info("\n=== Position Management ===")
        pos = position[0]
        position_side = pos['side']
        entry_price = float(pos['info']['openPriceAvg'])
        size = pos['contracts']
        
        logging.info(f"Current Position: {position_side.upper()} {size} contracts @ ${entry_price:.2f}")
        
        # Check for existing trigger orders
        trigger_orders = bitget.fetch_open_trigger_orders(params['symbol'])
        
        # Cancel any existing trigger orders before placing new ones
        if trigger_orders:
            logging.info("Cancelling existing trigger orders before placing new ones...")
            for order in trigger_orders:
                bitget.cancel_trigger_order(order['id'], params['symbol'])
                logging.info(f"Cancelled order {order['id']}")
        
        # Simple and clear logic for SL/TP placement
        if position_side == 'long':  
            close_side = 'sell'  # Close long position
            stop_loss = entry_price * (1 - params['stop_loss_pct'])
            take_profit = entry_price * (1 + params['take_profit_pct'])
        elif position_side == 'short':  
            close_side = 'buy'  # Close short position
            stop_loss = entry_price * (1 + params['stop_loss_pct'])
            take_profit = entry_price * (1 - params['take_profit_pct'])
        
        logging.info(f"Position Side: {position_side.upper()}")
        logging.info(f"Close Side: {close_side.upper()}")
        logging.info(f"Entry Price: ${entry_price:.2f}")
        logging.info(f"Stop Loss Price: ${stop_loss:.2f}")
        logging.info(f"Take Profit Price: ${take_profit:.2f}")
        
        try:
            # Place stop loss
            sl_order = bitget.place_trigger_market_order(
                symbol=params['symbol'],
                side=close_side,
                amount=size,
                trigger_price=stop_loss,
                reduce=True
            )
            logging.info(f"Stop Loss order placed for {position_side.upper()}: {sl_order['id']} at ${stop_loss:.2f}")
            
            # Place take profit
            tp_order = bitget.place_trigger_market_order(
                symbol=params['symbol'],
                side=close_side,
                amount=size,
                trigger_price=take_profit,
                reduce=True
            )
            logging.info(f"Take Profit order placed for {position_side.upper()}: {tp_order['id']} at ${take_profit:.2f}")
            
        except Exception as e:
            logging.error(f"Error placing SL/TP orders: {str(e)}")
            send_telegram_message(f"⚠️ Error placing SL/TP orders: {str(e)}")
    
    # Create status message with initialized variables
    message = (
        f"=== Momentum Bot Status Update ===\n\n"
        f"Position Status:\n"
        f"Current: {position_info}\n"
        f"Entry Price: {position_entry}\n"
        f"Unrealized PnL: {position_pnl}\n\n"
        
        f"Market Analysis:\n"
        f"Current Price: ${current_price:.2f}\n"
        f"Current Volume: {current_volume:.2f}\n"
        f"Volume vs Avg: {volume_ratio:.2f}x\n"
        f"Bid/Ask Ratio: {bid_ask_ratio:.2f} ({('Bullish' if bid_ask_ratio > 1 else 'Bearish')} pressure)\n\n"
        
        f"Technical Indicators:\n"
        f"MACD: {macd:.6f} vs Signal: {macd_signal:.6f}\n"
        f"MACD Status: {macd_status}\n"
        f"MACD Trend: {macd_trend}\n"
        f"EMA9: {ema9:.2f} (Price/EMA: {ema_ratio:.2f}%)\n"
        f"Volume Change: {volume_change:.2f}%\n\n"
        
        f"Trading Signals:\n"
        f"LONG Entry Conditions:\n"
        f"[*] No Existing Position: {'✅' if not has_position else '❌'} ({position_info})\n"
        f"[*] MACD Momentum: {'✅' if macd_increasing else '❌'} (needs 3 increasing bars)\n"
        f"   • Current diff: {macd_values[0]:.6f}\n"
        f"   • Previous diff: {macd_values[1]:.6f}\n"
        f"   • 2 bars ago: {macd_values[2]:.6f}\n"
        f"[*] Price/EMA9 Threshold: {'✅' if (current_price/ema9 - 1) > 0.006 else '❌'} ({(current_price/ema9 - 1)*100:.2f}% vs 0.6%)\n"
        f"[*] Volume Acceptable: {'✅' if volume_ratio > 0.7 else '❌'} ({volume_ratio:.2f}x)\n"
        f"[*] Strong Bullish Pressure: {'✅' if bid_ask_ratio > 1.2 else '❌'} ({bid_ask_ratio:.2f})\n"
        f"[*] Price Trend (3 bars): {'✅' if price_trend > 0.001 else '❌'} ({price_trend*100:.2f}% vs 0.1%)\n"
        f"Final Long Signal: {'✅' if long_entry else '❌'}\n\n"
        
        f"SHORT Conditions:\n"
        f"[*] No Existing Position: {'✓' if not has_position else '✗'} ({position_info})\n"
        f"[*] Not in Cooldown: {'✓' if can_trade else '✗'} ({params['cooldown_period']} min after loss)\n"
        f"[*] MACD Momentum: {'✓' if macd_decreasing else '✗'} (needs 3 decreasing bars)\n"
        f"   • Current diff: {macd_values[0]:.6f}\n"
        f"   • Previous diff: {macd_values[1]:.6f}\n"
        f"   • 2 bars ago: {macd_values[2]:.6f}\n"
        f"[*] Price/EMA9 Threshold: {'✓' if (ema9/current_price - 1) > params['ema_threshold'] else '✗'} ({(ema9/current_price - 1)*100:.2f}% vs 0.6%)\n"
        f"[*] Volume Acceptable: {'✓' if volume_ratio > 0.7 else '✗'} ({volume_ratio:.2f}x)\n"
        f"[*] Strong Bearish Pressure: {'✓' if bid_ask_ratio < 0.8 else '✗'} ({bid_ask_ratio:.2f})\n"
        f"[*] Price Trend (3 bars): {'✓' if price_trend < -params['min_price_trend'] else '✗'} ({price_trend*100:.2f}% vs -0.1%)\n"
        f"Short Standard Conditions: {'✓' if short_standard else '✗'}\n"
        f"Short Momentum Conditions: {'✓' if short_momentum else '✗'}\n"
        f"Final Short Signal: {'✓' if short_entry else '✗'}\n"
    )

    # Add pending orders if any
    open_orders = bitget.fetch_open_orders(params['symbol'])
    if open_orders:
        message += "\n\nPending Orders:\n"
        for order in open_orders:
            message += f"- {order['side'].upper()} {order['amount']} @ ${float(order['price']):.2f}\n"

    # Add stop loss/take profit orders if any
    trigger_orders = bitget.fetch_open_trigger_orders(params['symbol'])
    if trigger_orders:
        message += "\nActive TP/SL Orders:\n"
        for order in trigger_orders:
            message += f"- {order['side'].upper()} {order['amount']} @ ${float(order['triggerPrice']):.2f}\n"

    # Update Telegram message to include volume analysis
    message += (
        f"Volume Analysis:\n"
        f"Order Book Pressure: {'Buying' if bid_volume > ask_volume else 'Selling'}\n"
        f"Bid/Ask Ratio: {(bid_volume/ask_volume):.2f}\n\n"
    )

    send_telegram_message(message)

if __name__ == "__main__":
    try:
        logging.info("Starting momentum trading bot...")
        
        # Clean up any existing trigger orders at startup
        try:
            trigger_orders = bitget.fetch_open_trigger_orders(params['symbol'])
            if trigger_orders:
                logging.info("Cleaning up existing trigger orders at startup:")
                for order in trigger_orders:
                    bitget.cancel_trigger_order(order['id'], params['symbol'])
                    logging.info(f"[*] Cancelled order {order['id']}")
        except Exception as e:
            logging.error(f"Error cleaning up trigger orders at startup: {str(e)}")
        
        trade_logic()
        logging.info("Trade logic execution completed")
    except Exception as e:
        logging.error(f"Critical error in main execution: {str(e)}", exc_info=True)
        sys.exit(1)
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
    'stop_loss_pct': 0.01,
    'take_profit_pct': 0.10,
    'leverage': 3,
    'max_short_duration': 48,
    'short_underwater_threshold': -0.02,
    'strong_momentum_threshold': 0.015,  # 1.5% price move
    'strong_volume_multiplier': 2.0,     # 2x average volume
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
    """Check entry conditions with strong momentum override"""
    last_row = data.iloc[-1]
    macd, macd_signal, ema9 = last_row[['macd', 'macd_signal', 'ema9']]
    
    # Get previous MACD differences
    macd_diffs = [
        macd - macd_signal,
        data.iloc[-2]['macd'] - data.iloc[-2]['macd_signal'],
        data.iloc[-3]['macd'] - data.iloc[-3]['macd_signal']
    ]
    
    # Volume analysis
    avg_hourly_volume = data['volume'].tail(4).mean()
    avg_minute_volume = avg_hourly_volume / 60
    volume_ratio = current_volume / avg_minute_volume
    volume_acceptable = volume_ratio > 0.7

    # Standard conditions
    if direction == 'long':
        price_condition = (current_price > ema9 * (1 + params['ema_threshold']))
        macd_condition = all(macd_diffs[i] > macd_diffs[i+1] for i in range(2))
        volume_condition = volume_acceptable and (bid_volume > ask_volume * 1.2)
        
        # Strong momentum override for longs
        strong_momentum = (
            price_change_pct > params['strong_momentum_threshold'] and  # Big price move up
            volume_ratio > params['strong_volume_multiplier'] and       # High volume
            bid_volume > ask_volume * 1.5 and                          # Strong buying pressure
            current_price > ema9 and                                   # Above EMA
            macd > macd_signal                                         # MACD positive
        )
    else:  # short
        price_condition = (current_price < ema9 * (1 - params['ema_threshold']))
        macd_condition = all(macd_diffs[i] < macd_diffs[i+1] for i in range(2))
        volume_condition = volume_acceptable and (ask_volume > bid_volume * 1.2)
        
        # Strong momentum override for shorts
        strong_momentum = (
            price_change_pct < -params['strong_momentum_threshold'] and  # Big price move down
            volume_ratio > params['strong_volume_multiplier'] and        # High volume
            ask_volume > bid_volume * 1.5 and                           # Strong selling pressure
            current_price < ema9 and                                    # Below EMA
            macd < macd_signal                                          # MACD negative
        )

    # Return both standard conditions and strong momentum status
    standard_conditions = (macd_condition and price_condition and volume_condition)
    
    if strong_momentum:
        logging.info(f"Strong momentum override triggered for {direction.upper()}!")
        logging.info(f"Price change: {price_change_pct*100:.2f}%")
        logging.info(f"Volume ratio: {volume_ratio:.2f}x average")
        logging.info(f"Bid/Ask ratio: {(bid_volume/ask_volume):.2f}")
    
    return standard_conditions, strong_momentum

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
    macd_diffs = [
        macd - macd_signal,
        data.iloc[-2]['macd'] - data.iloc[-2]['macd_signal'],
        data.iloc[-3]['macd'] - data.iloc[-3]['macd_signal']
    ]
    
    # Check if MACD differences are increasing/decreasing
    macd_increasing = all(macd_diffs[i] > macd_diffs[i+1] for i in range(2))
    macd_decreasing = all(macd_diffs[i] < macd_diffs[i+1] for i in range(2))
    
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
    logging.info(f"Current MACD-Signal: {macd_diffs[0]:.6f}")
    logging.info(f"Previous MACD-Signal: {macd_diffs[1]:.6f}")
    logging.info(f"2 Candles Ago MACD-Signal: {macd_diffs[2]:.6f}")
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
        # Check entry conditions
        long_standard, long_momentum = check_entry_conditions(data, current_price, current_volume, bid_volume, ask_volume, price_momentum, price_change_pct, 'long')
        short_standard, short_momentum = check_entry_conditions(data, current_price, current_volume, bid_volume, ask_volume, price_momentum, price_change_pct, 'short')
        
        # Determine if we should enter
        long_entry = long_standard or long_momentum
        short_entry = short_standard or short_momentum
        
        if long_entry or short_entry:
            side = 'buy' if long_entry else 'sell'
            is_momentum_entry = long_momentum if long_entry else short_momentum
            
            # Calculate position size
            quantity = calculate_position_size(current_price)
            if quantity is None:
                logging.warning("Failed to calculate position size")
                return
            
            logging.info(f"\n=== Placing New {side.upper()} Order ===")
            logging.info(f"Amount: {quantity} contracts")
            
            try:
                # Use market order for momentum entries, limit order for standard entries
                if is_momentum_entry:
                    entry_order = bitget.place_market_order(
                        symbol=params['symbol'],
                        side=side,
                        amount=quantity
                    )
                    entry_price = current_price  # Use current price for market orders
                else:
                    entry_price = current_price * (1.001 if long_entry else 0.999)
                    entry_order = bitget.place_limit_order(
                        symbol=params['symbol'],
                        side=side,
                        amount=quantity,
                        price=entry_price
                    )
                
                logging.info(f"Entry order placed: {entry_order['id']} at ${entry_price:.2f}")
                
                # Calculate SL/TP levels based on entry direction
                if side == 'buy':  # Long position
                    stop_loss = entry_price * (1 - params['stop_loss_pct'])    # Below entry
                    take_profit = entry_price * (1 + params['take_profit_pct']) # Above entry
                    close_side = 'sell'  # Close long position with sell
                else:  # Short position
                    stop_loss = entry_price * (1 + params['stop_loss_pct'])    # Above entry
                    take_profit = entry_price * (1 - params['take_profit_pct']) # Below entry
                    close_side = 'buy'   # Close short position with buy
                
                # Place SL order
                sl_order = bitget.place_trigger_market_order(
                    symbol=params['symbol'],
                    side=close_side,  # Use the correct closing side
                    amount=quantity,
                    trigger_price=stop_loss,
                    reduce=True
                )
                logging.info(f"Stop Loss order placed for {side.upper()}: {sl_order['id']} at ${stop_loss:.2f}")
                
                # Place TP order
                tp_order = bitget.place_trigger_market_order(
                    symbol=params['symbol'],
                    side=close_side,  # Use the correct closing side
                    amount=quantity,
                    trigger_price=take_profit,
                    reduce=True
                )
                logging.info(f"Take Profit order placed for {side.upper()}: {tp_order['id']} at ${take_profit:.2f}")
                
                # Log the opened trade
                log_trade(
                    action="OPEN",
                    side=side,
                    entry_price=entry_price,
                    contracts=quantity,
                    exit_price=None,
                    pnl=None,
                    duration=None
                )
                
                # Send detailed Telegram message
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
        f"LONG Conditions:\n"
        f"[*] MACD Momentum: {'YES' if macd_increasing else 'NO'} (needs 3 increasing bars)\n"
        f"[*] Price Above EMA9: {current_price > ema9}\n"
        f"[*] Volume Acceptable: {volume_ratio > 0.7}\n"
        f"[*] Bullish Pressure: {bid_ask_ratio > 1}\n"
        f"Final Long Signal: {long_entry}\n\n"
        
        f"SHORT Conditions:\n"
        f"[*] MACD Momentum: {'YES' if macd_decreasing else 'NO'} (needs 3 decreasing bars)\n"
        f"[*] Price Below EMA9: {current_price < ema9}\n"
        f"[*] Volume Acceptable: {volume_ratio > 0.7}\n"
        f"[*] Bearish Pressure: {bid_ask_ratio < 1}\n"
        f"Final Short Signal: {short_entry}"
    )

    # Add trading recommendation
    if long_entry:
        message += ">> STRONG LONG OPPORTUNITY - All conditions met"
    elif short_entry:
        message += ">> STRONG SHORT OPPORTUNITY - All conditions met"
    else:
        message += ">> WAIT - Conditions not optimal for entry\n"
        if macd > macd_signal:
            message += "Watching for long setup (MACD bullish but other conditions not met)"
        else:
            message += "Watching for short setup (MACD bearish but other conditions not met)"

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
        f"Current vs Avg: {volume_ratio:.2f}x\n"
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
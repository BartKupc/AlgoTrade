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
    'short_underwater_threshold': -0.02
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
        ).iloc[:-1]
        
        # Calculate indicators for 1h timeframe
        data['macd'] = ta.trend.macd(data['close'])
        data['macd_signal'] = ta.trend.macd_signal(data['close'])
        data['ema9'] = ta.trend.ema_indicator(data['close'], window=params['ema_period'])
        
        # Get real-time market data
        ticker = bitget.fetch_ticker(params['symbol'])
        orderbook = bitget.fetch_order_book(params['symbol'])
        
        # Get real-time volume data (last minute)
        recent_trades = bitget.fetch_recent_trades(params['symbol'], limit=100)
        current_minute_volume = sum(trade['amount'] for trade in recent_trades 
                                  if trade['timestamp'] > (time.time() * 1000 - 60000))
        
        # Calculate order book volumes
        bid_volume = sum(bid[1] for bid in orderbook['bids'][:5])
        ask_volume = sum(ask[1] for ask in orderbook['asks'][:5])
        
        return data, ticker['last'], current_minute_volume, bid_volume, ask_volume
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
        position_value = usdt_balance * params['leverage']
        quantity = position_value / close_price
        return float(bitget.amount_to_precision(params['symbol'], quantity))
    except Exception as e:
        print(f"Error calculating position size: {e}")
        return None

def check_entry_conditions(data, current_price, current_volume, bid_volume, ask_volume, direction='long'):
    """Check entry conditions using both real-time and historical data"""
    last_row = data.iloc[-1]
    prev_row = data.iloc[-2]
    macd, macd_signal, ema9 = last_row[['macd', 'macd_signal', 'ema9']]
    
    # Get previous MACD differences from hourly data
    macd_diffs = [
        macd - macd_signal,
        data.iloc[-2]['macd'] - data.iloc[-2]['macd_signal'],
        data.iloc[-3]['macd'] - data.iloc[-3]['macd_signal']
    ]
    
    # Volume analysis combining historical and real-time data
    avg_hourly_volume = data['volume'].tail(4).mean()  # Last 4 hours average
    avg_minute_volume = avg_hourly_volume / 60
    
    # Real-time volume checks
    volume_ratio = current_volume / avg_minute_volume
    volume_acceptable = volume_ratio > 0.7  # Accept 70% of average volume
    
    # Order book pressure analysis
    if direction == 'long':
        price_condition = (current_price > ema9 * (1 + params['ema_threshold']))
        macd_condition = all(macd_diffs[i] > macd_diffs[i+1] for i in range(2))
        volume_condition = volume_acceptable and (bid_volume > ask_volume * 1.2)  # 20% more bids than asks
    else:  # short
        price_condition = (current_price < ema9 * (1 - params['ema_threshold']))
        macd_condition = all(macd_diffs[i] < macd_diffs[i+1] for i in range(2))
        volume_condition = volume_acceptable and (ask_volume > bid_volume * 1.2)  # 20% more asks than bids
    
    return (macd_condition and price_condition and volume_condition)

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
    data, current_price, current_volume, bid_volume, ask_volume = fetch_data()
    
    # Calculate volume metrics early
    avg_hourly_volume = data['volume'].tail(4).mean()
    avg_minute_volume = avg_hourly_volume / 60
    volume_ratio = current_volume / avg_minute_volume
    bid_ask_ratio = (bid_volume/ask_volume if ask_volume > 0 else 0)
    
    # Check for open positions and orders
    position = bitget.fetch_open_positions(params['symbol'])
    has_position = len(position) > 0
    
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
    if macd > macd_signal:
        if macd_increasing:
            macd_trend = "Strong Bullish (MACD positive and increasing for 3 bars)"
            macd_status = "BULLISH MOMENTUM"
        else:
            macd_trend = "Weakly Bullish (MACD positive but not consistently increasing)"
            macd_status = "MACD positive but no momentum"
    else:
        if macd_decreasing:
            macd_trend = "Strong Bearish (MACD negative and decreasing for 3 bars)"
            macd_status = "BEARISH MOMENTUM"
        else:
            macd_trend = "Weakly Bearish (MACD negative but not consistently decreasing)"
            macd_status = "MACD negative but no momentum"

    # Log MACD analysis
    logging.info("\n=== MACD Analysis ===")
    logging.info(f"Current MACD-Signal: {macd_diffs[0]:.6f}")
    logging.info(f"Previous MACD-Signal: {macd_diffs[1]:.6f}")
    logging.info(f"2 Candles Ago MACD-Signal: {macd_diffs[2]:.6f}")
    logging.info(f"MACD Trend: {macd_trend}")
    logging.info(f"MACD Status: {macd_status}")
    
    # Trading Signal Analysis
    logging.info("\n=== Trading Signal Analysis ===")
    logging.info("LONG Signal Analysis:")
    logging.info(f"[*] MACD Status: {macd_status}")
    logging.info(f"[*] MACD Momentum: {'YES' if macd_increasing else 'NO'} (needs 3 increasing bars)")
    logging.info(f"[*] Price Above EMA9: {current_price > ema9}")
    logging.info(f"[*] Volume Acceptable: {volume_ratio > 0.7}")
    logging.info(f"[*] Bid/Ask Pressure Bullish: {bid_ask_ratio > 1}")
    
    logging.info("\nSHORT Signal Analysis:")
    logging.info(f"[*] MACD Status: {macd_status}")
    logging.info(f"[*] MACD Momentum: {'YES' if macd_decreasing else 'NO'} (needs 3 decreasing bars)")
    logging.info(f"[*] Price Below EMA9: {current_price < ema9}")
    logging.info(f"[*] Volume Acceptable: {volume_ratio > 0.7}")
    logging.info(f"[*] Bid/Ask Pressure Bearish: {bid_ask_ratio < 1}")

    # Check entry conditions with detailed analysis
    long_entry = check_entry_conditions(data, current_price, current_volume, bid_volume, ask_volume, 'long')
    short_entry = check_entry_conditions(data, current_price, current_volume, bid_volume, ask_volume, 'short')
    
    logging.info("\n=== Trading Recommendation ===")
    if long_entry:
        logging.info(">> STRONG LONG OPPORTUNITY - All conditions met")
    elif short_entry:
        logging.info(">> STRONG SHORT OPPORTUNITY - All conditions met")
    else:
        logging.info(">> WAIT - Conditions not optimal for entry")
        if macd > macd_signal:
            logging.info("Watching for long setup (MACD bullish but other conditions not met)")
        else:
            logging.info("Watching for short setup (MACD bearish but other conditions not met)")

    # If no position is open, cancel all trigger orders
    if not has_position:
        try:
            trigger_orders = bitget.fetch_open_trigger_orders(params['symbol'])
            if trigger_orders:
                logging.info("Cancelling orphaned trigger orders:")
                for order in trigger_orders:
                    bitget.cancel_trigger_order(order['id'], params['symbol'])
                    logging.info(f"✓ Cancelled order {order['id']}")
        except Exception as e:
            logging.error(f"Error cancelling trigger orders: {str(e)}")
    
    # Position Management
    if has_position:
        pos = position[0]
        entry_price = float(pos['info']['openPriceAvg'])
        unrealized_pnl = float(pos['info'].get('unrealisedPnl', 0))
        
        logging.info("\n=== Position Status ===")
        logging.info(f"Current Position: {pos['side'].upper()} {pos['contracts']} contracts")
        logging.info(f"Entry Price: ${entry_price:.2f}")
        logging.info(f"Unrealized PnL: ${unrealized_pnl:.2f}")
        
        if check_exit_conditions(data, current_price, pos['side']):
            logging.info(f"🚨 Exit conditions met for {pos['side']} position")
            logging.info(f"Closing position at ${current_price:.2f}")
            
            # Log the closed trade
            duration = (datetime.now() - pd.to_datetime(pos['info']['openTime'])).total_seconds() / 3600  # hours
            log_trade(
                action="CLOSE",
                side=pos['side'],
                entry_price=entry_price,
                exit_price=current_price,
                contracts=pos['contracts'],
                pnl=unrealized_pnl,
                duration=f"{duration:.1f}h"
            )
            
            bitget.flash_close_position(params['symbol'])
            return

    # Place orders based on conditions
    quantity = calculate_position_size(current_price)
    if quantity is None:
        logging.warning("Failed to calculate position size")
        return

    if long_entry or short_entry:
        side = 'buy' if long_entry else 'sell'
        entry_price = current_price * (1.001 if long_entry else 0.999)
        
        logging.info(f"\n=== Placing New {side.upper()} Order ===")
        logging.info(f"Amount: {quantity} contracts")
        logging.info(f"Entry Price: ${entry_price:.2f}")
        
        # Place entry order and stop loss/take profit
        entry_order = bitget.place_limit_order(
            symbol=params['symbol'],
            side=side,
            amount=quantity,
            price=entry_price
        )
        
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
        
        logging.info(f"Entry order placed: {entry_order['id']}")
        
        stop_loss = entry_price * (1 - params['stop_loss_pct'])
        take_profit = entry_price * (1 + params['take_profit_pct'])
        
        # Log SL/TP levels
        logging.info(f"Stop Loss: ${stop_loss:.2f} ({params['stop_loss_pct']*100:.1f}%)")
        logging.info(f"Take Profit: ${take_profit:.2f} ({params['take_profit_pct']*100:.1f}%)")
        
        sl_order = bitget.place_trigger_market_order(
            symbol=params['symbol'],
            side='sell' if side == 'buy' else 'buy',
            amount=quantity,
            trigger_price=stop_loss,
            reduce=True
        )
        logging.info(f"Stop Loss order placed: {sl_order['id']}")
        
        tp_order = bitget.place_trigger_market_order(
            symbol=params['symbol'],
            side='sell' if side == 'buy' else 'buy',
            amount=quantity,
            trigger_price=take_profit,
            reduce=True
        )
        logging.info(f"Take Profit order placed: {tp_order['id']}")

    # Get position details if any exists
    position_info = "No Position"
    position_entry = "N/A"
    position_pnl = "N/A"
    if has_position:
        try:
            pos = position[0]
            side = pos['side']
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
            
            position_info = f"{side.upper()} {size} contracts"
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
    
    # Create enhanced Telegram message
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
        f"Final Short Signal: {short_entry}\n\n"
        
        f"Trading Recommendation:\n"
    )

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
        trade_logic()
        logging.info("Trade logic execution completed")
    except Exception as e:
        logging.error(f"Critical error in main execution: {str(e)}", exc_info=True)
        sys.exit(1)
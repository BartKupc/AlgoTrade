import json
import ta
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from pathlib import Path
import requests

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
    'limit': 300,
    'trend_bars': 20,
    'adx_period': 14,
    'adx_threshold': 20,
    'volume_period': 100,
    'volume_threshold': 1.1,
    'atr_period': 14,
    'atr_ma_period': 100,
    'atr_threshold': 1.1,
    'stop_loss_pct': 0.05,
    'take_profit_pct': 0.15,
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
        data = bitget.fetch_recent_ohlcv(params['symbol'], params['timeframe'], params['limit']).iloc[:-1]
        
        # Calculate indicators
        data['macd'] = ta.trend.macd(data['close'])
        data['macd_signal'] = ta.trend.macd_signal(data['close'])
        adx_indicator = ta.trend.ADXIndicator(data['high'], data['low'], data['close'], 
                                            window=params['adx_period'])
        data['adx'] = adx_indicator.adx()
        atr_indicator = ta.volatility.AverageTrueRange(data['high'], data['low'], data['close'], 
                                                      window=params['atr_period'])
        data['atr'] = atr_indicator.average_true_range()
        data['atr_ma'] = data['atr'].rolling(params['atr_ma_period']).mean()
        data['volume_ma'] = data['volume'].rolling(params['volume_period']).mean()
        
        return data
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

def trade_logic():
    logging.info("Fetching and processing data...")
    data = fetch_data()
    last_row = data.iloc[-1]
    macd, macd_signal, adx, close = last_row[['macd', 'macd_signal', 'adx', 'close']]
    prev_row = data.iloc[-2]
    prev_macd = prev_row['macd']
    prev_signal = prev_row['macd_signal']
    
    # Log current indicators and MACD crossover conditions
    logging.info(f"Current indicators - MACD: {macd:.6f}, Signal: {macd_signal:.6f}, ADX: {adx:.2f}, Close: {close:.2f}")
    logging.info(f"Previous indicators - MACD: {prev_macd:.6f}, Signal: {prev_signal:.6f}")
    
    # Log MACD conditions separately
    logging.info("MACD Conditions Check:")
    logging.info(f"Current MACD < Signal: {macd < macd_signal} ({macd:.6f} < {macd_signal:.6f})")
    logging.info(f"Previous MACD >= Signal: {prev_macd >= prev_signal} ({prev_macd:.6f} >= {prev_signal:.6f})")
    
    # Check for open positions and orders
    position = bitget.fetch_open_positions(params['symbol'])
    has_position = len(position) > 0
    open_orders = bitget.fetch_open_orders(params['symbol'])
    has_pending_orders = len(open_orders) > 0
    
    logging.info(f"Current state - Has position: {has_position}, Has pending orders: {has_pending_orders}")

    # Check time-based stop loss for shorts
    if has_position and position[0]['side'] == 'short':
        entry_time = position[0]['timestamp']
        current_time = datetime.now().timestamp() * 1000
        hours_in_position = (current_time - entry_time) / (1000 * 60 * 60)
        
        entry_price = float(position[0]['info']['openPriceAvg'])
        unrealized_pnl = (entry_price - close) / entry_price
        
        if hours_in_position >= params['max_short_duration'] and unrealized_pnl < params['short_underwater_threshold']:
            logging.info(f"Closing underwater short position after {hours_in_position:.1f} hours")
            bitget.flash_close_position(params['symbol'])
            return

    # Check pending orders against current conditions
    if has_pending_orders:
        is_long_signal = (macd > macd_signal and prev_macd <= prev_signal and 
                         check_trend(data, 'long') and check_volatility(data))
        is_short_signal = (macd < macd_signal and prev_macd >= prev_signal and 
                         check_trend(data, 'short') and check_volume(data, 'short') and 
                         check_volatility(data))
        
        orders_cancelled = False
        for order in open_orders:
            if not order.get('info', {}).get('reduceOnly', False):  # Not a SL/TP order
                order_matches_signal = (
                    (order['side'] == 'buy' and is_long_signal) or 
                    (order['side'] == 'sell' and is_short_signal)
                )
                
                if not order_matches_signal:
                    bitget.cancel_order(order['id'], params['symbol'])
                    logging.info(f"Cancelled {order['side']} entry order due to changed conditions")
                    return
        
        if not orders_cancelled:
            return

    # Check and place SL/TP orders if missing
    if has_position:
        pos = position[0]
        trigger_orders = bitget.fetch_open_trigger_orders(params['symbol'])
        
        if len(trigger_orders) == 0:
            entry_price = float(pos['info']['openPriceAvg'])
            amount = pos['contracts'] * pos['contractSize']
            
            if pos['side'] == 'long':
                stop_loss = entry_price * (1 - params['stop_loss_pct'])
                take_profit = entry_price * (1 + params['take_profit_pct'])
                close_side = 'sell'
            else:  # short
                stop_loss = entry_price * (1 + params['stop_loss_pct'])
                take_profit = entry_price * (1 - params['take_profit_pct'])
                close_side = 'buy'
            
            logging.info(f"Adding missing SL/TP orders for {pos['side']} position")
            bitget.place_trigger_market_order(
                symbol=params['symbol'],
                side=close_side,
                amount=amount,
                trigger_price=stop_loss,
                reduce=True
            )
            
            bitget.place_trigger_market_order(
                symbol=params['symbol'],
                side=close_side,
                amount=amount,
                trigger_price=take_profit,
                reduce=True
            )

    # Calculate position size for new trades
    quantity = calculate_position_size(close)
    if quantity is None:
        logging.info("Failed to calculate position size")
        return

    # Check if ADX is strong enough
    if adx < params['adx_threshold']:
        logging.info(f"ADX {adx:.2f} below threshold {params['adx_threshold']}")
        return
        
    # Log trend conditions
    is_long_trend = check_trend(data, 'long')
    is_short_trend = check_trend(data, 'short')
    is_volatile = check_volatility(data)
    is_volume_ok = check_volume(data, 'short')
    
    logging.info(f"Conditions - Long trend: {is_long_trend}, Short trend: {is_short_trend}")
    logging.info(f"Additional conditions - Volatility: {is_volatile}, Volume OK: {is_volume_ok}")

    # For Short Entry, log all conditions
    if macd < macd_signal and prev_macd >= prev_signal:
        logging.info("Short Entry Conditions:")
        logging.info(f"1. MACD Crossover: True (Current: {macd:.6f} < {macd_signal:.6f}, Previous: {prev_macd:.6f} >= {prev_signal:.6f})")
        logging.info(f"2. Short Trend: {is_short_trend}")
        logging.info(f"3. Volume OK: {is_volume_ok}")
        logging.info(f"4. Volatility: {is_volatile}")
    else:
        logging.info("Short Entry MACD Crossover not met:")
        if not (macd < macd_signal):
            logging.info(f"Current MACD not below Signal: {macd:.6f} >= {macd_signal:.6f}")
        if not (prev_macd >= prev_signal):
            logging.info(f"Previous MACD not above Signal: {prev_macd:.6f} < {prev_signal:.6f}")

    # Long entry conditions
    if (macd > macd_signal and prev_macd <= prev_signal and
        check_trend(data, 'long') and
        check_volatility(data)
    ):
        logging.info("Long entry conditions met!")
        
        if not has_position or (has_position and position[0]['side'] == 'short'):
            if has_position:
                bitget.flash_close_position(params['symbol'])
            
            entry_price = close * 1.001
            entry_order = bitget.place_limit_order(
                symbol=params['symbol'],
                side='buy',
                amount=quantity,
                price=entry_price
            )
            
            stop_loss = entry_price * (1 - params['stop_loss_pct'])
            take_profit = entry_price * (1 + params['take_profit_pct'])
            
            bitget.place_trigger_market_order(
                symbol=params['symbol'],
                side='sell',
                amount=quantity,
                trigger_price=stop_loss,
                reduce=True
            )
            
            bitget.place_trigger_market_order(
                symbol=params['symbol'],
                side='sell',
                amount=quantity,
                trigger_price=take_profit,
                reduce=True
            )
            
    # Short entry conditions
    elif (macd < macd_signal and prev_macd >= prev_signal and
          check_trend(data, 'short') and
          check_volume(data, 'short') and
          check_volatility(data)
    ):
        logging.info("Short entry conditions met!")
        
        if not has_position or (has_position and position[0]['side'] == 'long'):
            if has_position:
                bitget.flash_close_position(params['symbol'])
            
            entry_price = close * 0.999
            entry_order = bitget.place_limit_order(
                symbol=params['symbol'],
                side='sell',
                amount=quantity,
                price=entry_price
            )
            
            stop_loss = entry_price * (1 + params['stop_loss_pct'])
            take_profit = entry_price * (1 - params['take_profit_pct'])
            
            bitget.place_trigger_market_order(
                symbol=params['symbol'],
                side='buy',
                amount=quantity,
                trigger_price=stop_loss,
                reduce=True
            )
            
            bitget.place_trigger_market_order(
                symbol=params['symbol'],
                side='buy',
                amount=quantity,
                trigger_price=take_profit,
                reduce=True
            )
            
    else:
        logging.info("No entry conditions met")
    
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
            
            # Try different possible field names for unrealized PnL
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

    # Get current market price
    try:
        ticker = bitget.fetch_ticker(params['symbol'])
        current_price = ticker['last']
    except Exception as e:
        logging.error(f"Error fetching current price: {str(e)}")
        current_price = close  # fallback to close price if fetch fails

    # Update message format with both close and current price
    message = (
        f"🤖 Momentum Bot Status Update\n\n"
        f"💰 Position Status:\n"
        f"Current: {position_info}\n"
        f"Entry Price: {position_entry}\n"
        f"Unrealized PnL: {position_pnl}\n\n"
        
        f"📊 Price Action:\n"
        f"Current Market Price: ${current_price:.2f}\n"
        f"Last Candle Close: ${close:.2f}\n"
        f"24h Change: {((close - data.iloc[-24]['close'])/data.iloc[-24]['close']*100):.2f}%\n\n"
        
        f"📈 MACD Indicators:\n"
        f"MACD: {macd:.2f} | Signal: {macd_signal:.2f}\n"
        f"Prev MACD: {prev_macd:.2f} | Prev Signal: {prev_signal:.2f}\n"
        f"MACD < Signal: {macd < macd_signal} "
        f"({'Bearish/Short Signal' if macd < macd_signal else 'Bullish/Long Signal'})\n\n"
        
        f"🎯 Trading Conditions:\n"
        f"ADX: {adx:.2f} (>{params['adx_threshold']}: {adx > params['adx_threshold']})\n"
        f"Long Trend: {is_long_trend}\n"
        f"Short Trend: {is_short_trend}\n"
        f"Volatility OK: {is_volatile}\n"
        f"Volume OK: {is_volume_ok}\n\n"
        
        f"🔄 Entry Conditions:\n"
        f"Long Ready: {macd > macd_signal and prev_macd <= prev_signal and is_long_trend and is_volatile}\n"
        f"Short Ready: {macd < macd_signal and prev_macd >= prev_signal and is_short_trend and is_volume_ok and is_volatile}\n"
    )

    # Check if there are any pending orders
    if has_pending_orders:
        pending_orders_info = "\n🔶 Pending Orders:\n"
        for order in open_orders:
            pending_orders_info += f"- {order['side'].upper()} {order['amount']} @ ${float(order['price']):.2f}\n"
        message += pending_orders_info

    # Check if there are any stop loss/take profit orders
    trigger_orders = bitget.fetch_open_trigger_orders(params['symbol'])
    if trigger_orders:
        tp_sl_info = "\n⚡ Active TP/SL Orders:\n"
        for order in trigger_orders:
            tp_sl_info += f"- {order['side'].upper()} {order['amount']} @ ${float(order['triggerPrice']):.2f}\n"
        message += tp_sl_info

    send_telegram_message(message)

if __name__ == "__main__":
    try:
        logging.info("Starting momentum trading bot...")
        trade_logic()
        logging.info("Trade logic execution completed")
    except Exception as e:
        logging.error(f"Critical error in main execution: {str(e)}", exc_info=True)
        sys.exit(1)
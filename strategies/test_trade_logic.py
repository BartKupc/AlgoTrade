import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path
import logging

# Add parent directory to path to import our strategy
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Import our strategy functions
from strategies.live_moment2 import check_entry_conditions, trade_logic, params

class MockBitget:
    def __init__(self):
        self.positions = []
        self.orders = []
        
    def fetch_open_positions(self, symbol):
        return self.positions
    
    def fetch_ticker(self, symbol):
        return {
            'last': 2150,  # Price below EMA
            'quoteVolume': 2000  # High volume
        }
    
    def fetch_order_book(self, symbol):
        return {
            'bids': [[2149, 800]],
            'asks': [[2151, 1600]]  # More selling pressure
        }
    
    def fetch_balance(self):
        return {'USDT': {'free': 1000}}
    
    def amount_to_precision(self, symbol, amount):
        return str(round(amount, 2))
    
    def fetch_open_trigger_orders(self, symbol):
        return []
    
    def place_market_order(self, symbol, side, amount):
        order = {'id': '123', 'side': side, 'amount': amount}
        self.orders.append(order)
        logging.info(f"Mock placing market order: {side} {amount}")
        return order
    
    def place_trigger_market_order(self, symbol, side, amount, trigger_price, reduce=False):
        order = {'id': '124', 'side': side, 'amount': amount, 'trigger_price': trigger_price}
        self.orders.append(order)
        logging.info(f"Mock placing trigger order: {side} {amount} @ {trigger_price}")
        return order

def create_bearish_market_data():
    """Create sample data with strong bearish signals"""
    data = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=10, freq='1H'),
        'open': [2200, 2190, 2180, 2170, 2160, 2150, 2140, 2130, 2120, 2110],
        'high': [2205, 2195, 2185, 2175, 2165, 2155, 2145, 2135, 2125, 2115],
        'low': [2195, 2185, 2175, 2165, 2155, 2145, 2135, 2125, 2115, 2105],
        'close': [2190, 2180, 2170, 2160, 2150, 2140, 2130, 2120, 2110, 2100],
        'volume': [1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800],
        'macd': [5, 4, 3, 2, 1, 0, -1, -2, -3, -4],
        'macd_signal': [4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0, -0.5],
        'ema9': [2200, 2190, 2180, 2170, 2160, 2150, 2140, 2130, 2120, 2110]
    })
    return data

def test_short_entry():
    logging.info("\n=== Starting Short Entry Test ===")
    
    # Setup mock exchange
    mock_bitget = MockBitget()
    
    # Create bearish market conditions
    data = create_bearish_market_data()
    current_price = 2100  # Significantly below EMA
    current_volume = 2800  # Increasing volume
    bid_volume = 800
    ask_volume = 1600    # More selling pressure
    price_momentum = "down"
    price_change_pct = -0.02  # 2% down
    
    # Test entry conditions
    logging.info("\n=== Testing Entry Conditions ===")
    short_standard, short_momentum, price_trend, macd_values, macd_increasing, macd_decreasing = check_entry_conditions(
        data, current_price, current_volume, bid_volume, ask_volume, 
        price_momentum, price_change_pct, 'short'
    )
    
    logging.info(f"Short Standard: {short_standard}")
    logging.info(f"Short Momentum: {short_momentum}")
    logging.info(f"Price Trend: {price_trend}")
    logging.info(f"MACD Decreasing: {macd_decreasing}")
    
    # Test trade execution
    logging.info("\n=== Testing Trade Execution ===")
    orders_before = len(mock_bitget.orders)
    
    # Run trade logic
    from strategies.live_moment2 import trade_logic
    trade_logic()
    
    orders_after = len(mock_bitget.orders)
    orders_placed = orders_after - orders_before
    
    logging.info(f"\nOrders placed: {orders_placed}")
    if orders_placed > 0:
        logging.info("Trade execution successful!")
        for order in mock_bitget.orders[orders_before:]:
            logging.info(f"Order: {order}")
    else:
        logging.info("No orders placed")
    
    return {
        'short_standard': short_standard,
        'short_momentum': short_momentum,
        'orders_placed': orders_placed,
        'price_trend': price_trend,
        'macd_decreasing': macd_decreasing
    }

if __name__ == "__main__":
    results = test_short_entry()
    print("\nTest Results:")
    for key, value in results.items():
        print(f"{key}: {value}") 
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
    
    def fetch_recent_ohlcv(self, symbol, timeframe, limit):
        # Create sample declining price data
        dates = pd.date_range(end=datetime.now(), periods=limit, freq='1H')
        data = pd.DataFrame({
            'timestamp': dates,
            'open': np.linspace(2200, 2170, limit),
            'high': np.linspace(2205, 2175, limit),
            'low': np.linspace(2195, 2165, limit),
            'close': np.linspace(2200, 2170, limit),
            'volume': np.linspace(1000, 1300, limit)
        })
        return data
    
    def fetch_ticker(self, symbol):
        return {
            'last': 2160,
            'quoteVolume': 1500
        }
    
    def fetch_order_book(self, symbol):
        return {
            'bids': [[2159, 800]],  # price, volume
            'asks': [[2161, 1200]]
        }
    
    def fetch_balance(self):
        return {'USDT': {'free': 1000}}
    
    def amount_to_precision(self, symbol, amount):
        return str(round(amount, 2))
    
    def fetch_open_trigger_orders(self, symbol):
        return []
    
    def place_market_order(self, symbol, side, amount):
        logging.info(f"Mock placing market order: {side} {amount}")
        return {'id': '123'}
    
    def place_trigger_market_order(self, symbol, side, amount, trigger_price, reduce=False):
        logging.info(f"Mock placing trigger order: {side} {amount} @ {trigger_price}")
        return {'id': '124'}

def create_favorable_short_data():
    # Create sample data that matches real conditions
    data = pd.DataFrame({
        'timestamp': pd.date_range(end=datetime.now(), periods=10, freq='1H'),
        'open': [2200, 2195, 2190, 2185, 2180, 2175, 2170, 2165, 2160, 2155],
        'high': [2205, 2200, 2195, 2190, 2185, 2180, 2175, 2170, 2165, 2160],
        'low': [2195, 2190, 2185, 2180, 2175, 2170, 2165, 2160, 2155, 2150],
        'close': [2200, 2195, 2190, 2185, 2180, 2175, 2170, 2165, 2160, 2155],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        'macd': [14, 12, 10, 8, 6, 4, 2, 0, -2, -4],
        'macd_signal': [13, 12, 11, 10, 9, 8, 7, 6, 5, 4],
        'ema9': [2210, 2205, 2200, 2195, 2190, 2185, 2180, 2175, 2170, 2165]
    })
    return data

def test_trade_execution():
    logging.info("=== Starting Trade Logic Test ===")
    
    # Create mock exchange with more realistic data
    mock_bitget = MockBitget()
    
    # Create favorable conditions that match real market
    data = create_favorable_short_data()
    current_price = 2150  # Price below EMA
    current_volume = 2000  # Higher volume
    bid_volume = 800
    ask_volume = 1600    # More asks than bids (bearish)
    price_momentum = "down"
    price_change_pct = -0.02  # 2% down
    
    # Test entry conditions first
    short_standard, short_momentum, price_trend, macd_values, macd_increasing, macd_decreasing = check_entry_conditions(
        data, current_price, current_volume, bid_volume, ask_volume, 
        price_momentum, price_change_pct, 'short'
    )
    
    logging.info("\n=== Entry Conditions Test Results ===")
    logging.info(f"Short Standard: {short_standard}")
    logging.info(f"Short Momentum: {short_momentum}")
    logging.info(f"Price Trend: {price_trend}")
    logging.info(f"MACD Decreasing: {macd_decreasing}")
    
    # Now test full trade logic
    logging.info("\n=== Testing Full Trade Logic ===")
    trade_result = trade_logic()
    
    logging.info("\n=== Trade Logic Test Complete ===")
    return short_standard, short_momentum, trade_result

if __name__ == "__main__":
    print("Starting Trade Logic Test...")
    short_standard, short_momentum, trade_result = test_trade_execution()
    
    print("\nTest Results:")
    print(f"Short Standard Signal: {short_standard}")
    print(f"Short Momentum Signal: {short_momentum}")
    print(f"Trade Result: {trade_result}") 
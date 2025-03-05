import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add parent directory to path to import our strategy
base_dir = Path(__file__).resolve().parents[1]
sys.path.append(str(base_dir))

from strategies.live_moment2 import check_entry_conditions

# Create simulated data that should trigger a short entry
def create_favorable_short_data():
    # Create sample data with declining MACD and price
    data = pd.DataFrame({
        'close': [2200, 2190, 2180, 2170],  # Declining price
        'volume': [1000, 1100, 1200, 1300],  # Increasing volume
        'macd': [10, 8, 6, 4],              # Declining MACD
        'macd_signal': [9, 8, 7, 6],        # MACD signal line
        'ema9': [2200, 2195, 2190, 2185],   # EMA above price
    })
    
    # Simulate current market conditions
    current_price = 2160                     # Price below EMA
    current_volume = 1500                    # High volume
    bid_volume = 800                         # Lower bid volume
    ask_volume = 1200                        # Higher ask volume (bearish)
    price_momentum = "down"                  # Bearish momentum
    price_change_pct = -0.015               # -1.5% price change
    
    return data, current_price, current_volume, bid_volume, ask_volume, price_momentum, price_change_pct

def test_short_entry():
    # Create favorable short conditions
    data, current_price, current_volume, bid_volume, ask_volume, price_momentum, price_change_pct = create_favorable_short_data()
    
    # Test entry conditions
    short_standard, short_momentum, price_trend, macd_values, macd_increasing, macd_decreasing = check_entry_conditions(
        data, current_price, current_volume, bid_volume, ask_volume, 
        price_momentum, price_change_pct, 'short'
    )
    
    # Print detailed results
    print("\n=== Short Entry Test Results ===")
    print(f"Current Price: {current_price}")
    print(f"EMA9: {data['ema9'].iloc[-1]}")
    print(f"Price/EMA Deviation: {((data['ema9'].iloc[-1]/current_price - 1) * 100):.2f}%")
    print(f"MACD Values: {macd_values}")
    print(f"MACD Decreasing: {macd_decreasing}")
    print(f"Volume Ratio: {current_volume/data['volume'].mean():.2f}x")
    print(f"Bid/Ask Ratio: {bid_volume/ask_volume:.2f}")
    print(f"Price Trend: {price_trend*100:.2f}%")
    print("\nResults:")
    print(f"Short Standard: {short_standard}")
    print(f"Short Momentum: {short_momentum}")
    
    return short_standard, short_momentum

if __name__ == "__main__":
    print("Testing Short Entry Conditions...")
    short_standard, short_momentum = test_short_entry()
    
    if short_standard or short_momentum:
        print("\n✅ Short entry conditions met!")
    else:
        print("\n❌ Short entry conditions not met!") 
import time
import pandas as pd
import numpy as np
from bitget_futures import BitgetFutures
import os
import sys
import json
import ta
from datetime import datetime
from pathlib import Path



key_name = 'bitget'
base_dir = Path(__file__).resolve().parents[1]  # Navigate to the root: LiveTradingBotSample
key_path = base_dir / 'config/config.json'

# --- CONFIG ---
params = {
    'symbol': 'BTC/USDT:USDT',
    'timeframe': '1h',
}

# --- AUTHENTICATION ---
print(f"starting execution")
if not key_path.exists():
    raise FileNotFoundError(f"Key file not found at: {key_path}")
with open(key_path, "r") as f:
    api_setup = json.load(f)[key_name]
bitget = BitgetFutures(api_setup)

# # --- Fetch ---
# data = bitget.fetch_ohlcv(params['symbol'], params['timeframe'],"2023-01-01").iloc[:-1]

data = bitget.fetch_ohlcv(params['symbol'], params['timeframe'],"2023-01-01").iloc[:-1]

# --- Save ---
output_dir = base_dir / 'ohlc' 
symbol = params['symbol'].replace('/', '_')
current_time = datetime.now().strftime("%y_%m_%d_%H_%M")
data.to_csv(output_dir/f'{current_time}.csv')



# Test both methods
# print("\nTesting fetch_recent_ohlcv:")
# data = bitget.fetch_recent_ohlcv(params['symbol'], params['timeframe'], 10000)
# print(f"Number of rows in data: {len(data)}")

# print("\nTesting fetch_ohlcv:")
# data1 = bitget.fetch_ohlcv(
#     params['symbol'], 
#     params['timeframe'], 
#     "2024-01-01",
#     "2024-01-31"
# )
# print(f"Number of rows in data1: {len(data1)}")

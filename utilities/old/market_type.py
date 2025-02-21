import pandas as pd
import numpy as np
import pandas_ta as ta
from pathlib import Path
from hurst import compute_Hc

class MarketTypeDetector:
    def __init__(self, data):
        """
        Initialize with either a file path or a DataFrame
        """
        self.data = data
        if isinstance(data, (str, pd.DataFrame)):
            self.df = self.load_data()
        else:
            # Convert Backtesting data object to DataFrame
            self.df = pd.DataFrame({
                'Open': data.Open,
                'High': data.High,
                'Low': data.Low,
                'Close': data.Close,
                'Volume': data.Volume
            }, index=data.index)
            
        self.calculate_indicators()
        self.df['Market_Type'] = self.detect_market_type()

    def load_data(self):
        """Load data from file if path provided, otherwise use DataFrame directly"""
        if isinstance(self.data, str):
            return pd.read_csv(self.data, index_col=0, parse_dates=True)
        return self.data

    def calculate_indicators(self):
        """Calculate technical indicators needed for market type detection"""
        close = self.df['Close']
        high = self.df['High']
        low = self.df['Low']

        # Moving Averages
        self.df['SMA20'] = ta.sma(close, length=20)
        self.df['SMA50'] = ta.sma(close, length=50)
        
        # RSI
        self.df['RSI'] = ta.rsi(close, length=14)

        # Bollinger Bands & Volatility
        bb = ta.bbands(close, length=20)
        self.df['BB_Upper'] = bb['BBU_20_2.0']
        self.df['BB_Lower'] = bb['BBL_20_2.0']
        self.df['Volatility'] = self.df['BB_Upper'] - self.df['BB_Lower']
        self.df['Avg_Volatility'] = self.df['Volatility'].rolling(window=50).mean()

        # ADX (Trend Strength Indicator)
        adx_data = ta.adx(high, low, close, length=14)
        self.df['ADX'] = adx_data['ADX_14']

        # Rate of Change (Momentum Indicator)
        self.df['ROC'] = ta.roc(close, length=10)

        # Calculate Hurst exponent
        self.df['Hurst'] = np.nan
        if len(self.df) >= 100:
            for i in range(100, len(self.df)):
                try:
                    H, _, _ = compute_Hc(self.df['Close'].iloc[i-100:i].values, kind='price', simplified=True)
                    self.df.iloc[i, self.df.columns.get_loc('Hurst')] = H
                except:
                    self.df.iloc[i, self.df.columns.get_loc('Hurst')] = 0.5  # Neutral value if calculation fails

    def detect_market_type(self):
        """Detect market type based on technical indicators"""
        market_types = []
        roc_threshold = 2  # Define ROC threshold for momentum markets

        for i in range(len(self.df)):
            if i < 50:  # Not enough data for proper classification
                market_types.append("Momentum")  # Default type
                continue
            
            # Get current indicator values
            volatility = self.df['Volatility'].iloc[i]
            avg_volatility = self.df['Avg_Volatility'].iloc[i]
            rsi = self.df['RSI'].iloc[i]
            close = self.df['Close'].iloc[i]
            sma20 = self.df['SMA20'].iloc[i]
            sma50 = self.df['SMA50'].iloc[i]
            hurst = self.df['Hurst'].iloc[i] if not np.isnan(self.df['Hurst'].iloc[i]) else 0.5
            adx = self.df['ADX'].iloc[i] if not np.isnan(self.df['ADX'].iloc[i]) else 20
            roc = self.df['ROC'].iloc[i] if not np.isnan(self.df['ROC'].iloc[i]) else 0

            # Market classification logic
            if adx > 25:  # Strong trend present
                if hurst > 0.6:
                    market_types.append("Strong Trend")  # Moving Averages Work
                else:
                    market_types.append("Momentum")  # Trend exists but not strong enough for MAs
            elif adx < 20:  # Weak trend, possibly range-bound or mean-reverting
                if hurst < 0.4 and (rsi > 70 or rsi < 30):
                    market_types.append("Mean Reverting")
                elif 0.45 <= hurst <= 0.55:
                    market_types.append("Range-Bound")  # Sideways market
                else:
                    market_types.append("Choppy")  # Random Walk
            else:  # Mid-level ADX, check volatility & momentum
                if volatility > avg_volatility * 1.5:
                    market_types.append("Volatile Breakout")  # Donchian or breakout strategies
                elif roc > roc_threshold:
                    market_types.append("Momentum")  # Strong movement, momentum trading
                else:
                    market_types.append("Range-Bound")  # Default to range-bound

        return np.array(market_types)

    def get_latest_market_condition(self):
        return self.df[['Close', 'SMA20', 'SMA50', 'RSI', 'ADX', 'ROC', 'Volatility', 'Avg_Volatility', 'Hurst', 'Market_Type']].tail(1)

    def save_to_csv(self, save_path):
        self.df.to_csv(save_path)
        print(f"Market type saved to: {save_path}")

# Example usage
if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent
    file_path = base_dir / 'ohlc' / '25_02_12_09_54.csv'
    save_path = base_dir / 'market_type.csv'
    
    detector = MarketTypeDetector(file_path)
    print(detector.get_latest_market_condition())
    detector.save_to_csv(save_path)

AlgoTrade - Cryptocurrency Algorithmic Trading Bot

This repository contains a Python-based algorithmic trading bot designed for cryptocurrency markets, utilizing technical analysis indicators such as MACD, EMA, and real-time volume analysis to make informed trading decisions. The bot is integrated with Bitget Futures exchange and supports real-time trade notifications via Telegram.

Features

Automated cryptocurrency trading with momentum-based strategies

Supports long and short positions with configurable parameters

Incorporates MACD, EMA, volume analysis, and price momentum indicators

Implements protective measures like stop loss and take profit

Telegram integration for real-time alerts and trade notifications

Comprehensive logging for trades and market analysis

Prerequisites

Python 3.8 or later

ta library for technical analysis

Access to Bitget Futures account with API keys

Telegram bot for notifications

Installation

Clone the repository:

git clone https://github.com/BartKupc/AlgoTrade.git
cd AlgoTrade

Install dependencies:

pip install -r requirements.txt

Configure your API keys:

Create a file named config.json in the /config directory.

Structure your config.json as follows:

{
  "bitget": {
    "api_key": "your-api-key",
    "api_secret": "your-api-secret",
    "passphrase": "your-api-passphrase"
  }
}

Configure Telegram notifications:

Update the TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in the script with your Telegram bot credentials.

Running the Bot

To start the bot, execute:

python live_moment2.py

Customizing the Bot

You can adjust trading parameters by modifying the params dictionary in live_moment2.py:

symbol: Trading pair (e.g., ETH/USDT)

timeframe: Data interval (e.g., 1h, 15m)

Leverage, EMA period, thresholds, stop loss, and take profit percentages

Monitoring and Logs

Trade logs and market analysis are stored in the /trade_logs directory. You can monitor real-time bot activity and historical performance.

Safety and Risk Management

Always perform thorough testing in paper trading mode or with minimal capital before deploying significant funds.

Monitor trades regularly and maintain backups of your configuration and logs.

Contributing

Contributions are welcome! Please submit pull requests or report issues to help enhance the bot's functionality and reliability.

Disclaimer

This bot is for educational purposes only. Trading cryptocurrencies involves significant risk. The authors are not responsible for any financial losses incurred.
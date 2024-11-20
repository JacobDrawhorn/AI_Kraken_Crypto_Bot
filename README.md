# AI-Driven Kraken Trading Bot

## Overview
This is an advanced AI-powered cryptocurrency trading bot designed specifically for the Kraken exchange. The bot uses machine learning techniques to predict price movements and execute trades automatically.

## Features
- AI-driven trading strategy
- Support for multiple trading pairs (Bitcoin, Ethereum)
- Machine learning price prediction
- Secure API key management
- Comprehensive logging

## Prerequisites
- Python 3.8+
- Kraken API Key and Secret Key

## Installation
1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure API Credentials
   - Copy `.env.example` to `.env`
   - Add your Kraken API Key and Secret Key

## Configuration
Edit `kraken_bot.py` to:
- Modify trading pairs
- Adjust trading strategy
- Customize machine learning model

## Running the Bot
```bash
python kraken_bot.py
```

## Disclaimer
Trading cryptocurrencies involves significant risk. Use this bot at your own risk and always monitor its performance.

## License
MIT License

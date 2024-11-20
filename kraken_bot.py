import os
import sys
import logging
import krakenex
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import time
from datetime import datetime, timedelta
import math
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input
from tensorflow.keras.optimizers import Adam
import requests
from typing import List, Dict

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class KrakenAITrader:
    def __init__(self):
        """Initialize the trading bot with configuration"""
        # Load environment variables
        load_dotenv()
        
        # API setup
        self.k = krakenex.API()
        api_key = os.getenv('KRAKEN_API_KEY')
        api_secret = os.getenv('KRAKEN_SECRET_KEY')
        
        if not api_key or not api_secret:
            raise ValueError("Kraken API credentials not found in environment variables")
            
        self.k.key = api_key
        self.k.secret = api_secret
        
        # Trading parameters
        self.trading_pairs = [
            'XXBTZUSD',  # Bitcoin
            'XETHZUSD',  # Ethereum
            'SOLUSD',    # Solana
            'MATICUSD',  # Polygon
            'ATOMUSD',   # Cosmos
            'DOTUSD',    # Polkadot
            'ADAUSD',    # Cardano
            'LINKUSD',   # Chainlink
            'UNIUSD',    # Uniswap
            'AAVEUSD'    # Aave
        ]
        
        # Position sizing and risk parameters
        self.min_position_size = 25.0  # Minimum position size in USD
        self.max_position_size = 50.0  # Maximum position size in USD
        self.position_sizing_factor = 0.4  # Position sizing multiplier
        self.max_positions = 5  # Maximum number of concurrent positions
        self.confidence_threshold = 0.10  # Minimum prediction confidence
        
        # Trading interval
        self.interval = 60  # 1 minute between trades
        
        # Balance tracking
        self.available_balance = 0.0
        self.reserved_balance = 0.0
        self.daily_pnl = 0.0
        
        # Initialize ML components
        self.model = None
        self.scaler = None
        self.feature_scalers = {}
        self.sequence_length = 10
        self.initialize_model()

    def initialize_model(self):
        """Initialize LSTM model for price prediction"""
        try:
            # Model parameters
            n_features = 5  # open, high, low, close, volume
            
            # Define input layer
            inputs = Input(shape=(self.sequence_length, n_features))
            
            # LSTM layers
            x = LSTM(50, return_sequences=True)(inputs)
            x = Dropout(0.2)(x)
            x = LSTM(50, return_sequences=False)(x)
            x = Dropout(0.2)(x)
            
            # Output layer
            outputs = Dense(1, activation='linear')(x)
            
            # Create model
            model = Model(inputs=inputs, outputs=outputs)
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse',
                metrics=['mae']
            )
            
            self.model = model
            logger.info("LSTM model initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            self.model = None

    def prepare_features(self, data: pd.DataFrame, pair: str = None) -> pd.DataFrame:
        """Prepare features from OHLCV data"""
        try:
            # Create copy of relevant columns (only use 5 features)
            features = data[['open', 'high', 'low', 'close', 'volume']].copy()
            
            # Convert to float and handle NaN
            for col in features.columns:
                features[col] = pd.to_numeric(features[col], errors='coerce')
            
            # Fill NaN values using forward fill then backward fill
            features = features.ffill().bfill()
            
            # Scale features if needed
            if pair is not None:
                if pair not in self.feature_scalers:
                    self.feature_scalers[pair] = StandardScaler()
                    features_scaled = self.feature_scalers[pair].fit_transform(features)
                else:
                    features_scaled = self.feature_scalers[pair].transform(features)
                
                # Convert back to DataFrame
                features = pd.DataFrame(
                    features_scaled,
                    columns=features.columns,
                    index=features.index
                )
            
            return features
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            return None

    def prepare_data(self, data: pd.DataFrame) -> tuple:
        """Prepare data for LSTM model"""
        try:
            # Extract features
            features = ['open', 'high', 'low', 'close', 'volume']
            
            # Ensure all required columns exist
            for feature in features:
                if feature not in data.columns:
                    raise ValueError(f"Missing required column: {feature}")
            
            # Create sequences
            sequences = []
            targets = []
            
            for i in range(len(data) - self.sequence_length):
                sequence = data[features].iloc[i:(i + self.sequence_length)].values
                target = data['close'].iloc[i + self.sequence_length]
                sequences.append(sequence)
                targets.append(target)
            
            if not sequences:
                logger.error("No valid sequences created from data")
                return None, None
                
            X = np.array(sequences)
            y = np.array(targets)
            
            # Initialize scaler if needed
            if not hasattr(self, 'scaler') or self.scaler is None:
                self.scaler = StandardScaler()
                # Fit scaler on all data points
                X_reshape = X.reshape(-1, len(features))
                self.scaler.fit(X_reshape)
            
            # Scale the sequences
            X_scaled = np.zeros_like(X)
            for i in range(len(X)):
                X_scaled[i] = self.scaler.transform(X[i])
            
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            return None, None

    def train_model(self, pair: str):
        """Train LSTM model for a specific pair"""
        try:
            # Get historical data
            data = self.get_ohlcv_data(pair)
            if data is None:
                return
            
            # Prepare features
            features = self.prepare_features(data, pair)
            if features is None:
                return
            
            # Calculate returns for target
            returns = features['close'].pct_change().shift(-1)  # Next period's return
            features['target'] = returns
            
            # Drop rows with NaN
            features = features.dropna()
            
            if len(features) < self.sequence_length + 1:
                logger.error(f"Insufficient data for {pair} after preprocessing")
                return
            
            # Create sequences for training
            X, y = [], []
            for i in range(len(features) - self.sequence_length):
                # Input sequence
                seq = features[['open', 'high', 'low', 'close', 'volume']].iloc[i:i + self.sequence_length].values
                # Target is the return for the next period
                target = features['target'].iloc[i + self.sequence_length]
                X.append(seq)
                y.append(target)
            
            X = np.array(X)
            y = np.array(y)
            
            # Split into train and validation sets
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            # Train model with early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=32,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Log training results
            val_loss = history.history['val_loss'][-1]
            train_loss = history.history['loss'][-1]
            logger.info(f"Model trained for {pair} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
        except Exception as e:
            logger.error(f"Error training model for {pair}: {str(e)}")

    def predict_next_price(self, pair: str) -> tuple:
        """Predict the next price movement for a trading pair"""
        try:
            # Get current price
            current_price = self.get_current_price(pair)
            if current_price is None:
                return None, None, None, None
            
            # Get historical data
            data = self.get_ohlcv_data(pair)
            if data is None:
                return None, None, None, None
            
            # Prepare features
            features = self.prepare_features(data, pair)
            if features is None:
                return None, None, None, None
            
            if len(features) < self.sequence_length:
                logger.error(f"Insufficient data for prediction: {len(features)} < {self.sequence_length}")
                return None, None, None, None
            
            # Create sequence for prediction (last sequence_length rows)
            sequence = features.iloc[-self.sequence_length:].values
            sequence = np.expand_dims(sequence, axis=0)  # Add batch dimension
            
            # Make prediction
            predicted_return = float(self.model.predict(sequence, verbose=0)[0][0])
            
            # Calculate volatility-based confidence
            volatility = self.calculate_volatility(pair)
            if volatility is None or volatility == 0:
                confidence = 0
            else:
                # Normalize predicted return by volatility
                confidence = min(abs(predicted_return) / (2 * volatility), 1.0)
            
            # Clip prediction based on volatility
            max_return = min(0.05, 2 * volatility)  # Cap at 5% or 2x volatility, whichever is smaller
            predicted_return = np.clip(predicted_return, -max_return, max_return)
            
            # Calculate predicted price
            predicted_price = current_price * (1 + predicted_return)
            
            logger.info(f"Price prediction for {pair}: ${predicted_price:.2f} (current: ${current_price:.2f}, "
                       f"predicted return: {predicted_return:.2%}, confidence: {confidence:.2f})")
            
            return predicted_price, current_price, predicted_return, confidence
            
        except Exception as e:
            logger.error(f"Error predicting price for {pair}: {str(e)}")
            return None, None, None, None

    def calculate_position_size(self, pair: str, predicted_return: float, available_balance: float) -> float:
        """Calculate position size based on prediction confidence and available balance"""
        try:
            # Get volatility
            volatility = self.calculate_volatility(pair)
            if volatility is None:
                return None
                
            # Calculate base position size (0.2 = 20% of available balance)
            base_position_size = available_balance * 0.2
            
            # Adjust for volatility (reduce size for high volatility)
            volatility_factor = 1 / (1 + volatility)
            position_size = base_position_size * volatility_factor
            
            # Adjust for prediction confidence
            confidence = min(abs(predicted_return / volatility), 1.0)
            position_size *= confidence
            
            # Apply minimum and maximum constraints
            position_size = max(min(position_size, self.max_position_size), self.min_position_size)
            
            # Ensure we don't exceed available balance
            position_size = min(position_size, available_balance)
            
            logger.info(f"Calculated position size for {pair}: ${position_size:.2f} (confidence: {confidence:.2f}, max allowed: ${position_size:.2f})")
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {str(e)}")
            return None

    def place_order(self, pair: str, order_type: str, volume: float, price: float = None) -> dict:
        """Place an order on Kraken"""
        try:
            # Validate inputs
            if not pair or not order_type or not volume:
                logger.error("Missing required order parameters")
                return None
            
            # Get current balances
            balance = self.k.query_private('Balance')
            if 'error' in balance and balance['error']:
                logger.error(f"Error getting balance: {balance['error']}")
                return None
            
            # Check if we have enough balance
            if order_type == 'buy':
                # For buy orders, check USD balance
                usd_balance = float(balance['result'].get('ZUSD', 0))
                required_usd = volume * (price or self.get_current_price(pair))
                if usd_balance < required_usd:
                    logger.error(f"Insufficient USD balance for buy order. Required: ${required_usd:.2f}, Available: ${usd_balance:.2f}")
                    return None
            else:
                # For sell orders, check crypto balance
                # Extract base currency from pair (e.g., XBT from XXBTZUSD)
                base_currency = pair.replace('USD', '').replace('Z', '')
                if base_currency.startswith('X'):
                    base_currency = base_currency[1:]  # Remove X prefix
                crypto_balance = float(balance['result'].get(base_currency, 0))
                if crypto_balance < volume:
                    logger.error(f"Insufficient {base_currency} balance for sell order. Required: {volume}, Available: {crypto_balance}")
                    return None
                
            # Set up base order parameters
            params = {
                'pair': pair,
                'type': order_type,  # 'buy' or 'sell'
                'ordertype': 'market' if not price else 'limit',
                'volume': str(volume)
            }
            
            # Add price for limit orders
            if price:
                params['price'] = str(price)
            
            # Place the order
            logger.info(f"Placing {order_type} order for {pair}: {params}")
            result = self.k.query_private('AddOrder', params)
            
            if result.get('error'):
                logger.error(f"Order error: {result['error']}")
                return None
                
            logger.info(f"Order placed successfully: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return None

    def get_open_positions(self) -> list:
        """Get currently open positions"""
        try:
            response = self.k.query_private('OpenPositions')
            if 'error' in response and response['error']:
                logger.error(f"Error getting open positions: {response['error']}")
                return []
            return list(response['result'].values())
        except Exception as e:
            logger.error(f"Error fetching open positions: {str(e)}")
            return []

    def get_current_price(self, pair: str) -> float:
        """Get current price for a trading pair"""
        try:
            ticker = self.k.query_public('Ticker', {'pair': pair})
            if 'error' in ticker and ticker['error']:
                logger.error(f"Error getting ticker for {pair}: {ticker['error']}")
                return None
            return float(ticker['result'][pair]['c'][0])
        except Exception as e:
            logger.error(f"Error getting current price for {pair}: {str(e)}")
            return None

    def get_ohlcv_data(self, pair: str) -> pd.DataFrame:
        """Get OHLCV data for a trading pair"""
        try:
            # Get recent OHLCV data
            since = str(int(time.time() - 86400))  # Last 24 hours
            ohlcv = self.k.query_public('OHLC', {'pair': pair, 'interval': 1, 'since': since})
            
            if ohlcv['error']:
                logger.error(f"Error getting OHLCV data for {pair}: {ohlcv['error']}")
                return None
                
            data = ohlcv['result'][pair]
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            for col in ['open', 'high', 'low', 'close', 'vwap', 'volume']:
                df[col] = df[col].astype(float)
                
            return df
            
        except Exception as e:
            logger.error(f"Error getting OHLCV data for {pair}: {str(e)}")
            return None

    def create_sequences(self, features: pd.DataFrame, sequence_length: int) -> np.ndarray:
        """Create sequences for LSTM model"""
        try:
            feature_cols = ['returns', 'log_returns', 'volatility', 'rsi', 'macd']
            features_subset = features[feature_cols].values
            
            sequences = []
            for i in range(len(features_subset) - sequence_length + 1):
                seq = features_subset[i:(i + sequence_length)]
                sequences.append(seq)
            
            return np.array(sequences)
            
        except Exception as e:
            logger.error(f"Error creating sequences: {str(e)}")
            return None

    def calculate_order_volume(self, pair: str, position_size: float, current_price: float) -> float:
        """Calculate order volume in base currency units"""
        try:
            info = self.k.query_public('AssetPairs', {'pair': pair})
            if 'error' in info and info['error']:
                logger.error(f"Error getting pair info for {pair}: {info['error']}")
                return 0
                
            pair_info = info['result'][pair]
            min_order = float(pair_info.get('ordermin', 0))
            
            volume = abs(position_size) / current_price
            
            decimals = int(pair_info.get('lot_decimals', 8))
            volume = round(volume, decimals)
            
            if volume < min_order:
                logger.info(f"Order volume {volume} below minimum {min_order} for {pair}")
                return 0
                
            return volume
            
        except Exception as e:
            logger.error(f"Error calculating order volume for {pair}: {str(e)}")
            return 0

    def update_balance(self) -> float:
        """Update and return available balance"""
        try:
            # Get trades from last 24 hours
            since = int(time.time() - 24*60*60)  # 24 hours ago
            trades = self.k.query_private('TradesHistory', {'start': since})
            
            if 'error' in trades and trades['error']:
                logger.error(f"Error getting trade history: {trades['error']}")
                return 0
            
            # Calculate realized P&L from closed trades
            realized_pnl = 0
            for trade_id, trade in trades['result'].get('trades', {}).items():
                try:
                    cost = float(trade['cost'])
                    fee = float(trade['fee'])
                    if trade['type'] == 'buy':
                        realized_pnl -= (cost + fee)
                    else:  # sell
                        realized_pnl += (cost - fee)
                except (KeyError, ValueError) as e:
                    logger.error(f"Error calculating trade P&L: {str(e)}")
                    continue
            
            # Calculate unrealized P&L from open positions
            unrealized_pnl = 0
            positions = self.get_open_positions()
            
            for pos in positions:
                try:
                    current_price = self.get_current_price(pos['pair'])
                    if current_price is None:
                        continue
                        
                    volume = float(pos['vol'])
                    cost = float(pos['cost'])
                    fee = float(pos['fee'])
                    
                    # Calculate position value at current price
                    current_value = abs(volume) * current_price
                    
                    # Calculate P&L based on position type (long/short)
                    if volume > 0:  # Long position
                        unrealized_pnl += current_value - (cost + fee)
                    else:  # Short position
                        unrealized_pnl += (cost - fee) - current_value
                        
                except (KeyError, ValueError) as e:
                    logger.error(f"Error calculating position P&L: {str(e)}")
                    continue
            
            total_pnl = realized_pnl + unrealized_pnl
            
            # Log detailed P&L breakdown
            logger.info(f"Daily P&L Breakdown - Realized: ${realized_pnl:.2f}, Unrealized: ${unrealized_pnl:.2f}, Total: ${total_pnl:.2f}")
            
            # Get account balance
            balance = self.k.query_private('Balance')
            if 'error' in balance and balance['error']:
                logger.error(f"Error getting balance: {balance['error']}")
                return 0
            
            # Get USD balance (ZUSD)
            usd_balance = float(balance['result'].get('ZUSD', 0))
            
            # Get open positions
            positions = self.get_open_positions()
            
            # Calculate total position value
            position_value = 0
            for pos in positions:
                try:
                    volume = abs(float(pos['vol']))
                    current_price = self.get_current_price(pos['pair'])
                    if current_price is not None:
                        position_value += volume * current_price
                except (KeyError, ValueError) as e:
                    logger.error(f"Error calculating position value: {str(e)}")
                    continue
            
            # Update balance tracking
            self.reserved_balance = position_value
            self.available_balance = max(0, usd_balance - position_value)
            
            # Calculate daily P&L
            self.daily_pnl = total_pnl
            
            # Log balance information with more detail
            logger.info(f"Balance Status - Total: ${usd_balance:.2f}, Available: ${self.available_balance:.2f}, "
                       f"Reserved: ${self.reserved_balance:.2f}, Daily P&L: ${self.daily_pnl:.2f}")
            
            return self.available_balance
            
        except Exception as e:
            logger.error(f"Error updating balance: {str(e)}")
            return 0

    def calculate_daily_pnl(self) -> float:
        """Calculate daily profit/loss"""
        try:
            # Get trades from last 24 hours
            since = int(time.time() - 24*60*60)  # 24 hours ago
            trades = self.k.query_private('TradesHistory', {'start': since})
            
            if 'error' in trades and trades['error']:
                logger.error(f"Error getting trade history: {trades['error']}")
                return 0
            
            # Calculate realized P&L from closed trades
            realized_pnl = 0
            for trade_id, trade in trades['result'].get('trades', {}).items():
                try:
                    cost = float(trade['cost'])
                    fee = float(trade['fee'])
                    if trade['type'] == 'buy':
                        realized_pnl -= (cost + fee)
                    else:  # sell
                        realized_pnl += (cost - fee)
                except (KeyError, ValueError) as e:
                    logger.error(f"Error calculating trade P&L: {str(e)}")
                    continue
            
            # Calculate unrealized P&L from open positions
            unrealized_pnl = 0
            positions = self.get_open_positions()
            
            for pos in positions:
                try:
                    current_price = self.get_current_price(pos['pair'])
                    if current_price is None:
                        continue
                        
                    volume = float(pos['vol'])
                    cost = float(pos['cost'])
                    fee = float(pos['fee'])
                    
                    # Calculate position value at current price
                    current_value = abs(volume) * current_price
                    
                    # Calculate P&L based on position type (long/short)
                    if volume > 0:  # Long position
                        unrealized_pnl += current_value - (cost + fee)
                    else:  # Short position
                        unrealized_pnl += (cost - fee) - current_value
                        
                except (KeyError, ValueError) as e:
                    logger.error(f"Error calculating position P&L: {str(e)}")
                    continue
            
            total_pnl = realized_pnl + unrealized_pnl
            
            # Log detailed P&L breakdown
            logger.info(f"Daily P&L Breakdown - Realized: ${realized_pnl:.2f}, Unrealized: ${unrealized_pnl:.2f}, Total: ${total_pnl:.2f}")
            
            return total_pnl
            
        except Exception as e:
            logger.error(f"Error calculating daily P&L: {str(e)}")
            return 0

    def close_all_positions(self):
        """Close all open positions"""
        try:
            positions = self.get_open_positions()
            if not positions:
                logger.info("No open positions to close")
                return
                
            for pos in positions:
                try:
                    pair = pos['pair']
                    volume = abs(float(pos['vol']))
                    side = "sell" if float(pos['vol']) > 0 else "buy"
                    
                    self.place_order(pair, "market", volume)
                    logger.info(f"Closed position for {pair}")
                except Exception as e:
                    logger.error(f"Error closing position: {str(e)}")
                    continue
            
            logger.info("All positions closed")
            
        except Exception as e:
            logger.error(f"Error closing all positions: {str(e)}")

    def calculate_volatility(self, pair: str) -> float:
        """Calculate current market volatility using standard deviation of returns"""
        try:
            # Get recent OHLCV data
            since = str(int(time.time() - 86400))  # Last 24 hours
            ohlcv = self.k.query_public('OHLC', {'pair': pair, 'interval': 5, 'since': since})
            
            if ohlcv['error']:
                logger.error(f"Error getting OHLC data for {pair}: {ohlcv['error']}")
                return 0.1  # Default volatility
            
            # Calculate returns
            prices = pd.DataFrame(ohlcv['result'][pair])
            if len(prices) < 2:
                return 0.1  # Default volatility if not enough data
            
            prices = prices[4].astype(float)  # Close prices
            returns = np.log(prices / prices.shift(1)).dropna()
            
            # Calculate volatility
            volatility = np.std(returns) * np.sqrt(288)  # Annualized (288 5-min periods in day)
            
            # Cap volatility to reasonable range
            volatility = min(max(volatility, 0.1), 1.0)
            
            logger.debug(f"Calculated volatility for {pair}: {volatility:.4f}")
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility for {pair}: {str(e)}")
            return 0.1  # Default volatility on error

def main():
    """Main trading loop"""
    try:
        # Load environment variables
        load_dotenv()
        
        # Initialize trading bot
        trader = KrakenAITrader()
        
        logger.info("Starting trading bot...")
        
        while True:
            try:
                # Update balance
                trader.update_balance()
                
                # Log trading status
                logger.info(f"Trading Status - Daily P&L: ${trader.daily_pnl:.2f}")
                
                # Process trading pairs if we have available balance
                if trader.available_balance > trader.min_position_size:
                    for pair in trader.trading_pairs:
                        try:
                            # Get current price and predicted price
                            current_price = trader.get_current_price(pair)
                            if current_price is None:
                                continue
                            
                            predicted_price, current_price, predicted_return, confidence = trader.predict_next_price(pair)
                            if predicted_price is None:
                                continue
                            
                            # Skip if prediction confidence is too low
                            if confidence < trader.confidence_threshold:
                                logger.info(f"Skipping {pair} - Low confidence prediction: {confidence:.4f}")
                                continue
                            
                            # Calculate order volume in base currency
                            position_size = trader.calculate_position_size(pair, predicted_return, trader.available_balance)
                            if position_size == 0:
                                logger.info(f"Skipping {pair} - Position size too small: ${position_size:.2f}")
                                continue
                            
                            volume = trader.calculate_order_volume(pair, position_size, current_price)
                            if volume == 0:
                                logger.info(f"Skipping {pair} - Order volume too small: {volume:.8f}")
                                continue
                            
                            # Place order based on prediction
                            if predicted_return > 0:
                                # Place buy order
                                trader.place_order(
                                    pair=pair,
                                    order_type="buy",
                                    volume=volume
                                )
                                logger.info(f"Placed buy order for {pair} - Size: ${position_size:.2f}, Volume: {volume:.8f}")
                            else:
                                # Place sell order
                                trader.place_order(
                                    pair=pair,
                                    order_type="sell",
                                    volume=volume
                                )
                                logger.info(f"Placed sell order for {pair} - Size: ${position_size:.2f}, Volume: {volume:.8f}")
                        
                        except Exception as e:
                            logger.error(f"Error processing {pair}: {str(e)}")
                            continue
                    
                else:
                    logger.warning(f"Insufficient balance - Available: ${trader.available_balance:.2f}, "
                                 f"Minimum required: ${trader.min_position_size:.2f}")
                    time.sleep(trader.interval)
                    continue
                
                # Sleep for the trading interval
                time.sleep(trader.interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {str(e)}")
                time.sleep(60)  # Wait before retrying
                
    except KeyboardInterrupt:
        logger.info("Received shutdown signal...")
        trader.close_all_positions()
        
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        
    finally:
        logger.info("Shutting down trading bot...")

if __name__ == "__main__":
    main()

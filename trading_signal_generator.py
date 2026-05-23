"""
PIPRAISIER-INSPIRED: AI-DRIVEN TRADING SIGNAL GENERATOR
=========================================================

A machine learning-powered trading analytics platform that:
1. Ingests real-time market data (crypto, stocks, forex)
2. Engineers technical indicators as ML features
3. Predicts price direction using Random Forest
4. Generates buy/sell signals
5. Tracks performance metrics (win rate, profit factor, Sharpe ratio)

This project demonstrates:
- Data Engineering: Real-time API data ingestion
- Feature Engineering: Technical analysis indicators
- Machine Learning: Classification model for trend prediction
- Performance Analytics: Trading metrics calculation
- Software Architecture: Modular, extensible design

Created for: Pipraisier Trading Internship
Author: Madhav Bansal
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TradingSignalGenerator:
    """
    CORE CLASS: Handles data collection, feature engineering, ML model training, and signal generation.
    
    Why this architecture?
    - Modular: Separate concerns (data, features, model, signals)
    - Scalable: Can add new assets/markets easily
    - Testable: Each component can be tested independently
    - Maintainable: Clear responsibility for each method
    """
    
    def __init__(self, lookback_days=365):
        """
        Initialize the signal generator.
        
        Args:
            lookback_days: How many days of historical data to use for training
                          (365 = 1 year of data for robust model)
        """
        self.lookback_days = lookback_days
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = None
        self.signals_history = []
        
    def fetch_market_data(self, ticker, market_type='crypto'):
        """
        STEP 1: DATA INGESTION
        Fetches real-time OHLCV data from Yahoo Finance API.
        
        Why yfinance?
        - Free, no API key required
        - Covers crypto (BTC-USD), stocks (SPY), forex (EURUSD=X)
        - Real-time data with minimal latency
        
        Args:
            ticker: Symbol (e.g., 'BTC-USD', 'SPY', 'EURUSD=X')
            market_type: 'crypto', 'stock', or 'forex'
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"📊 Fetching {market_type.upper()} data for {ticker}...")
        
        try:
            # Download historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=self.lookback_days)
            
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if data.empty:
                print(f"❌ No data found for {ticker}")
                return None
            
            # Handle MultiIndex columns (yfinance sometimes returns these)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            
            print(f"✅ Downloaded {len(data)} candles for {ticker}")
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
            return None

    def get_realtime_price(self, ticker):
        """
        Fetches the current live price for a ticker.
        Uses yfinance's fast_info which pulls the latest market price.
        
        Works for:
        - Crypto: 'BTC-USD', 'ETH-USD', 'XRP-USD'  (24/7)
        - Stocks: 'SPY', 'QQQ'                       (market hours only)
        - Forex:  'EURUSD=X', 'USDINR=X'             (weekdays)
        """
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.fast_info
            price = info['last_price']
            return round(price, 6)
        except Exception as e:
            return None

    def engineer_features(self, data):
        """
        STEP 2: FEATURE ENGINEERING
        Creates technical indicators as features for ML model.
        
        WHY THESE INDICATORS?
        
        1. MOMENTUM INDICATORS (Trend strength):
           - RSI (14): Identifies overbought/oversold conditions
           - MACD: Momentum + trend direction
           
        2. VOLATILITY INDICATORS (Market uncertainty):
           - Bollinger Bands: Price range expectations
           - ATR: Average True Range (how much price moves daily)
           
        3. TREND INDICATORS (Direction):
           - EMA: Exponential Moving Average (recent price emphasis)
           - SMA: Simple Moving Average (overall trend)
           
        4. VOLUME (Confirmation):
           - Volume-based signals (is the move real?)
           
        Machine Learning Note:
        - These aren't magic; they're proxies for market psychology
        - Your ML model learns which combinations matter most
        - The more you feed it, the better it learns patterns
        
        Returns:
            DataFrame with original OHLCV + technical indicators
        """
        print("🔧 Engineering technical features...")
        
        df = data.copy()
        
        # MOMENTUM INDICATORS
        # RSI (Relative Strength Index): Measures speed/magnitude of price changes
        # High RSI (>70) = overbought, Low RSI (<30) = oversold
        rsi_14 = self._calculate_rsi(df['Close'], period=14)
        df['RSI_14'] = rsi_14.squeeze() if hasattr(rsi_14, 'squeeze') else rsi_14
        rsi_7 = self._calculate_rsi(df['Close'], period=7)
        df['RSI_7'] = rsi_7.squeeze() if hasattr(rsi_7, 'squeeze') else rsi_7
        
        # MACD (Moving Average Convergence Divergence): Trend + momentum
        macd, macd_signal, macd_hist = self._calculate_macd(df['Close'])
        df['MACD'] = macd.squeeze() if hasattr(macd, 'squeeze') else macd
        df['MACD_Signal'] = macd_signal.squeeze() if hasattr(macd_signal, 'squeeze') else macd_signal
        df['MACD_Hist'] = macd_hist.squeeze() if hasattr(macd_hist, 'squeeze') else macd_hist
        
        # TREND INDICATORS
        # Exponential Moving Averages: Quick response to price changes
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()
        
        # Simple Moving Averages: Smoother trend lines
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # VOLATILITY INDICATORS
        # Average True Range: How much does price move daily?
        atr = self._calculate_atr(df, period=14)
        df['ATR_14'] = atr.squeeze() if hasattr(atr, 'squeeze') else atr
        
        # Bollinger Bands: Price volatility bands
        bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['Close'])
        df['BB_Upper'] = bb_upper.squeeze() if hasattr(bb_upper, 'squeeze') else bb_upper
        df['BB_Middle'] = bb_middle.squeeze() if hasattr(bb_middle, 'squeeze') else bb_middle
        df['BB_Lower'] = bb_lower.squeeze() if hasattr(bb_lower, 'squeeze') else bb_lower
        # Calculate position within bands (0 = at lower band, 1 = at upper band)
        bb_range = df['BB_Upper'] - df['BB_Lower']
        bb_range = bb_range.replace(0, 1)  # Avoid division by zero
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / bb_range
        
        # PRICE PATTERNS
        # Rate of Change: How fast is price moving?
        df['ROC_12'] = (df['Close'] - df['Close'].shift(12)) / df['Close'].shift(12) * 100
        
        # Volume-based features
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        # Price-based features (relative changes)
        df['Close_Pct_Change'] = df['Close'].pct_change() * 100
        df['High_Low_Ratio'] = (df['High'] - df['Low']) / df['Close']
        
        print(f"✅ Created {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']])} technical indicators")
        
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """
        RSI (Relative Strength Index) Calculator
        
        Formula:
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        
        Interpretation:
        - RSI > 70: Overbought (potential pullback)
        - RSI < 30: Oversold (potential bounce)
        - RSI 30-70: Normal conditions
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.squeeze()
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """
        MACD (Moving Average Convergence Divergence) Calculator
        
        Components:
        - MACD Line: Fast EMA - Slow EMA
        - Signal Line: EMA of MACD
        - Histogram: MACD - Signal (shows momentum)
        
        Trading Signal: When MACD crosses above Signal line = BUY
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        # Ensure we return Series, not DataFrames
        return macd.squeeze(), signal_line.squeeze(), histogram.squeeze()
    
    def _calculate_atr(self, data, period=14):
        """
        ATR (Average True Range) Calculator
        
        True Range = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
        ATR = Rolling average of TR
        
        Usage: Measure volatility, set stop losses
        """
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        return atr.squeeze()
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """
        Bollinger Bands Calculator
        
        Middle Band: SMA (20-day)
        Upper Band: Middle + (2 × StdDev)
        Lower Band: Middle - (2 × StdDev)
        
        Usage: Identify overbought/oversold, volatility breakouts
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        # Ensure we return Series, not DataFrames
        return upper.squeeze(), middle.squeeze(), lower.squeeze()
    
    def prepare_training_data(self, df):
        """
        STEP 3: DATA PREPARATION FOR ML
        
        Creates training labels:
        - Label = 1 if price goes UP tomorrow (BUY signal)
        - Label = 0 if price goes DOWN tomorrow (SELL signal)
        
        This is SUPERVISED LEARNING:
        - We teach the model historical patterns
        - It learns what features predict price movement
        - Then it predicts future movements
        
        Returns:
            X: Features, y: Labels, features_list: Feature names
        """
        df = df.dropna()
        
        # CREATE TARGET VARIABLE
        # "Tomorrow's return": Will price be higher tomorrow?
        df['Tomorrow_Return'] = df['Close'].shift(-1) - df['Close']
        df['Signal'] = (df['Tomorrow_Return'] > 0).astype(int)
        
        # SELECT FEATURES (remove OHLCV and labels)
        feature_cols = [col for col in df.columns 
                       if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 
                                     'Adj Close', 'Tomorrow_Return', 'Signal']]
        
        X = df[feature_cols].dropna()
        y = df.loc[X.index, 'Signal']
        
        print(f"📈 Training data prepared: {len(X)} samples, {len(feature_cols)} features")
        print(f"   Class balance: {(y==1).mean():.1%} UP days, {(y==0).mean():.1%} DOWN days")
        
        self.feature_columns = feature_cols
        return X, y, feature_cols
    
    def train_model(self, X, y):
        """
        STEP 4: MACHINE LEARNING MODEL TRAINING
        
        Model Choice: Random Forest Classifier
        
        WHY RANDOM FOREST?
        ✅ Handles non-linear relationships (markets aren't linear)
        ✅ Works with many features (not overly sensitive)
        ✅ Robust to outliers
        ✅ Fast to train
        ✅ Provides feature importance (which indicators matter most?)
        ❌ Slower predictions (but still <1ms for our use case)
        
        Alternative models you could try:
        - Logistic Regression: Simple, interpretable
        - SVM: Good for binary classification
        - LSTM/Neural Networks: For time-series patterns
        - XGBoost: Often better performance than RF
        
        Hyperparameters explained:
        - n_estimators=100: Use 100 decision trees (more = better, slower)
        - max_depth=15: Prevent overfitting (too deep = memorizes noise)
        - min_samples_split=20: Require 20 samples minimum to split
        """
        print("🤖 Training Random Forest model...")
        
        # Scale features (important for many ML algorithms)
        X_scaled = self.scaler.fit_transform(X)
        
        # Initialize and train model
        self.model = RandomForestClassifier(
            n_estimators=100,      # Number of trees
            max_depth=15,          # Prevent overfitting
            min_samples_split=20,  # Minimum samples to split
            random_state=42,       # For reproducibility
            n_jobs=-1              # Use all CPU cores
        )
        
        self.model.fit(X_scaled, y)
        
        # Calculate training accuracy
        train_score = self.model.score(X_scaled, y)
        print(f"✅ Model trained! Training accuracy: {train_score:.2%}")
        
        # Show most important features
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\n🎯 Top 5 Most Important Features:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"   {row['feature']}: {row['importance']:.4f}")
    
    def generate_signals(self, current_data):
        """
        STEP 5: SIGNAL GENERATION
        
        Takes latest market data and generates trading signals.
        
        Outputs:
        - Signal: 1 (BUY) or 0 (SELL/HOLD)
        - Confidence: Model's confidence (0-1)
        - Reasoning: Which indicators are driving the signal
        """
        if self.model is None:
            print("❌ Model not trained yet!")
            return None
        
        # Get latest row
        latest = current_data.iloc[-1:]
        
        # Prepare features
        X_latest = latest[self.feature_columns].values
        X_latest_scaled = self.scaler.transform(X_latest)
        
        # Generate prediction
        signal = self.model.predict(X_latest_scaled)[0]
        confidence = self.model.predict_proba(X_latest_scaled)[0][signal]
        
        return {
            'signal': signal,
            'confidence': confidence,
            'timestamp': current_data.index[-1]
        }
    
    def backtest_signals(self, df):
        """
        STEP 6: BACKTESTING
        
        Simulate trading using the model's signals to evaluate performance.
        
        This answers: "Would this strategy have made money in the past?"
        
        Metrics calculated:
        - Win Rate: % of winning trades
        - Profit Factor: Gross Profit / Gross Loss
        - Sharpe Ratio: Risk-adjusted returns
        - Drawdown: Largest peak-to-trough decline
        """
        print("\n📊 Backtesting trading signals...")
        
        # Generate signals for entire history
        signals = []
        confidences = []
        
        for i in range(len(df) - 1):
            window = df.iloc[:i+1]
            signal_data = self.generate_signals(window)
            if signal_data:
                signals.append(signal_data['signal'])
                confidences.append(signal_data['confidence'])
            else:
                signals.append(0)
                confidences.append(0)
        
        df_backtest = df.iloc[:-1].copy()
        df_backtest['Signal'] = signals
        df_backtest['Confidence'] = confidences
        df_backtest['Returns'] = df_backtest['Close'].pct_change()
        df_backtest['Strategy_Returns'] = df_backtest['Signal'] * df_backtest['Returns']
        
        # Calculate metrics
        total_trades = (df_backtest['Signal'].diff() != 0).sum()
        winning_trades = (df_backtest['Strategy_Returns'] > 0).sum()
        losing_trades = (df_backtest['Strategy_Returns'] < 0).sum()
        
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        cumulative_return = (1 + df_backtest['Strategy_Returns']).prod() - 1
        buy_hold_return = (1 + df_backtest['Returns']).prod() - 1
        
        # Sharpe Ratio (risk-adjusted return)
        # Formula: (Return - Risk-free rate) / Volatility
        sharpe_ratio = (df_backtest['Strategy_Returns'].mean() * 252) / (df_backtest['Strategy_Returns'].std() * np.sqrt(252)) if df_backtest['Strategy_Returns'].std() > 0 else 0
        
        print(f"✅ Backtest Results:")
        print(f"   Total Trades: {total_trades}")
        print(f"   Win Rate: {win_rate:.2%}")
        print(f"   Strategy Return: {cumulative_return:.2%}")
        print(f"   Buy & Hold Return: {buy_hold_return:.2%}")
        print(f"   Outperformance: {(cumulative_return - buy_hold_return):.2%}")
        print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        
        return df_backtest
    
    def get_latest_signal(self, ticker, market_type='crypto'):
        """
        MAIN EXECUTION: Get latest signal for a ticker
        
        This is what Pipraisier's platform does:
        1. Fetch latest data
        2. Engineer features
        3. Generate signal
        4. Return to trader
        """
        # Fetch data
        data = self.fetch_market_data(ticker, market_type)
        if data is None:
            return None
        
        # Engineer features
        data = self.engineer_features(data)
        
        # Prepare and train
        X, y, _ = self.prepare_training_data(data)
        self.train_model(X, y)
        
        # Generate signal
        signal = self.generate_signals(data)
        
        return signal, data


class PerformanceAnalytics:
    """
    ANALYTICS MODULE: Track and analyze trading performance.
    
    This is what traders see in Pipraisier's dashboard.
    """
    
    @staticmethod
    def analyze_trades(df_backtest):
        """Calculate comprehensive trading metrics"""
        
        trades = []
        current_trade = None
        
        for i, row in df_backtest.iterrows():
            if row['Signal'] == 1 and (current_trade is None or current_trade['exit_price'] is not None):
                # Entry
                current_trade = {
                    'entry_price': row['Close'],
                    'entry_date': i,
                    'exit_price': None,
                    'exit_date': None
                }
            elif row['Signal'] == 0 and current_trade is not None and current_trade['exit_price'] is None:
                # Exit
                current_trade['exit_price'] = row['Close']
                current_trade['exit_date'] = i
                current_trade['pnl'] = current_trade['exit_price'] - current_trade['entry_price']
                current_trade['return'] = current_trade['pnl'] / current_trade['entry_price']
                trades.append(current_trade)
        
        return pd.DataFrame(trades)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("🚀 PIPRAISIER-INSPIRED: AI-DRIVEN TRADING SIGNAL GENERATOR")
    print("="*70)
    
    # Initialize generator
    generator = TradingSignalGenerator(lookback_days=365)
    
    # EXAMPLE 1: BITCOIN (Crypto)
    print("\n" + "="*70)
    print("EXAMPLE 1: BITCOIN TRADING SIGNALS")
    print("="*70)
    signal_btc, data_btc = generator.get_latest_signal('BTC-USD', 'crypto')
    if signal_btc:
        print(f"\n💡 Latest Bitcoin Signal:")
        print(f"   Signal: {'🟢 BUY' if signal_btc['signal'] == 1 else '🔴 SELL'}")
        print(f"   Confidence: {signal_btc['confidence']:.2%}")
        print(f"   Current Price: ${data_btc['Close'].iloc[-1]:.2f}")
    
    # Run backtest
    df_backtest_btc = generator.backtest_signals(data_btc)
    
    # EXAMPLE 2: S&P 500 (Stock Index)
    print("\n" + "="*70)
    print("EXAMPLE 2: S&P 500 TRADING SIGNALS")
    print("="*70)
    signal_spy, data_spy = generator.get_latest_signal('SPY', 'stock')
    if signal_spy:
        print(f"\n💡 Latest S&P 500 Signal:")
        print(f"   Signal: {'🟢 BUY' if signal_spy['signal'] == 1 else '🔴 SELL'}")
        print(f"   Confidence: {signal_spy['confidence']:.2%}")
        print(f"   Current Price: ${data_spy['Close'].iloc[-1]:.2f}")
    
    df_backtest_spy = generator.backtest_signals(data_spy)
    
    print("\n✅ Signal generation complete! Check your trading dashboard.")

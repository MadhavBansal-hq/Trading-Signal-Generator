"""
COMPREHENSIVE TEST SCRIPT
=========================
Tests each step of the pipeline to ensure everything works correctly.
"""

print("="*70)
print("COMPREHENSIVE TRADING SIGNAL GENERATOR TEST")
print("="*70)

# STEP 0: Import test
print("\n📦 STEP 0: Testing imports...")
try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import failed: {e}")
    exit(1)

# STEP 1: Import the module
print("\n🔧 STEP 1: Importing trading_signal_generator module...")
try:
    from trading_signal_generator import TradingSignalGenerator
    print("✅ Module imported successfully")
except Exception as e:
    print(f"❌ Module import failed: {e}")
    exit(1)

# STEP 2: Initialize generator
print("\n⚙️ STEP 2: Initializing TradingSignalGenerator...")
try:
    generator = TradingSignalGenerator(lookback_days=90)
    print("✅ Generator initialized")
    print(f"   - Lookback days: 90")
except Exception as e:
    print(f"❌ Initialization failed: {e}")
    exit(1)

# STEP 3: Fetch data
print("\n📊 STEP 3: Fetching Bitcoin data...")
try:
    data = generator.fetch_market_data('BTC-USD', 'crypto')
    if data is None:
        print("❌ fetch_market_data returned None")
        exit(1)
    print(f"✅ Data fetched successfully")
    print(f"   - Candles: {len(data)}")
    print(f"   - Columns: {list(data.columns)}")
    print(f"   - Date range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"   - Latest Close: ${data['Close'].iloc[-1]:,.2f}")
except Exception as e:
    print(f"❌ Data fetching failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# STEP 4: Engineer features
print("\n🔧 STEP 4: Engineering technical features...")
try:
    data_with_features = generator.engineer_features(data)
    print(f"✅ Features engineered successfully")
    print(f"   - Total columns: {len(data_with_features.columns)}")
    
    # Check individual indicators exist
    required_indicators = ['RSI_14', 'RSI_7', 'MACD', 'MACD_Signal', 'MACD_Hist',
                          'EMA_12', 'EMA_26', 'EMA_50', 'EMA_200',
                          'SMA_20', 'SMA_50', 'ATR_14',
                          'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Position',
                          'ROC_12', 'Volume_SMA', 'Volume_Ratio']
    
    missing = [ind for ind in required_indicators if ind not in data_with_features.columns]
    if missing:
        print(f"❌ Missing indicators: {missing}")
        exit(1)
    
    print(f"✅ All required indicators present")
    print(f"\n   Latest values (sample):")
    print(f"   - RSI_14: {data_with_features['RSI_14'].iloc[-1]:>8.2f}")
    print(f"   - MACD:   {data_with_features['MACD'].iloc[-1]:>8.4f}")
    print(f"   - EMA_12: {data_with_features['EMA_12'].iloc[-1]:>8.2f}")
    print(f"   - ATR_14: {data_with_features['ATR_14'].iloc[-1]:>8.2f}")
    print(f"   - BB_Pos: {data_with_features['BB_Position'].iloc[-1]:>8.2f}")
    
    # Verify data types - should be Series-compatible
    print(f"\n   Data type checks:")
    for col in ['RSI_14', 'MACD', 'ATR_14', 'BB_Upper', 'BB_Position']:
        col_type = type(data_with_features[col].iloc[-1])
        print(f"   - {col}: {col_type.__name__}")
        if not isinstance(data_with_features[col].iloc[-1], (int, float, np.number)):
            print(f"     ⚠️ Warning: Expected numeric, got {col_type}")
    
except Exception as e:
    print(f"❌ Feature engineering failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# STEP 5: Prepare training data
print("\n📈 STEP 5: Preparing training data for ML...")
try:
    X, y, feature_names = generator.prepare_training_data(data_with_features)
    print(f"✅ Data prepared successfully")
    print(f"   - Training samples: {len(X)}")
    print(f"   - Features: {len(feature_names)}")
    print(f"   - UP days: {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")
    print(f"   - DOWN days: {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
except Exception as e:
    print(f"❌ Data preparation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# STEP 6: Train model
print("\n🤖 STEP 6: Training Random Forest model...")
try:
    generator.train_model(X, y)
    print(f"✅ Model trained successfully")
    if generator.model is None:
        print("❌ Model is None after training")
        exit(1)
    print(f"   - Model type: {type(generator.model).__name__}")
except Exception as e:
    print(f"❌ Model training failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# STEP 7: Generate signal
print("\n💡 STEP 7: Generating trading signal...")
try:
    signal = generator.generate_signals(data_with_features)
    if signal is None:
        print("❌ Signal generation returned None")
        exit(1)
    print(f"✅ Signal generated successfully")
    print(f"   - Signal: {'BUY (1)' if signal['signal']==1 else 'SELL (0)'}")
    print(f"   - Confidence: {signal['confidence']:.2%}")
    print(f"   - Timestamp: {signal['timestamp']}")
except Exception as e:
    print(f"❌ Signal generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# STEP 8: Backtest
print("\n📊 STEP 8: Backtesting strategy...")
try:
    df_backtest = generator.backtest_signals(data_with_features)
    print(f"✅ Backtest completed successfully")
except Exception as e:
    print(f"❌ Backtesting failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# FINAL SUMMARY
print("\n" + "="*70)
print("✅ ALL TESTS PASSED SUCCESSFULLY!")
print("="*70)
print(f"""
Summary:
  - Data fetching: ✅ {len(data)} candles
  - Feature engineering: ✅ {len(data_with_features.columns)} total columns
  - Technical indicators: ✅ 18+ created
  - ML model: ✅ Random Forest trained
  - Signal generation: ✅ {signal['signal']} (confidence: {signal['confidence']:.1%})
  - Backtesting: ✅ Completed

Ready to run: python quick_start.py
""")

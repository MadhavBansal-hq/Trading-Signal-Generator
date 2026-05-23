# 📈 AI-Driven Trading Signal Generator

A machine learning-powered trading analytics platform that generates real-time buy/sell signals across crypto, stocks, and forex markets — built as a demonstration of quantitative analysis and ML pipeline development.

> Inspired by [Pipraisier](https://pipraisier.com) — an AI-driven trading insights platform.

---

## 🎯 What It Does

- **Ingests real-time market data** via Yahoo Finance API (crypto, stocks, forex)
- **Engineers 21 technical indicators** — RSI, MACD, Bollinger Bands, ATR, EMAs, and more
- **Trains a Random Forest classifier** to predict next-day price direction
- **Generates BUY/SELL signals** with confidence scores
- **Backtests strategies** and reports win rate, Sharpe ratio, and returns
- **Creates interactive HTML dashboards** for visual analysis

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Get live signals for BTC, ETH, XRP, S&P 500, NASDAQ, EUR/USD, USD/INR
python quick_start.py

# Generate interactive HTML dashboards
python create_dashboard.py

# Run full test suite
python comprehensive_test.py
```

---

## 📊 Sample Output

```
🚀 AI-DRIVEN TRADING SIGNALS — 2026-05-23 14:32:00

  Bitcoin  (BTC-USD)
  🔴 Signal:     SELL
  📊 Confidence: 59.4%
  💵 Live Price: 76,887.92 (live)
  📈 20d Return: -2.10%

  Ethereum (ETH-USD)
  🔴 Signal:     SELL
  📊 Confidence: 63.0%
  💵 Live Price: 2,123.85 (live)
  📈 20d Return: -8.52%

SIGNAL SUMMARY
  🟢 BUY signals:  2
  🔴 SELL signals: 5
  📊 Overall bias: Bearish 📉
```

---

## 🏗️ Architecture

```
fetch_market_data()       → Real-time OHLCV from Yahoo Finance API
    ↓
engineer_features()       → 21 technical indicators as ML features
    ↓
prepare_training_data()   → Labels (1=UP, 0=DOWN), feature scaling
    ↓
train_model()             → Random Forest (100 trees)
    ↓
generate_signals()        → BUY/SELL + confidence score
    ↓
backtest_signals()        → Win rate, Sharpe ratio, drawdown
```

---

## 📐 Technical Indicators

| Category | Indicators |
|----------|-----------|
| **Momentum** | RSI (7, 14), MACD, MACD Signal, MACD Histogram |
| **Trend** | EMA (12, 26, 50, 200), SMA (20, 50) |
| **Volatility** | Bollinger Bands (Upper/Middle/Lower/Position), ATR (14) |
| **Volume** | Volume SMA, Volume Ratio |
| **Price** | Rate of Change (12), High-Low Ratio, Close % Change |

---

## 📁 Project Structure

```
├── trading_signal_generator.py   # Core ML pipeline (main module)
├── quick_start.py                # Entry point — generates live signals
├── create_dashboard.py           # Interactive HTML dashboards (Plotly)
├── comprehensive_test.py         # Full test suite for all 8 pipeline steps
├── requirements.txt              # Dependencies
└── README.md
```

---

## ⚙️ Assets Supported

| Asset Class | Examples |
|-------------|---------|
| Crypto | BTC-USD, ETH-USD, XRP-USD |
| Stocks / Indices | SPY (S&P 500), QQQ (NASDAQ-100) |
| Forex | EURUSD=X, USDINR=X |

Any ticker supported by Yahoo Finance works out of the box.

---

## 🧠 Model Details

- **Algorithm**: Random Forest Classifier (100 estimators)
- **Training data**: Configurable lookback window (default 180 days)
- **Label**: Binary — price up (1) or down (0) next day
- **Accuracy**: ~52–55% on unseen data (realistic for financial markets)
- **Feature importance**: Automatically ranked; top features typically BB_Position, EMA_26, RSI

> Note: High training accuracy (~85%) indicates some overfitting on small datasets.
> Walk-forward validation is the recommended next step for production use.

---

## 📦 Dependencies

```
yfinance        — Market data API
pandas          — Data manipulation
numpy           — Numerical computing
scikit-learn    — Machine learning (Random Forest)
plotly          — Interactive dashboards
ta              — Technical analysis helpers
```

---

## ⚠️ Disclaimer

This project is for **educational purposes only**. It is not financial advice. Past performance does not guarantee future results. Always do your own research before making trading decisions.

---

## 👤 Author

**Madhav Bansal**  
B.E. Computer Science, BITS Pilani  
[LinkedIn](https://linkedin.com/in/your-profile) · [Email](mailto:bansal.madhav216@gmail.com)

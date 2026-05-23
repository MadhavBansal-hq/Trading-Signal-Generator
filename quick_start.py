#!/usr/bin/env python3
"""
QUICK START EXAMPLE
===================
Run this to get live trading signals with real-time prices.

Usage:
    python quick_start.py
"""

from trading_signal_generator import TradingSignalGenerator
import pandas as pd
from datetime import datetime

def main():
    print("\n" + "="*70)
    print("🚀 QUICK START: AI-DRIVEN TRADING SIGNALS")
    print(f"   Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    generator = TradingSignalGenerator(lookback_days=180)

    assets = [
        ('BTC-USD',  'crypto', 'Bitcoin'),
        ('ETH-USD',  'crypto', 'Ethereum'),
        ('XRP-USD',  'crypto', 'XRP'),
        ('SPY',      'stock',  'S&P 500'),
        ('QQQ',      'stock',  'NASDAQ-100'),
        ('EURUSD=X', 'forex',  'EUR/USD'),
        ('USDINR=X', 'forex',  'USD/INR'),
    ]

    all_signals = {}

    print("\n📡 Fetching real-time prices & generating signals...\n")

    for ticker, market_type, name in assets:
        print(f"{'─'*70}")
        print(f"  {name} ({ticker})")
        print(f"{'─'*70}")

        try:
            # Get live price first
            live_price = generator.get_realtime_price(ticker)

            # Generate signal using historical data + ML
            signal, data = generator.get_latest_signal(ticker, market_type)

            if signal is None:
                print(f"  ❌ Failed to get signal for {name}\n")
                continue

            # Use live price if available, fall back to last close
            display_price = live_price if live_price else data['Close'].iloc[-1]

            # 20-day return
            recent_return = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100

            signal_emoji = '🟢' if signal['signal'] == 1 else '🔴'
            signal_text  = 'BUY'  if signal['signal'] == 1 else 'SELL'

            print(f"  {signal_emoji} Signal:     {signal_text}")
            print(f"  📊 Confidence:  {signal['confidence']:.1%}")
            print(f"  💵 Live Price:  {display_price:,.4f}"
                  + (" (live)" if live_price else " (last close)"))
            print(f"  📈 20d Return:  {recent_return:+.2f}%\n")

            all_signals[name] = {
                'ticker':      ticker,
                'signal':      signal_text,
                'confidence':  f"{signal['confidence']:.1%}",
                'live_price':  display_price,
                '20d_return':  f"{recent_return:+.2f}%",
            }

        except Exception as e:
            print(f"  ⚠️  Error: {str(e)}\n")

    # Summary table
    print("\n" + "="*70)
    print("📋 SIGNAL SUMMARY")
    print("="*70)

    if all_signals:
        df = pd.DataFrame(all_signals).T
        print(df.to_string())

        buys  = sum(1 for s in all_signals.values() if s['signal'] == 'BUY')
        sells = len(all_signals) - buys
        print(f"\n  🟢 BUY signals:  {buys}")
        print(f"  🔴 SELL signals: {sells}")
        print(f"  📊 Overall bias: {'Bullish 📈' if buys > sells else 'Bearish 📉' if sells > buys else 'Neutral ➡️'}")

    print(f"\n  ⏱️  Signals generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*70)

if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
QUICK START EXAMPLE - ENHANCED WITH REAL-TIME PRICES
=====================================================

Shows you the fastest way to get trading signals WITH current market prices.
Run this to see real-time prices, signals, and market context all together.

Usage:
    python quick_start.py
"""

from trading_signal_generator import TradingSignalGenerator
import pandas as pd

def main():
    print("\n" + "="*70)
    print("🚀 QUICK START: AI-DRIVEN TRADING SIGNALS WITH REAL-TIME PRICES")
    print("="*70)
    
    # Initialize signal generators
    print("\n⚙️ Initializing signal generators...")
    generator_standard = TradingSignalGenerator(lookback_days=180)  # 6 months for crypto/stocks
    generator_forex = TradingSignalGenerator(lookback_days=365)      # 1 year for forex
    
    # Define assets to analyze
    standard_assets = [
        ('BTC-USD', 'crypto', 'Bitcoin'),
        ('ETH-USD', 'crypto', 'Ethereum'),
        ('XRP-USD', 'crypto', 'XRP'),
        ('SPY', 'stock', 'S&P 500'),
        ('QQQ', 'stock', 'NASDAQ-100'),
    ]
    
    forex_assets = [
        ('EURINR=X', 'forex', 'EUR/INR'),
        ('USDINR=X', 'forex', 'USD/INR'),
    ]
    
    # Store results
    all_signals = {}
    all_prices = {}
    
    # Get market indices for context
    print("\n📊 Fetching market context...")
    indices = generator_standard.get_market_indices()
    
    print("\n   Market Overview:")
    if indices:
        for name, data in indices.items():
            emoji = '📈' if data['change'] > 0 else '📉'
            print(f"   {emoji} {name}: {data['price']:>8.2f} ({data['change']:+.2f}%)")
    
    # Generate signals for standard assets (crypto/stocks)
    print("\n📊 Generating trading signals...\n")
    
    for ticker, market_type, name in standard_assets:
        print(f"\n{'─'*70}")
        print(f"📍 {name} ({ticker})")
        print(f"{'─'*70}")
        
        try:
            # GET LIVE PRICE
            live_price = generator_standard.get_realtime_price(ticker)
            
            if live_price:
                print(f"\n💵 LIVE PRICE:")
                print(f"   Current:      ${live_price['price']:>12,.2f}")
                print(f"   24h Change:   {live_price['percent_change']:>12.2f}%")
                print(f"   24h High:     ${live_price['high_24h']:>12,.2f}")
                print(f"   24h Low:      ${live_price['low_24h']:>12,.2f}")
                print(f"   Volume:       {live_price['volume']:>12,.0f}")
                
                all_prices[name] = {
                    'ticker': ticker,
                    'price': live_price['price'],
                    'change%': live_price['percent_change'],
                }
            
            # GET SIGNAL
            signal, data = generator_standard.get_latest_signal(ticker, market_type)
            
            if signal is None:
                print(f"❌ Failed to get signal for {name}")
                continue
            
            # Store signal
            all_signals[name] = {
                'ticker': ticker,
                'signal': 'BUY' if signal['signal'] == 1 else 'SELL',
                'confidence': signal['confidence'],
                'current_price': live_price['price'] if live_price else data['Close'].iloc[-1],
            }
            
            # DISPLAY SIGNAL
            signal_emoji = '🟢' if signal['signal'] == 1 else '🔴'
            print(f"{signal_emoji} Signal: {'BUY' if signal['signal'] == 1 else 'SELL'}")
            print(f"📊 Confidence: {signal['confidence']:.2%}")
            
            # Show recent performance if data available
            if len(data) > 20:
                recent_returns = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100
                print(f"📈 20-day return: {recent_returns:+.2f}%")
            
        except Exception as e:
            print(f"⚠️ Error processing {name}: {str(e)}")
    
    # Generate signals for forex assets (with more lookback)
    print("\n\n" + "="*70)
    print("💱 FOREX PAIRS (365-day analysis)")
    print("="*70)
    
    for ticker, market_type, name in forex_assets:
        print(f"\n{'─'*70}")
        print(f"📍 {name} ({ticker})")
        print(f"{'─'*70}")
        
        try:
            # GET LIVE PRICE
            live_price = generator_forex.get_realtime_price(ticker)
            
            if live_price:
                print(f"\n💵 LIVE PRICE:")
                print(f"   Current:      {live_price['price']:>12.6f}")
                print(f"   24h Change:   {live_price['percent_change']:>12.2f}%")
                print(f"   24h High:     {live_price['high_24h']:>12.6f}")
                print(f"   24h Low:      {live_price['low_24h']:>12.6f}")
                
                all_prices[name] = {
                    'ticker': ticker,
                    'price': live_price['price'],
                    'change%': live_price['percent_change'],
                }
            
            # GET SIGNAL
            signal, data = generator_forex.get_latest_signal(ticker, market_type)
            
            if signal is None:
                print(f"❌ Failed to get signal for {name}")
                continue
            
            # Store signal
            all_signals[name] = {
                'ticker': ticker,
                'signal': 'BUY' if signal['signal'] == 1 else 'SELL',
                'confidence': signal['confidence'],
                'current_price': live_price['price'] if live_price else data['Close'].iloc[-1],
            }
            
            # DISPLAY SIGNAL
            signal_emoji = '🟢' if signal['signal'] == 1 else '🔴'
            print(f"{signal_emoji} Signal: {'BUY' if signal['signal'] == 1 else 'SELL'}")
            print(f"📊 Confidence: {signal['confidence']:.2%}")
            
            # Show recent performance if data available
            if len(data) > 20:
                recent_returns = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100
                print(f"📈 20-day return: {recent_returns:+.2f}%")
            
        except Exception as e:
            print(f"⚠️ Error processing {name}: {str(e)}")
    
        print(f"\n{'─'*70}")
        print(f"📍 {name} ({ticker})")
        print(f"{'─'*70}")
        
        try:
            # GET LIVE PRICE
            live_price = generator_forex.get_realtime_price(ticker)
            
            if live_price:
                print(f"\n💵 LIVE PRICE:")
                print(f"   Current:      {live_price['price']:>12.6f}")
                print(f"   24h Change:   {live_price['percent_change']:>12.2f}%")
                print(f"   24h High:     {live_price['high_24h']:>12.6f}")
                print(f"   24h Low:      {live_price['low_24h']:>12.6f}")
                
                all_prices[name] = {
                    'ticker': ticker,
                    'price': live_price['price'],
                    'change%': live_price['percent_change'],
                }
            
            # GET SIGNAL
            signal, data = generator_forex.get_latest_signal(ticker, market_type)
            
            if signal is None:
                print(f"❌ Failed to get signal for {name}")
                continue
            
            # Store signal
            all_signals[name] = {
                'ticker': ticker,
                'signal': 'BUY' if signal['signal'] == 1 else 'SELL',
                'confidence': signal['confidence'],
                'current_price': live_price['price'] if live_price else data['Close'].iloc[-1],
            }
            
            # DISPLAY SIGNAL
            signal_emoji = '🟢' if signal['signal'] == 1 else '🔴'
            print(f"{signal_emoji} Signal: {'BUY' if signal['signal'] == 1 else 'SELL'}")
            print(f"📊 Confidence: {signal['confidence']:.2%}")
            
            # Show recent performance if data available
            if len(data) > 20:
                recent_returns = (data['Close'].iloc[-1] / data['Close'].iloc[-20] - 1) * 100
                print(f"📈 20-day return: {recent_returns:+.2f}%")
            
        except Exception as e:
            print(f"⚠️ Error processing {name}: {str(e)}")
    
    # Summary
    print("\n\n" + "="*70)
    print("📋 SIGNAL SUMMARY")
    print("="*70)
    
    if all_signals:
        # Create summary DataFrame
        summary_data = []
        for name, signal_data in all_signals.items():
            summary_data.append({
                'Asset': name,
                'Ticker': signal_data['ticker'],
                'Signal': signal_data['signal'],
                'Confidence': f"{signal_data['confidence']:.2%}",
                'Price': f"${signal_data['current_price']:,.2f}",
            })
        
        df_summary = pd.DataFrame(summary_data)
        print("\n" + df_summary.to_string(index=False))
        
        # Statistics
        buy_signals = sum(1 for s in all_signals.values() if s['signal'] == 'BUY')
        sell_signals = len(all_signals) - buy_signals
        
        print(f"\n📊 Portfolio View:")
        print(f"   🟢 BUY signals:  {buy_signals}")
        print(f"   🔴 SELL signals: {sell_signals}")
        print(f"   📈 Bullish tilt: {'Yes ✅' if buy_signals > sell_signals else 'No ❌'}")
    
    print("\n" + "="*70)
    print("✅ NEXT STEPS:")
    print("="*70)
    print("""
1. Run this regularly (e.g., every morning at market open)
   python quick_start.py
   
2. Try with real-time monitoring (uncomment code at bottom):
   python run_realtime.py
   
3. Extend your system:
   - Add more assets to monitor
   - Modify lookback_days parameter
   - Integrate with alerts or trading APIs
   
4. For your portfolio:
   - Record these daily
   - Track accuracy
   - Share results with Pipraisier
   
5. Interview preparation:
   - Explain how signals are generated
   - Discuss real vs backtested returns
   - Show live price integration
    """)

if __name__ == "__main__":
    main()


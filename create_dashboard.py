"""
BONUS: INTERACTIVE DASHBOARD CREATOR
=====================================

This script creates beautiful, interactive dashboards for your trading signals.
No experience needed - just run and open the HTML file in your browser!

Usage:
    python create_dashboard.py
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from trading_signal_generator import TradingSignalGenerator
from datetime import datetime

def create_trading_dashboard(ticker, market_type='crypto', output_name=None):
    """
    Create an interactive Plotly dashboard showing:
    - Price candles with moving averages
    - RSI indicator with overbought/oversold levels
    - MACD with signal line
    - Volume analysis
    - Trading signal overlay
    """
    
    print(f"📊 Creating dashboard for {ticker}...")
    
    # Fetch and process data
    generator = TradingSignalGenerator(lookback_days=180)
    data = generator.fetch_market_data(ticker, market_type)
    
    if data is None:
        print(f"❌ Could not fetch data for {ticker}")
        return
    
    data = generator.engineer_features(data)
    signal, _ = generator.get_latest_signal(ticker, market_type)
    
    # Prepare data for last 60 days (cleaner chart)
    df = data.iloc[-60:].copy()
    
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            f"Price Action & Moving Averages",
            "RSI (14)",
            "MACD",
            "Volume"
        ),
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # ============================================================
    # PLOT 1: PRICE WITH CANDLES & MOVING AVERAGES
    # ============================================================
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # EMA 12 (Fast trend)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA_12'],
            name='EMA 12 (Fast)',
            line=dict(color='blue', width=1.5),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # EMA 26 (Slow trend)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['EMA_26'],
            name='EMA 26 (Slow)',
            line=dict(color='red', width=1.5),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # SMA 50 (Longer-term trend)
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['SMA_50'],
            name='SMA 50',
            line=dict(color='orange', width=1, dash='dash'),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BB_Upper'],
            name='Upper BB',
            line=dict(color='rgba(100,100,100,0)'),
            showlegend=False
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['BB_Lower'],
            name='Lower BB',
            line=dict(color='rgba(100,100,100,0)'),
            fill='tonexty',
            fillcolor='rgba(100,100,100,0.1)',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # ============================================================
    # PLOT 2: RSI INDICATOR
    # ============================================================
    
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['RSI_14'],
            name='RSI (14)',
            line=dict(color='purple', width=2),
            showlegend=True
        ),
        row=2, col=1
    )
    
    # Overbought line (70)
    fig.add_hline(
        y=70,
        line_dash="dash",
        line_color="red",
        annotation_text="Overbought",
        annotation_position="right",
        row=2, col=1
    )
    
    # Oversold line (30)
    fig.add_hline(
        y=30,
        line_dash="dash",
        line_color="green",
        annotation_text="Oversold",
        annotation_position="right",
        row=2, col=1
    )
    
    # ============================================================
    # PLOT 3: MACD
    # ============================================================
    
    # MACD Line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MACD'],
            name='MACD',
            line=dict(color='blue', width=2),
            showlegend=True
        ),
        row=3, col=1
    )
    
    # Signal Line
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['MACD_Signal'],
            name='Signal Line',
            line=dict(color='red', width=2),
            showlegend=True
        ),
        row=3, col=1
    )
    
    # MACD Histogram (as bar chart)
    colors = ['green' if x > 0 else 'red' for x in df['MACD_Hist']]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['MACD_Hist'],
            name='MACD Histogram',
            marker=dict(color=colors),
            showlegend=True,
            opacity=0.3
        ),
        row=3, col=1
    )
    
    # ============================================================
    # PLOT 4: VOLUME
    # ============================================================
    
    # Volume bars
    colors_vol = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' 
                  for i in range(len(df))]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            name='Volume',
            marker=dict(color=colors_vol),
            showlegend=True,
            opacity=0.6
        ),
        row=4, col=1
    )
    
    # Volume SMA
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Volume_SMA'],
            name='Volume SMA',
            line=dict(color='orange', width=2),
            showlegend=True
        ),
        row=4, col=1
    )
    
    # ============================================================
    # UPDATE LAYOUT & STYLING
    # ============================================================
    
    # Determine signal color
    signal_color = 'green' if signal['signal'] == 1 else 'red'
    signal_text = '🟢 BUY SIGNAL' if signal['signal'] == 1 else '🔴 SELL SIGNAL'
    
    fig.update_layout(
        title={
            'text': f"<b>{ticker} Trading Dashboard</b><br>" +
                   f"<sub>{signal_text} | Confidence: {signal['confidence']:.2%} | " +
                   f"Price: ${df['Close'].iloc[-1]:,.2f}</sub>",
            'x': 0.5,
            'xanchor': 'center'
        },
        height=1200,
        hovermode='x unified',
        template='plotly_dark',
        xaxis4_title='Date',
        yaxis1_title='Price',
        yaxis2_title='RSI',
        yaxis3_title='MACD',
        yaxis4_title='Volume'
    )
    
    # Make y-axes match colors
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    # Save and show
    if output_name is None:
        output_name = f"{ticker}_{datetime.now().strftime('%Y%m%d')}_dashboard.html"
    
    fig.write_html(output_name)
    print(f"✅ Dashboard saved: {output_name}")
    print(f"📂 Open the HTML file in your browser to view!")
    
    return fig

def create_multi_asset_comparison(tickers_list, market_types):
    """
    Create a comparison dashboard showing signals for multiple assets.
    
    Args:
        tickers_list: List of tickers ['BTC-USD', 'ETH-USD', 'SPY']
        market_types: List of types ['crypto', 'crypto', 'stock']
    """
    
    generator = TradingSignalGenerator(lookback_days=365)
    
    print(f"📊 Creating multi-asset comparison...")
    
    # Fetch signals for all assets
    data_dict = {}
    signals_dict = {}
    
    for ticker, market_type in zip(tickers_list, market_types):
        try:
            data = generator.fetch_market_data(ticker, market_type)
            if data is not None:
                data = generator.engineer_features(data)
                signal, _ = generator.get_latest_signal(ticker, market_type)
                data_dict[ticker] = data
                signals_dict[ticker] = signal
        except:
            continue
    
    # Create subplots (one price chart per asset)
    n_assets = len(data_dict)
    fig = make_subplots(
        rows=n_assets, cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{t} | Signal: {'BUY' if signals_dict[t]['signal']==1 else 'SELL'} ({signals_dict[t]['confidence']:.1%})" 
                       for t in data_dict.keys()]
    )
    
    for idx, (ticker, data) in enumerate(data_dict.items(), 1):
        df = data.iloc[-90:].copy()
        
        # Add candlesticks
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name=ticker
            ),
            row=idx, col=1
        )
        
        # Add EMA 12 & 26
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['EMA_12'],
                name=f'{ticker} EMA12',
                line=dict(color='blue', width=1)
            ),
            row=idx, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['EMA_26'],
                name=f'{ticker} EMA26',
                line=dict(color='red', width=1)
            ),
            row=idx, col=1
        )
    
    fig.update_layout(
        title="Multi-Asset Trading Signals Comparison",
        height=400*n_assets,
        hovermode='x unified',
        template='plotly_dark'
    )
    
    output_name = f"multi_asset_comparison_{datetime.now().strftime('%Y%m%d')}.html"
    fig.write_html(output_name)
    
    print(f"✅ Comparison dashboard saved: {output_name}")
    
    return fig

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("🎨 INTERACTIVE TRADING DASHBOARD CREATOR")
    print("="*70)
    
    # Create individual dashboards
    print("\n📊 Creating dashboards for your assets...")
    
    assets = [
        ('BTC-USD', 'crypto'),
        ('ETH-USD', 'crypto'),
        ('SPY', 'stock'),
        ('EURINR=X', 'forex'),
        ('USDINR=X', 'forex'),
    ]
    
    for ticker, market_type in assets:
        try:
            create_trading_dashboard(ticker, market_type)
        except Exception as e:
            print(f"⚠️ Error creating dashboard for {ticker}: {e}")
    
    # Create comparison dashboard
    print("\n📈 Creating comparison dashboard...")
    try:
        tickers = [t[0] for t in assets]
        types = [t[1] for t in assets]
        create_multi_asset_comparison(tickers, types)
    except Exception as e:
        print(f"⚠️ Error creating comparison: {e}")
    
    print("\n" + "="*70)
    print("✅ DASHBOARDS CREATED!")
    print("="*70)
    print("""
Next step: Open the HTML files in your browser!

Files created:
  - BTC-USD_YYYYMMDD_dashboard.html
  - ETH-USD_YYYYMMDD_dashboard.html
  - SPY_YYYYMMDD_dashboard.html
  - multi_asset_comparison_YYYYMMDD.html

Interactive features:
  ✅ Hover over charts for exact values
  ✅ Click legend items to toggle indicators
  ✅ Zoom and pan
  ✅ Download chart as PNG
    """)

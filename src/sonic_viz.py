#!/usr/bin/env python3
"""
Sonic Protocol Visualization

This script generates visualizations for Sonic Protocol tokens, including:
1. Price comparison between S, wS, USDC.e, and scUSD
2. Volatility comparison between these tokens
3. Correlation heatmap showing relationships between their prices and volatility
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data", "volatility")
OUTPUT_DIR = os.path.join(ROOT_DIR, "data", "sonic_visualizations")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Token display names
TOKEN_NAMES = {
    "S": "Sonic (S)",
    "wS": "Wrapped Sonic (wS)",
    "USDC.e": "Sonic Bridged USDC.e",
    "scUSD": "Rings scUSD"
}

# Token colors for visualization
TOKEN_COLORS = {
    "S": "#3498db",     # Blue
    "wS": "#2980b9",    # Darker Blue
    "USDC.e": "#2ecc71", # Green
    "scUSD": "#f1c40f"  # Yellow
}


def print_header(title):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {title} ".center(70))
    print("=" * 70 + "\n")


def load_price_data(token_symbols, timeframe="daily"):
    """
    Load price data for multiple tokens.
    
    Args:
        token_symbols: List of token symbols
        timeframe: Data timeframe ("daily" or "hourly")
        
    Returns:
        Dictionary with token symbols as keys and DataFrames as values
    """
    data = {}
    
    for symbol in token_symbols:
        filename = f"{symbol}_{timeframe}.csv"
        file_path = os.path.join(DATA_DIR, filename)
        
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
        
        try:
            df = pd.read_csv(file_path)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            data[symbol] = df
            logger.info(f"Loaded {timeframe} data for {symbol}: {len(df)} rows")
        
        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
    
    return data


def create_price_comparison_chart(data, timeframe="daily"):
    """
    Create a price comparison chart for Sonic Protocol tokens.
    
    Args:
        data: Dictionary with token data
        timeframe: Data timeframe ("daily" or "hourly")
        
    Returns:
        Path to saved chart
    """
    if not data:
        logger.error("No data available for price comparison chart")
        return None
    
    # Set plot style
    plt.style.use('ggplot')
    sns.set_style("darkgrid")
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    # Plot 1: Price comparison for all tokens
    ax1 = axes[0]
    ax1.set_title("Sonic Protocol Tokens - Price Comparison", fontsize=16)
    
    for symbol, df in data.items():
        if 'price' in df.columns:
            df['price'].plot(
                ax=ax1, 
                label=TOKEN_NAMES.get(symbol, symbol),
                color=TOKEN_COLORS.get(symbol)
            )
    
    ax1.set_ylabel("Price (USD)", fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Plot 2: Normalized price comparison (starting at 100)
    ax2 = axes[1]
    ax2.set_title("Sonic Protocol Tokens - Normalized Price Comparison (Base 100)", fontsize=16)
    
    for symbol, df in data.items():
        if 'price' in df.columns:
            # Normalize price to 100 at first point
            normalized_price = df['price'] / df['price'].iloc[0] * 100
            normalized_price.plot(
                ax=ax2, 
                label=TOKEN_NAMES.get(symbol, symbol),
                color=TOKEN_COLORS.get(symbol)
            )
    
    ax2.set_ylabel("Normalized Price (Base 100)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"sonic_tokens_price_comparison_{timeframe}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved price comparison chart to {output_path}")
    return output_path


def create_volatility_comparison_chart(data, timeframe="daily"):
    """
    Create a volatility comparison chart for Sonic Protocol tokens.
    
    Args:
        data: Dictionary with token data
        timeframe: Data timeframe ("daily" or "hourly")
        
    Returns:
        Path to saved chart
    """
    if not data:
        logger.error("No data available for volatility comparison chart")
        return None
    
    # Set plot style
    plt.style.use('ggplot')
    sns.set_style("darkgrid")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    title = "Sonic Protocol Tokens - Volatility Comparison"
    if timeframe == "daily":
        title += " (Daily)"
    else:
        title += " (Hourly)"
        
    ax.set_title(title, fontsize=16)
    
    # Plot volatility for each token
    for symbol, df in data.items():
        if 'volatility' in df.columns:
            # Calculate rolling volatility (7-day or 24-hour window)
            window = 7 if timeframe == "daily" else 24
            rolling_vol = df['volatility'].rolling(window=window).mean()
            
            rolling_vol.plot(
                ax=ax, 
                label=TOKEN_NAMES.get(symbol, symbol),
                color=TOKEN_COLORS.get(symbol)
            )
    
    ax.set_ylabel("Volatility (%)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"sonic_tokens_volatility_comparison_{timeframe}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved volatility comparison chart to {output_path}")
    return output_path


def create_correlation_heatmap(data, timeframe="daily"):
    """
    Create a correlation heatmap for Sonic Protocol tokens.
    
    Args:
        data: Dictionary with token data
        timeframe: Data timeframe ("daily" or "hourly")
        
    Returns:
        Path to saved chart
    """
    if not data:
        logger.error("No data available for correlation heatmap")
        return None
    
    # Create a combined DataFrame with price data
    prices_df = pd.DataFrame()
    volatility_df = pd.DataFrame()
    
    for symbol, df in data.items():
        if 'price' in df.columns:
            prices_df[symbol] = df['price']
        if 'volatility' in df.columns:
            volatility_df[symbol] = df['volatility']
    
    # Skip if not enough data
    if len(prices_df.columns) < 2:
        logger.warning("Not enough token data for correlation heatmap")
        return None
    
    # Calculate correlation matrices
    price_corr = prices_df.corr()
    volatility_corr = volatility_df.corr() if not volatility_df.empty else None
    
    # Set plot style
    plt.figure(figsize=(15, 7))
    
    # Create price correlation heatmap
    plt.subplot(1, 2, 1)
    mask = np.triu(np.ones_like(price_corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        price_corr, 
        mask=mask,
        cmap=cmap,
        vmax=1.0, 
        vmin=-1.0,
        center=0,
        square=True, 
        linewidths=.5, 
        annot=True,
        fmt=".2f",
        cbar_kws={"shrink": .8}
    )
    
    plt.title(f"Price Correlation ({timeframe})", fontsize=14)
    
    # Create volatility correlation heatmap if available
    if volatility_corr is not None and not volatility_corr.empty:
        plt.subplot(1, 2, 2)
        mask = np.triu(np.ones_like(volatility_corr, dtype=bool))
        
        sns.heatmap(
            volatility_corr, 
            mask=mask,
            cmap=cmap,
            vmax=1.0, 
            vmin=-1.0,
            center=0,
            square=True, 
            linewidths=.5, 
            annot=True,
            fmt=".2f",
            cbar_kws={"shrink": .8}
        )
        
        plt.title(f"Volatility Correlation ({timeframe})", fontsize=14)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"sonic_tokens_correlation_{timeframe}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved correlation heatmap to {output_path}")
    return output_path


def create_price_candlestick_chart(data, token, timeframe="daily"):
    """
    Create a candlestick chart for a specific token.
    
    Args:
        data: Dictionary with token data
        token: Token symbol
        timeframe: Data timeframe ("daily" or "hourly")
        
    Returns:
        Path to saved chart
    """
    if token not in data:
        logger.error(f"No data available for {token} candlestick chart")
        return None
    
    df = data[token]
    
    # Check for required columns
    if not all(col in df.columns for col in ['price', 'percent_change_24h']):
        logger.error(f"Missing required columns for {token} candlestick chart")
        return None
    
    # Set plot style
    plt.style.use('ggplot')
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Main price chart
    ax1 = axes[0]
    ax1.set_title(f"{TOKEN_NAMES.get(token, token)} Price ({timeframe})", fontsize=16)
    
    # Plot price line
    ax1.plot(df.index, df['price'], color=TOKEN_COLORS.get(token, 'blue'), linewidth=2)
    
    # Customize price chart
    ax1.set_ylabel("Price (USD)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Percent change chart
    ax2 = axes[1]
    ax2.set_title(f"24h Percent Change", fontsize=14)
    
    # Create bar chart of percent changes
    bars = ax2.bar(
        df.index, 
        df['percent_change_24h'],
        color=[TOKEN_COLORS.get(token, 'blue') if x >= 0 else 'red' for x in df['percent_change_24h']]
    )
    
    # Customize percent change chart
    ax2.set_ylabel("Change (%)", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{token}_price_analysis_{timeframe}.png"
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved price analysis chart for {token} to {output_path}")
    return output_path


def main():
    """Main entry point for the script."""
    print_header("SONIC PROTOCOL VISUALIZATION")
    print(f"Starting at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # List of tokens
    tokens = ["S", "wS", "USDC.e", "scUSD"]
    
    # Load daily data
    print("Loading daily price data...")
    daily_data = load_price_data(tokens, "daily")
    
    if daily_data:
        # Create daily visualizations
        create_price_comparison_chart(daily_data, "daily")
        create_volatility_comparison_chart(daily_data, "daily")
        create_correlation_heatmap(daily_data, "daily")
        
        # Create individual token charts
        for token in tokens:
            if token in daily_data:
                create_price_candlestick_chart(daily_data, token, "daily")
    
    # Load hourly data
    print("Loading hourly price data...")
    hourly_data = load_price_data(tokens, "hourly")
    
    if hourly_data:
        # Create hourly visualizations
        create_price_comparison_chart(hourly_data, "hourly")
        create_volatility_comparison_chart(hourly_data, "hourly")
        create_correlation_heatmap(hourly_data, "hourly")
    
    # Print summary
    print_header("VISUALIZATION SUMMARY")
    print(f"Generated visualizations at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    
    # List output files
    output_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
    
    if output_files:
        print("\nGenerated visualization files:")
        for file in sorted(output_files):
            print(f"  - {file}")
    else:
        print("\nNo visualization files were generated.")
    
    print("\nTo view these files, open them with any image viewer or browser.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

def load_data():
    data_dir = Path("round_4/round_4_data")
    all_data = []
    
    for day in range(1, 4):
        prices_file = data_dir / f"prices_round_4_day_{day}.csv"
        
        if not os.path.exists(prices_file):
            print(f"Warning: Missing prices file for day {day}")
            continue
            
        try:
            df = pd.read_csv(prices_file, sep=';')
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            all_data.append(df)
        except Exception as e:
            print(f"Error processing day {day}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No data was successfully loaded")
    
    return pd.concat(all_data, ignore_index=True)

def calculate_spreads(df):
    # Filter data for VOLCANIC_ROCK and VOLCANIC_ROCK_VOUCHER_9500
    volcanic_rock = df[df['product'] == 'VOLCANIC_ROCK']
    voucher = df[df['product'] == 'VOLCANIC_ROCK_VOUCHER_10500']
    
    # Merge the dataframes on timestamp
    merged_df = pd.merge(volcanic_rock, voucher, on='timestamp', suffixes=('', '_voucher'))
    
    # Calculate mid prices
    merged_df['volcanic_rock_mid'] = merged_df['mid_price']
    merged_df['volcanic_rock_voucher_mid'] = merged_df['mid_price_voucher']
    
    # Calculate the theoretical value (spot - strike)
    merged_df['theoretical_value'] = merged_df['volcanic_rock_mid'] - 10500
    
    # Calculate the spread between voucher and theoretical value
    merged_df['spread'] = merged_df['volcanic_rock_voucher_mid'] - merged_df['theoretical_value']
    
    # Calculate bid-ask spreads
    merged_df['voucher_bid_ask_spread'] = merged_df['ask_price_1_voucher'] - merged_df['bid_price_1_voucher']
    merged_df['underlying_bid_ask_spread'] = merged_df['ask_price_1'] - merged_df['bid_price_1']
    
    return merged_df

def analyze_spreads(df):
    # Basic statistics
    spread_stats = {
        'mean_spread': df['spread'].mean(),
        'std_spread': df['spread'].std(),
        'min_spread': df['spread'].min(),
        'max_spread': df['spread'].max(),
        'median_spread': df['spread'].median(),
        'mean_voucher_bid_ask': df['voucher_bid_ask_spread'].mean(),
        'mean_underlying_bid_ask': df['underlying_bid_ask_spread'].mean()
    }
    
    # Calculate correlation between spread and underlying price
    correlation = df['spread'].corr(df['volcanic_rock_mid'])
    
    # Calculate spread percentiles
    percentiles = df['spread'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    
    return spread_stats, correlation, percentiles

def plot_analysis(df, spread_stats, correlation, percentiles):
    plt.figure(figsize=(15, 20))
    
    # Plot 1: Spread over time
    plt.subplot(4, 1, 1)
    plt.plot(df['timestamp'], df['spread'], label='Spread')
    plt.axhline(y=spread_stats['mean_spread'], color='r', linestyle='--', label='Mean Spread')
    plt.xlabel('Time')
    plt.ylabel('Spread')
    plt.title('Spread Between Voucher and Theoretical Value Over Time')
    plt.legend()
    
    # Plot 2: Spread vs Underlying Price
    plt.subplot(4, 1, 2)
    plt.scatter(df['volcanic_rock_mid'], df['spread'], alpha=0.1)
    plt.xlabel('Underlying Price')
    plt.ylabel('Spread')
    plt.title(f'Spread vs Underlying Price (Correlation: {correlation:.3f})')
    
    # Plot 3: Bid-Ask Spreads
    plt.subplot(4, 1, 3)
    plt.plot(df['timestamp'], df['voucher_bid_ask_spread'], label='Voucher Bid-Ask')
    plt.plot(df['timestamp'], df['underlying_bid_ask_spread'], label='Underlying Bid-Ask')
    plt.xlabel('Time')
    plt.ylabel('Bid-Ask Spread')
    plt.title('Bid-Ask Spreads Over Time')
    plt.legend()
    
    # Plot 4: Spread Distribution
    plt.subplot(4, 1, 4)
    plt.hist(df['spread'], bins=50, density=True, alpha=0.7)
    plt.axvline(x=spread_stats['mean_spread'], color='r', linestyle='--', label='Mean')
    plt.axvline(x=spread_stats['median_spread'], color='g', linestyle='--', label='Median')
    plt.xlabel('Spread')
    plt.ylabel('Density')
    plt.title('Spread Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('voucher_spread_analysis_4.png')
    plt.close()

def main():
    try:
        # Load and process data
        df = load_data()
        df = calculate_spreads(df)
        
        # Analyze spreads
        spread_stats, correlation, percentiles = analyze_spreads(df)
        
        # Print analysis results
        print("\nSpread Analysis Results:")
        print(f"Mean Spread: {spread_stats['mean_spread']:.2f}")
        print(f"Standard Deviation: {spread_stats['std_spread']:.2f}")
        print(f"Minimum Spread: {spread_stats['min_spread']:.2f}")
        print(f"Maximum Spread: {spread_stats['max_spread']:.2f}")
        print(f"Median Spread: {spread_stats['median_spread']:.2f}")
        print(f"\nBid-Ask Spreads:")
        print(f"Mean Voucher Bid-Ask: {spread_stats['mean_voucher_bid_ask']:.2f}")
        print(f"Mean Underlying Bid-Ask: {spread_stats['mean_underlying_bid_ask']:.2f}")
        print(f"\nCorrelation with Underlying Price: {correlation:.3f}")
        
        print("\nSpread Percentiles:")
        for p, value in percentiles.items():
            print(f"{p*100}th percentile: {value:.2f}")
        
        # Generate plots
        plot_analysis(df, spread_stats, correlation, percentiles)
        
        # Trading insights
        print("\nTrading Insights:")
        print("1. Look for spread mean reversion opportunities")
        print("2. Consider bid-ask spreads when entering/exiting positions")
        print("3. Monitor correlation with underlying price for potential arbitrage")
        print("4. Use percentiles to identify extreme spread values")
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
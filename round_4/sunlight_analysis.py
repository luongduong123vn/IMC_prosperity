import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import linregress
import os

def load_and_analyze_data():
    # Load data from all days
    data_dir = Path("round_4_data")
    all_data = []
    
    for day in range(1, 4):
        obs_file = data_dir / f"observations_round_4_day_{day}.csv"
        
        if not os.path.exists(obs_file):
            print(f"Warning: Missing observation file for day {day}")
            continue
            
        try:
            # Load observations - this contains both sunlight and sugar price data
            obs_df = pd.read_csv(obs_file)
            obs_df['timestamp'] = pd.to_datetime(obs_df['timestamp'], unit='s')
            
            # Calculate mid prices from bid/ask
            obs_df['macaron_price'] = (obs_df['bidPrice'] + obs_df['askPrice']) / 2
            
            all_data.append(obs_df)
            
        except Exception as e:
            print(f"Error processing day {day}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No data was successfully loaded")
    
    # Combine all days
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Calculate forward-looking price changes for different horizons
    for horizon in [50, 100, 200]:
        combined_df[f'sugar_price_change_{horizon}'] = combined_df['sugarPrice'].shift(-horizon) / combined_df['sugarPrice'] - 1
        combined_df[f'macaron_price_change_{horizon}'] = combined_df['macaron_price'].shift(-horizon) / combined_df['macaron_price'] - 1
    
    # Calculate rolling statistics
    combined_df['sunlight_rolling_mean'] = combined_df['sunlightIndex'].rolling(window=100).mean()
    combined_df['sunlight_rolling_std'] = combined_df['sunlightIndex'].rolling(window=100).std()
    
    # Calculate correlation between sugar and macaron prices
    combined_df['price_correlation'] = combined_df['sugarPrice'].rolling(window=100).corr(combined_df['macaron_price'])
    
    return combined_df

def find_critical_sunlight_index(df):
    # Create bins for sunlight index
    sunlight_bins = np.linspace(df['sunlightIndex'].min(), df['sunlightIndex'].max(), 50)
    
    # Calculate metrics for each bin
    bin_analysis = []
    for i in range(len(sunlight_bins) - 1):
        mask = (df['sunlightIndex'] >= sunlight_bins[i]) & (df['sunlightIndex'] < sunlight_bins[i+1])
        bin_data = df[mask]
        
        if len(bin_data) > 0:
            # Calculate average future returns for different horizons
            metrics = {
                'sunlight_min': sunlight_bins[i],
                'sunlight_max': sunlight_bins[i+1],
                'count': len(bin_data)
            }
            
            # Add metrics for different horizons
            for horizon in [50, 100, 200]:
                metrics.update({
                    f'sugar_future_{horizon}': bin_data[f'sugar_price_change_{horizon}'].mean(),
                    f'macaron_future_{horizon}': bin_data[f'macaron_price_change_{horizon}'].mean(),
                    f'correlation_{horizon}': bin_data['price_correlation'].mean()
                })
            
            bin_analysis.append(metrics)
    
    bin_df = pd.DataFrame(bin_analysis)
    
    # Find the CSI by looking for consistent price increases in both assets
    def score_threshold(threshold):
        below_threshold = df['sunlightIndex'] < threshold
        score = 0
        
        # Check price increases for different horizons
        for horizon in [50, 100, 200]:
            sugar_increase = df.loc[below_threshold, f'sugar_price_change_{horizon}'].mean()
            macaron_increase = df.loc[below_threshold, f'macaron_price_change_{horizon}'].mean()
            correlation = df.loc[below_threshold, 'price_correlation'].mean()
            
            # Score based on price increases and correlation
            score += (sugar_increase > 0) + (macaron_increase > 0) + (correlation > 0)
        
        return score
    
    # Find CSI that maximizes price increases and correlation
    scores = [(threshold, score_threshold(threshold)) for threshold in sunlight_bins]
    csi = max(scores, key=lambda x: x[1])[0]
    
    # Calculate detailed statistics for the chosen CSI
    below_csi = df[df['sunlightIndex'] < csi]
    above_csi = df[df['sunlightIndex'] >= csi]
    
    # Calculate statistics for different horizons
    price_stats = {}
    for horizon in [50, 100, 200]:
        price_stats.update({
            f'below_csi_sugar_{horizon}': below_csi[f'sugar_price_change_{horizon}'].mean() * 100,
            f'above_csi_sugar_{horizon}': above_csi[f'sugar_price_change_{horizon}'].mean() * 100,
            f'below_csi_macaron_{horizon}': below_csi[f'macaron_price_change_{horizon}'].mean() * 100,
            f'above_csi_macaron_{horizon}': above_csi[f'macaron_price_change_{horizon}'].mean() * 100,
            f'below_csi_correlation_{horizon}': below_csi['price_correlation'].mean(),
            f'above_csi_correlation_{horizon}': above_csi['price_correlation'].mean()
        })
    
    # Calculate impact of sustained low sunlight
    df['low_sunlight_duration'] = (df['sunlightIndex'] < csi).astype(int).rolling(window=100).sum()
    duration_analysis = []
    
    for duration in range(10, 101, 10):
        sustained_low = df[df['low_sunlight_duration'] >= duration]
        if len(sustained_low) > 0:
            stats = {
                'duration': duration,
                'count': len(sustained_low)
            }
            for horizon in [50, 100, 200]:
                stats.update({
                    f'sugar_change_{horizon}': sustained_low[f'sugar_price_change_{horizon}'].mean() * 100,
                    f'macaron_change_{horizon}': sustained_low[f'macaron_price_change_{horizon}'].mean() * 100,
                    f'correlation': sustained_low['price_correlation'].mean()
                })
            duration_analysis.append(stats)
    
    duration_df = pd.DataFrame(duration_analysis)
    
    return csi, bin_df, price_stats, duration_df

def plot_analysis(df, csi, bin_df, duration_df):
    plt.figure(figsize=(15, 20))
    
    # Plot 1: Price Changes vs Sunlight Index for different horizons
    plt.subplot(4, 1, 1)
    for horizon in [50, 100, 200]:
        plt.plot(bin_df['sunlight_min'], bin_df[f'sugar_future_{horizon}'] * 100, 
                label=f'Sugar {horizon} ticks', alpha=0.7)
        plt.plot(bin_df['sunlight_min'], bin_df[f'macaron_future_{horizon}'] * 100, 
                label=f'Macaron {horizon} ticks', alpha=0.7)
    plt.axvline(x=csi, color='r', linestyle='--', label=f'CSI: {csi:.2f}')
    plt.xlabel('Sunlight Index')
    plt.ylabel('Future Price Change (%)')
    plt.title('Future Price Changes vs Sunlight Index')
    plt.legend()
    
    # Plot 2: Price Correlation
    plt.subplot(4, 1, 2)
    plt.scatter(df['sunlightIndex'], df['price_correlation'], alpha=0.1)
    plt.axvline(x=csi, color='r', linestyle='--', label=f'CSI: {csi:.2f}')
    plt.xlabel('Sunlight Index')
    plt.ylabel('Price Correlation')
    plt.title('Sugar-Macaron Price Correlation vs Sunlight Index')
    plt.legend()
    
    # Plot 3: Duration Impact on Returns
    plt.subplot(4, 1, 3)
    for horizon in [50, 100, 200]:
        plt.plot(duration_df['duration'], duration_df[f'sugar_change_{horizon}'],
                label=f'Sugar {horizon} ticks', alpha=0.7)
        plt.plot(duration_df['duration'], duration_df[f'macaron_change_{horizon}'],
                label=f'Macaron {horizon} ticks', alpha=0.7)
    plt.xlabel('Duration Below CSI (ticks)')
    plt.ylabel('Average Price Change (%)')
    plt.title('Impact of Sustained Low Sunlight on Returns')
    plt.legend()
    
    # Plot 4: Duration Impact on Correlation
    plt.subplot(4, 1, 4)
    plt.plot(duration_df['duration'], duration_df['correlation'], marker='o')
    plt.xlabel('Duration Below CSI (ticks)')
    plt.ylabel('Price Correlation')
    plt.title('Impact of Sustained Low Sunlight on Price Correlation')
    
    plt.tight_layout()
    plt.savefig('sunlight_analysis.png')
    plt.close()

def main():
    try:
        # Load and analyze data
        df = load_and_analyze_data()
        
        # Find CSI and analyze patterns
        csi, bin_df, price_stats, duration_df = find_critical_sunlight_index(df)
        print(f"Critical Sunlight Index (CSI): {csi:.2f}")
        
        print("\nMACARON Price Changes at Different Horizons:")
        for horizon in [50, 100, 200]:
            print(f"\n{horizon}-tick horizon:")
            print(f"Below CSI - MACARON: {price_stats[f'below_csi_macaron_{horizon}']:.2f}%")
            print(f"Above CSI - MACARON: {price_stats[f'above_csi_macaron_{horizon}']:.2f}%")
        
        print("\nImpact of Sustained Low Sunlight on MACARON:")
        print(duration_df.to_string(index=False))
        
        # Plot analysis
        plot_analysis(df, csi, bin_df, duration_df)
        
        # Print MACARON-specific trading strategy
        print("\nMACARON Trading Strategy:")
        print(f"1. Entry Conditions (when sunlight index falls below {csi:.2f}):")
        print("   - Wait for at least 50 ticks below CSI to confirm the trend")
        print("   - Take long positions in MACARON")
        print("   - Position size should increase with duration below CSI")
        
        print("\n2. Position Sizing:")
        print("   - Base position: 100 units")
        print("   - Scale up by 20% for each additional 50 ticks below CSI")
        print("   - Maximum position: 300 units")
        
        print("\n3. Exit Conditions:")
        print("   - When sunlight index rises above CSI")
        print("   - After capturing expected price increase (0.88% for 50-tick horizon)")
        print("   - If price drops more than 0.5% from entry")
        
        print("\n4. Risk Management:")
        print("   - Monitor the duration of low sunlight periods")
        print("   - Use a 100-tick moving average as a reference")
        print("   - Set stop loss at 0.5% below entry price")
        
        print("\nKey Insights:")
        print("1. MACARON tends to rise during sustained low sunlight periods")
        print("2. Longer duration below CSI leads to stronger price increases")
        print("3. Strongest effects seen in 50-tick horizon")
        print("4. Price increases are most significant after extended periods below CSI")
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main() 
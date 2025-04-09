import pandas as pd
import numpy as np
from typing import Union, List

def calculate_time_series_slope_n(values: Union[List[float], np.ndarray, pd.Series]) -> float:
    """
    Calculate the slope of the line of best fit through a time series of values.
    """
    if isinstance(values, (list, pd.Series)):
        values = np.array(values)
    
    n = len(values)
    if n < 2:
        return 0
    
    # For n consecutive timestamps (0 to n-1), the mean is (n-1)/2
    t_mean = (n - 1) / 2
    v_mean = np.mean(values)
    
    # Create timestamp deviations from mean
    t_deviations = np.arange(n) - t_mean
    
    # Calculate slope (beta)
    numerator = np.sum(t_deviations * (values - v_mean))
    denominator = np.sum(t_deviations ** 2)
    
    return numerator / denominator if denominator != 0 else 0

def analyze_price_behavior(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
    """
    Analyze price behavior by calculating:
    1. Rolling volatility
    2. Price trend strength (absolute slope)
    3. Mean reversion strength (correlation with lagged returns)
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Print initial data info
    print(f"\nInitial data shape: {df.shape}")
    print(f"Missing values in initial data:")
    print(df[['ask_price_1', 'bid_price_1']].isnull().sum())
    
    # Calculate mid price
    df['mid_price'] = (df['ask_price_1'] + df['bid_price_1']) / 2
    
    # Calculate returns
    df['returns'] = df['mid_price'].pct_change()
    
    # Calculate rolling volatility
    df['volatility'] = df['returns'].rolling(window, min_periods=2).std()
    
    # Calculate trend strength (absolute slope of price over rolling window)
    df['trend_strength'] = df['mid_price'].rolling(window, min_periods=2).apply(
        lambda x: abs(calculate_time_series_slope_n(x))
    )
    
    # Calculate mean reversion strength (negative autocorrelation)
    df['lagged_returns'] = df['returns'].shift(1)
    df['mean_reversion'] = df['returns'].rolling(window, min_periods=2).corr(df['lagged_returns'])
    
    # Print diagnostic information
    print("\nMissing values after calculations:")
    print(df[['volatility', 'trend_strength', 'mean_reversion']].isnull().sum())
    print("\nSample of non-null data:")
    print(df[['mid_price', 'volatility', 'trend_strength', 'mean_reversion']].dropna().head())
    
    return df

def calculate_behavior_correlations(df: pd.DataFrame) -> dict:
    """
    Calculate correlations between volatility and price behavior metrics
    """
    # Drop rows where any of the required columns are NaN
    clean_df = df[['volatility', 'trend_strength', 'mean_reversion']].dropna()
    
    print(f"\nNumber of valid rows for correlation: {len(clean_df)}")
    print("\nValue ranges:")
    print(clean_df.describe())
    
    correlations = {
        'vol_trend_corr': clean_df['volatility'].corr(clean_df['trend_strength']),
        'vol_meanrev_corr': clean_df['volatility'].corr(clean_df['mean_reversion'])
    }
    return correlations

# Read and process the data
print("Reading data...")
df = pd.read_csv('./round_1/round_1_data/prices_round_1_day_-2.csv')

# Filter for SQUID_INK
squid_data = df[df['product'] == 'SQUID_INK'].copy()
print(f"\nSQUID_INK data shape: {squid_data.shape}")

# Analyze SQUID_INK
print("\nAnalyzing SQUID_INK...")
squid_analyzed = analyze_price_behavior(squid_data)

# Calculate correlations
print("\nCalculating correlations...")
squid_corr = calculate_behavior_correlations(squid_analyzed)

# Print results
print("\nSQUID_INK Analysis:")
print(f"Correlation between volatility and trend strength: {squid_corr['vol_trend_corr']:.4f}")
print(f"Correlation between volatility and mean reversion: {squid_corr['vol_meanrev_corr']:.4f}")

# Save processed data for further analysis
squid_analyzed.to_csv('squid_analyzed.csv')
print("\nSaved processed data to CSV file for further analysis.") 
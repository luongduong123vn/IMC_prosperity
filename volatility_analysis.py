import pandas as pd
import numpy as np
from typing import Union, List
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy import stats

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
    3. Return trend strength (absolute slope of returns)
    4. Mean reversion strength (correlation with lagged returns)
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
    
    # Calculate price trend strength (absolute slope of price over rolling window)
    df['price_trend_strength'] = df['mid_price'].rolling(window, min_periods=2).apply(
        lambda x: abs(calculate_time_series_slope_n(x))
    )
    
    # Calculate return trend strength (absolute slope of returns over rolling window)
    df['return_trend_strength'] = df['returns'].rolling(window, min_periods=2).apply(
        lambda x: abs(calculate_time_series_slope_n(x))
    )
    
    # Calculate mean reversion strength (negative autocorrelation)
    df['lagged_returns'] = df['returns'].shift(1)
    df['mean_reversion'] = df['returns'].rolling(window, min_periods=2).corr(df['lagged_returns'])
    
    # Print diagnostic information
    print("\nMissing values after calculations:")
    print(df[['volatility', 'price_trend_strength', 'return_trend_strength', 'mean_reversion']].isnull().sum())
    print("\nSample of non-null data:")
    print(df[['mid_price', 'volatility', 'price_trend_strength', 'return_trend_strength', 'mean_reversion']].dropna().head())
    
    return df

def calculate_behavior_correlations(df: pd.DataFrame) -> dict:
    """
    Calculate correlations between volatility and price behavior metrics with p-values
    """
    # Drop rows where any of the required columns are NaN
    clean_df = df[['volatility', 'price_trend_strength', 'return_trend_strength', 'mean_reversion']].dropna()
    
    print(f"\nNumber of valid rows for correlation: {len(clean_df)}")
    print("\nValue ranges:")
    print(clean_df.describe())
    
    # Calculate correlations and p-values
    def calculate_correlation_with_pvalue(x, y):
        correlation, p_value = stats.pearsonr(x, y)
        return correlation, p_value
    
    vol_price_corr, vol_price_p = calculate_correlation_with_pvalue(clean_df['volatility'], clean_df['price_trend_strength'])
    vol_return_corr, vol_return_p = calculate_correlation_with_pvalue(clean_df['volatility'], clean_df['return_trend_strength'])
    vol_meanrev_corr, vol_meanrev_p = calculate_correlation_with_pvalue(clean_df['volatility'], clean_df['mean_reversion'])
    
    correlations = {
        'vol_price_trend_corr': {
            'value': vol_price_corr,
            'p_value': vol_price_p,
            'significant': vol_price_p < 0.05
        },
        'vol_return_trend_corr': {
            'value': vol_return_corr,
            'p_value': vol_return_p,
            'significant': vol_return_p < 0.05
        },
        'vol_meanrev_corr': {
            'value': vol_meanrev_corr,
            'p_value': vol_meanrev_p,
            'significant': vol_meanrev_p < 0.05
        }
    }
    return correlations

def plot_correlations(df: pd.DataFrame):
    """
    Create scatter plots to visualize correlations
    """
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot volatility vs price trend strength
    axes[0].scatter(df['volatility'], df['price_trend_strength'], alpha=0.5)
    axes[0].set_xlabel('Volatility')
    axes[0].set_ylabel('Price Trend Strength')
    axes[0].set_title('Volatility vs Price Trend')
    
    # Plot volatility vs return trend strength
    axes[1].scatter(df['volatility'], df['return_trend_strength'], alpha=0.5)
    axes[1].set_xlabel('Volatility')
    axes[1].set_ylabel('Return Trend Strength')
    axes[1].set_title('Volatility vs Return Trend')
    
    # Plot volatility vs mean reversion
    axes[2].scatter(df['volatility'], df['mean_reversion'], alpha=0.5)
    axes[2].set_xlabel('Volatility')
    axes[2].set_ylabel('Mean Reversion')
    axes[2].set_title('Volatility vs Mean Reversion')
    
    plt.tight_layout()
    plt.show()

def calculate_autocorrelation(df: pd.DataFrame, lags: int = 20) -> dict:
    """
    Calculate autocorrelation of market maker's mid price returns with significance testing
    """
    # Calculate market maker's mid price
    df['mm_mid'] = (df['ask_price_1'] + df['bid_price_1']) / 2
    
    # Calculate returns
    df['mm_returns'] = df['mm_mid'].pct_change()
    
    # Drop NaN values from returns
    returns = df['mm_returns'].dropna()
    n = len(returns)
    
    # Calculate autocorrelation
    autocorr = acf(returns, nlags=lags, fft=True)
    
    # Calculate 95% confidence intervals
    conf_int = 1.96 / np.sqrt(n)
    
    # Create dictionary of autocorrelation values with significance
    autocorr_dict = {}
    for i in range(lags + 1):
        value = autocorr[i]
        is_significant = abs(value) > conf_int
        autocorr_dict[f'lag_{i}'] = {
            'value': value,
            'is_significant': is_significant,
            'confidence_interval': conf_int
        }
    
    return autocorr_dict

def plot_autocorrelation(autocorr_dict: dict):
    """
    Plot autocorrelation function of returns with confidence intervals
    """
    lags = list(range(len(autocorr_dict)))
    values = [v['value'] for v in autocorr_dict.values()]
    conf_int = list(autocorr_dict.values())[0]['confidence_interval']
    
    plt.figure(figsize=(10, 6))
    plt.stem(lags, values, markerfmt='C0o', linefmt='C0-', basefmt='C0-')
    plt.axhline(y=0, color='black', linestyle='-')
    plt.axhline(y=conf_int, color='red', linestyle='--', alpha=0.5)
    plt.axhline(y=-conf_int, color='red', linestyle='--', alpha=0.5)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Autocorrelation of Market Maker Mid Price Returns\n(Red lines show 95% confidence intervals)')
    plt.grid(True)
    plt.show()

# Read and process the data
print("Reading data...")
df = pd.read_csv('./round_1/round_1_data/prices_round_1_day_0.csv')

# Filter for SQUID_INK
squid_data = df[df['product'] == 'SQUID_INK'].copy()
print(f"\nSQUID_INK data shape: {squid_data.shape}")

# Calculate autocorrelation
print("\nCalculating autocorrelation...")
autocorr = calculate_autocorrelation(squid_data)

# Print autocorrelation results with significance
print("\nAutocorrelation of Market Maker Mid Price Returns:")
print("Lag\tValue\t\tSignificant?")
print("-" * 40)
for lag, data in autocorr.items():
    significance = "Yes" if data['is_significant'] else "No"
    print(f"{lag}\t{data['value']:.4f}\t\t{significance}")

# Plot autocorrelation
print("\nPlotting autocorrelation...")
plot_autocorrelation(autocorr)

# Analyze SQUID_INK
print("\nAnalyzing SQUID_INK...")
squid_analyzed = analyze_price_behavior(squid_data)

# Calculate correlations
print("\nCalculating correlations...")
squid_corr = calculate_behavior_correlations(squid_analyzed)

# Print results
print("\nSQUID_INK Analysis:")
print("Metric\t\t\tCorrelation\tP-value\t\tSignificant?")
print("-" * 60)
for metric, data in squid_corr.items():
    significance = "Yes" if data['significant'] else "No"
    print(f"{metric}\t{data['value']:.4f}\t\t{data['p_value']:.4f}\t\t{significance}")

# Plot correlations
print("\nPlotting correlations...")
plot_correlations(squid_analyzed)

# Save processed data for further analysis
squid_analyzed.to_csv('squid_analyzed.csv')
print("\nSaved processed data to CSV file for further analysis.") 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
import math
from typing import List, Tuple, Union
import jsonpickle

def calculate_slope(numbers: Union[List[float], np.ndarray, pd.Series]) -> float:
    """
    Calculate the slope of the line of best fit through a list of numbers.
    
    Args:
        numbers: List of numbers to fit the line through
        
    Returns:
        float: Slope of the line of best fit
    """
    # Convert to numpy array if it's a list or pandas Series
    if isinstance(numbers, (list, pd.Series)):
        numbers = np.array(numbers)
    
    # Create x values (indices)
    x = np.arange(len(numbers))
    
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(numbers)
    
    # Calculate slope (beta)
    numerator = np.sum((x - x_mean) * (numbers - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    # Return slope, handling division by zero
    return numerator / denominator if denominator != 0 else 0

def regress_series(series: Union[List[float], np.ndarray, pd.Series]) -> Tuple[float, float, float, float, float, np.ndarray]:
    """
    Perform linear regression on a series of numbers.
    
    Args:
        series: The series of numbers to regress (list, numpy array, or pandas Series)
        
    Returns:
        Tuple containing:
        - slope: float
        - intercept: float
        - r_value: float
        - p_value: float
        - std_err: float
        - regression_line: numpy array
    """
    # Convert to numpy array if it's a list or pandas Series
    if isinstance(series, (list, pd.Series)):
        series = np.array(series)
    
    # Create x values (indices)
    x = np.arange(len(series))
    
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(series)
    
    # Calculate slope (beta)
    numerator = np.sum((x - x_mean) * (series - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator if denominator != 0 else 0
    
    # Calculate intercept (alpha)
    intercept = y_mean - slope * x_mean
    
    # Calculate R-squared
    y_pred = slope * x + intercept
    ss_tot = np.sum((series - y_mean) ** 2)
    ss_res = np.sum((series - y_pred) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    r_value = math.sqrt(r_squared) if r_squared >= 0 else 0
    
    # Calculate standard error
    n = len(series)
    if n > 2:
        std_err = math.sqrt(ss_res / (n - 2))
    else:
        std_err = 0
    
    # Calculate p-value (using t-distribution)
    if n > 2 and std_err != 0:
        t_stat = slope / (std_err / math.sqrt(np.sum((x - x_mean) ** 2)))
        p_value = 2 * (1 - statistics.NormalDist().cdf(abs(t_stat)))
    else:
        p_value = 1.0
    
    # Create regression line
    regression_line = slope * x + intercept
    
    return slope, intercept, r_value, p_value, std_err, regression_line

def plot_regression(series: Union[List[float], np.ndarray, pd.Series], title: str = "Regression Analysis") -> None:
    """
    Plot the series and its regression line.
    
    Args:
        series: The series of numbers to plot
        title: Title for the plot
    """
    # Get regression results
    slope, intercept, r_value, p_value, std_err, regression_line = regress_series(series)
    
    # Create x values for plotting
    x = np.arange(len(series))
    
    # Create the plot using pandas plotting
    df = pd.DataFrame({
        'x': x,
        'y': series,
        'regression': regression_line
    })
    
    ax = df.plot.scatter(x='x', y='y', label='Data Points')
    df.plot.line(x='x', y='regression', ax=ax, color='red', label=f'Regression Line (R²={r_value**2:.3f})')
    ax.set_title(title)
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.grid(True)
    
    # Print regression statistics
    print(f"\nRegression Statistics for {title}:")
    print(f"Slope: {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    print(f"R-squared: {r_value**2:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Standard Error: {std_err:.4f}")

def calculate_time_series_slope(timestamps: Union[List[int], np.ndarray, pd.Series], 
                              values: Union[List[float], np.ndarray, pd.Series]) -> float:
    """
    Calculate the slope of the line of best fit through a time series.
    
    Args:
        timestamps: List of timestamps (x-values)
        values: List of corresponding values (y-values)
        
    Returns:
        float: Slope of the line of best fit
    """
    # Convert inputs to numpy arrays if they're lists or pandas Series
    if isinstance(timestamps, (list, pd.Series)):
        timestamps = np.array(timestamps)
    if isinstance(values, (list, pd.Series)):
        values = np.array(values)
    
    # Calculate means
    t_mean = np.mean(timestamps)
    v_mean = np.mean(values)
    
    # Calculate slope (beta)
    numerator = np.sum((timestamps - t_mean) * (values - v_mean))
    denominator = np.sum((timestamps - t_mean) ** 2)
    
    # Return slope, handling division by zero
    return numerator / denominator if denominator != 0 else 0

def calculate_time_series_slope_n(values: Union[List[float], np.ndarray, pd.Series]) -> float:
    """
    Calculate the slope of the line of best fit through a time series of values.
    The timestamps are assumed to be consecutive integers starting from 0.
    
    Args:
        values: List of values (y-values)
        
    Returns:
        float: Slope of the line of best fit
    """
    # Convert to numpy array if it's a list or pandas Series
    if isinstance(values, (list, pd.Series)):
        values = np.array(values)
    
    n = len(values)
    if n < 2:
        return 0
    
    # For n consecutive timestamps (0 to n-1), the mean is (n-1)/2
    t_mean = (n - 1) / 2
    
    # Calculate value mean
    v_mean = np.mean(values)
    
    # Create timestamp deviations from mean
    t_deviations = np.arange(n) - t_mean
    
    # Calculate slope (beta)
    numerator = np.sum(t_deviations * (values - v_mean))
    denominator = np.sum(t_deviations ** 2)
    
    # Return slope, handling division by zero
    return numerator / denominator if denominator != 0 else 0

# Read the data
df = pd.read_csv('./round_1_data/prices_round_1_day_-2.csv')

# Calculate spread
df['spread'] = df['ask_price_1'] - df['bid_price_1']

# Filter data for KELP and SQUID_INK
kelp = df[df['product'] == 'KELP']
squid = df[df['product'] == 'SQUID_INK']

# Filter for high volume (>20)
kelp_high_vol = kelp[(kelp['bid_volume_1'] > 20) | (kelp['ask_volume_1'] > 20)]
squid_high_vol = squid[(squid['bid_volume_1'] > 20) | (squid['ask_volume_1'] > 20)]

# Create the visualization
ax = kelp_high_vol.plot(x='timestamp', y='spread', label='KELP (Volume > 20)', figsize=(15, 5))
squid_high_vol.plot(x='timestamp', y='spread', label='SQUID_INK (Volume > 20)', ax=ax)
ax.set_title('Spread Analysis (High Volume)')
ax.set_xlabel('Timestamp')
ax.set_ylabel('Spread')
ax.grid(True)

# Save the plot
plt = ax.get_figure()
plt.savefig('spread_charts_high_volume.png')
print('Charts saved as spread_charts_high_volume.png')

# Print statistics
print("\nHigh Volume Spread Statistics:")
print("\nKELP:")
print(f"Number of high volume instances: {len(kelp_high_vol)}")
print(f"Average spread: {kelp_high_vol['spread'].mean():.2f}")
print(f"Min spread: {kelp_high_vol['spread'].min():.2f}")
print(f"Max spread: {kelp_high_vol['spread'].max():.2f}")

print("\nSQUID_INK:")
print(f"Number of high volume instances: {len(squid_high_vol)}")
print(f"Average spread: {squid_high_vol['spread'].mean():.2f}")
print(f"Min spread: {squid_high_vol['spread'].min():.2f}")
print(f"Max spread: {squid_high_vol['spread'].max():.2f}")

# Create tables for spread = 1
print("\nTimestamps where spread = 1:")
print("\nKELP:")
kelp_spread_1 = kelp[kelp['spread'] == 1]
if not kelp_spread_1.empty:
    print(kelp_spread_1[['timestamp', 'bid_price_1', 'ask_price_1', 'spread']].to_string(index=False))
else:
    print("No instances found")

print("\nSQUID_INK:")
squid_spread_1 = squid[squid['spread'] == 1]
if not squid_spread_1.empty:
    print(squid_spread_1[['timestamp', 'bid_price_1', 'ask_price_1', 'spread']].to_string(index=False))
else:
    print("No instances found")

# Perform regression analysis on spreads
print("\nRegression Analysis:")
plot_regression(kelp_high_vol['spread'].values, "KELP Spread Regression")
plt.savefig('kelp_regression.png')
print("\n")
plot_regression(squid_high_vol['spread'].values, "SQUID_INK Spread Regression")
plt.savefig('squid_regression.png') 
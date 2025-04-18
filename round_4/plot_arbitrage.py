import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file with semicolon separator
df = pd.read_csv('round_4/round_4_data/prices_round_4_day_3.csv', sep=';')

# Print column names to check the structure
print("Column names in the CSV file:")
print(df.columns.tolist())

# Group by timestamp and product to get the mid prices for each product
df_pivot = df.pivot(index='timestamp', columns='product', values='mid_price')

# Calculate price differences
df_pivot['price_3_minus_2'] = df_pivot['VOLCANIC_ROCK_VOUCHER_10500'] - df_pivot['VOLCANIC_ROCK_VOUCHER_10250']
df_pivot['price_2_minus_1'] = df_pivot['VOLCANIC_ROCK_VOUCHER_10250'] - df_pivot['VOLCANIC_ROCK_VOUCHER_10000']
df_pivot['difference_of_differences'] = df_pivot['price_3_minus_2'] - df_pivot['price_2_minus_1']

# Convert timestamp to datetime
df_pivot.index = pd.to_datetime(df_pivot.index.astype(int), unit='ms')

# Create the plot
plt.figure(figsize=(15, 8))

# Create two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), height_ratios=[2, 1])

# First subplot: Price differences
ax1.plot(df_pivot.index, df_pivot['price_3_minus_2'], label='Price 3 - Price 2', color='blue')
ax1.plot(df_pivot.index, df_pivot['price_2_minus_1'], label='Price 2 - Price 1', color='red')
ax1.axhline(y=2, color='green', linestyle='--', label='Arbitrage Threshold')
ax1.set_xlabel('Time')
ax1.set_ylabel('Price Difference')
ax1.set_title('Price Differences Between Adjacent Strike Prices')
ax1.legend()
ax1.grid(True)
ax1.tick_params(axis='x', rotation=45)

# Second subplot: Difference of differences
ax2.plot(df_pivot.index, df_pivot['difference_of_differences'], label='(Price 3 - Price 2) - (Price 2 - Price 1)', color='purple')
ax2.axhline(y=2, color='green', linestyle='--', label='Arbitrage Threshold')
ax2.fill_between(df_pivot.index, df_pivot['difference_of_differences'], 2, 
                 where=(df_pivot['difference_of_differences'] > 2),
                 color='green', alpha=0.3, label='Profitable Arbitrage')
ax2.set_xlabel('Time')
ax2.set_ylabel('Difference of Differences')
ax2.set_title('Arbitrage Opportunity: Difference of Price Differences')
ax2.legend()
ax2.grid(True)
ax2.tick_params(axis='x', rotation=45)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('arbitrage_opportunities_day_3.png')
plt.close()

# Print some statistics
print("\nStatistics for Price 3 - Price 2:")
print(df_pivot['price_3_minus_2'].describe())
print("\nStatistics for Price 2 - Price 1:")
print(df_pivot['price_2_minus_1'].describe())
print("\nStatistics for Difference of Differences:")
print(df_pivot['difference_of_differences'].describe())

# Calculate arbitrage opportunities
arbitrage_opportunities = df_pivot[df_pivot['difference_of_differences'] > 2]
print(f"\nNumber of arbitrage opportunities: {len(arbitrage_opportunities)}")
print(f"Percentage of time with arbitrage opportunities: {len(arbitrage_opportunities)/len(df_pivot)*100:.2f}%")

# Calculate average profit potential during arbitrage opportunities
avg_profit = arbitrage_opportunities['difference_of_differences'].mean() - 2
print(f"Average profit potential during arbitrage opportunities: {avg_profit:.2f}")
print(f"Maximum profit potential: {df_pivot['difference_of_differences'].max() - 2:.2f}")
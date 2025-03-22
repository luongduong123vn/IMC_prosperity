import pandas as pd
import os

# Read the CSV file from round_0/round_0_data folder with semicolon separator
df = pd.read_csv('round_0/round_0_data/trades_round_0_day_0_nn.csv', sep=';')

# Print column names to inspect the structure
print("Column names in the CSV file:", df.columns.tolist())

# Replace symbols in the 'symbol' column
df['symbol'] = df['symbol'].replace({
    'AMETHYSTS': 'RAINFOREST_RESIN',
    'STARFRUIT': 'KELP'
})

# Subtract 3000 from price where symbol is KELP
df.loc[df['symbol'] == 'KELP', 'price'] = df.loc[df['symbol'] == 'KELP', 'price'] - 3000

# Save back to the same file with semicolon separator
df.to_csv('round_0/round_0_data/trades_round_0_day_0_nn.csv', sep=';', index=False)

print("Symbols have been replaced successfully!")
print("Prices for KELP have been adjusted successfully!") 
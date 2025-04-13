import pandas as pd
import numpy as np

# Read the files
df_day0 = pd.read_csv('round_2/round_2_data/round2/prices_round_2_day_0.csv', sep=';')
df_day2 = pd.read_csv('round_2/round_2_data/round2/prices_round_2_day_2.csv')

# Convert empty values to empty strings
df_day2 = df_day2.replace({np.nan: ''})

# Save with semicolon separator
df_day2.to_csv('round_2/round_2_data/round2/prices_round_2_day_2_converted.csv', 
               sep=';', 
               index=False,
               na_rep='')

print("Conversion complete. Saved to prices_round_2_day_2_converted.csv") 
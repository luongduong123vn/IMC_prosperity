import pandas as pd
from pathlib import Path

def check_headers():
    data_dir = Path("round_4_data")
    
    # Check observations file
    print("Observations file headers:")
    obs_df = pd.read_csv(data_dir / "observations_round_4_day_1.csv", nrows=1)
    print(obs_df.columns.tolist())
    
    # Check prices file
    print("\nPrices file headers:")
    prices_df = pd.read_csv(data_dir / "prices_round_4_day_1.csv", nrows=1)
    print(prices_df.columns.tolist())

if __name__ == "__main__":
    check_headers() 
import json
import pandas as pd
from collections import defaultdict

def _process_data_(file):
    with open(file, 'r') as file:
        log_content = file.read()
    sections = log_content.split('Sandbox logs:')[1].split('Activities log:')
    sandbox_log =  sections[0].strip()
    activities_log = sections[1].split('Trade History:')[0]
    # sandbox_log_list = [json.loads(line) for line in sandbox_log.split('\n')]
    trade_history =  json.loads(sections[1].split('Trade History:')[1])
    # sandbox_log_df = pd.DataFrame(sandbox_log_list)
    market_data_df = pd.read_csv(io.StringIO(activities_log))
    trade_history_df = pd.json_normalize(trade_history)
    return market_data_df, trade_history_df

market_data, trade_history = _process_data_('round_2_data/oos_data.log')
market_data.to_csv("round_2_oos.csv", index=False)
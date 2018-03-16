import pandas as pd
import numpy as np
from ipdb import set_trace

from db import base_trade_dates
dates = base_trade_dates.load_trade_dates()
dates = dates[dates.td_type==2].index
df = pd.DataFrame(index=dates)[:-1]
codes = ['120000001', '120000002', '120000014', 'ERI000001', 'ERI000002']
for code in codes:
    df[code]=0
df['confidence'] = 0.2
df['riskparity'] = 0

def select(returns):
    if abs(returns) < 0.01:
        return 0
    # elif abs(returns) < 0.05:
    #     return 1 if returns > 0 else -1
    else:
        return 2 if returns > 0 else -2

from CommandMarkowitz import load_nav_series
future_returns = pd.DataFrame.from_dict({code: load_nav_series(code) for code in codes}).fillna(0)
future_returns = future_returns.reindex(df.index)
for today_idx in range(len(df)-4):
    returns = ((future_returns.iloc[today_idx+4]/future_returns.iloc[today_idx]) - 1).fillna(0)
    returns_parsed = returns.apply(select)
    df.iloc[today_idx, :len(codes)] = returns_parsed

# for day in df.index:
#     if day.month < 7:
#         df.loc[day, 'ERI000001'] = 1
#     else:
#         df.loc[day, 'ERI000001'] = -1

set_trace()
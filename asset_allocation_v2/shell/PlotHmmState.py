from db import *
from asset import *
from trade_date import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipdb import set_trace

#  sh300 = Asset('120000001')
#  tdate = ATradeDate.trade_date()
#  nav = sh300.nav(reindex=tdate)
#  hmm = asset_tc_timing_signal.load_series('21110109')
#  color_map = dict(zip([1,2,3,4,5], ['r', 'g', 'b', 'y', 'k']))
#  for i in hmm.unique():
    #  idx = hmm[hmm == i].index
    #  plt.plot(idx, nav[idx], '.', color=color_map[i], ms=3)
#  set_trace()

rise = asset_tc_timing_signal.load_series('21110204')
oscillate = asset_tc_timing_signal.load_series('21110205')
fall = asset_tc_timing_signal.load_series('21110206')

sh300 = Asset('120000002')
nav = sh300.nav(reindex=rise.index).dropna()


tmp = pd.Series([0 for i in range(len(nav))], index=nav.index)
tmp[rise==1] = 1
tmp[oscillate==1] = 0
tmp[fall==1] = -1
color_map = dict(zip([-1,0,1], ['r', 'g', 'b']))
for i in [-1,0,1]:
    idx = tmp[tmp==i].index
    plt.plot(idx, nav[idx], '.', color=color_map[i], ms=3)
set_trace()

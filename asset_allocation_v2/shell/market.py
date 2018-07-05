import pandas as pd
import numpy as np
from db import *
from asset import *
from trade_date import *
import matplotlib.pyplot as plt
from ipdb import set_trace

sh300 = Asset('120000015')
tdate = ATradeDate.trade_date()
nav = sh300.nav(reindex=tdate).dropna()
inc = np.log(nav).diff().fillna(0)
all_sample = inc.rolling(21*6).sum().shift(-21*3).dropna()
upper_bound = all_sample.quantile(0.667)
lower_bound = all_sample.quantile(0.333)

tmp = inc.rolling(21*6).apply(lambda x: 1 if sum(x)>upper_bound else -1 if sum(x)<lower_bound else 0).shift(-21*3).dropna()

color_map = dict(zip([-1,0,1], ['r', 'g', 'b']))
for i in [-1,0,1]:
    idx = tmp[tmp==i].index
    plt.plot(idx, nav[idx], '.', color=color_map[i], ms=3)

set_trace()



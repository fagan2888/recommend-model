#coding=utf8


import pandas as pd
import numpy as np
import datetime
import calendar
from ipdb import set_trace

def np_pad_to(x, to=1):
    x[np.nanargmax(x)] += (to - np.nansum(x))
    return x

def np_pad_to_2(x, to=1):
    y = (to /np.nansum(x))*np.array(x)
    return y

def nancorr(x, y):
    if x is np.nan or y is np.nan or (len(x) != len(y)):
        return np.nan
    else:
        return np.corrcoef(x,y)[1,0]

if __name__ == '__main__':


    #df_inc  = pd.read_csv('./testcases/portfolio_nav_inc_df.csv', index_col = 'date', parse_dates = ['date'] )
    #df_position = pd.read_csv('./testcases/portfolio_nav_position_df.csv', index_col = 'date', parse_dates = ['date'] )

    #print portfolio_nav(df_inc, df_position)

    # print(np_pad_to([0.1, 0.2, np.nan], 1.0))
    print(np_pad_to_2([0.1, 0.2, np.nan], 1.0))


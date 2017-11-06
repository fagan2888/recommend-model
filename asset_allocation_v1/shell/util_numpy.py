#coding=utf8


import pandas as pd
import numpy as np
import datetime
import calendar

def np_pad_to(x, to=1):
    x[np.nanargmax(x)] += (to - np.nansum(x))
    return x
     
if __name__ == '__main__':


    #df_inc  = pd.read_csv('./testcases/portfolio_nav_inc_df.csv', index_col = 'date', parse_dates = ['date'] )
    #df_position = pd.read_csv('./testcases/portfolio_nav_position_df.csv', index_col = 'date', parse_dates = ['date'] )

    #print portfolio_nav(df_inc, df_position)

    print np_pad_to([0.1, 0.2, np.nan], 1.0)


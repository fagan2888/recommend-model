#coding=utf8

import pandas as pd
import numpy as np


if __name__ == '__main__':

    df = pd.read_csv('./data/nav.csv', index_col = ['date'], parse_dates = ['date'])

    df = df.resample('M').last()

    low_df = df['risk1']
    high_df = df['risk10']

    #compute low_df three month return
    three_month_return = []
    dates = low_df.index
    for i in range(0, len(dates) - 12):
        startv = low_df.loc[dates[i]]
        endv = low_df.loc[dates[i + 12]]
        three_month_return.append( endv / startv - 1)


    three_month_return = []
    dates = high_df.index
    for i in range(0, len(dates) - 3):
        startv = high_df.loc[dates[i]]
        endv = high_df.loc[dates[i + 3]]
        three_month_return.append( endv / startv - 1)
        print dates[i], endv / startv - 1
    #print np.percentile(three_month_return, 5)
    #print np.mean(three_month_return)
    #print three_month_return

    #low_df =
    #print low_df
    #print high_df

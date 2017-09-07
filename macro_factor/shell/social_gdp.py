#coding=utf8

import pandas as pd
import numpy as np


if __name__ == '__main__':

    df = pd.read_csv('./data/gdp.csv', index_col = ['date'], parse_dates = ['date'])
    df['social_financing_3m'] = df['social_financing'].rolling(window = 3, min_periods = 1).sum()
    df = df.dropna()
    print df
    df.to_csv('sgdp.csv')

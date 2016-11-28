#coding=utf8


import pandas as pd
import numpy as np


if __name__ == '__main__':

    df = pd.read_csv('./data/fund_nav.csv', index_col = 'date', parse_dates = ['date'])
    dates = df.index
    dates = dates[-252:]
    print dates

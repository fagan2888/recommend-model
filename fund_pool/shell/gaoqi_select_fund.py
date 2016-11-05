#coding=utf8


import pandas as pd
import numpy as np
from datetime import datetime


if __name__ == '__main__':


    fund_size_df = pd.read_csv('./data/fund_size.csv', index_col = ['date'], parse_dates = ['date'])

    fund_df = pd.read_csv('./manager_fund_performance.csv', index_col = 'SECODE')


    #print fund_df.columns
    fund_df = fund_df['TOTYEARS']
    print fund_df

    #print fund_df

#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import pandas as pd
import sys
sys.path.append('shell')
from db import *
from db import trade_ts_holding_fund
from ipdb import set_trace
import  numpy as np


'''
df = pd.read_csv('ts_share_fund.csv', index_col = ['ts_date'], parse_dates = ['ts_date'])
df = df.loc[:, ['ts_fund_code', 'ts_amount']]
fund_pool = asset_ra_pool_fund.get_pool()
fund_pool = fund_pool.groupby('ra_fund_code').last()
fund_pool = fund_pool.ra_pool.to_dict()
df.ts_fund_code = df.ts_fund_code.apply(lambda x: fund_pool.get('%06d'%x,np.nan))
df = df.dropna()
set_trace()
df = df.reset_index()
df = df.groupby(['ts_date', 'ts_fund_code']).sum()
'''

def cal_ratio():
    fund_pool = asset_ra_pool_fund.get_pool()
    fund_pool = fund_pool.groupby('ra_fund_code').last()
    fund_pool = fund_pool.ra_pool.to_dict()

    date_range = pd.date_range('2017-01-01', '2018-05-30')
    pool_ids = ['11110100', '11110200', '11120200', '11120500', '11120501', '11310100', '11400100', '11210100', '11210200']
    df_result = {}
    for pool_id in pool_ids:
        df_result[pool_id] = {}
    for date in date_range:
        df = trade_ts_holding_fund.find(date)
        df.ts_fund_code = df.ts_fund_code.apply(lambda x: fund_pool.get(x,np.nan))
        df = df.dropna()
        df = df.groupby('ts_fund_code').sum()
        df_ratio = df/df.sum()
        for k, v in df_ratio.iterrows():
            df_result[k][date] = v['ts_amount']

    df_result = pd.DataFrame(df_result)
    df_result = df_result.fillna(0.0)
    df_result.to_csv('ratio.csv', index_label = 'date')


if __name__ == '__main__':
    cal_ratio()

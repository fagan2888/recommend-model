#coding=utf8


import sys
sys.path.append('shell')
import MySQLdb
import config
import pandas as pd
import re
from datetime import datetime
import numpy as np



def uid_indicator(uid):


    conn  = MySQLdb.connect(**config.db_asset)
    cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    conn.autocommit(True)

    sql = 'select is_date as date, is_nav as nav from is_investor_nav where is_investor_id = %d and is_type = 9' % uid
    df  = pd.read_sql(sql, conn, index_col = ['date'], parse_dates = ['date'])


    df['max'] = df.cummax()
    df['drawdown'] = 1 - df['nav'] / df['max']
    df['inc'] = df['nav'].pct_change().fillna(0.0)


    max_drawdown = max(df['drawdown'])
    ret = df['nav'][-1] / df['nav'][0] - 1
    annual_ret = ((ret + 1) ** (1.0 / len(df))) ** 360 - 1
    std = np.std(df['inc']) * (360 ** 0.5)
    sharpe = (np.mean(df['inc']) * 360 - 0.03) / std

    print ret, annual_ret, max_drawdown, std, sharpe
    


if __name__ == '__main__':
    uid_indicator(100)

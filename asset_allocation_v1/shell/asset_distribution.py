#coding=utf8


import sys
sys.path.append('shell')
import config
import MySQLdb
import pandas as pd

from sklearn import mixture
from sklearn.cluster import KMeans
import numpy as np


if __name__ == '__main__':


    conn  = MySQLdb.connect(**config.db_base)
    cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    conn.autocommit(True)

    sql = 'select ra_index_id, ra_nav_date, ra_nav from ra_index_nav where ra_index_id in (120000001, 120000002, 120000013, 120000014, 120000015)'

    df = pd.read_sql(sql, conn, index_col = ['ra_nav_date', 'ra_index_id'], parse_dates = ['ra_nav_date'])
    df = df[~df.duplicated()]
    df = df.unstack()
    df = df.fillna(method = 'pad')
    df = df.resample('W').last()
    df.columns = df.columns.droplevel(0)


    data = []
    n_clusters = 1
    for col in df.columns:
        vs = df[col]
        vs = vs.dropna()
        rs = vs.pct_change().fillna(0.0).values
        rs.sort()
        data.append([np.percentile(rs, 16), 0, np.percentile(rs, 83)])

    df = pd.DataFrame(data, index = df.columns, columns = ['short', 'mid', 'long'])
    df.index.name = 'index_id'
    df.to_csv('./data/history_long_short_r.csv')
    print df

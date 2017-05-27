#coding=utf8


import sys
sys.path.append('shell')
import config
import pandas as pd
import MySQLdb


if __name__ == '__main__':


    conn  = MySQLdb.connect(**config.db_asset)
    conn.autocommit(True)

    sql = 'select mz_date, mz_asset_id, mz_ratio from mz_highlow_pos where mz_highlow_id = 70052563'

    df = pd.read_sql(sql, conn, index_col = ['mz_date', 'mz_asset_id'])
    df = df.unstack()
    df = df.fillna(0.0)
    df['money'] = df[df.columns[0]] + df[df.columns[1]]

    df.to_csv('pinganbank_pos.csv')
    print df

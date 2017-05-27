#coding=utf8


import sys
sys.path.append('shell')
import config
import pandas as pd
import MySQLdb


if __name__ == '__main__':


    conn  = MySQLdb.connect(**config.db_asset)
    conn.autocommit(True)

    mz_highlow_ids = []
    for i in range(0, 10):
        mz_highlow_ids.append(70052560 + i)


    mz_highlow_ids_str = ','.join([str(highlow_id) for highlow_id in mz_highlow_ids])

    sql = 'select mz_highlow_id, mz_date, mz_nav from mz_highlow_nav where mz_highlow_id in (' + mz_highlow_ids_str + ')'

    df = pd.read_sql(sql, conn, index_col = ['mz_date', 'mz_highlow_id'])
    df = df.unstack()

    df.to_csv('pinganbank_nav.csv')
    print df

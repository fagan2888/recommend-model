#coding=utf8

import sys
sys.path.append('shell')
import config
import pandas as pd
import MySQLdb


if __name__ == '__main__':


    conn  = MySQLdb.connect(**config.db_asset)
    cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    conn.autocommit(True)



    from_id = 800000
    to_id = 800000


    sql = 'select on_online_id, on_date, on_pool_id, on_fund_id, on_fund_code, on_fund_type, on_fund_ratio, on_merged_from from on_online_fund'
    df = pd.read_sql(sql, conn)

    for k ,v in df.groupby(['on_date', 'on_pool_id', 'on_fund_type']):
        print v

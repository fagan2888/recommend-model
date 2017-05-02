#coding=utf8


import MySQLdb
import pandas as pd


asset_allocation = {
    "host": "rdsf4ji381o0nt6n2954.mysql.rds.aliyuncs.com",
    "port": 3306,
    "user": "jiaoyang",
    "passwd": "wgOdGq9SWruwATrVWGwi",
    "db":"asset_allocation",
    "charset": "utf8"
}

if __name__ == '__main__':


    conn  = MySQLdb.connect(**asset_allocation)
    conn.autocommit(True)

    ids = []
    for i in range(1, 9):
        ids.append(20200 + i)

    ids_str = ','.join([str(_id) for _id in ids])

    sql = 'select ra_asset_id, ra_date, ra_nav from ra_composite_asset_nav where ra_asset_id in (' + ids_str + ')'

    df = pd.read_sql(sql, conn, index_col = ['ra_date', 'ra_asset_id'])
    df = df.unstack()
    print df
    df.to_csv('baseline.csv')

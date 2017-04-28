#coding=utf8


import pandas as pd
import numpy as np
import scipy.optimize
import MySQLdb


db_base = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"asset_allocation",
    "charset": "utf8"
}


if __name__ == '__main__':


    conn  = MySQLdb.connect(**db_base)

    gids = []
    for i in range(0, 10):
        gid = 80042810 + i
        gids.append(gid)

    gids_str = ','.join([repr(gid) for gid in gids])

    sql = 'select ra_portfolio_id, ra_date, ra_nav from ra_portfolio_nav where ra_portfolio_id in (' + gids_str + ')'

    df = pd.read_sql(sql, conn, index_col = ['ra_date', 'ra_portfolio_id'])
    df = df.unstack()
    df = df.dropna()
    print df

    df.to_csv('data/nav.csv')

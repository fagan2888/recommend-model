#coding=utf8


import MySQLdb
import pandas as pd

db_asset = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"asset_allocation",
    "charset": "utf8"
}


if __name__ == '__main__':

    '''
    conn  = MySQLdb.connect(**db_asset)
    cur = conn.cursor()
    df = pd.read_csv('./fund_pool_nav.csv', index_col = 'date', parse_dates = ['date'])
    dfr = df.pct_change().fillna(0.0)

    item_base_sql = "replace into ra_composite_asset (globalid, ra_name, ra_calc_type, ra_begin_date) values (%d, '%s', 3, '%s')"
    nav_base_sql = "replace into ra_composite_asset_nav (ra_asset_id, ra_date, ra_nav, ra_inc) values (%d, '%s', '%f', '%f')"

    dates = df.index.values
    dates.sort()
    start_date = dates[0]
    id = 50000
    for col in df.columns:
        sql = item_base_sql % (id, col, start_date)
        print sql
        cur.execute(sql)
        for d in dates:
            ra_nav = df.loc[d, col]
            ra_inc = dfr.loc[d, col]
            sql =  nav_base_sql % (id, d, ra_nav, ra_inc)
            #print sql
            cur.execute(sql)
        conn.commit()
        id = id + 1

    cur.close()
    conn.close()
    #print df
    '''



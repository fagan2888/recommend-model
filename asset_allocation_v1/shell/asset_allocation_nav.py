#coding=utf8



import pandas as pd
import numpy  as np
from datetime import datetime
import MySQLdb



db_asset = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"asset_allocation",
    "charset": "utf8"
}



if __name__ == '__main__':


    conn  = MySQLdb.connect(**db_asset)
    #asset_id = 2016121590


    sql = 'select ra_date, ra_alloc_id, ra_nav from risk_asset_allocation_nav where ra_type = 9'
    df = pd.read_sql(sql, conn, index_col = ['ra_date', 'ra_alloc_id'])
    df = df.unstack()
    df.columns = df.columns.get_level_values(1)


    year = 365
    #mdf = df.resample('M').last()
    dfr = df.pct_change()
    dfr = dfr.rolling(year).sum()
    dfr = dfr.shift(-1 * year)
    dfr.to_csv('asset_allocation_nav.csv')


    vs = []
    ds = []
    dates = df.index
    for i in range(0, len(dates) - year):
        start_d = dates[i]
        last_d = dates[i + year]
        tmp_df = df[df.index >= start_d]
        tmp_df = tmp_df[tmp_df.index <= last_d]
        #print tmp_df
        maxdrawdown = (tmp_df / tmp_df.cummax() - 1).min()
        #print maxdrawdown
        vs.append(maxdrawdown.values)
        ds.append(start_d)
    drawdown_df = pd.DataFrame(vs, index = ds, columns = df.columns)
    #print drawdown_df
    drawdown_df.to_csv('asset_allocation_drawdown.csv')

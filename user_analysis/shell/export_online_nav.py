#coding=utf8


import MySQLdb
import pandas as pd
import datetime


asset_allocation = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"asset_allocation",
    "charset": "utf8"
}


mofang = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"mofang",
    "charset": "utf8"
}


if __name__ == '__main__':

    conn  = MySQLdb.connect(**asset_allocation)
    conn.autocommit(True)

    '''
    sql = 'select on_date as date, on_nav as nav from on_online_nav where on_online_id = 800000 and on_type = 8'
    sql = 'select on_date as date, on_nav as nav from on_online_nav where on_online_id = 800000 and on_type = 9'
    online_df = pd.read_sql(sql, conn, index_col = ['date'])
    print online_df
    online_df.to_csv('./tmp/online_nav.csv')
    df = pd.read_csv('data/index.csv', parse_dates = ['date'], index_col = ['date'])
    df = df[['000300.SH']]
    df = pd.concat([online_df, df], axis = 1, join_axes = [online_df.index])
    df = df.dropna()
    df = df / df.iloc[0]
    df.to_csv('./data/nav.csv')
    print df
    '''


    dfs = []
    for i in range(0, 10):
        sql = 'select on_date as date, on_nav as nav from on_online_nav where on_online_id = 80000%d and on_type = 8' % i
        df = pd.read_sql(sql, conn, index_col = ['date'], parse_dates = ['date'])
        df.columns = ['risk_' + str(i)]
        dfs.append(df)

    conn  = MySQLdb.connect(**mofang)
    conn.autocommit(True)
    sql = 'select ra_date as date, ra_nav as nav from ra_index_nav where ra_index_id = 120000016'
    df = pd.read_sql(sql, conn, index_col = ['date'], parse_dates = ['date'])


    dfs.append(df)
    df = pd.concat(dfs, axis = 1)
    df = df[df.index >= '2016-08-01']
    #df = df[df.index <= '2017-10-18']
    df = df / df.iloc[0]
    #print df
    df.to_csv('./tmp/online_nav.csv')




    '''
    mdf = df.resample('M').last()
    dates = []
    dates.append(df.index[0])
    for d in mdf.index:
        dates.append(d)

    ds = []
    max_drawdowns = []
    for i in range(0, len(dates) - 1):
        tmp_df = df[df.index >= dates[i]]
        tmp_df = tmp_df[tmp_df.index <= dates[i + 1]]
        ds.append(tmp_df.index[-1].strftime('%Y-%m'))

        mdds = []
        for col in tmp_df.columns:
            sr = tmp_df[col]
            cummax_sr = sr.cummax()
            drawdown_sr = 1.0 - sr / cummax_sr
            max_drawdown = max(drawdown_sr)
            mdds.append(max_drawdown)
        max_drawdowns.append(mdds)

    max_drawdown_df = pd.DataFrame(max_drawdowns, index = ds, columns = df.columns)
    #print max_drawdown_df
    max_drawdown_df.to_csv('tmp/max_drawdown.csv')

    #print df

    #df = pd.concat(dfs, axis = 1)
    #df = df[df.index >= '2016-07-01']
    #df = df[df.index <= '2017-07-31']
    #df = df / df.iloc[0]
    #print df


    #df.to_csv('./tmp/online_nav.csv')
    '''

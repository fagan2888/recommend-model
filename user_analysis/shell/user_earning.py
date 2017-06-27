#coding=utf8


import MySQLdb
import pandas as pd
import numpy as np


asset_allocation = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"asset_allocation",
    "charset": "utf8"
}

portfolio_statistics = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"portfolio_statistics",
    "charset": "utf8"
}


trade = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "passwd": "Mofang123",
    "db":"trade",
    "charset": "utf8"
}


def earning(nav_df):

    nav_df['io_share'] = nav_df['amount_io'] / nav_df['nav']
    nav_df['share'] = nav_df['io_share'].shift(1).fillna(0.0).cumsum()
    nav_df['amount'] = nav_df['share'] * nav_df['nav']
    nav_df['amount_diff'] = nav_df['amount'].diff().fillna(0.0)
    nav_df['earning'] = nav_df['amount_diff'] - nav_df['amount_io'].shift(1).fillna(0.0)

    return nav_df


if __name__ == '__main__':


    conn  = MySQLdb.connect(**trade)
    conn.autocommit(True)


    sql = 'select mp_uid, mp_risk, mp_placed_date from mf_portfolio_trade_statuses'
    df = pd.read_sql(sql, conn, index_col = ['mp_uid'], parse_dates = ['mp_placed_date'])

    high_risk_uids = []
    for k, v in df.groupby(df.index):
        if v['mp_risk'].values[-1] >= 0.7:
            high_risk_uids.append(k)


    conn  = MySQLdb.connect(**portfolio_statistics)
    conn.autocommit(True)


    sql = 'select ds_uid, ds_portfolio_id, ds_trade_date, ds_trade_type, ds_amount from ds_order'
    df = pd.read_sql(sql, conn, index_col = ['ds_uid'], parse_dates = ['ds_trade_date'])
    df = df.loc[high_risk_uids]
    conn.close()

    #print df['ds_trade_type'].isin([20, 21, 30, 31])
    mask = df['ds_trade_type'].isin([10, 11])
    mask[mask.values == True] = 1.0
    mask[mask.values == False] = -1.0

    df['ds_amount'] = df['ds_amount'] * mask

    asset_io = df.reset_index()
    asset_io = asset_io.reset_index()
    asset_io = asset_io[['ds_trade_date', 'ds_amount']]
    asset_io = asset_io.set_index('ds_trade_date')
    asset_io = asset_io.groupby(asset_io.index).sum()

    '''
    conn  = MySQLdb.connect(**asset_allocation)
    conn.autocommit(True)
    sql = 'select ra_date, ra_nav as nav from ra_portfolio_nav where ra_portfolio_id = 80062600 and ra_type = 9'
    nav_df = pd.read_sql(sql, conn, index_col = ['ra_date'])
    nav_df.index.name = 'date'
    asset_io = asset_io.reindex(nav_df.index).fillna(0.0)
    nav_df['amount_io'] = asset_io['ds_amount']
    nav_df = earning(nav_df)
    #nav_df.to_csv('user_nav.csv')
    print nav_df.sum()['earning']
    '''

    df = pd.read_csv('./data/nav.csv', index_col = ['date'], parse_dates = ['date'])
    df = df[df.index >= '2016-05-01']
    df = df.dropna(axis = 1)
    asset_io = asset_io.reindex(df.index)
    for code in df.columns:
        nav_df = df[[code]]
        nav_df.columns = ['nav']
        nav_df['amount_io'] = asset_io['ds_amount']
        nav_df = earning(nav_df)
        #nav_df.to_csv('user_nav.csv')
        print code, ',', nav_df.sum()['earning']

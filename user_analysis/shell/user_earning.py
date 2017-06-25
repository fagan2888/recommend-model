#coding=utf8


import MySQLdb
import pandas as pd


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


if __name__ == '__main__':


    conn  = MySQLdb.connect(**portfolio_statistics)
    conn.autocommit(True)


    sql = 'select ds_uid, ds_portfolio_id, ds_trade_date, ds_trade_type, ds_amount from ds_order'
    df = pd.read_sql(sql, conn, index_col = ['ds_uid'])
    conn.close()

    #print df['ds_trade_type'].isin([20, 21, 30, 31])
    mask = df['ds_trade_type'].isin([10, 11])
    mask[mask.values == True] = 1.0
    mask[mask.values == False] = -1.0

    df['ds_amount'] = df['ds_amount'] * mask

    #for uid in df.index.unique():
    #    if len(df.loc[uid]) > 10:
    #        print df.loc[uid]

    uid = 1000082620
    asset_io = df.loc[uid]
    asset_io = asset_io.reset_index()
    asset_io = asset_io[['ds_trade_date', 'ds_amount']]
    asset_io = asset_io.set_index('ds_trade_date')
    asset_io = asset_io.groupby(asset_io.index).sum()


    conn  = MySQLdb.connect(**asset_allocation)
    conn.autocommit(True)
    sql = 'select ra_date, ra_nav from ra_portfolio_nav where ra_portfolio_id = 80062600 and ra_type = 9'
    nav_df = pd.read_sql(sql, conn, index_col = ['ra_date'])

    asset_io = asset_io.reindex(nav_df.index).fillna(0.0)
    nav_df['ds_amount'] = asset_io['ds_amount']


    nav_df['io_share'] = nav_df['ds_amount'] / nav_df['ra_nav']
    nav_df['share'] = nav_df['io_share'].shift(1).fillna(0.0).cumsum()
    nav_df['amount'] = nav_df['share'] * nav_df['ra_nav']
    nav_df['amount_diff'] = nav_df['amount'].diff().fillna(0.0)
    nav_df['earning'] = nav_df['amount_diff'] - nav_df['ds_amount'].shift(1).fillna(0.0)
    print nav_df

    nav_df.to_csv('user_nav.csv')

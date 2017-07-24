#coding=utf8


import sys
sys.path.append('shell')
import MySQLdb
import config
import pandas as pd
import re
import statsmodels.api as sm



if __name__ == '__main__':

    portfolio_id = sys.argv[1]

    conn  = MySQLdb.connect(**config.db_base)
    cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    conn.autocommit(True)

    sql = 'select ra_date as date, ra_nav as nav from ra_index_nav where ra_index_id = 120000001'
    hs300_df = pd.read_sql(sql, conn, index_col = ['date'], parse_dates = ['date'])
    conn.close()

    conn  = MySQLdb.connect(**config.db_asset)
    cur   = conn.cursor(MySQLdb.cursors.DictCursor)
    conn.autocommit(True)

    sql = 'select ra_date as date, ra_nav as nav from ra_portfolio_nav where ra_portfolio_id = %d and ra_type = 9' % (int)(portfolio_id)
    portfolio_df = pd.read_sql(sql, conn, index_col = ['date'], parse_dates = ['date'])

    conn.close()

    hs300_dfr = hs300_df.pct_change().fillna(0.0)
    portfolio_dfr = portfolio_df.pct_change().fillna(0.0)

    dates = set(hs300_dfr.index) & set(portfolio_dfr.index)
    dates = list(dates)
    dates.sort()
    hs300_dfr = hs300_dfr.loc[dates]
    portfolio_dfr = portfolio_dfr.loc[dates]


    X = hs300_dfr['nav'].values
    y = portfolio_dfr['nav'].values
    X = sm.add_constant(X)
    results = sm.OLS(y, X).fit()
    params = results.params
    alpha = params[0]
    beta  = params[1]

    print 'alpha' , ':', alpha
    print 'beta' , ':', beta

    

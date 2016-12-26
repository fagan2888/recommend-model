#conding=utf8



import sys
sys.path.append('shell')
import config
import MySQLdb
import pandas as pd
import datetime



if __name__ == '__main__':


    conn  = MySQLdb.connect(**config.db_base)
    cur   = conn.cursor()

    df = pd.read_csv('./data/bank_fin.csv', index_col = ['date'], parse_dates = ['date'])
    dates = df.index
    startd = dates[0]
    lastd = dates[-1]

    dates = pd.date_range(startd, lastd)
    df = df.reindex(dates)
    df = df.fillna(method = 'pad')
    df = df / 100
    r_df = df / 365
    nav_df = (r_df + 1).cumprod()

    ra_type = 5
    ra_type_calc = 3
    ra_retime = dates[0]

    for col in df.columns:
        name = 'bank_fin_' + col
        code = None
        globalid = None
        if col == 'month':
            code = '999001'
            globalid = 33056941
        elif col == 'year':
            code = '999002'
            globalid = 33056942
        sql = "replace into ra_fund (globalid, ra_code, ra_name, ra_type, ra_type_calc, ra_regtime, ra_volume, ra_mask, created_at, updated_at) values (%d, '%s', '%s', %d, %d, '%s', 0.00, 0, '%s', '%s')" % (globalid, code, name, ra_type, ra_type_calc, ra_retime, datetime.datetime.now(), datetime.datetime.now())
        print sql
        cur.execute(sql)

        ds = df.index
        for d in ds:
            r = r_df.loc[d, col]
            nav = nav_df.loc[d, col]
            sql = "replace into ra_fund_nav (ra_fund_id, ra_code, ra_date, ra_type, ra_type_calc, ra_nav_date, ra_nav, ra_inc, ra_nav_acc, ra_nav_adjusted, ra_inc_adjusted, ra_return_daily, ra_mask, created_at, updated_at) values (%d, '%s', '%s', 3, 3, '%s', 1.0, 0.0, 1.0, %f, %f, %f, 0, '%s', '%s')" % (globalid, code, d, d, nav, r, r * 10000, datetime.datetime.now(), datetime.datetime.now())
            print sql
            cur.execute(sql)

    conn.commit()
    conn.close()

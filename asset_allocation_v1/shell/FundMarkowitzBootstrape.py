#coding=utf8


import sys
sys.path.append('shell')
import pandas as pd
from db import database
from sqlalchemy import MetaData, Table, select, func
import Portfolio


if __name__ == '__main__':

    db = database.connection('base')
    metadata = MetaData(bind=db)

    t = Table('ra_fund', metadata, autoload=True)
    columns = [
        t.c.ra_code.label('code'),
    ]
    s = select(columns).where(t.c.ra_type == 1).where(t.c.ra_mask == 0)
    code_df = pd.read_sql(s, db)
    codes = []
    for code in code_df.values:
        codes.append(code[0])

    t = Table('ra_fund_nav', metadata, autoload=True)
    columns = [
        t.c.ra_code.label('code'),
        t.c.ra_date.label('date'),
        t.c.ra_nav.label('nav'),
    ]
    s = select(columns).where(t.c.ra_type.in_(codes))
    df = pd.read_sql(s, db, index_col = ['date', 'code'])
    df = df.unstack()
    df = df.fillna(method = 'pad')
    df = df.resample('W-FRI').last()
    dfr = df.pct_change()

    dates = dfr.index
    for i in range(500, len(dates)):
        tmp_dfr = dfr.iloc[i - 26 : i].copy()
        tmp_dfr = tmp_dfr.dropna(axis = 1)
        bound = []
        for col in tmp_dfr.columns:
            bound.append({'sum1': 0,    'sum2' : 0,   'upper': 1.0,  'lower': 0.0})
        risk, ret, ws, sharpe = Portfolio.markowitz_bootstrape(tmp_dfr, bound)
        print dates[i], ws

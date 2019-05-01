#coding=utf-8
'''
Created on: Mar. 6, 2019
Modified on: May. 1, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import multiprocessing
import numpy as np
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_stock_financial_data(stock_ids, begin_date=None, end_date=None, reindex=None, fill_method='pad'):

    if isinstance(stock_ids, str):
        stock_ids = [stock_ids]
    elif isinstance(stock_ids, (tuple, set)):
        stock_ids = list(stock_ids)
    elif isinstance(stock_ids, dict):
        stock_ids = list(stock_ids.values())
    else:
        if isinstance(stock_ids, (pd.Index, pd.Series, pd.DataFrame)):
            stock_ids = stock_ids.values
        if isinstance(stock_ids, np.ndarray):
            stock_ids = stock_ids.reshape(-1).tolist()

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count//2)

    res = pool.map(load_stock_financial_data_df, stock_ids)

    pool.close()
    pool.join()

    df = pd.concat(res).unstack()
    if fill_method is not None:
        df.fillna(method=fill_method, inplace=True)

    if begin_date is not None:
        df = df.loc[begin_date:]
    if end_date is not None:
        df = df.loc[:end_date]
    if reindex is not None:
        df = df.reindex(reindex)

    return df

def load_stock_financial_data_df(stock_id):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_sk_finindic', metadata, autoload=True)

    columns = [
        t.c.SECODE.label('stock_id'),
        t.c.TRADEDATE.label('trade_date'),
        t.c.NEGOTIABLEMV.label('negotiable_market_value'),
        t.c.TOTMKTCAP.label('total_market_cap'),
        t.c.TURNRATE.label('turn_rate'),
        t.c.PELFY.label('pe_lfy'),
        t.c.PETTM.label('pe_ttm'),
        t.c.PEMRQ.label('pe_mrq'),
        t.c.PELFYNPAAEI.label('pe_lfy_npaaei'),
        t.c.PETTMNPAAEI.label('pe_ttm_npaaei'),
        t.c.PEMRQNPAAEI.label('pe_mrq_npaaei'),
        t.c.PB.label('pb'),
        t.c.PSLFY.label('ps_lfy'),
        t.c.PSTTM.label('ps_ttm'),
        t.c.PSMRQ.label('ps_mrq'),
        t.c.PCLFY.label('pc_lfy'),
        t.c.PCTTM.label('pc_ttm'),
        t.c.DY.label('dy'),
        t.c.EQV.label('eqv'),
        t.c.EV.label('ev'),
        t.c.EVEBITDA.label('evebitda'),
        t.c.EVPS.label('evps'),
        t.c.PEGTTM.label('pegttm'),
        t.c.PCNCF.label('pcncf'),
        t.c.PCNCFTTM.label('pcncf_ttm'),
        t.c.PETTMRN.label('pe_ttm_rn'),
        t.c.LYDY.label('lydy')
    ]

    s = select(columns).where(t.c.SECODE==stock_id)

    df = pd.read_sql(s, engine, index_col=['trade_date', 'stock_id'], parse_dates=['trade_date'])

    return df


if __name__ == '__main__':

    load_stock_financial_data(stock_ids=np.array([['2010000001']]))


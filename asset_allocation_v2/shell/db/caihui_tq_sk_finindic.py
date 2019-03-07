#coding=utf-8
'''
Created on: Mar. 6, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_stock_fin_indic(begin_data=None, end_data=None, stock_ids=None):

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('tq_sk_finindic', metadata, autoload=True)

    columns = [
        t.c.SYMBOL.label('stock_code'),
        t.c.SECODE.label('stock_id'),
        t.c.TRADEDATE.label('date'),
        t.c.LTDATE.label('ldate'),
        t.c.TCLOSE.label('tclose'),
        t.c.NEGOTIABLEMV.label('negotiablemv'),
        t.c.TOTMKTCAP.label('tot_mkt_cap'),
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
        t.c.ISVALID.label('is_valid'),
        t.c.TMSTAMP.label('tmstamp'),
        t.c.LYDY.label('lydy')
    ]

    s = select(columns)
    if begin_data is not None:
        s = s.where(t.c.TRADEDATE>=begin_date)
    if end_date is not None:
        s = s.where(t.c.TRADEDATE<=end_date)
    if stock_ids is not None:
        s = s.where(t.c.SECODE.in_(stock_ids))

    df = pd.read_sql(s, engine, parse_dates=['date'])

    df = df.set_index(['stock_id', 'date'])

    return df


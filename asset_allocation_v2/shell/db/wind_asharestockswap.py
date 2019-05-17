#coding=utf-8
'''
Created on: May. 8, 2019
Modified on: May. 17, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import numpy as np
import pandas as pd
from . import database
from . import wind_ashareeodprices
from . import util_db


logger = logging.getLogger(__name__)


def load_a_stock_swap(transferer_stock_ids=None):

    transferer_stock_ids = util_db.to_list(transferer_stock_ids)

    engine = database.connection('wind')
    metadata = MetaData(bind=engine)
    t = Table('AShareStockSwap', metadata, autoload=True)

    columns = [
        t.c.TRANSFERER_WINDCODE.label('transferer_stock_id'),
        t.c.TARGETCOMP_WINDCODE.label('targetcomp_stock_id'),
        t.c.TRANSFERER_CONVERSIONPRICE.label('transferer_conversion_prc'),
        t.c.TARGETCOMP_CONVERSIONPRICE.label('targetcomp_conversion_prc'),
        t.c.CONVERSIONRATIO.label('conversion_ratio'),
        t.c.LASTTRADEDATE.label('last_trade_date'),
        t.c.EQUITYREGISTRATIONDATE.label('equity_registration_date'),
        t.c.ANN_DT.label('ann_date')
    ]

    s = select(columns)
    if transferer_stock_ids is not None:
        s = s.where(t.c.TRANSFERER_WINDCODE.in_(transferer_stock_ids))

    df = pd.read_sql(s, engine, parse_dates=['last_trade_date', 'equity_registration_date', 'ann_date'])

    df.sort_values(
        by=['transferer_stock_id', 'targetcomp_stock_id', 'ann_date'],
        ascending=True,
        inplace=True
    )
    df.drop_duplicates(
        subset=['transferer_stock_id', 'targetcomp_stock_id'],
        keep='last',
        inplace=True
    )
    df.sort_values(
        by=['equity_registration_date', 'transferer_stock_id', 'targetcomp_stock_id'],
        ascending=True,
        inplace=True
    )

    df_prc = wind_ashareeodprices.load_a_stock_price(
        stock_ids=df.transferer_stock_id.drop_duplicates(),
        fill_method='pad'
    )
    if df.size > 0:
        df.loc[:, 'transferer_stock_prc'] = df.apply(
            lambda ser: df_prc.loc[ser.last_trade_date, ser.transferer_stock_id] if ser.notna().last_trade_date else np.nan,
            axis='columns'
        )
    else:
        df.loc[:, 'transferer_stock_prc'] = []

    df.drop(
        ['last_trade_date', 'ann_date'],
        axis='columns',
        inplace=True
    )

    return df


if __name__ == '__main__':

    load_a_stock_swap('600270.SH')


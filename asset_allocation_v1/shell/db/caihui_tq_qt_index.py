#coding=utf8

from sqlalchemy import MetaData, Table, select, func
import pandas as pd
import logging
import database
from dateutil.parser import parse
import sys
logger = logging.getLogger(__name__)
def load_index_daily_data(secode, start_date=None, end_date=None):
    db = database.connection('caihui')
    metadata = MetaData(bind=db)
    t = Table('tq_qt_index', metadata, autoload=True)

    columns = [
        t.c.TRADEDATE.label('date'),
        t.c.TCLOSE.label('close'),
        t.c.THIGH.label('high'),
        t.c.TLOW.label('low'),
        t.c.VOL.label('volume') if secode !='2070006521' else t.c.AMOUNT.label('volume'),
        t.c.TOPEN.label('open'),
    ]
    s = select(columns).where(t.c.SECODE == secode)
    if start_date:
        s = s.where(t.c.TRADEDATE >= start_date)
    if end_date:
        s = s.where(t.c.TRADEDATE <= end_date)
    s = s.where(t.c.ISVALID == 1).order_by(t.c.TRADEDATE.asc())
    df = pd.read_sql(s, db, index_col = ['date'], parse_dates=['date'])
    return df

if __name__ == "__main__":
    assets = {
            '120000001':'2070000060', #沪深300
            '120000002':'2070000187', #中证500
            '120000013':'2070006545', #标普500指数
            '120000014':'2070000626', #黄金指数
            '120000015':'2070000076', #恒生指数
            '120000028':'2070006521', #南华商品指数
            '120000029':'2070006789', #标普高盛原油商品指数收益率
    }
    for key, value in assets.items():
        df = load_index_daily_data(value, '20050101', '20170531')
        df.to_csv(key+"_ori_day_data.csv")








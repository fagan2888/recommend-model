#coding=utf8

from sqlalchemy import MetaData, Table, select, func
import pandas as pd
import logging
import database
from dateutil.parser import parse
from ipdb import set_trace
logger = logging.getLogger(__name__)
def load_view_strength(viewid):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('mc_view_strength', metadata, autoload=True)
    columns = [
        t.c.mc_date,
        t.c.mc_inc,
    ]
    s = select(columns).where(t.c.mc_view_id == viewid)
    df = pd.read_sql(s, db, index_col = ['mc_date'], parse_dates = ['mc_date'])
    return df

def get_view_id(asset):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('mc_view', metadata, autoload=True)
    columns = [
        t.c.globalid
    ]
    s = select(columns).where(t.c.mc_asset_id == asset)
    id_ = s.execute()
    id_ = id_.fetchall()
    id_ = id_[0][0]

    return id_


if __name__ == '__main__':
    vs = load_view_strength('MC.VW0001')
    print vs.tail()

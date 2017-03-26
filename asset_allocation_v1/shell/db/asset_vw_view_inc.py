#coding=utf8

from sqlalchemy import MetaData, Table, select, func
import pandas as pd
import logging
import database
from dateutil.parser import parse
logger = logging.getLogger(__name__)
def get_asset_newest_view(viewid):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('vw_view_inc', metadata, autoload=True)
    # Session = sessionmaker(bind=db)
    # qry = session.query(
    #     func.max(t.vw_date).label("newest_date"), \
    # )
    # metadata = MetaData(bind=db)
    # t = Table('vw_view_inc', metadata, autoload=True)
    columns = [
        func.max(t.c.vw_date).label('newest_date'),
    ]
    s = select(columns).where(t.c.vw_view_id == viewid)
    df = pd.read_sql(s, db)
    return df

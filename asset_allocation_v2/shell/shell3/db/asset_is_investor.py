#coding=utf8

from sqlalchemy import MetaData, Table, select, func
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
from . import database

from dateutil.parser import parse

logger = logging.getLogger(__name__)

#
# is_investor
#
def load(gids):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('is_investor', metadata, autoload=True)

    columns = [
        t1.c.globalid,
        t1.c.is_risk,
        t1.c.is_name,
    ]

    s = select(columns)

    if gids is not None:
        s = s.where(t1.c.globalid.in_(gids))
    
    df = pd.read_sql(s, db)

    return df

# def max_id_between(min_id, max_id):
#     db = database.connection('asset')
#     metadata = MetaData(bind=db)
#     t = Table('is_investor', metadata, autoload=True)

#     columns = [ t.c.globalid ]

#     # s = select([func.max(t.c.globalid.op('DIV')('10')).label('maxid')]).where(t.c.globalid.between(min_id, max_id))
#     s = select([func.max(t.c.globalid).label('maxid')]).where(t.c.globalid.between(min_id, max_id))

#     return s.execute().scalar()


#coding=utf8

from sqlalchemy import MetaData, Table, select, func
# import string
# from datetime import datetime, timedelta
# import pandas as pd
# import os
# import sys
import logging
import database

from dateutil.parser import parse

logger = logging.getLogger(__name__)

#
# mz_markowitz
#
def load(gids, xtypes=None):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t1 = Table('rm_risk_mgr', metadata, autoload=True)

    columns = [
        t1.c.globalid,
        t1.c.rm_inst_id,
    ]

    s = select(columns)

    if gids is not None:
        s = s.where(t1.c.globalid.in_(gids))
    # if xtypes is not None:
    #     s = s.where(t1.c.rm_type.in_(xtypes))
    
    df = pd.read_sql(s, db)

    return df

def max_id_between(min_id, max_id):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('rm_risk_mgr', metadata, autoload=True)

    columns = [ t.c.globalid ]

    s = select([func.max(t.c.globalid).label('maxid')]).where(t.c.globalid.between(min_id, max_id))

    return s.execute().scalar()


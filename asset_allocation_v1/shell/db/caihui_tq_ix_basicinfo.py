#coding=utf8

from sqlalchemy import MetaData, Table, select, func
import pandas as pd
import logging
import database
from dateutil.parser import parse
from ipdb import set_trace
import sys
logger = logging.getLogger(__name__)
def find_index(estclass):
    db = database.connection('caihui')
    metadata = MetaData(bind=db)
    t = Table('tq_ix_basicinfo', metadata, autoload=True)

    columns = [
        t.c.SECODE.label('secode'),
        t.c.INDEXNAME.label('name'),
        t.c.ESTCLASS.label('type'),
    ]
    s = select(columns).where(t.c.ESTCLASS.in_(estclass))

    df = pd.read_sql(s, db, index_col = ['type'])
    return df


def load_index_name(secodes):

    db = database.connection('caihui')
    metadata = MetaData(bind=db)
    t = Table('tq_ix_basicinfo', metadata, autoload=True)

    columns = [
        t.c.SECODE.label('secode'),
        t.c.INDEXNAME.label('name'),
    ]
    s = select(columns).where(t.c.SECODE.in_(secodes))

    df = pd.read_sql(s, db)
    df = df.set_index('secode')
    df = df.reindex(secodes)

    return df



if __name__ == "__main__":
    pass

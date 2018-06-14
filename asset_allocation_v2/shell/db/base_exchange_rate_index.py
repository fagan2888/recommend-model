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
# base.ra_index
#
def find(globalid):
    db = database.connection('base')
    metadata = MetaData(bind=db)
    t = Table('exchange_rate_index', metadata, autoload=True)

    columns = [
        t.c.globalid,
        t.c.eri_code,
        t.c.eri_name,
        t.c.eri_pcur,
        t.c.eri_excur,
        t.c.eri_pricetype,
        t.c.eri_datasource,
    ]

    s = select(columns).where(t.c.globalid == globalid)

    return s.execute().first()

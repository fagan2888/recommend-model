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

def load_company_names(company_ids):

    db = database.connection('base')
    metadata = MetaData(bind=db)
    t = Table('company_infos', metadata, autoload=True)

    columns = [
        t.c.ci_globalid,
        t.c.ci_name,
    ]

    s = select(columns).where(t.c.ci_globalid.in_(company_ids))

    df = pd.read_sql(s, db, index_col=['ci_globalid'])

    return df


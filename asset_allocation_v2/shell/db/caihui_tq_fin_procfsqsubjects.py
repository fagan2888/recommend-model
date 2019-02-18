#coding=utf-8
'''
Created at Jan. 10, 2019
Author: Shixun Su
Contact: sushixun@licaimofang.com
'''

import logging
from sqlalchemy import MetaData, Table, select, func
import pandas as pd
from . import database


logger = logging.getLogger(__name__)


def load_all():

    engine = database.connection('caihui')
    metadata = MetaData(bind=engine)
    t = Table('TQ_FIN_PROINDICDATASUB', metadata, autoload=True)

    s = 'SELECT * FROM TQ_FIN_PROINDICDATASUB'

    df = pd.read_sql(s, engine)

    return df


if __name__ == '__main__':

    df = load_all()
    df.to_csv('data/tq_fin_procfsqsubjects.csv')

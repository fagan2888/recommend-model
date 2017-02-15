#coding=utf8

from sqlalchemy import MetaData, Table, select, func
# import string
# from datetime import datetime, timedelta
import pandas as pd
# import os
# import sys
import logging
import database

from dateutil.parser import parse

logger = logging.getLogger(__name__)

def categories_name(category, default='未知'):
    tls = {
        11 : '大盘',
        12 : '小盘',
        13 : '上涨',
        14 : '震荡',
        15 : '下跌',
        16 : '成长',
        17 : '价值',
        
        21 : '利率债',
        22 : '信用债',
        23 : '可转债',
        
        31 : '货币',
        
        41 : '标普',
        42 : '黄金',
        43 : '恒生',
    }        

    if category in tls:
        return tls[category]
    else:
        return default

def load_asset_name(id_, category, xtype):
    if xtype == 1:
        s1 = '实验|'
    elif xtype == 9:
        s1 = '线上|'
    else:
        s1 = ''

    s2 = categories_name(category)

    return "%s%s资产" % (s1, s2)

#coding=utf8

from sqlalchemy import MetaData, Table, select, func
# import string
# from datetime import datetime, timedelta
import pandas as pd
import re
# import os
# import sys
import logging
import database
from db import base_ra_index

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

def find(globalid):
    db = database.connection('asset')
    metadata = MetaData(bind=db)
    t = Table('ra_pool', metadata, autoload=True)

    columns = [
        t.c.id,
        t.c.ra_type,
        t.c.ra_date_type,
        t.c.ra_fund_type,
        t.c.ra_lookback,
        t.c.ra_name,
    ]

    s = select(columns).where(t.c.id == globalid)

    return s.execute().first()
    

def match_asset_pool(gid):

    if gid.isdigit():
        xtype = int(gid) / 10000000
    else:
        xtype = re.sub(r'([\d]+)','',gid).strip()

    result = 0
    if xtype == 1:
        result =  gid
    elif xtype == 5:
        xtab = {
            41110100: 11110100, #大盘资产修型	
            41110101: 11110100, #大盘资产修型(实验)	
            41110102: 11110100, #巨潮大盘指数修型	
            41110105: 11110217, #价值资产修型	
            41110200: 11110200, #小盘资产修型	
            41110201: 11110200, #小盘资产修型(实验)	
            41110202: 11110200, #巨潮小盘指数修型	
            41110205: 11110213, #上涨资产修型	
            41110206: 11110214, #震荡资产修型	
            41110207: 11110215, #下跌资产修型	
            41110208: 11110216, #成长资产修型	
            41120200: 11120200, #标普资产修型	
            41120500: 11120500, #恒生资产修型	
            41120501: 11120500, #恒生资产修型(实验)	
            41120502: 11120500, #恒生指数修型	
            41400100: 11400100, #黄金资产修型	
            41400101: 11400100, #黄金资产修型(实验)	
            41400102: 11400100, #黄金指数修型	
        }
        if gid in xtab:
            result = xtab[gid]

    elif xtype == 12:
        #
        # 指数资产
        #
        xtab = {
            120000001: 11110100, # 沪深300, 大盘池
            120000002: 11110200, # 中证500, 小盘池
            120000003: 11110100, # 巨潮大盘, 大盘池
            120000004: 11110200, # 巨潮小盘, 小盘池
            120000010: 11210100, # 国债指数，利率债池
            120000011: 11210200, # 中证信用债指数，信用债池
            120000014: 11400100, # 沪金指数, 黄金池
            120000028: 11400300, # 原油指数, 原油池
            120000029: 11400400, # 商品指数, 商品池
            120000031: 11400500, # 房地产指数, 房地产池
        }
        if gid in xtab:
            result = xtab[gid]
        else:
            asset = base_ra_index.find(gid)
            name = asset['ra_name']
            if '标普' in name:
                result = 11120200
            elif '黄金' in name:
                result = 11400100
            elif '恒生' in name:
                result = 11120500
    elif xtype == 'ERI':

        xtab = {
            'ERI000001': 11120200, # 标普500人民币计价指数, 标普基金池
            'ERI000002': 11120500, # 恒生指数人民币计价指数, 恒生基金池
        }
        if gid in xtab:
            result = xtab[gid]
    else:
        pass

    return result

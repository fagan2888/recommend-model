# coding=utf-8
#'''
#1、各个客户的，近6个月收益
#2、各个客户的，最近1个月对应等级
#3、中证全债指数，上证指数净值
#4、风险等级的，近6个的月收益
#5、各个客户的，等级 对应  标杆的收益
#'''
import sys
sys.path.append('./shell')
import MySQLdb
import config
import trade_date
from db import database
import pandas as pd
import numpy as np
from config import uris
import logging
from sqlalchemy import *
from sqlalchemy.engine import Engine as engine
from dateutil.parser import parse
from datetime import datetime, timedelta
from time import time
from ipdb import set_trace


##########################################################################
# set_trace()
trade_month = trade_date.ATradeDate().month_trade_date(begin_date = '2018-01-01')
#print trade_month


####1、各个客户的，近6个月收益   ### trade库，ts_holding_nav表，ts_order表
start = time()
db =database.connection('trade')
metadata = MetaData(bind=db)
t1 = Table('ts_holding_nav',metadata,autoload=True)
columns = [
    t1.c.ts_uid,
    t1.c.ts_date,
    t1.c.ts_nav,
]
# s = select(columns).where(t1.c.ts_uid == 1000110749)
# s = select(columns).where(t1.c.ts_uid <= 1000000749)
s = select(columns)
df1 = pd.read_sql(s,db,index_col=['ts_date'],parse_dates=['ts_date'])
end = time()
print('耗时: %s 秒'%(end-start))
#print 'df1',df1.tail(30)


####2、各个客户的，最近1个月对应等级   ### trade库，ts_holding_nav表，ts_order表

start = time()
db =database.connection('trade')
metadata = MetaData(bind=db)
t2 = Table('ts_order',metadata,autoload=True)
columns = [
    t2.c.ts_uid,
    t2.c.ts_placed_date,
    t2.c.ts_risk,
]
s = select(columns)
df2 = pd.read_sql(s,db,index_col=['ts_placed_date'],parse_dates=['ts_placed_date'])
end = time()
df2 = df2.reset_index().set_index(['ts_uid','ts_placed_date'],drop=True).sort_index(ascending=False)
df2 = df2.groupby(level=[0]).first()
print('耗时：%s 秒'%(end - start))

####3、中正全债指数，上证指数  ###  mofang库，ra_index_nav表     base

start = time()
ids = ['120000009','120000016']#前者：中证全债指数，后者：上证指数
db =database.connection('base')
metadata = MetaData(bind=db)
t3 = Table('ra_index_nav',metadata,autoload=True)
columns = [
    t3.c.ra_index_id,
    t3.c.ra_date,
    t3.c.ra_nav,
]
s = select(columns).where(t3.c.ra_index_id.in_(ids))
df3 = pd.read_sql(s,db,index_col=['ra_date'],parse_dates=['ra_date'])
end = time()
print ('耗时：%s 秒'%(end - start))

####4、风险等级的，近6个月收益  ####asset_allocation库，ra_composition_asset表，ra_composition_asset_nav表

# 中证全债和上证指数金6个月的收益率
df_qz = df3[df3['ra_index_id'].isin([ids[0]])].drop(['ra_index_id'],axis=1,inplace=False).reindex(trade_month).pct_change().dropna()
df_sz = df3[df3['ra_index_id'].isin([ids[1]])].drop(['ra_index_id'],axis=1,inplace=False).reindex(trade_month).pct_change().dropna()

# 10个风险等级的近6个月收益，即标杆收益率
df_risk_net = pd.DataFrame(index = df_qz.index)
allocation_dict = {
    0: [1.0,0.0],
    1: [0.89,0.11],
    2: [0.78,0.22],
    3: [0.67,0.33],
    4: [0.56,0.44],
    5: [0.45,0.55],
    6: [0.34,0.66],
    7: [0.23,0.77],
    8: [0.12,0.88],
    9: [0.0,1.0]
}

for i in range(10):
    df_risk_net[str(i+1)] = df_qz.values * allocation_dict[i][0] + df_sz.values * allocation_dict[i][1]
print df_risk_net

####5、各个客户的，近6个月收益

uids = df2.index.intersection(df1.ts_uid.values).unique()
df_result = None

for uid in uids:
    print uid

    risk = df2.loc[uid].values[0]
    ratio_h = (risk * 10 - 1) / 9
    ratio_l = 1 - ratio_h
    df_threshold = ratio_h * df_sz + ratio_l * df_qz
    df_threshold = df_threshold.resample('m').last()

    df_ret = df1[df1.ts_uid == uid]
    df_ret = df_ret.loc[:, ['ts_nav']].pct_change().dropna()
    df_ret = df_ret.resample('m').sum()
    df_ret = df_ret[df_ret.index >='2018-02']
    df_ret = df_ret[df_ret.index <='2018-07-31']
    df_ret.columns = ['ra_nav']

    df_res = df_ret - df_threshold
    df_res['uid'] = uid
    if df_result is None:
        df_result = df_res
    else:
        df_result = pd.concat([df_result, df_res])

df_result.index.name = 'date'
df_result = df_result.reset_index()
df_result = df_result.set_index(['uid', 'date'])
df_result = df_result.unstack()
df_result.columns = df_result.columns.levels[1]
df_result = df_result.dropna(axis = 0)
df_result = (np.sign(df_result) + 1) / 2
df_result.to_csv('df_result.csv', index_label = ['uid'])

set_trace()

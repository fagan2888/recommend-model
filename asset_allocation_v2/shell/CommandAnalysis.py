#coding=utf8


import string
import os
import sys
sys.path.append('shell')
import click
import pandas as pd
import numpy as np
import os
import time
import logging
import re
import util_numpy as npu
import MySQLdb
import config


from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func, literal_column
from tabulate import tabulate
from db import database, base_exchange_rate_index, base_ra_index, asset_ra_pool_fund, base_ra_fund, asset_ra_pool, asset_on_online_nav, asset_ra_portfolio_nav, asset_on_online_fund
from util import xdict

import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)
@click.pass_context
def analysis(ctx):
    '''
        analysis something
    '''
    pass


@analysis.command()
@click.option('--portfolio-id', 'optid', default=True, help='portfolio_id')
@click.option('--start-date', 'optstartdate', default=None, help='portfolio pos startdate')
@click.option('--end-date', 'optenddate', default=None, help='portfolio pos endate')
@click.pass_context
def portfolio_turnover_freq(ctx, optid, optstartdate, optenddate):

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    portfolio_alloc_t = Table('ra_portfolio_alloc', metadata, autoload=True)
    portfolio_pos_t = Table('ra_portfolio_pos', metadata, autoload=True)

    alloc_columns = [
                    portfolio_alloc_t.c.globalid,
            ]

    s = select(alloc_columns).where(portfolio_alloc_t.c.ra_portfolio_id == optid.strip())

    alloc_df = pd.read_sql(s, db)

    pos_columns = [
                portfolio_pos_t.c.ra_date,
                portfolio_pos_t.c.ra_fund_code,
                portfolio_pos_t.c.ra_fund_ratio,
            ]


    nums = []
    for globalid in alloc_df['globalid'].ravel():
        s = select(pos_columns).where(portfolio_pos_t.c.ra_portfolio_id == globalid)
        pos_df = pd.read_sql(s, db, index_col = ['ra_date', 'ra_fund_code'], parse_dates = ['ra_date'])
        pos_df = pos_df.groupby(level=[0,1]).sum()
        pos_df = pos_df.unstack(1).fillna(0.0)
        dates = pos_df.index.ravel()
        if optstartdate is not None:
            pos_df = pos_df[pos_df.index >= optstartdate]
        if optenddate is not None:
            pos_df = pos_df[pos_df.index <= optenddate]
        #print globalid, ' change position num ' ,len(pos_df.index)
        nums.append(len(pos_df.index))


    change_position_num_df = pd.DataFrame(nums, index = alloc_df['globalid'], columns = ['num'])
    print(change_position_num_df)
    change_position_num_df.to_csv('tmp/' + str(optid.strip()) + '_change_position_num.csv')


@analysis.command()
@click.option('--new-portfolio-id', 'optnewid', default='', help='portfolio_new_id')
@click.option('--old-portfolio-id', 'optoldid', default='', help='portfolio_old_id')
@click.option('--fee', 'optfee', default=8, help='portfolio_fee')
@click.option('--start-date', 'optstartdate', default=None, help='portfolio pos startdate')
@click.option('--end-date', 'optenddate', default=None, help='portfolio pos endate')
@click.pass_context
def portfolio_compara(ctx, optnewid, optoldid, optfee, optstartdate, optenddate):

    db = database.connection('asset')
    metadata = MetaData(bind=db)
    portfolio_nav_t = Table('ra_portfolio_nav', metadata, autoload=True)

    nav_columns = [
                    portfolio_nav_t.c.ra_date,
                    portfolio_nav_t.c.ra_nav,
            ]

    s = select(nav_columns).where(portfolio_nav_t.c.ra_portfolio_id == optnewid.strip()).where(portfolio_nav_t.c.ra_type == optfee)
    new_nav_df = pd.read_sql(s, db, index_col = ['ra_date'], parse_dates = ['ra_date'])

    s = select(nav_columns).where(portfolio_nav_t.c.ra_portfolio_id == optoldid.strip()).where(portfolio_nav_t.c.ra_type == optfee)
    old_nav_df = pd.read_sql(s, db, index_col = ['ra_date'], parse_dates = ['ra_date'])


    old_nav_df = old_nav_df.reindex(new_nav_df.index).fillna(method = 'pad').fillna(1.0)


    new_nav = new_nav_df['ra_nav'].ravel()
    old_nav = old_nav_df['ra_nav'].ravel()


    interval = 7
    new_minus_old = []
    for i in range(0, len(new_nav_df) - interval):

        new_r = new_nav[i + interval] / new_nav[i] - 1
        old_r = old_nav[i + interval] / old_nav[i] - 1

        new_minus_old.append(new_r - old_r)
    new_minus_old = np.array(new_minus_old)

    print('平均每周多获得收益均值和方差', np.mean(new_minus_old), np.std(new_minus_old))
    print('周战胜概率', 1.0 * len(new_minus_old[new_minus_old > 0]) / len(new_minus_old))


    interval = 30
    new_minus_old = []
    for i in range(0, len(new_nav_df) - interval):

        new_r = new_nav[i + interval] / new_nav[i] - 1
        old_r = old_nav[i + interval] / old_nav[i] - 1

        new_minus_old.append(new_r - old_r)
    new_minus_old = np.array(new_minus_old)

    print('平均每月多获得收益均值和方差', np.mean(new_minus_old), np.std(new_minus_old))
    print('月战胜概率', 1.0 * len(new_minus_old[new_minus_old > 0]) / len(new_minus_old))


    interval = 60
    new_minus_old = []
    for i in range(0, len(new_nav_df) - interval):

        new_r = new_nav[i + interval] / new_nav[i] - 1
        old_r = old_nav[i + interval] / old_nav[i] - 1

        new_minus_old.append(new_r - old_r)
    new_minus_old = np.array(new_minus_old)

    print('平均每两月多获得收益均值和方差', np.mean(new_minus_old), np.std(new_minus_old))
    print('两月战胜概率', 1.0 * len(new_minus_old[new_minus_old > 0]) / len(new_minus_old))


    interval = 90
    new_minus_old = []
    for i in range(0, len(new_nav_df) - interval):

        new_r = new_nav[i + interval] / new_nav[i] - 1
        old_r = old_nav[i + interval] / old_nav[i] - 1

        new_minus_old.append(new_r - old_r)
    new_minus_old = np.array(new_minus_old)

    print('平均每季度多获得收益均值和方差', np.mean(new_minus_old), np.std(new_minus_old))
    print('季度战胜概率', 1.0 * len(new_minus_old[new_minus_old > 0]) / len(new_minus_old))


    interval = 180
    new_minus_old = []
    for i in range(0, len(new_nav_df) - interval):

        new_r = new_nav[i + interval] / new_nav[i] - 1
        old_r = old_nav[i + interval] / old_nav[i] - 1

        new_minus_old.append(new_r - old_r)
    new_minus_old = np.array(new_minus_old)

    print('平均每半年多获得收益均值和方差', np.mean(new_minus_old), np.std(new_minus_old))
    print('半年战胜概率', 1.0 * len(new_minus_old[new_minus_old > 0]) / len(new_minus_old))


    interval = 360
    new_minus_old = []
    for i in range(0, len(new_nav_df) - interval):

        new_r = new_nav[i + interval] / new_nav[i] - 1
        old_r = old_nav[i + interval] / old_nav[i] - 1

        new_minus_old.append(new_r - old_r)
    new_minus_old = np.array(new_minus_old)

    print('平均每年多获得收益均值和方差', np.mean(new_minus_old), np.std(new_minus_old))
    print('年战胜概率', 1.0 * len(new_minus_old[new_minus_old > 0]) / len(new_minus_old))


@analysis.command()
@click.option('--pool-id', 'optpool', default=None, help='pool id')
@click.pass_context
def fund_pool_info(ctx, optpool):


    db = database.connection('base')
    metadata = MetaData(bind=db)
    fund_info_t = Table('fund_infos', metadata, autoload=True)
    company_t = Table('company_infos', metadata, autoload=True)

    fund_info = [
            fund_info_t.c.fi_globalid,
            fund_info_t.c.fi_code,
            fund_info_t.c.fi_name,
            fund_info_t.c.fi_company_id,
    ]

    fund_info_df = pd.read_sql(select(fund_info), db)

    company_info = [
            company_t.c.ci_globalid,
            company_t.c.ci_name,
    ]


    company_info_df = pd.read_sql(select(company_info), db)


    fund_company_df = pd.merge(fund_info_df, company_info_df, left_on = ['fi_company_id'], right_on = ['ci_globalid'])
    fund_company_df = fund_company_df[['fi_globalid','fi_code', 'fi_name', 'ci_name']]
    fund_company_df = fund_company_df.set_index(['fi_globalid'])


    dfs = []
    for _id in optpool.split(','):

        pool_id = _id.strip()
        pool_name = asset_ra_pool.find(pool_id)['ra_name']
        pool_df = asset_ra_pool_fund.load(pool_id)
        pool_df = pool_df['ra_fund_code']
        pool_df = pool_df.reset_index()
        pool_df = pool_df.set_index('ra_fund_id')

        pool_fund_company_df = pd.concat([pool_df, fund_company_df], axis = 1, join_axes = [pool_df.index])
        pool_fund_company_df = pool_fund_company_df[['ra_date','ra_fund_code','fi_name','ci_name']]
        pool_fund_company_df['pool_name'] = pool_name
        #print pool_fund_company_df.head()

        dfs.append(pool_fund_company_df)

    df = pd.concat(dfs)
    print(df.head())
    df.to_csv('fund_pool.csv', encoding='gbk')
    #pool_fund_company_df.to_csv('pool_fund_company_df.csv', encoding = 'gbk')



    '''
    fund_info_df = base_ra_fund.load()
    fund_info_df = fund_info_df[['globalid', 'ra_code', 'ra_name']]
    fund_info_df = fund_info_df.set_index('globalid')
    print fund_info_df
    '''


#线上组合在所有偏股基金的收益回撤排名
@analysis.command()
@click.pass_context
def fund_online_portfolio(ctx):

    conn  = MySQLdb.connect(**config.db_base)
    conn.autocommit(True)

    '''
    sql = 'select ra_code as code from ra_fund where ra_type = 1'
    df = pd.read_sql(sql, conn)
    codes = ','.join(["'" + code[0] + "'" for code in df.values])

    sql = 'select ra_code as code, ra_date as date, ra_nav_adjusted as nav from ra_fund_nav where ra_code in (' + codes + ')'

    df = pd.read_sql(sql, conn, index_col = ['date', 'code'])
    df = df.unstack()

    df.columns = df.columns.get_level_values(1)
    df.index.name = 'date'
    df = df[df.index >= '2017-01-01']
    df.to_csv('./fund_nav_ra_type1.csv')
    '''

    '''
    df = pd.read_csv('./data/ra_fund_qdii.csv', index_col = ['code'])
    codes = ','.join(["'" + '%06d' % code + "'" for code in df.index])
    print codes

    sql = 'select ra_code as code, ra_date as date, ra_nav_adjusted as nav from ra_fund_nav where ra_code in (' + codes + ')'

    df = pd.read_sql(sql, conn, index_col = ['date', 'code'])
    df = df.unstack()

    df.columns = df.columns.get_level_values(1)
    df.index.name = 'date'
    df = df[df.index >= '2017-01-01']
    df.to_csv('./fund_nav_qdii_type4.csv')
    '''

    '''
    conn  = MySQLdb.connect(**config.db_asset)
    conn.autocommit(True)

    dfs = []
    for i in range(0, 10):
        sql = 'select on_date as date, on_nav as nav from on_online_nav where on_online_id = 80000%d and on_type = 8' % i
        df = pd.read_sql(sql, conn, index_col = ['date'], parse_dates = ['date'])
        df.columns = ['risk_' + str(i)]
        dfs.append(df)

    df = pd.concat(dfs, axis = 1)
    df = df[df.index >= '2017-01-01']
    df = df / df.iloc[0]
    df.to_csv('./online_nav.csv')
    '''

    fund_df = pd.read_csv('fund_nav_ra_type1.csv', index_col = ['date'], parse_dates = ['date'])
    qdii_df = pd.read_csv('fund_nav_qdii_type4.csv', index_col = ['date'], parse_dates = ['date'])
    online_df = pd.read_csv('online_nav.csv', index_col = ['date'], parse_dates = ['date'])

    df = pd.concat([fund_df, qdii_df, online_df], axis = 1, join_axes = [online_df.index])
    df = df[df.index >= '2017-01-01']
    df = df[df.index <= '2017-11-30']
    df = df.fillna(method = 'pad')
    df = df.dropna(axis = 1)
    df = df / df.iloc[0]

    r = df.iloc[-1] / df.iloc[0] - 1
    #print r.head()
    cummax_df = df.cummax()
    drawdown_df = 1 - df / cummax_df
    max_drawdown = drawdown_df.max()
    #print max_drawdown.head()

    dfr = df.pct_change().fillna(0.0)
    std = dfr.std()

    shape = (r - 0.03) / (std * (360 ** 0.5))
    #print shape.head()

    data = {}
    data['r'] = r
    data['max_drawdown'] = max_drawdown
    data['shape'] = shape
    df = pd.DataFrame(data)
    #print df.head()

    #df.to_csv('r_maxdrawdown_shape.csv')


    sql = 'select ra_code, ra_volume from ra_fund'
    volume_df = pd.read_sql(sql, conn, index_col = ['ra_code'])
    volume_df = volume_df / (1e8)
    #print volume_df.head()

    #df = df.loc[volume_df.index]
    df = pd.concat([df, volume_df], axis = 1, join_axes = [df.index])
    print(len(df))

    #df.to_csv('r_maxdrawdown_shape_volume.csv')

    sql = 'select fi_code, fi_yingmi_subscribe_status from fund_infos'
    fund_info_df = pd.read_sql(sql, conn, index_col = ['fi_code'])
    fund_info_df = fund_info_df[fund_info_df['fi_yingmi_subscribe_status'] == 0]
    codes = []
    for code in fund_info_df.index:
        codes.append('%06d' % code)
    fund_info_df.index = codes
    print(len(fund_info_df))

    codes = []
    for code in df.index:
        if code in set(fund_info_df.index):
            codes.append(code)
    print(df.index)
    print(fund_info_df.index)
    df = df.iloc[df.index & fund_info_df.index]
    print(len(df))


    #print df.head()
    #print df.tail()



#线上收益归因
@analysis.command()
@click.option('--start-date', 'optsdate', default=None, help='start date')
@click.option('--end-date', 'optedate', default=None, help='end date')
@click.pass_context
def online_return_reason(ctx, optsdate, optedate):

    conn  = MySQLdb.connect(**config.db_asset)
    conn.autocommit(True)


    trade_date_df = pd.read_sql('select on_date from on_online_contrib', conn, index_col = ['on_date'], parse_dates = ['on_date'])
    trade_date = trade_date_df.index
    trade_date = list(set(trade_date))
    trade_date.sort()

    #trade_date_df = trade_date_df[trade_date_df.index < datetime.strptime(optsdate, '%Y-%m-%d').date()]
    trade_date_df = trade_date_df[trade_date_df.index < optsdate]
    #print trade_date_df.index[-1]

    print(trade_date_df.tail())
    gid = 800000
    sql = "SELECT on_fund_id, sum(on_return_value) FROM on_online_contrib WHERE on_online_id = %d AND on_type = 8 \
            AND on_date BETWEEN '%s' AND '%s' GROUP BY on_fund_id" % (gid, optsdate, optedate)

    #print sql
    df = pd.read_sql(sql, conn, index_col = ['on_fund_id'])

    online_nav_ser = asset_on_online_nav.load_series(gid, 8)


    optsdate = trade_date_df.index[-1].strftime('%Y-%m-%d')
    start_nav = online_nav_ser.loc[optsdate]
    end_nav = online_nav_ser.loc[optedate]

    print(optsdate, start_nav, optedate, end_nav)

    print('月初净值 : ', start_nav)
    print('月末净值 : ', end_nav)
    print('净值增长 : ', end_nav - start_nav)
    print('净值增长率 : ', end_nav / start_nav - 1)

    conn.close()

    conn  = MySQLdb.connect(**config.db_base)
    conn.autocommit(True)

    globalids = ','.join([str(gid) for gid in df.index])
    sql = 'select globalid ,ra_code, ra_name from ra_fund where globalid in (' + globalids + ')'
    fund_df = pd.read_sql(sql, conn, index_col = ['globalid'])

    df = pd.concat([fund_df, df], axis = 1, join_axes = [df.index])
    df.columns = ['基金代码','基金名称','收益率贡献']
    df['收益率贡献百分比'] = df['收益率贡献'] / start_nav
    print(df)
    df.to_csv('风险10各个基金收益贡献百分比.csv', encoding = 'gbk')



#标杆组合有费率和沪深300比较
@analysis.command()
@click.option('--start_date', 'optsdate', default=None, help='start date')
@click.option('--end_date', 'optedate', default=None, help='end date')
@click.pass_context
def online_nav(ctx, optsdate, optedate):

    data = {}
    for i in range(0, 10):
        gid = 800000 + i
        risk = None
        if i == 0:
            risk = 'risk10'
        else:
            risk = 'risk' + str(i)

        data[risk] = asset_on_online_nav.load_series(gid, 8)

    online_nav_df = pd.DataFrame(data)
    print(online_nav_df)



#线上风险10和portfolio有费率数据月度收益率
@analysis.command()
@click.pass_context
def online_portfolio_month_nav(ctx):
    ser = asset_on_online_nav.load_series(800000, 8)
    ser = ser.groupby(ser.index.strftime('%Y-%m')).last()
    online_ser = ser.pct_change()
 
    ser = asset_ra_portfolio_nav.load_series('PO.000030', 8)
    ser = ser.groupby(ser.index.strftime('%Y-%m')).last()
    portfolio_ser = ser.pct_change()

    data = {}
    data['online'] = online_ser
    data['po.000030'] = portfolio_ser

    df = pd.DataFrame(data)

    df.to_csv('线上新模型有费率和标杆组合风险10有费率每月收益率.csv')



#线上风险10权益类资产被动管理型基金占比
@analysis.command()
@click.pass_context
def passive_fund_ratio(ctx):


    passive_funds = set()
    for line in open('./data/passive_fund.csv','r').readlines():
        passive_funds.add(line.strip()[0:6])


    dates = []
    data = []


    funds = asset_on_online_fund.load(800000)
    for date, group in funds.groupby(funds.index):
        group = group[(group.on_fund_type == 11101) | (group.on_fund_type == 11102) | (group.on_fund_type == 11202) | (group.on_fund_type == 11205)]

        print(group[group.on_fund_code.isin(passive_funds)])
        passive_ratio = group[group.on_fund_code.isin(passive_funds)].on_fund_ratio.sum()
        active_ratio = group[~group.on_fund_code.isin(passive_funds)].on_fund_ratio.sum()


        ratio_sum = active_ratio + passive_ratio
        if ratio_sum == 0:
            active_ratio = 0.0
            passive_ratio = 0.0
            pass
        else:
            active_ratio = active_ratio / ratio_sum
            passive_ratio = passive_ratio / ratio_sum

        #print date, active_ratio, passive_ratio

        dates.append(date)
        data.append([active_ratio, passive_ratio])

    df = pd.DataFrame(data, index=dates, columns = ['active_ratio', 'passive_ratio'])

    df.to_csv('active_passive_fund_ratio.csv')
    return

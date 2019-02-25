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
from ipdb import set_trace


from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import MetaData, Table, select, func, literal_column
from tabulate import tabulate
from db import database, base_exchange_rate_index, base_ra_index, asset_ra_pool_fund, base_ra_fund, asset_ra_pool, asset_on_online_nav, asset_ra_portfolio_nav, asset_on_online_fund, asset_mz_markowitz_nav, base_ra_index_nav, asset_ra_composite_asset_nav, base_exchange_rate_index_nav, base_ra_fund_nav, asset_mz_highlow_pos
from util import xdict
from trade_date import ATradeDate
from asset import Asset

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

    #conn  = MySQLdb.connect(**config.db_base)
    #conn.autocommit(True)

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

    conn  = MySQLdb.connect(**config.db_asset)
    conn.autocommit(True)

    dfs = []
    for i in range(0, 10):
        sql = 'select on_date as date, on_nav as nav from on_online_nav where on_online_id = 80000%d and on_type = 8' % i
        df = pd.read_sql(sql, conn, index_col = ['date'], parse_dates = ['date'])
        df.columns = ['risk_' + str(i)]
        dfs.append(df)

    df = pd.concat(dfs, axis = 1)
    df = df[df.index >= '2013-12-23']
    df = df / df.iloc[0]
    df.to_csv('./online_nav.csv')

    conn.close()
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
    '''

#线上10个风险等级配置比例
@analysis.command()
@click.pass_context
def online_portfolio_fund(ctx):


    asset_name = {
            '11110100':'大盘',
            '11110106':'高盈利',
            '11110108':'高财务质量',
            '11110110':'食品饮料',
            '11110112':'医药生物',
            '11110114':'银行',
            '11110116':'非银金融',
            '11110200':'小盘',
            '11120200':'美股',
            '11120201':'美股',
            '11120500':'恒生',
            '11120501':'恒生',
            '11210100':'信用债',
            '11210200':'利率债',
            '11310100':'货币',
            '11310101':'货币',
            '11400100':'沪金',
            '11400300':'原油',
        }

    writer = pd.ExcelWriter('10个风险等级月末仓位.xlsx')

    conn  = MySQLdb.connect(**config.db_asset)
    conn.autocommit(True)

    sql = 'select on_online_id, on_date, on_pool_id, on_fund_ratio from on_online_fund'

    df = pd.read_sql(sql, conn, index_col = ['on_online_id', 'on_date', 'on_pool_id'])
    df = df.groupby(level = [0,1,2]).sum()
    df = df.unstack().fillna(0.0)
    df.columns = df.columns.droplevel(0)
    for k , v in df.groupby(df.index.get_level_values(0)):
        v.index = v.index.droplevel(0)
        #print(k)
        dates = pd.date_range(v.index[0], v.index[-1])
        v = v.reindex(dates).fillna(method = 'pad').fillna(0.0)
        v = v[v.index >= '2013-06-15']
        v = v.resample('M').last()
        v = v.rename(columns = asset_name)
        v = v.T
        v = v.groupby(level=[0]).sum()
        v = v.T
        #print(v['大盘']+ v['小盘'])
        #print(v.columns)
        v[''] = ''
        v[''] = ''
        v['股基'] = v['大盘'] + v['小盘'] + v['美股'] + v['恒生']
        v['债基'] = v['信用债'] + v['利率债']
        v['海外'] = v['美股'] + v['恒生'] + v['原油']
        v['国内'] = v['大盘'] + v['小盘'] + v['信用债'] + v['利率债'] + v['货币'] + v['沪金']

        v = v[v.index <= '2018-09-30']
        v.to_excel(writer, '风险等级' + str(k)[-1])
    writer.save()


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
    trade_date_df = trade_date_df.sort_index()

    #trade_date_df = trade_date_df[trade_date_df.index < datetime.strptime(optsdate, '%Y-%m-%d').date()]
    trade_date_df = trade_date_df[trade_date_df.index < optsdate]
    #print trade_date_df.index[-1]

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
    risk_df = online_risk_reason(optsdate, optedate)
    risk_df = risk_df.loc[df.index]
    df['风险贡献'] = risk_df
    print(df)
    df.to_csv('风险10各个基金收益贡献百分比.csv', encoding = 'gbk')


#线上风险归因
def online_risk_reason(optsdate, optedate):

    conn  = MySQLdb.connect(**config.db_asset)

    trade_date_df = pd.read_sql('select on_date from on_online_contrib', conn, index_col = ['on_date'], parse_dates = ['on_date'])
    trade_date = trade_date_df.index
    trade_date = list(set(trade_date))
    trade_date.sort()


    gid = 800000
    sql = "SELECT on_fund_id, sum(on_return_value) FROM on_online_contrib WHERE on_online_id = %d AND on_type = 8 \
            AND on_date BETWEEN '%s' AND '%s' GROUP BY on_fund_id" % (gid, optsdate, optedate)

    df = pd.read_sql(sql, conn, index_col = ['on_fund_id'])
    conn.close()


    conn  = MySQLdb.connect(**config.db_base)
    conn.autocommit(True)

    globalids = ','.join([str(gid) for gid in df.index])
    sql = 'select globalid ,ra_code, ra_name from ra_fund where globalid in (' + globalids + ')'
    fund_df = pd.read_sql(sql, conn, index_col = ['globalid'])

    conn.close()


    conn  = MySQLdb.connect(**config.db_asset)
    conn.autocommit(True)
    sql = "select on_date, on_fund_id, on_fund_ratio from on_online_fund where on_online_id = %s" %  gid
    df = pd.read_sql(sql, conn, index_col = ['on_date', 'on_fund_id'])
    df = df.unstack().reindex(trade_date).fillna(method = 'pad').fillna(0.0)
    df.columns = df.columns.droplevel(0)

    df = df[df.index >= optsdate]
    df = df[df.index <= optedate]

    fund_ratio_df = df
    conn.close()


    conn  = MySQLdb.connect(**config.db_base)
    conn.autocommit(True)
    fund_ids = ','.join([str(fid) for fid in fund_ratio_df.columns])
    sql = "select ra_fund_id, ra_date, ra_nav_adjusted from ra_fund_nav where ra_fund_id in (%s)" % fund_ids
    df = pd.read_sql(sql, conn, index_col = ['ra_date', 'ra_fund_id'])
    df = df.unstack()
    df.columns = df.columns.droplevel(0)

    df = df[df.index >= optsdate]
    df = df[df.index <= optedate]
    df = df.fillna(method = 'pad')

    df_cum_max = df.cummax()

    df_drawdown = 1.0 - df / df_cum_max

    df_drawdown = df_drawdown[fund_ratio_df.columns]

    df_risk = df_drawdown * fund_ratio_df

    conn.close()

    return df_risk.max()

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
    online_nav_df.to_csv('online_nav.csv')
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


#线上风险10权益类资产被动管理型基金占比
@analysis.command()
@click.pass_context
def online_sharpe(ctx):

    df_ret = pd.DataFrame(columns = ['ret', 'std', 'mdd', 'calmar', 'sharpe'])
    for risk in range(800000, 800010):
        df_ret.loc[risk] = cal_online_indic(risk)

    df_ret.to_csv('data/result_fee.csv')
    set_trace()

    return

def cal_online_indic(risk):

    df = pd.read_csv('data/on_online_nav.csv', index_col = ['on_date'], parse_dates = ['on_date'])
    df = df[df.on_type == 8]
    df = df[df.on_online_id == risk]
    df = df['on_nav']

    reindex = ATradeDate.week_trade_date(begin_date = '2017-12-29', end_date = '2018-06-30')
    df = df.reindex(reindex)
    df_ret = df.pct_change().dropna()
    ret = df.iloc[-1] / df.iloc[0] - 1
    weeks = len(df) - 1
    year_ret = ret * 52 / weeks - 0.015
    mdd = -(df / df.rolling(min_periods=1, window=len(df)).max() - 1).min()

    calmar = year_ret / mdd
    std = df_ret.std()
    sharpe = df_ret.mean() / df_ret.std()

    return ret, std, mdd, calmar, sharpe



#线上风险10资产配置比例
@analysis.command()
@click.pass_context
def online_asset_ratio(ctx):
    ratio_df = pd.read_csv('on_online_fund.csv', index_col = ['date','type'], parse_dates = ['date'])
    ratio_df = ratio_df.groupby(level = [0, 1]).sum().unstack().fillna(0.0)

    nav_df = pd.read_csv('on_online_nav.csv', index_col = ['date'], parse_dates = ['date'])

    #ratio_df = ratio_df.reindex(nav_df.index).fillna(method = 'pad')

    df = pd.concat([ratio_df, nav_df], axis = 1, join_axes = [nav_df.index]).fillna(method = 'pad')
    df = df[df.index >= '2018-01-01']
    df.nav = df.nav / df.nav[0]

    df.to_csv('ratio_nav.csv')


#线上风险10权益类资产被动管理型基金占比
@analysis.command()
@click.pass_context
def online_sharpe(ctx):

    df_ret = pd.DataFrame(columns = ['ret', 'std', 'mdd', 'calmar', 'sharpe'])
    for risk in range(800000, 800010):
        df_ret.loc[risk] = cal_online_indic(risk)

    df_ret.to_csv('data/result_fee.csv')
    set_trace()

    return

def cal_online_indic(risk):

    df = pd.read_csv('data/on_online_nav.csv', index_col = ['on_date'], parse_dates = ['on_date'])
    df = df[df.on_type == 8]
    df = df[df.on_online_id == risk]
    df = df['on_nav']

    reindex = ATradeDate.week_trade_date(begin_date = '2017-12-29', end_date = '2018-06-30')
    df = df.reindex(reindex)
    df_ret = df.pct_change().dropna()
    ret = df.iloc[-1] / df.iloc[0] - 1
    weeks = len(df) - 1
    year_ret = ret * 52 / weeks - 0.015
    mdd = -(df / df.rolling(min_periods=1, window=len(df)).max() - 1).min()

    calmar = year_ret / mdd
    std = df_ret.std()
    sharpe = df_ret.mean() / df_ret.std()

    return ret, std, mdd, calmar, sharpe


#货币基金组合与所有货币基金排名
@analysis.command()
@click.pass_context
def monetary_fund_rank(ctx):

    allocate_nav = asset_mz_markowitz_nav.load_series('MZ.MONE00')
    allocate_inc = allocate_nav.pct_change()
    all_monetary_fund_df = base_ra_fund.find_type_fund(3)
    all_monetary_fund_df = all_monetary_fund_df.set_index(['globalid'])
    datas = {}
    for fund_globalid in all_monetary_fund_df.index:
        datas[fund_globalid] = Asset(str(fund_globalid)).nav()
    df_nav = pd.DataFrame(datas)
    df_nav = df_nav.loc[allocate_nav.index]
    df_inc = df_nav.pct_change()
    fund_month_inc = df_inc.groupby(df_inc.index.strftime('%Y-%m')).sum()
    allocate_month_inc = allocate_inc.groupby(allocate_inc.index.strftime('%Y-%m')).sum()

    ranks = []
    for date in allocate_month_inc.index:
        allocate_r = allocate_month_inc.loc[date]
        fund_month_r = fund_month_inc.loc[date].ravel()
        fund_month_r = list(fund_month_r[fund_month_r > 0.0])
        fund_month_r.append(allocate_r)
        fund_month_r.sort(reverse = True)
        print(fund_month_r.index(allocate_r), len(fund_month_r), 1.0 * fund_month_r.index(allocate_r) / len(fund_month_r))
        ranks.append(1.0 * fund_month_r.index(allocate_r) / len(fund_month_r))
    print(np.mean(ranks))



#调仓频率与次数
@analysis.command()
@click.pass_context
def online_turnover_freq(ctx):
    datas = []
    risks = []
    for i in range(0, 10):
        gid = 800000 + i
        df = asset_on_online_fund.load_fund_pos(gid)
        df = df.unstack().fillna(0.0)
        df = df[df.index <= '2018-09-30']
        day_num = len(pd.date_range(df.index[0], df.index[-1]))
        turnover = df.diff().abs().sum().sum() / len(df) / 2
        last_five_year_df = df[df.index >= '2013-09-30']
        last_five_year_turnover = last_five_year_df.diff().abs().sum().sum() / 2
        from_2017_df = df[df.index >= '2016-12-30']
        from_2017_turnover = from_2017_df.diff().abs().sum().sum() / 2
        print(day_num/len(df), turnover, len(last_five_year_df), last_five_year_turnover, len(from_2017_df), from_2017_turnover + 0.5)
        datas.append([day_num/len(df), turnover, len(last_five_year_df), last_five_year_turnover, len(from_2017_df), from_2017_turnover + 0.5]) 
        risks.append('等级' + str(i)) 

    df = pd.DataFrame(datas, index = risks, columns = ['多久调仓一次（单位：天）', '平均每次调仓比例', '过去5年调仓次数', '过去5年整体换手率', '2017年以来调仓次数',  '2017年以来整体换手率'])
    df.index.name = '风险等级'
    print(df)
    df.to_csv('每个风险等级调仓频率与次数.csv',encoding='gbk')


#调仓频率与次数
@analysis.command()
@click.pass_context
def index_nav(ctx):

    index_ids = ['120000001', '120000002', '120000009' , '120000010', '120000011', '120000016' ,'120000017' ,'120000039', '120000080', '120000081','120000082' ,'ERI000001', 'ERI000002']
    data = {}
    for _id in index_ids[0:-2]:
        data[_id] = base_ra_index_nav.load_series(_id)
    for _id in index_ids[-2:]:
        data[_id] = base_exchange_rate_index_nav.load_series(_id)
    df = pd.DataFrame(data)
    #df = df[df.index>='2018-01-01']
    #df = df/df.iloc[0]

    df.to_csv('tmp/index_nav.csv')


#标杆组合
@analysis.command()
@click.pass_context
def benchmark(ctx):

    index_ids = ['120000016', '120000010']
    data = {}
    for _id in index_ids:
        data[_id] = base_ra_index_nav.load_series(_id)
    df = pd.DataFrame(data)

    composite_asset_ids = ['20201','20202', '20203', '20204', '20205', '20206', '20207', '20208']

    data = {}

    for _id in composite_asset_ids:
        nav = asset_ra_composite_asset_nav.load_nav(_id)
        nav = nav.reset_index()
        nav = nav[['ra_date', 'ra_nav']]
        nav = nav.set_index(['ra_date'])
        data[_id] = nav.ra_nav

    bench_df = pd.DataFrame(data)
    benchmark_df = pd.concat([bench_df,df],axis = 1, join_axes = [bench_df.index])

    benchmark_df = benchmark_df[benchmark_df.index >= '2013-01-01']
    benchmark_df = benchmark_df / benchmark_df.iloc[0]
    #print(df.tail())
    #df.to_csv('benchmark.csv')


    conn  = MySQLdb.connect(**config.db_asset)
    conn.autocommit(True)

    dfs = []
    for i in range(0, 10):
        sql = 'select on_date as date, on_nav as nav from on_online_nav where on_online_id = 80000%d and on_type = 8' % i
        df = pd.read_sql(sql, conn, index_col = ['date'], parse_dates = ['date'])
        df.columns = ['risk_' + str(i)]
        dfs.append(df)

    df = pd.concat(dfs, axis = 1)
    df = df[df.index >= '2013-01-04']
    online_df = df / df.iloc[0]
    #df.to_csv('./online_nav.csv')

    online_df.head()
    conn.close()


    df = pd.concat([online_df, benchmark_df], axis = 1, join_axes = [benchmark_df.index])
    print(df.head())

    df.to_csv('./online_benchmark.csv')


#标杆组合
@analysis.command()
@click.pass_context
def fin_indicator(ctx):

    '''
    index_ids = ['120000009','120000014','120000016', '120000042', '120000043']
    data = {}
    for _id in index_ids:
        data[_id] = base_ra_index_nav.load_series(_id)
    df = pd.DataFrame(data)

    benchmark_df = df

    benchmark_df = benchmark_df[benchmark_df.index >= '2013-01-01']
    benchmark_df = benchmark_df / benchmark_df.iloc[0]

    benchmark_df_inc = benchmark_df.sum(axis = 1) / len(benchmark_df.columns)
    #print(benchmark_df_inc.tail())
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
    df = df[df.index >= '2012-12-31']
    online_df = df / df.iloc[0]
    #df.to_csv('./online_nav.csv')

    #print(online_df.tail())
  
    year = '2014'
    begin_date = '2013-12-31'
    end_date = '2014-12-31'

 
    fund_year = pd.read_csv('fund_year.csv', index_col = ['code'])
    funds = fund_year[year].dropna()
    codes = [code[0:6] for code in funds.index]
    #print(len(codes))
    navs = base_ra_fund_nav.load_daily(begin_date, end_date, codes = codes)
    navs = navs / navs.iloc[0]

    online_df = online_df.loc[navs.index]
    online_df = online_df['risk_0']

    #online_df = online_df.mean(axis = 1) 
    navs['online'] = online_df
    navs = navs / navs.iloc[0]

    std = navs.pct_change().fillna(0.0).std()
    rs = navs.iloc[-1]
    maxdrawdown = (1 - navs / navs.cummax()).max()
    #print(maxdrawdown)
    sharp = rs  / len(navs) / std

    std = std.sort_values(ascending = True)
    rs = rs.sort_values(ascending = False)
    maxdrawdown = maxdrawdown.sort_values(ascending = True)
    sharp = sharp.sort_values(ascending = False)

    print(year)
    print('sharp', sharp.index.tolist().index('online') / len(codes))
    print('rs', rs.index.tolist().index('online') / len(codes))
    print('maxdrawdown', maxdrawdown.index.tolist().index('online') / len(codes))
    print('std', std.index.tolist().index('online') / len(codes))
    #print(navs)
    #print(fund_year.tail()) 


#组合配置比例和净值
@analysis.command()
@click.pass_context
def allocate_nav(ctx):

    index_ids = ['120000001', '120000002', '120000013', '120000014', '120000015']
    data = {}
    for _id in index_ids:
        data[_id] = base_ra_index_nav.load_series(_id)
    df = pd.DataFrame(data)

    pos = asset_mz_highlow_pos.load('HL.000070')

    df = pd.concat([df, pos], axis = 1, join_axes = [pos.index])

    df.to_csv('allocate_nav.csv')



#投资理财魔方和投资指数的收益对比
@analysis.command()
@click.pass_context
def allocate_comp(ctx):


    index_ids = ['120000001', '120000018']
    data = {}
    for _id in index_ids:
        data[_id] = base_ra_index_nav.load_series(_id)
    index_df = pd.DataFrame(data)

    conn  = MySQLdb.connect(**config.db_asset)
    conn.autocommit(True)

    dfs = []
    for i in range(0, 10):
        sql = 'select on_date as date, on_nav as nav from on_online_nav where on_online_id = 80000%d and on_type = 8' % i
        df = pd.read_sql(sql, conn, index_col = ['date'], parse_dates = ['date'])
        df.columns = ['risk_' + str(i)]
        dfs.append(df)

    online_df = pd.concat(dfs, axis = 1)[['risk_7', 'risk_0']]


    df = pd.concat([online_df, index_df], axis = 1, join_axes = [online_df.index]).fillna(method = 'pad')
    tmp_df = df.resample('M').last()
    tmp_df = tmp_df.iloc[-25:-1]
    print(tmp_df.index)
    df = df[df.index >= tmp_df.index[0]]
    df = df / df.iloc[0]
    #print(df)
    old_df = df
    tmp_df.loc[df.index[-3]] = df.iloc[-3]
    #print(tmp_df.index)

    df = df[df.index >= tmp_df.index[0]]
    df = df[df.index <= tmp_df.index[-1]]
    df_inc = df.pct_change().fillna(0.0)

    data = {}
    for col in df_inc.columns:
        total_asset = 0.0
        user_asset = 0.0
        navs = {}
        for date in df_inc.index:
            if date in tmp_df.index:
                total_asset = total_asset + 1
                user_asset = user_asset + 1
            total_asset = total_asset * (df_inc.loc[date, col] + 1)
            navs[date] = 1.0 * total_asset / user_asset
        data[col] = pd.Series(navs) 
    new_df = pd.DataFrame(data)

    df = pd.concat([old_df, new_df], axis = 1, join_axes = [old_df.index]).fillna(method='pad')
    df.to_csv('allocate_comp.csv')
    print(df.tail())

    #print(tmp_df / tmp_df.iloc[0] - 1.0)
    #tmp_df = tmp_df.pct_change().fillna(0.0)

    #for col in tmp_df.columns:
    #    rs = tmp_df[col].ravel()
    #    total_asset = 0.0
    #    user_asset = 0.0
    #    for r in rs:
    #        total_asset = (total_asset + 1) * (r + 1)
    #        user_asset = user_asset + 1

    #    print(col, total_asset / user_asset - 1.0)

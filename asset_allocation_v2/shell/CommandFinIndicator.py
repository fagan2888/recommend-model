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
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from tabulate import tabulate
from db import database, base_exchange_rate_index, base_ra_index, asset_ra_pool_fund, base_ra_fund, asset_ra_pool, asset_on_online_nav, asset_ra_portfolio_nav, asset_on_online_fund, asset_mz_markowitz_nav, base_ra_index_nav, asset_ra_composite_asset_nav, base_exchange_rate_index_nav, base_ra_fund_nav, asset_mz_highlow_pos, asset_ra_pool_nav, asset_ra_portfolio_pos, asset_allocate
from util import xdict
from trade_date import ATradeDate
from asset import Asset
from monetary_fund_filter import MonetaryFundFilter

import pymysql
import pandas as pd
from ipdb import set_trace
from dateutil.parser import parse
from datetime import timedelta

import traceback, code

logger = logging.getLogger(__name__)

@click.group(invoke_without_command=True)
@click.pass_context
def indicator(ctx):
    '''
        analysis something
    '''
    pass



#组合和标杆组合,收益率对比
@indicator.command()
@click.pass_context
def allocate_benchmark_comp(ctx):

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

    conn  = MySQLdb.connect(**config.db_asset)
    conn.autocommit(True)

    dfs = []
    for i in range(0, 10):
        sql = 'select on_date as date, on_nav as nav from on_online_nav where on_online_id = 80000%d and on_type = 8' % i
        df = pd.read_sql(sql, conn, index_col = ['date'], parse_dates = ['date'])
        df.columns = ['risk_' + str(i)]
        dfs.append(df)

    df = pd.concat(dfs, axis = 1)

    conn.close()

    df = pd.concat([df, benchmark_df], axis = 1, join_axes = [df.index])
    df = df.fillna(method='pad')
    df = df.rename(columns = {'risk_0':'风险10','risk_1':'风险1','risk_2':'风险2','risk_3':'风险3','risk_4':'风险4','risk_5':'风险5',
                            'risk_6':'风险6','risk_7':'风险7','risk_8':'风险8','risk_9':'风险9',
                            '20201':'风险2比较基准','20202':'风险3比较基准', '20203':'风险4比较基准', '20204':'风险5比较基准', 
                            '20205':'风险6比较基准', '20206':'风险7比较基准', '20207':'风险8比较基准', '20208':'风险9比较基准',
                            '120000016':'风险10比较基准','120000010':'风险1比较基准'})
    cols = ['风险1', '风险2', '风险3', '风险4', '风险5', '风险6', '风险7', '风险8', '风险9', '风险10','风险1比较基准','风险2比较基准', '风险3比较基准', '风险4比较基准', '风险5比较基准', '风险6比较基准', '风险7比较基准', '风险8比较基准', '风险9比较基准', '风险10比较基准']
    df = df[cols]

    df.to_csv('online_benchmark.csv', encoding='gbk')

    result_df = pd.DataFrame(columns = df.columns)
    last_day = df.index[-1]
    print(last_day)
    result_df.loc[df.index[-1].strftime('%Y-%m-%d') + ' 当日'] = df.pct_change().iloc[-1]
    result_df.loc[df.index[-1].strftime('%Y-%m-%d') + ' 过去一周'] = df.loc[last_day] / df.loc[last_day - timedelta(weeks = 1)] - 1
    result_df.loc[df.index[-1].strftime('%Y-%m-%d') + ' 过去一月'] = df.loc[last_day] / df.loc[last_day - timedelta(days = 31)] - 1
    result_df.loc[df.index[-1].strftime('%Y-%m-%d') + ' 过去三个月'] = df.loc[last_day] / df.loc[last_day - timedelta(days = 91)] - 1
    result_df.loc[df.index[-1].strftime('%Y-%m-%d') + ' 过去六个月'] = df.loc[last_day] / df.loc[last_day - timedelta(days = 182)] - 1
    result_df.loc[df.index[-1].strftime('%Y-%m-%d') + ' 过去一年'] = df.loc[last_day] / df.loc[last_day - timedelta(days = 365)] - 1
    #result_df.to_csv('智能组合收益与比较基准收益比较.csv', encoding='gbk')

    df = result_df
    print(df)
    
    db_name_in = 'asset_allocation'
    table_name_risk = 'smart_risk_info_pdate (sir_risk1, sir_risk2, sir_risk3, sir_risk4, sir_risk5, sir_risk6, sir_risk7, sir_risk8, sir_risk9, sir_risk10, sir_risk1_standard, sir_risk2_standard, sir_risk3_standard, sir_risk4_standard, sir_risk5_standard, sir_risk6_standard, sir_risk7_standard, sir_risk8_standard, sir_risk9_standard, sir_risk10_standard, sir_date, sir_statistics_type)'

    conn, cursor = get_connection(db_name_in)
    df = df.reset_index()
    df = df.iloc[:6,:]

    df['date'] = pd.Series(None)
    df['statistics_type'] = pd.Series(None)


    df = df.apply(split_ct, axis=1)
    df.pop('index')

    df.iloc[:,:20] = round_col(df.iloc[:,:20], 6)

    length = len(df.columns)
    counts = 0
    for each in df.values:
        sql = 'INSERT INTO ' + table_name_risk + ' VALUES ('

        for i, n in enumerate(each):
            if i==length-1:
                if str(n) == 'nan':
                    sql = sql + '"' + '");'
                else:
                    sql = sql + '"' + str(n) + '");'
            else:
                if str(n) == 'nan':
                    sql = sql + '"' + '", '
                else:
                    sql = sql + '"' + str(n) + '", '
        #print(sql)
        cursor.execute(sql)
        conn.commit()
        counts += 1
        #print('成功添加了'+str(counts)+'条数据')

    


#基金池收益率排名
@indicator.command()
@click.pass_context
def pool_rank(ctx):


    engine = database.connection('base')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql_t = 'select ra_fund_id, ra_date, ra_nav_adjusted from ra_fund_nav where ra_date > "2018-01-01"'
    ra_fund_nav = pd.read_sql(sql=sql_t, con=session.bind)
    ra_fund_nav.ra_fund_id = ra_fund_nav.ra_fund_id.astype(str)
    session.commit()
    session.close()

    engine = database.connection('base')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql_t = 'select * from yinhe_type'
    yinhe_type = pd.read_sql(sql=sql_t, con=session.bind)
    yinhe_type.yt_fund_id = yinhe_type.yt_fund_id.astype(str)
    session.commit()
    session.close()

    engine = database.connection('asset')
    Session = sessionmaker(bind=engine)
    session = Session()
    sql_t = 'select ra_pool, ra_date, ra_nav from ra_pool_nav where ra_date > "2018-01-01"'
    ra_pool_nav = pd.read_sql(sql=sql_t, con=session.bind)
    ra_pool_nav.ra_pool = ra_pool_nav.ra_pool.astype(str)
    ra_pool_nav.ra_date = pd.to_datetime(ra_pool_nav.ra_date)
    ra_pool_nav.set_index('ra_date', inplace=True)
    session.commit()
    session.close()

    columns_debt = ['中短期标准债券型基金', '中短期标准债券型基金(B/C类)', '长期标准债券型基金(A类)', '长期标准债券型基金(B/C类)', '指数债券型基金(A类)', '指数债券型基金(B/C类)']
    loc_t_debt = yinhe_type.yt_l3_name.isin(columns_debt)

    columns_stock = ['指数股票型基金', '标准股票型基金', '行业股票型基金', '股票型分级子基金', '偏股型基金']
    loc_t_stock = yinhe_type.yt_l2_name.isin(columns_stock)
    #
    columns_cash = ['货币市场基金']
    loc_t_cash = yinhe_type.yt_l1_name.isin(columns_cash)

    rank_record = pd.DataFrame()
    for i_num in range(3):
        if i_num == 0:
            loc_t = loc_t_debt
            ra_pool = ['11210100', '11210200']
        elif i_num == 1:
            loc_t = loc_t_stock
            ra_pool = ['11110116', '11110114', '11110112', '11110110', '11110108', '11110106', '11110100', '11110200']
        else:
            loc_t = loc_t_cash
            ra_pool = ['11310102']

        yinhe_type_t = yinhe_type.loc[loc_t].copy()
        codes_list = list(yinhe_type_t.yt_fund_id.values)

        ra_fund_nav_t = ra_fund_nav.loc[ra_fund_nav.ra_fund_id.isin(codes_list)].copy()
        ra_fund_nav_t.ra_date = pd.to_datetime(ra_fund_nav_t.ra_date)
        ra_fund_nav_t.set_index('ra_date', inplace=True)
        ra_pool_nav_t = ra_pool_nav.loc[ra_pool_nav.ra_pool.isin(ra_pool)].copy()
        dates = pd.date_range(datetime.now().date() - timedelta(400) , datetime.now().date())
        dates = dates[0:-1]
        print(dates)
        date_list_begin = [dates[-2].strftime('%Y-%m-%d'), dates[-7].strftime('%Y-%m-%d'), dates[-31].strftime('%Y-%m-%d'), dates[-91].strftime('%Y-%m-%d'), dates[-182].strftime('%Y-%m-%d') , dates[-365].strftime('%Y-%m-%d')]
        date_end = dates[-1].strftime('%Y-%m-%d')
        ra_pool_nav_t = ra_pool_nav_t.reset_index()
        ra_pool_nav_t = ra_pool_nav_t.set_index(['ra_date', 'ra_pool'])
        ra_pool_nav_t = ra_pool_nav_t.unstack().fillna(method='pad')
        ra_pool_nav_t = ra_pool_nav_t.reindex(dates).fillna(method='pad')
        ra_pool_nav_t = ra_pool_nav_t.stack()

        # 截至日期排名
        for j_num, j_date in enumerate(date_list_begin):
            ra_fund_nav_t0 = ra_fund_nav_t.loc[j_date].copy()
            ra_fund_nav_t1 = ra_fund_nav_t.loc[date_end].copy()
            ra_fund_nav_t01 = pd.merge(ra_fund_nav_t0, ra_fund_nav_t1, how='inner', on='ra_fund_id', sort=False, suffixes=('_0', '_1'))
            ra_fund_nav_t01['NEXT_RETURN'] = (ra_fund_nav_t01['ra_nav_adjusted_1'] / ra_fund_nav_t01['ra_nav_adjusted_0']) - 1
            #
            ra_pool_nav_t0 = ra_pool_nav_t.loc[j_date:j_date, :].copy()
            ra_pool_nav_t1 = ra_pool_nav_t.loc[date_end:date_end, :].copy()
            ra_pool_nav_t01 = pd.merge(ra_pool_nav_t0, ra_pool_nav_t1, how='inner', on='ra_pool', sort=False, suffixes=('_0', '_1'))
            ra_pool_nav_t01['NEXT_RETURN'] = (ra_pool_nav_t01['ra_nav_1'] / ra_pool_nav_t01['ra_nav_0']) - 1
            #
            ra_fund_nav_add = ra_fund_nav_t01.iloc[0:1, :].copy()
            ra_fund_nav_add['ra_fund_id'] = 'select'
            ra_fund_nav_add['NEXT_RETURN'] = ra_pool_nav_t01['NEXT_RETURN'].mean()
            #
            ra_fund_nav_t01 = ra_fund_nav_t01.append(ra_fund_nav_add, ignore_index=True)
            ra_fund_nav_t01['rank'] = ra_fund_nav_t01['NEXT_RETURN'].rank(method='first', ascending=False)
            ra_fund_nav_t01.set_index('ra_fund_id', inplace=True)

            rank_record.at[j_date, str(i_num)+'_rank'] = ra_fund_nav_t01.at['select', 'rank']
            rank_record.at[j_date, str(i_num)+'_samples'] = ra_fund_nav_t01.shape[0] - 1
            rank_record.at[j_date, str(i_num)+'_return'] = ra_pool_nav_t01['NEXT_RETURN'].mean()

    df = pd.DataFrame(index = [date_end + ' 当日', date_end + ' 过去一周',date_end + ' 过去一月',date_end + ' 过去三个月',date_end + ' 过去六个月',date_end + ' 过去一年'])
    df['债券收益均值'] = rank_record['0_return'].ravel()
    df['债券收益排名'] = (rank_record['0_rank'].astype(int).astype(str)+ '/' + rank_record['0_samples'].astype(int).astype(str)).ravel()
    df['股票收益均值'] = rank_record['1_return'].ravel()
    df['股票收益排名'] = (rank_record['1_rank'].astype(int).astype(str)+ '/' + rank_record['1_samples'].astype(int).astype(str)).ravel()
    df['货币收益均值'] = rank_record['2_return'].ravel()
    df['货币收益排名'] = (rank_record['2_rank'].astype(int).astype(str)+ '/' + rank_record['2_samples'].astype(int).astype(str)).ravel()
    print(df)

    db_name_in = 'asset_allocation'
    table_name_pool = 'fund_pool_info_pdate (fp_date, fp_statistics_type, fp_average_return_bond, fp_rank_bond, fp_average_return_stock, fp_rank_stock, fp_average_return_currency, fp_rank_currency)'

    conn, cursor = get_connection(db_name_in)
    df = df.reset_index()
    df['date'] = pd.Series(None)
    df['statistics_type'] = pd.Series(None)


    df = df.apply(split_ct, axis=1)
    df.pop('index')
    order = list(df.columns)[-2:] + list(df.columns)[:6]
    df = df[order]

    df.iloc[:,[2,4,6]] = round_col(df.iloc[:,[2,4,6]], 6)



    length = len(df.columns)
    counts = 0
    for each in df.values:
        sql = 'INSERT INTO ' + table_name_pool + ' VALUES ('

        for i, n in enumerate(each):
            if i==length-1:
                if str(n) == 'nan':
                    sql = sql + '"' + '");'
                else:
                    sql = sql + '"' + str(n) + '");'
            else:
                if str(n) == 'nan':
                    sql = sql + '"' + '", '
                else:
                    sql = sql + '"' + str(n) + '", '
        print(sql)
        cursor.execute(sql)
        conn.commit()
        counts += 1
        print('成功添加了'+str(counts)+'条数据')


@indicator.command()
@click.pass_context
@click.option('--start-date', 'st_date', default=None, help='portfolio pos startdate')
@click.option('--end-date', 'ed_date', default=None, help='portfolio pos endate')
def macroview_retcompare(ctx,st_date,ed_date):
    startDate = parse(st_date)
    endDate  = parse(ed_date)
    assetsID = {'120000001':'沪深300','120000002':'中证500','120000013':'标普500','120000015':'恒生指数','120000014':'沪金指数','120000010':'中证国债','120000011':'中证信用债'}
    #assetsRet = {'120000001':0.2743,'120000002':0.3576,'120000013':0.0953,'120000015':0.0997,'120000014':-0.0029,'120000010':0.0009,'120000011':0.0032}
    assets = dict([(asset_id, base_ra_index_nav.load_series(asset_id)) for asset_id in list(assetsID.keys())])
    df_assets = pd.DataFrame(assets).loc[startDate:endDate,].fillna(method='pad')
    assetsRet = dict([(asset_id,df_assets.loc[endDate,asset_id] / df_assets.loc[startDate,asset_id] - 1.0) for asset_id in list(assetsID.keys())])
     #     #计算给定指数和日期的持有期收益
    df_assets = pd.DataFrame(assets)
    df_assets = df_assets.rolling(365).apply(lambda x : x[-1] / x[0] - 1,raw=True)
    df = df_assets.reset_index()
    tradedaysDiff = df[df.date==endDate].index.tolist()[0] - df[df.date==startDate].index.tolist()[0]
    annualMulty = 365/tradedaysDiff
    MacroCompare = []
    for key,value in assetsRet.items():
        values = value*annualMulty
        ser = df_assets[key].dropna()
        ser = ser.sort_values()
        if len(ser) == 0:
            continue
        MacroCompare.append((assetsID[key], 1 - len(ser[ser < values]) / len(ser)))
    df_MacroCompare = pd.DataFrame(MacroCompare,columns = ['ra_index','分位数'])
    #print(df_MacroCompare)
    ser = pd.Series(assetsRet)
    ret_df = pd.DataFrame(columns = ser.index)
    ret_df.loc[endDate] = ser
    ret_df = ret_df.rename(columns = assetsID)
    print(ret_df)
    df_MacroCompare['date'] = endDate
    df_MacroCompare = df_MacroCompare.set_index(['date','ra_index']).unstack()
    df_MacroCompare.columns = df_MacroCompare.columns.get_level_values(1)
    df_MacroCompare = df_MacroCompare[ret_df.columns]
    df_MacroCompare.columns = df_MacroCompare.columns + ' 分位数'
    df = pd.concat([ret_df, df_MacroCompare] ,axis = 1)
    print(df)

    # 开始注入数据
    db_name_in = 'asset_allocation'
    conn, cursor = get_connection(db_name_in)
    
    table_name_macro = 'opinion_info_pdate (`oi_date`, `oi_return_CSI300`, `oi_return_IC500`, `oi_return_S&P500`, `oi_return_HSI`, `oi_return_SQau`, `oi_return_CSI_TB`, `oi_return_CSI_DB`)'
    df_in = df.iloc[:,:7]
    sql = "INSERT INTO " + table_name_macro + " VALUES " + value_sql(df_in)

    cursor.execute(sql)
    conn.commit()


@indicator.command()
@click.pass_context
@click.option('--database-out', 'db_name_out', default=None, help='name of database out')
@click.option('--table-out', 'table_name_out', default=None, help='name of table out')
@click.option('--database-in', 'db_name_in', default=None, help='name of database in')
@click.option('--table-out', 'table_name_in', default=None, help='name of table in')
@click.option('--ra-portfolio-id', 'ra_portfolio_id', default=None, help='ra_portfolio_id')
@click.option('--ra-type', 'ra_type', default=None, help='rate of fee ==8 while no == 9')
def insertData_moderate(ctx, db_name_out, table_name_out, db_name_in, table_name_in, ra_portfolio_id, ra_type):

    db_name_in = 'asset_allocation'
    db_name_out = 'asset_allocation'


    table_name_out = 'ra_portfolio_nav'
    table_name_in = 'moderate_info_pdate'

    ra_portfolio_ids = ['PO.CB0010', 'PO.CB0020','PO.CB0030','PO.CB0040']
    ra_type = 8


    incs = []
    last_date = None
    for ra_portfolio_id in ra_portfolio_ids:
        conn_out, cursor_out = get_connection(db_name_out)
        # ra_type == 8: 有费率                   ra_type == 9: 无费率
        sql_out = "SELECT ra_date, ra_nav FROM " + table_name_out + " WHERE ra_portfolio_id='" + ra_portfolio_id + "' AND ra_type='" + str(ra_type) + "';"
        df = pd.read_sql(sql=sql_out, con=conn_out, parse_dates=['ra_date'])
        df = df.set_index('ra_date')
        #TimeSeries = df['ra_nav']
        #df_res = backroll(TimeSeries)
        #df_res = df_res.sort_index(ascending=True)
        #df_res['当日点评'] = pd.Series(None) 
        #df_original = pd.read_excel('副本每日数据报告_20190612(1).xlsx', sheetname='稳健组合', index_col=0, parse_dates=['稳健组合收益'])
        inc = df.ra_nav.iloc[-1] / df.ra_nav.iloc[0] - 1 
        #df_total = pd.concat([df_original, df_res])
        #df_total = df_total.sort_index(ascending=True)
        incs.append(inc)
        last_date = df.index[-1]

    conn_in, cursor_in = get_connection(db_name_in)

    # 截取表的所有列名，拼接成字符串，为sql语句做准备
    df_columns = pd.read_sql(sql="SELECT * FROM " + table_name_in + ";", con=conn_in)
    columns = list(df_columns.columns)[1:-2]
    columns_sql_part = "("
    for item in columns:
        if item != columns[-1]:
            columns_sql_part = columns_sql_part + "`" + item + "`, "
        else:
            columns_sql_part = columns_sql_part +  "`" + item + "`)"

    sql = "INSERT INTO moderate_info_pdate  (mi_date, mi_return_last_month, mi_return_last_three_month, mi_return_last_half_year, \
                mi_return_last_year)  VALUES (%s, %f, %f, %f, %f)" % (last_date.strftime('%Y-%m-%d'), incs[0], incs[1], incs[2], incs[3])
    cursor_in.execute(sql)
    conn_in.commit()
    conn_in.close()

    return


#-----------------predefined functions from Tong Cheng---------------
# 建立数据库连接
def get_connection(db_name):
    if db_name == 'mofang_api':
        conn = pymysql.connect(host=config.db_base['host'], user=config.db_base['user'], passwd=config.db_base['passwd'], db=db_name, charset='gbk')
    elif db_name == 'asset_allocation':
        conn = pymysql.connect(host=config.db_asset['host'], user=config.db_asset['user'], passwd=config.db_asset['passwd'], db=db_name, charset='gbk')
    cursor1 = conn.cursor()
    return conn, cursor1

# 拆分日期和中文描述混合的标号
def split_ct(x):
    L = x.iloc[0].split()
    x['date'] = parse(L[0])
    x['statistics_type'] = L[1]
    return x

# 截取前num位数据
def round_col(df, num):
    for columns in df.keys():
        df[columns] = df[columns].apply(lambda x: round(x,num))
    return df

# 生成sql语句中VALUES后面的部分
def value_sql(df):
    str_value_sql = ""
    for key in list(df.index):
        L = df.loc[key].values # array
        temp = "('" + str(key) + "', "
        for item in L:
            if str(item) == "nan":
                temp = temp + "'', "
            else:
                temp = temp + "'" + str(item) + "', "
        temp = temp[:-2] + ")"
        str_value_sql = str_value_sql + temp + ", "
    return str_value_sql[:-2]
#--------------------------------------------------------------------

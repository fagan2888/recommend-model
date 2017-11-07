#coding=utf8

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import calendar
from sqlalchemy import *

from db import database, Nav
from util.xdebug import dd


def get_date_df(df, start_date, end_date):
    _df = df[df.index <= end_date]
    _df = _df[_df.index >= start_date]
    return _df



def last_friday():
    date   = datetime.date.today()
    oneday = datetime.timedelta(days=1)

    while date.weekday() != calendar.FRIDAY:
       date -= oneday

    date = date.strftime('%Y-%m-%d')
    return date

def portfolio_nav(df_inc, df_position, result_col='portfolio') :
    '''calc nav for portfolio
    '''
    if df_position.empty:
        return pd.DataFrame(columns=(['portfolio'] + list(df_inc.columns)))
    #
    # 从第一次调仓开始算起.
    #
    # [XXX] 调仓日的处理是先计算收益,然后再调仓, 因为在实际中, 调仓的
    # 动作也是在收盘确认之后发生的
    #
    start_date = df_position.index.min()
    
    if start_date not in df_inc.index:
        df_inc.loc[start_date] = 0
        df_inc.sort_index(inplace=True)

    df = df_inc[start_date:].copy()

    df_position['cash'] = 1 - df_position.sum(axis=1)
    df['cash'] = 0.0

    assets_s = pd.Series(np.zeros(len(df.columns)), index=df.columns) # 当前资产
    assets_s[0] = 1

    #
    # 计算每天的各资产持仓情况
    #
    # 注意:
    #    1. 第一天是没有收益率的
    #    2. 对于调仓日, 先计算收益率, 然后再调仓
    #
    df_result = pd.DataFrame(index=df.index, columns=df.columns)
    for day,row in df.iterrows():
        # 如果不是第一天, 首先计算当前资产收益
        if day != start_date:
            assets_s = assets_s * (row + 1)

        # 如果是调仓日, 则调仓
        if day in df_position.index: # 调仓日
            if day == start_date:
                assets_s = 1 * df_position.loc[day]
            else:
                assets_s = assets_s.sum() * df_position.loc[day]

        # 日末各个基金持仓情况
        df_result.loc[day] = assets_s

    #
    # 计算资产组合净值
    #
    df_result.insert(0, result_col, df_result.sum(axis=1))               

    return df_result

def portfolio_nav2(df_pos, end_date=None) :
    '''calc nav for portfolio
    '''
    if df_pos.empty:
        return pd.DataFrame(columns=(['portfolio'] + list(df_inc.columns)))
    #
    # 从第一次调仓开始算起.
    #
    # [XXX] 调仓日的处理是先计算收益,然后再调仓, 因为在实际中, 调仓的
    # 动作也是在收盘确认之后发生的
    #
    if end_date is not None:
        max_date = end_date
    else:
        max_date = (datetime.now() - timedelta(days=1)) # yesterday

    dates = df_pos.index.get_level_values(0).unique()

    min_date = dates.min()
    if max_date < min_date:
        return pd.DataFrame(columns=(['portfolio'] + list(df_inc.columns)))
    
    dates = dates[(dates >= min_date) & (dates <= max_date)]
    
    pairs = zip(dates[0:-1], dates[1:])
    if max_date.strftime("%Y-%m-%d") not in dates:
        pairs.append((dates[-1], max_date))

    sr_nav_portfolio = pd.Series([1], index=[dates[0]]);

    for sdate, edate in pairs:
        #
        # 加载新仓位
        #
        df_ratio = df_pos.loc[sdate].T

        #
        # 加载收益率
        #
        days = pd.date_range(sdate, edate)
        df_nav = Nav.Nav().load(df_ratio.columns, sdate=sdate, edate=edate)
        df_inc = df_nav.pct_change().fillna(0.0)
        #
        # 有些基金在最后一次调仓的时候, 没有净值数据, 比如周一, 周二计算的QDII基金
        # 我们需要不足这些不存在的列
        #
        for column in df_ratio.columns:
            if column not in df_inc:
                df_inc[column] = 0

        if df_inc.empty:
            df_inc.loc[sdate] = 0
        #
        # 不足仓位补充现金
        #
        df_ratio['cash'] = 1 - df_ratio.sum(axis=1)
        df_inc['cash'] = 0.0

        #
        # 计算相对昨日的增长因子
        #
        df_inc += 1
        # 第一天是调仓日, 没有收益, 直接设置各资产比例
        df_inc.iloc[0] = sr_nav_portfolio[-1] * df_ratio.iloc[0]
        #
        # 后面所有日期累积乘即为每天净值
        #
        df_nav_new = df_inc.cumprod()
        #
        # 结果净值
        #
        sr_nav_portfolio = sr_nav_portfolio.append(df_nav_new.sum(axis=1)[1:])

    return sr_nav_portfolio

def load_nav_csv(csv, columns=None, reindex=None):
    if columns and 'date' not in columns:
        columns.insert(0, 'date')
        
    df = pd.read_csv(csv, index_col='date', parse_dates=['date'], usecols=columns)

    if reindex:
        #
        # 重索引的时候需要, 需要首先用两个索引的并集,填充之后再取
        # reindex, 否则中间可能会造成增长率丢失
        # 
        index = df.index.union(reindex)
        df = df.reindex(index, method='pad').reindex(reindex)
        
    return df

def load_inc_csv(csv, columns=None, reindex=None):
    if columns and 'date' not in columns:
        columns.insert(0, 'date')
    #
    # [XXX] 不能直接调用load_nav_csv, 否则会造成第一行的增长率丢失
    #
    df = pd.read_csv(csv, index_col='date', parse_dates=['date'], usecols=columns)

    if reindex is not None:
        #
        # 重索引的时候需要, 需要首先用两个索引的并集,填充之后再计算增长率
        # 
        index = df.index.union(reindex)
        df = df.reindex(index, method='pad')

    # 计算增长率
    dfr = df.pct_change().fillna(0.0)

    if reindex is not None:
        dfr = dfr.reindex(reindex)
        
    return dfr

def pad_sum_to_one(df, by, ratio='ratio'):
    df3 = df[ratio].groupby(by).agg(['sum', 'idxmax'])
    df.ix[df3['idxmax'], ratio] += (1 - df3['sum']).values
    # print df[ratio].groupby(by).sum()
    
    return df

def filter_by_turnover_rate(df, turnover_rate):
    df_result = pd.DataFrame(columns=['risk', 'date', 'category', 'fund', 'ratio'])
    for k0, v0 in df.groupby('risk'):
        df_tmp = filter_by_turnover_rate_per_risk(v0, turnover_rate)
        if not df_tmp.empty:
            df_result = pd.concat([df_result, df_tmp])
            
    return df_result

def filter_by_turnover_rate_per_risk(df, turnover_rate):
    df_result = pd.DataFrame(columns=['risk', 'date', 'category', 'fund', 'ratio'])
    df_last=None
    for k1, v1 in df.groupby(['risk', 'date']):
        if df_last is None:
            df_last = v1[['category', 'fund','ratio']].set_index(['category', 'fund'])
            df_result = pd.concat([df_result, v1])
        else:
            df_current = v1[['category', 'fund', 'ratio']].set_index(['category', 'fund'])
            df_diff = df_current - df_last
            xsum = df_diff['ratio'].abs().sum()
            if df_diff.isnull().values.any() or xsum >= turnover_rate:
                df_result = pd.concat([df_result, v1])
                df_last = df_current
                
    return df_result

def filter_by_turnover(df, turnover):
    result = {}
    sr_last=None
    for k, v in df.iterrows():
        vv = v.fillna(0)
        if sr_last is None:
            result[k] = v
            sr_last = vv
        else:
            xsum = (vv - sr_last).abs().sum()
            if xsum >= turnover:
                result[k] = v
                sr_last = vv
            else:
                #print "filter by turnover:", v.to_frame('ratio')
                pass
    return pd.DataFrame(result).T

def portfolio_import(df):
    pass;

def nav_drawdown(df_nav):
    ''' calc drawdown base on nav
    '''
    return df_nav/df_nav.cummax() - 1

def inc_drawdown(df_inc):
    ''' calc drawdown base on inc
    '''
    df_nav = (df_inc + 1).cumprod()
    return df_nav/df_nav.cummax() - 1

def nav_drawdown_window(df_nav, window, min_periods=1):
    ''' calc drawdown base on nav
    '''
    df_max = df_nav.rolling(window=window, min_periods=min_periods).max()
    return df_nav/df_max - 1

def nav_max_drawdown_window(df_nav, window, min_periods=1):
    ''' calc max draw base on slice window of nav
    '''
    return df_nav.rolling(
        window=window, min_periods=min_periods).apply(
            lambda x:(x/np.maximum.accumulate(x) - 1).min())


def merge_column_for_fund_id_type(df, code, usecols=['globalid', 'ra_type']):
    sr_code = df[code]
    
    db = database.connection('base')
    # 加载基金列表
    t = Table('ra_fund', MetaData(bind=db), autoload=True)
    columns = [
        t.c.globalid,
        t.c.ra_code,
        t.c.ra_type,
        t.c.ra_name,
    ]
    
    s = select(columns, (t.c.ra_code.in_(sr_code)))
    
    df_c2i = pd.read_sql(s, db, index_col = ['ra_code'])

    df_result = df.merge(df_c2i[usecols], left_on=code, right_index=True)

    return df_result

def filter_same_with_last(df, fill_value=0):
    df1 = df.fillna(fill_value)
    df2 = df.shift(1).fillna(fill_value)
    return df[(df1 != df2).any(axis=1)]

def categories_types(as_int=False):
    if as_int:
        return {
            'largecap'        : 11, # 大盘
            'smallcap'        : 12, # 小盘
            'rise'            : 13, # 上涨
            'oscillation'     : 14, # 震荡
            'decline'         : 15, # 下跌
            'growth'          : 16, # 成长
            'value'           : 17, # 价值

            'ratebond'        : 21, # 利率债
            'creditbond'      : 22, # 信用债
            'convertiblebond' : 23, # 可转债

            'money'           : 31, # 货币

            'SP500.SPI'       : 41, # 标普
            'GLNC'            : 42, # 黄金
            'HSCI.HI'         : 43, # 恒生
        }
        
    #
    # 输出配置数据
    #
    return {
        'largecap'        : '11', # 大盘
        'smallcap'        : '12', # 小盘
        'rise'            : '13', # 上涨
        'oscillation'     : '14', # 震荡
        'decline'         : '15', # 下跌
        'growth'          : '16', # 成长
        'value'           : '17', # 价值

        'ratebond'        : '21', # 利率债
        'creditbond'      : '22', # 信用债
        'convertiblebond' : '23', # 可转债

        'money'           : '31', # 货币

        'SP500.SPI'       : '41', # 标普
        'GLNC'            : '42', # 黄金
        'HSCI.HI'         : '43', # 恒生
    }

def categories_name(category, default='unknown'):
    tls = {
        11 : 'largecap'       , # 大盘
        12 : 'smallcap'       , # 小盘
        13 : 'rise'           , # 上涨
        14 : 'oscillation'    , # 震荡
        15 : 'decline'        , # 下跌
        16 : 'growth'         , # 成长
        17 : 'value'          , # 价值
        
        21 : 'ratebond'       , # 利率债
        22 : 'creditbond'     , # 信用债
        23 : 'convertiblebond', # 可转债
        
        31 : 'money'          , # 货币
        
        41 : 'SP500.SPI'      , # 标普
        42 : 'GLNC'           , # 黄金
        43 : 'HSCI.HI'        , # 恒生
    }        

    if category in tls:
        return tls[category]
    else:
        return default

def calc_turnover(df):
    return (df.fillna(0) - df.shift(1).fillna(0)).abs().sum(axis=1)


if __name__ == '__main__':


    df_inc  = pd.read_csv('./testcases/portfolio_nav_inc_df.csv', index_col = 'date', parse_dates = ['date'] )
    df_position = pd.read_csv('./testcases/portfolio_nav_position_df.csv', index_col = 'date', parse_dates = ['date'] )

    print portfolio_nav(df_inc, df_position)



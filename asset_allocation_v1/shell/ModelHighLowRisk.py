#coding=utf8


import os
import string
import pandas as pd
import numpy as np
import FundIndicator
from datetime import datetime
import Portfolio as PF
import DFUtil
import AllocationData

from Const import datapath

def highriskasset(dfr, his_week, interval):

    if 'HSCI.HI' in dfr.columns:
        dfr =  dfr.drop('HSCI.HI', axis = 1)

    if 'oscillation' in dfr.columns:
        dfr =  dfr.drop('oscillation', axis = 1)

    #interval = 26

    result_dates = []
    result_datas = []

    position_dates = []
    position_datas = []

    dates        = dfr.index

    portfolio_vs = [1]
    result_dates.append(dates[his_week - 1])

    fund_values  = {}
    fund_codes   = []


    for i in range(his_week + 1, len(dates)):


        if (i - his_week - 1 ) % interval == 0:

            start_date = dates[i- his_week].strftime('%Y-%m-%d')
            end_date   = dates[i - 1].strftime('%Y-%m-%d')

            allocation_dfr = dfr[dfr.index <= datetime.strptime(end_date, '%Y-%m-%d').date()]
            allocation_dfr = allocation_dfr[allocation_dfr.index >= datetime.strptime(start_date, '%Y-%m-%d').date()]


            uplimit   = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 0.3]
            downlimit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

            #risk, returns, ws, sharpe = PF.markowitz_r(allocation_dfr, None)
            risk, returns, ws, sharpe = PF.markowitz_r_spe(allocation_dfr, [downlimit, uplimit])
            #risk, returns, ws, sharpe = PF.markowitz_r(allocation_dfr, None)
            fund_codes = allocation_dfr.columns


            #for n in range(0, len(fund_codes)):
            #    ws[n] = 1.0 / len(fund_codes)


            last_pv = portfolio_vs[-1]
            fund_values = {}
            for n in range(0, len(fund_codes)):
                fund_values[n] = [last_pv * ws[n]]

            position_dates.append(end_date)
            position_datas.append(ws)


        pv = 0
        d = dates[i]
        for n in range(0, len(fund_codes)):
            vs = fund_values[n]
            code = fund_codes[n]
            fund_last_v = vs[-1]
            fund_last_v = fund_last_v + fund_last_v * dfr.loc[d, code]
            vs.append(fund_last_v)
            pv = pv + vs[-1]
        portfolio_vs.append(pv)
        result_dates.append(d)

        print d , pv


    result_datas  = portfolio_vs
    result_df = pd.DataFrame(result_datas, index=result_dates,
                                 columns=['high_risk_asset'])

    result_df.index.name = 'date'
    result_df.to_csv(datapath('highriskasset.csv'))


    highriskposition_df = pd.DataFrame(position_datas, index=position_dates, columns=dfr.columns)
    highriskposition_df.index.name = 'date'
    highriskposition_df.to_csv(datapath('highriskposition.csv'))

    #allocationdata.high_risk_position_df = highriskposition_df
    #allocationdata.high_risk_asset_df     = result_df

    return result_df



def lowriskasset(dfr, his_week, interval):


    #interval = 26

    result_dates = []
    result_datas = []


    position_dates = []
    position_datas = []


    dates        = dfr.index

    portfolio_vs = [1]
    result_dates.append(dates[his_week - 1])

    fund_values  = {}
    fund_codes   = []

    for i in range(his_week + 1, len(dates)):


        if (i - his_week - 1 ) % interval == 0:

            start_date = dates[i- his_week].strftime('%Y-%m-%d')
            end_date   = dates[i - 1].strftime('%Y-%m-%d')

            allocation_dfr = dfr[dfr.index <= datetime.strptime(end_date, '%Y-%m-%d').date()]
            allocation_dfr = allocation_dfr[allocation_dfr.index >= datetime.strptime(start_date, '%Y-%m-%d').date()]

            risk, returns, ws, sharpe = PF.markowitz_r(allocation_dfr, None)
            fund_codes = allocation_dfr.columns

            #for n in range(0, len(fund_codes)):
            #    ws[n] = 1.0 / len(fund_codes)

            last_pv = portfolio_vs[-1]
            fund_values = {}
            for n in range(0, len(fund_codes)):
                fund_values[n] = [last_pv * ws[n]]

            position_dates.append(end_date)
            position_datas.append(ws)


        pv = 0
        d = dates[i]
        for n in range(0, len(fund_codes)):
            vs = fund_values[n]
            code = fund_codes[n]
            fund_last_v = vs[-1]
            fund_last_v = fund_last_v + fund_last_v * dfr.loc[d, code]
            vs.append(fund_last_v)
            pv = pv + vs[-1]
        portfolio_vs.append(pv)
        result_dates.append(d)

        #print d , pv


    result_datas  = portfolio_vs
    result_df = pd.DataFrame(result_datas, index=result_dates,
                             columns=['low_risk_asset'])

    result_df.index.name = 'date'
    result_df.to_csv(datapath('lowriskasset.csv'))


    lowriskposition_df = pd.DataFrame(position_datas, index=position_dates, columns=dfr.columns)
    lowriskposition_df.index.name = 'date'
    lowriskposition_df.to_csv(datapath('lowriskposition.csv'))


    #allocationdata.low_risk_position_df = lowriskposition_df
    #allocationdata.low_risk_asset_df    = result_df

    return result_df



def highlowallocation(dfr):

    #print dfr

    dates           = dfr.index

    portfolio_vs    = [[1,1,1,1,1,1,1,1,1,1]]
    portfolio_dates = []
    portfolio_dates.append(dates[0])


    print 'risk rank'
    for i in range(1, len(dates)):


        d = dates[i]


        high_risk_asset_r = dfr.loc[d, 'high_risk_asset']
        low_risk_asset_r  = dfr.loc[d, 'low_risk_asset']


        print d,
        risk_vs = []
        for j in range(1, 11):
            high_w  = (j - 1) * 1.0 / 9
            low_w = 1 - high_w
            last_risk_vs= portfolio_vs[-1]
            v      = last_risk_vs[j - 1] * (1 +  high_risk_asset_r * high_w +  low_risk_asset_r * low_w)
            risk_vs.append(v)
            print v,
        print

        portfolio_vs.append(risk_vs)
        portfolio_dates.append(d)


    cols = []
    for i in range(0 , 10):
        cols.append(str(i + 1))
    portfolio_df = pd.DataFrame(portfolio_vs, index = portfolio_dates, columns = cols)
    portfolio_df.index.name = 'date'
    portfolio_df.to_csv(datapath('risk_portfolio.csv'))

    # allocationdata.riskhighlowriskasset = portfolio_df
    return portfolio_df

def asset_alloc_high_risk_per_day(day, lookback, df_inc=None, columns=None):
    '''perform asset allocation of high risk asset for single day
    '''
    if not columns:
        columns = ['date', 'largecap', 'smallcap', 'rise', 'decline', 'growth', 'value', 'SP500.SPI', 'GLNC']
    
    # 加载时间轴数据
    index = DBData.trade_date_lookback_index(end_date=day, lookback=lookback)

    # 加载数据
    if df_inc is None:
        df_nav  = pd.read_csv(datapath('equalriskasset.csv'), index_col='date', parse_dates=['date'], usecols=columns)
        df_inc  = df_nav.pct_change().fillna(0.0)

    #
    # 根据时间轴进行重采样
    #
    df_inc = df_inc.reindex(index, fill_value=0.0)
    
    #
    # 基于马克维茨进行资产配置
    #
    uplimit   = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.3, 0.3]
    downlimit = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    #risk, returns, ws, sharpe = PF.markowitz_r(allocation_dfr, None)
    risk, returns, ws, sharpe = PF.markowitz_r_spe(df_inc, [downlimit, uplimit])

    df_result = pd.concat([
        pd.DataFrame(ws, index=[day], columns=df_inc.columns),
        pd.DataFrame((sharpe, risk, returns), index=[day], columns=['sharpe','risk', 'return'])
    ], axis=1)

    return df_result

def asset_alloc_low_risk_per_day(day, lookback, df_inc=None, columns=None):
    '''perform asset allocation of low risk asset for single day
    '''
    if not columns:
        columns = ['ratebond','creditbond']

    # 加载时间轴数据
    index = DBData.trade_date_lookback_index(end_date=day, lookback=lookback)

    # 加载数据
    if df_inc is None:
        df_nav = pd.read_csv(datapath('labelasset.csv'), index_col='date', parse_dates=['date'], usecols=columns)
        df_inc  = df_nav.pct_change().fillna(0.0)
    #
    # 根据时间轴进行重采样
    #
    df_inc = df_inc.reindex(index, fill_value=0.0)
    
    #
    # 基于马克维茨进行资产配置
    #
    risk, returns, ws, sharpe = PF.markowitz_r(dfr_inc, None)

    df_result = pd.concat([
        pd.DataFrame(ws, index=[day], columns=df_inc.columns),
        pd.DataFrame((sharpe, risk, returns), index=[day], columns=['sharpe','risk', 'return'])
    ], axis=1)

    return df_result

def asset_alloc_high_low(start_date, end_date=None, lookback=26, adjust_period=None):
    '''perform asset allocation with constant-risk + high_low model.
    '''
    # 加载时间轴数据
    index = DBData.trade_date_index(start_date, end_date=end_date)

    # 根据调整间隔抽取调仓点
    if adjust_period:
        adjust_index = index[::adjust_period]
    else:
        adjust_index = index
        
    columns = {
        'low':['ratebond','creditbond'],
        'high':['date', 'largecap', 'smallcap', 'rise', 'decline', 'growth', 'value', 'SP500.SPI', 'GLNC']
    }
        
    #
    # 计算每个调仓点的最新配置
    #
    df_high = pd.DataFrame(index=pd.Index([], name='date'), columns=columns['high'])
    df_low =  pd.DataFrame(index=pd.Index([], name='date'), columns=columns['low'])
    for day in adjust_index:
        # 高风险资产配置
        df_tmp = asset_alloc_high_risk_per_day(day, lookback, columns=columns['high'])
        df_high = pd.concat([df_high, df_tmp], axis=0)
        # 底风险资产配置
        df_tmp = asset_alloc_low_risk_per_day(day, lookback, columns=columns['low'])
        df_low = pd.concat([df_low, df_tmp], axis=0)

    #
    # 保存高低风险配置结果
    #
    df_high.to_csv(datapath('high_position.csv'))
    df_low.to_csv(datapath('low_position.csv'))

    #
    # 计算高风险资产的资产净值
    #
    df_inc = DFUtil.load_inc_csv(datapath('equalriskasset.csv'), columns['high'], index)
    df_nav_high = DFUtil.portfolio_nav(df_inc, df_high[columns['high']])
    df_nav_high.to_csv(datapath('high_nav.csv'))
    
    #
    # 计算低风险资产的资产净值
    #
    df_inc = DFUtil.load_inc_csv(datapath('labelasset.csv'), columns['low'], index)
    df_nav_low = DFUtil.portfolio_nav(df_inc, df_low[columns['low']])
    df_nav_low.to_csv(datapath('low_nav.csv'))
        
    #
    # 混合后风险1-10的资产净值
    #
    # 高低风险净值增长率
    df_inc_high = df_nav_high.pct_change()
    df_inc_low = df_inc_low.pct_change()
    df_inc = pd.DataFrame({'high':df_inc_high['portfolio'], 'low':df_inc_low['portfolio']})
    # 高低风险配置比例
    dt = dict()
    for risk in range(1, 11):
        # 配置比例
        ratio_h  = (risk - 1) * 1.0 / 9
        ratio_l  = 1 - ratio_h
        # 按调仓日期,生成调仓矩阵
        data = [(ratio_h, ratio_l) for x in adjust_index]
        df_position = pd.DataFrame(data, index=adjust_index, columns=['high', 'low'])
        # 单个风险配置结果
        dt['risk'+str(risk)] = DFUtil.portfolio_nav(df_inc, df_position)

    #
    # 保存高低风险配置结果
    # 
    df_nav_result = pd.concat(dt, axis=1)
    df_nav_result.to_csv(datapath('portfolio_nav.csv'))
    #
    # 返回结果
    #
    return df_nav_result

def highlowriskasset(day, lookback):

    # 加载时间轴数据
    index = DBData.trade_date_lookback_index(end_date=day, lookback=lookback)

    #highriskassetdf  = allocationdata.equal_risk_asset_df
    highriskassetdf  = pd.read_csv(datapath('equalriskasset.csv'), index_col = 'date', parse_dates = ['date'])
    highriskassetdfr = highriskassetdf.pct_change().fillna(0.0)
    highriskassetdfr = highriskassetdfr.reindex(index, method='ffill')

    #lowriskassetdfr  = allocationdata.label_asset_df
    lowassetlabel    = ['ratebond','creditbond']
    lowriskassetdf   = pd.read_csv(datapath('labelasset.csv'), index_col = 'date', parse_dates = ['date'] )
    lowriskassetdfr  = lowriskassetdf.pct_change().fillna(0.0)
    lowriskassetdfr  = lowriskassetdfr[lowassetlabel]
    # lowriskassetdfr  = lowriskassetdfr.loc[highriskassetdfr.index]
    lowriskassetdfr  = lowriskassetdfr.reindex(index, method='ffill')

    highdf = highriskasset(highriskassetdfr)
    lowdf  = lowriskasset(lowriskassetdfr)


    df  = pd.concat([highdf, lowdf], axis = 1, join_axes=[index])
    dfr = df.pct_change().fillna(0.0)

    print dfr

    highlowdf = highlowallocation(dfr)

    #print "sharpe : ", FundIndicator.portfolio_sharpe(highlowdf['highlow_risk_asset'].values)
    #print "annual_return : ", FundIndicator.portfolio_return(highlowdf['highlow_risk_asset'].values)
    #print "maxdrawdown : ", FundIndicator.portfolio_maxdrawdown(highlowdf['highlow_risk_asset'].values)


    #highlowdf.to_csv(datapath('highlowrisk_net_value.csv'))
    #print "sharpe : ", fi.portfolio_sharpe(highdf['high_risk_asset'].values)
    #print "annual_return : ", fi.portfolio_return(highdf['high_risk_asset'].values)
    #print "maxdrawdown : ", fi.portfolio_maxdrawdown(highdf['high_risk_asset'].values)


    #print lowriskassetdfr



if __name__ == '__main__':


    highriskassetdf  = pd.read_csv(datapath('equalriskasset.csv'), index_col = 'date', parse_dates = ['date'] )
    highriskassetdfr = highriskassetdf.pct_change().fillna(0.0)


    lowassetlabel    = ['ratebond','creditbond']
    lowriskassetdfr  = pd.read_csv(datapath('labelasset.csv'), index_col = 'date', parse_dates = ['date'] )
    lowriskassetdfr  = lowriskassetdfr[lowassetlabel]
    lowriskassetdfr  = lowriskassetdfr.loc[highriskassetdfr.index]


    his_week = 26
    interval = 26


    highdf = highriskasset(highriskassetdfr, his_week, interval)
    lowdf  = lowriskasset(lowriskassetdfr, his_week, interval)


    df  = pd.concat([highdf, lowdf], axis = 1, join_axes=[highdf.index])
    dfr = df.pct_change().fillna(0.0)


    highlowdf = highlowallocation(dfr, his_week, interval)


    print "sharpe : ", fi.portfolio_sharpe(highlowdf['highlow_risk_asset'].values)
    print "annual_return : ", fi.portfolio_return(highlowdf['highlow_risk_asset'].values)
    print "maxdrawdown : ", fi.portfolio_maxdrawdown(highlowdf['highlow_risk_asset'].values)


    highlowdf.to_csv(datapath('highlowrisk_net_value.csv'))
    #print "sharpe : ", fi.portfolio_sharpe(highdf['high_risk_asset'].values)
    #print "annual_return : ", fi.portfolio_return(highdf['high_risk_asset'].values)
    #print "maxdrawdown : ", fi.portfolio_maxdrawdown(highdf['high_risk_asset'].values)


    #print lowriskassetdfr

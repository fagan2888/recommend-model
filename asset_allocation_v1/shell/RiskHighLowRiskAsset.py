#coding=utf8


import string
import pandas as pd
import numpy as np
import FundIndicator
from datetime import datetime
import Portfolio as PF


def highriskasset(dfr, his_week, interval):


    #interval = 26

    result_dates = []
    result_datas = []

    position_dates = []
    position_datas = []

    dates        = dfr.index

    portfolio_vs = [1]
    result_dates.append(dates[his_week])

    fund_values  = {}
    fund_codes   = []


    for i in range(his_week, len(dates)):


        if (i - his_week) % interval == 0:

            start_date = dates[i- his_week].strftime('%Y-%m-%d')
            end_date   = dates[i - 1].strftime('%Y-%m-%d')

            allocation_dfr = dfr[dfr.index <= end_date]
            allocation_dfr = allocation_dfr[allocation_dfr.index >= start_date]


            uplimit   = [1.0, 1.0, 1.0, 1.0]
            downlimit = [0.0, 0.0, 0.0, 0.0]

            #risk, returns, ws, sharpe = PF.markowitz_r(allocation_dfr, None)
            risk, returns, ws, sharpe = PF.markowitz_r(allocation_dfr, [downlimit, uplimit])
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

            #print end_date , ws
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
                                 columns=['high_risk_asset'])

    result_df.index.name = 'date'
    result_df.to_csv('./tmp/highriskasset.csv')


    highriskposition_df = pd.DataFrame(position_datas, index=position_dates, columns=dfr.columns)
    highriskposition_df.index.name = 'date'
    highriskposition_df.to_csv('./tmp/highriskposition.csv')


    #print "maxdrawdown : ", FundIndicator.portfolio_maxdrawdown(result_df['high_risk_asset'].values)


    return result_df



def lowriskasset(dfr, his_week, interval):


    #interval = 26

    result_dates = []
    result_datas = []


    position_dates = []
    position_datas = []


    dates        = dfr.index

    portfolio_vs = [1]
    result_dates.append(dates[his_week])

    fund_values  = {}
    fund_codes   = []

    for i in range(his_week, len(dates)):


        if (i - his_week) % interval == 0:

            start_date = dates[i- his_week].strftime('%Y-%m-%d')
            end_date   = dates[i - 1].strftime('%Y-%m-%d')

            allocation_dfr = dfr[dfr.index <= end_date]
            allocation_dfr = allocation_dfr[allocation_dfr.index >= start_date]

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
    result_df.to_csv('./tmp/lowriskasset.csv')


    lowriskposition_df = pd.DataFrame(position_datas, index=position_dates, columns=dfr.columns)
    lowriskposition_df.index.name = 'date'
    lowriskposition_df.to_csv('./tmp/lowriskposition.csv')


    #print "maxdrawdown : ", FundIndicator.portfolio_maxdrawdown(result_df['low_risk_asset'].values)
    return result_df


def highlowriskasset():


    df       = pd.read_csv('./data/fund.csv', index_col = 'date', parse_dates = 'date' ).fillna(method = 'pad')
    df       = df.resample('W-FRI').last()
    dfr      = df.pct_change().fillna(0.0)


    equaldf       = pd.read_csv('./tmp/equalriskasset.csv', index_col = 'date', parse_dates = 'date' )
    #equaldf       = pd.read_csv('./tmp/equalriskassetday.csv', index_col = 'date', parse_dates = 'date')
    equaldfr      = equaldf.pct_change().fillna(0.0)


    equaldfr      = equaldfr[equaldfr.columns[0:1]]
    dfr           = dfr[dfr.columns[1:4]]

    #print equaldfr
    #print dfr

    dfr = pd.concat([equaldfr, dfr], axis = 1, join_axes=[equaldfr.index])

    dfr      = df.pct_change().fillna(0.0)
    #print dfr

    #print dfr.columns

    cols = df.columns
    highdfr = dfr[cols[0:2]]
    lowdfr  = dfr[cols[2:4]]

    his_week = 13
    interval = 13

    #print highdfr.columns
    #print lowdfr.columns

    highdf = highriskasset(highdfr, his_week, interval)
    highdf = highdf[highdf.index >= datetime.strptime('2007-07-22', '%Y-%m-%d')]
    #print highdf
    print "maxdrawdown : ", FundIndicator.portfolio_maxdrawdown(highdf['high_risk_asset'].values)
    print "sharpe : ", FundIndicator.portfolio_sharpe(highdf['high_risk_asset'].values)
    print "return : ", FundIndicator.portfolio_return(highdf['high_risk_asset'].values)
    print "risk : ", FundIndicator.portfolio_risk(highdf['high_risk_asset'].values)

    lowdf  = lowriskasset(lowdfr, his_week, interval)
    lowdf  = lowdf[lowdf.index >= datetime.strptime('2007-07-22', '%Y-%m-%d')]
    #print lowdf
    print "maxdrawdown : ", FundIndicator.portfolio_maxdrawdown(lowdf['low_risk_asset'].values)
    print "sharpe : ", FundIndicator.portfolio_sharpe(lowdf['low_risk_asset'].values)
    print "return : ", FundIndicator.portfolio_return(lowdf['low_risk_asset'].values)
    print "risk : ", FundIndicator.portfolio_risk(lowdf['low_risk_asset'].values)

    #highdf.to_csv('highdf.csv')
    #lowdf.to_csv('lowdf.csv')

    #highdfr = dfr[cols[0:4]][-13:]
    #lowdfr  = dfr[cols[4:7]][-13:]

    #print highdfr
    #print lowdfr

    #uplimit   = [1.0, 1.0, 1.0, 1.0]
    #downlimit = [0.0, 0.0, 0.0, 0.0]

    #risk, returns, ws, sharpe = PF.markowitz_r(highdfr, [downlimit, uplimit])

    #print ws



if __name__ == '__main__':

    highlowriskasset()

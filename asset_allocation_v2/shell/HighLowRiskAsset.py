#coding=utf8



import os
import string
import pandas as pd
import numpy as np
import FundIndicator
from datetime import datetime
import Portfolio as PF
import AllocationData

from Const import datapath


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

        print(d , pv)


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
    result_dates.append(dates[his_week])

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


    # allocationdata.low_risk_position_df = lowriskposition_df
    # allocationdata.low_risk_asset_df    = result_df

    return result_df



def highlowallocation(dfr, his_week, interval):

    result_dates = []
    result_datas = []

    position_dates = []
    position_datas = []

    dates        = dfr.index

    portfolio_vs = [1]
    result_dates.append(dates[his_week])


    risk_drawdown = []
    risk_position = 1.0
    risk_index    = 0

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

            #ws[0] = ws[0] * risk_position
            #ws[1] = 1.0 - ws[0]

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
            fund_last_v = fund_last_v + fund_last_v * dfr.loc[d, code] * risk_position + fund_last_v * 0.03 / 52 * ( 1.0 - risk_position)
            #fund_last_v = fund_last_v + fund_last_v * dfr.loc[d, code]
            vs.append(fund_last_v)
            pv = pv + vs[-1]
        portfolio_vs.append(pv)
        result_dates.append(d)


        print(d , pv)

        if i - risk_index >= 4:
            risk_position = 1


        drawdown = FundIndicator.portfolio_drawdown(portfolio_vs)
        risk_drawdown.append(drawdown)
        risk_drawdown.sort()
        if len(risk_drawdown) >= 26:
            if drawdown > risk_drawdown[(int)(0.6 * len(risk_drawdown))]:
                risk_position = risk_position * 0.6
                risk_index    = i

        #print risk_drawdown


    result_datas  = portfolio_vs
    result_df = pd.DataFrame(result_datas, index=result_dates,
                             columns=['highlow_risk_asset'])

    result_df.index.name = 'date'
    result_df.to_csv(datapath('highlowriskasset.csv'))


    highlowriskposition_df = pd.DataFrame(position_datas, index=position_dates, columns=dfr.columns)

    highlowriskposition_df.index.name = 'date'

    highlowriskposition_df.to_csv(datapath('highlowriskposition.csv'))


    # allocationdata.highlow_risk_position_df = highlowriskposition_df
    # allocationdata.highlow_risk_asset_df    = result_df


    return result_df



def highlowriskasset(lookback, adjust_period):


    #highriskassetdf  = allocationdata.equal_risk_asset_df
    highriskassetdf  = pd.read_csv(datapath('equalriskasset.csv'), index_col = 'date', parse_dates = ['date'] )
    highriskassetdfr = highriskassetdf.pct_change().fillna(0.0)


    lowassetlabel    = ['ratebond','creditbond']
    #lowriskassetdfr  = allocationdata.label_asset_df
    lowriskassetdfr  = pd.read_csv(datapath('labelasset.csv'), index_col = 'date', parse_dates = ['date'] )
    lowriskassetdfr  = lowriskassetdfr[lowassetlabel]
    lowriskassetdfr  = lowriskassetdfr.loc[highriskassetdfr.index]


    his_week = lookback
    interval = adjust_period


    highdf = highriskasset(allocationdata, highriskassetdfr, his_week, interval)
    lowdf  = lowriskasset(allocationdata, lowriskassetdfr, his_week, interval)


    df  = pd.concat([highdf, lowdf], axis = 1, join_axes=[highdf.index])
    dfr = df.pct_change().fillna(0.0)


    highlowdf = highlowallocation(allocationdata, dfr, his_week, interval)


    print("sharpe : ", FundIndicator.portfolio_sharpe(highlowdf['highlow_risk_asset'].values))
    print("annual_return : ", FundIndicator.portfolio_return(highlowdf['highlow_risk_asset'].values))
    print("maxdrawdown : ", FundIndicator.portfolio_maxdrawdown(highlowdf['highlow_risk_asset'].values))


    highlowdf.to_csv(datapath('highlowrisk_net_value.csv'))
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


    print("sharpe : ", fi.portfolio_sharpe(highlowdf['highlow_risk_asset'].values))
    print("annual_return : ", fi.portfolio_return(highlowdf['highlow_risk_asset'].values))
    print("maxdrawdown : ", fi.portfolio_maxdrawdown(highlowdf['highlow_risk_asset'].values))


    highlowdf.to_csv(datapath('highlowrisk_net_value.csv'))
    #print "sharpe : ", fi.portfolio_sharpe(highdf['high_risk_asset'].values)
    #print "annual_return : ", fi.portfolio_return(highdf['high_risk_asset'].values)
    #print "maxdrawdown : ", fi.portfolio_maxdrawdown(highdf['high_risk_asset'].values)


    #print lowriskassetdfr
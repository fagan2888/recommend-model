#coding=utf8


import numpy as np
import string
import sys
sys.path.append("windshell")
import const
import Financial as fin
import stockfilter as sf
import stocktag as st
import portfolio as pf
import fundindicator as fi
import fund_selector as fs
import data
import datetime
from numpy import *
import fund_evaluation as fe
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':


    #df  = pd.read_csv('./wind/allocation_data.csv', index_col='date', parse_dates='date')
    #df        = pd.read_csv('./tmp/5fund.csv', index_col='date', parse_dates='date')
    #df        = pd.read_csv('./tmp/high_low_risk_value.csv', index_col='date', parse_dates='date')
    #df         = pd.read_csv('./tmp/eq_risk_data.csv', index_col='date', parse_dates='date')
    df        = pd.read_csv('./tmp/private_fund_highlowrisk_net_value.csv', index_col='date', parse_dates='date')
    ratio_df  = pd.read_csv('./tmp/eq_ratio.csv', index_col='date', parse_dates='date')
    low_df    = pd.read_csv('./tmp/high_low_risk_value.csv', index_col = 'date', parse_dates='date')


    risk_allocation_df    = pd.read_csv('./tmp/risk_allocation.csv', index_col='date', parse_dates='date')
    risk_allocation_dates = risk_allocation_df.index
    risk_allocation_cols  = risk_allocation_df.columns


    low_dfr   = low_df.pct_change().fillna(0.0)

    ratio_cols = ratio_df.columns

    ratio_dates = ratio_df.index

    df  = df.fillna(method='pad')

    dfr = df.pct_change().fillna(0.0)

    dates = dfr.index

    #dates.sort()

    #df.to_csv('hehe.csv')

    fundws = {}
    fund_values = {}
    fund_codes = []
    portfolio_vs = []
    portfolio_vs.append(1)


    risk_level = 10
    net_value_f = open('./tmp/net_value_' + str(risk_level) + '.csv', 'w')
    net_value_f.write('date, net_value\n')
    allocation_f = open('./tmp/allocation.csv', 'w')
    #allocation_f.write('date, largecap, smallcap, rise, oscillation, decline ,growth ,value, ratebond, creditbond, convertiblebond, money1, money2, SP500.SPI, SPGSGCTR.SPI, HSCI.HI\n')
    #allocation_f.write('date, largecap, smallcap, rise, oscillation, decline ,growth ,value, convertiblebond,  SP500.SPI, SPGSGCTR.SPI, HSCI.HI\n')
    #allocation_f.write('date, high, low\n')
    allocation_f.write('date, ltj, zp, lqs, wph\n')


    asset_allocation_f = open('./tmp/asset_alloction.csv','w')
    asset_allocation_f.write('date,ltj,zp,lqs,wph\n')
    asset_allocation_str = '%s,%f,%f,%f,%f\n'
    #allrisk_f = open('./tmp/risks.csv','w')
    #allrisk_f.write('date, largecap, smallcap, rise, oscillation, decline ,growth ,value, ratebond, creditbond, convertiblebond, money1, money2, SP500.SPI, SPGSGCTR.SPI, HSCI.HI\n')

    large_f  = open('./tmp/highrisk_net_value.csv','w')

    large_f.write('date,risk,return\n')

    asset_ratio = {}
    for col in ratio_df.columns:
        asset_ratio.setdefault(col, 1.0)


    risk_asset_ratio = {}
    for col in ratio_df.columns:
        risk_asset_ratio.setdefault(col, 0.0)


    ws = [0, 0]
    for i in range(8, len(dates)):

        if i % 4 == 0:

            start_date = dates[i - 5].strftime('%Y-%m-%d')
            end_date   = dates[i - 1].strftime('%Y-%m-%d')
            d          = dates[i].strftime('%Y-%m-%d')

            allocation_df = df[df.index <= datetime.datetime.strptime(end_date, '%Y-%m-%d')]
            allocation_df = allocation_df[allocation_df.index >= datetime.datetime.strptime(start_date, '%Y-%m-%d')]

            #print allocation_df

            fund_codes = allocation_df.columns

            #print allocation_df

            #uplimit   = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
            #downlimit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            uplimit   = [0.4, 0.4, 0.4, 0.4, 0.4]
            downlimit = [0, 0, 0, 0, 0]
            bound = [downlimit, uplimit]

            #risk, returns, ws, sharpe = pf.markowitz(allocation_df, bound, end_date)

            risk, returns, ws, sharpe = pf.markowitz(allocation_df, None, end_date)

            risk_data = fi.fund_risk(allocation_df)
            risk_dict = {}
            for k, v in risk_data:
                risk_dict[k] = v

            '''
            highrisk = risk_dict['highrisk_net_value']
            lowrisk  = risk_dict['lowrisk_net_value']

            highrisk = highrisk ** 2
            lowrisk  = lowrisk ** 2
            interval = (highrisk - lowrisk) / 9
            wlow     = (highrisk - (lowrisk + risk_level * interval)) / (highrisk - lowrisk)
            whigh    =  1 - wlow
            #print end_date, wlow, whigh

            print wlow, whigh
            ws = [whigh, wlow]
            '''


            #print allocation_df
            last_pv = portfolio_vs[-1]
            fund_values = {}
            for n in range(0, len(fund_codes)):
                fund_values[n] = [last_pv * ws[n]]

            ws_str = "%s, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n"
            ws_str = "%s, %f, %f, %f, %f, %f, %f, %f, %f\n"
            #ws_str = "%s, %f, %f\n"
            ws_str = "%s, %f, %f, %f, %f\n"
            #allocation_f.write(ws_str % (end_date, ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], ws[6], ws[7], ws[8], ws[9], ws[10]))
            #allocation_f.write(ws_str % (end_date, ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], ws[6], ws[7] ))
            #allocation_f.write(ws_str % (end_date, ws[0], ws[1]))
            #allocation_f.write(ws_str % (end_date, ws[0], ws[1], ws[2], ws[3]))

            return_data = fi.fund_return(allocation_df)
            risk_data = fi.fund_risk(allocation_df)
            return_dict = {}
            for k, v in return_data:
                return_dict[k] = v
            risk_dict = {}
            for k, v in risk_data:
                risk_dict[k] = v

            #riskreturn_f = open('./tmp/riskreturn_' + end_date + '.csv','w')
            #riskreturn_f.write('code, risk, return\n')
            #riskreturn_f_str = "%s, %f, %f\n"

            #large_f .write(end_date + "," + str(risk_dict['highrisk_net_value']) + "," + str(return_dict['highrisk_net_value']) + "\n")

            '''
            risks   = []
            returns = []
            for code in fund_codes:
                risks.append(risk_dict[code])
                returns.append(return_dict[code])
                riskreturn_f.write(riskreturn_f_str % (code, risk_dict[code], return_dict[code]))
            riskreturn_f.flush()
            riskreturn_f.close()
            '''


            #plt.plot(risks, returns, 'o', markersize=5)
            #plt.xlabel('std')
            #plt.ylabel('mean')
            #plt.title(end_date)
            #plt.show()

            #allrisk_f.write(ws_str % (end_date, risks[0], risks[1], risks[2], risks[3], risks[4], risks[5], risks[6], risks[7], risks[8], risks[9], risks[10], risks[11], risks[12], risks[13], risks[14]))


        pv = 0
        d = dates[i]
        for n in range(0, len(fund_codes)):
            vs = fund_values[n]
            code = fund_codes[n]
            fund_last_v = vs[-1]
            #print d, low_dfr.loc[d, 'lowrisk_net_value']
            #fund_last_v = fund_last_v + fund_last_v * dfr.loc[d, code] + fund_last_v * (1 - asset_ratio[code.strip()]) * (low_dfr.loc[d, 'lowrisk_net_value'])
            fund_last_v = fund_last_v + fund_last_v * dfr.loc[d, code]
            vs.append(fund_last_v)
            pv = pv + vs[-1]


        #print fund_values
        portfolio_vs.append(pv)
        print d, pv
        net_value_f.write(str(d) + "," + str(pv) + "\n")
        #print ws
        #print ws
        #print asset_ratio
        #print risk_asset_ratio
        asset_allocation_f.write(asset_allocation_str % (d, ws[0] * asset_ratio['ltj'] * risk_asset_ratio['ltj'], ws[0]  * asset_ratio['zp'] * risk_asset_ratio['zp'], ws[0] * asset_ratio['lqs']* risk_asset_ratio['lqs'], ws[0] * asset_ratio['wph'] * risk_asset_ratio['wph']))


        if d in set(ratio_dates):
            for col in ratio_cols:
                asset_ratio[col] = ratio_df.loc[d, col]


        if d in set(risk_allocation_dates):
            for col in risk_allocation_cols:
                risk_asset_ratio[col] = risk_allocation_df.loc[d, col]


    print "sharpe : " ,fi.portfolio_sharpe(portfolio_vs)
    print "annual_return : " ,fi.portfolio_return(portfolio_vs)
    print "maxdrawdown : " ,fi.portfolio_maxdrawdown(portfolio_vs)


    net_value_f.flush()
    net_value_f.close()
    allocation_f.flush()
    allocation_f.close()
    #allrisk_f.flush()
    #allrisk_f.close()
    large_f.flush()
    large_f.close()
    asset_allocation_f.flush()
    asset_allocation_f.close()

#coding=utf8



import string
import pandas as pd
import numpy as np
import FundIndicator
from datetime import datetime
import sys
sys.path.append('shell')



def riskmeanstd(risks):

    risk_mean = np.mean(risks)
    risk_std  = np.std(risks)

    rerisk = []
    risk_max = risk_mean + 2 * risk_std
    risk_min = risk_mean - 2 * risk_std

    for risk in risks:
        if risk >= risk_max or risk <= risk_min or np.isnan(risk):
            continue
        rerisk.append(risk)

    return np.mean(rerisk), np.std(rerisk)



def equalriskassetratio(dfr, pname='', debug='y'):

    #assetlabels = ['largecap','smallcap','rise','oscillation','decline','growth','value','convertiblebond','SP500.SPI','GLNC','HSCI.HI']
    assetlabels = dfr.columns.values
    #dfr         = df.pct_change().fillna(0.0)

    #dfr         = pd.read_csv('./tmp/labelasset.csv', index_col = 'date', parse_dates = 'date' )
    dates = dfr.index

    #interval = 30
    #his_week = 300 #kunge

    interval = 30
    his_week = 300 #kunge

    risk_day = 10
    day_return = {}

    #his_week = 13   #gaopeng


    record = []
    for asset in assetlabels:
        record.append(1.0)
        day_return.setdefault(asset, [0])

    result_datas  = [record]
    result_dates  = [dates[his_week]]


    for i in range(his_week + 1, len(dates)):

        d = dates[i]

        start_date = dates[i - his_week].strftime('%Y-%m-%d')
        end_date   = dates[i].strftime('%Y-%m-%d')
        allocation_date = dates[i - interval].strftime("%Y-%m-%d")

        allocation_dfr = dfr[dfr.index <= datetime.strptime(end_date, '%Y-%m-%d')]
        allocation_dfr = allocation_dfr[allocation_dfr.index >= datetime.strptime(allocation_date, '%Y-%m-%d')]

        #print dfr.index
        his_dfr = dfr[dfr.index <= datetime.strptime(end_date, '%Y-%m-%d')]
        #his_dfr = dfr[dfr.index <= datetime.strptime(allocation_date, '%Y-%m-%d')]
        his_dfr = his_dfr[his_dfr.index >= datetime.strptime(start_date, '%Y-%m-%d')]


        j = 0
        risks = {}
        while j <= len(his_dfr.index):

            riskdfr = his_dfr.iloc[j:j + interval]
            #print riskdfr

            risk_data = {}
            for code in riskdfr.columns:
                risk_data[code] = np.std(riskdfr[code].values)

            for k,v in risk_data.items():
                risk = risks.setdefault(k, [])
                risk.append(v)

            j = j + interval


        ratio_data = []
        for asset in assetlabels:
            mean, std = riskmeanstd(risks[asset])
            asset_std = np.std(allocation_dfr[asset].values)

            max_risk  = mean + 2 * std
            #print mean, std, asset_std, max_risk

            position = 0
            if asset_std >= max_risk:
                position = 0.0
            elif asset_std <= mean:
                position = 1.0
            else:
                position = mean / asset_std
            if position <= 0.5:
                position = 0.0
            ratio_data.append(position)

            #print d, asset, position

        asset_returns = []
        for asset in assetlabels:
            rs = []
            for n in range(0, risk_day):
                rs.append(dfr.loc[dates[i -n], asset])
            #print d, asset, np.mean(rs)
            asset_returns.append(np.mean(rs))    
            day_return[asset].append(np.mean(rs))


        last_ratio = result_datas[-1]
        current_ratio = []
        change_position = False    


        for m in range(0, len(ratio_data)):

            asset   = assetlabels[m]
            drs     = day_return[asset]
            drs.sort()
            r       = asset_returns[m]

            last    = last_ratio[m]
            current = ratio_data[m]

            
            if r <= drs[(int)(0.1 * len(drs))] and r < 0.0:
                #current = min(current, last)
                #if current <= last * 0.5 and (not (current == 0 and last == 0)):
                current  = last * 0.4
                change_position = True
                current_ratio.append(current)
                #if current <= last * 0.5 and (not (current == 0 and last == 0)):
                #    current_ratio.append(current)
                #else:
                #    current_ratio.append(last)    
            #elif current >= last * 2 and (not (current == 0 and last == 0)):
            elif current >= 0.8 and (not (current == 0 and last == 0)):
                #if r >= drs[(int)(0.5 * len(drs))]:
                #    current_ratio.append(current)
                #    change_position = True
                #else:
                #    current_ratio.append(last)    
                current_ratio.append(current)
                change_position = True
            else:
                current_ratio.append(last)    


            '''    
            if (current >= last * 2 or current <= last * 0.5) and (not (current == 0 and last == 0)):
                current_ratio.append(current)
                change_position = True
            else:
                current_ratio.append(last)        
            '''
    
        if change_position:
            #result_datas.append(ratio_data)
            result_datas.append(current_ratio)
            result_dates.append(d)


    
    result_df = pd.DataFrame(result_datas, index=result_dates, columns=assetlabels)
    result_df.index.name = 'date'
    if debug == 'y':
        result_df.to_csv('./result/equalriskassetratio.csv')
    #else:
    #    result_df.to_csv('/tmp/' + pname + 'equalriskassetratio.csv')

    
    return result_df

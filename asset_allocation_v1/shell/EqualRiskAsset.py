#coding=utf8



import string
import pandas as pd
import time
from datetime import datetime
import sys
sys.path.append('shell')
import AllocationData



rf = 0.03 / 252



def equalriskasset(allocationdata):


    #rf = 0.03 / 52


    ratio_df         = allocationdata.equal_risk_asset_ratio_df
    #ratio_df         = pd.read_csv('./tmp/equalriskassetratio.csv', index_col = 'date', parse_dates = 'date' )
    ratio_dates      = ratio_df.index
    start_date = ratio_dates[0]


    dfr              = allocationdata.label_asset_df.pct_change().fillna(0.0)
    #dfr              = pd.read_csv('./tmp/labelasset.csv', index_col = 'date', parse_dates = 'date' )
    dfr              = dfr[dfr.index >= start_date]


    dates = dfr.index
    ratio_dates = ratio_df.index

    #assetlabels  = ['largecap','smallcap','rise','oscillation','decline','growth','value','convertiblebond','SP500.SPI','GLNC','HSCI.HI']

    assetlabels  = ratio_df.columns
    asset_values = {}
    asset_ratio  = {}


    for asset in assetlabels:

        asset_values.setdefault(asset, [1.0])
        asset_ratio.setdefault(asset, 0)

    result_dates = []
    result_datas  = []


    for i in range(0, len(dates)):

        d = dates[i]

        #print d,
        for asset in assetlabels:
            vs = asset_values[asset]
            last_v = vs[-1]
            current_v = last_v + last_v * dfr.loc[d, asset] * asset_ratio[asset]
            vs.append(current_v)
        #    print dfr.loc[d, asset],
        #print 


        if d in ratio_dates:
            for asset in assetlabels:
                asset_ratio[asset] = ratio_df.loc[d, asset]

        asset_vs = []
        for col in ratio_df.columns:
            asset_vs.append(asset_values[col][-1])

        result_datas.append(asset_vs)
        result_dates.append(d)



    #new_assetlabels  = ['largecap','smallcap','rise','oscillation','decline','growth','value','SP500.SPI','GLNC','HSCI.HI']
    result_df = pd.DataFrame(result_datas, index=result_dates, columns=ratio_df.columns)

    result_df.index.name = 'date'

    #print result_df
    result_df.to_csv('./tmp/equalriskassetday.csv')

    result_df = result_df.resample('W-FRI').last()
    result_df = result_df.fillna(method='pad')
    result_df.to_csv('./tmp/equalriskasset.csv')
    allocationdata.equal_risk_asset_df = result_df



if __name__ == '__main__':

    df  = pd.read_csv('./data/qieman.csv', index_col = 'date' ,parse_dates = ['date']).fillna(method = 'pad')
    dfr = df.pct_change().fillna(0.0)
    #allocationdata = AllocationData.allocationdata()
    #allocationdata.label_asset_df = df
    ratio_df = pd.read_csv('./tmp/equalriskassetratio.csv', index_col = 'date' ,parse_dates = ['date'])

    print ratio_df
    #rdf = dfr.shift(1) * ratio_df
    #print rdf
    #allocationdata.equal_risk_asset_ratio_df = df
    #equalriskasset(allocationdata)

#coding=utf8



import pandas as pd
import numpy  as np
import sys
sys.path.append('shell')
import datetime
import AllocationData
from Const import datapath



def intervalreturn(df, interval):

    dates = df.index

    rs = []
    ds = []
    for i in range(interval, len(dates)):
        d = dates[i]
        r = 1.0 * df.iloc[i, 0] / df.iloc[i - interval, 0] - 1
        rs.append(r)
        ds.append(d)

    df = pd.DataFrame(rs, index=ds, columns=['nav'])
    return df



def periodstdmean(df, period):

    dates = df.index

    meanstd = []
    ds      = []
    for i in range(period, len(dates)):
        d  = dates[i]
        rs = df.iloc[i - period : i, 0]

        meanstd.append([np.std(rs), np.mean(rs)])
        ds.append(d)

    df = pd.DataFrame(meanstd, index=ds, columns=['std','mean'])

    return df



def fixrisk(allocationdata):

    
    interval     = 20
    short_period = 20
    long_period  = 252


    # alldf = allocationdata.label_asset_df
    alldf = pd.read_csv(datapath('labelasset.csv'), index_col='date', parse_dates=['date'])


    position_datas = []
    position_dates = []

    for code in alldf.columns:

        df = alldf[[code]]


        ma5_df  = pd.rolling_mean(df, 5)
        ma5_dfr = ma5_df.pct_change().fillna(0.0)
        ma10_df  = pd.rolling_mean(df, 10)
        ma10_dfr = ma10_df.pct_change().fillna(0.0)
        ma20_df  = pd.rolling_mean(df, 20)
        ma20_dfr = ma20_df.pct_change().fillna(0.0)
        ma60_df  = pd.rolling_mean(df, 60)
        ma60_dfr = ma20_df.pct_change().fillna(0.0)


        dfr = df.pct_change().fillna(0.0)

        interval_df = intervalreturn(df, interval)
        periodstdmean_df = periodstdmean(interval_df, short_period)


        dates = periodstdmean_df.index

        ps    = [0]
        pds   = [dates[long_period]]


        for i in range(long_period, len(dates) - 2):

            d = dates[i]

            risk    = periodstdmean_df.iloc[i, 0]
            r       = periodstdmean_df.iloc[i, 1]

            risks   = periodstdmean_df.iloc[i - long_period : i + 1 , 0]
            rs      = periodstdmean_df.iloc[i - long_period : i + 1 , 1]

            rerisks  = risks
            rers     = rs

            riskstd     = np.std(rerisks)
            riskmean    = np.mean(rerisks)

            rstd        = np.std(rers)
            rmean       = np.mean(rers)

            if ma5_dfr.loc[d, code] > 0 and ma10_dfr.loc[d, code] > 0 and ma20_dfr.loc[d, code] > 0:
                position = riskmean / risk
                if position >= 1.0:
                    position = 1.0
            elif risk >= riskmean + 2 * riskstd and r < rmean - 1 * rstd:
                position = riskmean / risk
            elif risk >= riskmean + 2 * riskstd and r > rmean + 1 * rstd:
                position = 0.0
                #position = riskmean / risk
            elif risk <= riskmean:
                position = 1.0
            else:
                position = riskmean / risk

            p = ps[-1]
            if position == 0.0:
                position = 0
            elif (not (position <= p * 0.5 or position >= p * 2.0)) and (not (position <= p - 0.2 or position >= p + 0.2)):
                position = p

            #if position < 0.1:
            #    position = 0.0

            ps.append(position)
            pds.append(dates[i + 1])

        position_datas.append(ps)
        position_dates = pds

    pdf = pd.DataFrame(np.matrix(position_datas).T, index = position_dates, columns = alldf.columns)
    pdf.index.name = 'date'
    pdf.to_csv('./tmp/equalriskassetratio.csv')

    allocationdata.equal_risk_asset_ratio_df = pdf


if __name__ == '__main__':


    df = pd.read_csv('./tmp/labelasset.csv', index_col = 'date', parse_dates = ['date'])
    allocationdata = AllocationData.allocationdata()
    allocationdata.label_asset_df = df
    fixrisk(allocationdata)


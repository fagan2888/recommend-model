#coding=utf8



import pandas as pd
import numpy  as np
import os
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



def fixrisk(interval=20, short_period=20, long_period=252):

    alldf = pd.read_csv(datapath('labelasset.csv'), index_col='date', parse_dates=['date'])
    timing_df = pd.read_csv(os.path.normpath(datapath( '../csvdata/000300_signals.csv')), index_col = 'date', parse_dates=['date'])

    position_datas = []
    position_dates = []

    for code in alldf.columns:

        df = alldf[[code]]

        dfr = df.pct_change().fillna(0.0)

        interval_df = intervalreturn(df, interval)
        periodstdmean_df = pd.DataFrame({
            'mean': interval_df['nav'].rolling(window=short_period).mean(),
            'std': interval_df['nav'].rolling(window=short_period).std(),
        }, columns=['std', 'mean'])

        periodstdmean_df.to_csv('periodstdmean.csv')

        dates = periodstdmean_df.index

        ps    = [0]
        pds   = [dates[long_period]]

        for i in range(long_period, len(dates) - 1):

            d = dates[i]

            signal = 0
            if d in timing_df.index:
                signal = timing_df.loc[d].values[0]

            risk    = periodstdmean_df.iloc[i, 0]
            r       = periodstdmean_df.iloc[i, 1]

            risks   = periodstdmean_df.iloc[i - long_period : i, 0]
            rs      = periodstdmean_df.iloc[i - long_period : i, 1]

            rerisks  = risks
            rers     = rs

            riskstd     = np.std(rerisks)
            riskmean    = np.mean(rerisks)

            rstd        = np.std(rers)
            rmean       = np.mean(rers)

            #
            # 风险修型规则:
            #
            # 1. 波动率大于等于两个标准差 & 收益率小于一个标准差, 则持有部分仓位
            #
            #        position = risk20_mean / risk20
            #    
            # 2. 波动率大于等于两个标准差 & 收益率大于一个标准差 => 空仓
            #
            # 3. 波动率小于波动率均值 则 全仓
            # 
            # 4. 其他情况, 则持有部分仓位
            #
            #        position = risk20_mean / risk20
            #


            if risk >= riskmean + 2 * riskstd and r < rmean - 1 * rstd:
                position = riskmean / risk
            elif risk >= riskmean + 2 * riskstd and r > rmean + 1 * rstd:
                position = 0.0
                #position = riskmean / risk
            elif risk <= riskmean:
                position = 1.0
            else:
                position = riskmean / risk

            #
            # 择时调整规则
            #
            # 1. 择时判断空仓 & 本期仓位小于上期仓位的20%或者是上次仓位的一半以下，
            #    则降低仓位至风险修型新算出的仓位
            #    
            # 2. 择时判断持仓 & 本期仓位大于上期仓位的20%或者是上次仓位的一倍以上，
            #    则增加仓位至风险修型新算出的仓位
            #    
            # 3. 否则, 维持原仓位不变
            #

            if (position <= ps[-1] * 0.5 or position <= ps[-1] - 0.2) and signal == -1:
                pass
            elif (position >= ps[-1] * 2 or position >= ps[-1] + 0.2 or position == 1.0) and signal == 1:
                pass
            else:
                position = ps[-1]

            #if position < 0.1:
            #    position = 0.0

            ps.append(position)
            pds.append(dates[i + 1])

        position_datas.append(ps)
        position_dates = pds

    pdf = pd.DataFrame(np.matrix(position_datas).T, index = position_dates, columns = alldf.columns)
    pdf.index.name = 'date'
    pdf.to_csv(datapath('equalriskassetratio.csv'))

if __name__ == '__main__':


    df = pd.read_csv('./tmp/labelasset.csv', index_col = 'date', parse_dates = ['date'])
    allocationdata = AllocationData.allocationdata()
    allocationdata.label_asset_df = df
    fixrisk(allocationdata)


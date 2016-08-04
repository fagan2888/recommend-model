#codeing=utf8


import string
import pandas as pd
import numpy as np
import FundIndicator
from datetime import datetime
import sys
import EqualRiskAssetRatio
import EqualRiskAsset
import Allocation
import RiskControl
import Position
import copy
sys.path.append('shell')


def fix_risk(risks):

    rerisks = []
    std  = np.std(risks)
    mean = np.mean(risks)
    for risk in risks:
        if risk > (std + mean):
            continue
        else:
            rerisks.append(risk)

    return rerisks


if __name__ == '__main__':


    code = '000905.SH'
    f    = './data/000905.csv'
    #code = '000001.SH'
    #f    = './data/000001.csv'
    start_index   = 0
    risk_lookback = 30


    df = pd.read_csv(f, index_col = 'date', parse_dates = ['date'])
    #df = pd.read_csv('./data/000001.csv', index_col = 'date', parse_dates = ['date'])
    ma5_df  = pd.rolling_mean(df[code], 5)
    ma10_df = pd.rolling_mean(df[code], 10)
    ma20_df = pd.rolling_mean(df[code], 20)

    df['ma5']         = ma5_df
    df['ma10']        = ma10_df
    df['ma20']        = ma20_df


    df = df.dropna()
    dfr = df.pct_change().fillna(0.0)

    risks         = []

    dates = dfr.index

    position_dates = []
    positions      = []


    for i in range(start_index, len(dates)):

        d = dates[i]

        now_risk = np.std(dfr.iloc[i - risk_lookback : i + 1, 0])
        all_risk_std  = np.std(dfr.iloc[0 : i + 1, 0])

        risks.append(all_risk_std)
        risks = fix_risk(risks)

        ma5  = dfr.loc[d, 'ma5']
        ma10 = dfr.loc[d, 'ma10']
        ma20 = dfr.loc[d, 'ma20']

        if now_risk >= np.mean(risks) + np.std(risks) and ma5 < 0.0:    
            if len(positions) == 0 or (not (0.0 == positions[-1])):
                positions.append(0.0)
                position_dates.append(d)
        elif ma10 > 0.0 and ma20 > 0.0 and ma5 > 0: 
            if len(positions) == 0 or (not (1.0 == positions[-1])):
                positions.append(1.0)
                position_dates.append(d)


    p_df = pd.DataFrame(positions, index = position_dates, columns = [code])


    df = pd.read_csv(f, index_col = 'date', parse_dates = ['date'])
        dfr = df.pct_change().fillna(0.0)
        equal_asset_df     = Allocation.allocation_asset(p_df, dfr)

        asset_df = equal_asset_df[['nav']]

    print asset_df
        print "sharpe : ", FundIndicator.portfolio_sharpe_day(asset_df['nav'].values)
        print "annual_return : ", FundIndicator.portfolio_return_day(asset_df['nav'].values)
        print "maxdrawdown : ", FundIndicator.portfolio_maxdrawdown(asset_df['nav'].values)

    p_df.to_csv('position.csv')
    asset_df.to_csv('asset.csv')

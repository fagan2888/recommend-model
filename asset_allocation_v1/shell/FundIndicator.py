#coding=utf8



import numpy as np
import string
import os
import sys
sys.path.append("windshell")
import Financial as fin
import Const
import Data
from numpy import isnan
from datetime import datetime
import pandas as pd
import FundFilter as ff

from Const import datadir


def fund_maxsemivariance(funddf):


    fundsemivariance = {}

    funddfr = funddf.pct_change().fillna(0.0)

    codes = funddfr.columns

    for code in codes:

        rs = []
        for r in funddfr[code].values:
            if not isnan(r):
                rs.append(r)
        max_semivariance = 0

        for i in range(5, len(rs) + 1):
            semivariance = fin.semivariance(rs[0 : i])
            if semivariance > max_semivariance:
                    max_semivariance = semivariance

        fundsemivariance[code] = max_semivariance

    return fundsemivariance


def fund_semivariance(funddf):


    fundsemivariance = {}

    funddfr = funddf.pct_change().fillna(0.0)

    codes = funddfr.columns


    for code in codes:
        rs = []
        for r in funddfr[code].values:
            if not isnan(r):
                rs.append(r)

        fundsemivariance[code] = fin.semivariance(rs)


    return fundsemivariance


def fund_weekly_return(funddf):

    fundweeklyreturn = {}

    funddfr = funddf.pct_change().fillna(0.0)

    codes = funddfr.columns

    for code in codes:
        rs = []
        for r in funddfr[code].values:
            if not isnan(r):
                rs.append(r)
        rs.sort()
        fundweeklyreturn[code] = rs


    return fundweeklyreturn


def fund_month_return(funddf):

    fundmonthreturn = {}

    length = len(funddf.index)


    tran_index = []
    for i in range(0, length):
        if i % 4 == 0:
            tran_index.append(i)

    funddf = funddf.iloc[tran_index]

    funddfr = funddf.pct_change().fillna(0.0)

    codes = funddfr.columns

    for code in codes:
        rs = []
        for r in funddfr[code].values:
            if not isnan(r):
                rs.append(r)

        rs.sort()
        fundmonthreturn[code] = rs


    return fundmonthreturn


def fund_sharp(funddf):

    fundsharp = {}

    funddfr = funddf.pct_change().fillna(0.0)

    codes = funddfr.columns

    for code in codes:
        rs = []
        for r in funddfr[code].values:
            if not isnan(r):
                rs.append(r)

        fundsharp[code] = fin.sharp(rs, const.rf)


    x = fundsharp
    sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
    sorted_sharp = sorted_x

    result = []
    for i in range(0, len(sorted_sharp)):
        result.append(sorted_sharp[i])

    return result


def fund_sharp_annual(funddf):

    fundsharp = {}

    funddfr = funddf.pct_change().fillna(0.0)

    codes = funddfr.columns

    for code in codes:
        rs = []
        for r in funddfr[code].values:
            rs.append(r)

        #if code == '000165':
        #print rs

        fundsharp[code] = fin.sharp_annual(rs, Const.rf)

    x = fundsharp
    sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
    sorted_sharp = sorted_x

    result = []
    for i in range(0, len(sorted_sharp)):
        result.append(sorted_sharp[i])


    return result


def fund_return(funddf):


    fundreturn = {}

    funddfr = funddf.pct_change().fillna(0.0)

    codes = funddfr.columns

    for code in codes:
        vs = funddfr[code].values
        #fundreturn[code] = vs[len(vs) -1] / vs[0] - 1
        fundreturn[code] = np.mean(vs)
        #print code, fundreturn[code]

    x = fundreturn
    sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
    sorted_return = sorted_x

    result = []
    for i in range(0, len(sorted_return)):
        result.append(sorted_return[i])


    return result



def fund_risk(funddf):


    fundrisk = {}

    funddfr = funddf.pct_change().fillna(0.0)

    codes = funddf.columns

    for code in codes:
        fundrisk[code] = np.std(funddfr[code].values)

    x = fundrisk
    sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
    sorted_risk = sorted_x

    result = []
    for i in range(0, len(sorted_risk)):
        result.append(sorted_risk[i])

    return result



def portfolio_sharpe(pvs):
    rs = []
    for i in range(1, len(pvs)):
        rs.append(pvs[i] / pvs[i-1] - 1)
    returns = np.mean(rs) * 52
    risk    = np.std(rs) * (52 ** 0.5)
    return (returns - 0.03) / risk


def portfolio_return(pvs):
    #print pvs
    #print pvs[0]
    #print pvs[len(pvs) - 1]
    rs = []
    for i in range(1, len(pvs)):
        rs.append(pvs[i] / pvs[i-1] - 1)
    return np.mean(rs) * 52
    #return pvs[len(pvs) - 1] / pvs[0] - 1


def portfolio_risk(pvs):
    rs = []
    for i in range(1, len(pvs)):
        rs.append(pvs[i] / pvs[i-1] - 1)
    return np.std(rs)



def portfolio_maxdrawdown(pvs):
    mdd = 0
    peak = pvs[0]
    for v in pvs:
        if v > peak:
            peak = v
        dd = (peak - v) / peak
        if dd > mdd:
            mdd = dd
    return mdd


#计算回撤
def portfolio_drawdown(pvs):
    m = max(pvs)
    return (m - pvs[-1]) / m



#基金的最大回撤
def fund_maxdrawdown(funddf):

    return 0


def fund_jensen(funddf, indexdf):

    rf = 0.03 / 52

    funddfr = funddf.pct_change().fillna(0.0)
    indexdfr = indexdf.pct_change().fillna(0.0)

    jensen = {}
    cols = funddfr.columns
    for col in cols:
        p = []
        m = []
        rs = funddfr[col].values
        #print col, rs
        indexrs = indexdfr.values
        for i in range(0, len(rs)):
            if isnan(rs[i]):
                continue
            else:
                p.append(rs[i])
                m.append(indexrs[i])

        jensen[col] = fin.jensen(p, m, rf)


    x = jensen
    sorted_x       = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
    sorted_measure = sorted_x


    return sorted_measure



if __name__ == '__main__':


    df = pd.read_csv(os.path.join(datadir,'highriskasset.csv'), index_col = 'date', parse_dates = ['date'])

    print "sharpe : ", portfolio_sharpe(df['high_risk_asset'].values)
    print "annual_return : ", portfolio_return(df['high_risk_asset'].values)
    print "maxdrawdown : ", portfolio_maxdrawdown(df['high_risk_asset'].values)


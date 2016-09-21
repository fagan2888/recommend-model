#coding=utf8


import numpy  as np
import pandas as pd
import math
from sklearn import datasets, linear_model
from cvxopt import matrix, solvers


rf = 0.025 / 52


#计算下方差
def semivariance(prs):
    mean       =        np.mean(prs)
    var        =        0.0
    for r in prs:
        if r <= mean:
            var = var + (r - mean) ** 2
    var        =    var / (len(prs) - 1)
    var        =    math.sqrt(var)

    return     var


#计算jensen测度
def jensen(prs, market, rf):

    p = []
    m = []
    for i in range(0, len(prs)):
        p.append(prs[i] - rf)
        m.append(market[i] - rf)

    p = np.array(p)
    m = np.array(m)

    clf       = linear_model.LinearRegression()
    clf.fit(m.reshape(len(m),1), p.reshape(len(p), 1))
    alpha = clf.intercept_[0]
    beta  = clf.coef_[0]


    return alpha


#索提诺比率
def sortino(prs, rf):
    pr        =    np.mean(prs)
    sortino_v = (pr - rf) / semivariance(prs)
    return    sortino_v



#positive peroid weight
def ppw(portfolio, benchmark):


    #print 'p', portfolio
    #print 'm', benchmark

    solvers.options['show_progress'] = False

    length = len(benchmark)
    A = []
    b = []

    for i in range(0, length):
        item = []
        for j in range(0, length + 3):
            item.append(0)
        A.append(item)

    for i in range(0,  length + 3):
        b.append(0)

    for i in range(0, length):
        A[i][i] = -1
        b[i]    = -1e-13

    i = length
    for j in range(0, length):
        A[j][i]     = 1
    b[i] = 1


    i = length + 1
    for j in range(0, length):
        A[j][i] = -1
    b[i] = -1


    i = length + 2
    for j in range(0, length):
        A[j][i] = -benchmark[j]

    b[i] = 0

    c = []
    for j in range(0, length):
        c.append(benchmark[j])


    A = matrix(A)
    b = matrix(b)
    c = matrix(c)

    #print A
    #print b
    #print c

    sol = solvers.lp(c, A, b)
    ppw = 0
    for i in range(0, len(sol['x'])):
        ppw = ppw + portfolio[i] * sol['x'][i]

    return ppw

#sharp
def sharpe_annual(prs, rf):

    r         =    np.mean(prs) * 52
    sigma     =    np.std(prs) * (52 ** 0.5)
    sharpe    = (r - rf) /sigma
    return    sharpe

def ratio_filter(measure, ratio):

    x = measure
    sorted_x       = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
    sorted_measure = sorted_x

    result = []
    for i in range(0, (int)(len(sorted_measure) * ratio)):
        result.append(sorted_measure[i])

    return result

def sharpe_filter(fdfr, ratio):

    fund_sharpe = {} 
    codes = fdfr.columns
    for code in codes:
        rs = fdfr[code].values
        fund_sharpe[code] = sharpe_annual(rs, rf)

    result = ratio_filter(fund_sharpe, ratio)
    return result


def sortino_filter(fdfr, ratio):

    fund_sortino = {} 
    codes = fdfr.columns
    for code in codes:
        rs = fdfr[code].values
        fund_sortino[code] = sortino(rs, rf)

    result = ratio_filter(fund_sortino, ratio)
    return result


def jensen_filter(fdfr, idfr, ratio):
    fund_jensen = {}
    codes = fdfr.columns
    for code in codes:
        rs = fdfr[code].values
        market = idfr.values
        fund_jensen[code] = jensen(rs, market, rf)

    result = ratio_filter(fund_jensen, ratio)
    return result


def ppw_filter(fdfr, idfr, ratio):
    fund_ppw = {}
    codes = fdfr.columns
    for code in codes:
        rs = fdfr[code].values
        market = idfr.values
        fund_ppw[code] = ppw(rs, market)

    result = ratio_filter(fund_ppw, ratio)
    return result



if __name__ == '__main__':


    fdf = pd.read_csv('./data/fund_value.csv', index_col = 'date', parse_dates = ['date'])
    idf = pd.read_csv('./data/index_value.csv', index_col = 'date', parse_dates = ['date'])

    fdf = fdf.iloc[-989:-1,]
    fdf.dropna(axis = 1, inplace = True)
    idf = idf.iloc[-989:-1,]
    idf.dropna(axis = 1, inplace = True)

    fdf  = fdf.resample('W-FRI').last()
    idf  = idf.resample('W-FRI').last()
    fdfr = fdf.pct_change().dropna()
    idfr = idf.pct_change().dropna()
    fdfr.to_csv('fdfr.csv')
    idfr = idfr['000905.SH']


    dates = fdfr.index
    his_back = 52
    interval = 1
    codes    = []
    allvs    = []
    vs       = []
    ds       = []
    rs       = []

    for i in range(his_back, len(dates)):

        d = dates[i]

        if (i - his_back) % interval == 0:
            codes = []
            tmp_fdfr  = fdfr.iloc[i - his_back : i, ]
            tmp_idfr = idfr.iloc[i - his_back : i, ]
            sharpe_result = sharpe_filter(tmp_fdfr, 0.5)
            sortino_result= sortino_filter(tmp_fdfr, 0.5)
            jensen_result = jensen_filter(tmp_fdfr, tmp_idfr, 0.5)
            ppw_result    = ppw_filter(tmp_fdfr, tmp_idfr, 0.5)

            sharpe_codes = []
            sortino_codes= []
            jensen_codes = []
            ppw_codes    = []

            #print result
            for record in sharpe_result:
                sharpe_codes.append(record[0])

            for record in sortino_result:
                sortino_codes.append(record[0])

            for record in jensen_result:
                jensen_codes.append(record[0])

            for record in ppw_result:
                ppw_codes.append(record[0])

            codes = list(set(sharpe_codes) & set(sortino_codes) & set(jensen_codes) & set(ppw_codes))
            #print len(codes), codes

        r = 0
        #codes = fdfr.columns
        for code in codes:
            r = r + fdfr.loc[d, code] / len(codes)
        if len(vs) == 0:
            vs.append(1.0)
        else:
            v = vs[-1] * (1 + r)
            vs.append(v)

        rs.append(r)
        ds.append(d)
        print d, vs[-1], r


    result_df = pd.DataFrame(np.matrix([vs,rs]).T, index = ds, columns = ['nav','rs'])
    result_df.to_csv('nav.csv')
    print result_df

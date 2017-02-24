#coding=utf8


import numpy as np
import string
import os
import sys
sys.path.append("windshell")
import Financial as fin
import Const
import Data
from numpy import *
from datetime import datetime
import multiprocessing
import random
from cvxopt import matrix, solvers
import cvxopt
import math

from Const import datapath

#strategicallocation


#indexallocation


#technicallocation


#中类资产配置
def indexallocation(indexdf):

    indexdfr = indexdf.pct_change()

    indexdfr = indexdfr.fillna(0.0)

    codes = indexdfr.columns

    return_rate = []
    for code in codes:
        return_rate.append(indexdfr[code].values)

    #print return_rate
    risks, returns, ws = fin.efficient_frontier_index(return_rate)

    rf = const.rf

    final_risk = 0
    final_return = 0
    final_ws = []
    final_sharp = -1000


    for i in range(0, len(risks)):


        sharp = (returns[i] - rf) / risks[i]

        if sharp > final_sharp:

                final_risk = risks[i]
                final_return = returns[i]
                final_ws     = ws[i]
                final_sharp  = sharp

    return final_risk, final_return, final_ws, final_sharp



#细类资产配置
def technicallocation(funddf, fund_rank):

    rf = const.rf

    funddfr = funddf.pct_change()

    funddfr = funddfr.fillna(0.0)

    final_risk = 0
    final_return = 0
    final_ws = []
    final_sharp = -10000000000000000.0
    final_codes = []

    for i in range(2, min(11, len(fund_rank))):

        codes = fund_rank[0 : i]
        dfr = funddfr[codes]

        #dfr.fillna(0.0)

        return_rate = []
        for code in codes:
            return_rate.append(dfr[code].values)

        #print return_rate
        risks, returns, ws = fin.efficient_frontier_fund(return_rate)


        for j in range(0, len(risks)):

            sharp = (returns[j] - rf) / risks[j]
            if sharp > final_sharp:

                final_risk = risks[i]
                final_return = returns[i]
                final_ws     = ws[i]
                final_sharp  = sharp


    return final_risk, final_return, final_ws, final_sharp


#markowitz
def markowitz(funddf, bounds, d):

    rf = const.rf
    funddfr = funddf.pct_change()
    funddfr = funddfr.fillna(0.0)

    final_risk = 0
    final_return = 0
    final_ws = []
    final_sharp = -10000000000000000000000000.0
    final_codes = []


    codes = funddfr.columns


    return_rate = []


    for code in codes:
        return_rate.append(funddfr[code].values)


    risks, returns, ws = fin.efficient_frontier(return_rate, bounds)

    for j in range(0, len(risks)):
        sharp = (returns[j] - rf) / risks[j]
        if sharp > final_sharp:
            final_risk = risks[j]
            final_return = returns[j]
            final_ws = ws[j]
            final_sharp = sharp

    f_str = '%s, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n'
    f = open(datapath('ef_' + d + '.csv'),'w')
    f.write('date, risk, return, largecap, smallcap, rise, oscillation, decline ,growth ,value, ratebond, creditbond, convertiblebond, money1, money2, SP500.SPI, SPGSGCTR.SPI, HSCI.HI\n')
    #for j in range(0, len(risks)):
    #    f.write(f_str % (d,risks[j], returns[j], ws[j][0], ws[j][1], ws[j][2], ws[j][3], ws[j][4], ws[j][5], ws[j][6], ws[j][7], ws[j][8], ws[j][9], ws[j][10], ws[j][11], ws[j][12], ws[j][13], ws[j][14] ))

    f.flush()
    f.close()

    return final_risk, final_return, final_ws, final_sharp


def markowitz_r(funddfr, bounds):

    rf = Const.rf

    final_risk = 0
    final_return = 0
    final_ws = []
    final_sharp = -10000000000000000000000000.0
    final_codes = []


    codes = funddfr.columns


    return_rate = []

    for code in codes:
        return_rate.append(funddfr[code].values)


    risks, returns, ws = fin.efficient_frontier(return_rate, bounds)

    for j in range(0, len(risks)):
        sharp = (returns[j] - rf) / risks[j]
        if sharp > final_sharp:
            final_risk = risks[j]
            final_return = returns[j]
            final_ws = ws[j]
            final_sharp = sharp

    return final_risk, final_return, final_ws, final_sharp


def markowitz_r_spe(funddfr, bounds):

    rf = Const.rf

    final_risk = 0
    final_return = 0
    final_ws = []
    final_sharp = -10000000000000000000000000.0
    final_codes = []


    codes = funddfr.columns


    return_rate = []

    for code in codes:
        return_rate.append(funddfr[code].values)


    risks, returns, ws = fin.efficient_frontier_spe(return_rate, bounds)

    for j in range(0, len(risks)):
        sharp = (returns[j] - rf) / risks[j]
        if sharp > final_sharp:
            final_risk = risks[j]
            final_return = returns[j]
            final_ws = ws[j]
            final_sharp = sharp

    return final_risk, final_return, final_ws, final_sharp

def m_markowitz(queue, random_index, df_inc, bound):
    for index in random_index:
        tmp_df_inc = df_inc.iloc[index]
        risk, returns, ws, sharpe = markowitz_r_spe(tmp_df_inc, bound)
        queue.put((risk, returns, ws, sharpe))


def markowitz_bootstrape(df_inc, bound):

    process_num = 8
    look_back = len(df_inc)
    loop_num = look_back * 4
    #loop_num = 20
    randoms = []
    rep_num = loop_num * (look_back / 2) / look_back
    day_indexs = range(0, look_back) * rep_num
    random.shuffle(day_indexs)
    day_indexs = np.array(day_indexs)
    #print day_indexs
    day_indexs = day_indexs.reshape(len(day_indexs) / (look_back / 2), look_back / 2)
    #print day_indexs
    for m in range(0, len(day_indexs)):
        randoms.append(list(day_indexs[m]))
    #print randoms

    q = multiprocessing.Queue()
    processes = []
    process_index_num = len(randoms) / process_num
    for m in range(0, process_num):
        indexs = randoms[m * process_index_num : (m + 1) * process_index_num]
        p = multiprocessing.Process(target = m_markowitz, args = (q, indexs, df_inc, bound,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    wss = np.zeros(len(df_inc.columns))
    risks = []
    returns = []
    sharpes = []
    for m in range(0, q.qsize()):
        record = q.get(m)
        ws = record[2]
        for n in range(0, len(ws)):
            w = ws[n]
            wss[n] = wss[n] + w
        risks.append(record[0])
        returns.append(record[1])
        sharpes.append(record[3])


    ws = wss / loop_num
    return np.mean(risks), np.mean(returns), ws, np.mean(sharpes)


def m_markowitz(queue, random_index, df_inc, bound):
    for index in random_index:
        tmp_df_inc = df_inc.iloc[index]
        risk, returns, ws, sharpe = markowitz_r_spe(tmp_df_inc, bound)
        queue.put((risk, returns, ws, sharpe))


def markowitz_bootstrape(df_inc, bound):

    process_num = 8
    look_back = len(df_inc)
    loop_num = look_back * 4
    #loop_num = 20
    randoms = []
    rep_num = loop_num * (look_back / 2) / look_back
    day_indexs = range(0, look_back) * rep_num
    random.shuffle(day_indexs)
    day_indexs = np.array(day_indexs)
    #print day_indexs
    day_indexs = day_indexs.reshape(len(day_indexs) / (look_back / 2), look_back / 2)
    #print day_indexs
    for m in range(0, len(day_indexs)):
        randoms.append(list(day_indexs[m]))
    #print randoms

    q = multiprocessing.Queue()
    processes = []
    process_index_num = len(randoms) / process_num
    for m in range(0, process_num):
        indexs = randoms[m * process_index_num : (m + 1) * process_index_num]
        p = multiprocessing.Process(target = m_markowitz, args = (q, indexs, df_inc, bound,))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    wss = np.zeros(len(df_inc.columns))
    risks = []
    returns = []
    sharpes = []
    for m in range(0, q.qsize()):
        record = q.get(m)
        ws = record[2]
        for n in range(0, len(ws)):
            w = ws[n]
            wss[n] = wss[n] + w
        risks.append(record[0])
        returns.append(record[1])
        sharpes.append(record[3])


    ws = wss / loop_num
    return np.mean(risks), np.mean(returns), ws, np.mean(sharpes)


#利用blacklitterman做战略资产配置
def strategicallocation(delta,    weq, V, tau, P, Q):

    P = np.array(P)
    Q = np.array(Q)

    tauV = tau * V

    Omega = np.dot(np.dot(P,tauV),P.T) * np.eye(Q.shape[0])

    res = fin.black_litterman(delta, weq, V, tau, P, Q, Omega)

    return re



def largesmallcapfunds(fund_tags):

    largecap          =   fund_tags['largecap']
    smallcap          =   fund_tags['smallcap']
    risefitness       =   fund_tags['risefitness']
    declinefitness    =   fund_tags['declinefitness']
    oscillationfitness=   fund_tags['oscillationfitness']
    growthfitness     =   fund_tags['growthfitness']
    valuefitness      =   fund_tags['valuefitness']


    largecap_set      =   set(largecap)
    smallcap_set      =   set(smallcap)

    largecap_fund     =   []
    smallcap_fund     =   []

    largecap_fund.append(largecap[0])
    smallcap_fund.append(smallcap[0])


    for code in risefitness:
        if code in largecap_set:
            largecap_fund.append(code)
            break

    for code in declinefitness:
        if code in largecap_set:
            largecap_fund.append(code)
            break


    for code in oscillationfitness:
        if code in largecap_set:
            largecap_fund.append(code)
            break

    for code in growthfitness:
        if code in largecap_set:
            largecap_fund.append(code)
            break

    for code in valuefitness:
        if code in largecap_set:
            largecap_fund.append(code)
            break


    for code in risefitness:
        if code in smallcap_set:
            smallcap_fund.append(code)
            break

    for code in declinefitness:
        if code in smallcap_set:
            smallcap_fund.append(code)
            break


    for code in oscillationfitness:
        if code in smallcap_set:
            smallcap_fund.append(code)
            break

    for code in growthfitness:
        if code in smallcap_set:
            smallcap_fund.append(code)
            break

    for code in valuefitness:
        if code in smallcap_set:
            smallcap_fund.append(code)
            break


    largecap_fund = list(set(largecap_fund))
    smallcap_fund = list(set(smallcap_fund))

    return largecap_fund, smallcap_fund


#
def boundlimit(n):

    bounds = []

    min_bound  = []
    max_bound  = []
    for i in range(0, n):
        min_bound.append(0.05)
        max_bound.append(0.4)

    bounds.append(min_bound)
    bounds.append(max_bound)

    return bounds


#资产配置
def asset_allocation(start_date, end_date, largecap_fund, smallcap_fund, P, Q):
#########################################################################

    delta = 2.5
    tau = 0.05

    ps = []
    for p in P:
        ps.append(np.array(p))

    P = np.array(ps)

    qs = []
    for q in Q:
        qs.append(np.array(q))

    Q = np.array(qs)


    indexdf = data.index_value(start_date, end_date, [const.largecap_code, const.smallcap_code])

    indexdfr = indexdf.pct_change().fillna(0.0)

    indexrs = []
    for code in indexdfr.columns:
        indexrs.append(indexdfr[code].values)

    #print indexdfr

    sigma = np.cov(indexrs)

    #print type(sigma)
    #print sigma
    #print np.cov(indexrs)
    #print indexdfr


    weq = np.array([0.5, 0.5])
    tauV = tau * sigma
    Omega = np.dot(np.dot(P,tauV),P.T) * np.eye(Q.shape[0])
    er, ws, lmbda = fin.black_litterman(delta, weq, sigma, tau, P, Q, Omega)


    sum = 0
    for w in ws:
        sum = sum + w
    for i in range(0, len(ws)):
        ws[i] = 1.0 * ws[i] / sum

    #print er
    indexws = ws
    #print indexws
    #largecap_fund, smallcap_fund = largesmallcapfunds(fund_tags)

    #print largecap_fund
    #risk, returns, ws, sharp = markowitz(
    #print smallcap_fund


    funddf = data.fund_value(start_date, end_date)

    bounds = boundlimit(len(largecap_fund))

    risk, returns, ws, sharp = markowitz(funddf[largecap_fund], bounds)

    largecap_fund_w = {}
    for i in range(0, len(largecap_fund)):
        code = largecap_fund[i]
        largecap_fund_w[code] = ws[i] * indexws[0]


    bounds = boundlimit(len(smallcap_fund))
    risk, returns ,ws ,sharp = markowitz(funddf[smallcap_fund], bounds)

    smallcap_fund_w = {}
    for i in range(0, len(smallcap_fund)):
        code = smallcap_fund[i]
        smallcap_fund_w[code] = ws[i] * indexws[1]


    '''
    #平均分配
    largecap_fund_w = {}
    for code in largecap_fund:
        largecap_fund_w[code] = 1.0 / len(largecap_fund) * indexws[0]


    smallcap_fund_w = {}
    for code in smallcap_fund:
        smallcap_fund_w[code] = 1.0 / len(smallcap_fund) * indexws[1]
    '''

    fundws = {}
    for code in largecap_fund:
        w = fundws.setdefault(code, 0)
        fundws[code] = w + largecap_fund_w[code]
    for code in smallcap_fund:
        w = fundws.setdefault(code, 0)
        fundws[code] = w + smallcap_fund_w[code]


#######################################################################

    #print largecap
    #print smallcap
    #print risefitness
    #print declinefitness
    #print oscillafitness
    #print growthfitness
    #print valuefitness
    #print


    fund_codes = []
    ws         = []
    for k, v in fundws.items():
        fund_codes.append(k)
        ws.append(v)

    #for code in largecap:

    return fund_codes, ws



def black_litterman(weq, df_inc, P, Q):

    tau = 0.05
    delta = 2.5
    weq = np.array(weq)
    P = np.array(P)
    Q = np.array(Q)

    return_rate = []
    for code in df_inc.columns:
        return_rate.append(df_inc[code].values)

    sigma = np.cov(return_rate)
    tauV = tau * sigma
    Omega = np.dot(np.dot(P, tauV), P.T) * np.eye(Q.shape[0])

    # Reverse optimize and back out the equilibrium returns
    # This is formula (12) page 6.
    pi = weq.dot(sigma * delta)
    # print(pi)
    # We use tau * sigma many places so just compute it once
    ts = tau * sigma
    # Compute posterior estimate of the mean
    # This is a simplified version of formula (8) on page 4.

    middle = linalg.inv(np.dot(np.dot(P,ts),P.T) + Omega)
    # print(middle)
    # print(Q-np.expand_dims(np.dot(P,pi.T),axis=1))
    er = np.expand_dims(pi,axis=0).T + np.dot(np.dot(np.dot(ts,P.T),middle),(Q - np.expand_dims(np.dot(P,pi.T),axis=1)))
    # Compute posterior estimate of the uncertainty in the mean
    # This is a simplified and combined version of formulas (9) and (15)
    posteriorSigma = sigma + ts - ts.dot(P.T).dot(middle).dot(P).dot(ts)
    # print(posteriorSigma)
    # Compute posterior weights based on uncertainty in mean
    w = er.T.dot(linalg.inv(delta * posteriorSigma)).T
    # Compute lambda value
    # We solve for lambda from formula (17) page 7, rather than formula (18)
    # just because it is less to type, and we've already computed w*.
    lmbda = np.dot(linalg.pinv(P).T,(w.T * (1 + tau) - weq).T)

    #print sigma
    #print
    #print posteriorSigma
    #print weq
    #print P
    #print Q
    #print 
    #print np.mean(return_rate, axis = 1)
    #print 
    #print er
    #print 
    #print w
    #print 
    #print np.dot(w.T, er)
    #print 
    #print sqrt(np.dot(w.T, posteriorSigma).dot(w))
    #print er.shape
    #print w.shape
    #print posteriorSigma.shape

    solvers.options['show_progress'] = False
    n_asset = len(df_inc.columns)
    S          =     matrix(posteriorSigma)
    pbar       =     matrix(er)

    G          =     matrix(0.0, (n_asset, n_asset))
    G[::n_asset + 1]  =  -1.0
    h                 =  matrix(0.0, (n_asset, 1))
    A                 =  matrix(1.0, (1, n_asset))
    b                 =  matrix(1.0)

    #print S
    N = 200
    mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
    portfolios = [ solvers.qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus ]
    returns = [ cvxopt.blas.dot(pbar,x) for x in portfolios ]
    risks = [ math.sqrt(cvxopt.blas.dot(x, S*x)) for x in portfolios ]

    rf = Const.rf

    final_risk = 0
    final_return = 0
    final_ws = []
    final_sharp = -np.inf
    final_codes = []

    for j in range(0, len(risks)):
        sharp = (returns[j] - rf) / risks[j]
        if sharp > final_sharp:
            final_risk = risks[j]
            final_return = returns[j]
            final_ws = portfolios[j]
            final_sharp = sharp

    #print np.mean(return_rate, axis = 1)
    #print
    #print Q
    #print
    #print final_ws
    #print 
    return final_risk, final_return, final_ws, final_sharp

    #print risks
    #print np.dot(er, w)
    #return [er, w, lmbda]

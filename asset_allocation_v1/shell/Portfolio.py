#coding=utf8


import numpy as np
import string
import os
import sys
import logging
sys.path.append("windshell")
import Financial as fin
import Const
import Data
from numpy import *
from datetime import datetime
import multiprocessing
import random
from multiprocessing import Manager
import scipy
import scipy.optimize


from Const import datapath
from util.xdebug import dd

logger = logging.getLogger(__name__)


def markowitz_r_spe(funddfr, bounds):

    rf = Const.rf

    final_risk = 0
    final_return = 0
    final_ws = list(1.0 * np.ones(len(funddfr.columns)) / len(funddfr.columns))
    final_sharp = -10000000000000000000000000.0
    final_codes = []


    codes = funddfr.columns
    return_rate = []
    for code in codes:
        return_rate.append(funddfr[code].values)


    risks, returns, ws = fin.efficient_frontier_spe(return_rate, bounds)

    for j in range(0, len(risks)):
        if risks[j] == 0:
            continue
        sharp = (returns[j] - rf) / risks[j]
        # if sharp < 0:
        #     sharp = 1 / sharp
        if sharp > final_sharp:
            final_risk = risks[j]
            final_return = returns[j]
            final_ws = ws[j]
            final_sharp = sharp

    return final_risk, final_return, final_ws, final_sharp


def markowitz_r_spe_bl(funddfr, P, eta, alpha, bounds):

    rf = Const.rf

    final_risk = 0
    final_return = 0
    final_ws = list(1.0 * np.ones(len(funddfr.columns)) / len(funddfr.columns))
    final_sharp = -10000000000000000000000000.0
    final_codes = []

    codes = funddfr.columns
    return_rate = []
    for code in codes:
        return_rate.append(funddfr[code].values)

    if eta.size == 0:
        risks, returns, ws = fin.efficient_frontier_spe(return_rate, bounds)
    else:
        risks, returns, ws = fin.efficient_frontier_spe_bl(return_rate, P, eta, alpha, bounds)

    for j in range(0, len(risks)):
        if risks[j] == 0:
            continue
        sharp = (returns[j] - rf) / risks[j]
        # if sharp < 0:
        #     sharp = 1 / sharp
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


def m_markowitz_bl(queue, random_index, df_inc, P, eta, alpha, bound):
    for index in random_index:
        tmp_df_inc = df_inc.iloc[index]
        risk, returns, ws, sharpe = markowitz_r_spe_bl(tmp_df_inc, P, eta, alpha, bound)
        queue.put((risk, returns, ws, sharpe))


def markowitz_bootstrape_bl(df_inc, P, eta, alpha, bound, cpu_count = 0, bootstrap_count=0):

    os.environ['OMP_NUM_THREADS'] = '1'

    if cpu_count == 0:
        count = multiprocessing.cpu_count()
        cpu_count = count if count > 0 else 1

    look_back = len(df_inc)
    if bootstrap_count <= 0:
        loop_num = look_back * 4
    elif bootstrap_count % 2:
        loop_num = bootstrap_count + 1
    else:
        loop_num = bootstrap_count

    # logger.info("bootstrap_count: %d, cpu_count: %d", loop_num, cpu_count)
    process_indexs = [[] for i in range(0, cpu_count)]

    #print process_indexs
    #loop_num = 20
    rep_num = loop_num * (look_back / 2) / look_back
    day_indexs = range(0, look_back) * rep_num
    random.shuffle(day_indexs)
    #print day_indexs
    day_indexs = np.array(day_indexs)


    day_indexs = day_indexs.reshape(len(day_indexs) / (look_back / 2), look_back / 2)
    for m in range(0, len(day_indexs)):
        indexs = day_indexs[m]
        mod = m % cpu_count
        process_indexs[mod].append(list(indexs))


    manager = Manager()
    q = manager.Queue()
    processes = []
    for indexs in process_indexs:
        if eta.size == 0:
            p = multiprocessing.Process(target = m_markowitz, args = (q, indexs, df_inc, bound,))
        else:
            p = multiprocessing.Process(target = m_markowitz_bl, args = (q, indexs, df_inc, P, eta, alpha, bound))
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




def markowitz_bootstrape(df_inc, bound, cpu_count = 0, bootstrap_count=0):

    os.environ['OMP_NUM_THREADS'] = '1'

    if cpu_count == 0:
        count = multiprocessing.cpu_count()
        cpu_count = count if count > 0 else 1

    look_back = len(df_inc)
    if bootstrap_count <= 0:
        loop_num = look_back * 4
    elif bootstrap_count % 2:
        loop_num = bootstrap_count + 1
    else:
        loop_num = bootstrap_count

    # logger.info("bootstrap_count: %d, cpu_count: %d", loop_num, cpu_count)
    process_indexs = [[] for i in range(0, cpu_count)]

    #print process_indexs
    #loop_num = 20
    rep_num = loop_num * (look_back / 2) / look_back
    day_indexs = range(0, look_back) * rep_num
    random.shuffle(day_indexs)
    #print day_indexs
    day_indexs = np.array(day_indexs)


    day_indexs = day_indexs.reshape(len(day_indexs) / (look_back / 2), look_back / 2)
    for m in range(0, len(day_indexs)):
        indexs = day_indexs[m]
        mod = m % cpu_count
        process_indexs[mod].append(list(indexs))


    manager = Manager()
    q = manager.Queue()
    processes = []
    for indexs in process_indexs:
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


def riskparity(dfr):
    cov = dfr.cov()
    cov = cov.values
    asset_num = len(cov)
    w = 1.0 * np.ones(asset_num) / asset_num
    bound = [ (0.0 , 1.0) for i in range(asset_num)]
    constrain = ({'type':'eq', 'fun': lambda w: sum(w)-1.0 })
    result = scipy.optimize.minimize(riskparity_obj_func, w, (cov), method='SLSQP', constraints=constrain, bounds=bound)
    ws = result.x
    return ws


def riskparity_obj_func(w, cov):
    n = len(cov)
    risk_sum = 0
    for i in range(0, n):
        for j in range(i, n):
            risk_sum = risk_sum + (np.dot(w, cov[i]) - np.dot(w , cov[j])) ** 2
    return risk_sum

#coding=utf8


import numpy as np
import string
import os
import sys
import logging
sys.path.append("windshell")
import Financial as fin
import Const
from numpy import *
from datetime import datetime
import multiprocessing
import random
from multiprocessing import Manager
import scipy
import scipy.optimize
from ipdb import set_trace


from Const import datapath
from util.xdebug import dd

logger = logging.getLogger(__name__)



def markowitz_r_spe(funddfr, bounds):

    rf = Const.rf

    final_risk = 0
    final_return = 0
    final_ws = list(1.0 * np.ones(len(funddfr.columns)) / len(funddfr.columns))
    final_sharp = -np.inf
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
    final_ws = []
    final_ws = list(1.0 * np.ones(len(funddfr.columns)) / len(funddfr.columns))
    final_sharp = -inf
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
        count = int(multiprocessing.cpu_count())
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
    rep_num = loop_num * (look_back // 2) // look_back
    day_indexs = list(range(0, look_back)) * rep_num
    random.shuffle(day_indexs)
    #print day_indexs
    day_indexs = np.array(day_indexs)


    day_indexs = day_indexs.reshape(len(day_indexs) // (look_back // 2), look_back // 2)
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
        count = int(multiprocessing.cpu_count())
        cpu_count = count if count > 0 else 1
    cpu_count = int(cpu_count)

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
    rep_num = loop_num * (look_back // 2) // look_back
    day_indexs = list(range(0, look_back)) * rep_num
    random.shuffle(day_indexs)
    #print day_indexs
    day_indexs = np.array(day_indexs)


    day_indexs = day_indexs.reshape(len(day_indexs) // (look_back // 2), look_back // 2)
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


# def markowitz_fixrisk(df_inc, bound, target_risk):

#     final_risk = 0
#     final_return = 0
#     final_ws = list(1.0 * np.ones(len(df_inc.columns)) / len(df_inc.columns))
#     final_sharp = -np.inf
#     final_codes = []

#     codes = df_inc.columns
#     return_rate = []
#     for code in codes:
#         return_rate.append(df_inc[code].values)

#     risks, returns, ws = fin.efficient_frontier_spe(return_rate, bound)
#     risk_diff = np.inf
#     for j in range(0, len(risks)):
#         if risks[j] == 0:
#             continue
#         if abs(risks[j] - target_risk) < risk_diff:
#             final_risk = risks[j]
#             final_return = returns[j]
#             final_ws = ws[j]
#             final_sharp = (returns[j] - Const.rf) / risks[j]
#             risk_diff = abs(risks[j] - target_risk)

#     # print(target_risk, final_risk, min(risks), max(risks))
#     return final_risk, final_return, final_ws, final_sharp


def markowitz_fixrisk(df_inc, bound, target_risk):

    w0 = [1/len(df_inc.columns)]*len(df_inc.columns)


    bnds = [(bound[i]['lower'], bound[i]['upper']) for i in range(len(bound))]
    ret = df_inc.mean().values
    vol = df_inc.cov().values

    cons = (
        {'type': 'eq', 'fun': lambda x : np.sum(x) - 1.0},
        {'type': 'ineq', 'fun': lambda x: target_risk - np.sqrt(np.dot(x,np.dot(vol,x)))},
    )

    res = scipy.optimize.minimize(risk_budget_objective, w0, args=[ret, vol, target_risk], method='SLSQP', bounds = bnds, constraints=cons, options={'disp': False, 'eps': 1e-3})

    final_risk = np.sqrt(np.dot(res.x,np.dot(vol,res.x)))
    final_return = np.dot(res.x, ret)
    final_ws = res.x
    final_sharp = (final_return - Const.rf) / final_risk

    # print()
    # print(target_risk, final_risk)

    return final_risk, final_return, final_ws, final_sharp


def risk_budget_objective(x,pars):
    LN = 10
    ret = pars[0]
    vol = pars[1]
    tr = pars[2]

    ret_p = np.dot(ret, x)
    vol_p = np.sqrt(np.dot(x, np.dot(vol,x)))
    # target = -ret_p + LN*np.abs(vol_p - tr)
    target = -ret_p

    return target


def total_weight_constraint(x):
    return np.sum(x)-1.0


def m_markowitz_fixrisk(queue, random_index, df_inc, bound, target_risk):

    for index in random_index:
        tmp_df_inc = df_inc.iloc[index]
        risk, returns, ws, sharpe = markowitz_fixrisk(tmp_df_inc, bound, target_risk)
        queue.put((risk, returns, ws, sharpe))


def markowitz_bootstrape_fixrisk(df_inc, bound, target_risk, cpu_count = 0, bootstrap_count=0):

    os.environ['OMP_NUM_THREADS'] = '1'

    if cpu_count == 0:
        count = int(multiprocessing.cpu_count())
        cpu_count = count if count > 0 else 1
    cpu_count = int(cpu_count)

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
    rep_num = loop_num * (look_back // 2) // look_back
    day_indexs = list(range(0, look_back)) * rep_num
    random.shuffle(day_indexs)
    #print day_indexs
    day_indexs = np.array(day_indexs)


    day_indexs = day_indexs.reshape(len(day_indexs) // (look_back // 2), look_back // 2)
    for m in range(0, len(day_indexs)):
        indexs = day_indexs[m]
        mod = m % cpu_count
        process_indexs[mod].append(list(indexs))


    manager = Manager()
    q = manager.Queue()
    processes = []
    for indexs in process_indexs:
        p = multiprocessing.Process(target = m_markowitz_fixrisk, args = (q, indexs, df_inc, bound, target_risk))
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


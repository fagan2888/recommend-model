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
    final_ws = []
    final_sharp = -10000000000000000000000000.0
    final_codes = []


    codes = funddfr.columns
    return_rate = []
    for code in codes:
        return_rate.append(funddfr[code].values)


    risks, returns, ws = fin.efficient_frontier_spe(return_rate, bounds)

    final_ws = ws[0]
    for j in range(0, len(risks)):
        if risks[j] == 0:
            if np.sum(ws[j] ** 2) > np.sum(final_ws ** 2):
                final_risk = risks[j]
                final_return = returns[j]
                final_ws = ws[j]
                final_sharp = np.inf
        else:
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

    final_ws = ws[0]
    for j in range(0, len(risks)):
        if risks[j] == 0:
            if np.sum(ws[j] ** 2) > np.sum(final_ws ** 2):
                final_risk = risks[j]
                final_return = returns[j]
                final_ws = ws[j]
                final_sharp = np.inf
        else:
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

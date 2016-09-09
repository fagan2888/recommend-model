#coding=utf8


import numpy as np
import string
import os
import sys
sys.path.append("windshell")
import Financial as fin
import Const
from numpy import *
from datetime import datetime
from Const import datadir
import cvxopt as opt
from cvxopt import matrix
from scipy import linalg
from cvxopt import blas, solvers


#markowizt 资产配置
def optimal_portfolio(returns):

    solvers.options['show_progress'] = False
    n = len(returns)
    returns = np.asmatrix(returns)
    
    N = 500
    mus = [10**(5.0 * t/N - 1.0) for t in range(N)]
    
    # Convert to cvxopt matrices
    S = opt.matrix(np.cov(returns))
    pbar = opt.matrix(np.mean(returns, axis=1))
    
    # Create constraint matrices
    G = -opt.matrix(np.eye(n))   # negative n x n identity matrix
    h = opt.matrix(0.0, (n ,1))
    A = opt.matrix(1.0, (1, n))
    b = opt.matrix(1.0)
    
    # Calculate efficient frontier weights using quadratic programming
    portfolios = [solvers.qp(mu*S, -pbar, G, h, A, b)['x'] 
                  for mu in mus]
    ## CALCULATE RISKS AND RETURNS FOR FRONTIER
    returns = [blas.dot(pbar, x) for x in portfolios]
    risks = [np.sqrt(blas.dot(x, S*x)) for x in portfolios]
    ## CALCULATE THE 2ND DEGREE POLYNOMIAL OF THE FRONTIER CURVE
    m1 = np.polyfit(returns, risks, 2)
    x1 = np.sqrt(m1[2] / m1[0])
    # CALCULATE THE OPTIMAL PORTFOLIO
    wt = solvers.qp(opt.matrix(x1 * S), -pbar, G, h, A, b)['x']

    return risks, returns, np.asarray(wt)


def new_markowitz(funddfr):

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

    risks, returns, ws = optimal_portfolio(return_rate)

    print ws

    for j in range(0, len(risks)):
        sharp = (returns[j] - rf) / risks[j]
        if sharp > final_sharp:
            final_risk   = risks[j]
            final_return = returns[j]
            final_sharp  = sharp

    final_ws = ws

    return final_risk, final_return, final_ws, final_sharp


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
    f = open(os.path.join(datadir,'ef_' + d + '.csv'),'w')
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


def black_litterman(dfr, delta, tau, ws, P, Q, ):

    solvers.options['show_progress'] = False

    rs = []
    cols = dfr.columns
    for col in cols:
        rs.append(dfr[col].values)

    delta = 2.5
    tau = 0.02
    n_asset = len(cols)
    P = np.array(P)
    Q = np.array(Q)

    weq = np.array(ws)

    sigma = np.cov(rs)
    tauV  = tau * sigma
    Omega = np.dot(np.dot(P,tauV),P.T) * np.eye(Q.shape[0])
    pi = weq.dot(sigma * delta)
    ts = tau * sigma
    middle = linalg.inv(np.dot(np.dot(P,ts),P.T) + Omega)
    er = np.expand_dims(pi,axis=0).T + np.dot(np.dot(np.dot(ts,P.T),middle),(Q - np.expand_dims(np.dot(P,pi.T),axis=1)))
    posteriorSigma = sigma + ts - ts.dot(P.T).dot(middle).dot(P).dot(ts)

    G                 =     matrix(0.0, (n_asset, n_asset))
    G[::n_asset + 1]  =  -1.0
    h                 =  matrix(0.0, (n_asset, 1))
    A                 =  matrix(1.0, (1, n_asset))
    b                 =  matrix(1.0)

    pbar              = matrix(er)
    S                 = matrix(posteriorSigma)

    min_ws    = solvers.qp(delta / 2.0 * S, -1.0 * pbar, G, h, A, b)['x']

    return min_ws



#min risk allocation model
def min_risk(funddfr):

    codes = funddfr.columns
    return_rate = []
    for code in codes:
        return_rate.append(funddfr[code].values)

    ws = fin.min_risk_allocation(return_rate)

    return ws

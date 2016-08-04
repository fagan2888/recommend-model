#coding=utf8


import string
import math

from math import sqrt
from cvxopt import matrix
from cvxopt.blas import dot
from cvxopt.solvers import qp
from sklearn import datasets, linear_model    
import numpy as np
from scipy.stats import norm
from cvxopt import matrix, solvers


#计算有效前沿
def efficient_frontier(return_rate):

    n_asset    =     len(return_rate)


    asset_mean = np.mean(return_rate, axis = 1)
    #print asset_mean

    cov        =     np.cov(return_rate)
    
    S       =     matrix(cov)
    pbar       =     matrix(asset_mean)    
    
    G          =     matrix(0.0, (n_asset, n_asset))    
    G[::n_asset + 1]  =  -1.0    
    h                 =  matrix(0.0, (n_asset, 1))
    A                 =  matrix(1.0, (1, n_asset))
    b                 =  matrix(1.0)

            
    N = 100
    mus = [ 10**(5.0*t/N-1.0) for t in range(N) ]
    portfolios = [ qp(mu*S, -pbar, G, h, A, b)['x'] for mu in mus ]
    returns = [ dot(pbar,x) for x in portfolios ]
    risks = [ sqrt(dot(x, S*x)) for x in portfolios ]    

    
    return risks, returns, portfolios



#计算下方差
def semivariance(portfolio):
    mean       =        np.mean(portfolio)
    var        =        0.0
    n          =        0    
    for d in portfolio:
        if d <= mean:
            n   = n + 1
            var = var + (d - mean) ** 2
    var        =    var / n

    return     math.sqrt(var)    
            

        
#jensen测度
def jensen(portfolio, market, rf):

    pr         =    np.mean(portfolio)
    mr         =    np.mean(market)
    beta       =    np.cov(portfolio, market)[0][1] / np.cov(market)
    return pr - (rf + beta * ( mr - rf) )    



#sharp
def sharp(portfolio, rf):
    r         =    np.mean(portfolio)
    sigma     =    np.std(portfolio)
    return    (r - rf) / sigma



#索提诺比率
def sortino(portfolio, rf):
    pr        =    np.mean(portfolio)
    return (pr - rf ) / semivariance(portfolio)



#treynor-mazuy模型
def tm(portfolio, market, rf):
    xparams   = []    
    yparams   = []
    for i in range(0, len(portfolio)):
        yparams.append(portfolio[i] - rf)
        xparams.append([market[i] - rf, (market[i] - rf) ** 2])                    
        
    clf       = linear_model.LinearRegression()
    clf.fit(xparams, yparams)
    return clf.intercept_,  clf.coef_[0], clf.coef_[1]



#henrikson-merton
def hm(portfolio, market, rf):
    xparams   = []    
    yparams   = []
    for i in range(0, len(portfolio)):
        yparams.append(portfolio[i] - rf)
        if rf - market[i] > 0:
            xparams.append([market[i] - rf, market[i] - rf])                    
        else:
            xparams.append([market[i] - rf, 0])    
                    
    clf       = linear_model.LinearRegression()
    clf.fit(xparams, yparams)
    return clf.intercept_,  clf.coef_[0], clf.coef_[1]



#value at risk
def var(portfolio):
    parray = np.array(portfolio)    
    valueAtRisk = norm.ppf(0.05, parray.mean(), parray.std())    
    return valueAtRisk





#positive peroid weight
def ppw(portfolio, benchmark, rf):    

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

    sol = solvers.lp(c, A, b)
    ppw = 0
    for i in range(0, len(sol['x'])):
        ppw = ppw + portfolio[i] * sol['x'][i]    

    return ppw




def grs(portfolio):
        
    return 0    



def black_litterman():
    return 0        



def printHello():
    print 'hello'


if __name__ == '__main__':


        rs = [[0.25,0.3,0.4, 0.3, 0.2, -0.1, -0.2], [0.1, 0.2, 0.3, -0.01, -0.2, 0.01, 0.02]]
    #print np.cov(rs)
           #print semivariance(rs[0])
    #print jensen(rs[0], rs[1], 0.02)
    #print sharp(rs[0], 0.02)
    #print sortino(rs[0], 0.02)
    #print tm(rs[0], rs[1], 0.02)
    print ppw(rs[0], rs[1])



    


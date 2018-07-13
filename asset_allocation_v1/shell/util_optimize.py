#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8
import numpy as np
import pandas as pd
import scipy
import Const


def solve_weights(R, C, rf):
    # method = 'SLSQP'
    method = 'L-BFGS-B'

    def fitness(W, R, C, rf):
        mean, var = port_mean_var(W, R, C)
        sr = (mean - rf) / np.sqrt(var)
        return -sr

    n = len(R)
    W = np.ones([n])/n
    b_ = [(-1.,1.) for i in range(n)]
    # c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1.0})
    # optimized = scipy.optimize.minimize(fitness, W, (R, C, rf),
                # method='SLSQP', constraints=c_, bounds=b_)
    optimized = scipy.optimize.minimize(fitness, W, (R, C, rf),
                method=method, bounds=b_)
    if not optimized.success:
        raise BaseException(optimized.message)
    return optimized.x


def port_mean(W, R):
    return sum(R * W)


def port_var(W, C):
    return np.dot(np.dot(W, C), W)


def port_mean_var(W, R, C):
    return port_mean(W, R), port_var(W, C)





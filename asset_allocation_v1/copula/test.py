#!/usr/bin/python
# coding=utf-8

import numpy as np
from scipy.stats import rankdata
from bvcopula.bvcopula import *
from ipdb import set_trace

a = np.random.normal(size=1000)
b = a*2 + np.random.normal(size=1000)

x = rankdata(a)/len(a)
y = rankdata(b)/len(b)

'''
# t_copula = bv_cop_mle(x, y, 2)
# t_copula = bv_cop_cdf(x, y, [0.5, 10], 2)
t_loglik = 0.0
split_point = range(0,1000,10)
for i in split_point:
    try:
        # t_loglik += bv_cop_loglik(x[i:i+10], y[i:i+10], [0.5, 10], 2)
        print bv_cop_loglik(x[i:i+10], y[i:i+10], [0.1, 20], 2)
        print x[i:i+10]
        print y[i:i+10]
    except:
        print x[i:i+10]
        print y[i:i+10]
'''

res = bv_cop_model_selection(x, y)
set_trace()
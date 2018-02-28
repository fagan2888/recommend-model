# -*- coding: utf-8 -*-
"""
Created on Sat Jun 18 12:07:21 2011

@author: Sat Kumar Tomer
@website: www.ambhas.com
@email: satkumartomer@gmail.com

This is a test script for the copulalib

"""


# import required library
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
# import copulalib
from copulalib.copulalib import Copula
from scipy.stats import rankdata

from ipdb import set_trace
from pdf import PDF

def load_data(id_):
    data = pd.read_csv("data/ra_index_nav.csv", index_col=["ra_date"], parse_dates=["ra_date"])
    data = data[data.ra_index_id == id_]
    return data

if __name__ == '__main__':
    sh300 = load_data(120000001).loc[:, ['ra_nav']].sort_index().pct_change()
    zz500 = load_data(120000002).loc[:, ['ra_nav']].sort_index().pct_change()

    df = pd.merge(sh300, zz500, left_index=True, right_index=True)
    df.columns = ['sh300', 'zz500']
    df = df.replace(0, np.nan).dropna()

    # generate random (normal distributed) numbers
    # SIZE = 100
    # x = np.random.normal(size=SIZE)
    # y = 2.5*x+ np.random.normal(size=SIZE)

    x = df.sh300.values
    y = df.zz500.values

    LIMIT = 100   
    x = rankdata(x)[-LIMIT:]/len(x)
    y = rankdata(y)[-LIMIT:]/len(y)
    # x = x[-LIMIT:]
    # y = y[-LIMIT:]

    # make the instance of Copula class with x, y and clayton family
    # foo_clayton = Copula(x, y, family='clayton')
    # X, Y = foo_clayton.generate_uv(1000)
    foo_frank = Copula(x, y, family='frank')
    X, Y = foo_frank.generate_uv(10000)
    # foo_gumbel = Copula(x, y, family='gumbel')
    # X, Y = foo_gumbel.generate_uv(1000)

    # plt.scatter(x, y)
    # plt.savefig('ori_data.pdf')
    # plt.scatter(X, Y)
    # plt.savefig('gen_data.pdf')

    pdf = PDF(X, Y, nbins = 20, plot = True) 
    copula_rho = [pdf(x[i], y[i]) for i in range(len(x))]
    plt.plot(range(len(copula_rho)), copula_rho)
    plt.savefig('copula_rho.pdf')
    # set_trace()
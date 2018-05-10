# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathos.multiprocessing import ProcessingPool as Pool
#  import multiprocessing
from Queue import Queue
import RiskMgrVaRs
from TimingGFTD import TimingGFTD
import DFUtil
from db import *
import warnings
import random
import os
import sys
import pickle
import os


import rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri
from scipy.stats.stats import pearsonr

warnings.filterwarnings("ignore")
r = ro.r
numpy2ri.activate()
pandas2ri.activate()

#Definitions of GARCH Model and functions in R to execute

r('''
require(rugarch)

garchOrder <- c(1,1)
armaOrder <- c(1,0)
varModel <- list(model="gjrGARCH", garchOrder = garchOrder)
spec <- ugarchspec(varModel, mean.model = list(armaOrder = armaOrder), distribution.model="norm")
funcfit <- function(x)(ugarchfit(x, spec=spec, solver="hybrid"))
funcsim <- function(fit, n)(fitted(ugarchsim(fit, n.sim=n, m.sim=1, startMethod = 'unconditional', rseed=sample(10^5, 1))))
funcpath <- function(fit, n, seed){
    fixed.p <- as.list(coef(fit))
    pspec <- ugarchspec(varModel, mean.model = list(armaOrder = armaOrder), distribution.model="norm", fixed.pars = fixed.p)
    X <- ugarchpath(pspec, n.sim=n, m.sim=1, rseed=seed)
    return(X)
}
funcmu <- function(fit)(coef(fit)['mu'])
''')

def get_seed(id_):
    tdate = base_trade_dates.load_origin_index_trade_date(id_)
    nav = database.load_nav_series(id_, reindex=tdate)
    initial_idx = random.randint(0, nav.size-600)
    nav_part = nav[initial_idx:initial_idx+600]
    inc = np.log(1+nav_part.pct_change().fillna(0))*100
    return inc

#  def gen_simulation(n, id_):
    #  seed = get_seed(id_).values
    #  for i in range(n):
        #  fit = r.funcfit(seed)
        #  simulation = np.array(r.funcsim(fit, 5)).flatten()
        #  seed = np.concatenate((seed, simulation))
    #  new_nav = np.exp((seed/100).cumsum())
    #  return new_nav

#  def gen_simulation_path(n, id_):
    #  seed = get_seed(id_).values
    #  path = np.array(r.funcpath(seed, n)).flatten()
    #  new_nav = np.exp((path/100).cumsum())
    #  return new_nav
def gen_simulation_with_seed(n, seed):
    fit = r.funcfit(seed)
    path = r.funcpath(fit, n, np.random.randint(2**32-1))
    #  sigma = np.array(r.sigma(path)).flatten()
    sigma = np.array(r.sigma(path)).ravel()
    #  seq = [stats.norm.rvs(0, i) for i in sigma]
    seq = [t_rvs(5, loc=0, scale=i, randseed=np.random.randint(2**32-1)) for i in sigma]
    seq.insert(0,0)
    seq = np.array(seq)
    new_nav = np.exp((seq/100).cumsum())
    return new_nav


def gen_simulation(n, id_):
    seed = get_seed(id_).values
    return gen_simulation_with_seed(n, seed)


#  def corr(verification, seq):
    #  return pearsonr(*map(lambda x:np.diff(np.log(x), 1), [verification, seq]))[0]

#  def parallel_gen(n, seed):
    #  count = multiprocessing.cpu_count() / 2
    #  pool = multiprocessing.Pool(processes = count)
    #  q = Queue()
    #  for i in range(n):
        #  q.put(pool.apply_async(gen_simulation_with_seed, args=(3000, seed)))
    #  pool.close()
    #  pool.join()
    #  return q

def gen_helper(fit, seedsize, numprocess):
    path = r.funcpath(fit, seedsize, np.random.randint(2**16-1)+numprocess)
    sigma = np.array(r.sigma(path)).ravel()
    sequence = np.array([t_rvs(5, loc=0, scale=i, randseed=np.random.randint(2**16-1)+numprocess) for i in sigma])
    return sequence

def t_rvs(df, loc, scale, randseed):
    rv = stats.t(df, loc=loc, scale=scale)
    rv.random_state = np.random.RandomState(seed=randseed)
    return rv.rvs()

def parallel(n, seed):
    pool = Pool()
    fit = r.funcfit(seed)
    sequences = pool.map(lambda x: gen_helper(fit, seed.size, x), range(n))
    return sequences

def get_sh300():
    tdate = base_trade_dates.load_origin_index_trade_date('120000001')
    nav = database.load_nav_series('120000001', reindex=tdate)
    inc = np.log(1+nav.pct_change().fillna(0))*100
    return inc

def divide_seq(n, seq):
    size = len(seq)
    start_idxs = [size/n*i for i in range(n)]
    idxs = zip(start_idxs, start_idxs[1:]+[None])
    for idx in idxs:
        yield seq[slice(*idx)]

def save_tmp_file(n, seq, path):
    gen = divide_seq(n, seq)
    i = 0
    for subseq in gen:
        with open(os.path.join(path, 'cached_%d.pickle' % i), 'w') as f:
            pickle.dump(subseq, f)
            i+=1

def convert_to_ndarray(src, dst, n):
    for i in range(n):
        with open(os.path.join(src, 'cached_%d.pickle' % i)) as f:
            tmp = pickle.load(f)
        tmp = np.array(tmp)
        with open(os.path.join(dst, 'cached_%d.pickle' % i), 'w') as f:
            pickle.dump(tmp, f)

def get_corr(seed, n, path):
    corr = np.array([], dtype=np.float64)
    for i in range(n):
        with open(os.path.join(path, 'cached_%d.pickle' % i)) as f:
            print "Load file num: %d" % (i+1)
            tmp_seq = pickle.load(f)
            tmp_corr = np.array([pearsonr(seed, tmp_seq[i])[0] for i in range(len(tmp_seq))])
            corr = np.append(corr, tmp_corr)
    return corr


if __name__ == "__main__":
    seed = get_sh300().loc['2013-01-01':'2018-05-01'].values
    path = 'montecarlo-check-sh300/last_5yrs'
    for i in range(1000):
        seqs = parallel(1000, seed)
        seqs = np.array(seqs)
        with open(os.path.join(path, 'cached_%d.pickle' % i), 'w') as f:
            pickle.dump(seqs, f)
        print "Generated %dth file!" % i


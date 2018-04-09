
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathos.multiprocessing import ProcessingPool as Pool
import click
from scipy import linalg
from Queue import Queue
import multiprocessing
from ipdb import set_trace

#Import rpy2
import rpy2
import rpy2.robjects as ro
import rpy2.robjects.packages
from rpy2.robjects import pandas2ri, numpy2ri

#Load basic libs from R
r = ro.r
rugarch = ro.packages.importr('rugarch')
spd = ro.packages.importr('spd')
copula = ro.packages.importr('copula')
rmgarch = ro.packages.importr('rmgarch')
numpy2ri.activate()
pandas2ri.activate()

# ---------------------------------------------
# # Load datas by Python
# codes = {'sh300':'120000001', 'zz500':'120000002'}
# codes = {'sh300':'120000001', 'au9999':'120000014'}
# # codes = {'sp500': '120000013', 'hsi': '120000015'}
# tdates = {k: base_trade_dates.load_origin_index_trade_date(v) for k,v in codes.items()}
# tdate = reduce(lambda x,y: x[1].intersection(y[1]), tdates.items())
# df_nav = pd.DataFrame({k: database.load_nav_series(v, reindex=tdate) for k,v in codes.items()})
# df_inc = np.log(1+df_nav.pct_change()).fillna(0)*100
# df_inc5d = df_inc.rolling(5).sum().fillna(0)
#-----------------------------------------------
# -----------------------------------------
# Define R objects of GARCH
garchOrder = r('c(1,1)')
armaOrder = r('c(1,0)')
varModel = rpy2.robjects.ListVector({'garchOrder':garchOrder, 'model': 'gjrGARCH'})
armaModel = rpy2.robjects.ListVector({'armaOrder':armaOrder})
# spec = rugarch.ugarchspec(**{'variance.model': varModel, 'mean.model':armaModel, 'distribution.model':'std'})
spec = rugarch.ugarchspec(**{'variance.model': varModel, 'mean.model':armaModel, 'distribution.model':'norm'})
uspec = rugarch.multispec(r.replicate(2, spec))
spec1 = rmgarch.dccspec(uspec=uspec, dccOrder = r('c(1,1)'), distribution='mvnorm')
#----------------------------------------------

class RiskMgrMGARCH(object):
    def __init__(self):
        pass
    
    def perform_joint(self, df):
        #Calculate joint distribution
        status_res = {}
        mu_res = {}
        cov_res = {}

        count = multiprocessing.cpu_count() / 2
        process_adjust_indexs = [[] for i in range(0, count)]

        for i in range(600, len(df.index)):
            process_adjust_indexs[i % count].append(i)

        m = multiprocessing.Manager()
        q = m.Queue()
        processes = []

        for indexes in process_adjust_indexs:
            p = multiprocessing.Process(target = joint_status_days, args=(indexes, df, q))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        for m in range(q.qsize()):
            day, status, mu, cov = q.get()
            status_res[day] = status
            mu_res[day] = mu
            cov_res[day] = cov
        
        sr_result = pd.Series(status_res)
        return sr_result
        
    
    def perform_single(self, df):
        var2d_res = {}
        var3d_res = {}
        var5d_res = {}
        count = multiprocessing.cpu_count() / 2
        process_adjust_indexs = [[] for i in range(0, count)]

        for i in range(600, len(df.index)):
            process_adjust_indexs[i % count].append(i)

        m = multiprocessing.Manager()
        q = m.Queue()
        processes = []

        for indexes in process_adjust_indexs:
            p = multiprocessing.Process(target = var_days, args=(indexes, df, q))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        for m in range(q.qsize()):
            day, var2d, var3d, var5d = q.get()
            var2d_res[day] = var2d
            var3d_res[day] = var3d
            var5d_res[day] = var5d
        
        df_result = pd.DataFrame({'VaR2d': var2d_res, 'VaR3d': var3d_res, 'VaR5d': var5d_res})
        return df_result


def joint_status_days(days, df, q):
    for day in days:
        try: 
            calc_joint_status(day, df, q)
        except:
            pass

def calc_joint_status(i, df, q):
    day = df.index[i]
    df_used = df[i::-5][::-1].fillna(0)
    multf = rugarch.multifit(uspec, df_used)
    fit = rmgarch.dccfit(spec1, data=df_used, **{'fit.control':ro.ListVector({"eval.se":True}), 'fit':multf})
    # mu = np.array(r('function (x) (coef(x)[1,])')(multf))
    mu = np.array(rugarch.fitted(fit))[-1]
    # cov_tmp = rmgarch.rcov(fit)
    # cov = np.matrix(r('function (x) (x[, , dim(x)[3]])')(cov_tmp))
    cov = np.array(rmgarch.rcov(fit))[:,:,-1]
    today = df_used.iloc[-1].values
    if np.dot((today - mu).T, np.dot(linalg.inv(cov), (today - mu))) > -2 * np.log(0.01) and (today < 0).all():
        print day
        status = True
    else:
        status = False
    q.put((day, status, mu, cov))


def var_days(days, df, q):
    for day in days:
        calc_var(day, df, q)

def calc_var(i, df, q):
    day = df.index[i]
    sr_inc2d = df['inc2d'][i::-2][::-1].fillna(0)
    sr_inc3d = df['inc3d'][i::-3][::-1].fillna(0)
    sr_inc5d = df['inc5d'][i::-5][::-1].fillna(0)
    inc = [sr_inc2d, sr_inc3d, sr_inc5d]
    fit = map(lambda x: rugarch.ugarchfit(x, spec=spec, solver='hybrid'), inc)
    VaRs = map(genVaR, fit)
    q.put(tuple([day]+VaRs))
    

# def genVaR(fit):
#     mu = np.array(rugarch.fitted(fit)).T[0]
#     sigma =np.array(rugarch.sigma(fit)).T[0]
#     nu = r('function(x) coef(x)["shape"]')(fit)[0]
#     return stats.t.ppf(0.01, nu, mu[-1], sigma[-1])

def genVaR(fit):
    mu = np.array(rugarch.fitted(fit)).T[0]
    sigma = np.array(rugarch.sigma(fit)).T[0]
    return mu[-1] - 3 * sigma[-1]

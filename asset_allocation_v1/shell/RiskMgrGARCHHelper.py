
'''
This class stands for calculating VaRs for univariate case, and calculating if the joint distribution got
breakthough in multivariable case. It will be a helper for RiskMgrGARCH. It is isolated cuz it highly depends
on the R interactive shell introduced by RPy2.
'''

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
import warnings
warnings.filterwarnings('ignore')

#Import rpy2
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri, numpy2ri

#Load basic libs from R
warnings.filterwarnings("ignore")
r = ro.r
numpy2ri.activate()
pandas2ri.activate()

#Definitions of GARCH Model and functions in R to execute
'''
The reason why we hardcode R codes here is that, once we load packages directly
by RPy2, some bugs may be introduced. For instance, maybe for a specific date, if
we fitted the inc5d till that day directly by RPy2, it may fall into infinite loop,
but technically it works well in the R interactive environment.
'''

r('''
require(rugarch)
require(rmgarch)

garchOrder <- c(1,1)
armaOrder <- c(1,0)
varModel <- list(model="gjrGARCH", garchOrder = garchOrder)
spec <- ugarchspec(varModel, mean.model = list(armaOrder = armaOrder), distribution.model="norm")
uspec <- multispec(replicate(2, spec))
dspec <- dccspec(uspec=uspec, dccOrder = c(1,1), distribution='mvnorm')

last <- (function(x) x[length(x)])

garch <- function(sr) {
    fit <- ugarchfit(sr, spec=spec, solver='hybrid', solver.control=list(tol=1e-6))
    mu <- last(fitted(fit))
    sigma <- last(sigma(fit))
    return(c(mu, sigma))
}

mgarch <- function(df) {
    multf <- multifit(uspec, df, solver.control=list(tol=1e-6))
    fit <- dccfit(dspec, data=df, fit.control=list(eval.se=TRUE), solver.control=list(tol=1e-6), fit=multf)
    mu. <- fitted(fit)
    mu <- mu.[dim(mu.)[1],]
    cov. <- rcov(fit)
    cov <- cov.[, , dim(cov.)[3]]
    return(list(mu, cov))
}
''')


#Univariate part
def perform_single(df):
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

    df_result = pd.DataFrame({'var_2d': var2d_res, 'var_3d': var3d_res, 'var_5d': var5d_res})
    return df_result


def var_days(days, df, q):
    for day in days:
        calc_var(day, df, q)


def calc_var(i, df, q):
    day = df.index[i]
    sr_inc2d = df['inc2d'][i::-2][::-1].fillna(0)
    sr_inc3d = df['inc3d'][i::-3][::-1].fillna(0)
    sr_inc5d = df['inc5d'][i::-5][::-1].fillna(0)
    inc = [sr_inc2d, sr_inc3d, sr_inc5d]
    fit = map(r.garch, inc)
    VaRs = map(lambda x: genVaR(*x), fit)
    q.put(tuple([day]+VaRs))


def genVaR(mu, sigma):
    '''
    Here is the definition of the specific way we calculate the VaR from mu and sigma (in univariate case)
    For instance, if we use the Student's t distribution and we wanna have the alpha quantile as the VaR,
    we shall return `stats.t.ppf(alpha, nu, mu, sigma)`, where the parameters coming from the fitted GARCH model.

    Currently we use normal distribution, and the quantile is approx. 0.00135 (3 sigmas off the mu).
    '''
    return mu - 3 * sigma



#Multivariate part
def perform_joint(df):
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

def joint_status_days(days, df, q):
    '''
    Sometimes, the model cannot got fitted by a reasonable bunch of parameters.
    In this case, R may raise a RuntimeError denoting it is not able to get a fitted model.
    I did a try-catch here to ignore such a case, if that happened, I consider it is not a
    signal to trigger the joint risk control.
    '''
    for day in days:
        try:
            calc_joint_status(day, df, q)
        except rpy2.rinterface.RRuntimeError, error:
            pass
            # print 'Day: %s' % df.index[day].strftime('%Y/%m/%d')
            # print error
            # print

def calc_joint_status(i, df, q):
    day = df.index[i]
    df_used = df[i::-5][::-1].fillna(0)
    mu, cov = map(np.array, r.mgarch(df_used))
    #Cuz the type we retrieved from the R interactive environment is `Matrix`,
    #we have to flatten the (1,2) matrix to (0,2) vector.
    mu = mu.flatten()
    today = df_used.iloc[-1].values
            # In the following code, we calculated the *Mahalanobis distance* from x to mu, where x denotes today's actual compounded return.
            # It's like the generalized Z-score in multivariate senario.
            # if np.dot((today - mu), np.dot(linalg.inv(cov), (today - mu))) > -2 * np.log(0.01) and (today < 0).all():
    # *Actually* right now we figured that the Mahalanobis distance method may not be vaild for calculating if the VaR got breakthrough.
    # The reason is that, it is more likely to be the extension of *confidence interval* under multivariable case instead of just quantile.
    # Here we use a more intuitive way: just calculate the cdf of corresponding vector and see if its value is larger than the threshold.
    # It is basically similar to comparing `today` and quantile(0.0013, mu, sigma) in univariable case.
    if stats.multivariate_normal.cdf(today, mean=mu, cov=cov) < 0.0013:
        status = 1
    else:
        status = 0
    q.put((day, status, mu, cov))



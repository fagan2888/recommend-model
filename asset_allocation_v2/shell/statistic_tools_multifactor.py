# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 14:33:12 2019

@author: Boyang ZHOU
@editor: Shixun Su
"""

import sys
import logging
# import matplotlib.pyplot as plt
# import pylab
import scipy as sp
# from scipy.stats import norm, poisson, gaussian_kde
# from scipy.special import psi
import numpy as np
import pandas as pd
import statsmodels as sm
from statsmodels.sandbox.regression.gmm import GMM
from statsmodels.stats.diagnostic import unitroot_adf
# from statsmodels.stats.sandwich_covariance import cov_hac
# from statsmodels.datasets import grunfeld
# from statsmodels.multivariate import pca
# from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
# import linearmodels
from linearmodels import IV2SLS, IVLIML, IVGMM, IVGMMCUE, PanelOLS, BetweenOLS, IVSystemGMM
# from collections import Counter
# import random
# from sklearn import decomposition
# import time
# from datetime import timedelta, datetime
# from __future__ import division


logger = logging.getLogger(__name__)


def auto_corr_normality_multi_tests(residuals_regression,regressor):

    '''
    Statistical hypothesis test for residuals

    I: Autocorrelation Test

        Methodologies:

            Ljung Box Q Testi:

                H0: The data are independently distributed
                H1: Serial correlateion effect
                Pitfall: Small sample size; All regressors are "strictly exogenous"

            Durbin Watson Test:

                H0: No serial correlateion effect
                H1: (Only) First order autoregressive
                Pitfall: The data must show an autocorrelation process

            Breusch Godfrey Test:

                H0: No seial correlation effect
                H1: Any order up to p autoregessive

        Reference:

            Maddala (2001) "Introduction to Econometrics (3d edition), ch 6.7, and 13. 5 p 528.

    II: Normality test

        Methodologies:

            Jarque Bera Test:
                H0: A joint hypothesis of the skewness being zero and the excess kurtosis being zero
                H1: Serial correlation effect
                Pitfall: Test depends on the skewness and kurtosis of matching a normal distribution (central moments based):
                    The sensitivity of chi-squared approximation of small sample size is high

            Kolmogorov Smirnov Test:

                H0: A good fit of the empirical distribution and the theoretical cdf
                H1: Alternative H0
                Pitfall: The distributions considered under the null hypothesis are continuous distributions but are otherwise unrestricted

            Anderson Darling Test:

                H0: A good fit of the empirical distribution and the theoretical cdf
                H1: Alternative H0
                Pitfall: Essentially the same test statistic can be used in the test of fit of a family of distributions,
                    but then it must be compared against the critical values appropriate to that family of theoretical distributions
                    and dependent also on the method used for parameter estimation.

            Reference:

                Jarque, Carlos M.; Bera, Anil K. (1987). "A test for normality of observations and regression residuals". International Statistical Review. 55 (2): 163–172.
                Scholz, F. W.; Stephens, M. A. (1987). "K-sample Anderson–Darling Tests". Journal of the American Statistical Association. 82 (399): 918–924.

    III: Homoskedasticity test

        Methodologies:

            Breusch Pagan Test:

            Goldfeld Quandt Test:

    '''

    # First I selected the first of 40 lags for Ljungbox
    A1 = sm.stats.diagnostic.acorr_ljungbox(residuals_regression)[1][0]
    A2 = sm.stats.stattools.durbin_watson(residuals_regression)

    # Normality test
    B1 = sm.stats.stattools.jarque_bera(residuals_regression,axis=0)[1]
    B2 = sm.stats.diagnostic.kstest_normal(residuals_regression,dist='norm', pvalmethod='approx')[0]
    B3 = sm.stats.diagnostic.normal_ad(residuals_regression, axis=0)[1]
    # QQ plot
    # sp.stats.probplot(residuals_regression,dist='norm', plot=pylab)
    # plt.show()

    # Homoskedasticity
    C1 = sm.stats.diagnostic.het_breuschpagan(residuals_regression, regressor)[3]
    C2 = sm.stats.diagnostic.het_goldfeldquandt(residuals_regression, regressor)[0]

    summary = pd.DataFrame({
            'Ljung Box': [A1],
            'Durbin Watson': [A2],
            'Jarque Bera': [B1],
            'Komlogorov Smirnov': [B2],
            'Anderson Darling': [B3],
            'Breusch Pagan': [C1],
            'Goldfeld Quandt': [C2]},
            index=['P value like'])

    print(summary)

    return summary

'''Regression Type I: OLS'''
def OLS_compare_summary(X,Y):

    X = pd.DataFrame(X)
    X = X[X.index.isin(Y.index)]
    Y = Y[Y.index.isin(X.index)]

    OLS1 = sm.regression.linear_model.OLS(X, Y).fit()
    OLS1.summary()
    OLS1_resid = OLS1.resid
    # 40 lags for Ljungbox
    sm.stats.diagnostic.acorr_ljungbox(OLS1_resid)
    sm.stats.stattools.durbin_watson(OLS1_resid)
    sm.stats.diagnostic.acorr_breusch_godfrey(OLS1)[2]
    # QQ plop
    # sp.stats.probplot(OLS1_resid, dist='norm', plot=pylab)
    # Normality test
    sm.stats.stattools.jarque_bera(OLS1_resid,axis=0)[1]
    sm.stats.diagnostic.kstest_normal(OLS1_resid,dist='norm', pvalmethod='approx')[0]
    sm.stats.diagnostic.normal_ad(OLS1_resid, axis=0)[1]

    '''Homoskedastic estimator'''
    print('OLS Homoskedastic estimator')
    OLS1 = IV2SLS(X, Y, None, None).fit(cov_type='unadjusted')
    print(OLS1.summary)
    # OLS1_resid = OLS1._resid
    auto_corr_normality_multi_tests(OLS1._resid,X)

    '''Robust to heteroskedasticity'''
    print('OLS Robust to heteroskedasticity')
    OLS1 = IV2SLS(X, Y, None, None).fit(cov_type='robust')
    print(OLS1.summary)
    auto_corr_normality_multi_tests(OLS1._resid,X)

    '''Clustering to account for additional sources of dependence between the model scores'''
    print('OLS Cluster')
    OLS1 = IV2SLS(X, Y, None, None).fit(cov_type='clustered')
    print(OLS1.summary)
    auto_corr_normality_multi_tests(OLS1._resid,X)

    '''A HAC robust covariance estimator'''
    print('OLS HAC')
    OLS1 = IV2SLS(X, Y, None, None).fit(cov_type='kernel')
    print(OLS1.summary)
    auto_corr_normality_multi_tests(OLS1._resid,X)

    return

'''Regression Type II: (F)GLS'''
def GLS_compare_summary(Y, X):
    X = pd.DataFrame(X)
    Y = pd.DataFrame(Y)
    X = X[X.index.isin(Y.index)]
    Y = Y[Y.index.isin(X.index)]

    '''Got the weighted sigma from OLS regression'''
    # TODO: Any solid methodology for HAC estimator instead of fit the OLS model in an arbitrage way?
    '''
    HC0_se

        White's (1980) heteroskedasticity rost standard errors.
        Defined as sqrt(diag(X.T X)^(-1)X.T diag(e_i^(2)) X(X.T X)^(-1)
        where e_i = resid[i]bu
        HC0_se is a cached property.
        When HC0_se or cov_HC0 is called the RegressionResults instance will
        then have another attribute `het_scale`, which is in this case is just
        resid**2.

    HC1_se

        MacKinnon and White's (1985) alternative heteroskedasticity robust
        standard errors.
        Defined as sqrt(diag(n/(n-p)*HC_0)
        HC1_see is a cached property.
        When HC1_se or cov_HC1 is called the RegressionResults instance will
        then have another attribute `het_scale`, which is in this case is
        n/(n-p)*resid**2.

    HC2_se

        MacKinnon and White's (1985) alternative heteroskedasticity robust
        standard errors.
        Defined as (X.T X)^(-1)X.T diag(e_i^(2)/(1-h_ii)) X(X.T X)^(-1)
        where h_ii = x_i(X.T X)^(-1)x_i.T
        HC2_see is a cached property.
        When HC2_se or cov_HC2 is called the RegressionResults instance will
        then have another attribute `het_scale`, which is in this case is
        resid^(2)/(1-h_ii).

    HC3_se

        MacKinnon and White's (1985) alternative heteroskedasticity robust
        standard errors.
        Defined as (X.T X)^(-1)X.T diag(e_i^(2)/(1-h_ii)^(2)) X(X.T X)^(-1)
        where h_ii = x_i(X.T X)^(-1)x_i.T
        HC3_see is a cached property.
        When HC3_se or cov_HC3 is called the RegressionResults instance will
        then have another attribute `het_scale`, which is in this case is
        resid^(2)/(1-h_ii)^(2).

    Reference: help(statsmodels.regression.linear_model.get_robustcov_results)
    '''
    OLS_resid = sm.regression.linear_model.OLS(Y, X).fit(cov_type='HC1').resid
    rho = IV2SLS(OLS_resid[1:],OLS_resid[:-1], None, None).fit().params[0]
    sigma = rho ** sp.linalg.toeplitz(np.arange(len(OLS_resid)))


    GLS = sm.regression.linear_model.GLS(Y, X, sigma)
    # TODO: The same as statsmodels.regression.linear_model.RegressionResults.get_robustcov_results
    GLS_results = GLS.fit(cov_type='HAC', use_t=True, cov_kwds={'maxlags': 1, 'kernel': 'bartlett'})
    print(GLS_results.summary())
    Hypothesis_Test_Summary = auto_corr_normality_multi_tests(GLS_results._results.resid, Y)

    return GLS_results, Hypothesis_Test_Summary


if __name__ == '__main__':

    # OLS_compare_summary(Ret_indexing, PS_ratio.mean(axis=1))
    # GLS_compare_summary(data['H500'], data['500_Quality'])[1]
    pass


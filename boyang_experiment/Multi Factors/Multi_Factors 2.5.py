# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 14:33:12 2019

@author: Boyang ZHOU
"""
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, poisson, gaussian_kde
from collections import Counter
import pylab
import statsmodels as sm
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
import seaborn.apionly as sns
import random
import time
from datetime import timedelta, datetime

from statsmodels.stats.diagnostic import unitroot_adf
from statsmodels.multivariate import pca
from sklearn import decomposition
from statsmodels.stats.sandwich_covariance import cov_hac


import linearmodels
from statsmodels.datasets import grunfeld
from linearmodels import IV2SLS, IVLIML, IVGMM, IVGMMCUE,PanelOLS,BetweenOLS,IVSystemGMM

#from __future__ import division
from scipy.special import psi
from statsmodels.sandbox.regression.gmm import GMM

###########################################################

def Weighted_by_factors(factor):
    Weight_matrix=factor.divide(factor.sum(axis=1).values,axis=0)
    return Weight_matrix

def Single_factor_retrace(factor,Return,Ret_indexing):
#    The Ret_indexing is the mkt cap weighted benchmark or index benchmark
#    The Return is the time series returns of the underlyings
#    The factor equally weighted by each underlyings
    Factor_weighted_portfolio=pd.DataFrame(np.sum(Return*Weighted_by_factors(factor),axis=1)).dropna()
    Compare_result=pd.merge(Ret_indexing,Factor_weighted_portfolio,right_index=True,left_index=True,how='inner')
    Compare_result.columns=['Mkt Weight','Factor Weight']
    Compare_result.cumsum().plot()
    plt.show()
#    return Compare_result

def Normalization_factor(factor_df):
#    Ref: recommend_model/asset_allocation2/shell/stock_factor.py
    factor_df = factor_df.fillna(np.nan)
#    Null/Na data tolerance level sets as 20% here
    if factor_df.isnull().sum()/factor_df.shape[0]>0.2:
        print('Garbage data')
        
    factor_median = factor_df.median(axis = 1)

    factor_df_sub_median = abs(factor_df.sub(factor_median, axis = 0))
    factor_df_sub_median_median = factor_df_sub_median.median(axis = 1)

    max_factor_df = factor_median + 10.000 * factor_df_sub_median_median
    min_factor_df = factor_median - 10.000 * factor_df_sub_median_median

    stock_num = len(factor_df.columns)
    stock_ids = factor_df.columns
    max_factor_df = pd.concat([max_factor_df]*stock_num, 1)
    min_factor_df = pd.concat([min_factor_df]*stock_num, 1)
    max_factor_df.columns = stock_ids
    min_factor_df.columns = stock_ids

    factor_df = factor_df.mask(factor_df < min_factor_df, min_factor_df)
    factor_df = factor_df.mask(factor_df > max_factor_df, max_factor_df)
  
    factor_std  = factor_df.std(axis = 1)
    factor_mean  = factor_df.mean(axis = 1)

    factor_df = factor_df.sub(factor_mean, axis = 0)
    factor_df = factor_df.div(factor_std, axis = 0)

    return factor_df

# In[0]:
###########################################################
'Example Data'
stock_data = pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\Multi Factors\data\cls_px.csv",index_col=[0])

stock_data.index = stock_data.index.map(lambda x: pd.Timestamp(str(int(x))))
stock_data=stock_data.apply(lambda x: np.nan if type(x)==float else np.float64(x))
stock_data.groupby(stock_data.index.strftime('%Y-%m')).last()



stock_mkt_cap=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\Multi Factors\data\mkt_cap.csv",index_col=[0])
stock_mkt_cap.index = stock_mkt_cap.index.map(lambda x: pd.Timestamp(str(int(x))))

PE_ratio=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\Multi Factors\data\PE_ratio.csv",index_col=[0])
PE_ratio.index = PE_ratio.index.map(lambda x: pd.Timestamp(str(int(x))))

PB_ratio=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\Multi Factors\data\PB_ratio.csv",index_col=[0])
PB_ratio.index = PB_ratio.index.map(lambda x: pd.Timestamp(str(int(x))))
PB_ratio=PB_ratio[PB_ratio.index.isin(stock_data.index)]


PS_ratio=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\Multi Factors\data\PSales_ratio.csv",index_col=[0])
PS_ratio.index=PS_ratio.index.map(lambda x: pd.Timestamp(str(int(x))))
PS_ratio=PS_ratio[PS_ratio.index.isin(stock_data.index)]

USDCNY=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\Multi Factors\data\usdcny.csv",index_col=[0])
USDCNY.index=USDCNY.index.map(lambda x: pd.Timestamp(str(x)))
#USDCNY=USDCNY[USDCNY.index.isin(stock_data.index)]
Fin_data=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\Multi Factors\data\caihui_tq_fin_procfsqsubjects_50.csv",index_col=[0])
Fin_data.PUBLISHDATE=Fin_data.FIRSTPUBLISHDATE.map(lambda x: pd.Timestamp(str(x)))


start_date=pd.Timestamp('2004-01-01')
stock_data=stock_data[stock_data.index>=start_date]
stock_mkt_cap=stock_mkt_cap[stock_mkt_cap.index>=start_date]
PE_ratio=PE_ratio[PE_ratio.index>=start_date]
PB_ratio=PB_ratio[PB_ratio>=start_date]

PS_ratio=PS_ratio[PS_ratio.index>=start_date]
Fin_data=Fin_data[Fin_data.PUBLISHDATE>=start_date]
#PE_ratio=PE_ratio.applymap(lambda x: 1000 if type(x)==np.str else np.float64(x))
PE_ratio=PE_ratio.replace('--',np.nan)
PS_ratio=PS_ratio.replace('--',np.nan)
'filter nan'
#PE_ratio.isnull().sum()/PE_ratio.shape[0]
#print(Counter(Fin_data.isnull().sum()/Fin_data.shape[0]).most_common()[0])
#print(Counter(stock_mkt_cap.isnull().sum()/stock_mkt_cap.shape[0]).most_common()[0])
#print(Counter(PE_ratio.isnull().sum()/PE_ratio.shape[0]).most_common()[0])

Null_NB=Fin_data.isnull().sum()

valid_columns=[stock_data.columns[i] if stock_data.iloc[:,i].isnull().sum()/stock_data.shape[0] <0.1 else np.nan for i in range(stock_data.shape[1])]
valid_columns=[x for x in valid_columns if x==x]

stock_data=stock_data[valid_columns].ffill()
stock_mkt_cap=stock_mkt_cap[valid_columns].ffill()
PE_ratio=PE_ratio[valid_columns].ffill()
PS_ratio=PS_ratio[valid_columns].ffill()
PB_ratio=PB_ratio[valid_columns].ffill()


PE_ratio=PE_ratio.apply(lambda x:np.float64(x))
PS_ratio=PS_ratio.apply(lambda x:np.float64(x))
PB_ratio=PB_ratio.apply(lambda x:np.float64(x))

'Arithmetic return instead of logarithm return according to FMB93'
Return=stock_data.pct_change()
Return=Return.dropna()

Test_individual_underlying=Fin_data[Fin_data.Ticker==Return.columns[1]]

#'HAC cov'
#Covariance=sm.stats.sandwich_covariance.cov_hac_simple(Return.cov())
#

# In[1]:
'Mkt Cap weighted index'
Indexing=np.sum(Return*(stock_mkt_cap.divide(stock_mkt_cap.sum(axis=1).values,axis=0)),axis=1).fillna(value=0).add(1,axis=0).cumprod()
Ret_indexing=pd.DataFrame(Indexing,columns=['Indexing']).pct_change().dropna()
Mkt_weight=stock_mkt_cap.divide(stock_mkt_cap.sum(axis=1).values,axis=0)
stock_mkt_cap_log=np.log(stock_mkt_cap)

######################################################################################################################
######################################################################################################################

def Auto_Corr_Normality_multi_tests(residuals_regression,regressor):
    
    '''
    Statistical hypothesis test for residuals
    
    I: Autocorrelation Test
    
    Methodologies:
        
            Ljung Box Q Test: 
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
               H1: Serial correlateion effect
               Pitfall: Test depends on the skewness and kurtosis of matching a normal distribution (central moments based): The sensitivity of chi-squared approximation of small sample size is high  

            Kolmogorov Smirnov Test:
               H0: A good fit of the empirical distribution and the theoretical cdf
               H1: Alternative H0
               Pitfall: The distributions considered under the null hypothesis are continuous distributions but are otherwise unrestricted

            Anderson Darling Test:
               H0: A good fit of the empirical distribution and the theoretical cdf
               H1: Alternative H0
               Pitfall: Essentially the same test statistic can be used in the test of fit of a family of distributions, but then it must be compared against the critical values appropriate to that family of theoretical distributions and dependent also on the method used for parameter estimation.
         
    Reference:
    Jarque, Carlos M.; Bera, Anil K. (1987). "A test for normality of observations and regression residuals". International Statistical Review. 55 (2): 163–172
    Scholz, F. W.; Stephens, M. A. (1987). "K-sample Anderson–Darling Tests". Journal of the American Statistical Association. 82 (399): 918–924.        
 
    
    III: Homoskedasticity test
    
    Methodologies:
        
            Breusch Pagan Test: 
               H0: Homoscedasticity 
               H1: Heteroscedasticity
               Pitfall: 
                   
            Goldfeld Quandt Test:
               H0: Homoscedasticity 
               H1: Heteroscedasticity
               Pitfall: Not very robust to specification errors, where it detects non-homoskedasticity errors but cannot distinguish between heteroskedasticity error structure and an underlying specification problem.
            
    
    Reference:
    Thursby, Jerry (May 1982). "Misspecification, Heteroscedasticity, and the Chow and Goldfeld-Quandt Tests". The Review of Economics and Statistics. 64 (2): 314–321.
    '''  
      
    #40 lags for Ljungbox
    A1=sm.stats.diagnostic.acorr_ljungbox(residuals_regression)[1]
    A2=sm.stats.stattools.durbin_watson(residuals_regression)

    #Normality test
    B1=sm.stats.stattools.jarque_bera(residuals_regression,axis=0)[1]
    B2=sm.stats.diagnostic.kstest_normal(residuals_regression,dist='norm',pvalmethod='approx')[0]
    B3=sm.stats.diagnostic.normal_ad(residuals_regression, axis=0)[1]
    #QQ plop
    sp.stats.probplot(residuals_regression,dist='norm',plot=pylab)
    plt.show()    
    
    #Homoskedasticity
    C1=sm.stats.diagnostic.het_breuschpagan(residuals_regression,regressor)[3]
    C2=sm.stats.diagnostic.het_goldfeldquandt(residuals_regression,regressor)[0]
    
    summary=pd.DataFrame({
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
    
##########################
# In[2]:
'''
Test I: Different Regression Model Comparision'
        
        Portfolio: Full filtered dataset
        Factors: PE_ratio
        Factors type: Single factor
        Factors Joint Methodology: Equally Weighted
        
'''

'Regression Type I: OLS'
def OLS_compare_summary(X,Y):
    
    X=pd.DataFrame(X)
    X=X[X.index.isin(Y.index)]
    Y=Y[Y.index.isin(X.index)]

    OLS1=sm.regression.linear_model.OLS(X,Y).fit()
    OLS1.summary()
    OLS1_resid=OLS1.resid
    #40 lags for Ljungbox
    sm.stats.diagnostic.acorr_ljungbox(OLS1_resid)
    sm.stats.stattools.durbin_watson(OLS1_resid)
    sm.stats.diagnostic.acorr_breusch_godfrey(OLS1)[2]
    #QQ plop
    sp.stats.probplot(OLS1_resid,dist='norm',plot=pylab)
    #Normality test
    sm.stats.stattools.jarque_bera(OLS1_resid,axis=0)[1]
    sm.stats.diagnostic.kstest_normal(OLS1_resid,dist='norm',pvalmethod='approx')[0]
    sm.stats.diagnostic.normal_ad(OLS1_resid, axis=0)[1]
    
    
    'Homoskedastic estimator'
    print('OLS Homoskedastic estimator')
    OLS1=IV2SLS(X,Y,None,None).fit(cov_type='unadjusted')
    print(OLS1.summary)
#    OLS1_resid=OLS1._resid
    Auto_Corr_Normality_multi_tests(OLS1._resid,X)
    
    'Robust to heteroskedasticity'
    print('OLS Robust to heteroskedasticity')
    OLS1=IV2SLS(X,Y,None,None).fit(cov_type='robust')
    print(OLS1.summary)
    Auto_Corr_Normality_multi_tests(OLS1._resid,X)
    
    'Clustering to account for additional sources of dependence between the model scores'
    print('OLS Cluster')
    OLS1=IV2SLS(X,Y,None,None).fit(cov_type='clustered')
    print(OLS1.summary)
    Auto_Corr_Normality_multi_tests(OLS1._resid,X)
    
    'A HAC robust covariance estimator'
    print('OLS HAC')
    OLS1=IV2SLS(X,Y,None,None).fit(cov_type='kernel')
    print(OLS1.summary)
    Auto_Corr_Normality_multi_tests(OLS1._resid,X)


#OLS_compare_summary(Ret_indexing,PS_ratio.mean(axis=1))
#OLS_compare_summary(Ret_indexing,PE_ratio.mean(axis=1))
#OLS_compare_summary(Ret_indexing,stock_mkt_cap_log.mean(axis=1))


'Regression Type II: (F)GLS'
def GLS_compare_summary(Y,X):
    X=pd.DataFrame(X)
    Y=pd.DataFrame(Y)
    X=X[X.index.isin(Y.index)]
    Y=Y[Y.index.isin(X.index)]
    
    'Got the weighted sigma from OLS regression'
    # TODO: Any solid methodology for HAC estimator instead of fit the OLS model in an arbitrage way?
    OLS_resid=sm.regression.linear_model.OLS(Y,X).fit(cov_type='HC1').resid
    rho=IV2SLS(OLS_resid[1:],OLS_resid[:-1], None, None).fit().params[0]
    sigma=rho**sp.linalg.toeplitz(np.arange(len(OLS_resid)))
    
    
    GLS=sm.regression.linear_model.GLS(Y,X,sigma)
    GLS_results=GLS.fit()
    print(GLS_results.summary())
    Auto_Corr_Normality_multi_tests(GLS_results._results.resid,Y)

    return GLS_results


#GLS_compare_summary(Ret_indexing,PS_ratio.mean(axis=1))
#GLS_compare_summary(Ret_indexing,PE_ratio.mean(axis=1))
#GLS_compare_summary(Ret_indexing,stock_mkt_cap_log.mean(axis=1))

'Regression Type III: GMM'

'''
Reference: Ravi J.; Georgios S.; Zhenyu W. (2010) "The Analysis of the Cross-Section of Security Returns"
           Hansen, Lars Peter (1982). "Large Sample Properties of Generalized Method of Moments Estimators". Econometrica. 50 (4): 1029–1054.
'''

class GMMGamma(GMM):

    def __init__(self, *args, **kwds):
        # set appropriate counts for moment conditions and parameters
        # TODO: Finish the no exog and instruments GMM
        kwds.setdefault('k_moms', 4)
        kwds.setdefault('k_params', 2)
        super(GMMGamma, self).__init__(*args, **kwds)


    def momcond(self, params):
        p0, p1 = params
        endog = self.endog
        error1 = endog - p0 / p1
        error2 = endog**2 - (p0 + 1) * p0 / p1**2
        error3 = 1 / endog - p1 / (p0 - 1)
        error4 = np.log(endog) + np.log(p1) - psi(p0)
        g = np.column_stack((error1, error2, error3, error4))
        return g


##model = GMMGamma(Ret_indexing, PS_ratio[PS_ratio.index.isin(Ret_indexing.index)].mean(axis=1), None)
#model = GMMGamma(Ret_indexing, PE_ratio[PE_ratio.index.isin(Ret_indexing.index)], None)
#
#beta0 = np.array([-2,-2])
##res = model.fit(beta0, maxiter=2, weights_method='cov', optim_method='nm', wargs=dict(centered=False))
#res = model.fit(beta0, maxiter=0, optim_method='nm', wargs=dict(centered=False))
#
#print(res.summary())
#

# In[3]:
'''
Numerical Factors
'''

Month12std=Return.rolling(252).std().dropna()

MomentumR12=Return.rolling(252).sum().dropna()
MomentumR6=Return.rolling(int(252/2)).sum().dropna()
MomentumR1=Return.rolling(int(252/12)).sum().dropna()

Single_factor_retrace(PB_ratio,Return,Ret_indexing)

def Momentum(stock_return,R,H):
#    windows in month: Rolling Period as R Holding Period as H
    R=int(R*252/12)
    H=int(H*252/12)
    
    Momentum1=stock_return.rolling(R).sum().dropna()
#    Equally weighted L-S strategy at 30% rank quantile of portfolio 
    Nb_stock=int(stock_return.shape[1]*0.3)
#    set the left column values as 0.0001 just for further re-scale the weight matrix, and I can set it to any number if I want to set.
    Weight=Momentum1.rank(ascending=False,axis=1).applymap(lambda x: 1/Nb_stock if x<=Nb_stock else (-1/Nb_stock if x>=stock_return.shape[1]-Nb_stock else 0.0001))
    Weight_index=Weight.index.isin([Weight.index[0+i*H] for i in range(Weight.shape[0]//H)])*1
    Weight_index=pd.DataFrame(data=np.repeat(list(Weight_index),Weight.shape[1]).reshape(len(Weight_index),Weight.shape[1]),index=Weight.index,columns=Weight.columns)
    
    Weight_index=Weight*Weight_index
    Weight_index=Weight_index.applymap(lambda x: float('Nan') if x==0 else x)
#    Now I delete the 0.0001 or any other number here 
    Weight_index=Weight_index.fillna(method='ffill').applymap(lambda x: 0 if x ==0.0001 else x)
    Mometum_portfolio=Weight_index*Return
    Mometum_portfolio=Mometum_portfolio.dropna()
   
    return Mometum_portfolio
    

Momentum_test=Momentum(Return,12,3)

#
#Single_factor_retrace(Month12std,Return,Ret_indexing)
#Single_factor_retrace(PE_ratio,Return,Ret_indexing)
#Single_factor_retrace(np.log(stock_mkt_cap),Return,Ret_indexing)
#Single_factor_retrace(Momentum_test,Return,Ret_indexing)
#Single_factor_retrace(PB_ratio,Return,Ret_indexing)
#
#GLS_compare_summary(Ret_indexing,Weighted_by_factors(Month12std).mean(axis=1))

#GLS_compare_summary(Ret_indexing,PB_ratio.mean(axis=1))
#GLS_compare_summary(Ret_indexing,Momentum_test.mean(axis=1))
#GLS_compare_summary(Ret_indexing,PS_ratio.mean(axis=1))
#GLS_compare_summary(Ret_indexing,PB_ratio)
#GLS_compare_summary(Ret_indexing,PS_ratio)

Beta=Ret_indexing.cov()/Ret_indexing.std()**2

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
from linearmodels import IV2SLS, IVLIML, IVGMM, IVGMMCUE,PanelOLS,BetweenOLS
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

AA=Fin_data.isnull().sum()

valid_columns=[stock_data.columns[i] if stock_data.iloc[:,i].isnull().sum()/stock_data.shape[0] <0.1 else np.nan for i in range(stock_data.shape[1])]
valid_columns=[x for x in valid_columns if x==x]

stock_data=stock_data[valid_columns].ffill()
stock_mkt_cap=stock_mkt_cap[valid_columns].ffill()
PE_ratio=PE_ratio[valid_columns].ffill()
PS_ratio=PS_ratio[valid_columns].ffill()

PE_ratio=PE_ratio.apply(lambda x:np.float64(x))
PS_ratio=PS_ratio.apply(lambda x:np.float64(x))
'Arithmetic return instead of logarithm return according to FMB93'
Return=stock_data.pct_change()
Return=Return.dropna()

Test_individual_underlying=Fin_data[Fin_data.Ticker==Return.columns[1]]

#'HAC cov'
#Covariance=sm.stats.sandwich_covariance.cov_hac_simple(Return.cov())
#


'Mkt Cap weighted index'
Indexing=np.sum(Return*(stock_mkt_cap.divide(stock_mkt_cap.sum(axis=1).values,axis=0)),axis=1).fillna(value=0).add(1,axis=0).cumprod()
Ret_indexing=pd.DataFrame(Indexing,columns=['Indexing']).pct_change().dropna()
Mkt_weight=stock_mkt_cap.divide(stock_mkt_cap.sum(axis=1).values,axis=0)
###########################################################
'Example Factor'
stock_mkt_cap_log=np.log(stock_mkt_cap)

GLS=sm.regression.linear_model.GLS(Indexing,stock_data)
GLS_results=GLS.fit()
print(GLS_results.summary())



ivolsmod = IV2SLS(Indexing, stock_data, None, None)
res_ols = ivolsmod.fit()
print(res_ols)
res_ols._params
res_ols._resid
def stat_summary_compare_by_fit(Y,X):
    X=pd.DataFrame(X)
    X=X[X.index.isin(Y.index)]
    Y=Y[Y.index.isin(X.index)]
    
    ivolsmod = IV2SLS(Y, X, None, None)
    res_ols = ivolsmod.fit()
    print(res_ols)
    #res_ols.resids
    'unadjusted: the classic homoskedastic estimator'
    ivolsmod = IV2SLS(Y, X, None, None)
    res_ols = ivolsmod.fit(cov_type='unadjusted')
    print(res_ols)
    'robust: robust to heteroskedasticity'
    ivolsmod = IV2SLS(Y, X, None, None)
    res_ols = ivolsmod.fit(cov_type='robust')
    print(res_ols)
    'clustered: one- or two-way clustering to account for additional sources of dependence between the model scores'
    ivolsmod = IV2SLS(Y, X, None, None)
    res_ols = ivolsmod.fit(cov_type='clustered')
    print(res_ols)
    'kernel: a heteroskedasticity-autocorrelation robust covariance estimator'
    ivolsmod = IV2SLS(Y, X, None, None)
    res_ols = ivolsmod.fit(cov_type='kernel')
    print(res_ols)


#stat_summary_compare_by_fit(Ret_indexing,(Return*Mkt_weight).sum(axis=1))
#
#
#stat_summary_compare_by_fit(Ret_indexing,stock_mkt_cap_log)
#
#stat_summary_compare_by_fit(Ret_indexing,(PE_ratio*Mkt_weight).sum(axis=1))
#stat_summary_compare_by_fit(Ret_indexing,(PS_ratio*Mkt_weight).sum(axis=1))
#
#stat_summary_compare_by_fit(Ret_indexing,PS_ratio)
stat_summary_compare_by_fit(Ret_indexing,PE_ratio)
#
def stats_in_m(Y,X):
    X=pd.DataFrame(X)
    X=X[X.index.isin(Y.index)]
    Y=Y[Y.index.isin(X.index)]

    X=X.groupby(X.index.strftime('%Y-%m')).last()
    Y=Y.groupby(Y.index.strftime('%Y-%m')).last()
    ivolsmod = IV2SLS(Y, X, None, None)
    res_ols = ivolsmod.fit(cov_type='kernel')
    print(res_ols)
    return res_ols

stats_in_m(Ret_indexing,Return)
A=stats_in_m(Ret_indexing,USDCNY.pct_change())
A.tstats.values[0]
A._params.values[0]
A.resids



stat_summary_compare_by_fit(Ret_indexing,USDCNY.pct_change())

def stats_ts(Y,X):
    
    X=pd.DataFrame(X)
    Y=pd.DataFrame(Y)
    X=X[X.index.isin(Y.index)]
    Y=Y[Y.index.isin(X.index)]
    
#    ivolsmod = IV2SLS(Y, X, None, None)
#    res_ols = ivolsmod.fit(cov_type='kernel')
    tstats=[]
    _params=[]
    for i in range(Y.shape[0]-1000-1):
        
        ivolsmod = IV2SLS(Y.iloc[:1000+i,:], X.iloc[:1000+i,:], None, None)        
        res_ols = ivolsmod.fit(cov_type='kernel')
        
        tstats.append(res_ols.tstats.values[0])
        _params.append(res_ols._params.values[0])
        
        
    return tstats,_params


def GLS(Y,X):
#    new_index = X.index.intersect(Y.index)
#    X = X.reindex(new_index)
#    Y = Y.reindex(new_index)
    
    ##
#    Intersect_index=set(X.index.tolist(),Y.index.tolist())
    X=pd.DataFrame(X)
    Y=pd.DataFrame(Y)
    X=X[X.index.isin(Y.index)]
    Y=Y[Y.index.isin(X.index)]

    GLS=sm.regression.linear_model.GLS(Y,X)
    GLS_results=GLS.fit()
    print(GLS_results.summary())
    return GLS_results

#GLS(Ret_indexing,USDCNY.pct_change())

GLS_TEST=GLS(Ret_indexing,PS_ratio)
GLS_TEST_params=GLS_TEST._results.params
GLS_TEST_resid=GLS_TEST._results.resid
GLS_TEST_params.mean()

GLS(Ret_indexing,PS_ratio.mean(axis=1))



#df = pd.concat([USDCNY.pct_change(),Ret_indexing],1)
#df=df.dropna()
#df=df.groupby(df.index.strftime('%Y-%m')).last()
#plt.scatter(df['USDCNY'], df['Indexing'])
#

def indexed_data(Y,X):
    X=pd.DataFrame(X)
    Y=pd.DataFrame(Y)
    X=X[X.index.isin(Y.index)]
    Y=Y[Y.index.isin(X.index)]

    return X,Y


#linearmodels.asset_pricing.LinearFactorModelGMM(PE_ratio,PS_ratio).fit()
    
PanelOLS(Ret_indexing, pd.DataFrame(PS_ratio[PS_ratio.index.isin(Ret_indexing.index)].sum(axis=1)),entity_effects=True).fit(cov_type='clustered')

PanelOLS(Ret_indexing, pd.DataFrame(Return[Return.index.isin(Ret_indexing.index)].sum(axis=1))).fit(cov_type='unadjusted')

A=pd.merge(pd.DataFrame(Return[Return.index.isin(Ret_indexing.index)].sum(axis=1)), pd.DataFrame(PS_ratio[PS_ratio.index.isin(Ret_indexing.index)].sum(axis=1)),right_index=True,left_index=True,how='outer')

PanelOLS(Ret_indexing, A,entity_effects=True).fit(cov_type='unadjusted')



pd.concat(pd.DataFrame(PE_ratio[PE_ratio.index.isin(Ret_indexing.index)].sum(axis=1)),pd.DataFrame(PS_ratio[PS_ratio.index.isin(Ret_indexing.index)].sum(axis=1)))


np.sort(list(set(Fin_data.index)))[1]
Fin_data[Fin_data.index==np.sort(list(set(Fin_data.index)))[1]].ROEAVGPRE.isnull().sum()

linearmodels.asset_pricing.TradedFactorModel(Ret_indexing,pd.DataFrame(PE_ratio[PE_ratio.index.isin(Ret_indexing.index)].sum(axis=1))).fit()

#############
#def stat_summary_compare_by_fit(Y,X):
#
#    X=X[X.index.isin(Y.index)]
#    ivolsmod = IV2SLS(Y, X, None, None)
#    res_ols = ivolsmod.fit()
#    print(res_ols)
#    #res_ols.resids
#    'unadjusted: the classic homoskedastic estimator'
#    ivolsmod = IV2SLS(Y, X, None, None)
#    res_ols = ivolsmod.fit(cov_type='unadjusted')
#    print(res_ols)
#    'robust: robust to heteroskedasticity'
#    ivolsmod = IV2SLS(Y, X, None, None)
#    res_ols = ivolsmod.fit(cov_type='robust')
#    print(res_ols)
#    'clustered: one- or two-way clustering to account for additional sources of dependence between the model scores'
#    ivolsmod = IV2SLS(Y, X, None, None)
#    res_ols = ivolsmod.fit(cov_type='clustered')
#    print(res_ols)
#    'kernel: a heteroskedasticity-autocorrelation robust covariance estimator'
#    ivolsmod = IV2SLS(Y, X, None, None)
#    res_ols = ivolsmod.fit(cov_type='kernel')
#    print(res_ols)
#
#
#stat_summary_compare_by_fit(Ret_indexing,(Return*Mkt_weight).sum(axis=1))
#
#
#stat_summary_compare_by_fit(Ret_indexing,stock_mkt_cap_log)
#
#stat_summary_compare_by_fit(Ret_indexing,(PE_ratio*Mkt_weight).sum(axis=1))
#stat_summary_compare_by_fit(Ret_indexing,(PS_ratio*Mkt_weight).sum(axis=1))
#
#stat_summary_compare_by_fit(Ret_indexing,PS_ratio)
#stat_summary_compare_by_fit(Ret_indexing,PE_ratio)





#PanelOLS(Indexing, stock_mkt_cap,entity_effect=True).fit(debiased=True)
#ivolsmod = IV2SLS(Indexing, PE_ratio, None, None)
#res_ols = ivolsmod.fit()
#print(res_ols)

#
#ivmod =IVGMM((Indexing, stock_data)
#res_gmm = ivmod.fit()


#ivmod = IV2SLS(data.ldrugexp, data[controls], data.hi_empunion, data.ssiratio)
#res_2sls = ivmod.fit()
#print(res_2sls.summary)
#MomentumR6H1=stock_data.pct_change().rolling(126).sum()

#
#class FamaMacBeth(PooledOLS):
#    r"""
#    Pooled coefficient estimator for panel data
#
#    Parameters
#    ----------
#    dependent : array-like
#        Dependent (left-hand-side) variable (time by entity)
#    exog : array-like
#        Exogenous or right-hand-side variables (variable by time by entity).
#    weights : array-like, optional
#        Weights to use in estimation.  Assumes residual variance is
#        proportional to inverse of weight to that the residual time
#        the weight should be homoskedastic.
#
#    Notes
#    -----
#    The model is given by
#
#    .. math::
#
#        y_{it}=\beta^{\prime}x_{it}+\epsilon_{it}
#
#    The Fama-MacBeth estimator is computed by performing T regressions, one
#    for each time period using all available entity observations.  Denote the
#    estimate of the model parameters as :math:`\hat{\beta}_t`.  The reported
#    estimator is then
#
#    .. math::
#
#        \hat{\beta} = T^{-1}\sum_{t=1}^T \hat{\beta}_t
#
#    While the model does not explicitly include time-effects, the
#    implementation based on regressing all observation in a single
#    time period is "as-if" time effects are included.
#
#    Parameter inference is made using the set of T parameter estimates with
#    either the standard covariance estimator or a kernel-based covariance,
#    depending on ``cov_type``.
#    """
#
#    def __init__(self, dependent, exog, *, weights=None):
#        super(FamaMacBeth, self).__init__(dependent, exog, weights=weights)
#        self._validate_blocks()
#
#    def _validate_blocks(self):
#        x = self._x
#        root_w = np.sqrt(self._w)
#        wx = root_w * x
#
#        exog = self.exog.dataframe
#        wx = pd.DataFrame(wx[self._not_null], index=exog.notnull().index, columns=exog.columns)
#
#        def validate_block(ex):
#            return ex.shape[0] >= ex.shape[1] and matrix_rank(ex) == ex.shape[1]
#
#        valid_blocks = wx.groupby(level=1).apply(validate_block)
#        if not valid_blocks.any():
#            err = 'Model cannot be estimated. All blocks of time-series observations are rank\n' \
#                  'deficient, and so it is not possible to estimate any cross-sectional ' \
#                  'regressions.'
#            raise ValueError(err)
#        if valid_blocks.sum() < exog.shape[1]:
#            import warnings
#            warnings.warn('The number of time-series observation available to estimate '
#                          'cross-sectional\nregressions, {0}, is less than the number of '
#                          'parameters in the model. Parameter\ninference is not '
#                          'available.'.format(valid_blocks.sum()), InferenceUnavailableWarning)
#        elif valid_blocks.sum() < valid_blocks.shape[0]:
#            import warnings
#            warnings.warn('{0} of the time-series regressions cannot be estimated due to '
#                          'deficient rank.'.format(valid_blocks.shape[0] - valid_blocks.sum()),
#                          MissingValueWarning)
#
#    def fit(self, cov_type='unadjusted', debiased=True, **cov_config):
#        """
#        Estimate model parameters
#
#        Parameters
#        ----------
#        cov_type : str, optional
#            Name of covariance estimator. See Notes.
#        debiased : bool, optional
#            Flag indicating whether to debiased the covariance estimator using
#            a degree of freedom adjustment.
#        **cov_config
#            Additional covariance-specific options.  See Notes.
#
#        Returns
#        -------
#        results :  PanelResults
#            Estimation results
#
#        Examples
#        --------
#        >>> from linearmodels import FamaMacBeth
#        >>> mod = FamaMacBeth(y, x)
#        >>> res = mod.fit(cov_type='kernel', kernel='Parzen')
#
#        Notes
#        -----
#        Four covariance estimators are supported:
#
#        * 'unadjusted', 'homoskedastic', 'robust', 'heteroskedastic' - Use the
#          standard covariance estimator of the T parameter estimates.
#        * 'kernel' - HAC estimator. Configurations options are:
#
#          * ``kernel`` - One of the supported kernels (bartlett, parzen, qs).
#            Default is Bartlett's kernel, which is implements the the
#            Newey-West covariance estimator.
#          * ``bandwidth`` - Bandwidth to use when computing the kernel.  If
#            not provided, a naive default is used.
#        """
#        y = self._y
#        x = self._x
#        root_w = np.sqrt(self._w)
#        wy = root_w * y
#        wx = root_w * x
#
#        dep = self.dependent.dataframe
#        exog = self.exog.dataframe
#        index = self.dependent.index
#        wy = pd.DataFrame(wy[self._not_null], index=index, columns=dep.columns)
#        wx = pd.DataFrame(wx[self._not_null], index=exog.notnull().index, columns=exog.columns)
#
#        yx = pd.DataFrame(np.c_[wy.values, wx.values], columns=list(wy.columns) + list(wx.columns),
#                          index=wy.index)
#
#        def single(z: pd.DataFrame):
#            exog = z.iloc[:, 1:].values
#            if exog.shape[0] < exog.shape[1] or matrix_rank(exog) != exog.shape[1]:
#                return pd.Series([np.nan] * len(z.columns), index=z.columns)
#            dep = z.iloc[:, :1].values
#            params = lstsq(exog, dep)[0]
#            return pd.Series(np.r_[np.nan, params.ravel()], index=z.columns)
#
#        all_params = yx.groupby(level=1).apply(single)
#        all_params = all_params.iloc[:, 1:]
#        params = all_params.mean(0).values[:, None]
#        all_params = all_params.values
#
#        wy = wy.values
#        wx = wx.values
#        index = self.dependent.index
#        fitted = pd.DataFrame(self.exog.values2d @ params, index, ['fitted_values'])
#        effects = pd.DataFrame(np.full_like(fitted.values, np.nan), index, ['estimated_effects'])
#        idiosyncratic = pd.DataFrame(self.dependent.values2d - fitted.values, index,
#                                     ['idiosyncratic'])
#
#        eps = self.dependent.values2d - fitted.values
#        weps = wy - wx @ params
#        w = self.weights.values2d
#        root_w = np.sqrt(w)
#        #
#        residual_ss = float(weps.T @ weps)
#        y = e = self.dependent.values2d
#        if self.has_constant:
#            e = y - (w * y).sum() / w.sum()
#        total_ss = float(w.T @ (e ** 2))
#        r2 = 1 - residual_ss / total_ss
#
#        if cov_type in ('robust', 'unadjusted', 'homoskedastic', 'heteroskedastic'):
#            cov_est = FamaMacBethCovariance
#        elif cov_type == 'kernel':
#            cov_est = FamaMacBethKernelCovariance
#        else:
#            raise ValueError('Unknown cov_type')
#
#        cov = cov_est(wy, wx, params, all_params, debiased=debiased, **cov_config)
#        df_resid = wy.shape[0] - params.shape[0]
#        res = self._postestimation(params, cov, debiased, df_resid, weps, wy, wx, root_w)
#        index = self.dependent.index
#        res.update(dict(df_resid=df_resid, df_model=x.shape[1], nobs=y.shape[0],
#                        residual_ss=residual_ss, total_ss=total_ss,
#                        r2=r2, resids=eps, wresids=weps, index=index, fitted=fitted,
#                        effects=effects, idiosyncratic=idiosyncratic))
#        return PanelResults(res)
#
#    @classmethod
#    def from_formula(cls, formula, data, *, weights=None):
#        """
#        Create a model from a formula
#
#        Parameters
#        ----------
#        formula : str
#            Formula to transform into model. Conforms to patsy formula rules.
#        data : array-like
#            Data structure that can be coerced into a PanelData.  In most
#            cases, this should be a multi-index DataFrame where the level 0
#            index contains the entities and the level 1 contains the time.
#        weights: array-like, optional
#            Weights to use in estimation.  Assumes residual variance is
#            proportional to inverse of weight to that the residual times
#            the weight should be homoskedastic.
#
#        Returns
#        -------
#        model : FamaMacBeth
#            Model specified using the formula
#
#        Notes
#        -----
#        Unlike standard patsy, it is necessary to explicitly include a
#        constant using the constant indicator (1)
#
#        Examples
#        --------
#        >>> from linearmodels import BetweenOLS
#        >>> mod = FamaMacBeth.from_formula('y ~ 1 + x1', panel_data)
#        >>> res = mod.fit()
#        """
#        parser = PanelFormulaParser(formula, data)
#        dependent, exog = parser.data
#        mod = cls(dependent, exog, weights=weights)
#        mod.formula = formula
#        return mod

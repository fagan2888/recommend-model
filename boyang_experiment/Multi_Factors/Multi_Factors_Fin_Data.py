# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:09:28 2019

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



Fin_data=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\Multi Factors\data\caihui_tq_fin_procfsqsubjects.csv",index_col=[0])
Fin_data=Fin_data.sort_values(['COMPCODE','FIRSTPUBLISHDATE'])


Fin_data_code=pd.read_csv(r"C:\Users\yshlm\Desktop\licaimofang\Multi Factors\data\tq_oa_stcode.csv")
Fin_data_code.columns=['Ticker','COMPNAME','COMPCODE']

Fin_data_code.Ticker=Fin_data_code.Ticker.apply(lambda x: str(x)+str('.SH'))

Fin_data_code=Fin_data_code[Fin_data_code.Ticker.isin(stock_data.columns)]

Fin_data=Fin_data[Fin_data.COMPCODE.isin(Fin_data_code.COMPCODE)]
A=Fin_data.copy()
A.index=A.COMPCODE
Fin_data_code.index=Fin_data_code.COMPCODE
AA=pd.merge(A,Fin_data_code,right_index=True,left_index=True,how='outer')
AA.to_csv(r'caihui_tq_fin_procfsqsubjects_50.csv')

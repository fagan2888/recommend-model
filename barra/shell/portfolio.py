#coding=utf8

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings("ignore")

start = time.time()
print 'Programme Start', start
print '----------------------------------------------------'
#Data Clearn

low_beta    = pd.read_excel('low_beta_returns.xlsx', index_col='Date', parse_dates=True)[[-1]]
high_beta   = pd.read_excel('high_beta_returns.xlsx', index_col='Date', parse_dates=True)[[-1]]        
small_bm    = pd.read_excel('small_bm_returns.xlsx', index_col='Date', parse_dates=True)[[-1]]        
big_bm      = pd.read_excel('big_bm_returns.xlsx', index_col='Date', parse_dates=True)[[-1]]        
small_size  = pd.read_excel('small_size_returns.xlsx', index_col='Date', parse_dates=True)[[-1]]        
big_size    = pd.read_excel('big_size_returns.xlsx', index_col='Date', parse_dates=True)[[-1]]        
low_m       = pd.read_excel('low_momentum_returns.xlsx', index_col='Date', parse_dates=True)[[-1]]        
high_m      = pd.read_excel('high_momentum_returns.xlsx', index_col='Date', parse_dates=True)[[-1]]
high_alpha  = pd.read_excel('high_alpha_returns.xlsx', index_col='Date', parse_dates=True)[[-1]]  
low_alpha   = pd.read_excel('low_alpha_returns.xlsx', index_col='Date', parse_dates=True)[[-1]]  
ffc         = pd.read_excel('FFC.xlsx', index_col='Date', parse_dates=True)

low_beta.columns    = ['low_beta']
high_beta.columns   = ['high_beta']
small_bm.columns    = ['small_bm']
big_bm.columns      = ['big_bm']
small_size.columns  = ['small_size']
big_size.columns    = ['big_size']
low_m.columns       = ['low_m']
high_m.columns      = ['high_m']
high_alpha.columns  = ['high_alpha']
low_alpha.columns   = ['low_alpha']

df = pd.merge(low_m, high_m, left_index=True, right_index=True,how = 'left') 
df = pd.merge(df, big_size, left_index=True, right_index=True,how = 'left') 
df = pd.merge(df, small_size, left_index=True, right_index=True,how = 'left') 
df = pd.merge(df, big_bm, left_index=True, right_index=True,how = 'left') 
df = pd.merge(df, small_bm, left_index=True, right_index=True,how = 'left') 
df = pd.merge(df, high_beta, left_index=True, right_index=True,how = 'left') 
df = pd.merge(df, low_beta, left_index=True, right_index=True,how = 'left')
df = pd.merge(df, high_alpha, left_index=True, right_index=True,how = 'left') 
df = pd.merge(df, low_alpha, left_index=True, right_index=True,how = 'left')
df = pd.merge(df, ffc, left_index=True, right_index=True,how = 'left')

#print df

df.to_excel('portfolio.xlsx')

end = time.time()
print 'Programme End', end
print 'Costs (Mins)', (end-start)/60.0	 
print '----------------------------------------------------'





#coding=utf8


import scipy
import pandas as pd
import numpy as np
import itertools
import time
from sklearn import datasets, linear_model
import datetime
import statsmodels.api as sm


factor_num = 6


def max_sq_r(fund_dfr, factor_dfr):


    factors = list(factor_dfr.columns.values)

    vs = []
    #for i in range(1, len(factors)):
    for cols in list(itertools.combinations(factors, factor_num)):
        #reg   = clf.fit(factor_dfr[list(cols)], fund_dfr.values)
        #score = reg.score(factor_dfr[list(cols)], fund_dfr.values)
        #print  score, reg.intercept_

        X = factor_dfr[list(cols)].values
        y = fund_dfr.values

        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        alpha   = results.params[0]
        ws      = results.params[1:]
        rsq = results.rsquared_adj
        vs.append([alpha, rsq, cols, ws])

        #print results.summary()
        #print results.use_t
        #print results.params
        #print results.bse
        #print(results.pvalues)


        '''
        pvalues = results.pvalues
        sig_factor   = []        
        for i in range(1, len(pvalues)):
            if pvalues[i] <= 0.05:
                sig_factor.append(cols[i-1])

        #print sig_factor            
        X = factor_dfr[sig_factor].values
        X = sm.add_constant(X)
        model = sm.OLS(y, X)
        results = model.fit()
        '''
        #print results.rsquared_adj
        #print 


    df = pd.DataFrame(vs, columns = ['alpha','rsq', 'factor', 'weight'])                
    return df



def multi_factor(fund_df, factor_df):

    fund_score = {}

    factor_dfr = factor_df.pct_change().fillna(0.0)
    fund_dfr   = fund_df.pct_change().fillna(0.0)
    cols     = fund_dfr.columns

    for col in cols:

        tmp_fund_dfr = fund_dfr[col]
        df = max_sq_r(tmp_fund_dfr, factor_dfr)
        df = df.sort(['rsq'])
        alphas = df['alpha'].values
        index = df.index
        last_index = index[-1]       
        score = df.loc[last_index, 'alpha'] * 0.25 + 1.0 / df.loc[last_index, 'rsq'] * 0.25 + np.mean(alphas) / np.std(alphas) * 0.5 
        fund_score[col] = score
        print col, score, df.loc[last_index, 'factor'], df.loc[last_index, 'weight']

if __name__ == '__main__':


    barra_df = pd.read_csv('./data/diff.csv', index_col='date', parse_dates=['date'])
    industry_df = pd.read_csv('./data/industry.csv', index_col='date', parse_dates=['date'])
    bond_df = pd.read_csv('./data/bonds.csv', index_col='date', parse_dates=['date'])
    fund_df = pd.read_csv('./data/stock.csv', index_col='date', parse_dates=['date'])

    fund_df = fund_df[-52:]
    factor_df = pd.concat([barra_df, bond_df], axis = 1, join_axes = [fund_df.index[-52:]])

    fund_score = multi_factor(fund_df, factor_df)

    print fund_score

#coding=utf8

import scipy
import pandas as pd
import numpy as np
import itertools
import time
from sklearn import datasets, linear_model
import datetime
import statsmodels.api as sm
import string
from statsmodels import regression

def lin(x,y):
	lin  = linear_model.LinearRegression()
	reg  = lin.fit(x, y)
	coef  = reg.coef_
	return coef[0][0], coef[0][1], coef[0][2], coef[0][3]


def max_sqr(fund_dfr, factor_dfr):

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
    df.to_csv('test/max_sqr.csv')
    return df


def multi_factor(fund_df, factor_df, rf):

    fund_score = {}

    factor_dfr   = factor_df.pct_change().fillna(0.0)
    fund_dfr     = fund_df.pct_change().fillna(0.0)
    rf           = (1.0 + rf / 100.0) ** (1.0 / 52.0) - 1.0
    date         = fund_dfr[[-1]]
    date.columns = ['tmp']
    rf           = pd.merge(date, rf, right_index=True, left_index=True, how='left')
    rf           = rf.fillna(method='ffill')
    rf           = rf.fillna(0)
    del rf['tmp']
    stocks = list(fund_df.columns)
    for j in stocks:
        fund_df[j] = fund_df[j] - rf['rf']

    cols         = fund_dfr.columns

    data = []
    for col in cols:

        tmp_fund_dfr = fund_dfr[col]
        df = max_sqr(tmp_fund_dfr, factor_dfr)
        df = df.sort(['rsq'])
        alphas = df['alpha'].values
        index = df.index
        last_index = index[-1]
        score = df.loc[last_index, 'alpha'] * 0.25 + 1.0 / df.loc[last_index, 'rsq'] * 0.25 + np.mean(alphas) / np.std(alphas) * 0.5
        yj_score = np.mean(alphas) / np.std(alphas) - 1.0 / df.loc[last_index, 'rsq']
        fund_score[col] = score
        fund_score[col] = score
        #print col, yj_score, score, df.loc[last_index, 'alpha'] , df.loc[last_index, 'rsq'] , np.mean(alphas)/ np.std(alphas)
        data.append([col, yj_score, score, df.loc[last_index, 'alpha'] , df.loc[last_index, 'rsq'] , np.mean(alphas)/ np.std(alphas), df.loc[last_index, 'factor'], df.loc[last_index, 'weight'] ])

    df = pd.DataFrame(data, index = cols, columns = ['code', 'yj_score', 'score', 'alpha', 'rsq', 'alpha_persistence','factor', 'weight'])
    #print type(df['factor'])


    fvs = [] 
    for i in range(0, len(df)):

        tmp_df = df.iloc[i,:]

        code = tmp_df['code']
        F1 = tmp_df['factor'][0]
        F2 = tmp_df['factor'][1]
        F3 = tmp_df['factor'][2]
        F4 = tmp_df['factor'][3]

        optimal_factors = factor_dfr[[F1, F2, F3, F4]]

        #def fmcar(fund_dfr, optimal_factors):
        b1,b2,b3,b4 = lin(optimal_factors,fund_dfr)

        cov = optimal_factors.cov()
        cov = cov.values

        ar_squared = (fund_dfr.iloc[i,:].std()) ** 2
        fmcar1 = (b1 * (b4 * cov[0, 3] + b3 * cov[0, 2] + b2 * cov[0, 1] + b1 * cov[0, 0])) / ar_squared
        fmcar2 = (b2 * (b4 * cov[1, 3] + b3 * cov[1, 2] + b2 * cov[1, 1] + b1 * cov[1, 0])) / ar_squared
        fmcar3 = (b3 * (b4 * cov[2, 3] + b3 * cov[2, 2] + b2 * cov[2, 1] + b1 * cov[2, 0])) / ar_squared
        fmcar4 = (b4 * (b4 * cov[3, 3] + b3 * cov[3, 2] + b2 * cov[3, 1] + b1 * cov[3, 0])) / ar_squared
        #print ('F1 Risk Contribution:', fmcar1)
        #print ('F2 Risk Contribution:', fmcar2)
        #print ('F3 Risk Contribution:', fmcar3)
        #print ('F4 Risk Contribution:', fmcar4)

        #fmcar = fmcar(fund_dfr, optimal_factors)

        factor_v = []
        for col in factor_df.columns:
            if col == F1:
                factor_v.append(fmcar1.values[0])
            elif col == F2:
                factor_v.append(fmcar2.values[0])
            elif col == F3:
                factor_v.append(fmcar3.values[0])
            elif col == F4:
                factor_v.append(fmcar4.values[0])
            else:
                factor_v.append(np.nan)

        fvs.append(factor_v)
    fmcar_df = pd.DataFrame(fvs, index = df.index, columns = factor_df.columns)
        #print fmcar_df

    print fmcar_df
    return df

if __name__ == '__main__':

    factor_num = 4

    barra_df  = pd.read_csv('./data/diff.csv', index_col='date', parse_dates=True)
    bond_df   = pd.read_csv('./data/bonds.csv', index_col='date', parse_dates=True)
    winda_df  = pd.read_csv('./data/windA.csv', index_col='date', parse_dates=True)
    fund_df   = pd.read_csv('./data/stock.csv', index_col='date', parse_dates=True)
    rf        = pd.read_excel('data/rf.xls', index_col='Date', parse_dates=True)
    fund_df   = fund_df.iloc[:,0:3]

    factor_df = pd.concat([barra_df, bond_df, winda_df], axis = 1, join_axes = [fund_df.index])

    test      = multi_factor(fund_df, factor_df, rf)


















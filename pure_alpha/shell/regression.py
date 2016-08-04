#coding=utf8


import scipy
import pandas as pd
import numpy as np
import itertools
import time
from sklearn import datasets, linear_model
import datetime
import statsmodels.api as sm


def stepwise_regression():
    return 0


def max_sq_r(fund_dfr, factor_dfr):


	clf = linear_model.LinearRegression()

	factors = list(factor_dfr.columns.values)


	vs = []
	#for i in range(1, len(factors)):
	for cols in list(itertools.combinations(factors, 4)):
		#reg   = clf.fit(factor_dfr[list(cols)], fund_dfr.values)
		#score = reg.score(factor_dfr[list(cols)], fund_dfr.values)
		#print  score, reg.intercept_

		X = factor_dfr[list(cols)].values
		y = fund_dfr.values

		X = sm.add_constant(X)
		model = sm.OLS(y, X)
		results = model.fit()
		alpha   = results.params[0]
		rsq = results.rsquared_adj
		vs.append([alpha, rsq, cols])

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


	df = pd.DataFrame(vs, columns = ['alpha','rsq', 'factor'])				
	return df


if __name__ == '__main__':


	factor_dfr = pd.read_csv('./data/factor.csv', index_col='date', parse_dates=['date']).pct_change().fillna(0.0)
	fund_dfr = pd.read_csv('./data/fund.csv', index_col='date', parse_dates=['date']).pct_change().fillna(0.0)

	cols     = fund_dfr.columns
	for col in cols:

		tmp_fund_dfr = fund_dfr[col]
		df = max_sq_r(tmp_fund_dfr, factor_dfr)

		df = df.sort(['rsq'])
		print df
		#df.to_csv(str(col) + '.csv')
		#alphas = df['alpha'].values
		#print alphas
		#print df
		#index = df.index
		#last_index = index[-1]		
		#print df.loc[last_index, : 'alpha'].values
		#print df.loc[last_index, , 'rsq'].values
		#print col, df.loc[last_index, 'alpha'] * 0.25 + 1.0 / df.loc[last_index, 'rsq'] * 0.25 + np.mean(alphas) / np.std(alphas) * 0.5
		#print col, np.mean(alphas) / np.std(alphas) * 0.5
	#max_sq_r(None, None)
	#print 'hehe'


	'''
    url = "http://data.princeton.edu/wws509/datasets/salary.dat"
    data = pd.read_csv(url, sep='\\s+')

    print data

    model = forward_selected(data, 'sl')

    print model.model.formula
    # sl ~ rk + yr + 1

    print model.rsquared_adj
    # 0.835190760538
	'''

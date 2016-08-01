#coding=utf8


import scipy
import pandas as pd
import itertools
import time
from sklearn import datasets, linear_model



def stepwise_regression():
    return 0


def max_sq_r(fund_dfr, factor_dfr):


	clf = linear_model.LinearRegression()

	factors = list(factor_dfr.columns.values)

	for i in range(1, len(factors)):
		for cols in list(itertools.combinations(factors, i)):
			#print cols #clf.fit(factor_dfr[list[cols]], fund_dfr)
			#print type(factor_dfr[list(cols)])
			#print fund_dfr
			print cols
			reg = clf.fit(factor_dfr[list(cols)], fund_dfr.values)
			print reg.score(factor_dfr[list(cols)], fund_dfr.values)


if __name__ == '__main__':


	factor_dfr = pd.read_excel('./data/factors.xlsx', index_col='Date')
	factor_dfr = factor_dfr.fillna(0.0)
	#print factor_dfr
	fund_dfr = pd.read_excel('./data/ls.xlsx', index_col='Date')
	fund_dfr = fund_dfr.fillna(0.0)

	cols     = fund_dfr.columns
	fund_dfr = fund_dfr[cols[0]]
	max_sq_r(fund_dfr, factor_dfr)

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

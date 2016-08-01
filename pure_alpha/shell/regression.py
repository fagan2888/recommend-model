#coding=utf8



import scipy
import statsmodels.formula.api as smf
import pandas as pd
import itertools



def stepwise_regression():
    return 0



def max_sq_r(fund_dfr, factor_dfr):

	factors = factor_dfr.columns

	for i in range(1, len(factors)):

		print list(itertools.combinations(['a', 'b', 'c'], 2))



if __name__ == '__main__':


	factor_dfr = pd.read_excel('./data/factors.xlsx', index_col='Date')
	#print factor_dfr

	fund_dfr = pd.read_excel('./data/ls.xlsx', index_col='Date')
	cols     = fund_dfr.columns

	fund_dfr = fund_dfr[cols[0]]
	print fund_dfr


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

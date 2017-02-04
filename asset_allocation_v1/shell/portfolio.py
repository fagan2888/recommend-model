#coding=utf8


import numpy as np
import string
import sys
sys.path.append("code")
import pandas as pd


def portfolio_sharpe(pvs, period_type):

	period = 52
	if 'w' == period_type:
		period = 52
	elif 'd' == period_type:
		period = 252

	rs = []
	for i in range(1, len(pvs)):
		rs.append(pvs[i] / pvs[i-1] - 1)

	returns = np.mean(rs) * period
	risk    = np.std(rs) * (period ** 0.5)


	return (returns - 0.03) / risk


def portfolio_return(pvs, period_type):
	#print pvs
	#print pvs[0]
	#print pvs[len(pvs) - 1]
	period = 52
	if 'w' == period_type:
		period = 52
	elif 'd' == period_type:
		period = 252

	r = ((pvs[-1] / pvs[0]) ** (1.0 * period / len(pvs))) - 1
	return r
	#return pvs[len(pvs) - 1] / pvs[0] - 1


def portfolio_risk(pvs):
	rs = []
	for i in range(1, len(pvs)):
		rs.append(pvs[i] / pvs[i-1] - 1)
	return np.std(rs)



def portfolio_maxdrawdown(pvs):
	mdd = 0
	peak = pvs[0]
	for v in pvs:
		if v > peak:
			peak = v
		dd = (peak - v) / peak
		if dd > mdd:
			mdd = dd
	return mdd
#coding=utf8


import numpy as np
import string
import sys
sys.path.append("windshell")
import const
import Financial as fin
import stockfilter as sf
import stocktag as st
import portfolio as pf
import fundindicator as fi
import fund_selector as fs
import data
import datetime
from numpy import *
import fund_evaluation as fe
import pandas as pd


if __name__ == '__main__':

	df  = pd.read_csv('./wind/allocation_data.csv', index_col='date', parse_dates='date')

	df  = df.fillna(method='pad')

	dfr = df.pct_change().fillna(0.0)

	dates = dfr.index
	#dates.sort()

	#df.to_csv('hehe.csv')

	fundws = {}
	fund_values = {}
	fund_codes = []
	portfolio_vs = []
	portfolio_vs.append(1)


	net_value_f = open('./tmp/net_value.csv', 'w')
	net_value_f.write('date, net_value\n')
	allocation_f = open('./tmp/allocation.csv', 'w')
	#allocation_f.write('date, largecap, smallcap, rise, oscillation, decline ,growth ,value, ratebond, creditbond, convertiblebond, money1, money2, SP500.SPI, SPGSGCTR.SPI, HSCI.HI\n')


	for i in range(13, len(dates)):

		if i % 13 == 0:

			start_date = dates[i - 13].strftime('%Y-%m-%d')
			end_date = dates[i - 1].strftime('%Y-%m-%d')

			allocation_df = df[df.index <= datetime.datetime.strptime(end_date, '%Y-%m-%d')]
			allocation_df = allocation_df[allocation_df.index >= datetime.datetime.strptime(start_date, '%Y-%m-%d')]

			#print allocation_df

			fund_codes = allocation_df.columns

			#print allocation_df

			uplimit   = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4]
			downlimit = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

			bound = [uplimit, downlimit]

			risk, returns, ws, sharpe = pf.markowitz(allocation_df, bound)

			last_pv = portfolio_vs[-1]
			fund_values = {}
			for n in range(0, len(fund_codes)):
				fund_values[n] = [last_pv * ws[n]]

			#print ws


			#ws_str = "%s, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n"
			ws_str = "%s, %f, %f, %f, %f, %f, %f, %f, %f\n"
			#allocation_f.write(ws_str % (end_date, ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], ws[6], ws[7], ws[8], ws[9], ws[10], ws[11], ws[12], ws[13], ws[14]))
			allocation_f.write(ws_str % (end_date, ws[0], ws[1], ws[2], ws[3], ws[4], ws[5], ws[6], ws[7] ))


		pv = 0
		d = dates[i]
		for n in range(0, len(fund_codes)):
			vs = fund_values[n]
			vs.append(vs[-1] + vs[-1] * dfr.loc[d, fund_codes[n]])
			pv = pv + vs[-1]

		#print fund_values
		portfolio_vs.append(pv)
		print d, pv
		net_value_f.write(str(d) + "," + str(pv) + "\n")


	print "sharpe : " ,fi.portfolio_sharpe(portfolio_vs)
	print "annual_return : " ,fi.portfolio_return(portfolio_vs)
	print "maxdrawdown : " ,fi.portfolio_maxdrawdown(portfolio_vs)


	net_value_f.flush()
	net_value_f.close()
	allocation_f.flush()
	allocation_f.close()

#coding=utf8

import pandas as pd
import numpy as np

class RSI_strategy(object):
	
	#n1 = n日rsi
	#n2 = n日短rsi
	#n3 = n日长rsi
	def __init__(self, n1, n2, n3):
		self.n1 = n1
		self.n2 = n2
		self.n3 = n3


	#计算n日rsi的值
	def compute_rsi(self, n1, df_nav):
		close_price = pd.DataFrame({'close_price':df_nav.tc_close}, index = df_nav.index)
		sum_rise = 0
		sum_low = 0
		rs_up = pd.DataFrame({'rs_up':[0.0]*len(df_nav)}, index = df_nav.index)
		rs_down = pd.DataFrame({'rs_down':[0.0]*len(df_nav)}, index = df_nav.index)
		diff_close = close_price.diff()
		rsi_values = pd.DataFrame({'rsi':[0.0]*len(df_nav)}, index = df_nav.index)
		
		for i in range(n1-1, len(df_nav)):
			for j in range (i-n1+1, i+1):
				if diff_close.iloc[j][0] > 0:
					sum_rise += diff_close.iloc[j][0]	
				elif diff_close.iloc[j][0] < 0:
					sum_low += abs(diff_close.iloc[j][0])
			rs_up.iloc[i][0] = sum_rise
			rs_down.iloc[i][0] = sum_low
		rs_up = rs_up.diff()
		rs_down = rs_down.diff()
		
		for k in range(0, len(df_nav)):
			rs_up.iloc[k][0] = rs_up.iloc[k][0]/n1
			rs_down.iloc[k][0] = rs_down.iloc[k][0]/n1
			rsi_values.iloc[k][0] = 100*(rs_up.iloc[k][0]/(rs_up.iloc[k][0]+rs_down.iloc[k][0]))
		return rsi_values

	#择时：
	#若n2日rsi值在50到80之间，不属于超买现象，但人们购买热情，且n1日短rsi从下往上穿过n3长rsi：购买
	#若n2日rsi值在20到50之间，不属于超卖现象，但人们欲望减少，且n1日短rsi从上往下穿过n3长rsi：赎回

	def timing(self, df_nav):
		n1_rsi = self.compute_rsi(self.n1, df_nav)
		n2_rsi = self.compute_rsi(self.n2, df_nav)
		n3_rsi = self.compute_rsi(self.n3, df_nav)
		#all_rsi = pd.DataFrame({'short_rsi':n1_rsi.rsi, 'rsi':n2_rsi.rsi, \
		#						'long_rsi':n3_rsi.rsi}, index = df_nav.index)
		all_rsi = pd.DataFrame({'short_rsi':n1_rsi.rsi, 'rsi':n2_rsi.rsi, 'long_rsi':n3_rsi.rsi},\
					columns=['short_rsi','rsi','long_rsi'], index = df_nav.index)
		singnal = pd.DataFrame({'singnal':[0]*len(df_nav)}, index = df_nav.index)
		for i in range(1, len(df_nav)):
			if all_rsi.iloc[i][1] > 50.0 and all_rsi.iloc[i][1] < 80.0 and \
			   all_rsi.iloc[i-1][0] < all_rsi.iloc[i-1][2] and \
			   all_rsi.iloc[i][0] > all_rsi.iloc[i][2]:
				singnal.iloc[i][0] = 1
			elif all_rsi.iloc[i][1] > 20.0 and all_rsi.iloc[i][1] < 50.0 and \
 			     all_rsi.iloc[i-1][0] > all_rsi.iloc[i-1][2] and \
 			     all_rsi.iloc[i][0] < all_rsi.iloc[i][2]:
 				singnal.iloc[i][0] = -1

		return singnal




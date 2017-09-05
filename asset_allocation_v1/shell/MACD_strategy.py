# coding=utf8


import pandas as pd
import numpy as np

class MACD_strategy(object):

	#MACD求值过程：
	#	离差值DIF = ewmaCal(12) - ewmaCal(26)
	#	DEA = ewmaCal(DIF,9)
	#	DIF,DEA > 0 DIF向上突破DEA 释放买入信号
	#	DIF,DEA < 0 DIF向下跌破DEA 释放卖出信号
	
	def timing(self, df_nav):
		df_nav = df_nav[['tc_close']]
		Ewma12 = pd.ewma(df_nav.values, span = 12)
		#Ewma12 = pd.DataFrame({'Ewma12':Ewma12}, index = df_nav.index)
		#print Ewma12[100]
		Ewma26 = pd.ewma(df_nav.values, span = 26)
		#Ewma26 = pd.DataFrame({'Ewma26':Ewma26}, index = df_nav.index)
		DIF = Ewma12-Ewma26
		#DIF = pd.Series(DIF, index = df_nav.index)
		#print DIF.head(100)
		DEA = pd.ewma(DIF, span = 9)
		DEA = pd.Series({'DEA':DEA},index = df_nav.index)
		
		Singnal = pd.Series(0.0, index = df_nav.index)
		for i in range(1, len(DIF)):
			if DIF[i] > DEA[i] > 0.0 and DIF[i-1] < DEA[i-1]:
				Singnal[i] = 1
			elif DIF[i] < DEA[i] < 0.0 and DIF[i-1] > DEA[i-1]:
				Singnal[i] = -1
		df = pd.DataFrame()
		df['singnal'] = Singnal
		df['DIF'] = DIF
		df['DEA'] = DEA
		df.index = df_nav.index
		#print df.head(200)
		#return df
	
		

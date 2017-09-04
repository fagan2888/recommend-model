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
		Ewma12 = pd.DataFrame(Ewma12,index = df_nav.index)
		Ewma26 = pd.ewma(df_nav.values, span = 26)
		Ewma26 = pd.DataFrame(Ewma26,index = df_nav.index)
		DIF = Ewma12 - Ewma26
		DEA = pd.ewma(DIF, span = 9)
		DEA = pd.DataFrame(DEA, index = df_nav.index)
		Singnal = pd.Series(0.0, index = df_nav.index)
		for i in range(1, len(DIF)):
			if DIF.iloc[i].values > DEA.iloc[i].values > 0.0 and DIF.iloc[i-1].values < DEA.iloc[i-1].values:
				Singnal[i] = 1
			elif DIF.iloc[i].values < DEA.iloc[i].values < 0.0 and DIF.iloc[i-1].values > DEA.iloc[i-1].values:
				Singnal[i] = -1
		#df = pd.merge(DIF,DEA,how='left',left_index=True,right_index=True)
		df = pd.DataFrame()
		print df.head(30)
		df['singnal'] = Singnal
		df['DIF'] = DIF
		df['DEA'] = DEA
		df.index = DIF.index
		return df
	
		

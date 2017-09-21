#coding=utf8


import pandas as pd
import numpy as np


class LLTStrategy(object):
	
	#设置a为权重指数
	def __init__(self,d):
		self.a = 2.0/(d+1)

	
	#列出每日llt的值
	def llt(self,df_nav,a):
		Llt = pd.DataFrame({'LLT':0.0}, index = df_nav.index)
		num = len(df_nav)
		for i in range(2, num):
		#	Llt[i] = (a-(a*a)/4)*df_nav.iloc[i][3]+((a**2)*Close_price.iloc[i-1]\
		#	-(a-(3*a**2)/4)*Close_price.iloc[i-2]+2*(1-a)*Llt[i-1]-(1-a)**2*Llt[i-2]
			Llt.iloc[i][0] = (a-(a*a)/4)*df_nav.iloc[i][3]+((a*a)/2)*df_nav.iloc[i-1][3]\
					 -(a-3*(a*a)/4)*df_nav.iloc[i-2][3]+2*(1-a)*Llt.iloc[i-1][0]-(1-a)*\
					 (1-a)*Llt.iloc[i-2][0]
		return Llt
		

	# 择时：
	#获得llt在当日该点切线k
	# k < 0 singnal=-1 通知用户赎回
	# k > 0 singnal=1  通知用户购买
	# k = 0 singnal=0  维持原状
	def signal(self,Llt):
		Signal = pd.DataFrame({'tc_signal':[0]*len(Llt)}, index = Llt.index)
		Llt = Llt.diff()
		num = len(Llt)
		for i in range(1, num):
			if Llt.iloc[i][0] < 0:
				Signal.iloc[i][0] = -1
			elif Llt.iloc[i][0] > 0:
				Signal.iloc[i][0] = 1
		return Signal
		


	def timing(self,df_nav):
		Llt = self.llt(df_nav, self.a)
		Signal = self.signal(Llt)
		Signal.replace(0,np.nan, inplace = True)
		Signal.fillna(method='ffill',inplace = True)
		Signal.dropna(inplace = True)
		return Signal






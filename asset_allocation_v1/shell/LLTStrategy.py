#coding=utf8


import pandas as pd
import numpy as np


class LLTStrategy(object):
	
	#设置a为权重指数
	def __init__(self,a):
		self.a = a

	
	#计算修改指数移动平均
	def change_ewma_Cal(self,Close_price, period, a):
		Ewma = pd.Series(0.0, index = Close_price.index)
		Ewma[period-1] = np.mean(Close_price.tc_close[:period])
		for i in range(period, len(Close_price)):
			Ewma[i] = a*((Close_price.tc_close[i]+Close_price.tc_close[i-1])/2)+\
					  (1-a)*Ewma[period-1]
		return Ewma

	#列出每日llt的值
	def llt(self,Close_price,a):
		Llt = pd.Series(0.0, index = Close_price.index)
		#print Close_price[0]
		Llt[0] = Close_price.iloc[0]
		Llt[1] = Close_price.iloc[1]
		for i in range(2, len(Close_price)):
			Llt[i] = (a-a**2/4)*Close_price.iloc[i]+(a**2)*Close_price.iloc[i-1]\
			-(a-(3*a**2)/4)*Close_price.iloc[i-2]+2*(1-a)*Llt[i-1]-(1-a)**2*Llt[i-2]
		#print Llt.head(8)
		return Llt
		

	# 择时：
	#获得llt在当日该点切线k
	# k < 0 singnal=-1 通知用户赎回
	# k > 0 singnal=1  通知用户购买
	# k = 0 singnal=0  维持原状
	def choose_timing(self,Llt):
		Singnal = pd.Series(0.0, index = Llt.index)
		for i in range(1, len(Llt)):
			if Llt[i]-Llt[i-1] < 0:
				Singnal[i] = -1
			elif Llt[i]-Llt[i-1] > 0:
				Singnal[i] = 1
		return Singnal
		


	def timing(self,df_nav):
		df_nav = df_nav[['tc_close']]
		#print df_nav.head(8)
		print df_nav.iloc[0]
		Llt = self.llt(df_nav, self.a)
		Singnal = self.choose_timing(Llt)
		df = pd.DataFrame({'tc_close':df_nav.tc_close,'Singnal':Singnal.values}, index = Singnal.index)
		return df






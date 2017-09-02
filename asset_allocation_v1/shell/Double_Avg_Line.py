#coding=utf8


import pandas as pd
import numpy as np

class Double_Avg_Line(object):
	#设置choose_Ssma为短期均线，choose_Lsma为长期均线
	def __init__(self,choose_Ssma,choose_Lsma):
		self.choose_Ssma =choose_Ssma
		self.choose_Lsma =choose_Lsma


	#简单移动平均计算收盘价均值
	def smaCal(self,Close_price,k):
		Sma = pd.Series(0.000, index = Close_price.index)
		Sma = Close_price.tc_close.rolling(window = k, center = False).mean() 
		
		return Sma


	#加权移动平均计算收盘价均值
	def wmaCal(self,Close_price,k):
		k = len(weight)
		arrWeight = np.array(weight)
		Wma = pd.Series(0.0, index = Close_price.index)
		for i in range(k-1, len(Close_price.index)):
			Wma[i] = sum(arrWeight*Close_price[(i-k+1):(i+1)])
		return(Wma)


	#指数加权平均计算收盘价均值
	def ewmaCal(self,Close_price, period = 5, exponential = 0.2):
		Ewma = pd.Series(0.0, index = Close_price.index)
		Ewma[period-1] = np.mean(Close_price[:period])
		for i in range(period, len(Close_price)):
			Ewma[i] = exponential*Close_price[i]+\
					(1-exponential)*Ewma[period-1]
		return(Ewma)


	#择时：
	#当短期均线从下往上穿过长期均线时，释放买入信号
	#当短期均线从上往下穿过长期均线时，释放卖出信号
	def choose_Line_Timing(self,Ssma, Lsma):
		SLSingnal = pd.Series(0,index = Lsma.index)
		for i in range(1,len(Lsma)):
			if Ssma[i] > Lsma[i] and Ssma[i-1] < Lsma[i-1]:
				SLSingnal[i] = 1
			elif Ssma[i] < Lsma[i] and  Ssma[i-1] > Lsma[i-1]:
				SLSingnal[i] = -1
		return SLSingnal


	def timing(self,df_nav):
		df_nav = df_nav[['tc_close']]
		Ssma = self.smaCal(df_nav,self.choose_Ssma)
		Lsma = self.smaCal(df_nav,self.choose_Lsma)
		SLSingnal = self.choose_Line_Timing(Ssma,Lsma)
		df = pd.DataFrame({'tc_close':df_nav.tc_close,'5_date_avg':Ssma.values,'20_date_avg':Lsma.values,'Singnal':SLSingnal.values}, index = SLSingnal.index)
		return df


	






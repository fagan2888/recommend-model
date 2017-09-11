#coding=utf8


import pandas as pd
import numpy as np

class Volume_strategy(object):

	#计算5日成交平均值，10日成交平均值
	#将两个值进行平均
	def compute_volume(self, df_volume):
		volume = pd.DataFrame({'volume':df_volume.tc_volume}, index = df_volume.index)
		volSMA5 = volume.rolling(window=5).mean()
		volSMA10 = volume.rolling(window=10).mean()
		volSMA = pd.DataFrame({'volSMA5':volSMA5.volume, 'volSMA10':volSMA10.volume,\
							 'volSMA':(volSMA5.volume+volSMA10.volume)/2, 'volume':\
							 df_volume.tc_volume}, index=df_volume.index)
		return volSMA

	#计算5日收盘价均值，20日收盘价均值
	def compute_price(self, df_volume):
		close_price = pd.DataFrame({'tc_close':df_volume.tc_close}, index = df_volume.index)
		avg_5 = close_price.rolling(window=5).mean()
		avg_20 = close_price.rolling(window=20).mean()
		close_avg = pd.DataFrame({'avg_5':avg_5.tc_close, 'avg_20':avg_20.tc_close}, \
								index = close_price.index)
		return close_avg

	#策略1:将当日成交量与volSMA比较
	#		若当日成交量>volSMA ：释放买入信号
	#		若当日成交量<volSMA ：释放卖出信号
	#策略2:
	#		若5日均值向上突破20日均值 ：释放买入信号
	#		若5日均值向下穿过20日均值 ：释放卖出信号
	#		两个都买入则买入
	#		两个都卖出则卖出
	def singnal(self, volSMA, close_avg):
		singnal_volume = pd.DataFrame({'singnal':[0]*len(volSMA)}, index=volSMA.index)
		singnal_close = pd.DataFrame({'singnal':[0]*len(close_avg)}, index=close_avg.index)
		
		for i in range(0, len(volSMA)):
			if volSMA.iloc[i, 3] > volSMA.iloc[i, 2]:
				singnal_volume.iloc[i,0] = 1
			elif volSMA.iloc[i, 3] < volSMA.iloc[i ,2]:
				singnal_volume.iloc[i, 0] = -1

		for i in range(0, len(close_avg)):
			if close_avg.iloc[i-1, 0] < close_avg.iloc[i-1 ,1] and \
			   close_avg.iloc[i, 0] > close_avg.iloc[i, 1]:
				singnal_close.iloc[i, 0] = 1
			elif close_avg.iloc[i-1, 0] > close_avg.iloc[i-1 ,1] and \
               close_avg.iloc[i, 0] < close_avg.iloc[i, 1]:
				singnal_close.iloc[i, 0] = -1

		singnal = pd.DataFrame({'singnal':singnal_volume.singnal+singnal_close.singnal}, \
								index = singnal_volume.index)
		return singnal


	def timing(self, df_volume):
		volSMA = self.compute_volume(df_volume)
		close_avg = self.compute_price(df_volume)
		singnal = self.singnal(volSMA,close_avg)
		return singnal

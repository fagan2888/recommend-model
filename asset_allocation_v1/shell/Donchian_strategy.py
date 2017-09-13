#coding=utf8

import pandas as pd
import numpy as np

class Donchian_strategy:

    #设定为几日
    def __init__(self, n):
        self.n = n


    #计算n日最高阶，最低价，以及中间价
    #中间价 = （最低价+最高价）／2
    def compute_price(self, n, df_nav):
        n_high = df_nav.tc_high.rolling(window=n).max()
        n_low = df_nav.tc_low.rolling(window=n).min()
        n_mid = pd.DataFrame({'mid':(n_high.values + n_low.values)/2}, index = n_low.index)
        high_low = pd.DataFrame({'n_day_high':n_high.values, 'n_day_low':n_low.values,\
								 'mid':n_mid.mid}, index = n_low.index)
        return high_low

	#择时：
	#若当日收盘价超出n日最高价，则通知买入
	#若当日收盘价低于n日最低价，则通知赎回
    def singnal(self, df_nav, high_low):
		singnal = pd.DataFrame([0]*len(df_nav), index=df_nav.index)
		close_price = pd.DataFrame({'close_price':df_nav.tc_close}, index = df_nav.index)
		for i in range(0,len(df_nav)):
			if close_price.iloc[i][0] > high_low.iloc[i][0]:
				singnal.iloc[i][0] = 1
			elif close_price.iloc[i][0] < high_low.iloc[i][1]:
				singnal.iloc[i][0] = -1
				#if singnal.iloc[i][0] == singnal.iloc[i-1][0] == -1:
 				#	singnal.iloc[i][0] = 0
		return singnal


    def timing(self, df_nav):
		high_low = self.compute_price(self.n, df_nav)
		singnal = self.singnal(df_nav, high_low)
		return singnal



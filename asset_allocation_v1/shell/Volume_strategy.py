#coding=utf8


import pandas as pd
import numpy as np

class Volume_strategy(object):

	#计算5日成交平均值，10日成交平均值
	#将两个值进行平均
    def compute_volume(self, df_nav):
        volume = pd.DataFrame({'volume':df_nav.tc_volume}, index = df_nav.index)
        volSMA5 = volume.rolling(window=5).mean()
        volSMA10 = volume.rolling(window=10).mean()
        volSMA = pd.DataFrame({'volSMA5':volSMA5.volume, 'volSMA10':volSMA10.volume,\
							   'volume': df_nav.tc_volume,\
                               'volSMA':(volSMA5.volume+volSMA10.volume)/2},\
                              columns=['volSMA5','volSMA10','volume','volSMA'],\
							  index=df_nav.index)
        return volSMA

	#计算5日收盘价均值，20日收盘价均值
    def compute_price(self, df_nav):
        close_price = pd.DataFrame({'tc_close':df_nav.tc_close}, index = df_nav.index)
        avg_10 = close_price.rolling(window=10).mean()
        avg_240 = close_price.rolling(window=240).mean()
        close_avg = pd.DataFrame({'avg_10':avg_10.tc_close, 'avg_240':avg_240.tc_close}, \
								columns=['avg_10','avg_240'], index = df_nav.index)
        return close_avg

	#策略1:将当日成交量与volSMA比较
	#		若当日成交量>volSMA ：释放买入信号
	#		若当日成交量<volSMA ：释放卖出信号
	#策略2:
	#		若5日均值向上突破20日均值 ：释放买入信号
	#		若5日均值向下穿过20日均值 ：释放卖出信号
	#		两个都买入则买入
	#		两个都卖出则卖出
    def signal(self, volSMA, close_avg, df_nav):
        #num = len(close_avg)
        signal = pd.DataFrame({'tc_signal':[0]*len(close_avg)},index = close_avg.index)
        for i in range(1, len(close_avg)):
            if df_nav.iloc[i, 4] > volSMA.iloc[i, 3] and \
			   close_avg.iloc[i-1, 0] < close_avg.iloc[i-1 ,1] and \
               close_avg.iloc[i, 0] > close_avg.iloc[i, 1]:
                signal.iloc[i,0] = 1
            elif df_nav.iloc[i, 4] < volSMA.iloc[i ,3] and \
			   close_avg.iloc[i-1, 0] > close_avg.iloc[i-1 ,1] and \
               close_avg.iloc[i, 0] < close_avg.iloc[i, 1]:
                signal.iloc[i,0] = -1

        return signal


    def timing(self, df_nav):
		
        volSMA = self.compute_volume(df_nav)
        close_avg = self.compute_price(df_nav)
        signal = self.signal(volSMA,close_avg, df_nav)
        signal.replace(0,np.nan,inplace = True)
        signal.fillna(method = 'ffill', inplace = True)
        signal = signal.dropna()
		
        return signal

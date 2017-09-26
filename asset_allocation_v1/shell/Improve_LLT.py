#coding=utf8


import pandas as pd
import numpy as np


class Improve_LLT(object):

    #设置a为权重指数
    #设置b为权重指数
    def __init__(self, d, f):
        self.a = 2.0/(d+1)
        self.b = 2.0/(f+1)


    #列出每日llt的值
    def llt(self, df_nav, a, b):
        #print df_nav.head()
        Llt = pd.DataFrame({'LLT_short':0.0,'LLT_long':0.0},\
                            columns=['LLT_short','LLT_long'],index = df_nav.index)
        #print Llt.head()
        num = len(df_nav)
        Llt.iloc[1][0] = Llt.iloc[1][1] = df_nav.iloc[1][3]
        Llt.iloc[2][0] = Llt.iloc[2][1] = df_nav.iloc[2][3]
        for i in range(3, num):
            
            Llt.iloc[i][0] = (a-(a*a)/4)*df_nav.iloc[i][3]+((a*a)/2)*df_nav.iloc[i-1][3]\
                     -(a-3*(a*a)/4)*df_nav.iloc[i-2][3]+2*(1-a)*Llt.iloc[i-1][0]-(1-a)*\
                     (1-a)*Llt.iloc[i-2][0]

            Llt.iloc[i][1] = (b-(b*b)/4)*df_nav.iloc[i][3]+((b*b)/2)*df_nav.iloc[i-1][3]\
                     -(b-3*(b*b)/4)*df_nav.iloc[i-2][3]+2*(1-b)*Llt.iloc[i-1][1]-(1-b)*\
                     (1-b)*Llt.iloc[i-2][1]
        #print Llt.head()
        return Llt


    # 择时：
    #获得llt在当日该点切线k
    # k < 0 singnal=-1 通知用户赎回
    # k > 0 singnal=1  通知用户购买
    # k = 0 singnal=0  维持原状
    def signal(self, df_nav, Llt):
        Signal = pd.DataFrame({'tc_signal':[0]*len(Llt)}, index = df_nav.index)
        #avg_10 = df_nav.rolling(window=10).mean()
        #print avg_10.tail()
        #avg_240 = df_nav.rolling(window=240).mean()
        #diff = Llt.diff()
        num = len(Llt)
        for i in range(2, num):
            if Llt.iloc[i-1][0] < Llt.iloc[i-1][1] and Llt.iloc[i][0] > Llt.iloc[i][1]:
                #and diff.iloc[i][0] > 0 and diff.iloc[i][1] > 0 :
                #and diff.iloc[i-1][0] > 0 and diff.iloc[i-1][1] > 0 \
                Signal.iloc[i][0] = 1
                #if avg_10.iloc[i-1][3] < avg_240.iloc[i-1][3] and avg_10.iloc[i][3] > avg_240.iloc[i][3]:
                #if ((df_nav.iloc[i][1]+df_nav.iloc[i][2])/2) >= df_nav.iloc[i][3]:
                if df_nav.iloc[i][3] >= df_nav.iloc[i][0]:
                    Signal.iloc[i+1][0] = 1
                else:
                    Signal.iloc[i+1][0] = -1
            elif Llt.iloc[i-1][0] > Llt.iloc[i-1][1] and Llt.iloc[i][0] < Llt.iloc[i][1]:
                #and diff.iloc[i][0] < 0 and diff.iloc[i][1] < 0 :
                #and diff.iloc[i-1][0] < 0 and diff.iloc[i-1][1] < 0 \
                Signal.iloc[i][0] = -1
                #if avg_10.iloc[i-1][3] > avg_240.iloc[i-1][3] and avg_10.iloc[i][3] < avg_240.iloc[i][3]:
                #if ((df_nav.iloc[i][1]+df_nav.iloc[i][2])/2) <= df_nav.iloc[i][3]:
                if df_nav.iloc[i][3] <= df_nav.iloc[i][0]:
                    Signal.iloc[i+1][0] = -1
                else:
                    Signal.iloc[i+1][0] = 1
        return Signal



    def timing(self,df_nav):
        Llt = self.llt(df_nav, self.a, self.b)
        Signal = self.signal(df_nav,Llt)
        Signal.replace(0,np.nan, inplace = True)
        Signal.fillna(method='ffill',inplace = True)
        Signal.dropna(inplace = True)
        return Signal

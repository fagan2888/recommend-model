#coding=utf8

import pandas as pd
import numpy as np

class count_df_nav(object):
   
    def timing(self, df_nav):
        data = pd.DataFrame({'tc_open':df_nav.tc_open,'tc_close':df_nav.tc_close},\
                            columns=['tc_open','tc_close'], index = df_nav.index)
        close_diff = pd.DataFrame({'number':df_nav.tc_close.diff()}, index = df_nav.index)
        #print close_diff        
        count_rise = 0
        count_low = 0  
        num = len(df_nav)

        for i in range(0, num-5):
            if data.iloc[i][0] <= data.iloc[i][1]:
                if close_diff.iloc[i][0] > 0 and close_diff.iloc[i+1][0] > 0\
                   and close_diff.iloc[i+2][0] > 0 and close_diff.iloc[i+3][0] > 0 \
                   and close_diff.iloc[i+4][0] > 0:
                    count_rise = count_rise + 1
                
        '''for m in range(0, num-3):
            if data.iloc[i][0] >= data.iloc[i][1]:
                 for n in range(m, m+3):
                    if close_diff.iloc[n][0] >= 0:
                        continue
                    else:
                        count_low = count_low + 1'''
        for i in range(0, num-5):
            if data.iloc[i][0] >= data.iloc[i][1]:
                if close_diff.iloc[i][0] < 0 and close_diff.iloc[i+1][0] < 0\
                   and close_diff.iloc[i+2][0] < 0 and close_diff.iloc[i+3][0] < 0\
                   and close_diff.iloc[i+4][0] < 0:
                    count_low = count_low + 1
        
        count1 = 0
        count2 = 0
        b = data.pct_change().rolling(window=3).sum().shift(-3).fillna(0)
        
        for i in range(0, len(b)-3):
            
            if data.iloc[i][0] <= data.iloc[i][1]:
                if b.iloc[i][1] > 0 and b.iloc[i+1][1] >0 and b.iloc[i+2][1]>0:
                    count1 = count1 + 1
            elif data.iloc[i][0] >= data.iloc[i][1]:
                if b.iloc[i][1] < 0 and b.iloc[i+1][1] <0 and b.iloc[i+2][1]<0:
                    count2 = count2 + 1

        print 'num',num
        print 'count_rise',count_rise
        print 'count_low',count_low
        print 'count1',count1
        print 'count2',count2
                
                
                
                           

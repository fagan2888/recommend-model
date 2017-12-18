#coding=utf8

import pandas as pd
import numpy as np


feat07_df = pd.read_csv('user_prediction/feat.csv.20171209', index_col = ['ts_date'], parse_dates = ['ts_date'])
feat15_df = pd.read_csv('user_prediction/feat.csv.20171215', index_col = ['ts_date'], parse_dates = ['ts_date'])


feat07 = feat07_df.loc['2017-12-09']
feat15 = feat15_df.loc['2017-12-09']


feat07 = feat07.reset_index().set_index(['ts_uid'])
feat15 = feat15.reset_index().set_index(['ts_uid'])

for uid in feat07.index:
    print uid
    feat07_record = feat07.loc[uid]
    feat15_record = feat15.loc[uid]
    for index in feat07_record.index:
        #print feat07_record[index]
        #print pd.isnull(feat07_record[index])
        if pd.isnull(feat07_record[index]) and pd.isnull(feat15_record[index]):
            pass
        elif feat07_record[index] == feat15_record[index]:
            #print index, feat07_record[index], feat15_record[index]
            pass
        else:
            print index, feat07_record[index], feat15_record[index]
            pass

#print feat15

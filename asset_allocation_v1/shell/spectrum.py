from __future__ import division
import pandas as pd
import numpy as np
from cal_spec import cal_indic
#import os


def load_data():
    df = pd.read_csv('~/recommend_model/asset_allocation_v1/120000001_ori_min_data.csv', \
            index_col=0, parse_dates = True)
    df = df[1:]
    return df

def cal_spectrum(ori_data, qs = np.arange(-10, 10, 0.1), dtype = 1):
    window = 48
    if dtype == 2:
        window = 240
    dx_list = []
    df_list = []
    Rf_list = []
    date_list = []
    index = ori_data.index
    s_num = 238
    in_num = s_num + window
    e_num = len(ori_data)
    ret = np.array(ori_data['close'])
    while in_num < e_num:
        in_time = index[in_num -1]
        date_list.append(in_time.date())
        dx, df, Rf = cal_indic(ret[s_num:in_num], qs, dtype)
        dx_list.append(dx)
        df_list.append(df)
        Rf_list.append(Rf)
        s_num += window
        in_num += window

        print index[s_num]
        #print index[in_num]
        #print dx
        #print df

    #ori_data = ori_data.ix[window - 1:,['close', 'pct_chg']]
    #ori_data['dx'] = dx_list
    #ori_data['df'] = df_list

    result_dict = {
            'dx': dx_list,
            'df': df_list,
            'Rf': Rf_list
            }
    result_df = pd.DataFrame(result_dict, index = date_list)
    result_df.to_csv('~/recommend_model/asset_allocation_v1/dxdf_w.csv')

    return result_df

def res_sta():
    res = pd.read_csv('~/recommend_model/asset_allocation_v1/dxdf_w.csv', \
            index_col = 0)
    res_dict = {
            'floor':[],
            'upper':[],
            'return':[],
            }

    for i in range(0, 100, 10):
        floor = np.percentile(res.df, i)
        upper = np.percentile(res.df, i+10)
        count = res[(res.df > floor) & (res.df < upper)].mean()
        res_dict['floor'].append(floor)
        res_dict['upper'].append(upper)
        res_dict['return'].append(count.pct_chg1)

    res_df = pd.DataFrame(res_dict, columns = ['floor', 'upper', 'return'])
    res_df.to_csv('~/recommend_model/asset_allocation_v1/countdf.csv', \
            index = False)


def handle():
    data = load_data()
    data = cal_spectrum(data, dtype = 2)
    print data
#    data.to_csv('output/dxdf_day.csv')


if __name__ == '__main__':
    handle()
    #res_sta()

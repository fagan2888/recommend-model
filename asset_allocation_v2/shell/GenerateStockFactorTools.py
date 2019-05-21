#coding=utf-8
'''
Created on: May. 19, 2019
Author: Ning Yang
Contact: yangning@licaimofang.com
'''

import numpy as np
import pandas as pd
import statsmodels.api as sm

def closing_dt(x):
    if x.month == 3:
        return pd.Timestamp(x.year, 4, 30)
    elif x.month == 6:
        return pd.Timestamp(x.year, 8, 31)
    elif x.month == 9:
        return pd.Timestamp(x.year, 10, 31)
    elif x.month == 12:
        return pd.Timestamp(x.year+1, 4, 30)
    else:
        return pd.Timestamp(2035, 1, 1)  # 特殊标记


def lt_growth(data):
    data_t = data[~np.isnan(data)].copy()
    x = np.arange(data_t.shape[0]) + 1
    y = data_t
    ols_results = sm.OLS(y, sm.add_constant(x)).fit()
    growth_t = ols_results.params[1] / np.mean(data_t)
    return growth_t


def e_vol(data):
    data_t = data[~np.isnan(data)].copy()
    growth_list = list()
    for i in range(1, data_t.shape[0]):
        if data_t[i-1] != 0:
            growth_t = (data_t[i] - data_t[i-1]) / np.abs(data_t[i-1])
        else:
            growth_t = 0
        growth_list.append(growth_t)
    growth_array = np.array(growth_list)
    fun_var = np.cov(growth_array)
    fun_std = np.sqrt(fun_var)
    return fun_std


def select_fs(df_FS, trade_date):
    df_FS_t = df_FS.loc[data_fs.ann_date < trade_date].copy()
    df_FS_t.sort_values(by=['stock_id', 'ann_date'], ascending=False, inplace=True)
    df_FS_t.drop_duplicates(subset=['stock_id'], keep='first', inplace=True)    
    return df_FS_t


def z_score(df, columns, weight=None):
    df_t = df.copy()
    for i_column in columns:
        loc_t = df_t[i_column].isin([np.nan, - np.inf, np.inf])
        df_t.loc[loc_t, i_column] = np.nan
        # limit outliers
        df_t.loc[df_t[i_column] > df_t[i_column].mean(skipna=True) + 3 * df_t[i_column].std(skipna=True), i_column] = df_t[i_column].mean(skipna=True) + 3 * df_t[i_column].std(skipna=True)
        df_t.loc[df_t[i_column] < df_t[i_column].mean(skipna=True) - 3 * df_t[i_column].std(skipna=True), i_column] = df_t[i_column].mean(skipna=True) - 3 * df_t[i_column].std(skipna=True)
        # calculate factor exposures
        cap_weighted_mean = np.dot(df_t.loc[~loc_t, i_column], df_t.loc[~loc_t, weight]) / df_t.loc[~loc_t, weight].sum()
        equal_weighted_std = df_t.loc[~loc_t, i_column].std()
        df_t[i_column] = (df_t[i_column] - cap_weighted_mean) / equal_weighted_std
    return df_t



#coding=utf8

import pandas as pd


if __name__ == '__main__':

    domestic_macro_df      = pd.read_csv('./data/domestic_macro.csv', parse_dates = ['date'], index_col = ['date'], encoding = 'gbk')

    index_monthly_inc_df   = pd.read_csv('./data/index_monthly_inc.csv', parse_dates = ['date'], index_col = ['date'], encoding = 'gbk')

    #print index_monthly_inc_df

    print domestic_macro_df

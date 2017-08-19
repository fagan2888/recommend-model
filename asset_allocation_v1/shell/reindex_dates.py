#coding=utf8


import pandas as pd


if __name__ == '__main__':

    date_df = pd.read_csv('./dates.csv', index_col = ['date'], parse_dates = ['date'])

    t_df = pd.read_csv('./tc_timing_signal.csv', index_col = ['date'], parse_dates = ['date'])

    t_df = t_df.reindex(date_df.index).fillna(method = 'pad')

    t_df.to_csv('t_df.csv')
    print t_df

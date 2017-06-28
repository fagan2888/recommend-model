#coding=utf8

import pandas as pd
from pykalman import KalmanFilter


if __name__ == '__main__':


    index_df = pd.read_csv('./data/m1_m2_yoy.csv', index_col = ['date'], parse_dates = ['date'])
    #m1_m2_df = money_df.fillna(method = 'pad')
    #m1_m2_df = m1_m2_df.resample('M', how = 'last')
    #m1_m2_df = m1_m2_df.iloc[0 : -2]

    hs = index_df['000001.SH']
    dates = []
    vs = []
    for i in range(252 : len(hs)):

    #m1_m2_yoy_vs = m1_m2_df['m1_m2_yoy'].values
    #hs000001_vs = m1_m2_df['000001.SH'].values
    #kf = KalmanFilter(initial_state_mean = 0, n_dim_obs = 1)
    #(us, vs) = kf.em(m1_m2_yoy_vs).filter(m1_m2_yoy_vs)
    #m1_m2_df['m1_m2_yoy_kf'] = us
    #(us, vs) = kf.em(hs000001_vs).filter(hs000001_vs)
    #m1_m2_df['000001.SH_kf'] = us
    #print m1_m2_df
    #m1_m2_df.to_csv('tmp.csv')

#coding=utf8


import pandas as pd
import numpy as np


if __name__ == '__main__':

    gdp_df = pd.read_csv('./data/gdp.csv', index_col = ['date'], parse_dates = ['date'])
    tsf_df = pd.read_csv('./data/tsf.csv', index_col = ['date'], parse_dates = ['date'])

    gdp_df.index = gdp_df.index.strftime('%Y-%m')
    tsf_df.index = tsf_df.index.strftime('%Y-%m')

    gdp_df = gdp_df[['gdp']]

    df = pd.concat([tsf_df, gdp_df], axis = 1, join_axes = [tsf_df.index])
    df = df.interpolate()

    print df
    df.to_csv('tsf_gdp.csv')

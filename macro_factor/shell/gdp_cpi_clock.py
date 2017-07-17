#coding=utf8


import pandas as pd
import numpy as np



if __name__ == '__main__':

    gdp_cpi_df = pd.read_csv('./data/gdp_cpi.csv', parse_dates = ['date'], index_col = ['date'])

    index_df = pd.read_csv('./data/index.csv', parse_dates = ['date'], index_col = ['date'])
    dates = pd.date_range(index_df.index[0], index_df.index[-1])
    index_df = index_df.reindex(dates)
    index_df = index_df.fillna(method = 'pad')

    gdp_cpi_df = gdp_cpi_df.shift(1)

    gdp_cpi_df = gdp_cpi_df[['gdp_yoy','cpi_yoy']]

    gdp_cpi_df = gdp_cpi_df.reindex(index_df.index)

    gdp_cpi_df = gdp_cpi_df.shift(17)

    gdp_cpi_df = gdp_cpi_df.dropna()

    df = pd.concat([gdp_cpi_df, index_df], axis = 1, join_axes = [gdp_cpi_df.index])

    df = df.dropna()

    df['cycle'] = np.nan
    df['better_asset'] = np.nan

    df['SHIBOR3M.IR_inc'] = df['SHIBOR3M.IR'].pct_change().fillna(0.0).shift(-1)
    df['000300.SH_inc'] = df['000300.SH'].pct_change().fillna(0.0).shift(-1)
    df['H11001.CSI_inc'] = df['H11001.CSI'].pct_change().fillna(0.0).shift(-1)
    df['NH0100.NHF_inc'] = df['NH0100.NHF'].pct_change().fillna(0.0).shift(-1)

    dates = df.index
    for i in range(1, len(dates)):

        d = dates[i]
        pre_d = dates[i - 1]

        if df.loc[d, 'gdp_yoy'] >= df.loc[pre_d, 'gdp_yoy'] and df.loc[d, 'cpi_yoy'] <= df.loc[pre_d, 'cpi_yoy']:
            df.loc[d, 'cycle'] = 'recovery'
            df.loc[d, 'better_asset'] = '000300.SH'
        elif df.loc[d, 'gdp_yoy'] >= df.loc[pre_d, 'gdp_yoy'] and df.loc[d, 'cpi_yoy'] >= df.loc[pre_d, 'cpi_yoy']:
            df.loc[d, 'cycle'] = 'overheated'
            df.loc[d, 'better_asset'] = 'NH0100.NHF'
        elif df.loc[d, 'gdp_yoy'] <= df.loc[pre_d, 'gdp_yoy'] and df.loc[d, 'cpi_yoy'] >= df.loc[pre_d, 'cpi_yoy']:
            df.loc[d, 'cycle'] = 'stagflation'
            df.loc[d, 'better_asset'] = 'SHIBOR3M.IR'
        elif df.loc[d, 'gdp_yoy'] <= df.loc[pre_d, 'gdp_yoy'] and df.loc[d, 'cpi_yoy'] <= df.loc[pre_d, 'cpi_yoy']:
            df.loc[d, 'cycle'] = 'recession'
            df.loc[d, 'better_asset'] = 'H11001.CSI'


    print df
    df.to_csv('gdp_cpi_index_cycle.csv')

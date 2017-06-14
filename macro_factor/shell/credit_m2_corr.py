#coding=utf8

import pandas as pd


if __name__ == '__main__':

    df = pd.read_csv('./data/credit_m2.csv', index_col = ['date'], parse_dates = ['date'])
    df = df.rolling(window = 12).apply(lambda x : x[-1] / x[0] - 1)
    df = df.dropna()

    for i in range(12, len(df)):
        date = df.index[i]
        tmp_df = df[i - 12 : i]
        print date, tmp_df.corr().values[0][1]

#coding=utf8

import pandas as pd


if __name__ == '__main__':

    money_df = pd.read_csv('./data/money_policy.csv', index_col = ['date'], parse_dates = ['date'])
    money_df = money_df.fillna(method = 'pad')
    money_df = money_df.resample('M', how = 'last')


    index_df = pd.read_csv('./data/index.csv', index_col = ['date'], parse_dates = ['date'])
    index_df = index_df.resample('M', how = 'last')

    df = pd.concat([money_df, index_df], axis = 1, join_axes = [index_df.index])
    print df
    df.to_csv('tmp.csv')
    #print df.columns
    #df = df[['reverse_money']]
    #print df.dropna()

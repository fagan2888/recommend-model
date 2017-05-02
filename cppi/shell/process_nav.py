#coding=utf8


import pandas as pd


if __name__ == '__main__':

    df = pd.read_csv('./data/nav.csv', index_col = ['date'])
    df = df / df.iloc[0]
    print df
    df.to_csv('./data/nav.csv')

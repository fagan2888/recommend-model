#coding=utf8


import pandas as pd
import string
import statsmodels.api as sm
import numpy as np
import time


#data standardization
def factor_standard(df):

    n = 2
    index = df.index
    for i in index:
        vs = df.loc[i]
        nona_vs = vs.dropna()
        median = np.median(nona_vs)
        differences = []
        for v in nona_vs:
            differences.append(abs(v - median))
        diff_median = np.median(differences)
        std_vs = []
        uplimit = median + n * diff_median
        downlimit = median - n * diff_median
        #if downlimit < 0:
        #    downlimit = 0
        #print i, median, diff_median, uplimit, downlimit
        for v in vs:
            if np.isnan(v):
                std_vs.append(v)
            elif v > uplimit:
                std_vs.append(uplimit)
            elif v < downlimit:
                std_vs.append(downlimit)
            else:
                std_vs.append(v)
        df.loc[i] = std_vs

    return df
    #print df
    #df.to_csv('std_market_value.csv')


def momentum20(df):
    dfr = df.pct_change().fillna(0.0)
    result_df = dfr.rolling(window = 20).sum()
    return result_df


def momentum60(df):
    dfr = df.pct_change().fillna(0.0)
    result_df = dfr.rolling(window = 20).sum()
    return result_df


def momentum120(df):
    dfr = df.pct_change().fillna(0.0)
    result_df = dfr.rolling(window = 120).sum()
    return result_df


def momentum240(df):
    dfr = df.pct_change().fillna(0.0)
    result_df = dfr.rolling(window = 240).sum()
    return result_df

def std20(df):
    dfr = df.pct_change().fillna(0.0)
    result_df = dfr.rolling(window = 20).std()
    return result_df

def std60(df):
    dfr = df.pct_change().fillna(0.0)
    result_df = dfr.rolling(window = 60).std()
    return result_df

def std120(df):
    dfr = df.pct_change().fillna(0.0)
    result_df = dfr.rolling(window = 120).std()
    return result_df

def std240(df):
    dfr = df.pct_change().fillna(0.0)
    result_df = dfr.rolling(window = 240).std()
    return result_df


if __name__ == '__main__':

    #market_value_df = pd.read_csv('./data/stock_market_value.csv', index_col = ['date'], parse_dates = ['date'])
    #market_value_df = market_value_df * 10000
    #factor_standard(market_value_df)
    stock_price_adjust_df = pd.read_csv('./data/stock_price_adjust.csv', index_col = ['date'], parse_dates = ['date'])
    #print stock_price_adjust_df
    momentum20(stock_price_adjust_df)
    momentum60(stock_price_adjust_df)
    momentum120(stock_price_adjust_df)
    mom240_df = momentum240(stock_price_adjust_df)

    std20(stock_price_adjust_df)
    std60(stock_price_adjust_df)
    std120(stock_price_adjust_df)
    std240_df = std240(stock_price_adjust_df)
    std240_df = factor_standard(std240_df)

    print std240_df

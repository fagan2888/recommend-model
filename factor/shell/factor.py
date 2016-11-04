#coding=utf8


import pandas as pd
import string
import statsmodels.api as sm
import numpy as np
import time


#beta factor
def beta(stock_dfr, index_dfr):

    back = 252
    half_life = 63
    dates = stock_dfr.index

    betas = []
    ds    = []
    for i in range(back, len(dates) + 1):
        tmp_stock_dfr = stock_dfr.iloc[i - back: i,]
        tmp_index_dfr = index_dfr.loc[tmp_stock_dfr.index]
        d = dates[i - 1]
        beta = []
        for col in tmp_stock_dfr.columns:
            vs= tmp_stock_dfr[col]
            vs= vs.dropna()
            if len(vs) < 252 * 0.8:
                beta.append(np.nan)
            else:
                X = tmp_stock_dfr[col].values
                for j in range(0, len(X)):
                    life = len(X) - j - 1
                    X[j] = X[j] * (0.5 ** (life / half_life))
                X = sm.add_constant(X)
                y = tmp_index_dfr.values
                try:
                    model  = sm.OLS(y,X)
                    result = model.fit()
                    #print result.params
                    beta.append(result.params[1])
                except:
                    beta.append(np.nan)
        betas.append(beta)
        ds.append(d)
        print d

    beta_df = pd.DataFrame(betas, index = ds, columns = stock_dfr.columns)
    #print beta_df
    beta_df.index.name = 'date'
    beta_df.to_csv('./tmp/beta.csv')
    print 'beta done'
    return beta_df


def ht_momentum(stock_dfr, stock_turnover_df):

    w_df = stock_turnover_df
    df = w_df * stock_dfr
    one_month_df = pd.rolling_sum(df, 21)
    two_month_df = pd.rolling_sum(df, 21 * 2)
    three_month_df = pd.rolling_sum(df, 21 * 3)
    six_month_df = pd.rolling_sum(df, 21 * 6)
    twelve_month_df = pd.rolling_sum(df, 21 * 12)
    w_sum = 1.0 / 1 + 1.0 / 2 + 1.0 / 3 + 1.0 / 6 + 1.0 / 12
    w1 = 1.0 / 1 / w_sum
    w2 = 1.0 / 2 / w_sum
    w3 = 1.0 / 3 / w_sum
    w6 = 1.0 / 6 / w_sum
    w12= 1.0 / 12 / w_sum
    momentum_df = w1 * one_month_df + w2 * two_month_df + w3 * three_month_df + w6 * six_month_df + w12 * twelve_month_df
    momentum_df = momentum_df.iloc[252:]
    #print momentum_df
    momentum_df.to_csv('./tmp/momentum.csv')
    return momentum_df


def ht_std(stock_dfr):

    stock_dfr = stock_df.pct_change()

    std1m_df = stock_dfr.rolling(window=21).std()
    std2m_df = stock_dfr.rolling(window=21 * 2).std()
    std3m_df = stock_dfr.rolling(window=21 * 3).std()
    std6m_df = stock_dfr.rolling(window=21 * 6).std()
    std12m_df = stock_dfr.rolling(window=21 * 12).std()

    high_low_1m = stock_df.rolling(window = 21).max() / stock_df.rolling(window = 21).min()
    high_low_2m = stock_df.rolling(window = 21 * 2).max() / stock_df.rolling(window = 21 * 2).min()
    high_low_3m = stock_df.rolling(window = 21 * 3).max() / stock_df.rolling(window = 21 * 3).min()
    high_low_6m = stock_df.rolling(window = 21 * 6).max() / stock_df.rolling(window = 21 * 6).min()
    high_low_12m = stock_df.rolling(window = 21 * 12).max() / stock_df.rolling(window = 21 * 12).min()

    w_sum = 1.0 / 1 + 1.0 / 2 + 1.0 / 3 + 1.0 / 6 + 1.0 / 12
    w1 = 1.0 / 1 / w_sum
    w2 = 1.0 / 2 / w_sum
    w3 = 1.0 / 3 / w_sum
    w6 = 1.0 / 6 / w_sum
    w12= 1.0 / 12 / w_sum

    std_df = w1 * std1m_df + w2 * std2m_df + w3 * std3m_df + w6 * std6m_df + w12 * std12m_df + w1 * high_low_1m + w2 * high_low_2m + w3 * high_low_3m + w6 * high_low_6m + w12 * high_low_12m
    std_df = std_df.iloc[252:]
    #print std_df
    std_df.to_csv('./tmp/stock_dastd.csv')
    return std_df


#momentum factor
def momentum(stock_dfr):

    T = 504
    L = 21
    half_life = 126

    dfr = pd.rolling_sum(stock_dfr, L)
    cols = dfr.columns
    dates = dfr.index

    indexs = []
    j = T - 1
    while j >= 0:
        indexs.append(j)
        j = j - L

    momentums = []
    ds        = []
    for i in range(T, len(dates)):
        d = dates[i - 1]
        tmp_dfr = dfr.iloc[ i - T : i, ]
        log_dfr = np.log(1 + tmp_dfr)
        values = log_dfr.values
        trans_values = []
        for j in range(0, len(log_dfr)):
            life = len(dates) - j - 1
            w = 0.5 ** (life / half_life)
            trans_values.append(values[j] * w)
        log_dfr = pd.DataFrame(trans_values, index = log_dfr.index, columns = log_dfr.columns)
        log_dfr = log_dfr.iloc[indexs]
        momentum_df = np.sum(log_dfr)
        moment = []
        for col in cols:
            moment.append(momentum_df[col])
        print d
        momentums.append(moment)
        ds.append(d)

    mom_df = pd.DataFrame(momentums, index = ds, columns = cols)
    mom_df.index.name = 'date'
    mom_df.to_csv('./tmp/momentum.csv')

    print 'momentum done'
    return mom_df


def cap_size(stock_market_value_df):
    stock_market_value_df = np.log(stock_market_value_df)
    stock_market_value_df.index.name = 'date'
    stock_market_value_df.to_csv('./tmp/market_value.csv')
    print 'stock cap size done'
    return stock_market_value_df

def cube_size(stock_market_value_df):
    stock_market_value_df = np.power(stock_market_value_df, 3)
    stock_market_value_df.index.name = 'date'
    stock_market_value_df.to_csv('./tmp/cube_size.csv')
    print 'stock cube size done'
    return stock_market_value_df

def dastd(stock_dfr):
    back = 252
    stock_dastd = pd.rolling_std(stock_dfr, back)
    stock_dastd.index.name = 'date'
    stock_dastd.to_csv('./tmp/stock_dastd.csv')
    print 'stock dastd done'
    return stock_dastd


def cmra(stock_dfr):

    T = 12
    L = 21

    dfr   = pd.rolling_sum(stock_dfr, L)
    cols  = dfr.columns
    dates = dfr.index

    indexs = []
    j = T * L - 1
    while j >= 0:
        indexs.append(j)
        j = j - L

    cmras     = []
    ds        = []
    for i in range(T * L, len(dates)):
        d = dates[i - 1]
        log_dfr = np.log(1 + tmp_dfr)
        log_dfr = log_dfr.iloc[indexs]

        zts = []
        for j in range(1, T + 1):
            zt      = np.sum(log_dfr.iloc[0 : j,])
            zts.append(zt)

        print d
        momentums.append(moment)
        ds.append(d)


    mom_df = pd.DataFrame(momentums, index = ds, columns = cols)
    mom_df.index.name = 'date'
    mom_df.to_csv('./tmp/momentum.csv')

    print 'momentum done'
    return mom_df


def egrlf(stock_dfr):
    return 1


def bp(stock_bp):
    stock_bp.to_csv('./tmp/bp.csv')
    return stock_bp

def pe(stock_pe):
    stock_pe[stock_pe < 0] = np.nan
    stock_pe.to_csv('./tmp/pe.csv')
    return stock_pe

def liquidity(stock_turnover_df):

    tau = 21
    stom_df = np.log(pd.rolling_sum(stock_turnover_df, tau))

    codes = stom_df.columns
    dates = stom_df.index
    T = 3
    indexs = [-1, -1 - tau, -1 - tau * 2]
    values = []
    ds     = []
    for i in range(tau * T, len(dates)):
        d = dates[i - 1]
        tmp_df = stom_df.iloc[i - 1 - tau * (T - 1): i, ]
        l = len(tmp_df)
        tmp_df = tmp_df.dropna(axis = 1, thresh = (int)(0.8 * l))
        tmp_df = tmp_df.iloc[indexs]
        tmp_df = np.log(np.sum(np.exp(tmp_df)) / T)
        vs = []
        for code in codes:
            if code in tmp_df.index:
                vs.append(tmp_df[code])
            else:
                vs.append(np.nan)
        values.append(vs)
        ds.append(d)
        print d
    stoq_df = pd.DataFrame(values, index = ds, columns = codes)


    codes = stom_df.columns
    dates = stom_df.index
    T = 12
    indexs = []
    for i in range(0, T):
        indexs.append(-1 - tau * i)
    values = []
    ds     = []
    for i in range(tau * T, len(dates)):
        d = dates[i - 1]
        tmp_df = stom_df.iloc[i - 1 - (T - 1) * tau: i, ]
        l = len(tmp_df)
        tmp_df = tmp_df.dropna(axis = 1, thresh = (int)(0.8 * l))
        tmp_df = tmp_df.iloc[indexs]
        tmp_df = np.log(np.sum(np.exp(tmp_df)) / T)
        vs = []
        for code in codes:
            if code in tmp_df.index:
                vs.append(tmp_df[code])
            else:
                vs.append(np.nan)
        values.append(vs)
        ds.append(d)
        print d
    stoa_df = pd.DataFrame(values, index = ds, columns = codes)


    liquidity_df = 0.35 * stom_df + 0.35 * stoq_df + 0.30 * stoa_df
    liquidity_df.index.name = 'date'
    #liquidity_df = liquidity_df.dropna( thresh = (int)(0.5 * len(liquidity_df.columns)))
    liquidity_df.to_csv('./tmp/liquidity.csv')

    return liquidity


if __name__ == '__main__':


    stock_df              = pd.read_csv('./data/stock_price_adjust.csv', index_col = 'date', parse_dates = ['date'])
    stock_market_value_df = pd.read_csv('./data/stock_market_value.csv', index_col = 'date', parse_dates = ['date'])
    stock_bp_df           = pd.read_csv('./data/stock_bp.csv', index_col = 'date', parse_dates = ['date'])
    stock_pe_df           = pd.read_csv('./data/stock_pe.csv', index_col = 'date', parse_dates = ['date'])
    stock_turnover_df     = pd.read_csv('./data/stock_turnover.csv', index_col = 'date', parse_dates = ['date'])
    index_df              = pd.read_csv('./data/index_price.csv', index_col = 'date', parse_dates = ['date'])

    index_df = index_df[['000300']]
    stock_dfr = stock_df.pct_change()
    index_dfr = index_df.pct_change().fillna(0.0)
    stock_market_value_df = stock_market_value_df.fillna(method = 'pad')
    #stock_market_value_df = stock_market_value_df[stock_df.columns]
    #stock_bp_df = stock_bp_df[stock_df.columns]

    #beta(stock_dfr, index_dfr)
    #ht_momentum(stock_dfr, stock_turnover_df)
    #momentum(stock_dfr)
    #cap_size(stock_market_value_df)
    #cube_size(stock_market_value_df)
    pe(stock_pe_df)
    #dastd(stock_dfr)
    #ht_std(stock_df)
    #bp(stock_bp_df)
    #liquidity(stock_turnover_df)

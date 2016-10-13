#coding=utf8


import pandas as pd
import numpy  as np
import statsmodels.api as sm


def alpha_beta(factor_index_df, fund_nav_df):

    fund_nav_dfr     = fund_nav_df.pct_change()
    factor_dates     = factor_index_df.index.values
    fund_nav_dates   = fund_nav_dfr.index.values
    dates = list(set(factor_dates) & set(fund_nav_dates))
    dates.sort()
    fund_nav_dfr     = fund_nav_dfr.loc[dates]
    fund_nav_dfr.dropna(axis = 1, inplace = True)
    factor_index_df = factor_index_df.loc[dates]

    X = factor_index_df.values
    X = sm.add_constant(X)

    cols = list(factor_index_df.columns.values)
    cols.insert(0, 'alpha')
    cols.insert(1, 'rsquared_adj')
    #print fund_nav_dfr
    betas = []
    codes = fund_nav_dfr.columns
    cs    = []
    for code in codes:
        y = fund_nav_dfr[code].values
        model  = sm.OLS(y,X)
        result = model.fit()
        #print code, result.params
        alpha = result.params[0]
        rsquared_adj = result.rsquared_adj
        #if alpha < 0 or rsquared_adj < 0:
        #    continue
        cs.append(code)
        #print code, result.pvalues, result.rsquared_adj
        #print result.summary()
        #print code, result.rsquared_adj
        vs = []
        vs.append(alpha)
        vs.append(rsquared_adj)
        for i in range(1, len(result.params)):
            vs.append(result.params[i])
        betas.append(vs)

    alpha_beta_df = pd.DataFrame(betas, index = cs, columns = cols)
    alpha_beta_df.dropna(inplace = True)
    alpha_beta_df['zscore_alpha']        = zscore(alpha_beta_df['alpha'].values)
    alpha_beta_df['zscore_rsquared_adj'] = zscore(alpha_beta_df['rsquared_adj'].values)
    alpha_beta_df.to_csv('./tmp/alpha_beta.csv')
    return alpha_beta_df


def zscore(vs):
    mean = np.mean(vs)
    std  = np.std(vs)
    zscore_vs = []
    for v in vs:
        zscore_vs.append((v - mean) / std)
    return zscore_vs


def correl(factor_index_df, fund_nav_df):

    fund_nav_dfr     = fund_nav_df.pct_change()
    factor_dates     = factor_index_df.index.values
    fund_nav_dates   = fund_nav_dfr.index.values
    dates = list(set(factor_dates) & set(fund_nav_dates))
    dates.sort()
    fund_nav_dfr     = fund_nav_dfr.loc[dates]
    fund_nav_dfr.dropna(axis = 1, inplace = True)
    factor_index_df = factor_index_df.loc[dates]

    df = pd.concat([factor_index_df, fund_nav_dfr] , axis = 1)
    corr_df = df.corr()
    corr_df = corr_df[factor_index_df.columns]
    #print corr_df
    corr_df.to_csv('./tmp/corr.csv')
    return corr_df


    '''
    print factor_index_df
    codes = fund_nav_dfr.columns
    for code in codes:
        y = fund_nav_dfr[code].values
    '''

if __name__ == '__main__':


    factor_index_df  = pd.read_csv('./tmp/factor_index.csv', index_col = 'date', parse_dates = ['date'])
    #factor_index_df  = pd.read_csv('./tmp/factor_index_diff.csv', index_col = 'date', parse_dates = ['date'])
    fund_nav_df      = pd.read_csv('./data/fund_value.csv', index_col = 'date', parse_dates = ['date'])
    factor_index_df.dropna(inplace = True)


    diff_cols       = ['beta', 'market_value', 'momentum', 'dastd', 'bp', 'liquidity']
    factor_index_df = factor_index_df[diff_cols]
    alpha_beta_df = alpha_beta(factor_index_df, fund_nav_df)

    #print alpha_beta_df
    correl(factor_index_df, fund_nav_df)
    #print factor_index_df

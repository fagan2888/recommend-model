#coding=utf8


import pandas as pd
import numpy  as np
import datetime
import statsmodels.api as sm
import time


rf = 0.03 / 252
back = 252

notna_stock_ratio = 0.9
month     = 21
cmra_T    = 12
cmra_tau  = 21
stom_back = 21
stoq_back = 21 * 3
stoa_back = 21 * 12
stoq_t    = 3
stoa_t    = 12


 
def residual_volatility(df, windadf):


    #########################################################################
    ## daily standard deviation
    dfr = df.pct_change()
    std_df = pd.rolling_std(dfr.fillna(0.0), back)
    dates = dfr.index
    data  = []
    for i in range(back, len(dates)):
        d  = dates[i]
        cols = dfr.iloc[i - back :i , : ].dropna(axis = 1, thresh = (int)(notna_stock_ratio * back)).columns
        tmp_std_df = std_df.loc[d, cols]
        codes = set(tmp_std_df.index.values)
        vs = []
        for col in dfr.columns:
            if col in codes:
                vs.append(tmp_std_df[col])
            else:
                vs.append(np.nan)
        data.append(vs)
    dastd_df = pd.DataFrame(data, index = dates[back : ], columns = dfr.columns)
    dastd_df.to_csv('./tmp/dastd.csv')
        #print d, tmp_std_df
    #########################################################################

    #########################################################################
    ## cumulative range
    '''
    dfr = np.log(df / df.shift(21))
    dates = dfr.index

    for i in range(back, len(dates)):
        d = dates[i]
        tmp_dfr = dfr.iloc[i - back:i,:] 
        tmp_dfr = tmp_dfr.dropna(axis = 1)
        zt_dict = {}
        cmra = {}
        for col in tmp_dfr.columns:
            for j in range(0, len(tmp_dfr)):
                if j % cmra_tau == 0:
                    zt = zt_dict.setdefault(col, [])
                    if len(zt) > 0:
                        zt.append(zt[-1] + tmp_dfr.loc[tmp_dfr.index[j], col])
                    else:
                        zt.append(tmp_dfr.loc[tmp_dfr.index[j], col])
            zt = zt_dict[col]
            #cmra[col] = np.log(1 + np.max(zt)) - np.log(1 + np.min(zt))
            cmra[col] = np.max(zt) - np.min(zt)
        for k,v in cmra.items():
            print d, k, v
    '''
    #########################################################################


    #########################################################################
    ## higsma
    dfr = df.pct_change()
    windadfr = windadf.pct_change().fillna(0.0) 
    dates = dfr.index
    data = []
    #for i in range(back, len(dates)):
    for i in range(back, len(dates)):
        d       = dates[i]
        tmp_dfr = dfr.iloc[i - back :i , : ]
        tmp_dfr = tmp_dfr.dropna(axis = 1, thresh = (int)(notna_stock_ratio * back)).fillna(0.0)
        vs = []
        for col in dfr.columns:
            if col in set(tmp_dfr.columns.values):
                X = tmp_dfr[col].values
                y = windadfr.loc[tmp_dfr.index].values
                X = sm.add_constant(X)
                model = sm.OLS(y, X)
                ret = model.fit()
                residual = np.matrix(y) - np.matrix(X).dot(np.matrix(ret.params).T)
                vs.append(np.std(residual))
            else:
                vs.append(np.nan)
        data.append(vs)
    higsma_df = pd.DataFrame(data, index = dates[back : ], columns = dfr.columns)
    higsma_df.to_csv('./tmp/higsma.csv')
    #########################################################################

    residual_volatility_df = 0.75 * dastd_df + 0.1 * higsma_df
    residual_volatility_df.to_csv('./tmp/residual_volatility.csv')

    return residual_volatility_df


def liquidity(df):

    stom_df = np.log(pd.rolling_sum(df, stom_back))

    dates   = stom_df.index 
    exp_stom_df = np.exp(stom_df)

    #######################################################################
    ##stom
    data = []
    for i in range(0 , len(dates)):
        d = dates[i]
        tmp_stom_df = stom_df.iloc[i,:]
        tmp_stom_df = tmp_stom_df.dropna()
        vs = []
        code_set = set(tmp_stom_df.index.values)
        for col in stom_df.columns:
            if col in code_set:
                vs.append(tmp_stom_df[col])
            else:
                vs.append(np.nan)
        data.append(vs)
    stom_indicator_df = pd.DataFrame(data, index = dates, columns = stom_df.columns)
    stom_indicator_df.to_csv('./tmp/stom.csv')
    ##########################################################################


    ######################################################################
    ##stoq
    data = []
    for i in range(stoq_back, len(dates)):
        d = dates[i]
        j = 0
        tmp_df = None
        while j < stoq_t:
            tmp_exp_stom_df = exp_stom_df.iloc[ i - j * month,:] 
            if tmp_df is None:
                tmp_df = tmp_exp_stom_df
            else:
                tmp_df = tmp_df + tmp_exp_stom_df
            j = j + 1
        tmp_df = np.log(tmp_df.dropna() / stoq_t)
        vs = []
        code_set = set(tmp_df.index.values)
        for col in exp_stom_df.columns:
            if col in code_set:
                vs.append(tmp_df[col])
            else:
                vs.append(np.nan)
        data.append(vs)
    stoq_indicator_df = pd.DataFrame(data, index = dates[stoq_back:], columns = exp_stom_df.columns)
    stoq_indicator_df.to_csv('./tmp/stoq.csv')
    ############################################################################ 


    ###########################################################################
    ##stoa
    data = []
    for i in range(stoa_back, len(dates)):
        d = dates[i]
        j = 0
        tmp_df = None
        while j < stoa_t:
            tmp_exp_stom_df = exp_stom_df.iloc[ i - j * month,:]    
            if tmp_df is None:
                tmp_df = tmp_exp_stom_df
            else:
                tmp_df = tmp_df + tmp_exp_stom_df
            j = j + 1
        tmp_df = np.log(tmp_df.dropna() / stoa_t)
        vs = []
        code_set = set(tmp_df.index.values)
        for col in exp_stom_df.columns:
            if col in code_set:
                vs.append(tmp_df[col])
            else:
                vs.append(np.nan)
        data.append(vs)
    stoa_indicator_df = pd.DataFrame(data, index = dates[stoa_back:], columns = exp_stom_df.columns)
    stoa_indicator_df.to_csv('./tmp/stoa.csv')
    ##########################################################################

    liquidity_df = 0.35 * stom_indicator_df + 0.35 * stoq_indicator_df + 0.30 * stoa_indicator_df
    return liquidity_df



if __name__ == '__main__':

    #df = pd.read_csv('./data/stock_price.csv', index_col = 'date', parse_dates = ['date'])
    #winda_df = pd.read_csv('./data/windA.csv', index_col = 'date', parse_dates = ['date'])
    #residual_volatility(df, winda_df)
    df = pd.read_csv('./data/stock_turnover.csv', index_col = 'date', parse_dates = ['date'])
    liquidity(df)

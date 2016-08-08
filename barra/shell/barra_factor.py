#coding=utf8


import pandas as pd
import numpy  as np
import datetime
import statsmodels.api as sm
import time


rf = 0.03 / 252
back = 252


notna_stock_ratio = 0.95


month     = 21
cmra_T    = 12
cmra_tau  = 21
stoq_back = 21 * 3
stoa_back = 21 * 12
stoq_t    = 3
stoa_t    = 12


 
def residual_volatility(df, windadf):


    #########################################################################
    ## daily standard deviation
    '''
    dfr = df.pct_change()
    std_df = pd.rolling_std(dfr.fillna(0.0), back)
    dates = dfr.index
    for i in range(back, len(dates)):
        d  = dates[i]
        cols = dfr.iloc[i - back :i , : ].dropna(axis = 1, thresh = (int)(notna_stock_ratio * back)).columns
        tmp_std_df = std_df.loc[d, cols]
        print tmp_std_df
        #print d, tmp_std_df
    '''
    #########################################################################

    #########################################################################
    ## cumulative range
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
    #########################################################################


    #########################################################################
    ## higsma
    '''
    dfr = df.pct_change()
    windadfr = windadf.pct_change().fillna(0.0) 
    dates = dfr.index
    #for i in range(back, len(dates)):
    for i in range(1000, len(dates)):
        d       = dates[i]
        tmp_dfr = dfr.iloc[i - back :i , : ]
        tmp_dfr = tmp_dfr.dropna(axis = 1, thresh = (int)(notna_stock_ratio * back)).fillna(0.0)
        for col in tmp_dfr.columns: 
            X = tmp_dfr[col].values
            y = windadfr.loc[tmp_dfr.index].values
            X = sm.add_constant(X)
            model = sm.OLS(y, X)
            ret = model.fit()
            residual = np.matrix(y) - np.matrix(X).dot(np.matrix(ret.params).T)
            print d, col, np.std(residual)
    '''
    #########################################################################


    #print dates[back]


        #for col in std_df.columns:
        #    if col not in set(cols):
        #        std_df.loc[d, col] = np.nan
        #print std_df.iloc[i, : ]
    #print std_df

    return 0




def liquidity(df):

    stom_df = np.log(pd.rolling_sum(df, 21))

    dates   = stom_df.index 
    exp_stom_df = np.exp(stom_df)

    #######################################################################
    ##stom
    '''
    for i in range(0 , len(dates)):
        d = dates[i]
        tmp_stom_df = stom_df.iloc[i,:]
        tmp_stom_df = tmp_stom_df.dropna()
        for code in tmp_stom_df.index:
            print d, code, tmp_stom_df[code]  
    '''
    ##########################################################################

    ######################################################################
    ##stoq
    '''
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
        for code in tmp_df.index:
            print d, code , tmp_df[code]
    '''
    ############################################################################ 


    ###########################################################################
    ##stoa
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
        for code in tmp_df.index:
            print d, code , tmp_df[code]
    ##########################################################################
    return 0.0


if __name__ == '__main__':

    #df = pd.read_csv('./data/stock_price.csv', index_col = 'date', parse_dates = ['date'])
    #winda_df = pd.read_csv('./data/windA.csv', index_col = 'date', parse_dates = ['date'])
    #residual_volatility(df, winda_df)
    df = pd.read_csv('./data/stock_turnover.csv', index_col = 'date', parse_dates = ['date'])
    liquidity(df)

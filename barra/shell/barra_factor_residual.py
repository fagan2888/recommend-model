#coding=utf8

import pandas as pd
import numpy  as np
import datetime
import statsmodels.api as sm


rf = 0.03 / 252
back = 252

notna_stock_ratio = 0.95

cmra_T   = 12
cmra_tau = 21


 
def residual_volatility(df, windadf):


    #########################################################################
    ## daily standard deviation
    dfr = df.pct_change()
    std_df = pd.rolling_std(dfr.fillna(0.0), back)
    dates = dfr.index
    for i in range(back, len(dates)):
        d  = dates[i]
        cols = dfr.iloc[i - back :i , : ].dropna(axis = 1, thresh = (int)(notna_stock_ratio * back)).columns
        tmp_std_df = std_df.loc[d, cols]
        for i in range(0, len(tmp_std_df.index)):
            print d, tmp_std_df.index[i], tmp_std_df.loc[tmp_std_df.index[i]]
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
            cmra[col] = np.log(1 + np.max(zt)) - np.log(1 + np.min(zt))
        print d, cmra
    '''
    #########################################################################

    #########################################################################
    ## higsma
    '''
    dfr = df.pct_change()
    windadfr = windadf.pct_change().fillna(0.0) 
    dates = dfr.index
    #for i in range(back, len(dates)):
    for i in range(back, len(dates)):
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


if __name__ == '__main__':

    #notna_stock_dict = notna_stock()
    df = pd.read_csv('./data/stock_price.csv', index_col = 'date', parse_dates = ['date'])
    winda_df = pd.read_csv('./data/windA.csv', index_col = 'date', parse_dates = ['date'])
    #df = df.dropna(axis=1, thresh = 2000)
    #print df
    residual_volatility(df, winda_df)

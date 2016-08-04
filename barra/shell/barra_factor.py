#coding=utf8

import pandas as pd
import numpy  as np
import datetime

rf = 0.03 / 252
back = 252

notna_stock_ratio = 0.95

cmra_T   = 12
cmra_tau = 21


 
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
        print d, tmp_std_df
    '''
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
                zt = zt_dict.setdefault(col, [])
                if j % cmra_tau == 0:
                    if len(zt) > 0:
                        zt.append(zt[-1] + tmp_dfr.loc[tmp_dfr.index[j], col])
                    else:
                        zt.append(tmp_dfr.loc[tmp_dfr.index[j], col])
            cmra[col] = np.log(1 + np.max(zt_dict[col])) - np.log(1 + np.min(zt_dict[col]))
        print d, cmra
    '''
    #########################################################################

    #########################################################################
    ## higsma
    dfr = df.pct_change()
    dates = dfr.index
    for i in range(back, len(dates)):
        d       = dates[i]
        cols    = dfr.iloc[i - back :i , : ].dropna(axis = 1, thresh = (int)(notna_stock_ratio * back)).columns
        tmp_dfr = dfr.loc[dates[i - back:i], cols]
         
        
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
    #df = df.dropna(axis=1)
    #print df
    residual_volatility(df, windadf)

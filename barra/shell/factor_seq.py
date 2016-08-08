#conding=utf8


import pandas as pd
import string
import numpy  as np
import datetime


def ind_seq(indicators):

    inds = {}
    for indicator in indicators:
        vec = indicator.strip().split()
        date = vec[0].strip()
        code = vec[1]
        v    = string.atof(vec[2])
        code_v = inds.setdefault(date, {})
        code_v[code] = v
    return inds



def residual_volatility():

    lines = open('./data/dastd','r')
    dastd_ind = ind_seq(lines)
    print 'dastd done'

    lines = open('./data/cmra','r')
    cmra_ind = ind_seq(lines)
    print 'cmra done'

    lines = open('./data/hsigma','r')
    hsigma_ind = ind_seq(lines)
    print 'hsigma done'


    dates = set(dastd_ind.keys()) & set(cmra_ind.keys()) & set(hsigma_ind.keys())
    dates = list(dates)
    dates.sort()

    inds = {}
   
 
    for d in dates:
        dastd  = dastd_ind[d] 
        cmra   = cmra_ind[d] 
        hsigma = hsigma_ind[d] 
        
        codes  = set(dastd.keys()) & set(cmra.keys()) & set(hsigma.keys())
       
        code_v = inds.setdefault(d, {})
        for code in codes:
            code_v[code] = 0.75 * dastd[code] + 0.1 * hsigma[code]     
            print d, code, 0.75 * dastd[code] + 0.1 * hsigma[code]
  
    return inds 



def liquidity():

    lines = open('./data/stom','r')
    stom_ind = ind_seq(lines)
    print 'stom done'

    lines = open('./data/stoq','r')
    stoq_ind = ind_seq(lines)
    print 'stoq done'

    lines = open('./data/stoa','r')
    stoa_ind = ind_seq(lines)
    print 'stoa done'


    dates = set(stom_ind.keys()) & set(stoq_ind.keys()) & set(stoa_ind.keys())
    dates = list(dates)
    dates.sort()

    inds = {}
    
    for d in dates:
        stom  = stom_ind[d] 
        stoq   = stoq_ind[d] 
        stoa = stoa_ind[d] 

        codes  = set(stom.keys()) & set(stoq.keys()) & set(stoa.keys())
 
        code_v = inds.setdefault(d, {})
        for code in codes:
            code_v[code] = 0.35 * stom[code] + 0.35 * stoq[code] + 0.30 * stoa[code]
            print d, code, 0.35 * stom[code] + 0.35 * stoq[code] + 0.30 * stoa[code]

    return inds 




def factor_seq(df, lines):

    dfr = df.pct_change().fillna(0.0)
    factor_ind = ind_seq(lines)

    dates = factor_ind.keys()
    dates.sort()

    low_factor_vs  = []
    high_factor_vs = []
    
    
    for d in dates:

        ind = factor_ind[d]

        sorted_ind = sorted(ind.iteritems(), key=lambda ind : ind[1], reverse = False)
        length = len(sorted_ind)
        high_v = sorted_ind[(int)(0.95 * length)][1]
        low_v  = sorted_ind[(int)(0.05 * length)][1]
  
        ind = {}
        for k,v in sorted_ind:
            if v >= high_v:
                v = high_v
            elif v <= low_v:
                v = low_v
            ind[k] = (v - low_v) / (high_v - low_v) + 1


        median = np.median(ind.values()) 

        low_sum  = 0
        high_sum = 0

        for k,v in ind.items():
            if v < median:
                low_sum = low_sum + 1.0 / v
            else:
                high_sum = high_sum + v


        low_vs  = []
        high_vs = []
        for col in dfr.columns:
            if ind.has_key(col):
                v = ind[col]
                if v < median:
                    low_vs.append(1.0 / v / low_sum)
                    high_vs.append(0.0)
                else:
                    low_vs.append(0.0)
                    high_vs.append(v / high_sum)
            else:
                    low_vs.append(0.0)
                    high_vs.append(0.0)
   
   
        low_factor_vs.append(low_vs)
        high_factor_vs.append(high_vs)
         
    low_df  = pd.DataFrame(low_factor_vs,  index = dates, columns=dfr.columns)
    high_df = pd.DataFrame(high_factor_vs, index = dates, columns=dfr.columns)


    low_df.to_csv('low_df.csv')
    high_df = (high_df.shift(1) * dfr).sum(axis = 1, skipna = True)
    low_df  = (low_df.shift(1) * dfr).sum(axis = 1, skipna = True)

    #print high_df.shift(1)
    #print low_df.shift(1)
    return high_df, low_df



if __name__ == '__main__':

    df = pd.read_csv('./data/stock_price.csv', index_col = 'date', parse_dates = ['date'])
    #residual_volatility()
    lines = open('./data/liquidity','r')
    high_df, low_df = factor_seq(df, lines)

    liquidity_high_low_df = pd.concat([high_df, low_df] ,axis = 1, join_axes = [high_df.index])
    liquidity_high_low_df.to_csv('high_low_liquidity')
 
    #lines = open('./data/residual_volatility','r')
    #high_df, low_df = factor_seq(df, lines)
    #residual_high_low_df = pd.concat([high_df, low_df] ,axis = 1, join_axes = [high_df.index])
    #residual_high_low_df.to_csv('high_low_residual.csv')

    #liquidity()
    #print df

#coding=utf8

import pandas as pd


if __name__ == '__main__':

    index = pd.read_csv('./data/index.csv', index_col = 'date', parse_dates = ['date'])
    #index = index.iloc[:,0:-1]
    index = index.fillna(method = 'pad').dropna()
    index = index / index.iloc[0]
    riskfixpdf = pd.read_csv('./allpdf.csv', index_col = 'date', parse_dates = ['date'])
    #riskfixpdf = riskfixpdf.iloc[:,0:-1]
    #print riskfixpdf
    markowitzpdf = pd.read_csv('./robustmarkowitzposition.csv', index_col = 'date', parse_dates = ['date'])

    markowitzpdf = markowitzpdf.reindex(riskfixpdf.index)
    markowitzpdf = markowitzpdf.fillna(method = 'pad')
    #print markowitzpdf
    #print riskfixpdf

    pdf = riskfixpdf * markowitzpdf
    df_inc = index.pct_change().fillna(0.0)
    df_inc = df_inc.reindex(pdf.index)
    df_inc = df_inc.fillna(0.0)
    dfr = pdf * df_inc
    dfr['asset'] = dfr.apply(lambda x: x.sum(), axis = 1)
    #print pdf
    vdf = (dfr + 1).cumprod()

    markowitzvdf = pd.read_csv('./robustmarkowitznav.csv', index_col = 'date', parse_dates = ['date'])
    dates = vdf.index & markowitzvdf.index
    vdf = vdf.loc[dates]
    markowitzvdf = markowitzvdf.loc[dates]
    vdf = pd.concat([vdf, markowitzvdf], join_axes = [vdf.index], axis = 1)
    vdf = vdf / vdf.iloc[0]
    print vdf
    vdf.to_csv('asset.csv')

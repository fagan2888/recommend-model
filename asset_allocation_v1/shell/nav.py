#coding=utf8

import pandas as pd


if __name__ == '__main__':

    index = pd.read_csv('./data/index.csv', index_col = 'date', parse_dates = ['date'])
    #index = index.iloc[:,0:-1]
    index = index.fillna(method = 'pad').dropna()
    index = index.resample('W-FRI').last()
    index = index / index.iloc[0]
    df_inc = index.pct_change().fillna(0.0)
    #riskfixpdf = pd.read_csv('./allpdf.csv', index_col = 'date', parse_dates = ['date'])
    #riskfixpdf = riskfixpdf.iloc[:,0:-1]
    #print riskfixpdf
    pdf = pd.read_csv('./robustmarkowitzposition.csv', index_col = 'date', parse_dates = ['date'])

    wss = []
    for d in pdf.index:
        ws = pdf.loc[d]
        if len(wss) == 0:
            wss.append(ws)
        else:
            pws = wss[-1]
            if sum(abs(ws - pws)) > 0.2:
                wss.append(ws)
            else:
                wss.append(pws)

    pdf = pd.DataFrame(wss, index = pdf.index, columns = pdf.columns)
    presult = pdf.rolling(window = 2, min_periods = 1).apply(lambda x : x[1] - x[0] if len(x) > 1 else x[0])
    presult = presult.abs().sum(axis = 1).to_frame('turnover').sum()
    print presult

    df_inc = df_inc.loc[pdf.index]
    dfr = pdf * df_inc
    dfr['asset'] = dfr.apply(lambda x: x.sum(), axis = 1)
    dfr.iloc[0] = 0
    #print pdf
    vdf = (dfr + 1).cumprod()
    print vdf
    vdf.to_csv('asset.csv')
    pdf.to_csv('pdf.csv')
    #print markowitzpdf
    #print riskfixpdf
    presult = pdf.rolling(window = 2, min_periods = 1).apply(lambda x : x[1] - x[0] if len(x) > 1 else x[0])
    presult = presult.abs().sum(axis = 1).to_frame('turnover').sum()
    print presult


    '''
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
    '''

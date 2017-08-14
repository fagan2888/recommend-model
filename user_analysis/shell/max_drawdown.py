#coding=utf8


import numpy as np
import pandas as pd



if __name__ == '__main__':

    df = pd.read_csv('./tmp/portfolio_nav.csv', index_col = ['date'], parse_dates = ['date'])

    datas = []
    for col in df.columns:
        df['cummax'] = df[col].cummax()
        df['drawdown'] = 1 - df[col] / df['cummax']
        max_drawdown = max(df['drawdown'])
        df = df[df.index >= '2016-07-01']
        df = df[df.index <= '2017-07-31']
        print col, max_drawdown
        tmp_df = df.resample('M').last()
        tmp_df.loc[df.index[0]] = df.iloc[0]
        tmp_df = tmp_df.sort_index()
        tmp_dfr = tmp_df.pct_change()
        dates = tmp_dfr.index
        for i in range(0, len(dates) - 1):
            tdf = df[df.index >= dates[i]]
            tdf = tdf[tdf.index <= dates[i + 1]]
            tdf['cummax'] = tdf[col].cummax()
            tdf['drawdown'] = 1 - tdf[col] / tdf['cummax']
            max_drawdown = max(tdf['drawdown'])
            #print dates[i + 1], col, tmp_dfr.iloc[i + 1][col], max_drawdown
            datas.append([dates[i + 1], col, tmp_dfr.iloc[i + 1][col], max_drawdown])
        #print col, tmp_dfr[col]
    df = pd.DataFrame(datas)
    df.columns = ['date', 'risk_level', 'r', 'max_drawdown']

    rdf = df.iloc[:,[0,1,2]]
    rdf = rdf.set_index(['date','risk_level'])
    rdf = rdf.unstack()
    rdf.to_csv('./tmp/r.csv')

    df = df.iloc[:,[0,1,3]]
    df = df.set_index(['date','risk_level'])
    df = df.unstack()
    df.to_csv('./tmp/max_drawdown.csv')

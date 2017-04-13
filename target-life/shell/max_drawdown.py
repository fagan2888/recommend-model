#coding = utf8


import pandas as pd
import numpy as np


if __name__ == '__main__':


    df = pd.read_csv('./data/risk_asset_allocation.csv', index_col = ['date'], parse_dates = ['date'])
    #df = pd.read_csv('./data/000300.csv', index_col = ['date'], parse_dates = ['date'])
    #df = df / df.iloc[0]
    #print df
    df = df.resample('M', how = 'last')

    ds = []
    contents = {}
    for n in range(0, 10):
        risk = 'risk' + str(n + 1)
        #risk = 'close'
        tmp_df = df[[risk]]
        dates = tmp_df.index
        ds = []
        for i in range(0, len(dates) - 12):
            d = dates[i]
            vs = []
            share = 0
            principal = 0
            for j in range(0, 12):
                tmp_d = dates[ i + j ]
                v = tmp_df.loc[tmp_d, risk]
                share = share + 1.0 / v
                principal = principal + 1
                vs.append(share * v / principal)
            max_drawdown = np.inf
            for n in range(1, len(vs) + 1):
                drawdown = vs[n - 1] - 1
                if drawdown < max_drawdown:
                    max_drawdown = drawdown
            #print d, vs

            ds.append(d)
            cont = contents.setdefault(risk + '_drawdown', [])
            cont.append(max_drawdown)
            cont = contents.setdefault(risk + '_fix_invest_r', [])
            cont.append(vs[-1] / 1 - 1)
            cont = contents.setdefault(risk + '_r', [])
            cont.append(tmp_df.loc[dates[i + 11], risk] / tmp_df.loc[dates[i], risk] - 1)


    df = pd.DataFrame(contents, index = ds)
    df.to_csv('fix_invest.csv')
    print df

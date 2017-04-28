#coding=utf8


import pandas as pd
import numpy as np


def cppi(df, high_risk_level, low_risk_level, max_loss, multi_v, turnover, weeks):

    low_df = df[low_risk_level]
    high_df = df[high_risk_level]

    low_dfr = low_df.pct_change().fillna(0.0)
    low_dfr = low_dfr.rolling(weeks).sum().fillna(0.0)
    high_dfr = high_df.pct_change().fillna(0.0)

    low_average_return = np.percentile(low_dfr.values, 50)
    #print low_average_return
    #high_great_loss = np.percentile(high_dfr.values, 5)

    dates = df.index

    vs = []
    ds = []
    high_pos = []

    for i in range(0, len(dates)):

        d = dates[i]

        if len(vs) == 0:
            v = 1.0
        else:
            v = vs[-1]

        high_v = (v * (1 + low_average_return) - 1 * (1 - max_loss)) * multi_v

        if high_v > v:
            high_v = v
        elif high_v < 0:
            high_v = 0
        low_v = v - high_v

        high_r = high_dfr.loc[d]
        low_r = low_dfr.loc[d]

        now_v = high_v * (1 + high_r) + low_v * (1 + low_r)
        high_pos.append(high_v / v)
        vs.append(now_v)
        ds.append(d)
        #print d, high_v / v, now_v


    result_df = pd.DataFrame(np.matrix([high_pos, vs]).T, index = ds, columns = ['pos', 'nav'])
    result_df.index.name = 'date'
    #print df
    #print low_average_return, high_great_loss

    pos_df = result_df[['pos']]
    pos_df = filter_by_turnover(pos_df, 0.2)
    pos_df.columns = [high_risk]
    pos_df[low_risk] = 1 - pos_df[high_risk]
    pos_df.to_csv('cppi_pos.csv')

    dfr = df.pct_change().fillna(0.0)
    dfr = dfr[[high_risk, low_risk]]

    pos_df = pos_df.reindex(dfr.index).fillna(method = 'pad')
    rdf = (dfr * pos_df).sum(axis = 1).to_frame(name = 'r')
    vdf = (rdf + 1).cumprod()
    vdf.columns = [['v']]
    vdf.to_csv('cppi_v.csv')

    df = pd.concat([pos_df, vdf], axis = 1)

    #df.to_csv('cppi.csv')
    return df


def filter_by_turnover(df, turnover):
    result = {}
    sr_last=None
    for k, v in df.iterrows():
        vv = v.fillna(0)
        if sr_last is None:
            result[k] = v
            sr_last = vv
        else:
            xsum = (vv - sr_last).abs().sum()
            if xsum >= turnover:
                result[k] = v
                sr_last = vv
            else:
                #print "filter by turnover:", v.to_frame('ratio')
                pass
    return pd.DataFrame(result).T


if __name__ == '__main__':


    df = pd.read_csv('./data/nav.csv', index_col = ['date'], parse_dates = ['date'])
    #df = df.iloc[-400:]
    df = df.resample('W-FRI').last()
    #df = df.iloc[-30:]

    high_risk = 'risk10'
    low_risk = 'risk1'
    max_loss = 0.01
    multi_v = 10
    turnover = 0.2
    weeks = 12

    result_df = cppi(df, high_risk, low_risk, max_loss, multi_v, turnover, weeks)
    result_df.to_csv('cppi.csv')
    #print result_df.iloc[-1]
    print result_df

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


    risk1_pos_df = pd.read_csv('./data/risk1_pos.csv', index_col = ['date'], parse_dates = ['date'])
    risk10_pos_df = pd.read_csv('./data/risk10_pos.csv', index_col = ['date'], parse_dates = ['date'])
    #print risk1_pos_df
    #print risk10_pos_df
    #print risk10_pos_df

    df = pd.read_csv('./data/nav.csv', index_col = ['date'], parse_dates = ['date'])
    #df = df.iloc[-400:]
    df = df.resample('W-FRI').last()
    #df = df.iloc[-30:]

    high_risk = 'risk10'
    low_risk = 'risk1'
    max_loss = 0.00
    multi_v = 10
    turnover = 0.2
    weeks = 12

    dates = df.index
    num = 0
    total = 0
    rs = []

    '''
    for i in range(0, len(dates) - 12):
        tmp_df = df.iloc[ i : ]
        result_df = cppi(tmp_df, high_risk, low_risk, max_loss, multi_v, turnover, weeks)
        vs = result_df['v']
        #tmp_df = tmp_df / tmp_df.iloc[0]
        #vs = tmp_df['risk10']
        if vs[12] > 1.0:
            num += 1
        total += 1
        print dates[i], vs[12]
        rs.append(vs[12] / 1.0 - 1)

    print 1.0 * num / total
    print np.mean(rs)
        #print dates[i], np.min(result_df['v'])
    #result_df.to_csv('cppi.csv')
    #print result_df.iloc[-1]
    #print result_df
    '''

    result_df = cppi(df, high_risk, low_risk, max_loss, multi_v, turnover, weeks)
    #print result_df
    cppi_pos_df = result_df[['risk1', 'risk10']]
    #print cppi_pos_df.index
    #print risk1_pos_df.index
    cppi_pos_df.reindex(risk1_pos_df.index)
    cppi_pos_df = cppi_pos_df.fillna(method = 'pad')
    #risk1_pos_df = risk1_pos_df * cppi_pos_df[['risk1']]
    print risk1_pos_df
    print cppi_pos_df[['risk1']]
    #print cppi_pos_df


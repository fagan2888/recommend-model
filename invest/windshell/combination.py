#coding=utf8

import pandas as pd
import numpy as np

#asset = ['ltj','zp','lqs','wph','ydh']
asset = ['ltj','zp','lqs','wph']

result = []
ratio  = []
dates = []
for asset_name in asset:

    df = pd.read_csv('./tmp/' + asset_name + '.csv', index_col = 'date', parse_dates = 'date' )
    risks = df['risk'].values

    rdf = pd.read_csv('./tmp/5fund.csv', index_col = 'date', parse_dates = 'date' )
    rdf = rdf.loc[df.index]

    #rdf.to_csv('./hehe.csv')

    rdfr = rdf.pct_change().fillna(0.0)

    #print df
    risk_std = np.std(risks)
    risk_mean = np.mean(risks)

    max_risk = risk_mean + 2 * risk_std
    min_risk = risk_mean - 2 * risk_std

    norm_risk = []

    for risk in risks:
        if risk > max_risk or risk < min_risk:
            continue
        norm_risk.append(risk)

    risk_mean = np.mean(norm_risk)

    dates   = df.index
    risks   = df['risk'].values
    returns = df['return'].values


    f = open('./tmp/' + asset_name + '_mean_std.csv', 'w')
    f.write('date, ratio, risk, return, real_return, net_value\n')
    f_str = '%s, %f, %f, %f, %f, %f\n'


    pre_value = 1
    pre_w     = 0


    asset_net_value = []
    asset_net_value.append(1)
    asset_ratio = []


    for i in range(0, len(dates)):

        d      = dates[i]
        risk   = risks[i]
        ret    = returns[i]
        r      = rdfr.loc[d, asset_name]
        v      = pre_value * (1 + r * pre_w)
        asset_net_value.append(v)
        pre_value = v
        if risk > max_risk or risk < min_risk:
            f.write(f_str % (d, 0.0, risk, 0.0, r, v))
            pre_w = 0.0
            asset_ratio.append(0)
        elif risk <= risk_mean:
            #print d, risk, ret
            f.write(f_str % (d, 1.0, risk, ret, r, v))
            pre_w = 1.0
            asset_ratio.append(1)
        else:
            w = risk_mean / risk
            #print d, risk_mean, ret * w
            f.write(f_str % (d, w, risk_mean, ret * w, r, v))
            pre_w = w
            asset_ratio.append(w)


    f.flush()
    f.close()


    del asset_net_value[0]
    #print len(dates)
    #print len(asset_net_value)
    #print
    result.append(asset_net_value)
    ratio.append(asset_ratio)


result_df = pd.DataFrame(np.matrix(result).T, index=dates, columns=asset)
result_df.to_csv('./tmp/eq_risk.csv')


ratio_df = pd.DataFrame(np.matrix(ratio).T, index=dates, columns=asset)
ratio_df.to_csv('./tmp/eq_ratio.csv')

#df.to_csv('./tmp/norm_largecap.csv')
#print df

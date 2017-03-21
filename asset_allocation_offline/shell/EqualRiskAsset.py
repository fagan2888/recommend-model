#coding=utf8


import string
import pandas as pd
import time
from datetime import datetime
import sys
sys.path.append('shell')


def equalriskasset(equal_ratio_df, dfr):

    rf = 0.03 / 250

    ratio_df         = equal_ratio_df
    ratio_dates      = ratio_df.index
    start_date = ratio_dates[0]


    dfr               = dfr[dfr.index >= start_date]

    dates = dfr.index


    assetlabels   = dfr.columns    
    asset_values = {}
    asset_combination = {}
    asset_ratio  = {}


    ratio_index = 0

    for asset in assetlabels:

        asset_values.setdefault(asset, [1.0])
        asset_ratio.setdefault(asset, 0)
        asset_combination.setdefault(asset, [[0.0, 0,0]])


    result_dates = []
    result_datas  = []


    for i in range(0, len(dates)):

        d = dates[i]

        for asset in assetlabels:

            cvs = asset_combination[asset]
            last_cvs = cvs[-1]
            current_cvs = [last_cvs[0] * (1 + dfr.loc[d, asset]), last_cvs[1] * (1 + rf)]
            cvs.append(current_cvs)
            vs = asset_values[asset]
            if current_cvs[0] + current_cvs[1] == 0:
                continue
            else:
                vs.append(current_cvs[0] + current_cvs[1])


        if ratio_index < len(ratio_dates) and d >= ratio_dates[ratio_index]:
            for asset in assetlabels:
                asset_ratio[asset] = ratio_df.loc[ratio_dates[ratio_index], asset]
                vs = asset_values[asset]
                cvs = asset_combination[asset]
                cvs.append([vs[-1] * asset_ratio[asset], vs[-1] * (1 - asset_ratio[asset])])
            ratio_index = ratio_index + 1


        asset_vs = []
        for label in assetlabels:
            asset_vs.append(asset_values[label][-1])


        result_datas.append(asset_vs)
        result_dates.append(d)


    result_df = pd.DataFrame(result_datas, index=result_dates, columns=assetlabels)


    result_df.index.name = 'date'
    result_df.to_csv('./result/equalriskasset.csv')

    return result_df

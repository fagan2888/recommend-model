#coding=utf8


import sys
sys.path.append("shell")
import FundValue as fv
import Financial as fin
import pandas as pd
import FundFilter as fi


idfr = None
rf   = 0.025 / 52


def equal_weight(dfr):
    ws = {}
    codes = dfr.columns
    for i in range(0, len(codes)):
        code = codes[i]
        ws[code] = (1.0 / len(codes))
    return ws


def allocation(funddfr):

    indexdfr = idfr.loc[funddfr.index]
    ws = {}
    jensen_measure = fi.jensenmeasure(funddfr, indexdfr, rf)
    jensen_data    = fi.ratio_filter(jensen_measure, 0.5)
    print jensen_data

    #按照索提诺比率过滤
    sortino_measure = fi.sortinomeasure(funddfr, rf)
    sortino_data    = fi.ratio_filter(sortino_measure, 0.5)

    #按照ppw测度过滤
    ppw_measure    = fi.ppwmeasure(funddfr, indexdfr, rf)
    ppw_data       = fi.ratio_filter(ppw_measure, 0.5)

    #ppw_data       = sf.ppwfilter(funddf, indexdf, rf, 1.0)
    #print ppw_data

    #stability_measure = stabilitymeasure(funddf)
    #stability_data    = ratio_filter(stability_measure, 2.0 / 3)

    #stability_data = sf.stabilityfilter(funddf, 1.0)

    sharpe_measure = fi.sharpemeasure(funddfr, rf)
    sharpe_data    = fi.ratio_filter(sharpe_measure, 0.5)
    
    sharpe_set = set()
    jensen_set = set()
    sortino_set = set()
    ppw_set = set()

    for record in sharpe_data:
        code = record[0]
        sharpe_set.add(code)

    for record in jensen_data:
        code = record[0]
        jensen_set.add(code)

    for record in ppw_data:
        code = record[0]
        ppw_set.add(code)

    for record in sortino_data:
        code = record[0]
        sortino_set.add(code)

    #codes = sharpe_set & jensen_set & ppw_set & sortino_set
    codes = jensen_set
    for code in codes:
        ws[code] = 1.0 / len(codes)
    return ws

if __name__ == '__main__':


    fdf = pd.read_csv('./data/fund_value.csv', index_col = 'date', parse_dates = ['date'])
    idf = pd.read_csv('./data/index_value.csv', index_col = 'date', parse_dates = ['date'])

    fdf = fdf.iloc[-2000:-1,]
    fdf.dropna(axis = 1, inplace = True)
    fdf = fdf.resample('W-FRI').last()
    fdfr = fdf.pct_change().fillna(0.0)

    idf = idf.iloc[-2000:-1,]
    idf.dropna(axis = 1, inplace = True)
    idf = idf.resample('W-FRI').last()
    idfr = idf.pct_change().fillna(0.0)
    idfr = idfr['000300.SH']

    his_back = 52
    interval = 13
    df = fv.FundValue(fdfr, his_back, interval, allocation)
    #df = fv.FundValue(fdfr, his_back, interval, equal_weight)
    df.to_csv('nav.csv')

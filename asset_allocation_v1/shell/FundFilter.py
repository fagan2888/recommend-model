#coding=utf8



import numpy as np
import string
import os
import sys
sys.path.append("windshell")
import Financial as fin
import Data
from numpy import *
from datetime import datetime
import Const
import FundIndicator as fi
import pandas as pd
import AllocationData


from Const import datapath

rf = Const.rf



#按照基金成立时间过滤
def fundsetuptimefilter(codes, start_date, indicator_df):

    establish_date_code = set()

    for code in indicator_df.index:

            date = indicator_df['establish_date'][code]
            if datetime.strptime(date,'%Y-%m-%d') <= datetime.strptime(start_date, '%Y-%m-%d'):
                    establish_date_code.add(code)

    final_codes = []

    for code in codes:

        if code in establish_date_code:

            final_codes.append(code)

    return final_codes



#按照jensen测度过滤
def jensenmeasure(funddf, indexdf, rf):


    funddfr = funddf.pct_change().fillna(0.0)
    indexdfr = indexdf.pct_change().fillna(0.0)

    jensen = {}
    cols = funddfr.columns
    for col in cols:
        p = []
        m = []
        rs = funddfr[col].values
        #print col, rs
        indexrs = indexdfr.values
        for i in range(0, len(rs)):
            if isnan(rs[i]):
                continue
            else:
                p.append(rs[i])
                m.append(indexrs[i])

        jensen[col] = fin.jensen(p, m, rf)

    return jensen



#按照sortino测度过滤
def sortinomeasure(funddf, rf):

    funddfr = funddf.pct_change().fillna(0.0)
    indexdfr = funddf.pct_change().fillna(0.0)

    sortino = {}
    cols = funddfr.columns
    for col in cols:
        p = []
        rs = funddfr[col].values
        for i in range(0, len(rs)):
            if isnan(rs[i]):
                continue
            else:
                p.append(rs[i])
        sortino_value = fin.sortino(p,rf)
        #if np.isinf(sortino_value):
        #    continue
        sortino[col] = sortino_value


    return sortino



#按照ppw测度过滤
def ppwmeasure(funddf, indexdf, rf):


    length = len(funddf.index)

    '''
    tran_index = []
    for i in range(0, length):
        if i % 4 == 0:
            tran_index.append(i)

    funddf = funddf.iloc[tran_index]
    funddfr = funddf.pct_change()

    indexdf = indexdf.iloc[tran_index]
    indexdfr = indexdf.pct_change()
    '''

    funddfr = funddf.pct_change().fillna(0.0)
    indexdfr = indexdf.pct_change().fillna(0.0)


    ppw = {}
    cols = funddfr.columns
    for col in cols:
        p = []
        m = []
        rs = funddfr[col].values
        indexrs = indexdfr.values
        for i in range(0, len(rs)):
            if isnan(rs[i]):
                continue
            else:
                p.append(rs[i] - rf)
                m.append(indexrs[i] - rf)
        #print p
        #print m

        ppw[col] = fin.ppw(p, m)

    return ppw



#基金稳定性测度
def stabilitymeasure(funddf):


    length = len(funddf.index)
    tran_index = []
    for i in range(0, length):
        if i % 4 == 0:
            tran_index.append(i)

    funddf = funddf.iloc[tran_index]
    funddfr = funddf.pct_change()

    length = len(funddfr)

    fundstab = {}
    fundscore = {}

    for i in range(1, length):

        fr = {}
        for code in funddfr.columns:
            r = funddfr[code].values[i]
            fr[code] = r


        x = fr
        sorted_x = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
        sorted_fr = sorted_x

        l = len(sorted_fr)
        frank = {}


        for i in range(0, l):
            k,v = sorted_fr[i]
            if i <= 0.2 * l:
                frank[k] = 5
            elif i > 0.2 * l and i <= 0.4 * l:
                frank[k] = 4
            elif i > 0.4 * l and i <= 0.6 * l:
                frank[k] = 3
            elif i > 0.6 * l and i <= 0.8 * l:
                frank[k] = 2
            else:
                frank[k] = 1


        for code in frank.keys():
            stab = fundstab.setdefault(code, [])
            score = fundscore.setdefault(code, [])

            rank = frank[code]

            if len(stab) == 0:
                stab.append(rank)
                score.append(0)
                continue

            lastrank = stab[len(stab) - 1]
            lastscore = score[len(score) - 1]


            if rank >= lastrank:
                score.append(5)
            else:
                score.append(lastscore - (lastrank - rank))

            stab.append(rank)


    final_fund_stability = {}
    for k, v in fundscore.items():
        final_fund_stability[k] = np.sum(v)


    return final_fund_stability


#按照规模过滤
def scalefilter(ratio):


    fund_scale_df =  Data.scale_data()
    stock_codes   =  Data.stock_fund_code()

    scale = {}
    for code in fund_scale_df.index:
        v = fund_scale_df.loc[code].values
        if code in stock_codes:
            #if string.atof(v[0]) >= 10000000000.0:
            continue

        scale[code] = v


    return ratio_filter(scale, ratio)

    #print 'scale 000457 : ' ,scale['000457.OF']



def ratio_filter(measure, ratio):


    x = measure
    sorted_x       = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
    sorted_measure = sorted_x


    result = []
    for i in range(0, (int)(len(sorted_measure) * ratio)):
        result.append(sorted_measure[i])



    return result



def stockfundfilter(allocationdata, funddf, indexdf):


    dates = funddf.index.values
    dates.sort()
    end_date   = dates[-1].strftime('%Y-%m-%d')
    start_date = dates[0].strftime('%Y-%m-%d')

    indicator = {}


    #funddf  = Data.fund_value(start_date, end_date)
    #indexdf = Data.index_value(start_date, end_date, '000300.SH')


    #按照规模过滤
    #scale_data     = scalefilter(2.0 / 3)
    #scale_data     = sf.scalefilter(1.0)
    #print scale_data
    #按照基金创立时间过滤


    #setuptime_data = fundsetuptimefilter(funddf.columns, start_date, Data.establish_data())


    #print setuptime_data
    #按照jensen测度过滤
    jensen_measure = jensenmeasure(funddf, indexdf, rf)
    jensen_data    = ratio_filter(jensen_measure, 0.5)
    #jensen_data    = sf.jensenfilter(funddf, indexdf, rf, 1.0)

    #按照索提诺比率过滤
    sortino_measure = sortinomeasure(funddf, rf)
    sortino_data    = ratio_filter(sortino_measure, 0.5)
    #sortino_data   = sf.sortinofilter(funddf, rf, 1.0)

    #按照ppw测度过滤
    ppw_measure    = ppwmeasure(funddf, indexdf, rf)
    ppw_data       = ratio_filter(ppw_measure, 0.5)

    #ppw_data       = sf.ppwfilter(funddf, indexdf, rf, 1.0)
    #print ppw_data

    stability_measure = stabilitymeasure(funddf)
    stability_data    = ratio_filter(stability_measure, 2.0 / 3)

    #stability_data = sf.stabilityfilter(funddf, 1.0)

    sharpe_data    = fi.fund_sharp_annual(funddf)

    #print stability_data

    #print 'jensen'
    jensen_dict = {}
    for k,v in jensen_data:
        jensen_dict[k] = v
        #print k, v
    #print
    #print 'sortino'


    sortino_dict = {}
    for k,v in sortino_data:
        sortino_dict[k] = v
        #print k,v

    #print
    #print 'ppw'
    ppw_dict = {}
    for k,v in ppw_data:
        ppw_dict[k] = v
        #print k,v

    #print
    #print 'statbility'
    stability_dict = {}
    for k,v in stability_data:
        stability_dict[k] = v
        #print k,v


    sharpe_dict = {}
    for k,v in sharpe_data:
        sharpe_dict[k] = v


    #scale_set = set()
    #for k, v in scale_data:
    #    scale_set.add(k)


    #setuptime_set = set(setuptime_data)


    jensen_set = set()
    for k, v in jensen_data:
        jensen_set.add(k)


    sortino_set = set()
    for k, v in sortino_data:
        sortino_set.add(k)


    ppw_set = set()
    for k, v in ppw_data:
        ppw_set.add(k)


    stability_set = set()
    for k, v in stability_data:
        stability_set.add(k)

    codes = []

    for code in jensen_set:
        if  (code in sortino_set) and (code in ppw_set) and (code in stability_set):
            codes.append(code)


    for code in codes:
        ind = indicator.setdefault(code, {})
        #if np.isinf(sharpe_dict[code]):
        #    continue
        #if np.isinf(sortino_dict[code]):
        #    continue
        ind['sharpe']    = sharpe_dict[code]
        ind['jensen']    = jensen_dict[code]
        ind['sortino']   = sortino_dict[code]
        ind['ppw']     = ppw_dict[code]
        ind['stability'] = stability_dict[code]


    indicator_codes = []
    indicator_datas = []


    indicator_set = set(funddf.columns)
    #indicator_set.add(code)


    for code in indicator_set:

        indicator_codes.append(code)
        indicator_datas.append([sharpe_dict.setdefault(code, None), jensen_measure.setdefault(code, None), sortino_measure.setdefault(code, None), ppw_measure.setdefault(code, None), stability_measure.setdefault(code, None)])


    indicator_df = pd.DataFrame(indicator_datas, index = indicator_codes, columns=['sharpe', 'jensen', 'sortino', 'ppw', 'stability'])
    indicator_df.index.name = 'code'
    indicator_df.to_csv(datapath('stock_indicator_' + end_date + '.csv'))


    allocationdata.stock_fund_measure[end_date] = indicator_df


    '''
    f = open(datapath('stockfilter_codes_' + end_date + '.csv'),'w')
    for code in codes:
        f.write(str(code) + '\n')

    f.flush()
    f.close()
    '''


    return codes, indicator


def bondfundfilter(allocationdata, funddf, indexdf):


    dates = funddf.index.values
    dates.sort()
    end_date   = dates[-1].strftime('%Y-%m-%d')
    start_date = dates[0].strftime('%Y-%m-%d')


    indicator = {}


    #funddf  = Data.bond_value(start_date, end_date)
    #indexdf = Data.bond_index_value(start_date, end_date, 'H11001.CSI')

    #按照基金创立时间过滤
    #setuptime_data = fundsetuptimefilter(funddf.columns, start_date, Data.bond_establish_data())


    '''
    #print setuptime_data
    #按照jensen测度过滤
    #jensen_data    = jensenfilter(funddf, indexdf, rf, 0.5)
    #jensen_data    = sf.jensenfilter(funddf, indexdf, rf, 1.0)

    #按照索提诺比率过滤
    #sortino_data   = sortinofilter(funddf, rf, 0.5)
    #sortino_data   = sf.sortinofilter(funddf, rf, 1.0)

    #按照ppw测度过滤
    #ppw_data       = ppwfilter(funddf, indexdf, rf, 0.5)
    #ppw_data       = sf.ppwfilter(funddf, indexdf, rf, 1.0)
    #print ppw_data

    #stability_data = stabilityfilter(funddf, 2.0 / 3)
    #stability_data = sf.stabilityfilter(funddf, 1.0)
    '''


    #按照jensen测度过滤
    jensen_measure = jensenmeasure(funddf, indexdf, rf)
    jensen_data    = ratio_filter(jensen_measure, 0.5)
    #jensen_data    = sf.jensenfilter(funddf, indexdf, rf, 1.0)

    #按照索提诺比率过滤
    sortino_measure = sortinomeasure(funddf, rf)
    sortino_data    = ratio_filter(sortino_measure, 0.5)
    #sortino_data   = sf.sortinofilter(funddf, rf, 1.0)

    #按照ppw测度过滤
    ppw_measure    = ppwmeasure(funddf, indexdf, rf)
    ppw_data       = ratio_filter(ppw_measure, 0.5)
    #ppw_data       = sf.ppwfilter(funddf, indexdf, rf, 1.0)
    #print ppw_data

    stability_measure = stabilitymeasure(funddf)
    stability_data    = ratio_filter(stability_measure, 2.0 / 3)
    #stability_data = sf.stabilityfilter(funddf, 1.0)

    sharpe_data    = fi.fund_sharp_annual(funddf)

    #print 'jensen'
    jensen_dict = {}
    for k,v in jensen_data:
        jensen_dict[k] = v
    #print k, v
    #print
    #print 'sortino'


    sortino_dict = {}
    for k,v in sortino_data:
        sortino_dict[k] = v
    #print k,v

    #print
    #print 'ppw'
    ppw_dict = {}
    for k,v in ppw_data:
        ppw_dict[k] = v
    #print k,v

    #print
    #print 'statbility'
    stability_dict = {}
    for k,v in stability_data:
        stability_dict[k] = v
    #print k,v


    sharpe_dict = {}
    for k,v in sharpe_data:
        sharpe_dict[k] = v


    #setuptime_set = set(setuptime_data)


    jensen_set = set()
    for k, v in jensen_data:
        jensen_set.add(k)


    sortino_set = set()
    for k, v in sortino_data:
        sortino_set.add(k)


    ppw_set = set()
    for k, v in ppw_data:
        ppw_set.add(k)


    stability_set = set()
    for k, v in stability_data:
        stability_set.add(k)

    codes = []



    for code in jensen_set:
        if (code in sortino_set) and (code in ppw_set) and (code in stability_set):
            codes.append(code)


    for code in codes:
        ind = indicator.setdefault(code, {})
        ind['sharpe']    = sharpe_dict[code]
        ind['jensen']    = jensen_dict[code]
        ind['sortino']   = sortino_dict[code]
        ind['ppw']         = ppw_dict[code]
        ind['stability'] = stability_dict[code]


    indicator_set = set(funddf.columns)

    indicator_codes = []
    indicator_datas = []


    for code in indicator_set:

        if (not sharpe_dict.has_key(code)) or (not sortino_dict.has_key(code)):
            continue
        if np.isinf(sharpe_dict[code]):
            continue
        if np.isinf(sortino_dict[code]):
            continue

        indicator_codes.append(code)
        indicator_datas.append([sharpe_dict.setdefault(code, None), jensen_measure.setdefault(code, None), sortino_measure.setdefault(code, None), ppw_measure.setdefault(code, None), stability_measure.setdefault(code, None)])


    indicator_df = pd.DataFrame(indicator_datas, index = indicator_codes, columns=['sharpe', 'jensen', 'sortino', 'ppw', 'stability'])
    indicator_df.index.name = 'code'
    indicator_df.to_csv(datapath('bond_indicator_' + end_date + '.csv'))


    allocationdata.bond_fund_measure[end_date] = indicator_df


    '''
    f = open(datapath('bondfilter_codes_' + end_date + '.csv'),'w')
    for code in codes:
        f.write(str(code) + '\n')

    f.flush()
    f.close()
    '''

    return codes, indicator



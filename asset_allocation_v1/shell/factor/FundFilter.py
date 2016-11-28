#coding=utf8



import numpy as np
import Financial as fin
from numpy import *
from datetime import datetime
import pandas as pd



#按照jensen测度过滤
def jensenmeasure(funddfr, indexdfr, rf):


    jensen = {}
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
                p.append(rs[i])
                m.append(indexrs[i])

        jensen[col] = fin.jensen(p, m, rf)

    return jensen



#按照sortino测度过滤
def sortinomeasure(funddfr, rf):

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
        sortino[col] = sortino_value

    return sortino


#按照ppw测度过滤
def ppwmeasure(funddfr, indexdfr, rf):


    length = len(funddfr.index)

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
        ppw[col] = fin.ppw(p, m)

    return ppw


def sharpemeasure(funddfr, rf):

    sharpe = {}
    cols = funddfr.columns
    for col in cols:
        rs = funddfr[col].values
        sharpe[col] = (np.mean(rs) - rf) / np.std(rs)
    return sharpe

def ratio_filter(measure, ratio):

    x = measure
    sorted_x       = sorted(x.iteritems(), key=lambda x : x[1], reverse=True)
    sorted_measure = sorted_x


    result = []
    for i in range(0, (int)(len(sorted_measure) * ratio)):
        result.append(sorted_measure[i])


    return result

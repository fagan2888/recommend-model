#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import sys
sys.path.append('shell')
import numpy as np
import pandas as pd
from ipdb import set_trace

from db import *

def load_pe():

    indexpe = pd.read_csv("data/macro/indexpe.csv", index_col = ['date'], parse_dates = ['date'])
    pe_sz = indexpe[['120000016']]
    pe_sz.columns = ['pe']
    pe_sz_pt = pe_sz.rolling(window = len(pe_sz), min_periods = 100).apply(lambda x: len(x[x<x[-1]]) / float(len(x)))
    pe_sz_pt.columns = ['pe_percentile']
    df_pe = pd.concat([pe_sz, pe_sz_pt],1).dropna()

    df_sz = base_ra_index_nav.load_series('120000001')
    df_sz = df_sz.to_frame(name = 'nav')
    df_res =  pd.merge(df_sz, df_pe, left_index = True, right_index = True, how = 'left').dropna()
    df_res.to_csv('data/macro/indexpe_percentile.csv', index_label = 'date')

    return


if __name__ == '__main__':

    load_pe()

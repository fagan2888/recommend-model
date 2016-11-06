#coding=utf8


import pandas as pd
import numpy as np
from datetime import datetime


if __name__ == '__main__':


    fund_size_df = pd.read_csv('./data/fund_size.csv', index_col = ['date'], parse_dates = ['date'])

    fund_df = pd.read_csv('./manager_fund_performance.csv', index_col = 'SECODE')

    all_fund_df = pd.read_csv('./data/tq_fd_basicinfo', index_col = 'SECODE')

    all_fund_df = all_fund_df['FSYMBOL']

    secode_symbol_dict = {}

    for secode in all_fund_df.index:
        code = all_fund_df.loc[secode]
        code = '%06d' % code
        secode_symbol_dict[secode] = code

    #print all_fund_df
    #print fund_df.columns
    yearcodes = set()
    totalyear = {}
    fund_df = fund_df['TOTYEARS']
    for secode in fund_df.index:
        code = secode_symbol_dict[secode]
        year = fund_df.loc[secode]
        if type(year) == str:
            vec  = year.split('年')
            try:
                year = int(vec[0])
            except:
                year = 0
        else:
            try:
                year = int(year.values[0].split('年')[0])
            except:
                year = 0
        if year >= 3:
            yearcodes.add(code)
            #print code

    #print yearcodes
    #print fund_df
    #print fund_df

    size_code = set()
    for code in fund_size_df.iloc[-1].index:
        size = fund_size_df.iloc[-1].loc[code]
        if size > 200000 and size < 500000:
            #print code ,size
            size_code.add('%06d' % int(code))

    final_codes = size_code &  yearcodes

    df = pd.DataFrame(list(final_codes))
    df.to_csv('codes.csv')
    #print len(final_codes)
    #print fund_df

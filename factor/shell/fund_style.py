#coding=utf8


import pandas as pd
import numpy as np
import datetime


if __name__ == '__main__':

    stock_df              = pd.read_csv('./data/stock_price_adjust.csv', index_col = 'date', parse_dates = ['date'])
    stock_market_value_df = pd.read_csv('./data/stock_market_value.csv', index_col = 'date', parse_dates = ['date'])
    stock_pe_df           = pd.read_csv('./data/stock_pe.csv', index_col = 'date', parse_dates = ['date'])
    all_fund_df           = pd.read_csv('./data/all_stock_fund.csv')
    stock_code_df         = pd.read_csv('./data/stock_code.csv', index_col = ['SECODE'])['SYMBOL']
    fund_position_df      = pd.read_csv('./data/tq_fd_skdetail', index_col = ['ENDDATE', 'SECODE', 'SKCODE'], parse_dates = ['ENDDATE'])['ACCSTKRTO']
    #fund_position_df      = pd.read_csv('./data/tq_fd_skdetail', index_col = ['ENDDATE'], parse_dates = ['ENDDATE'])
    #fund_position_df      = fund_position_df[fund_position_df.index >= datetime.datetime.strptime('2014-01-02', '%Y-%m-%d')]
    #fund_position_df.to_csv('tq_fd_skdetail')
    #print fund_position_df

    #print stock_market_value_df.index
    #print stock_pe_df.index

    stock_code_dict = {}
    for secode in stock_code_df.index:
        symbol = stock_code_df.loc[secode]
        stock_code_dict[secode] = symbol

    fund_code_dict = {}
    print all_fund_df
    #for secode in all_fund_df.index:
    #    print all_fund_df.loc[secode]
    #print   stock_code_dict

    #print fund_position_df.index.get_level_values(0)
    #print stock_code_dict.keys()
    for date in fund_position_df.index.get_level_values(0):
        secode_skcode_ratio_df = fund_position_df.loc[date]
        for secode in secode_skcode_ratio_df.index.get_level_values(0):
            skcode_ratio_df = secode_skcode_ratio_df.loc[secode]
            for skcode in skcode_ratio_df.index:
                code = '%06d'  % stock_code_dict[skcode]
                ratio  = skcode_ratio_df[skcode] / 100.0
                size = stock_market_value_df.loc[date, code]
                pe = stock_pe_df.loc[date, code]
                print date, secode, code, ratio, size, pe

        #print secode_skcode_ratio
        #for secode in
        #print date, secode, skcode, ratio

    '''
    print skcode_ratio
    skcode_ratio.index = skcode_ratio.index.droplevel(level = 0).droplevel(level = 0) 
    for skcode in skcode_ratio.index:
        #print skcode
        code = '%06d' % stock_code_dict[skcode]
        ratio  = skcode_ratio[skcode]
        size = stock_market_value_df.loc[date, code]
        pe = stock_pd_df.loc[date, code]
        print code, size, pe, ratio
        #print code, ratio
    '''
    #print all_fund_df

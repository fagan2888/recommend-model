#coding=utf8


import pandas as pd
import numpy as np
import datetime


if __name__ == '__main__':


    dates = pd.date_range('2014-01-01', '2016-11-04')
    #print dates
    #stock_df              = pd.read_csv('./data/stock_price_adjust.csv', index_col = 'date', parse_dates = ['date'])
    stock_market_value_df = pd.read_csv('./data/stock_market_value.csv', index_col = 'date', parse_dates = ['date'])
    stock_market_value_df = stock_market_value_df.reindex(dates).fillna(method='pad')
    stock_pe_df           = pd.read_csv('./data/stock_pe.csv', index_col = 'date', parse_dates = ['date'])
    stock_pe_df           = stock_pe_df.reindex(dates).fillna(method='pad')
    all_fund_df           = pd.read_csv('./data/all_stock_fund.csv', index_col = 'SECODE')['FSYMBOL']
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

    #print all_fund_df
    fund_code_dict = {}
    for secode in all_fund_df.index:
        fcode = all_fund_df.loc[secode]
        fund_code_dict[secode] = fcode
    #print fund_code_dict
    #print   stock_code_dict

    #print fund_position_df.index.get_level_values(0)
    #print stock_code_dict.keys()
    secodes = []
    fpes = []
    fsizes = []
    fsymbols = []
    dates = fund_position_df.index.get_level_values(0).unique().values[0:11]
    #print dates
    for date in dates:
        print date
        secode_skcode_ratio_df = fund_position_df.loc[date]
        #dates.append(date)
        secodes = secode_skcode_ratio_df.index.get_level_values(0).unique()
        symbols = []
        for secode in secodes:
            #print secode, fund_code_dict[secode]
            symbols.append(fund_code_dict[secode])
        fsymbols = symbols
        fpe_record = []
        fsize_record = []
        for secode in secodes:
            skcode_ratio_df = secode_skcode_ratio_df.loc[secode]
            fpe = 0
            fsize = 0
            for skcode in skcode_ratio_df.index:
                if not stock_code_dict.has_key(skcode):
                    continue
                code = '%06d'  % stock_code_dict[skcode]
                ratio  = skcode_ratio_df[skcode] / 100.0
                size = stock_market_value_df.loc[date, code]
                pe = stock_pe_df.loc[date, code]
                fpe = fpe + ratio * pe
                fsize = fsize + ratio * size
            fpe_record.append(fpe)
            fsize_record.append(fsize)
        fpes.append(fpe_record)
        fsizes.append(fsize_record)
        #print date, fpe_record, fsize_record
                #print date, secode, code, ratio, size, pe

        #print secode_skcode_ratio
        #for secode in
        #print date, secode, skcode, ratio


    fsize_df = pd.DataFrame(fsizes, index = dates, columns = fsymbols)
    fpe_df = pd.DataFrame(fpes, index = dates, columns = fsymbols)
    fsize_df.to_csv('fsize.csv')
    fpe_df.to_csv('fpe.csv')


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

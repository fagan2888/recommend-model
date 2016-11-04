#coding=utf8


import pandas as pd
import numpy as np
import datetime


if __name__ == '__main__':


    dates = pd.date_range('2016-01-06', '2016-11-04')

    stock_market_value_df = pd.read_csv('./data/stock_market_value.csv', index_col = 'date', parse_dates = ['date'])
    stock_market_value_df = stock_market_value_df.reindex(dates).fillna(method='pad')
    stock_pe_df           = pd.read_csv('./data/stock_pe.csv', index_col = 'date', parse_dates = ['date'])
    stock_pe_df[stock_pe_df < 0] = np.nan
    stock_pe_df           = stock_pe_df.reindex(dates).fillna(method='pad')
    all_fund_df           = pd.read_csv('./data/all_stock_fund.csv', index_col = 'SECODE')['FSYMBOL']
    stock_code_df         = pd.read_csv('./data/stock_code.csv', index_col = ['SECODE'])['SYMBOL']
    fund_position_df      = pd.read_csv('./data/tq_fd_skdetail', index_col = ['ENDDATE', 'SECODE', 'SKCODE'], parse_dates = ['ENDDATE'])['ACCSTKRTO']

    stock_code_dict = {}
    for secode in stock_code_df.index:
        symbol = stock_code_df.loc[secode]
        stock_code_dict[secode] = symbol

    fund_code_dict = {}
    for secode in all_fund_df.index:
        fcode = all_fund_df.loc[secode]
        fund_code_dict[secode] = fcode

    secodes = []
    fpes = []
    fsizes = []
    fsymbols = []
    dates = fund_position_df.index.get_level_values(0).unique().values[8:11]
    for date in dates:
        print date
        secode_skcode_ratio_df = fund_position_df.loc[date]
        secodes = secode_skcode_ratio_df.index.get_level_values(0).unique()
        symbols = []
        for secode in secodes:
            symbols.append(fund_code_dict[secode])
        fsymbols = symbols
        fpe_record = []
        fsize_record = []
        for secode in secodes:
            skcode_ratio_df = secode_skcode_ratio_df.loc[secode]
            fpe = 0
            fsize = 0
            n = 0
            for skcode in skcode_ratio_df.index:
                if not stock_code_dict.has_key(skcode):
                    continue
                code = '%06d'  % stock_code_dict[skcode]
                ratio  = skcode_ratio_df[skcode] / 100.0
                size = stock_market_value_df.loc[date, code]
                pe = stock_pe_df.loc[date, code]
                if np.isnan(ratio) or np.isnan(size) or np.isnan(pe):
                    continue
                fpe = fpe + ratio * pe
                fsize = fsize + ratio * size
                n = n + 1
            #print n, fpe, fsize
            if n == 0:
                fpe_record.append(np.nan)
                fsize_record.append(np.nan)
            else:
                fpe_record.append(fpe / n)
                fsize_record.append(fsize / n)
        fpes.append(fpe_record)
        fsizes.append(fsize_record)


    fsize_df = pd.DataFrame(fsizes, index = dates, columns = fsymbols)
    fpe_df = pd.DataFrame(fpes, index = dates, columns = fsymbols)
    fsize_df.to_csv('fsize.csv')
    fpe_df.to_csv('fpe.csv')

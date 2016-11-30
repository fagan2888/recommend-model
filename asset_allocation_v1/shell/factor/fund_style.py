#coding=utf8


import pandas as pd
import numpy as np
import datetime


if __name__ == '__main__':


    reallocation_dates = pd.DatetimeIndex(['2012-06-30','2012-12-31','2013-06-30','2013-12-31','2014-06-30','2014-12-31','2015-06-30','2015-12-31','2016-06-30'])

    dates = pd.date_range('2010-01-01','2016-10-31')

    #print dates
    stock_market_value_df = pd.read_csv('./data/stock_market_value.csv', index_col = 'date', parse_dates = ['date'])
    stock_market_value_df = stock_market_value_df.reindex(dates).fillna(method='pad')
    stock_pe_df           = pd.read_csv('./data/stock_pe.csv', index_col = 'date', parse_dates = ['date'])
    stock_pe_df[stock_pe_df < 0] = np.nan
    stock_pe_df           = stock_pe_df.reindex(dates).fillna(method='pad')
    all_fund_df           = pd.read_csv('./data/all_stock_fund.csv', index_col = 'SECODE')['FSYMBOL']
    stock_code_df         = pd.read_csv('./data/stock_code.csv', index_col = ['SECODE'])['SYMBOL']
    fund_position_df      = pd.read_csv('./data/tq_fd_skdetail.csv', index_col = ['ENDDATE'], parse_dates = ['ENDDATE'])
    #fund_position_df      = fund_position_df.loc[reallocation_dates]
    #print fund_position_df

    fund_position_df = fund_position_df.loc[reallocation_dates]
    #print reallocation_dates
    #print fund_position_df.index
    #print fund_position_df['SECODE']
    fund_position_df = fund_position_df.reset_index()
    fund_position_df = fund_position_df.set_index(['SECODE'])
    secodes = list(set(fund_position_df.index.values) & set(all_fund_df.index.values))
    fund_position_df = fund_position_df.loc[secodes]
    #print fund_position_df['ENDDATE']
    fund_position_df = fund_position_df.reset_index()
    #print fund_position_df['ENDDATE']
    fund_position_df = fund_position_df.set_index(['ENDDATE', 'SECODE', 'SKCODE'])['NAVRTO']
    #print fund_position_df.index.get_level_values(0).unique()

    stock_code_dict = {}
    for secode in stock_code_df.index:
        symbol = stock_code_df.loc[secode]
        stock_code_dict[secode] = symbol

    fund_code_dict = {}
    for secode in all_fund_df.index:
        fcode = all_fund_df.loc[secode]
        fund_code_dict[secode] = fcode

    '''
    secodes = []
    fpes = []
    fsizes = []
    fsymbols = []
    dates = fund_position_df.index.get_level_values(0).unique()

    for date in dates:
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
            allratio = 0
            n = 0
            for skcode in skcode_ratio_df.index:
                if not stock_code_dict.has_key(skcode):
                    continue
                code = '%06d'  % stock_code_dict[skcode]
                ratio  = skcode_ratio_df[skcode] / 100.0
                allratio = allratio + ratio
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
                fpe_record.append(fpe / allratio)
                fsize_record.append(fsize / allratio)
        fpes.append(fpe_record)
        fsizes.append(fsize_record)


    fsize_df = pd.DataFrame(fsizes, index = dates, columns = fsymbols)
    fpe_df = pd.DataFrame(fpes, index = dates, columns = fsymbols)
    fsize_df.to_csv('fsize.csv')
    fpe_df.to_csv('fpe.csv')
    '''

    fsize_dict = {}
    fpe_dict   = {}

    dates = fund_position_df.index.get_level_values(0).unique().values
    dates.sort()
    #dates = [dates[-1]]
    for date in dates:
        secode_skcode_ratio_df = fund_position_df.loc[date]
        secodes = secode_skcode_ratio_df.index.get_level_values(0).unique()
        for secode in secodes:
            skcode_ratio_df = secode_skcode_ratio_df.loc[secode]
            fund_code = fund_code_dict[secode]
            fcode ='%06d'  % fund_code_dict[secode]
            ratios = []
            pes    = []
            sizes  = []

            allratio = 0
            for skcode in skcode_ratio_df.index:
                if not stock_code_dict.has_key(skcode):
                    continue
                code = '%06d'  % stock_code_dict[skcode]
                ratio  = skcode_ratio_df[skcode]
                size = stock_market_value_df.loc[date, code]
                pe = stock_pe_df.loc[date, code]
                if np.isnan(ratio) or np.isnan(size) or np.isnan(pe):
                    continue
                #print date, fcode, code, ratio, size, pe
                allratio = allratio + ratio
                ratios.append(ratio)
                sizes.append(size * ratio)
                pes.append(pe * ratio)

            if allratio <= 0.1:
                continue

            size_mean = np.mean(sizes)
            size_std  = np.std(sizes)
            pe_mean = np.mean(pes)
            pe_std  = np.std(pes)

            fcode ='%06d'  % fund_code_dict[secode]
            fund_size = 0
            sratio = 0
            for i in range(0, len(sizes)):
                s = sizes[i]
                if s >= size_mean - 2 * size_std and s <= size_mean + 2 * size_std:
                    sratio += ratios[i]
                    fund_size += s

            fund_pe   = 0
            pratio = 0
            for i in range(0, len(pes)):
                p = pes[i]
                if p >= pe_mean - 2 * pe_std and p <= pe_mean + 2 * pe_std:
                    pratio += ratios[i]
                    fund_pe += p

            fund_size = fund_size / sratio
            fund_pe   = fund_pe / pratio

            #print date, fund_code, allratio #,fund_size, fund_pe
            fcode ='%06d'  % fund_code_dict[secode]
            print date, fcode, 'done', fund_size, fund_pe
            fsize = fsize_dict.setdefault(date, {})
            fsize[fcode] = fund_size
            fpe = fpe_dict.setdefault(date, {})
            fpe[fcode] = fund_pe

    #print fsize_dict
    fsize_df = pd.DataFrame(fsize_dict).T
    #print fpe_dict
    fpe_df   = pd.DataFrame(fpe_dict).T

    #print fsize_df
    #print fpe_df
    fsize_df.to_csv('fsize.csv')
    fpe_df.to_csv('fpe.csv')

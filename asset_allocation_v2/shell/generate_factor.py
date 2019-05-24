#coding=utf-8
'''
Created on: May. 19, 2019
Author: Ning Yang
Contact: yangning@licaimofang.com
在计算股票在t时刻的因子时，可能会用到收盘时(收盘价)或收盘后(财务报表)的数据。
因此，为了避免引入未来数据，建议最早在t+1时刻使用t时刻计算的因子。
'''
import numpy as np
import pandas as pd
import statsmodels.api as sm
from calc_descriptor import calc_financial_descriptor, calc_numerical_descriptor
from db import wind_asharecalendar, wind_aindexmembers, wind_ashareeodderivativeindicator, wind_ashareeodprices, wind_aindexeodprices, wind_ashareswindustriesclass, generate_factor_database
from generate_factor_tools import *
from ipdb import set_trace
from trade_date import ATradeDate


def generate_stock_factor(begin_date=None, end_date=None, trade_date=None, percentage=0.8, calc_method='z-score'):
    if calc_method == 'z-score':
        percentage = 1.0
    if (calc_method != 'z-score') and (calc_method != 'raw'):
        return 'wrong calc_method'
    look_back = 300  # > 256+21
    table_name = 'stock_factor' + '_' + calc_method + '_' + str(percentage)
    if begin_date is None:
        reindex = [pd.Timestamp(trade_date)]
    else:
        reindex = wind_asharecalendar.load_a_trade_date(begin_date=pd.Timestamp(begin_date), end_date=pd.Timestamp(end_date))
        reidnex = list(reindex)

    reindex_total = ATradeDate.trade_date(begin_date=reindex[0], end_date=reindex[-1], lookback=look_back+2)
    # load data
    df_index_historical_constituents = wind_aindexmembers.load_a_index_historical_constituents(index_id='000985.CSI')
    stock_ids_total = df_index_historical_constituents.stock_id.unique()
    df_financial_descriptor = calc_financial_descriptor(stock_ids_total=stock_ids_total)
    df_derivative_indicator = wind_ashareeodderivativeindicator.load_a_stock_derivative_indicator(stock_ids=stock_ids_total, reindex=reindex_total)
    df_price = wind_ashareeodprices.load_a_stock_adj_price(stock_ids=stock_ids_total, reindex=reindex_total)
    df_price['benchmark'] = wind_aindexeodprices.load_a_index_nav(index_ids='000985.CSI', reindex=reindex_total)
    df_return = df_price.sort_index().pct_change().iloc[1:]
    df_sw = wind_ashareswindustriesclass.load_a_stock_historical_sw_industry_level1(stock_ids=stock_ids_total)
    # load existing tradedate
    existing_tradedate = generate_factor_database.query_table(table_name=table_name)
    df_descriptor = pd.DataFrame()
    for i_tradedate in reindex:
        if i_tradedate in existing_tradedate:
            continue
        stock_ids_t = df_index_historical_constituents.loc[
            (df_index_historical_constituents.in_date <= i_tradedate) &
            ((df_index_historical_constituents.out_date > i_tradedate) | (df_index_historical_constituents.out_date.isna()))
        ].stock_id.unique()
        df_sw_t = df_sw.loc[(df_sw.entry_date <= i_tradedate) & ((df_sw.remove_date > i_tradedate) | (df_sw.remove_date.isna()))].copy()
        df_sw_t = df_sw_t.reset_index().drop_duplicates(subset=['stock_id']).set_index('stock_id')
        
        df_return_t = df_return.loc[:i_tradedate].iloc[-look_back:].copy()
        df_derivative_indicator_t = df_derivative_indicator.loc[:i_tradedate].iloc[-look_back:].copy()
        df_numerical_descriptor_t = calc_numerical_descriptor(df_return=df_return_t, df_derivative_indicator=df_derivative_indicator_t, stock_ids=stock_ids_t)
        df_financial_descriptor_t = select_fs(df_FS=df_financial_descriptor, trade_date=i_tradedate, stock_ids=stock_ids_t)
        
        df_descriptor_t = pd.merge(df_financial_descriptor_t, df_numerical_descriptor_t, how='left', left_index=True, right_index=True)
        
        df_derivative_descriptor_t = df_derivative_indicator_t.loc[i_tradedate, ['free_float_market_value', 'pb', 'pe_ttm', 'pocf_ttm', 'pncf_ttm', 'ps_ttm', 'pdps_ttm']].unstack().T.reindex(stock_ids_t).copy()
        df_derivative_descriptor_t[['pb', 'pe_ttm', 'pocf_ttm', 'pncf_ttm', 'ps_ttm', 'pdps_ttm']] = 1 / df_derivative_descriptor_t[['pb', 'pe_ttm', 'pocf_ttm', 'pncf_ttm', 'ps_ttm', 'pdps_ttm']]
        df_derivative_descriptor_t['pdps_ttm'] = df_derivative_descriptor_t['pdps_ttm'].fillna(0.0)
        df_derivative_descriptor_t.rename(columns={'pb': 'book_to_price', 'pe_ttm': 'earnings_to_price', 'pocf_ttm': 'ocf_to_price','pncf_ttm': 'ncf_to_price', 'ps_ttm': 'sales_to_price', 'pdps_ttm': 'dividend_yield', 'free_float_market_value': 'weight'}, inplace=True)
        
        df_descriptor_t = pd.merge(df_descriptor_t, df_derivative_descriptor_t, how='left', left_index=True, right_index=True)
        df_descriptor_t = pd.merge(df_descriptor_t, df_sw_t, how='left', left_index=True, right_index=True)
        
        select_num = int(df_descriptor_t.shape[0]*percentage)
        df_descriptor_t = df_descriptor_t.sort_values(by=['weight'],ascending=False).iloc[:select_num]
        df_descriptor_t.dropna(subset=['sw_ind_lv1_code'], inplace=True)
        
        financial_descriptor = [
            'book_to_price', 'earnings_to_price', 'ocf_to_price', 'ncf_to_price', 'sales_to_price', 'dividend_yield',
            'roa', 'roe', 'cash_flow_accrual', 'gross_profit_margin', 'assets_turnover', 'earnings_growth','ocf_growth', 
            'gross_profit_growth', 'gross_profit_margin_growth', 'assets_turnover_growth','earnings_volatility', 'assets_growth'
        ]
        numerical_descriptor = ['short_term_reverse', 'middle_term_momentum', 'beta', 'monthly_turnover', 'quarterly_turnover', 'annual_turnover']
        all_descriptor = financial_descriptor + ['linear_size', 'non_linear_size'] + numerical_descriptor
        # size
        df_descriptor_t = z_score(df=df_descriptor_t, columns=['linear_size'], weight='weight')
        df_descriptor_t['non_linear_size'] = calc_non_linear_size(df=df_descriptor_t)
        df_descriptor_t = z_score(df=df_descriptor_t, columns=['non_linear_size'], weight='weight')
        if calc_method == 'z-score':
            # 对基本面因子进行分行业的标准化处理
            df_descriptor_t = z_score_cbi(df=df_descriptor_t, columns=financial_descriptor, industry='sw_ind_lv1_code', weight='weight', min_count=30)
            # 对基本面因子和数值因子进行标准化处理
            df_descriptor_t = z_score(df=df_descriptor_t, columns=financial_descriptor + numerical_descriptor, weight='weight')
            df_descriptor_t[all_descriptor] = df_descriptor_t[all_descriptor].fillna(df_descriptor_t[all_descriptor].mean())
        df_descriptor_t['trade_date'] = i_tradedate
        columns_retain = ['trade_date', 'sw_ind_lv1_code'] + financial_descriptor + ['linear_size', 'non_linear_size'] + numerical_descriptor
        df_descriptor = df_descriptor.append(df_descriptor_t[columns_retain], ignore_index=False)
    # write
    return_str = generate_factor_database.write_factor(df_descriptor, table_name=table_name)
    return return_str

if __name__ == '__main__':
    df = generate_stock_factor(begin_date='2018-11-02', end_date='2019-05-01')


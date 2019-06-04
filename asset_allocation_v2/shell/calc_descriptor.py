#coding=utf-8
'''
Created on: May. 19, 2019
Author: Ning Yang
Contact: yangning@licaimofang.com
'''

import sys
import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
# from ipdb import set_trace
sys.path.append('shell')
from db import wind_asharecalendar, wind_asharebalancesheet, wind_ashareincome, wind_asharecashflow, wind_aindexmembers
from util_timestamp import closing_date_of_report_period


logger = logging.getLogger(__name__)


def calc_stock_numerical_descriptor(df_return, df_derivative_indicator, stock_ids):

    stock_ids_t = list(stock_ids)
    df_return_t = df_return.reindex(stock_ids_t+['benchmark'], axis=1).fillna(0.0)
    df_alpha_t = df_return_t.sub(df_return_t.benchmark, axis=0).reindex(stock_ids_t, axis=1)
    df_derivative_indicator_t = df_derivative_indicator.reindex(stock_ids_t, axis=1, level=1).copy()
    # short_term_reverse middle_term_momentum
    df_rs_s = df_alpha_t.iloc[-70:].rolling(63).apply(lambda x: calc_rs(x, half_life=10), 'raw=True')
    df_rs_m = df_alpha_t.rolling(252).apply(lambda x: calc_rs(x, half_life=63), 'raw=True')
    df_strev = pd.DataFrame(df_rs_s.iloc[-3:].mean(), columns=['short_term_reverse'])
    df_mtm = pd.DataFrame(df_rs_m.iloc[-20:-10].mean(), columns=['middle_term_momentum'])
    # beta
    df_beta = df_return_t.reindex(stock_ids_t, axis=1).apply(lambda x: calc_beta(x, df_return_t.benchmark))
    df_beta = pd.DataFrame(df_beta, columns=['beta'])
    # turnover
    df_turnover = df_derivative_indicator_t['free_turnover'].reindex(stock_ids_t, axis=1) / 100.0
    df_stom = pd.DataFrame(np.log(df_turnover.iloc[-21:].mean(skipna=True).replace(0, np.nan) * 21), columns=['monthly_turnover'])
    df_stoq = pd.DataFrame(np.log(df_turnover.iloc[-63:].mean(skipna=True).replace(0, np.nan) * 21), columns=['quarterly_turnover'])
    df_stoa = pd.DataFrame(np.log(df_turnover.iloc[-252:].mean(skipna=True).replace(0, np.nan) * 21), columns=['annual_turnover'])
    #
    df_mv = df_derivative_indicator_t['free_float_market_value'].reindex(stock_ids_t, axis=1)
    df_linear_size = pd.DataFrame(np.log(df_mv.iloc[-1].rename('linear_size')))
    # concat
    df_numerical_descriptor = pd.concat([df_strev, df_mtm, df_beta, df_stom, df_stoq, df_stoa, df_linear_size], axis=1, join='outer', sort=False)
    return df_numerical_descriptor

def calc_stock_financial_descriptor(stock_ids):

    df_BS = wind_asharebalancesheet.load_a_stock_balancesheet(stock_ids=stock_ids)
    df_IS = wind_ashareincome.load_a_stock_income(stock_ids=stock_ids)
    df_CF = wind_asharecashflow.load_a_stock_cashflow(stock_ids=stock_ids)

    '''financial statement'''
    df_FS = pd.merge(df_BS, df_IS, how='left', on=['stock_id', 'report_period'])
    df_FS = pd.merge(df_FS, df_CF, how='left', on=['stock_id', 'report_period'])

    df_FS['closing_date'] = df_FS.report_period.map(closing_date_of_report_period)
    df_FS.loc[df_FS.ann_date.isna(), 'ann_date'] = df_FS.loc[df_FS.ann_date.isna(), 'closing_date']
    del df_FS['closing_date']

    df_FS['wc_cf'] = df_FS[['depr_fa_coga_dpba', 'amort_intang_assets', 'amort_lt_deferred_exp', 'others']].sum(axis=1, skipna=True, min_count=1)
    df_FS['d_a'] = df_FS[['depr_fa_coga_dpba', 'amort_intang_assets', 'amort_lt_deferred_exp']].sum(axis=1, skipna=True, min_count=1)
    df_FS.set_index(['stock_id', 'report_period'], inplace=True)
    df_FS.sort_index(inplace=True)

    #bug: comp 远古数据有空值

    columns_ttm = ['oper_rev', 'less_oper_cost', 'net_profit_after_ded_nr_lp', 'net_cash_flows_oper_act', 'wc_cf', 'd_a']
    for column_ttm in columns_ttm:
        df_FS[f'{column_ttm}_ttm'] = (df_FS[column_ttm] - comp(df_FS[column_ttm])).add(lfy(df_FS[column_ttm]), fill_value=0)

    df_FS['roa'] = 2 * df_FS['net_cash_flows_oper_act_ttm'] / (df_FS['tot_assets'] + comp(df_FS['tot_assets']))
    df_FS['roe'] = 2 * df_FS['net_profit_after_ded_nr_lp_ttm'] / (df_FS['tot_shrhldr_euqity_excl_min_int'] + comp(df_FS['tot_shrhldr_euqity_excl_min_int']))
    df_FS['cash_flow_accrual'] = 2 * (df_FS['wc_cf_ttm'] - df_FS['d_a_ttm']) / (df_FS['tot_assets'] + comp(df_FS['tot_assets']))
    df_FS['gross_profit_margin'] = (df_FS['oper_rev_ttm'] - df_FS['less_oper_cost_ttm']) / df_FS['oper_rev_ttm']
    df_FS['assets_turnover'] = 2 * df_FS['oper_rev_ttm'] / (df_FS['tot_assets'] + comp(df_FS['tot_assets']))
    df_FS['gross_profit'] = df_FS['oper_rev_ttm'] - df_FS['less_oper_cost_ttm']

    df_FS['earnings_growth'] = df_FS['net_profit_after_ded_nr_lp_ttm'] / np.abs(comp(df_FS['net_profit_after_ded_nr_lp_ttm'])) - 1
    df_FS['ocf_growth'] = df_FS['net_cash_flows_oper_act_ttm'] / np.abs(comp(df_FS['net_cash_flows_oper_act_ttm'])) - 1
    df_FS['gross_profit_growth'] = df_FS['gross_profit'] / np.abs(comp(df_FS['gross_profit'])) - 1
    df_FS['gross_profit_margin_growth'] = df_FS['gross_profit_margin'] / np.abs(comp(df_FS['gross_profit_margin'])) - 1
    df_FS['assets_turnover_growth'] = df_FS['assets_turnover'] / np.abs(comp(df_FS['assets_turnover'])) - 1

    columns_add = ['earnings_volatility', 'assets_growth']
    for column_add in columns_add:
        df_FS[column_add] = np.nan

    for stock_id in df_FS.index.levels[0].unique():
        df_FS.loc[stock_id, 'earnings_volatility'] = df_FS.loc[stock_id, 'net_profit_after_ded_nr_lp_ttm'] \
            .rolling(20, min_periods=12).apply(e_vol, raw=True).values
        df_FS.loc[stock_id, 'assets_growth'] = df_FS.loc[stock_id, 'tot_assets'] \
            .rolling(20, min_periods=12).apply(lt_growth, raw=True).values

    df_FS.reset_index(inplace=True)
    columns_retain = [
        'stock_id', 'ann_date', 'report_period', 'roa', 'roe', 'cash_flow_accrual', 'gross_profit_margin', 'assets_turnover','earnings_growth',
        'ocf_growth', 'gross_profit_growth', 'gross_profit_margin_growth', 'assets_turnover_growth', 'earnings_volatility', 'assets_growth'
    ]
    df_FD = df_FS.loc[:, columns_retain]

    return df_FD

def lfy(ser):

    index_lfy = ser.index.map(lambda x: (x[0], x[1]+relativedelta(years=-1, month=12, day=31)))
    ser_lfy = pd.Series(ser.reindex(index_lfy).values, index=ser.index)

    return ser_lfy

def comp(ser):

    index_comp = ser.index.map(lambda x: (x[0], x[1]+relativedelta(years=-1)))
    ser_comp = pd.Series(ser.reindex(index_comp).values, index=ser.index)

    return ser_comp

def lt_growth(data):

    data = data[~np.isnan(data)]
    x = np.arange(data.shape[0]) + 1
    y = data
    ols_results = sm.OLS(y, sm.add_constant(x)).fit()
    growth_t = ols_results.params[1] / np.mean(y)

    return growth_t

def e_vol(data):

    data = data[~np.isnan(data)]
    growth_list = []
    for i in range(1, data.shape[0]):
        if data[i-1] != 0:
            growth_t = (data[i] - data[i-1]) / np.abs(data[i-1])
        else:
            growth_t = 0
        growth_list.append(growth_t)
    growth_array = np.array(growth_list)
    fun_var = np.cov(growth_array)
    fun_std = np.sqrt(fun_var)

    return fun_std

def calc_beta(stock, benchmark):
    weight_t = (0.5 ** (1 / 63)) ** (np.arange(len(stock) - 1, -1, -1))
    y = stock * weight_t
    x = benchmark * weight_t
    beta = sm.OLS(y, sm.add_constant(x)).fit().params[1]
    return beta

def calc_rs(data, half_life):
    weight_t = (0.5 ** (1 / half_life)) ** (np.arange(len(data) - 1, -1, -1))
    relative_strength = np.dot(data, weight_t)
    return relative_strength


if __name__ == '__main__':

    df_index_historical_constituents = wind_aindexmembers.load_a_index_historical_constituents(index_id='000985.CSI')
    stock_ids = pd.Index(df_index_historical_constituents.stock_id.unique())
    df = calc_stock_financial_descriptor(stock_ids=stock_ids)


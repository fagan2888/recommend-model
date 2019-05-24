# coding=utf-8
'''
Created on: May. 19, 2019
Author: Ning Yang
Contact: yangning@licaimofang.com
'''

import numpy as np
import pandas as pd
import statsmodels.api as sm
from db import wind_asharecalendar, wind_asharebalancesheet, wind_ashareincome, wind_asharecashflow, wind_aindexmembers
from generate_factor_tools import *
from ipdb import set_trace


def calc_numerical_descriptor(df_return, df_derivative_indicator, stock_ids):
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


def calc_financial_descriptor(stock_ids_total):
    df_BS = wind_asharebalancesheet.load_a_stock_balancesheet(stock_ids=stock_ids_total)
    df_IS = wind_ashareincome.load_a_stock_income(stock_ids=stock_ids_total)
    df_CF = wind_asharecashflow.load_a_stock_cashflow(stock_ids=stock_ids_total)

    df_FS = pd.merge(df_BS, df_IS, how='left', on=['stock_id', 'report_period'])
    df_FS = pd.merge(df_FS, df_CF, how='left', on=['stock_id', 'report_period'])

    df_FS['closing_date'] = df_FS.report_period.map(lambda x: closing_dt(x))
    df_FS.loc[df_FS.ann_date.isna(), 'ann_date'] = df_FS.loc[df_FS.ann_date.isna(), 'closing_date']  # 尚未测试
    del df_FS['closing_date']

    df_FS['wc_cf'] = df_FS[['depr_fa_coga_dpba', 'amort_intang_assets', 'amort_lt_deferred_exp', 'others']].sum(axis=1, skipna=True, min_count=1)
    df_FS['d_a'] = df_FS[['depr_fa_coga_dpba', 'amort_intang_assets', 'amort_lt_deferred_exp']].sum(axis=1, skipna=True, min_count=1)

    df_FS['lfy'] = df_FS.report_period.map(lambda x: pd.Timestamp(x.year - 1, 12, 31))
    df_FS['comp'] = df_FS.report_period.map(lambda x: pd.Timestamp(x.year - 1, x.month, x.day))
    df_FS_t = df_FS.copy()
    df_FS = pd.merge(df_FS, df_FS, how='left', left_on=['stock_id', 'lfy'], right_on=['stock_id', 'report_period'], suffixes=('', '_lfy'))
    df_FS = pd.merge(df_FS, df_FS_t, how='left', left_on=['stock_id', 'comp'], right_on=['stock_id', 'report_period'], suffixes=('', '_comp'))

    columns_ttm = ['oper_rev', 'less_oper_cost', 'net_profit_after_ded_nr_lp', 'net_cash_flows_oper_act', 'wc_cf', 'd_a']
    for i_ttm in columns_ttm:
        df_FS[i_ttm + '_ttm'] = (df_FS[i_ttm] - df_FS[i_ttm + '_comp']).add(df_FS[i_ttm + '_lfy'], fill_value=0)

    df_FS['roa'] = 2 * df_FS['net_cash_flows_oper_act_ttm'] / (df_FS['tot_assets'] + df_FS['tot_assets_comp'])
    df_FS['roe'] = 2 * df_FS['net_profit_after_ded_nr_lp_ttm'] / (df_FS['net_profit_after_ded_nr_lp'] + df_FS['net_profit_after_ded_nr_lp_ttm'])
    df_FS['cash_flow_accrual'] = 2 * (df_FS['wc_cf_ttm'] - df_FS['d_a_ttm']) / (df_FS['tot_assets'] + df_FS['tot_assets_comp'])
    df_FS['gross_profit_margin'] = (df_FS['oper_rev_ttm'] - df_FS['less_oper_cost_ttm']) / df_FS['oper_rev_ttm']
    df_FS['assets_turnover'] = 2 * df_FS['oper_rev_ttm'] / (df_FS['tot_assets'] + df_FS['tot_assets_comp'])
    df_FS['gross_profit'] = df_FS['oper_rev_ttm'] - df_FS['less_oper_cost_ttm']

    df_FS = pd.merge(df_FS, df_FS, how='left', left_on=['stock_id', 'comp'], right_on=['stock_id', 'report_period'], suffixes=('', '_comp'))
    df_FS['earnings_growth'] = df_FS['net_profit_after_ded_nr_lp_ttm'] / np.abs(df_FS['net_profit_after_ded_nr_lp_ttm_comp']) - 1
    df_FS['ocf_growth'] = df_FS['net_cash_flows_oper_act_ttm'] / np.abs(df_FS['net_cash_flows_oper_act_ttm_comp']) - 1
    df_FS['gross_profit_growth'] = df_FS['gross_profit'] / np.abs(df_FS['gross_profit_comp']) - 1
    df_FS['gross_profit_margin_growth'] = df_FS['gross_profit_margin'] / np.abs(df_FS['gross_profit_margin_comp']) - 1
    df_FS['assets_turnover_growth'] = df_FS['assets_turnover'] / np.abs(df_FS['assets_turnover_comp']) - 1

    columns_add = ['earnings_volatility', 'assets_growth']
    for i_add in columns_add:
        df_FS[i_add] = np.nan

    df_FS = df_FS.set_index('stock_id').sort_values(by=['stock_id', 'report_period'])

    for i_stock in df_FS.index.unique():
        df_FS.loc[i_stock, ['earnings_volatility']] = df_FS.loc[i_stock, ['net_profit_after_ded_nr_lp_ttm']] \
            .rolling(20, min_periods=12).apply(lambda x: e_vol(x), 'raw=True').values
        df_FS.loc[i_stock, ['assets_growth']] = df_FS.loc[i_stock, ['tot_assets']] \
            .rolling(20, min_periods=12).apply(lambda x: lt_growth(x), 'raw=True').values

    df_FS.reset_index(inplace=True)
    columns_retain = [
        'stock_id', 'ann_date', 'report_period', 'roa', 'roe', 'cash_flow_accrual', 'gross_profit_margin',
        'assets_turnover', 'earnings_growth',
        'ocf_growth', 'gross_profit_growth', 'gross_profit_margin_growth', 'assets_turnover_growth',
        'earnings_volatility', 'assets_growth'
    ]
    
    return df_FS[columns_retain]

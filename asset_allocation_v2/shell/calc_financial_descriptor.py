# calculate financial indicator ---> calculate TTM indicator ---> calculate TTM descriptor --->
# calculate the growth of TTM descriptor ---> calculate the rolling descriptor
import numpy as np
import pandas as pd
import statsmodels.api as sm


def closing_dt(x):
    if x.month == 3:
        return pd.Timestamp(x.year, 4, 30)
    elif x.month == 6:
        return pd.Timestamp(x.year, 8, 31)
    elif x.month == 9:
        return pd.Timestamp(x.year, 10, 31)
    elif x.month == 12:
        return pd.Timestamp(x.year+1, 4, 30)
    else:
        return pd.Timestamp(1990, 1, 1)  # 特殊标记


def _lt_growth(data):
    data_t = data[~np.isnan(data)].copy()
    x = np.arange(data_t.shape[0]) + 1
    y = data_t
    ols_results = sm.OLS(y, sm.add_constant(x)).fit()
    growth_t = ols_results.params[1] / np.mean(data_t)
    return growth_t


def _e_vol(data):
    data_t = data[~np.isnan(data)].copy()
    growth_list = list()
    for i in range(1, data_t.shape[0]):
        if data_t[i-1] != 0:
            growth_t = (data_t[i] - data_t[i-1]) / np.abs(data_t[i-1])
        else:
            growth_t = 0
        growth_list.append(growth_t)
    growth_array = np.array(growth_list)
    fun_var = np.cov(growth_array)
    fun_std = np.sqrt(fun_var)
    return fun_std


def calc_financial_descriptor(data):
    # fill the null actual announce date with closing date
    data_FS = data.copy()
    data_FS.REPORT_PERIOD = pd.to_datetime(data_FS.REPORT_PERIOD.astype(str)).astype(pd.Timestamp)
    loc_t = data_FS.ACTUAL_ANN_DT.isna()
    data_FS.loc[loc_t, 'ACTUAL_ANN_DT'] = 19900101.00
    data_FS.ACTUAL_ANN_DT = pd.to_datetime(data_FS.ACTUAL_ANN_DT.astype(int).astype(str)).astype(pd.Timestamp)
    data_FS['CLOSING_DT'] = data_FS.REPORT_PERIOD.map(lambda x: closing_dt(x))
    data_FS.loc[loc_t, 'ACTUAL_ANN_DT'] = data_FS.loc[loc_t, 'CLOSING_DT'].astype(pd.Timestamp)
    # calculate financial indicator
    data_FS['WC_CF'] = data_FS[['DECR_INVENTORIES', 'DECR_OPER_PAYABLE', 'INCR_OPER_PAYABLE', 'OTHERS']].sum(axis=1, skipna=True, min_count=1)
    data_FS['D_A'] = data_FS[['DEPR_FA_COGA_DPBA', 'AMORT_INTANG_ASSETS', 'AMORT_LT_DEFERRED_EXP']].sum(axis=1, skipna=True, min_count=1)
    # merge
    data_FS['LFY'] = data_FS.REPORT_PERIOD.map(lambda x: pd.Timestamp(x.year - 1, 12, 31)).astype(pd.Timestamp)
    data_FS['COMP'] = data_FS.REPORT_PERIOD.map(lambda x: pd.Timestamp(x.year - 1, x.month, x.day)).astype(pd.Timestamp)
    data_FS_t = data_FS.copy()
    data_FS = pd.merge(data_FS, data_FS, how='left', left_on=['WIND_CODE', 'LFY'], right_on=['WIND_CODE', 'REPORT_PERIOD'], suffixes=('', '_LFY'))
    data_FS = pd.merge(data_FS, data_FS_t, how='left', left_on=['WIND_CODE', 'COMP'], right_on=['WIND_CODE', 'REPORT_PERIOD'], suffixes=('', '_COMP'))
    columns_del = ['ACTUAL_ANN_DT_LFY', 'REPORT_PERIOD_LFY', 'LFY_LFY', 'COMP_LFY', 'ACTUAL_ANN_DT_COMP', 'REPORT_PERIOD_COMP', 'LFY_COMP', 'COMP_COMP']
    data_FS.drop(columns_del, axis=1, inplace=True)
    # calculate TTM indicator
    descriptor_TTM = ['OPER_REV', 'LESS_OPER_COST', 'NET_PROFIT_AFTER_DED_NR_LP', 'NET_CASH_FLOWS_OPER_ACT', 'WC_CF', 'D_A']
    for i_ttm in descriptor_TTM:
        data_FS[i_ttm + '_TTM'] = (data_FS[i_ttm] - data_FS[i_ttm + '_COMP']).add(data_FS[i_ttm + '_LFY'], fill_value=0)
    # calculate TTM descriptor
    data_FS['QUALITY_ROA'] = 2 * data_FS['NET_CASH_FLOWS_OPER_ACT_TTM'] / (data_FS['TOT_ASSETS'] + data_FS['TOT_ASSETS_COMP'])
    data_FS['QUALITY_ROE'] = 2 * data_FS['NET_PROFIT_AFTER_DED_NR_LP_TTM'] / (data_FS['TOT_SHRHLDR_EQY_EXCL_MIN_INT'] + data_FS['TOT_SHRHLDR_EQY_EXCL_MIN_INT_COMP'])
    data_FS['QUALITY_ACCF'] = 2 * (data_FS['WC_CF_TTM'] - data_FS['D_A_TTM']) / (data_FS['TOT_ASSETS'] + data_FS['TOT_ASSETS_COMP'])
    data_FS['QUALITY_GPM'] = (data_FS['OPER_REV_TTM'] - data_FS['LESS_OPER_COST_TTM']) / data_FS['OPER_REV_TTM']
    data_FS['ATO'] = 2 * data_FS['OPER_REV_TTM'] / (data_FS['TOT_ASSETS'] + data_FS['TOT_ASSETS_COMP'])
    data_FS['GP'] = data_FS['OPER_REV_TTM'] - data_FS['LESS_OPER_COST_TTM']
    # calculate the growth of TTM descriptor
    data_FS = pd.merge(data_FS, data_FS, how='left', left_on=['WIND_CODE', 'COMP'], right_on=['WIND_CODE', 'REPORT_PERIOD'], suffixes=('', '_COMP'))
    data_FS['GROWTH_EGRO'] = data_FS['NET_PROFIT_AFTER_DED_NR_LP_TTM'] / np.abs(data_FS['NET_PROFIT_AFTER_DED_NR_LP_TTM_COMP']) - 1
    data_FS['GROWTH_CGRO'] = data_FS['NET_CASH_FLOWS_OPER_ACT_TTM'] / np.abs(data_FS['NET_CASH_FLOWS_OPER_ACT_TTM_COMP']) - 1
    data_FS['GROWTH_GPGRO'] = data_FS['GP'] / np.abs(data_FS['GP_COMP']) - 1
    data_FS['GROWTH_GPMGRO'] = data_FS['QUALITY_GPM'] / np.abs(data_FS['QUALITY_GPM_COMP']) - 1
    data_FS['GROWTH_ATOGRO'] = data_FS['ATO'] / np.abs(data_FS['ATO_COMP']) - 1
    # calculate the rolling descriptor
    columns_add = ['QUALITY_VERN', 'QUALITY_AGRO']
    for i_add in columns_add:
        data_FS[i_add] = np.nan
    data_FS = data_FS.set_index('WIND_CODE').sort_values(by=['WIND_CODE', 'REPORT_PERIOD'], ascending=True)
    for i_code in data_FS.index.unique():
        data_FS.loc[i_code, ['QUALITY_VERN']] = data_FS.loc[i_code, ['NET_PROFIT_AFTER_DED_NR_LP_TTM']].rolling(20, min_periods=12).apply(lambda x: _e_vol(x), 'raw=True').values
        data_FS.loc[i_code, ['QUALITY_AGRO']] = data_FS.loc[i_code, ['TOT_ASSETS']].rolling(20, min_periods=12).apply(lambda x: _lt_growth(x), 'raw=True').values

    data_FS.reset_index(inplace=True)
    columns_retain = ['WIND_CODE', 'stock_id', 'ACTUAL_ANN_DT', 'REPORT_PERIOD', 'QUALITY_ROA', 'QUALITY_ROE',
                      'QUALITY_ACCF', 'QUALITY_GPM', 'QUALITY_VERN', 'QUALITY_AGRO', 'GROWTH_EGRO', 'GROWTH_CGRO',
                      'GROWTH_GPGRO', 'GROWTH_GPMGRO', 'GROWTH_ATOGRO']
    columns_negative = ['QUALITY_VERN', 'QUALITY_AGRO']
    data_FS[columns_negative] = - data_FS[columns_negative]
    stock_financial_descriptor = data_FS[columns_retain].copy()
    return stock_financial_descriptor


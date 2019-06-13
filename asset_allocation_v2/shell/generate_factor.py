#coding=utf-8
'''
Created on: May. 19, 2019
Modified on: Jun. 4, 2019
Author: Ning Yang, Shixun Su
Contact: yangning@licaimofang.com
在计算股票在t时刻的因子时，可能会用到收盘时(收盘价)或收盘后(财务报表)的数据。
因此，为了避免引入未来数据，建议最早在t+1时刻使用t时刻计算的因子。
'''

import sys
import logging
import multiprocessing
import numpy as np
import pandas as pd
import statsmodels.api as sm
# from ipdb import set_trace
sys.path.append('shell')
from db import wind_asharecalendar, wind_aindexmembers, wind_ashareeodderivativeindicator, wind_ashareeodprices, wind_aindexeodprices, wind_ashareswindustriesclass, factor_sf_stock_factor_exposure
from calc_descriptor import calc_stock_financial_descriptor, calc_stock_numerical_descriptor
from trade_date import ATradeDate


logger = logging.getLogger(__name__)


class YNStockFactor:

    def __init__(self, reindex, look_back=300, percentage=1.0, calc_method='z_score'):

        self.reindex = reindex
        self.look_back = look_back # > 256 + 21
        self.percentage = percentage
        if (calc_method != 'z_score') and (calc_method != 'raw'):
            raise ValueError
        self.calc_method = calc_method

        self.reindex_total = ATradeDate.trade_date(begin_date=self.reindex[0], end_date=self.reindex[-1], lookback=self.look_back+2)

        self.df_index_historical_constituents = wind_aindexmembers.load_a_index_historical_constituents(index_id='000985.CSI')
        self.stock_ids_total = pd.Index(self.df_index_historical_constituents.stock_id.unique())

        self.df_financial_descriptor = calc_stock_financial_descriptor(stock_ids=self.stock_ids_total)

        self.df_derivative_indicator = wind_ashareeodderivativeindicator.load_a_stock_derivative_indicator(
            stock_ids=self.stock_ids_total,
            reindex=self.reindex_total
        )
        self.df_price = wind_ashareeodprices.load_a_stock_adj_price(
            stock_ids=self.stock_ids_total,
            reindex=self.reindex_total
        )
        self.df_price.loc[:, 'benchmark'] = wind_aindexeodprices.load_a_index_nav(index_ids='000985.CSI', reindex=self.reindex_total)
        self.df_return = self.df_price.pct_change().iloc[1:]
        del self.df_price
        self.df_sw = wind_ashareswindustriesclass.load_a_stock_historical_sw_industry(stock_ids=self.stock_ids_total)

    def generate_stock_factor_days(self, override=False):

        table_name = f'sf_stock_factor_exposure_{self.calc_method}'

        # load existing tradedate
        if override:
            reindex_to_update = self.reindex
        else:
            existing_tradedate = factor_sf_stock_factor_exposure.query_table(table_name=table_name)
            reindex_to_update = self.reindex.difference(existing_tradedate)

        df_descriptor = pd.DataFrame(index=reindex_to_update)
        pool = multiprocessing.Pool(min(multiprocessing.cpu_count()//2, 16))
        res = pool.map(self.generate_stock_factor, reindex_to_update.to_list())
        pool.close()
        pool.join()
        df_descriptor = pd.concat(res)
        # write
        if not df_descriptor.empty:
            factor_sf_stock_factor_exposure.write_factor(df_descriptor, table_name=table_name)

        return df_descriptor

    def generate_stock_factor(self, trade_date):

        stock_ids = self.df_index_historical_constituents.loc[
            (self.df_index_historical_constituents.in_date <= trade_date) &
            ((self.df_index_historical_constituents.out_date > trade_date) |
            (self.df_index_historical_constituents.out_date.isna()))
        ].stock_id.unique()
        df_sw = self.df_sw.loc[
            (self.df_sw.entry_date <= trade_date) &
            ((self.df_sw.remove_date > trade_date) |
             (self.df_sw.remove_date.isna()))
        ]
        df_sw = df_sw.reset_index().drop_duplicates(subset=['stock_id']).set_index('stock_id')

        df_return = self.df_return.loc[:trade_date].iloc[-self.look_back:]
        df_derivative_indicator = self.df_derivative_indicator.loc[:trade_date].iloc[-self.look_back:]
        df_numerical_descriptor = calc_stock_numerical_descriptor(
            df_return=df_return,
            df_derivative_indicator=df_derivative_indicator,
            stock_ids=stock_ids
        )
        df_financial_descriptor = self.select_fs(df_FS=self.df_financial_descriptor, trade_date=trade_date, stock_ids=stock_ids)

        df_descriptor = pd.merge(df_financial_descriptor, df_numerical_descriptor, how='left', left_index=True, right_index=True)

        df_derivative_descriptor = self.df_derivative_indicator.loc[trade_date, ['float_market_value', 'pb', 'pe_ttm', 'pocf_ttm', 'pncf_ttm', 'ps_ttm', 'pdps_ttm']].unstack().T.reindex(stock_ids)
        df_derivative_descriptor[['pb', 'pe_ttm', 'pocf_ttm', 'pncf_ttm', 'ps_ttm', 'pdps_ttm']] = 1.0 / df_derivative_descriptor[['pb', 'pe_ttm', 'pocf_ttm', 'pncf_ttm', 'ps_ttm', 'pdps_ttm']]
        df_derivative_descriptor.loc[:, 'pdps_ttm'] = df_derivative_descriptor['pdps_ttm'].fillna(0.0)
        df_derivative_descriptor.rename(columns={'pb': 'book_to_price', 'pe_ttm': 'earnings_to_price', 'pocf_ttm': 'ocf_to_price','pncf_ttm': 'ncf_to_price', 'ps_ttm': 'sales_to_price', 'pdps_ttm': 'dividend_yield', 'float_market_value': 'weight'}, inplace=True)

        df_descriptor = pd.merge(df_descriptor, df_derivative_descriptor, how='left', left_index=True, right_index=True)
        df_descriptor = pd.merge(df_descriptor, df_sw, how='left', left_index=True, right_index=True)

        select_num = int(df_descriptor.shape[0] * self.percentage)
        df_descriptor = df_descriptor.sort_values(by=['weight'], ascending=False).iloc[:select_num]
        df_descriptor.dropna(subset=['sw_lv1_ind_code', 'weight'], inplace=True)

        financial_descriptor = [
            'book_to_price', 'earnings_to_price', 'ocf_to_price', 'ncf_to_price', 'sales_to_price', 'dividend_yield',
            'roa', 'roe', 'cash_flow_accrual', 'gross_profit_margin', 'assets_turnover', 'earnings_growth','ocf_growth',
            'gross_profit_growth', 'gross_profit_margin_growth', 'assets_turnover_growth','earnings_volatility', 'assets_growth'
        ]
        numerical_descriptor = ['short_term_reverse', 'middle_term_momentum', 'beta', 'monthly_turnover', 'quarterly_turnover', 'annual_turnover']
        style = ['value', 'quality_earnings', 'quality_risk', 'growth', 'liquidity']
        all_descriptor = financial_descriptor + ['linear_size', 'non_linear_size'] + numerical_descriptor + style
        # size
        df_descriptor = self.z_score(df=df_descriptor, columns=['linear_size'], weight='weight')
        df_descriptor['non_linear_size'] = self.calc_non_linear_size(df=df_descriptor)
        df_descriptor = self.z_score(df=df_descriptor, columns=['non_linear_size'], weight='weight')
        if self.calc_method == 'z_score':
            # 对基本面因子进行分行业的标准化处理
            df_descriptor = self.z_score_cbi(df=df_descriptor, columns=financial_descriptor, industry='sw_lv1_ind_code', weight='weight', min_count=30)
            # 对基本面因子和数值因子进行标准化处理
            df_descriptor = self.z_score(df=df_descriptor, columns=financial_descriptor + numerical_descriptor, weight='weight')
            # add style
            df_descriptor['value'] = df_descriptor[['book_to_price', 'earnings_to_price', 'ocf_to_price', 'ncf_to_price']].mean(axis=1, skipna=True)
            df_descriptor['quality_earnings'] = df_descriptor[['roa', 'roe', 'gross_profit_margin', 'assets_turnover']].mean(axis=1, skipna=True)
            df_descriptor['cash_flow_accrual'] = - df_descriptor['cash_flow_accrual']
            df_descriptor['quality_risk'] = df_descriptor[['cash_flow_accrual', 'earnings_volatility', 'assets_growth']].mean(axis=1, skipna=True)
            df_descriptor['growth'] = df_descriptor[['earnings_growth', 'ocf_growth', 'gross_profit_growth', 'gross_profit_margin_growth', 'assets_turnover_growth']].mean(axis=1, skipna=True)
            df_descriptor['liquidity'] = df_descriptor[['monthly_turnover', 'quarterly_turnover', 'annual_turnover']].mean(axis=1, skipna=True)
            df_descriptor['cash_flow_accrual'] = - df_descriptor['cash_flow_accrual']
            # 对风格因子进行标准化处理
            df_descriptor = self.z_score(df=df_descriptor, columns=style, weight='weight')
            df_descriptor[all_descriptor] = df_descriptor[all_descriptor].fillna(df_descriptor[all_descriptor].mean())

        df_descriptor['trade_date'] = trade_date
        df_descriptor = df_descriptor.reset_index().set_index(['trade_date', 'stock_id'])
        columns_retain = ['sw_lv1_ind_code', 'weight'] + financial_descriptor + ['linear_size', 'non_linear_size'] + numerical_descriptor
        if self.calc_method == 'z_score':
            columns_retain += style
        df_descriptor = df_descriptor.loc[:, columns_retain]

        return df_descriptor

    def select_fs(self, df_FS, trade_date, stock_ids):

        df_FS_t = df_FS.loc[df_FS.ann_date < trade_date].copy()
        df_FS_t.sort_values(by=['stock_id', 'ann_date'], ascending=False, inplace=True)
        df_FS_t.drop_duplicates(subset=['stock_id'], keep='first', inplace=True)
        df_FS_t = df_FS_t.set_index('stock_id').reindex(stock_ids)

        return df_FS_t

    def z_score(self, df, columns, weight=None):

        df_t = df.copy()
        for i_column in columns:
            loc_t = df_t[i_column].isin([np.nan, - np.inf, np.inf])
            df_t.loc[loc_t, i_column] = np.nan
            # limit outliers
            df_t.loc[df_t[i_column] > df_t[i_column].mean(skipna=True) + 3 * df_t[i_column].std(skipna=True), i_column] = df_t[i_column].mean(skipna=True) + 3 * df_t[i_column].std(skipna=True)
            df_t.loc[df_t[i_column] < df_t[i_column].mean(skipna=True) - 3 * df_t[i_column].std(skipna=True), i_column] = df_t[i_column].mean(skipna=True) - 3 * df_t[i_column].std(skipna=True)
            # calculate factor exposures
            if weight is not None:
                cap_weighted_mean = np.dot(df_t.loc[~loc_t, i_column], df_t.loc[~loc_t, weight]) / df_t.loc[~loc_t, weight].sum()
                equal_weighted_std = df_t.loc[~loc_t, i_column].std()
                df_t[i_column] = (df_t[i_column] - cap_weighted_mean) / equal_weighted_std
            else:
                equal_weighted_mean = df_t.loc[~loc_t, i_column].mean()
                equal_weighted_std = df_t.loc[~loc_t, i_column].std()
                df_t[i_column] = (df_t[i_column] - equal_weighted_mean) / equal_weighted_std

        return df_t

    def z_score_cbi(self, df, columns, industry='sw_lv1_ind_code', weight='weight', min_count=30):

        df_t = df.copy()
        df_t.dropna(subset=[industry], inplace=True)
        industry_t = list(df_t[industry].unique())
        df_filter = pd.DataFrame()
        for i_industry in industry_t:
            df_t2 = df_t.loc[df_t[industry] == i_industry].copy()
            if len(df_t2.shape) == 2 and df_t2.shape[0] >= min_count:
                df_t2 = self.z_score(df=df_t2, columns=columns, weight=weight)
                df_filter = df_filter.append(df_t2, ignore_index=False)

        return df_filter

    def calc_non_linear_size(self, df):

        df_t = df.copy()
        loc_t = ~df_t.linear_size.isna()
        x = df_t.loc[loc_t, 'linear_size'].values
        y = x ** 3
        ols_result = sm.OLS(y, x).fit()
        df_t.loc[loc_t, 'non_linear_size'] = ols_result.resid

        return df_t[['non_linear_size']]


if __name__ == '__main__':

    reindex = wind_asharecalendar.load_a_trade_date(begin_date='20190101', end_date='20190531')
    stock_factor = YNStockFactor(reindex)
    df = stock_factor.generate_stock_factor_days()


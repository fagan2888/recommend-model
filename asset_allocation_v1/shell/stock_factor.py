#coding=utf8


import getopt
import string
import json
import os
import sys
import logging
sys.path.append('shell')
import click
import config
import pandas as pd
import numpy as np
import time


from datetime import datetime, timedelta
from dateutil.parser import parse
from Const import datapath
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from sqlalchemy import distinct
from tabulate import tabulate
import statsmodels.api as sm
import statsmodels
from db import database, base_trade_dates, base_ra_index_nav, asset_ra_pool_sample, base_ra_fund_nav, base_ra_fund, asset_stock, asset_stock_factor
from factor import Factor
from asset import StockAsset
from db import asset_trade_dates
from multiprocessing import Pool
import math
import scipy.stats as stats
import json
import stock_util


logger = logging.getLogger(__name__)


class StockFactor(Factor):

    def __init__(self, factor_id, asset_ids, exposure = None, factor_name = None):
        super(StockFactor, self).__init__(factor_id, asset_ids, exposure, factor_name)


    #插入股票合法性表
    @staticmethod
    def valid_stock_table():


	all_stocks = StockAsset.all_stock_info()
	all_stocks = all_stocks.reset_index()
	all_stocks = all_stocks.set_index(['sk_secode'])

	st_stocks = stock_util.stock_st()

	all_stocks.sk_listdate = all_stocks.sk_listdate + timedelta(365)

	engine = database.connection('caihui')
	Session = sessionmaker(bind=engine)
	session = Session()
	sql = session.query(asset_stock.tq_qt_skdailyprice.tradedate, asset_stock.tq_qt_skdailyprice.secode ,asset_stock.tq_qt_skdailyprice.tclose, asset_stock.tq_qt_skdailyprice.amount).filter(asset_stock.tq_qt_skdailyprice.secode.in_(all_stocks.index)).statement

	#过滤停牌股票
	quotation_amount = pd.read_sql(sql, session.bind , index_col = ['tradedate', 'secode'], parse_dates = ['tradedate'])

	quotation = quotation_amount[['tclose']]
	quotation = quotation.replace(0.0, np.nan)
	quotation = quotation.unstack()
	quotation.columns = quotation.columns.droplevel(0)

	#60个交易日内需要有25个交易日未停牌
	quotation_count = quotation.rolling(60).count()
	quotation[quotation_count < 25] = np.nan

	#过滤掉过去一年日均成交额排名后20%的股票
	amount = quotation_amount[['amount']]
	amount = amount.unstack()
	amount.columns = amount.columns.droplevel(0)

	year_amount = amount.rolling(252, min_periods = 100).mean()

	def percentile20nan(x):
            x[x <= np.percentile(x,20)] = np.nan
	    return x

	year_amount = year_amount.apply(percentile20nan, axis = 1)

	quotation[year_amount.isnull()] = np.nan

	session.commit()
	session.close()

	#过滤st股票
	for i in range(0, len(st_stocks)):
            secode = st_stocks.index[i]
            record = st_stocks.iloc[i]
            selecteddate = record.selecteddate
            outdate = record.outdate
            if secode in set(quotation.columns):
                    #print secode, selecteddate, outdate
		quotation.loc[selecteddate:outdate, secode] = np.nan

	#过滤上市未满一年股票
	for secode in all_stocks.index:
            if secode in set(quotation.columns):
		quotation.loc[:all_stocks.loc[secode, 'sk_listdate'], secode] = np.nan


	engine = database.connection('asset')
	Session = sessionmaker(bind=engine)
	session = Session()

	records = session.query(distinct(asset_stock_factor.valid_stock_factor.trade_date)).all()
	dates = [record[0] for record in records]

	dates.sort()
	last_date = dates[-20]

	quotation = quotation[quotation.index >= last_date.strftime('%Y-%m-%d')]

	for date in quotation.index:
            #records = []
            for secode in quotation.columns:
		globalid = all_stocks.loc[secode, 'globalid']
		value = quotation.loc[date, secode]
		if np.isnan(value):
                    continue
		valid_stock = asset_stock_factor.valid_stock_factor()
		valid_stock.stock_id = globalid
		valid_stock.secode = secode
		valid_stock.trade_date = date
		valid_stock.valid = 1.0
		#records.append(valid_stock)
		session.merge(valid_stock)


            #session.add_all(records)
            session.commit()

            logger.info('stock validation date %s done' % date.strftime('%Y-%m-%d'))

	session.commit()
	session.close()

	pass


    #过滤掉不合法股票
    @staticmethod
    def stock_factor_filter(factor_df):

	engine = database.connection('asset')
	Session = sessionmaker(bind=engine)
	session = Session()

	for stock_id in factor_df.columns:
            sql = session.query(stock_factor_stock_valid.trade_date, stock_factor_stock_valid.valid).filter(stock_factor_stock_valid.stock_id == stock_id).statement
            valid_df = pd.read_sql(sql, session.bind, index_col = ['trade_date'], parse_dates = ['trade_date'])
            valid_df = valid_df[valid_df.valid == 1]
            if len(factor_df) == 0:
		facto_df.stock_id = np.nan
            else:
		factor_df[stock_id][~factor_df.index.isin(valid_df.index)] = np.nan

	session.commit()
	session.close()

	logger.info('vailid filter done')

	return factor_df








if __name__ == '__main__':

    StockFactor.valid_stock_table()

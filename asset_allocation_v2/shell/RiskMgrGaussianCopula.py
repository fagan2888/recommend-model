# -*- coding: utf-8 -*-
"""
Created at Nov 23, 2016
Author: shengyitao
Contact: shengyitao@licaimofang.com
Company: LCMF
"""


import pandas as pd
import datetime
import numpy as np
import utils
import os
import sys
import click
import DFUtil
from scipy import stats
import random
import scipy as sp
import functools
from multiprocess import Pool
from sqlalchemy.orm import sessionmaker
from db import asset_allocate, database
from sqlalchemy import *



class RiskMgrGaussianCopula(object):

    def __init__(self, index_id, empty=5, maxdd=-0.075, mindd=-0.05, period=252):
        self.maxdd = maxdd
        self.mindd = mindd
        self.empty = empty
        self.period = period
        self.confidence_level = 0.99
        self.index_id = index_id

    def perform(self, asset, df_input):
        #
        # 计算回撤矩阵 和 0.97, 0.75置信区间
        #
        sr_inc = df_input['nav'].pct_change().fillna(0.0)
        sr_log_inc = np.log(sr_inc + 1)
        #gaussion_copula_simulation_cdf = [x/(1e6) for x in range(0, int(1e6))]

        status, empty_days, action, position = 0, 0, 0, 1.0


        def log_ret_var(sr_log_inc, confidence_level, day):
            _log_ret = sr_log_inc[sr_log_inc.index <= day]

            scale_log_ret = (_log_ret - _log_ret.min()) / (_log_ret.max() - _log_ret.min())
            normalization_maxmin_log_ret = _log_ret / scale_log_ret.sum()
            z_score_normalization_maxmin_log_ret = (normalization_maxmin_log_ret - normalization_maxmin_log_ret.mean()) / normalization_maxmin_log_ret.std()
            z_score_normalization_maxmin_log_ret_len = len(z_score_normalization_maxmin_log_ret)
            ecdf_z_score_normalization_maxmin_log_ret = np.arange(1, z_score_normalization_maxmin_log_ret_len + 1) / (1.0 * z_score_normalization_maxmin_log_ret_len)
            inverse_ecdf_z_score_normalization_maxmin_log_ret_func = sp.interpolate.interp1d(ecdf_z_score_normalization_maxmin_log_ret, z_score_normalization_maxmin_log_ret, kind='linear', bounds_error=False, fill_value='extrapolate')

            gaussion_copula_simulation_cdf = [x/(1e6) for x in range(0, int(1e6))]
            inverse_ecdf_z_score_normalization_maxmin_log_ret = inverse_ecdf_z_score_normalization_maxmin_log_ret_func(gaussion_copula_simulation_cdf)

            _ret = inverse_ecdf_z_score_normalization_maxmin_log_ret * _log_ret.std() + _log_ret.mean()

            return pd.Series(_ret).quantile( 1.0 - self.confidence_level)


        engine = database.connection('asset')
        Session = sessionmaker(bind=engine)
        session = Session()
        start_date = session.query(asset_allocate.rm_riskmgr_ecdf_vars.ix_date).filter(asset_allocate.rm_riskmgr_ecdf_vars.index_id == self.index_id).order_by(desc(asset_allocate.rm_riskmgr_ecdf_vars.ix_date)).first()
        session.commit()
        session.close()

        if start_date is None:
            days = sr_log_inc.index[888:]
        else:
            days = sr_log_inc[sr_log_inc.index > start_date[0].strftime('%Y-%m-%d')].index

        pool = Pool(32)
        log_vars = pool.map(functools.partial(log_ret_var, sr_log_inc, self.confidence_level), days)
        pool.close()
        pool.join()

        log_ret_var_ser = pd.Series(log_vars, index = days)

        engine = database.connection('asset')
        Session = sessionmaker(bind=engine)
        session = Session()
        records = []
        for day in log_ret_var_ser.index:
            ecdf_vars = asset_allocate.rm_riskmgr_ecdf_vars()
            ecdf_vars.index_id = self.index_id
            ecdf_vars.ix_date = day
            ecdf_vars.ecdf_var = log_ret_var_ser.loc[day]
            ecdf_vars.created_at = datetime.datetime.now()
            ecdf_vars.updated_at = datetime.datetime.now()
            records.append(ecdf_vars)
        session.add_all(records)
        session.commit()
        session.close()

        engine = database.connection('asset')
        Session = sessionmaker(bind=engine)
        session = Session()
        sql = session.query(asset_allocate.rm_riskmgr_ecdf_vars.ix_date, asset_allocate.rm_riskmgr_ecdf_vars.ecdf_var).filter(asset_allocate.rm_riskmgr_ecdf_vars.index_id == self.index_id).statement
        df = pd.read_sql(sql, session.bind, index_col = ['ix_date'], parse_dates = ['ix_date'])
        log_ret_var_ser = df['ecdf_var']
        log_ret_var_ser = log_ret_var_ser.sort_index()
        session.commit()
        session.close()


        _sr_log_inc = sr_log_inc.loc[log_ret_var_ser.index]
        signal = _sr_log_inc < log_ret_var_ser
        signal = signal.sort_index()

        def riskmgr_signal_pos_nav(riskmgr_signal, ret, start, end):
            riskmgr_signal_num = pd.Series([0] * len(riskmgr_signal), index = riskmgr_signal.index)
            for i in range(len(riskmgr_signal)):
                sig = riskmgr_signal.iloc[i]
                if sig:
                    start_index, end_index = i + start, i + end
                    riskmgr_signal_num.iloc[start_index:end_index,] = riskmgr_signal_num.iloc[start_index:end_index,] + 1
            riskmgr_signal_num = riskmgr_signal_num.reindex(ret.index).fillna(0.0)
            riskmgr_signal_pos = riskmgr_signal_num.apply(lambda signal_num : 1 if signal_num == 0 else (np.exp(-signal_num)-1)/(-signal_num))
            riskmgr_signal_nav = (ret * riskmgr_signal_pos + 1).cumprod()
            return riskmgr_signal_pos, riskmgr_signal_nav


        def best_start_end(signal, ret, end_date):
            _signal = signal[signal.index<=end_date]
            _ret = ret.loc[_signal.index]
            data = []
            for start in range(2, 20):
                for end in range(start + 5, 90):
                    riskmgr_signal_pos, riskmgr_signal_nav = riskmgr_signal_pos_nav(_signal, _ret, start, end)
                    data.append([start, end, riskmgr_signal_nav.iloc[-1]])
            df = pd.DataFrame(data, columns = ['start', 'end', 'nav'])
            df = df.sort_values(by='nav', ascending=False)
            threshold = int((len(df) / 5))
            return (df.iloc[0:threshold].start.mode().values[0], df.iloc[0:threshold].end.mode().values[0])



        engine = database.connection('asset')
        Session = sessionmaker(bind=engine)
        session = Session()
        start_date = session.query(asset_allocate.rm_riskmgr_index_best_start_end.ix_date).filter(asset_allocate.rm_riskmgr_index_best_start_end.index_id == self.index_id).order_by(desc(asset_allocate.rm_riskmgr_index_best_start_end.ix_date)).first()
        session.commit()
        session.close()


        if start_date is None:
            days = signal[signal].index
        else:
            _signal = signal[signal]
            days = _signal[_signal.index > start_date[0].strftime('%Y-%m-%d')].index


        ret = sr_inc.loc[signal.index]
        pool = Pool(32)
        index_best_start_ends = pool.map(functools.partial(best_start_end, signal, ret), days)
        pool.close()
        pool.join()
        index_best_start_end_ser = pd.Series(index_best_start_ends, index = days)

        engine = database.connection('asset')
        Session = sessionmaker(bind=engine)
        session = Session()
        records = []
        for day in index_best_start_end_ser.index:
            index_best_start_end = asset_allocate.rm_riskmgr_index_best_start_end()
            index_best_start_end.index_id = self.index_id
            index_best_start_end.ix_date = day
            index_best_start_end.start = index_best_start_end_ser.loc[day][0]
            index_best_start_end.end = index_best_start_end_ser.loc[day][1]
            index_best_start_end.created_at = datetime.datetime.now()
            index_best_start_end.updated_at = datetime.datetime.now()
            records.append(index_best_start_end)
        session.add_all(records)
        session.commit()
        session.close()

        engine = database.connection('asset')
        Session = sessionmaker(bind=engine)
        session = Session()
        sql = session.query(asset_allocate.rm_riskmgr_index_best_start_end.ix_date, asset_allocate.rm_riskmgr_index_best_start_end.start, asset_allocate.rm_riskmgr_index_best_start_end.end).filter(asset_allocate.rm_riskmgr_index_best_start_end.index_id == self.index_id).statement
        df = pd.read_sql(sql, session.bind, index_col = ['ix_date'], parse_dates = ['ix_date'])
        data = {}
        for day in df.index:
            data[day] = (df['start'].loc[day], df['end'].loc[day])
        index_best_start_end_ser = pd.Series(data)
        index_best_start_end_ser = index_best_start_end_ser.sort_index()
        session.commit()
        session.close()


        riskmgr_signal_num = pd.Series([0] * len(signal), index = signal.index)
        for i in range(len(signal)):
            sig = signal.iloc[i]
            if sig:
                start, end = index_best_start_end_ser.loc[signal.index[i]]
                start_index , end_index = i + start , i + end
                riskmgr_signal_num.iloc[start_index:end_index,] = riskmgr_signal_num.iloc[start_index:end_index,] + 1

        riskmgr_signal_pos = riskmgr_signal_num.apply(lambda signal_num : 1 if signal_num == 0 else (np.exp(-signal_num)-1)/(-signal_num))

        #result_pos = {} # 结果仓位
        #result_act = {} # 结果动作
        df_result = pd.DataFrame({'rm_pos': riskmgr_signal_pos, 'rm_action': riskmgr_signal_num})
        df_result.index.name = 'rm_date'

        return df_result

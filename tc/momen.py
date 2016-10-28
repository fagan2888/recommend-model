# -*- coding: UTF-8 -*-
"""
Created at Aug 14, 2016
Author: shengyitao
Contact: shengyitao@licaimofang.com
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
import datetime
import os

class Tcmomen(object):

    # T1 = 504
    # T2 = 21
    # T3 = 63
    # RS training days
    T1 = 64
    # RS previous days
    T2 = 11
    # half-life dyas
    T3 = 63
    # weight days
    T4 = 64

    def __init__(self, asset_handle, mrt_handle, rf_handle, s_date, e_date):
        self.rs = None
        self.rstr = None
        self.halpha = None
        self.momentum = None
        self.fund_name = "nav"
        # 择时开始时间
        self.s_date = s_date  # datetime.datetime(2014, 1, 3)
        # 择时结束时间
        self.e_date = e_date  # datetime.datetime(2016, 8, 25)
        self.asset = asset_handle  # pd.read_csv(asset_handle, index_col='Date',  parse_dates=["Date"], low_memory=False)
        self.asset = self.asset[self.s_date:self.e_date]
        # 收益计算延迟时间
        self.delay = 2
        # market returns
        self.mrt = mrt_handle  # pd.read_csv(mrt_handle, index_col='Date', parse_dates=["Date"])
        self.rf = self.annualized_rate_2_daily(rf_handle)

        self.cal_h_alpha()
        self.cal_asset_rstr()
        self.get_momentum()

        # 以两天为单位计算动量斜率的结果：分别为胜率、最终净值、最大回撤
        self.two_win_ratio = None
        self.two_nav = None
        self.two_max_draw = None

        # 以大于两天为单位计算动量斜率的结果：分别为胜率、最终净值、最大回撤
        self.long_win_ratio = None
        self.long_nav = None
        self.long_max_draw = None

    def get_return_two_day_momen_grad(self):
        """
        :usage: 以两天为单位计算动量斜率进行择时
        :return: 返回择时净值、最大回撤、信号列表（数据格式为dataframe）
        """
        rt_asset = self.asset.copy()
        momen = self.momentum.copy()
        rt_ratio = rt_asset.pct_change()
        # print type(rt_ratio)
        momen_ratio = momen.pct_change()
        momen_ratio = momen_ratio[1:]
        # print type(momen_ratio)
        momen_cols = list(momen_ratio.columns.values)
        momen_ratio[momen_cols[0]] = np.where(momen_ratio[momen_cols[0]] > 1.0, 1.0, momen_ratio[momen_cols[0]])
        momen_ratio[momen_cols[0]] = np.where(momen_ratio[momen_cols[0]] < -1.0, -1.0, momen_ratio[momen_cols[0]])
        inter_val = pd.merge(momen_ratio, rt_ratio, left_index=True, right_index=True, how='inner')
        momen_cols = list(inter_val.columns.values)

        col_num = len(momen_cols)

        dates_index = inter_val.index

        # 择时收益率
        self_ratio = np.zeros(len(inter_val))
        self_ratio = pd.DataFrame({"self_ratio": self_ratio}, index=dates_index)
        inter_val = inter_val.join(self_ratio)
        # 动量斜率小于0则卖出空仓
        inter_val["self_ratio"] = np.where(inter_val["momen"] < 0.0, 0.0,
                                           inter_val[self.fund_name])

        # delay
        delay_ratio = []
        for _ in range(self.delay):
            delay_ratio.append(0)
        self_ratio_in = list(inter_val["self_ratio"])
        self_ratio_in = self_ratio_in[:-self.delay]
        delay_ratio.extend(self_ratio_in)
        inter_val["self_ratio"] = delay_ratio

        asset_nav = 1.0
        asset_nav_list = []
        for date in dates_index:
            asset_nav *= (1.0 + inter_val[date:date][self.fund_name].values[0])
            asset_nav_list.append(asset_nav)

        dates_index = inter_val.index
        is_holding = np.zeros(len(inter_val))
        add_data = pd.DataFrame({"asset_nav":asset_nav_list, "is_holding":is_holding}, index=dates_index)
        inter_val = inter_val.join(add_data)

        inter_val["is_holding"] = np.where(inter_val["self_ratio"] == 0, -1, 1)
        [td_win, td_total, win_ratio, holding_days, holding_ratio] = self.win_ratio(inter_val)
        [inter_val, f_nav, f_max_draw] = self.cal_nav_maxdown(inter_val)
        inter_val.to_csv("tc_two_day.csv")

        self.two_nav = f_nav
        self.two_max_draw = f_max_draw
        self.two_win_ratio = win_ratio
        return inter_val

    def cal_nav_maxdown(self, union_data):
        """
        :usage: calculate the nav and max drawdown according to the return ratio
        :param union_data:origin data with return ratio
        :return: origin data with nav and max drawdown
        """
        dates = union_data.index
        nav = 1.0
        max_nav = 1.0
        nav_list = []
        max_drawdown_list = []
        for date in dates:
            ratio = union_data[date:date]["self_ratio"].values[0]
            nav *= (1.0 + ratio)
            nav_list.append(nav)
            if nav > max_nav:
                max_nav = nav

            max_drawdown = (nav - max_nav) / max_nav
            max_drawdown_list.append(max_drawdown)

        pd_ret = pd.DataFrame({"self_nav": nav_list, "max_drawdown": max_drawdown_list}, index=dates)
        union_data = union_data.join(pd_ret)
        final_nav = nav_list[-1]
        final_max_draw = min(max_drawdown_list)
        print "nav:", final_nav, "max drawdown:", final_max_draw
        return union_data, final_nav, final_max_draw

    def win_ratio(self, union_data):
        """
        :usage: cal the ratio of TD winning
        :param union_data: origin data with TD signal
        :return: ratio of winning within specific
        """
        dates = union_data.index
        data_num = len(union_data)
        cur_num = 0
        # 择时开始收盘价
        start_close = 0.0
        # 择时结束收盘价
        end_close = 0.0
        # 是否持仓
        is_holding = False
        # 是否是择时开始点
        started = True
        # 是否是择时结束点
        ended = False
        # 总的择时信号
        td_total = 0.0
        # 择时成功数目
        td_win = 0.0

        holding_days = 0.0

        for date in dates:
            cur_num += 1
            cur_row = union_data[date:date]
            hold_type = cur_row["is_holding"].values[0]
            close = cur_row["asset_nav"].values[0]
            ratio = cur_row[self.fund_name].values[0]
            if hold_type == 1:
                is_holding = True
                holding_days += 1
            else:
                is_holding = False

            if cur_num == 1:
                started = True
                start_close = close
            else:
                pre_date = self.get_move_day(dates, date, 1)
                pre_type = union_data[pre_date:pre_date]["is_holding"].values[0]
                if hold_type != pre_type:
                    started = True
                    start_close = close

            if cur_num == data_num:
                ended = True
                end_close = close
            else:
                next_date = self.get_move_day(dates, date, 1, previous=False)
                next_type = union_data[next_date:next_date]["is_holding"].values[0]
                if hold_type != next_type:
                    ended = True
                    end_close = close

            if ended:
                td_total += 1
                if started:
                    if is_holding and ratio > 0.0:
                        td_win += 1
                    elif is_holding and ratio < 0.0:
                        td_win += 1
                else:
                    if is_holding:
                        if end_close > start_close:
                            td_win += 1
                    else:
                        if end_close < start_close:
                            td_win += 1
            started = False
            ended = False

        print [td_win, td_total, td_win / td_total, holding_days, holding_days / data_num]
        return [td_win, td_total, td_win / td_total, holding_days, holding_days / data_num]

    def get_move_day(self, dates, cur_date, counts, previous=True):
        """
        :usage: get the previous or next date depend on pre_count
        :param dates: date list(DateIndex)
        :param cur_date: current date
        :param counts: how many date you want to move
        :param previous: if is "True" get previous date, else get next date
        :return: date
        """

        date = (cur_date - datetime.timedelta(days=counts)) if previous else (cur_date + datetime.timedelta(days=counts))
        date_array = list(dates)
        while date not in date_array:
            counts += 1
            date = (cur_date - datetime.timedelta(days=counts)) if previous \
                else (cur_date + datetime.timedelta(days=counts))

        return date

    def get_return_long_momen_grad(self, period=7, thres=0.002):
        """
        :usage: 以大于两天的天数为单位计算动量斜率进行择时
        :param period: 计算动量斜率的天数，默认为7天
        :param thres: 判断是否持有的阈值，小于thres则卖出空仓
        :return: 返回择时净值、最大回撤、信号列表（数据格式为dataframe）
        """
        rt_asset = self.asset.copy()
        momen = self.momentum.copy()
        rt_ratio = rt_asset.pct_change()

        x = range(0, period)
        momen_num = len(momen)
        pos = 0
        momen_col = list(momen.columns.values)
        momen_grad = []
        for _ in x:
            momen_grad.append(0.0)

        while pos < momen_num - period:
            y = list(momen[momen_col[0]][pos:pos + period])
            lin = linear_model.LinearRegression()
            reg = lin.fit(np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1))
            coef = reg.coef_
            momen_grad.append(coef[0, 0])
            pos += 1
        index_date = momen.index
        grad_inter = pd.DataFrame({"momen_grad": momen_grad}, index=index_date)
        momen_with_grad = momen.join(grad_inter)
        inter_val = pd.merge(momen_with_grad, rt_ratio, left_index=True, right_index=True, how='inner')
        momen_cols = list(inter_val.columns.values)
        col_num = len(momen_cols)

        dates_index = inter_val.index

        # 择时收益率
        self_ratio = np.zeros(len(inter_val))
        self_ratio = pd.DataFrame({"self_ratio":self_ratio}, index=dates_index)
        inter_val = inter_val.join(self_ratio)
        mg = inter_val["momen_grad"]
        mg_avg = np.average(mg)
        mg_avg *= 2.0
        # print inter_val
        inter_val["self_ratio"] = np.where(inter_val["momen_grad"] < thres, 0.0,
                                                      inter_val[self.fund_name])
        delay_ratio = []
        for _ in range(self.delay):
            delay_ratio.append(0)
        self_ratio_in = list(inter_val["self_ratio"])
        self_ratio_in = self_ratio_in[:-self.delay]
        delay_ratio.extend(self_ratio_in)
        inter_val["self_ratio"] = delay_ratio

        asset_nav = 1.0
        asset_nav_list = []
        for date in dates_index:
            asset_nav *= (1.0 + inter_val[date:date][self.fund_name].values[0])
            asset_nav_list.append(asset_nav)
        dates_index = inter_val.index
        is_holding = np.zeros(len(inter_val))
        # print len(nav), len(is_holding), len(inter_val), len(dates_index)
        add_data = pd.DataFrame({"asset_nav":asset_nav_list, "is_holding": is_holding}, index=dates_index)
        inter_val = inter_val.join(add_data)
        inter_val["is_holding"] = np.where(inter_val["self_ratio"] == 0, -1, 1)
        [td_win, td_total, win_ratio, holding_days, holding_ratio] = self.win_ratio(inter_val)
        [inter_val, f_nav, f_max_draw] = self.cal_nav_maxdown(inter_val)

        inter_val.to_csv("tc_period_grad.csv")
        # print len(inter_val)

        self.long_nav = f_nav
        self.long_max_draw = f_max_draw
        self.long_win_ratio = win_ratio
        return inter_val

    def get_momentum(self):
        """
        :usage: cal the momentum with rstr and h-alpha
        :return:
        """

        rstr_alpha = pd.merge(self.rstr, self.halpha, left_index=True, right_index=True, how='inner')
        momen_cols = list(rstr_alpha.columns.values)

        momen = list(0.6 * rstr_alpha[momen_cols[0]] + 0.4 * rstr_alpha[momen_cols[1]])
        index_date = rstr_alpha.index
        momen_inter = pd.DataFrame({"momen": momen}, index=index_date)
        momen = rstr_alpha.join(momen_inter)
        for col in momen_cols:
            del momen[col]

        self.momentum = momen
        momen.to_csv('momentum.csv')

        return momen

    def cal_asset_rstr(self):
        """
        :usage: get the RSTR from RS
        :return: RSTR
        """
        rs = self.cal_asset_rs()
        rstr = pd.rolling_sum(rs, self.T2) / self.T2
        # rstr = rs.rolling(self.T2).sum() / self.T2
        rstr = rstr[(self.T1 + self.T2):]
        self.rstr = rstr
        rstr.to_csv('rstr.csv')

        return rstr

    def cal_asset_rs(self):
        """
        :usage: calculate the rs of asset
        :return: rs of asset
        """
        returns = self.nav_2_rt(self.asset.copy())

        stocks = list(returns.columns.values)

        # rf = get_rf(pd.read_excel(rf_data_handle, index_col='Date'))
        rf = self.rf.copy()
        rs = pd.merge(returns, rf, left_index=True, right_index=True, how='inner')
        rs_num = len(rs)
        hl_weight = []
        for i in range(rs_num):
            hl_weight.append(self.half_life_weight(self.T4, i % self.T4))

        for j in stocks:
            rs[j] = np.log(rs[j] + 1) - np.log(rs['rf'] + 1)
        del rs['rf']

        hl_weight = np.array(hl_weight).reshape(-1, 1)

        rs = rs * hl_weight
        rs = pd.rolling_sum(rs, self.T1)
        # rs = rs.rolling(self.T1).sum()
        self.rs = rs
        rs.to_csv("rs.csv")

        return rs

    def cal_h_alpha(self):
        """
        :usage: calculate the historical alpha
        :return: historical alpha
        """
        rt_asset = self.nav_2_rt(self.asset.copy())
        # rf = get_rf(pd.read_excel(rf_data_handle, index_col='Date'))
        rf = self.rf.copy()
        RS = self.nav_2_rt(self.mrt.copy())

        combond = pd.merge(rt_asset, rf,
                           left_index=True, right_index=True, how='inner')
        combond = pd.merge(combond, RS,
                           left_index=True, right_index=True, how='inner')
        col_names = list(combond.columns.values)
        h_alpha_126 = []
        for i in range(self.T3):
            h_alpha_126.append(0.0)
        combond_num = len(combond)
        pos = 0
        combond.to_csv("combond.csv")
        while pos < combond_num - self.T3:
            y = combond[col_names[0]][pos:pos + self.T3] - combond[col_names[1]][pos:pos + self.T3]
            x = combond[col_names[2]][pos:pos + self.T3] #- combond[col_names[1]][pos:pos + self.T3]
            lin = linear_model.LinearRegression()
            reg = lin.fit(np.array(x).reshape(-1, 1), np.array(y).reshape(-1, 1))
            alpha = reg.intercept_
            h_alpha_126.append(alpha[0])
            pos += 1
        index_date = combond.index
        alpha_inter = pd.DataFrame({"halpha": h_alpha_126}, index=index_date)
        combond_with_alpha = combond.join(alpha_inter)

        for col in col_names:
            del combond_with_alpha[col]

        combond_with_alpha.to_csv("h_alpha.csv")
        self.halpha = combond_with_alpha
        return combond_with_alpha

    @staticmethod
    def nav_2_rt(asset_nav):
        """
        :usage: cal rf from asset net value
        :param asset_nav: nav of asset(format is dataframe)
        :return: rf of asset(format is dataframe)
        """
        asset_nav.fillna(method='pad')
        rt = np.log(asset_nav / asset_nav.shift(1))
        rt = rt[1:]
        stocks = list(rt.columns.values)
        for j in stocks:
            rt[j] = np.where(rt[j] == float('inf'), 0.0, rt[j])
            rt[j] = np.where(rt[j] == float('-inf'), 0.0, rt[j])
        return rt

    @staticmethod
    def annualized_rate_2_daily(asset_rate):
        """
        :usage: transfer annualized rate of return  to daily
        :param asset: asset with annualized rate(format is dataframe with 'Date' and 'rate')
        :return: asset daily rate of return
        """
        asset_rate.fillna(method='pad')
        return (1.0 + asset_rate / 100.0) ** (1.0 / 252.0) - 1.0

    @staticmethod
    def half_life_weight(T, t):
        """
        :usage: cal the half-life weight with T and t
        :param T: decay time period
        :param t: current time
        :return: half-life weight
        """
        return (0.5 ** (1.0 / T)) ** (T - t)

    @staticmethod
    def get_file_name(path_name):
        """
        :usage: get file name without extension name from path name
        :param path_name: string of path name
        :return: string of file name
        """
        path_list = path_name.split("/")
        path_name = path_list[-1].split(".")[0]
        return path_name

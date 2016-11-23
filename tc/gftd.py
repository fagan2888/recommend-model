# -*- coding: UTF-8 -*-
"""
Created at Sep 14, 2016
Author: shengyitao
Contact: shengyitao@licaimofang.com
"""
import pandas as pd
import datetime
import numpy as np
import os
from load_data import load_index

class GFTD(object):

    def __init__(self, index_code, stime, etime):
        #self.data = pd.read_csv(file_handle, index_col="Date", parse_dates=["Date"], low_memory=False)
        self.index_code = index_code
        self.data = load_index(index_code, stime, etime)
        # 择开始时间
        self.s_date = datetime.datetime(2013, 1, 1)
        # 择结束日期
        self.e_date = datetime.datetime(2016, 6, 30)
        self.n1 = 4
        self.n2 = 4
        self.n3 = 4
        # 坤哥建议卖出时间可以提前一天
        self.n4 = 4
        self.uds = []
        self.uds_sum = []
        self.union_data = pd.DataFrame()
        self.give_td(self.data.copy())
        # 择时最后净值
        self.nav = 0.0
        # 择时最大回撤
        self.max_drawdown = 0.0
        # 总信号数
        self.td_times = 0.0
        # 获胜的信号数
        self.win_times = 0.0
        # 胜率
        self.win_ratio = 0.0
        # 数据总数
        self.total_num = 0.0
        # 持有总数
        self.holding_num = 0.0
        # 持有率
        self.holding_ratio = 0.0

    def give_td(self, ori_data):
        """
        :usage: give the timing decision(sell or buy)
        :param ori_data: origin data
        :return: time and decision data
        """
        test_stime = datetime.datetime(2013, 3, 31)
        test_etime = datetime.datetime(2016, 5, 15)
        # retrieve data with specific date
        used_data = ori_data[self.s_date:self.e_date]
        close_data = used_data["close"]

        # get uds
        uds = self.get_uds(close_data)
        # print self.uds

        # get sum of uds
        uds_sum = self.sum_uds(uds)
        # print uds_sum
        # uds_sum = np.array(uds_sum)
        # print uds_sum

        # unit data
        union_data = self.union_data_uds(used_data, uds, uds_sum)
        # union_data = union_data[test_stime:test_etime]

        # give buy or sell signal
        union_data = self.signal_out(union_data)

        # buy or sell according to the buy/sell counts and stop loss strategy
        union_data = self.buy_sell(union_data)

        # cal the nav and max drawdown
        union_data = self.cal_nav_maxdown(union_data)
        union_data.to_csv(self.index_code + " gftd_result.csv")

        # cal the chance of winning and holding ratio
        win, total, win_ratio, holding_days = self.win_ratio(union_data)
        # 以下输出择时总信号数、获胜次数、胜率
        print "td times:", total, "win times:", win, "win ratio:", win_ratio
        self.td_times = total
        self.win_times = win
        self.win_ratio = win_ratio
        # 以下输出持仓天数、择时时间区间内的总天数、持仓率
        print "holding days:", holding_days, "total_days:", len(union_data), "holding ratio:", holding_days / len(union_data)
        self.holding_num = holding_days
        self.total_num = len(union_data)
        self.holding_ratio = holding_days / len(union_data)

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
            type = cur_row["trade_types"].values[0]
            close = cur_row["close"].values[0]

            if type == 1:
                is_holding = True
                holding_days += 1
            else:
                is_holding = False

            if cur_num == 1:
                started = True
                start_close = close
            else:
                pre_date = self.get_move_day(dates, date, 1)
                pre_type = union_data[pre_date:pre_date]["trade_types"].values[0]
                if type != pre_type:
                    started = True
                    start_close = close

            if cur_num == data_num:
                ended = True
                end_close = close
            else:
                next_date = self.get_move_day(dates, date, 1, previous=False)
                next_type = union_data[next_date:next_date]["trade_types"].values[0]
                if type != next_type:
                    ended = True
                    end_close = close

            if ended:
                td_total += 1
                if is_holding:
                    if end_close > start_close:
                        td_win += 1
                else:
                    if end_close < start_close:
                        td_win += 1
            ended = False

        return [td_win, td_total, td_win / td_total, holding_days]

    def buy_sell(self, union_data):
        """
        :usage: buy or sell according to GFTD2 model
        :param union_data: origin data with buy/sell counts
        :return: origin data with user nav
        """
        dates = union_data.index
        # 是否持有
        is_holding = False
        # return ratio list
        returns = []
        buy_low = 0.0

        # 交易记录
        trade_type = 0
        trade_list = []
        for date in dates:
            cur_row = union_data[date:date]
            buy_counts = cur_row["buy_counts"].values[0]
            sell_counts = cur_row["sell_counts"].values[0]
            ratio = cur_row["ratio"].values[0]
            cur_low = cur_row["buy_low"].values[0]
            cur_high = cur_row["sell_high"].values[0]

            if buy_counts == self.n3:
                if not is_holding:
                    trade_type = 1
                    is_holding = True
                    buy_low = cur_row["buy_low"].values[0]
            if is_holding and sell_counts == self.n4:
                is_holding = False
                trade_type = -1

            if is_holding and cur_low < buy_low:
                is_holding = False
                trade_type = -2
            # trade_list.append(trade_type)
            trade_type = 0
            if is_holding:
                returns.append(ratio)
                trade_list.append(1)
            else:
                returns.append(0.0)
                trade_list.append(-1)

        index_date = union_data.index
        pd_ret = pd.DataFrame({"self_ratio": returns, "trade_types":trade_list}, index=index_date)
        union_data = union_data.join(pd_ret)

        return union_data

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
            nav *= (1.0 + ratio / 100.0)
            nav_list.append(nav)
            if nav > max_nav:
                max_nav = nav

            max_drawdown = (nav - max_nav) / max_nav
            max_drawdown_list.append(max_drawdown)

        pd_ret = pd.DataFrame({"self_nav": nav_list, "max_drawdown": max_drawdown_list}, index=dates)
        union_data = union_data.join(pd_ret)

        # 输出择时最后净值、最大回撤
        print "self_nav:", nav_list[-1], "max drawdown:", min(max_drawdown_list)
        return union_data

    def signal_out(self, union_data):
        """
        :usage: give the sell and buy signal with uds_sum
        :param union_data: origin with uds and uds_sum
        :return: origin data with signal
        """
        # print union_data
        dates = union_data.index
        # 是否启动买入记数
        buy_count_signal = False
        # 是否启动卖出记数
        sell_count_signal = False
        # 买入记数
        buy_count = 0
        buy_counts = []
        buy_count_previous_date = None

        # 卖出记数
        sell_count = 0
        sell_counts = []
        sell_count_previous_date = None

        # 止损最高价、最低价
        ## 卖出计数过程中的最高价
        stop_loss_high = 0.0
        stop_loss_high_list = []
        ## 买入计数过程中的最低价
        stop_loss_low = 0.0
        stop_loss_low_list = []

        for date in dates:

            cur_close_for_stop = union_data[date:date]["close"].values[0]
            # 买入启动后的买入计数
            # 是否计数
            is_buy_count = False
            if buy_count_signal:
                # 初始化买入计数条件
                cond_one = False
                cond_two = False
                cond_three = False

                # 当前交易日收盘价
                cur_close = union_data[date:date]["close"].values[0]
                # 当前交易日最高价
                cur_high = union_data[date:date]["high"].values[0]

                # 当前日期之前第2个交易日
                pre_2_date = self.get_move_day(dates, date, 2)
                # 当前日期之前第1个交易日
                pre_1_date = self.get_move_day(dates, date, 1)

                # 之前第2根K线最高价
                pre_2_high = union_data[pre_2_date:pre_2_date]["high"].values[0]
                # 之前第1根K线最高价
                pre_1_high = union_data[pre_1_date:pre_1_date]["high"].values[0]

                # 条件1判断:收盘价大于或者等于之前第2根K线最高价
                if cur_close >= pre_2_high:
                    cond_one = True

                # 条件2判断:最高价大于之前第1根K线的最高价
                if cur_high > pre_1_high:
                    cond_two = True

                # 条件3判断:收盘价大于之前第1个计数的收盘价(这里默认是计数成功且如果之前没有计数成功的条件3成立)
                if buy_count_previous_date is None:
                    cond_three = True
                else:
                    # 之前第1个计数的收盘价
                    pre_first_count_close = union_data[buy_count_previous_date:buy_count_previous_date]["close"].values[0]
                    if cur_close > pre_first_count_close:
                        cond_three = True

                # 三个条件同时成立才能计数
                if cond_one and cond_two and cond_three:
                    is_buy_count = True

                # 更新之前第1个计数日期
                if is_buy_count:
                    buy_count_previous_date = date
                    buy_count += 1
                # print date, buy_count
                # if date == datetime.datetime(2013, 4, 18):
                #     print buy_count_signal
                #     print date, is_count, cond_one, cond_two, cond_three
                #     print date, cur_close, pre_2_high
                #     print buy_count_previous_date
            if is_buy_count:
                # print date, buy_count
                buy_counts.append(buy_count)
            else:
                buy_counts.append(0)

            # 卖出启动后的买入计数
            # 是否计数
            is_sell_count = False
            if sell_count_signal:
                # 初始化买入计数条件
                cond_one = False
                cond_two = False
                cond_three = False

                # 当前交易日收盘价
                cur_close = union_data[date:date]["close"].values[0]
                # 当前交易日最低价
                cur_low = union_data[date:date]["low"].values[0]

                # 当前日期之前第2个交易日
                pre_2_date = self.get_move_day(dates, date, 2)
                # 当前日期之前第1个交易日
                pre_1_date = self.get_move_day(dates, date, 1)

                # 之前第2根K线最低价
                pre_2_low = union_data[pre_2_date:pre_2_date]["low"].values[0]
                # 之前第1根K线最低价
                pre_1_low = union_data[pre_1_date:pre_1_date]["low"].values[0]

                # 收盘价小于或者等于之前第2根K线最低价
                if cur_close <= pre_2_low:
                    cond_one = True

                # 最低价小于之前第1根K线的最低价
                if cur_low < pre_1_low:
                    cond_two = True

                # 收盘价小于之前第1个计数的收盘价
                if sell_count_previous_date is None:
                    cond_three = True
                else:
                    # 之前第1个计数的收盘价
                    pre_first_count_close = union_data[sell_count_previous_date:sell_count_previous_date]["close"].values[0]
                    if cur_close < pre_first_count_close:
                        cond_three = True

                if cond_one and cond_two and cond_three:
                    is_sell_count = True

                # 更新之前第1个计数日期
                if is_sell_count:
                    sell_count_previous_date = date
                    sell_count += 1

            # if sell_count == 4:
            #     print date
            if is_sell_count:
                # print date, buy_count
                sell_counts.append(sell_count)
            else:
                sell_counts.append(0)

            if buy_count_signal:
                if stop_loss_low == 0.0 or stop_loss_low > cur_close_for_stop:
                    stop_loss_low = cur_close_for_stop

            if sell_count_signal:
                if stop_loss_high == 0.0 or stop_loss_high < cur_close_for_stop:
                    stop_loss_high = cur_close_for_stop
            stop_loss_high_list.append(stop_loss_high)
            stop_loss_low_list.append(stop_loss_low)
            # 当计数达到n3,重新计数
            if buy_count == self.n3:
                buy_count = 0
                # 达到计数终止此次买入启动后的计数
                buy_count_signal = False
            if sell_count == self.n3:
                sell_count = 0
                # 达到计数终止此次卖出启动后的计数
                sell_count_signal = False
            # 判断是否买入或者卖出计数, 放在计数逻辑后面是为了启动后一个交易日计数
            row_data = union_data[date:date]
            cur_uds_sum = row_data["uds_sum"].values[0]
            if cur_uds_sum == self.n2:
                sell_count_signal = True
                # 取消上一次卖出计数
                sell_count = 0
                sell_count_previous_date = None
                stop_loss_high = 0.0
            elif cur_uds_sum == -self.n2:
                buy_count_signal = True
                # 取消上一次买入计数
                buy_count = 0
                buy_count_previous_date = None
                stop_loss_low = 0.0

        # print union_data
        # print buy_counts
        # print union_data
        union_data = self.union_data_sell_buy_counts(union_data, sell_counts, buy_counts)
        union_data = self.union_data_high_low(union_data, stop_loss_low_list, stop_loss_high_list)

        return union_data

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

    def union_data_high_low(self, ori_data, sl, tp):
        """

        :param ori_data:
        :param sl:
        :param tp:
        :return:
        """
        index_date = ori_data.index
        high_low = pd.DataFrame({"buy_low":sl, "sell_high":tp}, index=index_date)
        union_data = ori_data.join(high_low)

        return union_data

    def union_data_dataframe(self, ori_data, columns, datas):
        """

        :param ori_data:
        :param columns:
        :param datas:
        :return:
        """
    def union_data_sell_buy_counts(self, ori_data, sell_counts, buy_counts):
        """
        :usage: 把买入和卖出计数加到原始数据中
        :param ori_data: 原始数据(Dataframe)
        :param sell_counts: 每天的卖出计数
        :param buy_counts: 每天的买入计数
        :return: 合并数据
        """

        index_date = ori_data.index
        sell_buy_counts_data = pd.DataFrame({"buy_counts":buy_counts, "sell_counts":sell_counts}, index=index_date)
        union_data = ori_data.join(sell_buy_counts_data)

        return union_data

    def union_data_uds(self, ori_data, uds, uds_sum):
        """
        :usage: add uds and uds_sum to origin data with close/open/high/low
        :param ori_data: origin data(Dataframe)
        :param uds:
        :param uds_sum:
        :return: origin with uds and uds_sum
        """
        index_date = ori_data.index
        uds_data = pd.DataFrame({"uds": uds, "uds_sum": uds_sum}, index=index_date)
        union_data = ori_data.join(uds_data)

        self.union_data = union_data

        return union_data

    def sum_uds(self, uds):
        """
        :usage: sum uds day by day
        :param uds: uds
        :return: result of uds sum(data type:list)
        """
        sum_uds = []
        sum_uds.append(0)
        uds_num = len(uds)
        sums = 0
        for ite in range(1, uds_num):
            if uds[ite] == uds[ite - 1]:
                if sums == -4 or sums == 4:
                    sums = uds[ite]
                else:
                    sums += uds[ite]
            else:
                sums = uds[ite]
            sum_uds.append(sums)

        self.uds_sum = sum_uds

        return sum_uds

    def get_uds(self, close_data):
        """
        :usage: get ud series from close price of stock
        :param: close price of stock
        :return: stock_close data with ud series
        """
        # uds init
        uds = []
        for _ in range(self.n1):
            uds.append(0)
        # cal uds and record
        cur_pos = self.n1
        for value in close_data[self.n1:]:
            ud = 0
            if value > close_data[cur_pos - self.n1]:
                ud = 1
            elif value < close_data[cur_pos - self.n1]:
                ud = -1
            uds.append(ud)
            cur_pos += 1

        self.uds = uds

        return uds

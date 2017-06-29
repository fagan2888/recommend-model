# -*- coding: utf-8 -*-
"""
Created at Jun. 20, 2017
Author: shengyitao
Contact: shengyitao@licaimofang.com
"""
import sys
sys.path.append('./shell')
import datetime
import calendar
import numpy as np
import pandas as pd
import os
from db import portfolio_statistics_ds_orders_pdate as ds_order
from db import portfolio_statistics_ds_share as ds_share
from db import portfolio_statistics_rpt_srrc_apportion as rpt_srrc_apportion
from db import portfolio_statistics_rpt_retention_data as rpt_retention_data
from db import portfolio_statistics_rpt_srrc_rolling as rpt_srrc_rolling

def hprint(con):
    print con
    os._exit(0)

def getBetweenMonth(s_date, e_date):
    """
    给出时间区间(s_date, e_date)内年月列表
    :param s_date: 起始时间(0000-00-00)
    :param e_date: 结束时间(0000-00-00)
    :return: list[(year, month, month days, true days),......]
    current days只对起始时间和结果时间有用，例如结束时间是2017-06-26, month days
    为30，true days为26
    """
    first_days = s_date.day
    end_days = e_date.day
    cursor_year = s_date.year
    cursor_month = s_date.month
    start_date_com_str = str(s_date.year) + str(s_date.month)
    end_date_com_str = str(e_date.year) + str(e_date.month)
    month_list = []
    while str(cursor_year) + str(cursor_month) <= end_date_com_str:
        month_days = calendar.monthrange(cursor_year, cursor_month)[1]
        # if str(cursor_year) + str(cursor_month) == start_date_com_str:
        #     month_list.append((cursor_year, cursor_month, month_days, \
        #         first_days))
        if str(cursor_year) + str(cursor_month) == end_date_com_str:
            month_list.append((cursor_year, cursor_month, month_days, \
                end_days))
        else:
            month_list.append((cursor_year, cursor_month, month_days, \
                month_days))
        if cursor_month == 12:
            cursor_month = 1
            cursor_year += 1
        else:
            cursor_month += 1
    return month_list

class MonthlyStaApportion(object):
    def __init__(self):
        self.min_date = ds_order.get_min_date()
        self.max_date = ds_order.get_max_date()
        self.min_year = self.min_date.year
        self.min_month = self.min_date.month
        self.max_year = self.max_date.year
        self.max_month = self.max_date.month
        self.last_day = self.max_date.day
    def get_month_range(self, cur_year, cur_month):
        cursor_year = self.min_year
        cursor_month = self.min_month
        end_com = datetime.date(cur_year, cur_month, 1)
        end_com_str = str(cur_year) + str(cur_month)
        past_year_month = []
        while datetime.date(cursor_year, cursor_month, 1) <= end_com:
            month_days = calendar.monthrange(cursor_year, cursor_month)[1]
            if str(cursor_year) + str(cursor_month) == end_com_str and \
                str(self.max_year) + str(self.max_month) == end_com_str: \
                past_year_month.append((cursor_year, cursor_month, \
                    month_days, self.last_day))
            else:
                past_year_month.append((cursor_year, cursor_month, \
                    month_days, month_days))
            if cursor_month == 12:
                cursor_month = 1
                cursor_year += 1
            else:
                cursor_month += 1
        return past_year_month
    def get_old_data(self):
        """
        虽然时间滚动到当前月，但上个月数据有可能更新，因为用户份额没有确认完,
        另外要做增量更新
        """
        # 数据库中最新所属月份
        cur_month = rpt_srrc_apportion.get_max_date()
        if cur_month == None:
            old_dict = {}
            old_dict['rp_tag_id'] = []
            old_dict['rp_date'] = []
            old_dict['rp_date_apportion'] = []
            old_dict['rp_user_resub'] = []
            old_dict['rp_user_clear'] = []
            old_dict['rp_user_retain'] = []
            old_dict['rp_amount_resub'] = []
            old_dict['rp_amount_redeem'] = []
            old_dict['rp_amount_aum'] = []
            old_df = pd.DataFrame(old_dict).set_index([ \
                    'rp_date', 'rp_date_apportion'])
        else:
            cur_month = cur_month[1]
            last_year_num = cur_month.year if cur_month.month != 1 \
                else cur_month.year - 1
            last_month_num = cur_month.month - 1 if cur_month.month != 1 else 12
            last_month = datetime.date(last_year_num, last_month_num, \
                    calendar.monthrange(last_year_num, last_month_num)[1])
            old_df = rpt_srrc_apportion.get_old_data([last_month, cur_month])
            old_df = pd.DataFrame(old_df)
            old_df = old_df.iloc[:, :-2]
            old_df = old_df.set_index(['rp_date', 'rp_date_apportion'])
        return old_df
    def incremental_update(self):
        """
        增量更新
        """
        # 数据库中最新所属月份
        newest_month = rpt_srrc_apportion.get_max_date()
        old_df = self.get_old_data()
        if newest_month == None:
            new_df = self.deal_data(self.min_year, self.min_month)
            #rpt_srrc_apportion.batch(new_df, old_df)
        else:
            newest_month = newest_month[1]
            last_year_num = newest_month.year if newest_month.month != 1 \
                else newest_month.year - 1
            last_month_num = newest_month.month - 1 if newest_month.month != 1 else 12
            # last_month = datetime.date(last_year_num, last_month_num, \
            #         calendar.monthrange(last_year_num, last_month_num)[1])
            new_df = self.deal_data(last_year_num, last_month_num)
        # new_df.sort_index(inplace=True)
        # old_df.sort_index(inplace=True)
        rpt_srrc_apportion.batch(new_df, old_df)
    def insert_db(self):
        old_df = self.get_old_data()
        new_df = self.deal_data()
        rpt_srrc_apportion.batch(new_df, old_df)
    def deal_data(self, min_year, min_month):
        end_year = self.max_year
        end_month = self.max_month
        cursor_year = min_year
        cursor_month = min_month
        end_com_str = str(end_year) + str(end_month)
        end_com = datetime.date(end_year, end_month, 1)
        past_year_month = []
        rp_tag_id = []
        rp_date = []
        rp_date_apportion = []
        rp_user_resub = []
        rp_user_clear = []
        rp_user_retain = []
        rp_amount_resub = []
        rp_amount_redeem = []
        rp_amount_aum = []
        while datetime.date(cursor_year, cursor_month, 1) <= end_com:
            start_date = datetime.date(cursor_year, cursor_month, 1)
            month_days = calendar.monthrange(cursor_year, cursor_month)[1]
            past_year_month = self.get_month_range(cursor_year, cursor_month)
            end_date_rp = datetime.date(cursor_year, cursor_month, month_days)
            end_date = datetime.date(cursor_year, cursor_month, \
                    past_year_month[-1][3])
            monthly_order_data = ds_order.get_monthly_data(start_date, end_date)
            clear_uids = set(monthly_order_data[ \
                monthly_order_data['ds_trade_type'] == 30]['ds_uid'] \
            )
            resub_uids = set(monthly_order_data[ \
                monthly_order_data['ds_trade_type'] == 11]['ds_uid'] \
            )
            retain_uids = ds_share.get_specific_month_hold_users(end_date)
            if len(retain_uids) > 0:
                retain_uids = np.array( \
                    retain_uids).reshape(1,len(retain_uids))[0]
            # 处理清仓用户数
            for date_tube in past_year_month:
                rp_tag_id.append(0)
                s_date = datetime.date(date_tube[0], date_tube[1], 1)
                e_date = datetime.date(date_tube[0], date_tube[1], date_tube[3])
                e_date_rp = datetime.date(date_tube[0], date_tube[1], date_tube[2])
                rp_date.append(end_date_rp)
                rp_date_apportion.append(e_date_rp)
                if len(clear_uids) > 0:
                    clear_monthly_num = np.array(ds_order.get_specific_month_num( \
                        s_date, e_date, 10, clear_uids))
                    rp_user_clear.append(clear_monthly_num[0][0])
                else:
                    rp_user_clear.append(0)
                if len(resub_uids) > 0:
                    resub_monthly_num = np.array(ds_order.get_specific_month_num( \
                        s_date, e_date, 10, resub_uids))
                    rp_user_resub.append(resub_monthly_num[0][0])
                else:
                    rp_user_resub.append(0)

                if len(retain_uids) > 0:
                    retain_monthly_num = np.array(ds_order.get_specific_month_num( \
                        s_date, e_date, 10, retain_uids))
                    rp_user_retain.append(retain_monthly_num[0][0])
                else:
                    rp_user_retain.append(0)
                first_buy_uids = ds_order.get_specific_month_uids(s_date, e_date, 10)
                if len(first_buy_uids) > 0:
                    first_buy_uids = np.array( \
                        first_buy_uids).reshape(1,len(first_buy_uids))[0]
                    # 复购金额
                    resub_monthly_amount = ds_order.get_specific_month_amount(start_date, end_date, [11], first_buy_uids)
                    rp_amount_resub.append(resub_monthly_amount[0][0])
                    # 赎回金额
                    redeem_monthly_amount = ds_order.get_specific_month_amount(start_date, end_date, \
                                        [20, 21, 30, 31], first_buy_uids)
                    rp_amount_redeem.append(redeem_monthly_amount[0][0])
                    # 在管资产
                    hold_monthly_amount = ds_share.get_specific_date_amount(end_date, first_buy_uids)
                    rp_amount_aum.append(hold_monthly_amount[0][0])
                else:
                    rp_amount_resub.append(0)
                    rp_amount_redeem.append(0)
                    rp_amount_aum.append(0)
            if cursor_month == 12:
                cursor_month = 1
                cursor_year += 1
            else:
                cursor_month += 1
        new_dict = {}
        new_dict['rp_tag_id'] = rp_tag_id
        new_dict['rp_date'] = rp_date
        new_dict['rp_date_apportion'] = rp_date_apportion
        new_dict['rp_user_resub'] = rp_user_resub
        new_dict['rp_user_clear'] = rp_user_clear
        new_dict['rp_user_retain'] = rp_user_retain
        new_dict['rp_amount_resub'] = rp_amount_resub
        new_dict['rp_amount_redeem'] = rp_amount_redeem
        new_dict['rp_amount_aum'] = rp_amount_aum
        new_df = pd.DataFrame(new_dict).set_index([ \
                'rp_date', 'rp_date_apportion'])
        new_df = new_df.ix[:, ['rp_tag_id',  \
                    'rp_user_resub', 'rp_user_clear', 'rp_user_retain', \
                    'rp_amount_resub', 'rp_amount_redeem', 'rp_amount_aum']]
        new_df.fillna(0, inplace=True)
        return new_df

class MonthlyStaRetention(object):
    """
    统计rpt_retention_data里的数据
    """
    def __init__(self):
        self.retention_type = {100:0, 101:1, 102:2, 103:3, 104:4, 105:5, \
                                106:6, 107:7, 108:8, 109:9, 110:10, 111:11, \
                                112:12}
        # 开始有交易的时间
        self.start_date = ds_order.get_min_date()
        # 当前交易时间
        self.end_date = ds_order.get_max_date()

    def handle(self):
        """
        取出所有交易月份，按月处理
        """
        month_list = getBetweenMonth(self.start_date, self.end_date)
        # 为得到old_df
        dates_list = []
        if len(month_list) >= 14:
            month_list = month_list[-14:]
        for month_tube in month_list:
            m_index = month_list.index(month_tube)
            cur_date = datetime.date(month_tube[0], month_tube[1], month_tube[2])
            old_df = self.get_old_data([cur_date])
            new_df = self.process_by_month(month_tube, month_list[m_index:])
            if new_df != None:
                self.insert_db(old_df, new_df)
    def insert_db(self, old_df, new_df):
        rpt_retention_data.batch(new_df, old_df)
    def get_old_data(self, month_dates):
        old_data = rpt_retention_data.get_old_data(month_dates)
        if len(old_data) == 0:
            old_dict = {}
            old_dict['rp_tag_id'] = []
            old_dict['rp_date'] = []
            old_dict['rp_retention_type'] = []
            old_dict['rp_user_resub'] = []
            old_dict['rp_user_hold'] = []
            old_dict['rp_amount_resub'] = []
            old_dict['rp_amount_redeem'] = []
            old_dict['rp_amount_aum'] = []
            old_df = pd.DataFrame(old_dict).set_index([ \
                    'rp_tag_id', 'rp_date', 'rp_retention_type'])
        else:
            old_df = pd.DataFrame(old_data)
            old_df = old_df.iloc[:, :-2]
            old_df = old_df.set_index(['rp_tag_id', 'rp_date', 'rp_retention_type'])
        return old_df
    def process_by_month(self, cur_month, cur_month_list):
        # 当前月开始时间和结束时间
        s_date = datetime.date(cur_month[0], cur_month[1], 1)
        e_date = datetime.date(cur_month[0], cur_month[1], cur_month[2])
        # 当前月新购买用户
        first_buy_uids = ds_order.get_specific_month_uids(s_date, e_date, 10)
        if len(first_buy_uids) > 0:
            first_buy_uids = np.array( \
                        first_buy_uids).reshape(1, len(first_buy_uids))[0]
        else:
            return None
        rp_tag_id = []
        rp_date = []
        rp_retention_type = []
        rp_user_resub = []
        rp_user_hold = []
        rp_amount_resub = []
        rp_amount_redeem = []
        rp_amount_aum = []
        for re_type, re_value in self.retention_type.iteritems():
            # 没有对应留存类型的数据
            if re_value > len(cur_month_list) - 1:
                break;
            # 留存类型对应日期
            retype_date_tube = cur_month_list[re_value]
            end_date = datetime.date(retype_date_tube[0], \
                    retype_date_tube[1], retype_date_tube[2])
            end_date_rp = datetime.date(retype_date_tube[0], \
                    retype_date_tube[1], retype_date_tube[3])
            rp_tag_id.append(0)
            rp_date.append(e_date)
            rp_retention_type.append(re_type)
            # 复购用户uid
            resub_uids = ds_order.get_specific_month_in_uids(s_date, end_date, \
                        [11], first_buy_uids)
            if len(resub_uids) > 0:
                resub_uids = np.array( \
                        resub_uids).reshape(1, len(resub_uids))[0]
            # 赎回用户uid
            redeem_uids = ds_order.get_specific_month_in_uids(s_date, end_date, \
                        [20, 21, 30, 31], first_buy_uids)
            if len(redeem_uids) > 0:
                redeem_uids = np.array( \
                        redeem_uids).reshape(1, len(redeem_uids))[0]
            # 留存用户uid
            retain_uids = ds_share.get_hold_users_date_uids(end_date_rp, \
                        first_buy_uids)
            if len(retain_uids) > 0:
                retain_uids = np.array( \
                        retain_uids).reshape(1, len(retain_uids))[0]

            resub_num = len(resub_uids)
            redeem_num = len(redeem_uids)
            retain_num = len(retain_uids)
            # 复购用户数
            rp_user_resub.append(resub_num)
            # 留存用户数
            rp_user_hold.append(retain_num)

            # 复购金额
            resub_amount = ds_order.get_specific_month_amount(s_date, end_date, [11], resub_uids)
            rp_amount_resub.append(resub_amount[0][0])
            # 赎回金额
            redeem_amount = ds_order.get_specific_month_amount(s_date, end_date, \
                                [20, 21, 30, 31], redeem_uids)
            rp_amount_redeem.append(redeem_amount[0][0])
            # 在管资产
            hold_amount = ds_share.get_specific_date_amount(end_date_rp, retain_uids)
            rp_amount_aum.append(hold_amount[0][0])
        new_dict = {}
        new_dict['rp_tag_id'] = rp_tag_id
        new_dict['rp_date'] = rp_date
        new_dict['rp_retention_type'] = rp_retention_type
        new_dict['rp_user_resub'] = rp_user_resub
        new_dict['rp_user_hold'] = rp_user_hold
        new_dict['rp_amount_resub'] = rp_amount_resub
        new_dict['rp_amount_redeem'] = rp_amount_redeem
        new_dict['rp_amount_aum'] = rp_amount_aum
        new_df = pd.DataFrame(new_dict).set_index([ \
                'rp_tag_id', 'rp_date', 'rp_retention_type'])
        new_df = new_df.ix[:, [ \
                    'rp_user_resub', 'rp_user_hold', \
                    'rp_amount_resub', 'rp_amount_redeem', 'rp_amount_aum']]
        new_df.fillna(0, inplace=True)
        return new_df

class MonthlyStaRolling(object):
    def __init__(self):
        # 开始有交易的时间
        self.start_date = ds_order.get_min_date()
        # 当前交易时间
        self.end_date = ds_order.get_max_date()
        # 统计类型
        self.rolling_types = {100:30, 101:60, 102:90, 103:120, 104:150, \
                                105:180, 106:210, 107:240, 108:270, 109:300, \
                                110:330, 111:360}#, 112:390}
    def handle(self):
        """
        从有交易日期开始按自然日处理
        """
        cursor_date = self.start_date
        end_date = self.end_date
        if cursor_date < (end_date - datetime.timedelta(390)):
            cursor_date = end_date - datetime.timedelta(390)
        while cursor_date <= end_date:
            new_df = self.process_by_day(cursor_date)
            old_df = self.get_old_data([cursor_date])
            self.insert_db(new_df, old_df)
            cursor_date = cursor_date + datetime.timedelta(1)
    def get_old_data(self, cur_date):
        old_data = rpt_srrc_rolling.get_old_data(cur_date) 
        if len(old_data) == 0:
            old_dict = {}
            old_dict['rp_tag_id'] = []
            old_dict['rp_date'] = []
            old_dict['rp_retention_type'] = []
            old_dict['rp_user_redeem_ratio'] = []
            old_dict['rp_user_resub_ratio'] = []
            old_dict['rp_amount_redeem_ratio'] = []
            old_dict['rp_amount_resub_ratio'] = []
            old_df = pd.DataFrame(old_dict).set_index([ \
                    'rp_tag_id', 'rp_date', 'rp_retention_type'])
        else:
            old_df = pd.DataFrame(old_data)
            old_df = old_df.iloc[:, :-2]
            old_df = old_df.set_index(['rp_tag_id', 'rp_date', 'rp_retention_type'])
        return old_df
    def insert_db(self, new_df, old_df):
        rpt_srrc_rolling.batch(new_df, old_df)
    def process_by_day(self, cur_date):
        rp_tag_id = []
        rp_date = []
        rp_retention_type = []
        rp_user_redeem_ratio = []
        rp_user_resub_ratio = []
        rp_amount_redeem_ratio = []
        rp_amount_resub_ratio = []
        for rType, day_num in self.rolling_types.iteritems():
            rp_tag_id.append(0)
            rp_date.append(cur_date)
            rp_retention_type.append(rType)
            pre_date = cur_date - datetime.timedelta(days=day_num)
            # # 如果起始时间小于有交易时间则把开始时间作为起始时间
            # if pre_date < self.start_date:
            #     pre_date = self.start_date
            first_buy_uids = ds_order.get_specific_month_uids( \
                            pre_date, pre_date, 10)
            first_buy_num = len(first_buy_uids)
            if first_buy_num > 0:
                first_buy_uids = np.array( \
                            first_buy_uids).reshape(1, len(first_buy_uids))[0]

            # 留存用户uid
            retain_uids = ds_share.get_hold_users_date_uids(cur_date, \
                        first_buy_uids)
            retain_num = len(retain_uids)
            if retain_num > 0:
                retain_uids = np.array( \
                        retain_uids).reshape(1, len(retain_uids))[0]
            # 赎回用户uid
            redeem_uids = ds_order.get_specific_month_in_uids( \
                        pre_date, cur_date, [20, 21, 30, 31], first_buy_uids)
            redeem_num = len(redeem_uids)
            if redeem_num > 0:
                redeem_uids = np.array( \
                        redeem_uids).reshape(1, len(redeem_uids))[0]
            # 复购用户uid
            resub_uids = ds_order.get_specific_month_in_uids( \
                        pre_date, cur_date, [11], first_buy_uids)
            resub_num = len(resub_uids)
            if resub_num > 0:
                resub_uids = np.array( \
                        resub_uids).reshape(1, len(resub_uids))[0]
            # 统计各种赎回、复购率
            retain_ratio = 0
            redeem_ratio = 0
            resub_ratio = 0
            if first_buy_num > 0:
                retain_ratio = float(retain_num) / first_buy_num
                redeem_ratio = float(redeem_num) / first_buy_num
                resub_ratio = float(resub_num) / first_buy_num
            rp_user_redeem_ratio.append(redeem_ratio)
            rp_user_resub_ratio.append(resub_ratio)
            # 首次购买金额
            first_buy_amount = ds_order.get_specific_month_amount(pre_date, \
                                pre_date, [10], first_buy_uids)
            first_buy_amount = first_buy_amount[0][0]
            # 赎回金额
            redeem_amount = ds_order.get_specific_month_amount(pre_date, cur_date, \
                                [20, 21, 30, 31], first_buy_uids)
            redeem_amount = redeem_amount[0][0]
            # 复购金额
            resub_amount = ds_order.get_specific_month_amount(pre_date, \
                                cur_date, [11], first_buy_uids)
            resub_amount = resub_amount[0][0]
            # 统计各种赎回、复购金额率
            redeem_amount_ratio = 0
            resub_amount_ratio = 0
            # if cur_date == datetime.date(2016, 9, 21):
            #     print "###########"
            #     print cur_date, pre_date
            #     print first_buy_num
            #     print first_buy_amount, resub_amount
            #     print "###########"
            #     os._exit(0)
            if first_buy_amount > 0:
                if redeem_amount != None:
                    redeem_amount_ratio = redeem_amount / first_buy_amount
                if resub_amount != None:
                    resub_amount_ratio = resub_amount / first_buy_amount
            rp_amount_redeem_ratio.append(redeem_amount_ratio)
            rp_amount_resub_ratio.append(resub_amount_ratio)
        new_dict = {}
        new_dict['rp_tag_id'] = rp_tag_id
        new_dict['rp_date'] = rp_date
        new_dict['rp_retention_type'] = rp_retention_type
        new_dict['rp_user_resub_ratio'] = rp_user_resub_ratio
        new_dict['rp_user_redeem_ratio'] = rp_user_redeem_ratio
        new_dict['rp_amount_resub_ratio'] = rp_amount_resub_ratio
        new_dict['rp_amount_redeem_ratio'] = rp_amount_redeem_ratio
        new_df = pd.DataFrame(new_dict).set_index([ \
                'rp_tag_id', 'rp_date', 'rp_retention_type'])
        new_df = new_df.ix[:, [ \
                    'rp_user_redeem_ratio', 'rp_user_resub_ratio', \
                    'rp_amount_redeem_ratio', 'rp_amount_resub_ratio']]
        new_df.fillna(0, inplace=True)
        return new_df
if __name__ == "__main__":
    # obj = MonthlyStaApportion()
    # obj.incremental_update()
    # obj_reten = MonthlyStaRetention()
    # obj_reten.handle()
    obj_rolling = MonthlyStaRolling()
    obj_rolling.handle()

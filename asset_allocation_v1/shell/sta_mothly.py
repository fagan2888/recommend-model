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
from db import portfolio_statistics_ds_orders as ds_order
from db import portfolio_statistics_ds_share as ds_share
from db import portfolio_statistics_rpt_srrc_apportion as rpt_srrc_apportion

def hprint(con):
    print con
    os._exit(0)
class MothlySta(object):
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
            print past_year_month
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
                    print "#######"
                    print end_date
                    print "######"
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
        print len(rp_amount_aum)
        new_df = pd.DataFrame(new_dict).set_index([ \
                'rp_date', 'rp_date_apportion'])
        new_df = new_df.ix[:, ['rp_tag_id',  \
                    'rp_user_resub', 'rp_user_clear', 'rp_user_retain', \
                    'rp_amount_resub', 'rp_amount_redeem', 'rp_amount_aum']]
        new_df.fillna(0, inplace=True)
        print len(new_df)
        return new_df


if __name__ == "__main__":
    obj = MothlySta()
    obj.incremental_update()

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
    def deal_data(self):
        curent_year = self.min_year
        curent_month = self.min_month
        end_year = self.max_year
        end_month = self.max_month
        cursor_year = self.min_year
        cursor_month = self.min_month
        end_com = str(end_year) + str(end_month)
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
        while str(cursor_year) + str(cursor_month) <= end_com:
            start_date = datetime.date(cursor_year, cursor_month, 1)
            month_days = calendar.monthrange(cursor_year, cursor_month)[1]
            end_date = datetime.date(cursor_year, cursor_month, month_days)
            monthly_order_data = ds_order.get_monthly_data(start_date, end_date)
            past_year_month.append((cursor_year, cursor_month, 1))
            # clear_uids = set(monthly_order_data[ \
            #     monthly_order_data['ds_trade_type'] == 31]['ds_uid'] \
            # )
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
            print end_date, len(retain_uids)
            # 处理清仓用户数
            for date_tube in past_year_month:
                rp_tag_id.append(0)
                s_date = datetime.date(date_tube[0], date_tube[1], 1)
                e_date = datetime.date(date_tube[0], date_tube[1], date_tube[2])
                rp_date.append(end_date)
                rp_date_apportion.append(e_date)
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
                    resub_monthly_amount = ds_order.get_specific_month_amount(start_date, end_date, 11, first_buy_uids)
                    rp_amount_resub.append(resub_monthly_amount[0][0])
                else:
                    rp_amount_resub.append(0)
            print(np.where(rp_amount_resub == None, 0, rp_amount_resub))
            print "##############"
            if cursor_month == 12:
                cursor_month = 1
                cursor_year += 1
            else:
                cursor_month += 1

if __name__ == "__main__":
    obj = MothlySta()
    obj.deal_data()

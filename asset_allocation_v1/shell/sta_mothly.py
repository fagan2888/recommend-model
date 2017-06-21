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
        inserted = pd.DataFrame()
        while str(cursor_year) + str(cursor_month) <= end_com:
            start_date = datetime.date(cursor_year, cursor_month, 1)
            month_days = calendar.monthrange(cursor_year, cursor_month)[1]
            end_date = datetime.date(cursor_year, cursor_month, month_days)
            monthly_order_data = ds_order.get_monthly_data(start_date, end_date)
            past_year_month.append((cursor_year, cursor_month, month_days))
            # clear_uids = set(monthly_order_data[ \
            #     monthly_order_data['ds_trade_type'] == 31]['ds_uid'] \
            # )
            clear_uids = set(monthly_order_data[ \
                monthly_order_data['ds_trade_type'] == 30]['ds_uid'] \
            )
            print len(clear_uids)
            for date_tube in past_year_month:
                s_date = datetime.date(date_tube[0], date_tube[1], 1)
                e_date = datetime.date(date_tube[0], date_tube[1], date_tube[2])
                clear_monthly_num = ds_order.get_specific_month_num( \
                    s_date, e_date, 10, clear_uids)
                print clear_monthly_num
            print "##############"
            if cursor_month == 12:
                cursor_month = 1
                cursor_year += 1
            else:
                cursor_month += 1

if __name__ == "__main__":
    obj = MothlySta()
    obj.deal_data()

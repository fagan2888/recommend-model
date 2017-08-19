# -*- coding: utf-8 -*-
import datetime
import math
import numpy as np
import pandas as pd
import os

def get_file_name(path_name):
    """
    :usage: get file name without extension name from path name
    :param path_name: string of path name
    :return: string of file name
    """
    path_list = path_name.split("/")
    path_name = path_list[-1].split(".")[0]
    return path_name


def get_move_day(dates, cur_date, counts, previous=True):
    """
    :usage: get the previous or next date depend on parameter pre_count and dates
    :param dates: date list(DateIndex)
    :param cur_date: current date
    :param counts: how many date you want to move
    :param previous: if is "True" get previous date, else get next date
    :return: date
    """

    # date = (cur_date - datetime.timedelta(days=counts)) if previous else (cur_date + datetime.timedelta(days=counts))
    date = cur_date
    date_array = list(dates)
    date_num = len(date_array)
    date_count = 1
    previous_count = 0
    while previous_count < counts:
        if date in date_array:
            previous_count += 1
            if previous_count >= counts:
                break
        date = (cur_date - datetime.timedelta(days=date_count)) if previous \
            else (cur_date + datetime.timedelta(days=date_count))

        if date_count > date_num:
            break
        date_count += 1

    return date


def sigmoid(x):
    """
    :usage: sigmoid function
    :param x: data array
    :return: data array of sigmoid
    """
    return 1.0 / (1.0 + np.exp(-x))

def cal_nav_maxdrawdown(return_lsit):
    """
    :usage: 根据收益率列表求净值
    :param retun_lsit: 收益率列表
    :return: 净值列表
    """
    num = len(return_lsit)
    nav_list = []
    max_drawdown_list = []
    cur_nav = 1.0
    max_nav = 1.0
    for i in range(num):
        cur_nav *= (1.0 + return_lsit[i])
        nav_list.append(cur_nav)
        if cur_nav < max_nav:
            drawdown = (cur_nav - max_nav) / max_nav
            max_drawdown_list.append(drawdown)
        else:
            max_drawdown_list.append(0.0)
            max_nav = cur_nav

    return [nav_list, max_drawdown_list]

def day_2_week(data_df, trade_dates):
    """
    :usage: 把日数据转化成周数据（包括周最高价最低价等）
    :param data_df: 输入数据
        type: DataFrame
        format:
                        close_pre   close    open    high     low     volume  \
            date
            2005-01-04    1000.00  982.79  994.77  994.77  980.66  7412869.0
            2005-01-05     982.79  992.56  981.58  997.32  979.88  7119109.0
            2005-01-06     992.56  983.17  993.33  993.79  980.33  6288029.0
            2005-01-07     983.17  983.96  983.05  995.71  979.81  7298694.0
            2005-01-10     983.96  993.88  983.76  993.96  979.79  5791698.0
    :param trade_dates: 输入数据
        type: DataFrame
        format:
            date,trade_type
            1997-07-15,0
            1997-07-16,0
            1997-07-17,0
    :return week_df: 返回数据
        type: DataFrame
        format: same to input data
    """
    high = []
    low = []
    close = []
    volume = []
    popen = []

    data_index = data_df.index
    start_date = data_index[0]
    end_date = data_index[-1]
    #print end_date

    used_trade_dates = trade_dates[trade_dates.index.get_level_values(0) <= end_date]
    #used_trade_dates = trade_dates[trade_dates.index.get_level_values(0) <= '2017-08-17']
    used_trade_dates = used_trade_dates[used_trade_dates.index.get_level_values(0) >= start_date]
    used_trade_dates = used_trade_dates.sort_index()
    #print used_trade_dates.tail()

    week_dates = used_trade_dates[(used_trade_dates['trade_type'] & 2) > 0].index
    if week_dates[-1].date() < datetime.datetime.now().date():
        week_dates = week_dates.append(used_trade_dates.index[-1:])
    #print week_dates
    #os._exit(0)
    low_date = used_trade_dates.index[0] - datetime.timedelta(days=1)
    high_date = week_dates[0]
    data_df.fillna(method='ffill', inplace=True)
    for date in week_dates:
        high_date = date
        tmp_df = data_df[data_df.index.get_level_values(0) > low_date]
        tmp_df = tmp_df[tmp_df.index.get_level_values(0) <= high_date]
        high.append(tmp_df['high'].max())
        low.append(tmp_df['low'].min())
        volume.append(tmp_df['volume'].sum())
        if not tmp_df.empty:
            close.append(tmp_df['close'][-1])
            popen.append(tmp_df['open'][0])
        else:
            close.append(np.nan)
            popen.append(np.nan)
        low_date = high_date
    week_df = pd.DataFrame({"high":high, "low":low, "close":close, \
                "volume":volume, "open":popen}, index=week_dates)
    # week_df.fillna(method='ffil', inplace=True)
    # week_df.to_csv("W00003_data_week.csv", encoding='utf8')
    return week_df
    # week_df.to_csv("000300_data_week.csv", encoding='utf8')
def rolling_window(a, window, axis=-1):
  '''Return a windowed array.

  from github of joe-antognini's code.
  https://gist.github.com/joe-antognini/ebef3ecfb7624d2980eae7ead8007cfc

  Parameters:
    a: A numpy array
    window: The size of the window

  Returns:
    An array that has been windowed
  '''

  if axis == -1:
    axis = a.ndim - 1

  if 0 <= axis <= a.ndim - 1:
    shape = (a.shape[:axis] + (a.shape[axis] - window + 1, window) +
      a.shape[axis+1:])
    strides = a.strides[:axis] + (a.strides[axis],) + a.strides[axis:]
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
  else:
    raise ValueError('rolling_window: axis out of bounds')
if __name__ == "__main__":
    data_day = pd.read_csv("../tmp/W00003_data.csv", index_col=['date'], \
                            parse_dates=['date'])
    trade_dates = pd.read_csv("../tmp/trade_dates.csv", index_col=['date'], \
                            parse_dates=['date'])
    day_2_week(data_day, trade_dates)

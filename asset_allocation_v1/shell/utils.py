# -*- coding: utf-8 -*-
import datetime
import math
import numpy as np


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

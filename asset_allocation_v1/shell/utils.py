# -*- coding: UTF-8 -*-
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

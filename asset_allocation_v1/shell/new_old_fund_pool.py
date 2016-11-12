#coding=utf8


import pandas as pd
import numpy as np


if __name__ == '__main__':

    fund_pool_new_lines = open('./data/fundpool_new.csv').readlines()
    fund_pool_old_lines = open('./data/fundpool_old.csv').readlines()

    new_date_dict = {}
    old_date_dict = {}
    code_name = {}

    for line in fund_pool_new_lines:
        vec = line.strip().split(',')
        d = vec[0].strip()
        code = vec[1].strip()
        name = vec[2].strip()
        codes = new_date_dict.setdefault(d, [])
        codes.append(code)
        code_name[code] = name



    for line in fund_pool_old_lines:
        vec = line.strip().split(',')
        d = vec[0].strip()
        code = vec[1].strip()
        name = vec[2].strip()
        codes = old_date_dict.setdefault(d, [])
        codes.append(code)
        code_name[code] = name


    dates = new_date_dict.keys()
    dates = list(dates)
    dates.sort()
    for d in dates:
        new_codes = new_date_dict[d]
        old_codes = old_date_dict[d]
        codes = set(new_codes) & set(old_codes)
        for code in codes:
            print d, code, code_name[code]

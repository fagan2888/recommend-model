#coding=utf8


import pandas as pd
from math import e

#week_return : 周收益率
#week_num    : 预测的收益率周数
def predict_r(week_return, week_num):
    pr = e ** (week_return * week_num) - 1
    return pr


if __name__ == '__main__':
    print predict_r(0.1, 10)

# -*- coding: UTF-8 -*-
import xlrd
import pandas as pd
import os
COLUMNS = [u"证券代码",u"买卖方向",u"成交数量",\
                    u"成交价格", u"佣金", u"成交金额", u"业务标志", u"交易类别", u"证券类别"]
filename1 = u"20130101（马丽）.xls"
filename2 = u"20140101（马丽）.xls"
filename3 = u"20150101（马丽）.xlsx"
oriDir = "../origindata/"
interMedDir = "../intermediates/"
path1 = oriDir + filename1
path2 = oriDir + filename2
path3 = oriDir + filename3
index = u"交易日期"
table1 = pd.read_excel(open(path1, 'rb'), 0, index_col = index)
table2 = pd.read_excel(open(path2, 'rb'), 0, index_col = index)
table3 = pd.read_excel(open(path3, 'rb'), 0, index_col = index)
col1 = table1.get(COLUMNS)
col2 = table2.get(COLUMNS)
col3 = table3.get(COLUMNS)
concats = pd.concat([table1, table2], axis = 0)
concats = pd.concat([concats, table3], axis = 0)
colcon = pd.concat([col1, col2], axis = 0)
colcon = pd.concat([colcon, col3], axis = 0)

print colcon.size
colcon.to_csv(interMedDir + "mali.csv", encoding = "utf8")
#concats.to_csv("mali.csv", encoding = "utf8")


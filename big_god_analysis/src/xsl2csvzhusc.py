# -*- coding: UTF-8 -*-
import xlrd
import pandas as pd
import os
COLUMNS = ["证券代码","成交数量","成交价格","业务类型"]
filename1 = "朱士春-2014交割单.csv"
filename2 = "朱士春2015年对账单.csv"
oriDir = "../origindata/"
interMedDir = "../intermediates/"
path1 = oriDir + filename1
path2 = oriDir + filename2
index = "交割日期"
table1 = pd.read_csv(path1, index_col = index)
table2 = pd.read_csv(path2, index_col = index)
col1 = table1.get(COLUMNS)
col2 = table2.get(COLUMNS)
#concats = pd.concat([table1, table2], axis = 0)
#concats = pd.concat([concats, table3], axis = 0)
colcon = pd.concat([col1, col2], axis = 0)

print colcon.size
colcon.to_csv(interMedDir + "zhushichun.csv", encoding = "utf8")
#concats.to_csv("mali.csv", encoding = "utf8")


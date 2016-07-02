# -*- coding: UTF-8 -*-
import xlrd
import pandas as pd
import os
COLUMNS = ["证券代码","成交数量","成交均价","摘要"]
filename1 = "吴广智2013对账单.csv"
filename2 = "吴广智2014对账单.csv"
filename3 = "吴广智2015对账单.csv"
oriDir = "../origindata/"
interMedDir = "../intermediates/"
path1 = oriDir + filename1
path2 = oriDir + filename2
path3 = oriDir + filename3
index = "成交日期"
table1 = pd.read_csv(path1, index_col = index)
table2 = pd.read_csv(path2, index_col = index)
table3 = pd.read_csv(path3, index_col = index)
col1 = table1.get(COLUMNS)
col2 = table2.get(COLUMNS)
col3 = table3.get(COLUMNS)
#concats = pd.concat([table1, table2], axis = 0)
#concats = pd.concat([concats, table3], axis = 0)
colcon = pd.concat([col1, col2, col3], axis = 0)

print colcon.size
colcon.to_csv(interMedDir + "wuguangzhi.csv", encoding = "utf8")
#concats.to_csv("mali.csv", encoding = "utf8")


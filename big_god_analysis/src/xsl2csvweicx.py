# -*- coding: UTF-8 -*-
import pandas as pd
import os
COLUMNS = [u"证券代码",u"成交数量", u"成交价格", u"摘要"]
filenames = {
'filename1':u"2012.1（魏长雄）.xlsx",
'filename2':u"2012.2（魏长雄）.xlsx",
'filename3':u"2012.3（魏长雄）.xlsx",
'filename4':u"2012.4（魏长雄）.xlsx",
'filename5':u"2012.5（魏长雄）.xlsx",
'filename6':u"2012.6（魏长雄）.xlsx",
'filename7':u"2012.7（魏长雄）.xlsx",
'filename8':u"2012.8（魏长雄）.xlsx",
'filename9':u"2012.9（魏长雄）.xlsx",
'filename10':u"2012.10（魏长雄）.xlsx",
'filename11':u"2012.11（魏长雄）.xlsx",
'filename12':u"2012.12（魏长雄）.xlsx",
'filename13':u"2013.1（魏长雄）.xlsx",
'filename14':u"2013.2（魏长雄）.xlsx",
'filename15':u"2013.3（魏长雄）.xlsx",
'filename16':u"2013.4（魏长雄）.xlsx",
'filename17':u"2013.5（魏长雄）.xlsx",
'filename18':u"2013.6（魏长雄）.xlsx",
'filename19':u"2013.7（魏长雄）.xlsx",
'filename20':u"2013.8（魏长雄）.xlsx",
'filename21':u"2013.9（魏长雄）.xlsx",
'filename22':u"2013.10（魏长雄）.xlsx",
'filename23':u"2013.11（魏长雄）.xlsx",
'filename24':u"2013.12（魏长雄）.xlsx",
'filename25':u"2014.1（魏长雄）.xlsx",
'filename26':u"2014.2（魏长雄）.xlsx",
'filename27':u"2014.3（魏长雄）.xlsx",
'filename28':u"2014.4（魏长雄）.xlsx",
'filename29':u"2014.5（魏长雄）.xlsx",
'filename30':u"2014.6（魏长雄）.xlsx",
'filename31':u"2014.7（魏长雄）.xlsx",
'filename32':u"2014.8（魏长雄）.xlsx",
'filename33':u"2014.9（魏长雄）.xlsx",
'filename34':u"2014.10（魏长雄）.xlsx",
'filename35':u"2014.11（魏长雄）.xlsx",
'filename36':u"2014.12（魏长雄）.xlsx",
'filename37':u"2015.1（魏长雄）.xlsx",
'filename38':u"2015.2（魏长雄）.xlsx",
'filename39':u"2015.3（魏长雄）.xlsx",
'filename40':u"2015.4（魏长雄）.xlsx",
'filename41':u"2015.5（魏长雄）.xlsx",
'filename42':u"2015.6（魏长雄）.xlsx",
'filename43':u"2015.7（魏长雄）.xlsx",
'filename44':u"2015.8（魏长雄）.xlsx",
'filename45':u"2015.9（魏长雄）.xlsx",
'filename46':u"2015.10（魏长雄）.xlsx",
'filename47':u"2015.11（魏长雄）.xlsx",
'filename48':u"2015.12（魏长雄）.xlsx"
};
filename1 = "2013.1（魏长雄）.xlsx"
oriDir = "../origindata/"
interMedDir = "../intermediates/"
index = u"交收日期"
table = pd.read_excel(open(oriDir + filename1, 'rb'), 0, index_col = index)
colcon = table.get(COLUMNS)
for i in range(14,49):
    back_str = str(i)
    fname = 'filename' + back_str
    pathx = oriDir + filenames[fname]
    table1 = pd.read_excel(open(pathx, 'rb'), 0, index_col = index)
    cols = table1.get(COLUMNS)
    colcon = pd.concat([colcon, cols], axis = 0)
#os._exit(0)
#path1 = oriDir + filename1
#path2 = oriDir + filename2
#path3 = oriDir + filename3
#index:u"交收日期"
#table1 = pd.read_excel(open(path1, 'rb'), 0, index_col = index)
#table2 = pd.read_excel(open(path2, 'rb'), 0, index_col = index)
#table3 = pd.read_excel(open(path3, 'rb'), 0, index_col = index)
#col1 = table1.get(COLUMNS)
#col2 = table2.get(COLUMNS)
#col3 = table3.get(COLUMNS)
#concats = pd.concat([table1, table2], axis = 0)
#concats = pd.concat([concats, table3], axis = 0)
#colcon = pd.concat([col1, col2], axis = 0)
#colcon = pd.concat([colcon, col3], axis = 0)

print colcon.size
colcon.to_csv(interMedDir + "weichangxiong.csv", encoding = "utf8")
#concats.to_csv("mali.csv", encoding = "utf8")


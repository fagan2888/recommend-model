# -*- coding: UTF-8 -*-
import xlrd
import pandas as pd
from datetime import datetime
import os
#oriDir = "../origindata/"
#table = pd.ExcelFile(oriDir+"20130101（马丽）.xls")
#print table.sheet_names
#sheet = table.parse("Sheet1")
#print sheet.irow(5).real
csvfile = pd.read_csv("file.csv", index_col = "交易日期")
print csvfile
os._exit(0)

COLUMNS = {'code':{'total':0, 'holdings':{}}}
def readXls(path, sheet):
    table = pd.read_excel(open(path, 'rb'), sheet, index_col = u"交易日期")
    return table
def getRowValue (rows, cName, ite):
    row = rows.get(cName)
    res = row.values[ite]
    return res
if __name__ == '__main__':
    path = "../origindata/"+"20130101（马丽）.xls"
    sheetName = "Sheet1"
    table = readXls(path, sheetName)
    dates = table.index.unique()
    columns = table.get([u"证券代码",u"买卖方向",u"成交数量",\
            u"成交价格", u"佣金", u"成交金额", u"成交数量"])
    #period = table[table.index <= 20130107]
    #period = period[period.index >= 20130107]
    #periodCol = period.get([u"证券代码",u"买卖方向",u"成交数量",\
    #        u"成交价格", u"佣金", u"成交金额", u"成交数量"])
    #print periodCol
    for oneDay in dates:
        virFund = {oneDay:{'total':0, 'holdings':{}}}
        #period = table[table.index == oneDay]
        #periodCol = period.get([u"证券代码",u"买卖方向",u"成交数量",\
        #    u"成交价格", u"佣金", u"成交金额", u"成交数量"])
        #periodCol = period[[u"证券代码",u"买卖方向",u"成交数量",\
        #                u"成交价格", u"佣金", u"成交金额"]]
        rows = table.loc[oneDay, [u"证券代码",u"买卖方向",u"成交数量",\
                                    u"成交价格", u"佣金", u"成交金额"]]
        rows.to_csv("file.csv", encoding='utf8')
        rsize = rows.size
        for i in range(rsize):
            row = rows.get(u"证券代码")
            code = row.values
            #buyInOut = 
            print values[1]
        row = rows.get(u"证券代码")
        values = row.values
        print values[1]
        #_iters = rows.iteritems()
        #print _iters
        #print _iters.next()
        #print ites
        #print rows.at(1)
        #for iter in ites:
            #print iter
        #virFund = {oneDay:{'total':0, 'holdings':{}}}
        #print virFund[oneDay]
        #virFund['new'] = 55
        #print virFund
        os._exit(0)


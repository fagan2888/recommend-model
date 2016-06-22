# -*- coding: UTF-8 -*-
import pandas as pd
import os
import VirFund
# steps:
#   1. get csv datas from intermediates
#   2. retrive datas day by day
#   3. update holdings everyday and calculate the yields
fileName = "mali.csv"
interDir = "../intermediates/"
COLUMNS = ["证券代码","买卖方向","成交数量",\
                    "成交价格", "佣金", "成交金额"]
vf = VirFund.virFund()
def calHoldings(stocks, stockValues):
    holdings = 0.0
    for key in stocks.keys():
        holding += stocks[key]['amount'] * stockValues[key]
    return holding
if __name__ == '__main__':
    table = pd.read_csv(interDir + fileName, index_col = "交易日期")
    dates = table.index.unique()
    #按天处理数据
    for oneDay in dates:
        #print oneDay
        operates = table.loc[oneDay, COLUMNS]
        dims = operates.ndim
        row = operates.get(COLUMNS)
        values = row.values
        ites = values.size / 6
        #print ites
        #处理每天的操作数据
        for ite in range(ites):
            if ites == 1:
                value = values
            else:
                value = values[ite]
            code = value[0]
            opt = value[1]
            dealAmount = abs(value[2])
            dealPrice = value[3]
            bonus = value[4]
            dealMoney = value[5]
            if opt == "卖出":
                vf.sell(code, dealAmount, dealPrice)
            elif opt == "买入":
                vf.buy(code, dealAmount, dealPrice)
            else:
                print "不能识别的操作"
            #if opt == "买入":
            #print code,opt, dealAmount, dealPrice, dealMoney
        #vf.setHoldings(2000)
        #print vf.origin
        #print vf.holdings
        #print vf.holdingStocks
        #os._exit(0)
    print vf
    #vf.setHoldings(vf.holdings, stockValues)

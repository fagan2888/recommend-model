#coding=utf8
import pandas as pd
import os
import VirFund
import json
import datetime

# steps:
#   1. get csv datas from intermediates
#   2. retrive datas day by day
#   3. update holdings everyday and calculate the yields
fileName = "weichangxiong.csv"
openName = "open.csv"
closeName = "close.csv"
interDir = "../intermediates/"
st_code = set(['sh', 'sz', 'cyb'])
stock_types = set(["股票", "创业板"])
COLUMNS = ["证券代码","成交数量", "成交价格", "摘要"]
#vf = VirFund.virFund()
def calHoldings(stocks, close_prices, date):
    holdings = 0.0
    for key in stocks.keys():
        stock_value = close_prices.loc[date, key]
        holdings += stocks[key]['amount'] * stock_value
    return holdings
def dealCodes(oriCode):
    clen = len(str(oriCode))
    retCode = oriCode
    if clen == 6:
        retCode = str(oriCode)+"S"
    elif clen < 6:
        tmpCode = str(oriCode)
        for i in range(6 - clen):
            tmpCode = "0"+tmpCode

            retCode = str(tmpCode)+"S"
    return retCode
if __name__ == '__main__':
    table = pd.read_csv(interDir + fileName, index_col = "交收日期")
    #table.dropna(0)
    open_table = pd.read_csv(interDir + openName, index_col = "dates")
    close_table = pd.read_csv(interDir + closeName, index_col = "dates")
    vf = VirFund.virFund()
    user_ratio = []
    user_info = {}
    #print sz_table.loc['2012-12-03', '000001.SZ']
    #print sz_table
    #os._exit(0)
    dates = table.index.unique()
    print len(dates)
    date_count = 0
    pre_date = None
    user_returns = 0.0
    #按天处理数据
    for oneDay in dates:
        date_count += 1
        #print oneDay
        operates = table.loc[oneDay, COLUMNS]
        dims = operates.ndim
        row = operates.get(COLUMNS)
        values = row.values
        ites = values.size / 4
        #print values
        #os._exit(0)
        #print ites
        #处理每天的操作数据
        for ite in range(ites):
            if ites == 1:
                value = values
            else:
                value = values[ite]
            code = value[0]
            if code == code:
                code = int(code)
            else:
                continue
            dealAmount = abs(value[1])
            dealPrice = value[2]
            opt = value[3]
            dcode = dealCodes(code)
            print code
            print dcode
            if (dcode != "600656S") and (dcode != "600832S") and (dcode != "000024S"):
                if opt == "证券卖出":
                    vf.sell(dcode, dealAmount, dealPrice)
                elif opt == "证券买入":
                    vf.buy(dcode, dealAmount, dealPrice)
                else:
                    print "不能识别的操作"
            else:
                print "退市"
        dDate = datetime.datetime.strptime(str(oneDay),'%Y%m%d').strftime("%Y-%m-%d")
        current_holdings = calHoldings(vf.holdingStocks, close_table, dDate)
        vf.setHoldings(current_holdings)
        vf.reset_assets()
        if date_count == 1:
            vf.setReturns(vf.total_assets - vf.origin)
            cur_returns_ratio = (vf.returns / vf.origin) * 100.0
            vf.set_ratio(cur_returns_ratio)
            user_returns += (vf.total_assets - vf.origin)
        else:
            print pre_date
            today_cash_in = vf.origin - user_info[pre_date]['origin']
            today_captical = today_cash_in + user_info[pre_date]["total_assets"]
            vf.setReturns(vf.total_assets - today_captical)
            cur_returns_ratio = (vf.returns / today_captical) * 100.0
            vf.set_ratio(cur_returns_ratio)
            user_returns += (vf.total_assets - today_captical)
        vf.set_total_returns(user_returns)
        vf.set_total_ratio((user_returns / vf.origin) * 100)
        pre_date = dDate
        #print vf
        #print current_holdings
        #os._exit(0)
        user_info[dDate] = {}
        user_info[dDate]['balance'] = float(str(vf.balance))
        user_info[dDate]['origin'] = float(str(vf.origin))
        user_info[dDate]['holdings'] = float(str(vf.holdings))
        user_info[dDate]['returns'] = float(str(vf.returns))
        user_info[dDate]['ratio'] = float(str(vf.ratio))
        user_info[dDate]['total_assets'] = float(str(vf.total_assets))
        user_info[dDate]['total_returns'] = float(str(vf.total_returns))
        user_info[dDate]['total_ratio'] = float(str(vf.total_ratio))
        user_info[dDate]['stocks'] = json.loads(json.dumps(vf.holdingStocks))
        user_ratio.append([dDate, user_info[dDate]['returns'], user_info[dDate]['ratio'], \
            user_info[dDate]['balance'], user_info[dDate]['holdings'], user_info[dDate]['total_assets'],\
            user_info[dDate]['origin'], user_info[dDate]['total_returns'], user_info[dDate]['total_ratio'], user_info[dDate]['stocks']])
        #print user_info
        #os._exit(0)
        #if oneDay == 20130110:
            #print user_info
            #print vf.holdingStocks
            #os._exit(0)
            #if opt == "买入":
            #print code,opt, dealAmount, dealPrice, dealMoney
        #vf.setHoldings(2000)
        #print vf.origin
        #print vf.holdings
        #print vf.holdingStocks
        #os._exit(0)
    print user_returns
    print date_count
    user_info = sorted(user_info.iteritems(), key=lambda d:d[0])
    #print type(user_info)
    udf = pd.DataFrame(user_info, columns=['Date', 'DateValue'])
    udf.to_csv('../outputs/user_info_'+fileName, encoding="utf8")

    urf = pd.DataFrame(user_ratio, columns = \
        ['日期', '当日收益', '当日收益率(%)', '账户可用金额', '当日持仓金额', \
        '当日总资产', '截止当日总投入本金', '截止当日总回报', '戴上当日总回报率(%)', '当前持仓'])
    #['dates', 'returns', 'ratio', 'balance', 'holdings', 'total_assets', 'invest', 'total_returns', 'total_ratio'])
    urf.to_csv('../outputs/user_daily_ratio_'+fileName, encoding="utf8")
    #print user_info[20151201]
    #print vf
    #print vf
    #print vf.holdingStocks.keys()
    #vf.setHoldings(vf.holdings, stockValues)

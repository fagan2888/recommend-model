# -*- coding: UTF-8 -*-
class virFund:
    #截止到当天用户原始投入资金
    origin = 0.0
    #当天可用现金(卖出、赎回的钱)
    balance = 0.0
    #当天持仓资产(金额)
    holdings = 0.0
    #用户当天总资产
    total_assets = balance + holdings
    #当天回报
    returns = 0.0
    #截止到当天总回报(金额)
    total_returns = 0.0
    #当天收益率
    ratio = 0.0
    #截止到当天收益率
    total_ratio = 0.0
    startTime = None
    endTime = None
    #当天持仓股票
    holdingStocks = {}
    def __init(self):
        self.origin = 0.0
        self.holdingStocks = {}
    def setClosePrice(self, code, price):
        self.holdingStocks[code]['close_price'] = price
    def setReturns(self, returns):
        self.returns = returns
    def addReturns(self, returns):
        self.returns += returns
    def reset_assets(self):
        self.total_assets = self.balance + self.holdings
    def set_ratio(self, pratio):
        self.ratio = pratio
    def set_total_ratio(self, pratio):
        self.total_ratio = pratio
    def set_total_returns(self, returns):
        self.total_returns = returns
    def sell(self, code, amount, price):
        isSuccess = False
        if not self.holdingStocks.has_key(code):
            #卖出没有持仓的股票
            print "has no " + code +" in holdings"
        else:
            curAmount = self.holdingStocks[code]['amount']
            if curAmount < amount:
                #卖出份额比持有的多
                print "sell:this stock by before, all will be sold " + code
                self.holdingStocks[code]['amount'] = 0
                del self.holdingStocks[code]
                #self.holdings = self.holdings - curAmount * price
                self.balance += curAmount * price
                isSuccess = True
            else:
                #正常卖出
                self.holdingStocks[code]['amount'] = curAmount - amount
                #删除持仓为0的股票
                if self.holdingStocks[code]['amount'] == 0.0:
                    del self.holdingStocks[code]
                #self.holdings = self.holdings - amount * price
                self.balance += amount * price
                isSuccess = True
        return isSuccess
    def buy(self,code, amount, price):
        money = amount * price
        if self.balance < money:
            self.origin += (money - self.balance)
            self.balance = 0.0
        else:
            self.balance -= money
        #self.holdings += money
        if not self.holdingStocks.has_key(code):
            self.holdingStocks[code] = {}
            self.holdingStocks[code]['amount'] = amount
        else:
            self.holdingStocks[code]['amount'] += amount
        return True
    def setHoldings(self, holdings):
        self.holdings = holdings
    def __repr__(self):
        return "balance:" + str(self.balance) + "origin:" + str(self.origin) +" holdings:" + str(self.holdings) \
                + " holdingStocks:" + self.holdingStocks.__str__
    def __str__(self):
        return "total_assets:" + str(self.total_assets) + "  balance:" + str(self.balance) + " origin:" + str(self.origin) +" holdings:" + str(self.holdings) \
                + " returns:" + str(self.returns) + "ratio:" + str(self.ratio) + " holdingStocks:" + self.holdingStocks.__str__()
if __name__ == '__main__':
    ss = virFund()
    ss.buy(10001, 100, 23)
    ss.sell(10001, 40, 25)
    ss.setHoldings(2000)
    print ss.holdings
    print ss.origin
    print ss.holdingStocks


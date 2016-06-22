# -*- coding: UTF-8 -*-
class virFund:
    origin = 0.0
    holdings = 0.0
    startTime = None
    endTime = None
    holdingStocks = {}
    def __init(self):
        self.origin = 0.0
        self.holdingStocks = {}
    def sell(self, code, amount, price):
        if not self.holdingStocks.has_key(code):
            print "has no " + str(code) +" in holdings"
        else:
            curAmount = self.holdingStocks[code]['amount']
            if curAmount < amount:
                print "sell:this stock by before, all will be sold " + str(code)
                self.holdingStocks[code]['amount'] = 0
                #self.holdings = self.holdings - curAmount * price
                self.origin += curAmount * price
            else:
                self.holdingStocks[code]['amount'] = curAmount - amount
                #self.holdings = self.holdings - amount * price
                self.origin += amount * price
    def buy(self,code, amount, price):
        money = amount * price
        if self.origin < money:
            self.origin = 0.0
        else:
            self.origin -= money
        #self.holdings += money
        if not self.holdingStocks.has_key(code):
            self.holdingStocks[code] = {}
            self.holdingStocks[code]['amount'] = amount
        else:
            self.holdingStocks[code]['amount'] += amount
    def setHoldings(self, holdings):
        self.holdings = holdings
    def __repr__(self):
        return "origin:" + str(self.origin) +" holdings:" + str(self.holdings) \
                + " holdingStocks:" + self.holdingStocks.__str__
    def __str__(self):
        return "origin:" + str(self.origin) +" holdings:" + str(self.holdings) \
                + " holdingStocks:" + self.holdingStocks.__str__()
if __name__ == '__main__':
    ss = virFund()
    ss.buy(10001, 100, 23)
    ss.sell(10001, 40, 25)
    ss.setHoldings(2000)
    print ss.holdings
    print ss.origin
    print ss.holdingStocks


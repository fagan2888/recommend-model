#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import click
from asset import *
from BondIndex import *
from trade_date import *
import DBData
import Const
import matplotlib
import itertools
import random
import datetime
myfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=10)

from ipdb import set_trace
from trade_date import ATradeDate
warnings.filterwarnings("ignore")


class MetaClassMethodWrapper(type):
    '''
    This metaclass aims to automatically change all callable attributes in a specific class into classmethod, for the convenience of implementing a class with no instance.
    '''
    def __new__(cls, name, bases, dct):
        attrs = dict()
        for k, v in dct.items():
            if (not k.startswith('__')) and callable(v):
                attrs[k] = classmethod(v)
            else:
                attrs[k] = v
        return type.__new__(cls, name, bases, attrs)


class BondFactor(object):

    __metaclass__ = MetaClassMethodWrapper
    name = 'Prototype'

    def inc(cls, begin_date=None, end_date=None, reindex=None):
        pass

    def nav(cls, begin_date=None, end_date=None, reindex=None):
        return (1+cls.inc(begin_date, end_date, reindex)).cumprod()

    def deal_with_rf(cls, idx_inc):
        timedelta = []
        for i in range(5):
            rand_i = random.randint(0, len(idx_inc)-1)
            timedelta.append((idx_inc.index[rand_i] - idx_inc.index[rand_i-1]).days)
        timedelta = min(timedelta)
        inc = idx_inc - Const.annual_rf / float(365 // timedelta)
        return inc




class RLevel(BondFactor):

    index = BondIndex("2070000044")
    name = 'level'

    def inc(cls, begin_date=None, end_date=None, reindex=None):
        idx_inc = cls.index.inc(begin_date, end_date, reindex)
        inc = cls.deal_with_rf(idx_inc)
        inc[0] = 0
        inc.name = 'RLevel'
        return inc



class RSlope(BondFactor):

    index_long_period = BondIndex("2070000067")
    index_short_period = BondIndex("2070000066")
    name = 'slope'

    def inc(cls, begin_date=None, end_date=None, reindex=None):
        x_inc = cls.index_long_period.inc(begin_date, end_date, reindex)
        y_inc = cls.index_short_period.inc(begin_date, end_date, reindex)
        x_duration, _ = load_cbdindex_dur_cvx(cls.index_long_period.globalid, reindex=x_inc.index)
        y_duration, _ = load_cbdindex_dur_cvx(cls.index_short_period.globalid, reindex=y_inc.index)
        x_weight = -x_duration/(x_duration - y_duration)
        y_weight = 1 - x_weight
        inc = x_weight * x_inc + y_weight * y_inc
        inc[0] = 0
        inc.name = 'RSlope'
        return inc



class RCurvature(BondFactor):
    lvl_map = {'2070000129' : 1,
               '2070000143' : 3,
               '2070000156' : 5,
               '2070000167' : 7,
               '2070000177' : 10}
    indexes = [BondIndex(k) for k in lvl_map.keys()]
    __nav = None

    def nav(cls, begin_date=None, end_date=None, reindex=None):
        if cls.__nav is None:
            res = []
            yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            for comb in itertools.combinations(cls.indexes, 3):
                comb_inc = pd.DataFrame(map(lambda x: x.inc('2002-01-01', yesterday, ATradeDate.trade_date()), comb)).T
                collection = map(lambda x: load_cbdindex_dur_cvx(x.globalid, reindex=comb_inc.index), comb)
                collection_T = zip(*collection)
                duration, convexity = map(lambda x: pd.DataFrame(list(x)).T, collection_T)
                pre_df = {}
                for day in duration.index:
                    params = np.array([duration.loc[day].values, convexity.loc[day].values, [1,1,1]])
                    weight = np.linalg.solve(params, np.array([0,0,1]))
                    pre_df[day] = weight.dot(comb_inc.loc[day].values)
                final = pd.Series(pre_df)
                res.append(final)
            #  res = np.array(res).sum(axis=0) /10.0
            inc = pd.DataFrame(res).mean().T
            inc[0] = 0
            inc.name = 'RCurvature'
            cls.__nav = (1+inc).cumprod()
        if begin_date is None:
            begin_date = '2010-01-01'
        if end_date is None:
            end_date = '2018-05-01'
        if reindex is None:
            reindex = ATradeDate.week_trade_date()
        nav = cls.__nav.reindex(reindex)
        nav = nav[(begin_date <= nav.index) & (nav.index <= end_date)]
        return nav/nav[0]

    def inc(cls, begin_date=None, end_date=None, reindex=None):
        return cls.nav(begin_date, end_date, reindex).pct_change().fillna(0)




class RCredit(BondFactor):

    index_enterpriseAAA = BondIndex("2070005063")
    index_CDB = BondIndex("2070000275")
    name = 'credit'

    def inc(cls, begin_date=None, end_date=None, reindex=None):
        longpos = cls.index_enterpriseAAA.inc(begin_date, end_date, reindex)
        shortpos = cls.index_CDB.inc(begin_date, end_date, reindex)
        inc = longpos - shortpos
        inc[0] = 0
        inc.name = 'RCredit'
        return inc



#  class RDefault(BondFactor):

    #  index_enterpriseAAA = BondIndex("2070005063")
    #  index_hrenterprise = BondIndex("2070004908")
    #  name = 'default'

    #  def inc(cls, begin_date=None, end_date=None, reindex=None):
        #  longpos = cls.index_hrenterprise.inc(begin_date, end_date, reindex)
        #  shortpos = cls.index_enterpriseAAA.inc(begin_date, end_date, reindex)
        #  inc = longpos - shortpos
        #  inc[0]=0
        #  inc.name = 'RDefault'
        #  return inc



class RConvertible(BondFactor):

    index_convertible = BondIndex('2070003819')
    name = 'convertible'

    def inc(cls, begin_date=None, end_date=None, reindex=None):
        idx_inc = cls.index_convertible.inc(begin_date, end_date, reindex)
        inc = cls.deal_with_rf(idx_inc)
        inc[0] = 0
        inc.name = 'RConvertible'
        return inc



class REquity(BondFactor):

    index_zz800 = BondIndex("2070000191")
    name = 'equity'

    def inc(cls, begin_date=None, end_date=None, reindex=None):
        idx_inc = cls.index_zz800.inc(begin_date, end_date, reindex)
        inc = idx_inc
        inc.name='REquity'
        return inc



class RCurrency(BondFactor):

    index_money = BondIndex('2070006913')
    name = 'currency'

    def inc(cls, begin_date=None, end_date=None, reindex=None):
        idx_inc = cls.index_money.inc(begin_date, end_date, reindex)
        inc = cls.deal_with_rf(idx_inc)
        inc[0] = 0
        inc.name = 'RCurrency'
        return inc

if __name__ == '__main__':
    from trade_date import ATradeDate
    for factor in BondFactor.__subclasses__():
        factor.nav('2011-01-04', '2018-01-04').plot()
        #  factor.nav().plot()
    #  BondIndex('2070000044').nav('2011-01-04', '2018-01-04', ATradeDate.trade_date()).plot()
    plt.legend(loc=0, borderaxespad=0.)
    plt.show()
    #  RCurvature.inc()
    set_trace()

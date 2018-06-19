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
        for i in range(1, 10):
            timedelta.append((idx_inc.index[i] - idx_inc.index[i-1]).days)
        timedelta = min(timedelta)
        #  set_trace()
        inc = idx_inc - Const.annual_rf / float(365 // timedelta)
        return inc

    def duration_immute(cls, x, y, *args):
        x_inc = x.inc(*args)
        y_inc = y.inc(*args)
        x_duration = x.duration(*args)
        y_duration = y.duration(*args)
        x_weight = -y_duration/(x_duration - y_duration)
        y_weight = 1 - x_weight
        mask = abs(x_weight/y_weight + 1) < 0.1
        x_weight[mask] = 1
        y_weight[mask] = -1
        inc = x_weight * x_inc + y_weight * y_inc
        inc[0] = 0
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

    '''
        To acheieve the neutralization of the duration, one have to satisfy:
            \sum_i{ w_i d_i } = 0
        and \sum_i{ w_i } = 1
        See the following code.
    '''
    def inc(cls, begin_date=None, end_date=None, reindex=None):
        inc = -cls.duration_immute(cls.index_long_period, cls.index_short_period, begin_date, end_date, reindex)
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
    name = 'curvature'

    ''' To achieve the neutralization of the duration and the convexity simutaniously,
        the following equations must be satisfied:
            for triple assets:
                \sum_i{ w_i d_i } = 0
                \sum_i{ w_i c_i } = 0
                \sum_i{ w_i } = 0
    '''
    def nav(cls, begin_date=None, end_date=None, reindex=None):
        if cls.__nav is None:
            res = []
            yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            for comb in itertools.combinations(cls.indexes, 3):
                comb_inc = pd.DataFrame(map(lambda x: x.inc('2002-01-01', yesterday, ATradeDate.trade_date()), comb)).T
                collection = map(lambda x: load_cbdindex_dur_cvx(x.globalid, reindex=comb_inc.index), comb)
                collection_T = zip(*collection)
                duration, convexity = map(lambda x: pd.DataFrame(list(x)).T, collection_T)
                # f(i,j) denotes c_i d_j (i,j in {1,2,3}), where the solution consists of this form.
                f = lambda i,j: convexity.iloc[:, i-1] * duration.iloc[:, j-1]
                denominator = -(f(2,1)-f(3,1)-f(1,2)+f(3,2)+f(1,3)-f(2,3))
                w_1 = (-f(3,2) + f(2,3)) / denominator
                w_2 = (f(3,1) - f(1,3)) / denominator
                w_3 = (1-w_1-w_2)
                weights = pd.DataFrame([w_1, w_2, w_3]).T
                weights.columns = comb_inc.columns
                final = (weights*comb_inc).sum(axis=1)
                res.append(final)
            inc = pd.DataFrame(res).mean().T
            inc[0] = 0
            inc.name = 'RCurvature'
            #  cls.__nav = (1+inc).cumprod()
            cls.__nav = (1-inc).cumprod()
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
    index_CDB = BondIndex("2070000276")
    name = 'credit'

    def inc(cls, begin_date=None, end_date=None, reindex=None):
        inc = cls.duration_immute(cls.index_enterpriseAAA, cls.index_CDB, begin_date, end_date, reindex)
        inc.name = 'RCredit'
        return inc



class RDefault(BondFactor):

    index_enterpriseAAA = BondIndex("2070005063")
    # A quickfix
    index_hrenterprise = BondIndex("2070004908", nav_sr = pd.read_excel('2070004908.xls', index_col=0, parse_dates=True).squeeze(), dur = pd.read_excel('2070004908_durcf.xlsx', index_col=0, parse_dates=True).squeeze().sort_index())
    name = 'default'

    def inc(cls, begin_date=None, end_date=None, reindex=None):
        inc = cls.duration_immute(cls.index_enterpriseAAA, cls.index_hrenterprise, begin_date, end_date, reindex)
        inc.name = 'RDefault'
        return inc



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

#Utils
def plotfactors():
    for factor in BondFactor.__subclasses__():
        factor.nav('2011-01-04', '2018-01-04', ATradeDate.trade_date()).plot()
        #  factor.nav().plot()
    #  BondIndex('2070000044').nav('2011-01-04', '2018-01-04', ATradeDate.trade_date()).plot()
    plt.legend(loc=0, borderaxespad=0.)
    plt.show()

def showheatmap():
    import seaborn
    mat = pd.DataFrame(np.corrcoef(map(lambda x:x.inc('2011-01-04', '2018-05-04', ATradeDate.trade_date()), BondFactor.__subclasses__())))
    #  mat = pd.DataFrame(np.corrcoef(map(lambda x:x.inc(), BondFactor.__subclasses__())))
    mat.index = mat.columns = map(lambda x: x.name, BondFactor.__subclasses__())
    mat = abs(mat)
    seaborn.heatmap(mat, annot=True)
    plt.show()


if __name__ == '__main__':
    plotfactors()
    #  showheatmap()
    set_trace()

#coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from asset import *
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from db import *
from trade_date import *
from sklearn.linear_model import Lasso
from scipy.stats import ttest_rel

warnings.filterwarnings("ignore")

class BondIndex(Asset):

    def __init__(self, secode):
        secode = str(secode)
        name = load_bond_index_info(secode).loc[secode, "INDEXNAME"]
        nav_sr = load_cbdindex_nav(secode)
        if nav_sr.empty:
            nav_sr = load_index_nav(secode)
        super(BondIndex, self).__init__(secode, name=name, nav_sr=nav_sr)

    def nav(self, begin_date = None, end_date = None, reindex = None):
        if begin_date is None:
            begin_date = '2013-01-01'
        if end_date is None:
            end_date = '2018-05-01'
        if reindex is None:
            reindex = ATradeDate.week_trade_date()
        nav = super(BondIndex, self).nav(begin_date, end_date, reindex).dropna()
        nav = nav.loc[begin_date:end_date]
        if nav.empty:
            return pd.Series()
        return nav/nav[0]

    def inc(self, begin_date = None, end_date = None, reindex = None):
        return self.nav(begin_date, end_date, reindex).pct_change().fillna(0)

#读取数据库
def load_bond_index_info(secode = None):
    db = database.connection('caihui')
    t1 = Table('tq_ix_basicinfo', MetaData(bind=db), autoload=True)
    columns = [t1.c.SECODE, t1.c.INDEXNAME, t1.c.ESTCLASS]
    if secode == None:
        s = select(columns).where((t1.c.INDEXTYPE == 4) and t1.c.ISVALID)
    else:
        s = select(columns).where(t1.c.SECODE == secode)
    df = pd.read_sql(s, db, index_col="SECODE")
    return df

def load_cbdindex_nav(secode):
    db = database.connection('caihui')
    t1 = Table('tq_qt_cbdindex', MetaData(bind=db), autoload=True)
    columns = [t1.c.TRADEDATE, t1.c.TCLOSE]
    s = select(columns).where(t1.c.SECODE == secode)
    df = pd.read_sql(s, db, index_col='TRADEDATE', parse_dates=['TRADEDATE'])
    return df.squeeze()

def load_index_nav(secode):
    db = database.connection('caihui')
    t1 = Table('tq_qt_index', MetaData(bind=db), autoload=True)
    columns = [t1.c.TRADEDATE, t1.c.TCLOSE]
    s = select(columns).where(t1.c.SECODE == secode)
    df = pd.read_sql(s, db, index_col='TRADEDATE', parse_dates=['TRADEDATE'])
    return df.squeeze()

def get_fund_nav(gid):
    if os.path.exists("tmpfund/%s.csv" % gid):
        nav = pd.read_csv("tmpfund/%s.csv" % gid, index_col=0, parse_dates=True, header=None).squeeze()
        nav.index.name = "date"
        nav.name = "nav"
    else:
        nav = base_ra_fund_nav.load_series(gid)
        if not nav.empty:
            nav.to_csv("tmpfund/%s.csv" % gid)
        else:
            print gid
    return nav.reindex(tdate).loc['2013-01-01':'2018-05-01']

def get_fund_inc(gid):
    return get_fund_nav(gid).pct_change().fillna(0)

bond_fund = base_ra_fund.find_type_fund(2).set_index("globalid")
bond_fund_ids = bond_fund.index.ravel()

#准备因子
lasso = Lasso(alpha=0, fit_intercept=True, positive=True)
tdate = ATradeDate.week_trade_date()
benchmark = BondIndex("2070006886")

#国债/信用债
#  treasuryBond = Asset('120000010')
#  creditBond = Asset('120000011')

#  treasuryBondNav = treasuryBond.nav(begin_date='2013-01-01', end_date='2018-05-01', reindex=tdate)
#  creditBondNav = creditBond.nav(begin_date='2013-01-01', end_date='2018-05-01', reindex=tdate)

#  treasuryBondNav = treasuryBondNav / treasuryBondNav[0]
#  creditBondNav = creditBondNav / creditBondNav[0]

#  treasuryBondInc = treasuryBondNav.pct_change().fillna(0)
#  creditBondInc = creditBondNav.pct_change().fillna(0)

#  index_inc_matrix = np.vstack([treasuryBondInc.values, creditBondInc.values]).T

#pair t test
indexes = pd.read_excel('Book1.xlsx', index_col=0)
index_ids = indexes.index[1:]
def run_ttest_rel():
    blacklist = [2070007385, 2070000071, 2070007387]
    pv = {}
    mean = {}
    stat = {}
    for id_ in index_ids:
        if id_ not in blacklist:
            targetIndex = BondIndex(id_)
            print "========================================"
            print targetIndex.name
            stat_, pvalue =  ttest_rel(targetIndex.inc(), benchmark.inc())
            print pvalue
            stat[id_] = stat_
            pv[id_] = pvalue
            mean[id_] = targetIndex.inc().mean()
    res = pd.DataFrame({'stat':stat, 'pvalue':pv, 'mean':mean})
    res["name"] = indexes.loc[res.index].squeeze()
    return res[(res.pvalue<0.05) & (res.stat>0)]

#  used_factor = [2070000278, 2070006893]
stAA = BondIndex(2070000278)
enterp = BondIndex(2070006893)

def matrix_from_obj(*x):
    return np.vstack(map(BondIndex.inc, x)).T

def factor_regression(gid, factor_matrix):
    fund_name = bond_fund.loc[gid].ra_name
    fund_id = bond_fund.loc[gid].ra_code
    fund_inc = get_fund_inc(gid)
    res = lasso.fit(factor_matrix, fund_inc)
    score = res.score(factor_matrix, fund_inc)
    coefstAA, coefetpr = tuple((res.coef_/res.coef_.sum()))
    return pd.DataFrame([{"fund_id":fund_id, "fund_name":fund_name, "score":score, "coefstAA":coefstAA, "coefetpr":coefetpr}])


def run():
    df_set = []
    for i in bond_fund_ids:
        df_set.append(factor_regression(i, matrix_from_obj(stAA, enterp)))
    return pd.concat(df_set).set_index("fund_id")

if __name__ == "__main__":
    from ipdb import set_trace
    #  res = pd.DataFrame({'name':indexes.loc[sr_pv.index].squeeze(), 'pvalue':sr_pvalue})
    #  df = run()
    #  threshold = 0.03
    #  df_nan = df[np.isnan(df.coeft)]
    #  df = df.dropna()
    #  df_irfund = df[(df.score > threshold) & (df.coefi > df.coeft)]
    #  df_cfund = df[(df.score > threshold) & (df.coeft < df.coefi)]
    #  df_filtered_out = df[(df.score < 0.03)]
    set_trace()

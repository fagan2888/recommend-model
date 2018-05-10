# coding=utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathos.multiprocessing import ProcessingPool as Pool
import RiskMgrVaRs
from TimingGFTD import TimingGFTD
from RiskMgrMonteCarloGenerator import gen_simulation
from RiskMgrSimple import RiskMgrSimple
from CommandTiming import load_index_ohlc
import DFUtil
from db import *
import warnings
import random
import os
import sys
import json

from sqlalchemy import *

from pandas_datareader import data as pdr
import fix_yahoo_finance as yf
yf.pdr_override()

# DMA timing
def dma(sr_nav):
    #  sr_nav = np.exp(sr_inc.cumsum())
    ma120 = sr_nav.rolling(120).mean().dropna()
    ma20 = sr_nav.rolling(20).mean().dropna().reindex(ma120.index)
    signal = pd.Series(np.nan, index=sr_nav.index)
    df = pd.DataFrame({'ma20':ma20, 'ma120':ma120})
    for day, row in df.iterrows():
        if row['ma20'] > row['ma120']:
            signal[day] = 1
        if row['ma20'] < row['ma120']:
            signal[day] = -1
    return signal.fillna(1)

#GFTD timing
def gftd(secode):
    df_ohlc = load_stock_OHLC(secode)
    df_timing = TimingGFTD(4,4,4,4).timing(df_ohlc)
    return df_timing.tc_signal


#Read the info of stock
def load_stock_info():
    db = database.connection('base')
    t1 = Table('ra_stock', MetaData(bind=db), autoload=True)
    columns = [t1.c.sk_code, t1.c.sk_secode, t1.c.sk_name]
    s = select(columns).where(t1.c.sk_liststatus==1)
    df = pd.read_sql(s, db, index_col = 'sk_secode')
    return df


# Load the vaild stocks
def load_vaild_stocks():
    db = database.connection('asset')
    t1 = Table('stock_factor_stock_valid', MetaData(bind=db), autoload=True)
    columns = [t1.c.secode]
    s = select(columns).where(t1.c.valid)
    df = pd.read_sql(s, db, index_col = 'secode')
    return df.index


#  def load_stock(secode):
    #  db = database.connection('caihui')
    #  t1 = Table('tq_sk_dquoteindic', MetaData(bind=db), autoload=True)
    #  columns = [t1.c.TRADEDATE, t1.c.TCLOSEAF, t1.c.VOL]
    #  index_col = ['TRADEDATE']
    #  s = select(columns).where((t1.c.SECODE == secode) and t1.c.ISVALID and (t1.c.VOL != 0.0))
    #  df = pd.read_sql(s, db, index_col=index_col, parse_dates=['TRADEDATE'])
    #  df.index.name = 'tdate'
    #  df.columns = ['closeAF', 'vol']
    #  return df[df.vol!=0].closeAF.sort_index()

def load_stock(secode):
    df = load_stock_OHLC(secode)
    sr = df.tc_close
    sr.name = 'nav'
    return sr


def load_stock_OHLC(secode):
    db = database.connection('caihui')
    t1 = Table('tq_sk_dquoteindic', MetaData(bind=db), autoload=True)
    columns = [t1.c.TRADEDATE, t1.c.TOPENAF, t1.c.THIGHAF, t1.c.TLOWAF, t1.c.TCLOSEAF, t1.c.VOL]
    index_col = ['TRADEDATE']
    s = select(columns).where((t1.c.SECODE == secode) and t1.c.ISVALID and (t1.c.VOL != 0))
    df = pd.read_sql(s, db, index_col=index_col, parse_dates=['TRADEDATE'])
    df = df[df.VOL!=0].drop(['VOL'], axis=1)
    df.index.name = 'tdate'
    df.columns = ['tc_open', 'tc_high', 'tc_low', 'tc_close']
    return df.dropna().sort_index()


def load_stock_us(symbol, data):
    df = data.minor_xs(symbol)
    sr = df['Adj Close'].sort_index()
    sr.name = 'nav'
    return sr


def load_stock_OHLC_us(symbol, data):
    df = data.minor_xs(symbol)
    df_selected = df.loc[:, ['Open', 'High', 'Low', 'Close']]
    df.index.name = 'tdate'
    df.columns = ['tc_open', 'tc_high', 'tc_low', 'tc_close']
    return df.sort_index()


def generate_df_for_garch(nav, timing):
    inc = np.log(1+nav.pct_change()).fillna(0)*100
    inc2d = inc.rolling(2).sum().fillna(0)
    inc3d = inc.rolling(3).sum().fillna(0)
    inc5d = inc.rolling(5).sum().fillna(0)
    df = pd.DataFrame({'inc2d':inc2d,
                       'inc3d':inc3d,
                       'inc5d':inc5d,
                       'timing':timing})
    return df



def garch(sr_nav, timing, df_vars, modified = False):
    empty = 5
    ddlookback = 5
    maxdd = -0.15
    df = generate_df_for_garch(sr_nav, timing)
    status, empty_days, action = 0, 0, 0
    result_status = {}
    result_pos = {} #结果仓位
    result_act = {} #结果动作

    for day, row in df.iterrows():
        if not (day in df_vars.index):
            pass
        else:
            if modified:
                tmp_vars = df_vars.loc[:day]
                mad = np.abs(tmp_vars - tmp_vars.median()).median()*1.483
                median = tmp_vars.median()
                day_vars = tmp_vars.iloc[-1]
                day_vars[day_vars < (median - 3*mad)] = (median - 3*mad)
                tmp_nav = sr_nav.loc[:day]
                local_max = tmp_nav.iloc[-ddlookback:].max()
                local_drawdown = (sr_nav[day] - local_max)/local_max
                if local_drawdown < maxdd*0.75:
                    status, empty_days, position, action = 2, 0, 0, 6
            else:
                day_vars = df_vars.loc[day]
            #  tmp_nav = sr_nav.loc[:day]
            #  local_max = tmp_nav.iloc[-ddlookback:].max()
            #  local_drawdown = (sr_nav[day] - local_max)/local_max
            #  # if status != 2:
            #  if local_drawdown < maxdd*0.75:
                #  status, empty_days, position, action = 2, 0, 0, 6
            if row['inc2d'] < day_vars['var_2d']:
                status, empty_days, position, action = 2, 0, 0, 2
            elif row['inc3d'] < day_vars['var_3d']:
                status, empty_days, position, action = 2, 0, 0, 3
            elif row['inc5d'] < day_vars['var_5d']:
                status, empty_days, position, action = 2, 0, 0, 5

        if status == 0:
            #不在风控中
            status, position, action = 0, 1, 0
        else:
            #风控中
            if empty_days >= empty:
                #择时满仓
                if row['timing'] == 1.0:
                    status, position, action = 0, 1, 8
                else:
                    #空仓等待择时
                    empty_days += 1
                    if status == 2:
                        status, position, action = 2, 0, 7
                    # else:
                    #     status, position, action = 1, self.ratio, 7
            else:
                empty_days += 1
        result_status[day] = status
        result_act[day] = action
        result_pos[day] = position

    df_result = pd.DataFrame({'rm_pos': result_pos, 'rm_action': result_act, 'rm_status': result_status})
    df_result.index.name = 'rm_date'
    return df_result
    #  return df_result


def original_riskctrl(nav, timing):
    risk_mgr = RiskMgrSimple()
    return risk_mgr.perform('0', pd.DataFrame({'nav':nav, 'timing':timing}))



def calc_nav(nav, pos, name):
    df_position = pos.to_frame(name)
    df_position = DFUtil.filter_same_with_last(df_position)

    # 加载基金收益率
    min_date = df_position.index.min()
    #  max_date = (datetime.now() - timedelta(days=1)) # yesterday
    max_date = pos.index[-1]

    df_inc = nav.pct_change().fillna(0.0).to_frame(name)

    # 计算复合资产净值
    df_nav_portfolio = DFUtil.portfolio_nav(df_inc, df_position, result_col='portfolio')

    df_result = df_nav_portfolio[['portfolio']].rename(columns={'portfolio':'rm_nav'}).copy()
    df_result.index.name = 'rm_date'
    df_result = df_result.reset_index().set_index(['rm_date'])
    return df_result.squeeze()


def calc_drawdown(sr_nav):
    max_nav = sr_nav[0]
    drawdown = []
    for day in sr_nav:
        if max_nav < day:
            max_nav = day
        drawdown.append((day - max_nav)/max_nav)
    return pd.Series(drawdown)


def find_local_max(nav, i, period):
    nav = pd.Series(nav)
    tmp_nav = nav.loc[:i]
    return (tmp_nav.iloc[-period:]).max()

def calc_drawdown_from_local_max(nav, pos):
    riskmgr_interval_constructor = pos.rolling(2,1).apply(lambda x: 1 if x[-1]!=x[0] else 0)
    riskmgr_interval_constructor = riskmgr_interval_constructor[riskmgr_interval_constructor == 1]
    intervals = zip(riskmgr_interval_constructor.index[::2], riskmgr_interval_constructor.index[1::2])
    drawdown_from_local_max = []
    for start, end in intervals:
        localmax = find_local_max(nav, start, period=100)
        drawdown = (nav.loc[start]-localmax)/localmax
        drawdown_from_local_max.append(drawdown)
    return pd.Series(drawdown_from_local_max)


def calc_winrate(nav, pos):
    riskmgr_interval_constructor = pos.rolling(2,1).apply(lambda x: 1 if x[-1]!=x[0] else 0)
    riskmgr_interval_constructor = riskmgr_interval_constructor[riskmgr_interval_constructor == 1]
    intervals = zip(riskmgr_interval_constructor.index[::2], riskmgr_interval_constructor.index[1::2])
    inc = np.log(1+nav.pct_change()).fillna(0)
    summary = np.array([inc.loc[slice(*i)].sum() for i in intervals])
    return (summary.size, summary[summary<0].size)

def calc_sharpe_ratio(nav):
    inc = nav.pct_change().fillna(0)
    return (inc.mean() - 0.03/252)/inc.std()

def stat(df):
    pd.set_option('display.max_columns', 100)
    pd.set_option('display.width', 1000)
    results = pd.DataFrame(columns=['origin', 'old_risk_ctrl', 'new_risk_ctrl', 'modified_new_risk_ctrl'])
    old_total, old_win = calc_winrate(df.nav, df.pos_old)
    new_total, new_win = calc_winrate(df.nav, df.pos_new)
    modified_total, modified_win = calc_winrate(df.nav, df.pos_new_modified)
    results.loc['Triggered'] = [np.nan, old_total, new_total, modified_total]
    results.loc['Winned'] = [np.nan, old_win, new_win, modified_win]
    results.loc['Winrate'] = [np.nan, float(old_win)/old_total, float(new_win)/new_total, float(modified_win)/modified_total]
    #  results.loc['Return'] = [df.nav[-1]-1, df.nav_old[-1]-1, df.nav_new[-1]-1]
    results.loc['Return'] = map(lambda sr: sr.iloc[-1]-1, [df.nav, df.nav_old, df.nav_new, df.nav_new_modified])
    results.loc['Avg. Drawdown'] = map(lambda x: calc_drawdown(x).mean(), [df.nav, df.nav_old, df.nav_new, df.nav_new_modified])
    results.loc['Max. Drawdown'] = map(lambda x: calc_drawdown(x).min(), [df.nav, df.nav_old, df.nav_new, df.nav_new_modified])
    results.loc['Avg. dd fr. locm'] = [np.nan, calc_drawdown_from_local_max(df.nav, df.pos_old).mean(), calc_drawdown_from_local_max(df.nav, df.pos_new).mean(), calc_drawdown_from_local_max(df.nav, df.pos_new_modified).mean()]
    results.loc['Sharpe Ratio'] = map(calc_sharpe_ratio, [df.nav, df.nav_old, df.nav_new, df.nav_new_modified])
    return results


#  def run_stock(secode):
    #  closeaf = load_stock(secode)
    #  timing = gftd(secode)
    #  # Calc VaRs by GARCH
    #  VaRs = RiskMgrVaRs.RiskMgrVaRs()
    #  df_vars = VaRs.perform(closeaf.copy())
    #  df_vars.index.name = 'rm_date'
    #  df_vars.to_csv(os.path.join('result/vars', secode+'.csv'))
    #  # Construct df for risk mgr
    #  df = generate_df_for_garch(closeaf, timing)
    #  df_result = garch(df, df_vars)
    #  df_result.to_csv(os.path.join('result/signals', secode+'.csv'))
    #  sr_nav = nav(closeaf.copy(), df_result.rm_pos, secode)
    #  df_nav = pd.DataFrame({'pos':df_result.rm_pos, 'closeaf':closeaf/closeaf[1], 'result':sr_nav})
    #  df_nav.index.name='rm_date'
    #  df_nav.to_csv(os.path.join('result/navs', secode+'.csv'))
    #  calc_winrate(df_nav)


#  def run_us_stock(symbol, stock_data):
    #  closeaf = load_stock_us(symbol, stock_data)
    #  # substitute gftd because of the difference of loading OHLC sequence
    #  df_ohlc = load_stock_OHLC_us(symbol, stock_data)
    #  df_gftd = TimingGFTD(4,4,4,4).timing(df_ohlc)
    #  timing = df_gftd.tc_timing
    #  # END
    #  VaRs = RiskMgrVaRs.RiskMgrVaRs()
    #  df_vars = VaRs.perform(closeaf)
    #  df_vars.index.name = 'rm_date'
    #  df_vars.to_csv(os.path.join('result/vars', symbol+'.csv'))
    #  df = generate_df_for_garch(closeaf, timing)
    #  df_result = garch(df, df_vars)
    #  df_result.to_csv(os.path.join('result/signals', symbol+'.csv'))
    #  sr_nav = nav(closeaf, df_result.rm_pos, symbol)
    #  df_nav = pd.DataFrame({'pos':df_result.rm_pos, 'closeaf':closeaf/closeaf[1], 'result':sr_nav})
    #  df_nav.index.name='rm_date'
    #  df_nav.to_csv(os.path.join('result/navs', symbol+'.csv'))
    #  calc_winrate(df_nav)

def run_index(index_id):
    tdate = base_trade_dates.load_trade_dates(index_id)
    sr_nav = database.load_nav_series(index_id)
    df_ohlc = load_index_ohlc(index_id, reindex=tdate, begin_date=None, end_date=None, mask=[0,2])
    df_ohlc = df_ohlc.sort_index()
    df_gftd = TimingGFTD(4,4,4,4).timing(df_ohlc)
    timing = df_gftd.tc_timing
    VaRs = RiskMgrVaRs.RiskMgrVaRs()
    df_vars = VaRs.perform(sr_nav)
    df_result_new = garch(sr_nav, timing, df_vars)
    df_result_new_modified = garch(sr_nav, timing, df_vars, modified=True)
    df_result_old = original_riskctrl(sr_nav, timing)
    df_result_old.rm_pos.iloc[:600] = 1
    sr_nav_new = calc_nav(sr_nav, df_result_new.rm_pos, index_id)
    sr_nav_new_modified = calc_nav(sr_nav, df_result_new_modified.rm_pos, index_id)
    sr_nav_old = calc_nav(sr_nav, df_result_old.rm_pos, index_id)
    sr_nav = sr_nav/sr_nav[0]
    df_nav = pd.DataFrame({'pos_new':df_result_new.rm_pos,
                           'pos_new_modified': df_result_new_modified.rm_pos,
                           'pos_old':df_result_old.rm_pos,
                           'nav':sr_nav,
                           'nav_new':sr_nav_new,
                           'nav_new_modified':sr_nav_new_modified,
                           'nav_old':sr_nav_old})
    df_result = pd.concat([df_nav, df_vars], axis=1)
    df_result.to_csv(os.path.join('result/index', index_id+'.csv'))
    result = stat(df_nav)
    print result
    result.to_csv(os.path.join('result/index', index_id+'_stat.csv'))



def run_montecarlo(n):
    symbol = str(n)
    seed_pool = ['120000001', '120000002', '120000013', '120000015']
    sr_nav = pd.Series(gen_simulation(3000, random.choice(seed_pool)))
    timing = dma(sr_nav)
    VaRs = RiskMgrVaRs.RiskMgrVaRs()
    df_vars = VaRs.perform(sr_nav)
    df_result_new = garch(sr_nav, timing, df_vars)
    df_result_new_modified = garch(sr_nav, timing, df_vars, modified=True)
    df_result_old = original_riskctrl(sr_nav, timing)
    df_result_old.rm_pos.iloc[:600] = 1
    sr_nav_new = calc_nav(sr_nav, df_result_new.rm_pos, symbol)
    sr_nav_new_modified = calc_nav(sr_nav, df_result_new_modified.rm_pos, symbol)
    sr_nav_old = calc_nav(sr_nav, df_result_old.rm_pos, symbol)
    sr_nav = sr_nav/sr_nav[0]
    df_nav = pd.DataFrame({'pos_new':df_result_new.rm_pos,
                           'pos_new_modified': df_result_new_modified.rm_pos,
                           'pos_old':df_result_old.rm_pos,
                           'nav':sr_nav,
                           'nav_new':sr_nav_new,
                           'nav_new_modified':sr_nav_new_modified,
                           'nav_old':sr_nav_old})
    df_result = pd.concat([df_nav, df_vars], axis=1)
    df_result.to_csv(os.path.join('result/montecarlo', symbol+'.csv'))
    result = stat(df_nav)
    print result
    result.to_csv(os.path.join('result/montecarlo', symbol+'_stat.csv'))
    #  stat(df_nav)


def stock_test(n):
    stocks = load_vaild_stocks()
    secodes = [random.choice(stocks) for i in range(n)]
    stocks_info = load_stock_info()
    for i in secodes:
        print "======================================="
        print "Working on : %s" % i
        print "Stock: %s" % stocks_info.loc[i].sk_name
        run_stock(i)
        print "======================================="

def index_test():
    db = database.connection(


def monte_carlo_test(n):
    for i in range(n):
        print "======================================="
        print "Working on : %d" % i
        #  run_montecarlo(i)
        run_montecarlo(i)
        print "======================================="

def summary(n, path):
    dfdict = {}
    for i in range(n):
        dfdict[i] = pd.read_csv(os.path.join(path, '%d_stat.csv' % i), index_col=0).T
    panel = pd.Panel(dfdict)
    print "old_risk_ctrl summary:"
    print panel.major_xs('old_risk_ctrl').T.describe().loc['mean']
    print "new_risk_ctrl summary:"
    print panel.major_xs('new_risk_ctrl').T.describe().loc['mean']
    return panel

if __name__ == "__main__":
    monte_carlo_test(20)
    #  stock_test(21)

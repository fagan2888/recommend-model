# coding=utf-8

import sys
import click
sys.path.append('shell')
import logging
import pandas as pd
import numpy as np
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker
from ipdb import set_trace

import config
from db import database, asset_trade_dates, base_ra_index_nav
from db.asset_fundamental import *
from calendar import monthrange
from datetime import datetime, timedelta
from ipdb import set_trace
from asset import Asset
from trade_date import ATradeDate
from hmmlearn.hmm import GaussianHMM
import matplotlib.pyplot as plt
import scipy.stats

logger = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.pass_context
def ms(ctx):
    '''
    macro timing
    '''
    if ctx.invoked_subcommand is None:
        ctx.invoke(ma)
    else:
        pass


@ms.command()
@click.pass_context
def ma(ctx):

    #ids = ['120000001', '120000002', '120000013', '120000014', '120000015', '120000010']
    #ids = ['120000001','120000010', '120000029']
    ids = ['120000001']
    nav = {}
    for _id in ids:
        nav[_id] = Asset(_id).nav(reindex = ATradeDate.trade_date())

    nav = pd.DataFrame(nav)
    nav = nav.dropna()
    nav = nav / nav.iloc[0]
    nav = nav[nav.index <= '2018-05-15']
    pct = nav.pct_change().fillna(0.0)
    #print(nav.rolling(20).mean())
    long_short_pct = (nav.rolling(20).mean() / nav.rolling(120).mean() - 1).dropna()
    nav = nav.loc[long_short_pct.index]
    ob = long_short_pct
    #print(ob);
    '''
    dates = []
    all_states = []
    for i in range(1000, len(ob)):
        now_date = ob.index[i]
        _tmp_ob = ob.iloc[:i,]
        model = GaussianHMM(n_components=5, covariance_type="diag", n_iter=10000).fit(_tmp_ob)
        predicts = model.predict(_tmp_ob)
        means_ = model.means_.reshape(1,5)[0]
        state_dict = dict(zip(range(0, len(means_)), scipy.stats.rankdata(means_)))
        #print(predicts, means_, np.argsort(means_),state_dict)
        states = []
        for predict in predicts:
            states.append(state_dict[predict])
        #print(states)
        print(now_date, states[-1])
        dates.append(now_date)
        all_states.append(states[-1])

        #print(now_date, means_[predicts[-1]])
        #print(np.argsort(means_))
        #print(predicts)
    #print(model.covars_)
    #print(model.transmat_)

    states_df = pd.DataFrame(all_states, index = dates, columns = ['state'])
    states_df.index.name = 'date'
    states_df.to_csv('states_hs300')
    #df_states = pd.DataFrame(model.predict(ob), index = nav.index)
    #print(df_states.tail(40))
    '''

    #model = GaussianHMM(n_components=5, covariance_type="diag", n_iter=10000).fit(ob)
    #states = model.predict(ob)
    #states_number = list(set(model.predict(ob)))
    #states_number.sort()

    #colors = {0:'red', 1:'blue', 2:'green', 3:'yellow', 4:'black'}
    #for state in states_number:
    #    plt.plot(nav.index[states == state], nav[states == state], '.', color = colors[state], ms = 3)

    #plt.show()
    #print(df_states.tail(200))
    states_df = pd.read_csv('states_hs300', index_col = ['date'], parse_dates = True)
    print(states_df.tail())
    states_number = list(set(states_df.state))
    states_number.sort()

    for state in states_number:
        dates = states_df.index[states_df.state == state]
        print(state, pct.loc[dates].mean())

    '''
    states = states_df.state.ravel() - 1
    states_number = list(set(states))
    states_number.sort()
    nav = nav.loc[states_df.index]

    colors = {0:'red', 1:'blue', 2:'green', 3:'yellow', 4:'black'}
    for state in states_number:
        plt.plot(nav.index[states == state], nav[states == state], '.', color = colors[state], ms = 5)

    plt.show()
    '''

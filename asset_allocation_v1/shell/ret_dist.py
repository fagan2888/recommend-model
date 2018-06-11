#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import sys
sys.path.append('shell')
from db import *
import numpy as np
import pandas as pd
from datetime import datetime
from ipdb import set_trace
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.style.use('seaborn')
myfont = matplotlib.font_manager.FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc', size=30)


def load_nav():

    nav = pd.read_csv('ts_holding_nav.csv', index_col = ['ts_date', 'ts_uid'], parse_dates = ['ts_date'])
    nav = nav.loc[:, ['ts_inc']]
    nav = nav.unstack()
    nav.columns = nav.columns.droplevel(0)
    # nav = nav.groupby(nav.index.strftime('%Y-%m')).apply(lambda x: np.prod(1+x)-1, skipna=False)
    nav = nav.groupby(nav.index.strftime('%Y-%m')).apply(lambda x: pd.Series((1+x.values).prod(0)-1, index = nav.columns))
    set_trace()

    return nav


def load_online():

    df = pd.read_csv('data/on_online_nav.csv', index_col = ['on_date'], parse_dates = ['on_date'])
    df = df.loc['2016-08':, ['on_online_id', 'on_inc']]
    df = df.reset_index()
    df = df.set_index(['on_date', 'on_online_id'])
    df = df.unstack()
    df.columns = df.columns.droplevel(0)
    df = df.groupby(df.index.strftime('%Y-%m')).apply(lambda x: (1+x).prod() - 1)

    return df


def load_base():

    high = base_ra_index_nav.load_series('120000016').pct_change().loc['2016-08':]
    low = base_ra_index_nav.load_series('120000010').pct_change().loc['2016-08':]
    df_base = pd.DataFrame(index = high.index)

    for risk in range(1, 11):
        df_base['risk%d'%risk] = (10-risk)/9.0*low + (risk-1)/9.0*high

    df_base = df_base.groupby(df_base.index.strftime('%Y-%m')).apply(lambda x: (1+x).prod() - 1)

    return df_base

def ret_distrbute():

    nav = pd.read_csv('ret_month.csv', index_col = 0, parse_dates = True)
    nav = nav.stack().reset_index()
    nav.columns = ['date', 'uid', 'ret']
    nav = nav.set_index('date')
    nav.index = nav.index.strftime('%Y-%m')
    risk = pd.read_csv('risk.csv', index_col = ['month'])

    df_online = load_online()
    df_base = load_base()

    dates = pd.date_range('2016-08-01', '2018-06-01', freq = 'M')
    # dates = pd.date_range('2018-05-01', '2018-06-01', freq = 'M')
    '''
    for date in dates:
        for rl in np.arange(0.1, 1.1, 0.1):
            month = date.strftime('%Y-%m')
            try:
                uids = risk[risk.risk == rl].loc[month].uid.values
                uids = map(str, uids)
                tmp_nav = nav[nav.uid.isin(uids)].loc[month]
                tmp_nav = tmp_nav.replace(0.0, np.nan)
                tmp_nav = tmp_nav.dropna()
            except:
                continue
            tmp_nav.columns = ['用户ID','本月收益率']
            tmp_nav.to_csv('data/user_ret/%s-风险%d.csv'%(month, int(rl*10)), index_label = '日期', encoding = 'gbk')
    '''
    for date in dates:
        for rl in np.arange(0.1, 1.1, 0.1):
            month = date.strftime('%Y-%m')
            try:
                uids = risk[risk.risk == rl].loc[month].uid.values
                uids = map(str, uids)
                tmp_nav = nav[nav.uid.isin(uids)].loc[month]
                tmp_nav = tmp_nav.replace(0.0, np.nan)
                tmp_nav = tmp_nav.dropna()
            except:
                continue

            online_ret = df_online[800000+(int(rl*10)%10)].loc[month]
            base_ret = df_base['risk%d'%int(rl*10)].loc[month]

            left = round(tmp_nav.ret.min(), 3)
            right = round(tmp_nav.ret.max(), 3)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            # ax.hist(tmp_nav.ret, normed=1, alpha = 0.75, rwidth = 0.8, label = '标杆组合收益: %1.4f\n比较基准收益: %1.4f'%(online_ret, base_ret))
            ax.hist(tmp_nav.ret,bins = 20, normed=1, alpha = 0.75, rwidth = 0.8, label = 'model return: %1.4f\n threshold return: %1.4f'%(online_ret, base_ret))
            plt.xticks(np.arange(left, right, 0.002), rotation = 70)
            plt.legend(fontsize = 15)
            plt.savefig('data/user_ret_png/%s-风险%d.png'%(month, int(rl*10)))



def ret_sta():

    nav = pd.read_csv('ret_month.csv', index_col = 0, parse_dates = True)
    nav = nav.stack().reset_index()
    nav.columns = ['date', 'uid', 'ret']
    nav = nav.set_index('date')
    nav.index = nav.index.strftime('%Y-%m')
    risk = pd.read_csv('risk.csv', index_col = ['month'])

    df_online = load_online()
    df_base = load_base()

    dates = pd.date_range('2016-08-01', '2018-06-01', freq = 'M')
    # dates = pd.date_range('2018-05-01', '2018-06-01', freq = 'M')
    # for rl in np.arange(0.1, 1.1, 0.1):
    for rl in [0.3]:

        online_ret = df_online[800000+(int(rl*10)%10)]
        base_ret = df_base['risk%d'%int(rl*10)]
        online_ret = online_ret.to_frame('online_ret')
        base_ret = base_ret.to_frame('base_ret')
        online_ret = online_ret.loc[:'2018-05']
        base_ret = base_ret.loc[:'2018-05']

        user_ret = pd.DataFrame(columns = ['user_ret', 'user_std'])
        for date in dates:
            try:
                month = date.strftime('%Y-%m')
                uids = risk[risk.risk == rl].loc[month].uid.values
                uids = map(str, uids)
                tmp_nav = nav[nav.uid.isin(uids)].loc[month]
                tmp_nav = tmp_nav.replace(0.0, np.nan)
                tmp_nav_ = tmp_nav.dropna().ret.values.mean()
                tmp_std_ = tmp_nav.dropna().ret.values.std()
            except:
                tmp_nav_ = np.nan
                tmp_std_ = np.nan

            user_ret.loc[month] = [tmp_nav_, tmp_std_]

        df_ret = pd.concat([base_ret, online_ret, user_ret], 1)
        set_trace()

        df_ret.columns = ['标杆收益','模型收益','用户收益均值','用户收益标准差']
        df_ret.to_csv('data/user_sta/风险%d模型、标杆及用户收益.csv'%(int(rl*10)), index_label = '日期', encoding = 'gbk')












if __name__ == '__main__':

    # load_nav()
    # load_online()
    # load_base()

    # ret_distrbute()
    ret_sta()

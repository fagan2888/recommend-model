#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import sys
import click
sys.path.append('shell')
import logging
import pandas as pd
import numpy as np
from sqlalchemy import *
from sqlalchemy.orm import sessionmaker

import config
from db import asset_on_online_nav
from db import database, asset_trade_dates, base_ra_index_nav
from db.asset_fundamental import *
from calendar import monthrange
from datetime import datetime, timedelta
from ipdb import set_trace

from asset import Asset

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from scipy.stats import rankdata, spearmanr
import matplotlib.pyplot as plt
from xgboostextension import XGBRanker
logger = logging.getLogger(__name__)
import warnings
warnings.filterwarnings('ignore')


@click.group(invoke_without_command=True)
@click.pass_context
def asv(ctx):
    '''
    A Share View
    '''
    if ctx.invoked_subcommand is None:
        ctx.invoke(macro_view_update)
    else:
        pass


@asv.command()
@click.option('--start-date', 'startdate', default='2012-07-27', help='start date to calc')
@click.option('--end-date', 'enddate', default=datetime.today().strftime('%Y-%m-%d'), help='start date to calc')
@click.option('--viewid', 'viewid', default='BL.000001', help='macro timing view id')
@click.option('--index', 'idx', default='sz', help='macro timing view id')
@click.pass_context
def macro_view_update(ctx, startdate, enddate, viewid, idx):

    union_mv_df = pd.DataFrame(union_mv, columns = ['globalid', 'bl_date', 'bl_view', 'created_at', 'updated_at'])

    for index_id in ['120000001', '120000002', '120000053', '120000056', '120000058', '120000073', 'MZ.FA0010', 'MZ.FA0050', 'MZ.FA0070', 'MZ.FA1010', 'ALayer']:

        df_new = union_mv_df
        df_new['bl_index_id'] = index_id
        df_new = df_new.set_index(['globalid','bl_date','bl_index_id'])

        db = database.connection('asset')
        metadata = MetaData(bind=db)
        t = Table('ra_bl_view', metadata, autoload = True)
        columns = [
            t.c.globalid,
            t.c.bl_date,
            t.c.bl_view,
            t.c.bl_index_id,
            t.c.created_at,
            t.c.updated_at,
        ]
        s = select(columns).where(t.c.globalid == viewid).where(t.c.bl_index_id == index_id)
        df_old = pd.read_sql(s, db, index_col = ['globalid', 'bl_date', 'bl_index_id'], parse_dates = ['bl_date'])
        database.batch(db, t, df_new, df_old, timestamp = False)

        print(df_new.tail())


class AShareView:

    def __init__(self, start_date, data_path):

        self.start_date = start_date
        self.data_path = data_path
        self.spilt_year = '2012'
        self.forcast_month = 1

    def preprocess(self):

        # original data
        data = pd.read_excel(data_path)
        data = data.iloc[1:-2]
        data.columns = ['date', 'gdp', 'house', 'bond', 'm2', 'cpi', 'sf']
        data = data.set_index('date')
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
        data = data.shift(1)
        data = data.fillna(method='pad')
        data = data[data.index > self.start_date]
        data['house'] = data['house'].pct_change(12)*100
        data['sf'] = data['sf'].pct_change(12)*100
        data = data.dropna()

        # cal momentum
        snavd = base_ra_index_nav.load_series('120000016')
        snavm = snavd.resample('m').last()
        sretm = snavm.pct_change()*100
        sretm = sretm.to_frame('sret')

        data = pd.merge(data, sretm, left_index=True, right_index=True, how='left')
        sret = snavm.pct_change(self.forcast_month)*100
        sret = sret.shift(-self.forcast_month)
        sret = sret.loc[data.index]
        sret = np.sign(sret)

        # train test split
        self.sretm = sretm
        self.train_x = data[data.index < self.spilt_year]
        self.test_x = data[data.index >= self.spilt_year]
        self.train_y = sret[sret.index < self.spilt_year]
        self.test_y = sret[sret.index >= self.spilt_year]

    def train(self, x, y):

        # model = LogisticRegression()
        model = GradientBoostingClassifier(random_state=1)
        # model = GradientBoostingClassifier()
        model.fit(x, y)

        return model

    def predict(self, model, x):

        return model.predict(x)

    def cal_indicator(self, model, x, y):

        train_pre = model.predict(x)
        total = 0.0
        correct = 0.0
        for y_pre, y in zip(train_pre, y.values):

            if not np.isnan(y):
                total += 1.0
                if y == y_pre:
                    correct += 1.0

        print("Correct Ratio: ", correct / total, "correct: ", correct, "total", total)

    def timing(self, model, x):

        y_pre = model.predict(x)
        sretm = self.sretm / 100
        sretm = sretm.loc[x.index]
        sretm['view'] = y_pre
        sretm['view'] = sretm['view'].shift(1)
        sretm['view'] = (sretm['view'] + 1) / 2
        sretm = sretm.fillna(0.0)
        sretm['nav'] = (sretm['sret'] + 1).cumprod()
        sretm['tret'] = sretm['sret']*sretm['view']
        sretm['tnav'] = (sretm['tret'] + 1).cumprod()
        sretm = sretm.loc[:, ['nav', 'tnav']]
        sretm.plot()
        plt.show()

    def threshhold(self):

        total = 0.0
        correct = 0.0
        for y_pre, y in zip(self.test_y.values[:-1], self.test_y.values[1:]):
            total += 1.0
            if y_pre == y:
                correct += 1.0

        print("Threshold Correct Ratio: ", correct / total, "correct: ", correct, "total", total)

    def handle(self):

        self.preprocess()
        model = self.train(self.train_x, self.train_y)
        self.threshhold()
        self.cal_indicator(model, self.train_x, self.train_y)
        self.cal_indicator(model, self.test_x, self.test_y)
        self.timing(model, self.train_x)
        self.timing(model, self.test_x)


class AssetRankView:

    def __init__(self, start_date, data_path):

        self.start_date = start_date
        self.data_path = data_path
        self.split_year = '2012'
        self.forcast_month = 1
        self.asset_ids = ['120000016', '120000013', '120000015', '120000010', '120000039', '120000080', '120000028']

    def preprocess(self):

        # original data
        data = pd.read_excel(data_path)
        data = data.iloc[1:-2]
        data.columns = ['date', 'gdp', 'house', 'bond', 'm2', 'cpi', 'sf']
        data = data.set_index('date')
        data.index = pd.to_datetime(data.index, format='%Y-%m-%d')
        data = data.shift(1)
        data = data.fillna(method='pad')
        data = data[data.index > self.start_date]
        data['house'] = data['house'].pct_change(12)*100
        data['sf'] = data['sf'].pct_change(12)*100
        data = data.dropna()
        data['group'] = range(len(data))

        df_nav = self.load_asset_data()
        df_nav = df_nav.resample('m').last()
        df_ret = df_nav.pct_change()*100
        df_ret = df_ret.reindex(data.index)
        df_ret = df_ret.fillna(method='bfill')
        arr_rank = np.apply_along_axis(rankdata, 1, df_ret.values)
        df_rank = pd.DataFrame(data=arr_rank, columns=df_ret.columns, index=df_ret.index)
        df_rank = df_rank.shift(-1)

        df_feature = pd.DataFrame(columns=np.append(data.columns, 'ret'))
        df_label = pd.Series()
        for asset_id in self.asset_ids:
            asset_ret = df_ret[[asset_id]]
            asset_ret.columns = ['ret']
            tmp_feature = pd.concat([data, asset_ret], 1)
            tmp_label = df_rank[asset_id]
            df_feature = pd.concat([df_feature, tmp_feature])
            df_label = pd.concat([df_label, tmp_label])
        df_feature = df_feature[['group', 'gdp', 'house', 'bond', 'm2', 'cpi', 'sf', 'ret']]

        self.train_x = df_feature[df_feature.index < self.split_year]
        self.test_x = df_feature[df_feature.index >= self.split_year]
        self.train_y = df_label[df_label.index < self.split_year]
        self.test_y = df_label[df_label.index >= self.split_year]
        self.train_ret = df_ret[df_ret.index < self.split_year]
        self.test_ret = df_ret[df_ret.index >= self.split_year]

    def load_asset_data(self):

        df_asset_nav = {}
        for asset_id in self.asset_ids:
            df_asset_nav[asset_id] = Asset.load_nav_series(asset_id)
        df_asset_nav = pd.DataFrame(df_asset_nav)

        return df_asset_nav

    def train(self, x, y):

        ranker = XGBRanker(max_depth=4, learning_rate=0.1, n_estimators=150, subsample=1.0, random_state=0)
        # ranker = XGBRanker(max_depth=7, learning_rate=0.01, n_estimators=5000, subsample=1.0, random_state=0)
        ranker.fit(x, y, eval_metric=['ndcg', 'map@5-'])

        return ranker

    def predict(self, model, x):

        dates = x.index.unique()
        y_pre = model.predict(x.values)
        y_pre = y_pre.reshape(len(dates), -1)
        df_pre = pd.DataFrame(data=y_pre, columns=self.asset_ids, index=dates)
        arr_rank = np.apply_along_axis(rankdata, 1, df_pre.values)
        df_pre_rank = pd.DataFrame(data=arr_rank, columns=df_pre.columns, index=df_pre.index)

        return df_pre_rank

    def cal_indicator(self, pre_rank, real_rank):

        dates = pre_rank.index
        real_rank_values = real_rank.values.reshape(-1, len(pre_rank)).T
        real_rank = pd.DataFrame(data=real_rank_values, columns=pre_rank.columns, index=dates)
        real_rank = real_rank.fillna(0.0)
        df_rankcorr = pd.Series(index=dates)
        for i in range(len(pre_rank)):
            tmp_pr = pre_rank.iloc[i].values
            tmp_rr = real_rank.iloc[i].values
            df_rankcorr.loc[dates[i]] = spearmanr(tmp_pr, tmp_rr)[0]
        print("Rank corr:", df_rankcorr.mean())

        rank = real_rank[pre_rank > 6].mean(1).mean()
        print("Rank percentile:", (rank-1)/6)

    def timing(self, df_pre_rank, df_ret):

        df_ret = df_ret.shift(-1)
        df_ret = df_ret.fillna(0.0)
        df_ret = df_ret / 100.0
        df_pos = df_pre_rank.copy()
        df_pos = 100**(df_pos)
        df_pos = df_pos.div(df_pos.sum(1), 0)

        timing_ret = df_ret * df_pos
        timing_ret = timing_ret.sum(1)
        timing_nav = (1 + timing_ret).cumprod()
        timing_nav = timing_nav.to_frame('tnav')
        df_nav = (1 + df_ret).cumprod()
        df_nav = pd.concat([df_nav, timing_nav], 1)

        df_nav.plot()
        plt.show()

    def handle(self):

        self.preprocess()
        model = self.train(self.train_x, self.train_y)

        train_rank = self.predict(model, self.train_x)
        self.cal_indicator(train_rank, self.train_y)
        self.timing(train_rank, self.train_ret)

        # test_rank = self.predict(model, self.test_x)
        # self.cal_indicator(test_rank, self.test_y)
        # self.timing(test_rank, self.test_ret)

        # test_rank.to_csv('test_rank.csv', index_label='date')


if __name__ == '__main__':

    data_path = '/home/yaojiahui/recommend_model6/recommend_model/asset_allocation_v2/data/ori_macro_data.xls'
    # asv = AShareView('2002-01-01', data_path)
    # asv.handle()

    arv = AssetRankView('2002-01-01', data_path)
    arv.handle()

    # df1 = asset_on_online_nav.load_series('800001', 8)
    # df1 = df1.to_frame('risk1')
    # df2 = pd.read_csv('currency.csv', index_col=['date'], parse_dates=['date'])
    # df = pd.merge(df2, df1, left_index=True, right_index=True, how='left')
    # df = df / df.iloc[0]


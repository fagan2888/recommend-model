from scipy.ndimage.filters import gaussian_filter
from sklearn.linear_model import LinearRegression
from datetime import datetime
from scipy.signal import hilbert
from numpy import mat
from decimal import Decimal
import sys
import pandas as pd
import numpy as np

from db import caihui_tq_qt_index as load_index

class Music(object):

    def __init__(self):
        self.stock_id = [
                '120000013',
                '120000015',
                '120000016',
                #'120000019',
                #'120000023',
                '120000024',
                ]
        self.bond_id = [
                '120000009',
                '120000027',
                ]
        self.commodity_id = [
                '120000028',
                '120000029',
                ]
        self.cash_id = ['120000039']
        self.asset_id = {
                '120000001': '2070000060',
                '120000002': '2070000187',
                '120000009': '2070006886',
                '120000013': '2070006545',
                '120000015': '2070000076',
                '120000016': '2070000005',
                '120000019': '2070009557',
                '120000023': '2070009556',
                '120000024': '2070008974',
                '120000027': '2070006678',
                '120000028': '2070006521',
                '120000029': '2070006789',
                '120000039': '2070006913',
                }
        self.start_date = '19910201'
        self.end_date = None
        self.test_start = '20050930'
        #self.data = pd.read_csv('music_month.csv', index_col = 0, parse_dates = True)
        #self.assets = self.data.columns
        self.cycle = np.array([42, 100, 200])

    def load_process_data(self, asset_class):
        class_id = {
                'stock': self.stock_id,
                'bond': self.bond_id,
                'commodity': self.commodity_id,
                }
        asset_data = {}
        for idx in class_id.get(asset_class):
            secode = self.asset_id[idx]
            tmp_data = load_index.load_index_daily_data(secode, \
                    start_date = self.start_date)
            tmp_data = tmp_data.groupby(tmp_data.index.strftime('%Y-%m')).last()
            if not asset_data:
                index = tmp_data.index
            asset_data[idx] = tmp_data['close']
        asset_data = pd.DataFrame(asset_data, index = index)
        asset_data.to_csv('tmp/%s_oridata.csv'%asset_class, index_label = 'date')
        asset_data = asset_data.rolling(12).apply(lambda x: np.log(x[-1]/x[0]))
        asset_data = asset_data.replace([np.inf, -np.inf], np.nan)
        asset_data = asset_data.dropna()
        if asset_class == 'bond':
            bond_fill = pd.read_csv('tmp/bond_fill.csv', index_col = 0, \
                    parse_dates = True)
            bond_fill.index = bond_fill.index.strftime('%Y-%m')
            for bond_id in self.bond_id:
                bond_fill[bond_id] = bond_fill['close']
            del bond_fill['close']
            asset_data = pd.concat([bond_fill[:142], asset_data])
        #asset_data.to_csv('tmp/%s_oridata.csv'%asset_class, index_label = 'date')

    def cal_equal_weight_index(self):
        class_id = {
                'stock': self.stock_id,
                'bond': self.bond_id,
                'commodity': self.commodity_id,
                }
        equal_weight_index = {}
        index = []
        for asset_id in class_id:
            asset_data = {}
            for idx in class_id.get(asset_id):
                secode = self.asset_id[idx]
                tmp_data = load_index.load_index_daily_data(secode, \
                        start_date = self.test_start)
                #print tmp_data.head()
                tmp_data = tmp_data.groupby(tmp_data.index.strftime('%Y-%m')).last()
                if len(index) == 0:
                    index = tmp_data.index
                asset_data[idx] = tmp_data['close']
            asset_data = pd.DataFrame(asset_data, index = index)
            asset_data = asset_data.pct_change()
            asset_data = asset_data.fillna(0) + 1
            asset_data = asset_data.sum(1)/asset_data.shape[1]
            asset_data = np.cumprod(asset_data)
            equal_weight_index[asset_id] = asset_data
        equal_weight_index = pd.DataFrame(equal_weight_index, index = index, \
                columns = ['stock', 'bond', 'commodity'])
        equal_weight_index['cash'] = 1
        #equal_weight_index.to_csv('tmp/equal_weight_index.csv')
        #print equal_weight_index

    def cal_stock_nav(self):
        index = []
        asset_data = {}
        stock_list =['120000001','120000002','120000013','120000015']
        for idx in stock_list:
            secode = self.asset_id[idx]
            tmp_data = load_index.load_index_daily_data(secode, \
                    start_date = self.test_start)
            tmp_data = tmp_data.groupby(tmp_data.index.strftime('%Y-%m')).last()
            if len(index) == 0:
                index = tmp_data.index
            asset_data[idx] = tmp_data['close']
        asset_data = pd.DataFrame(asset_data, index=index, columns = stock_list)
        asset_data = asset_data.pct_change()
        asset_data = asset_data.fillna(0) + 1
        asset_data = np.cumprod(asset_data)
        asset_data.columns = ['sh300', 'zz500', 'sp500', 'hsi']
        asset_data['cash'] = 1
        #asset_data.to_csv('tmp/stock_nav.csv')
        #print asset_data

    @staticmethod
    def sumple(X, maxiter = 100):
        X = X.astype(Decimal)
        ncor, n = X.shape
        X = hilbert(X ,axis = 0)
        X = mat(X)
        x = X[:ncor, :]
        i = 1
        weight = mat(np.ones(n).reshape(1,-1))
        weight = weight.astype(Decimal)
        while i <= maxiter:
            i += 1
            #weight = np.mean(np.conj(x.dot(weight.conj().T).dot(np.ones((1,n))) - \
            #        x.dot(np.diag(weight[0])))*x,0).reshape(1,-1)
            #weight = (np.sqrt(n/(weight.dot(weight.conj().T))))*weight
            weight = np.mean(np.multiply(np.conj(x*weight.conj().T*np.ones((1,n)) - \
                    x*np.diag(np.array(weight)[0])), x),0)
            w = weight*weight.conj().T
            w_real = np.real(w[0,0])
            weight = np.sqrt(n/w_real)*weight
            print weight.dtype
        s = (X.dot(weight.conj().T))/np.sum(np.abs(weight))
        r = np.real(s)
        return r

    def cal_yoy_seq(self):
        self.data = self.data.rolling(12).apply(lambda x: np.log(x[-1]/x[0]))
        self.data = 100*self.data
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.data = self.data.dropna()

    def cal_cycle(self):
        columns = self.data.columns
        for cycle in self.cycle:
            for column in columns:
                filter_data = gaussian_filter(self.data[column], cycle/6)
                self.data['%s-%d'%(column, cycle)] = filter_data

    def training(self):
        for asset in self.assets:
            column = self.data.columns
            used_column = [item for item in column if item.startswith(asset)]
            x = self.data.loc[:, used_column[1:]]
            y = self.data.loc[:, used_column[0]]
            lr = LinearRegression()
            lr.fit(x, y)
            #pre = lr.predict(x)
            print asset, lr.score(x,y), lr.coef_

    def handle(self):
        self.cal_yoy_seq()
        self.cal_cycle()
        self.training()

if __name__ == '__main__':
    music = Music()
    #music.cal_equal_weight_index()
    #music.cal_stock_nav()
    df = pd.read_csv('tmp/equal_weight_index.csv',  index_col = 0, parse_dates = True)
    #df = df.iloc[:len(df)-3, [0,2,3]]
    #df['cash'] = 1
    df_v = df.values
    df_sumple = music.sumple(df_v)

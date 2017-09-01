# -*- coding: utf-8 -*-
from scipy.ndimage.filters import gaussian_filter
from sklearn.linear_model import LinearRegression
#from datetime import datetime
from scipy.signal import hilbert
from numpy import mat
#from utils import cal_nav_maxdrawdown
#from decimal import Decimal
#import sys
import pandas as pd
import numpy as np

from db import caihui_tq_qt_index as load_index

class Cycle(object):

    def __init__(self):
        self.asset_secode = {
                '120000001': '2070000060',#沪深300
                '120000002': '2070000187',#中证500
                '120000009': '2070006886',#中证全债
                '120000013': '2070006545',#标普400
                #'120000014': '2070000626',#A股黄金
                '120000014': '2070006523',#暂时用标普黄金代替A股黄金
                '120000015': '2070000076',#恒生指数
                '120000016': '2070000005',
                '120000019': '2070009557',
                '120000023': '2070009556',
                '120000024': '2070008974',
                '120000027': '2070006678',
                '120000028': '2070006521',
                '120000029': '2070006789',
                '120000039': '2070006913',
                }
        self.asset_id = ['120000001','120000002','120000013','120000014','120000015']
        self.asset_name = ['sh300', 'zz500', 'sp500', 'au', 'hsi']
        self.asset_num = len(self.asset_id)
        self.start_date = '19910130'
        self.end_date = None
        self.cycle = np.array([42, 100, 200])
        self.window = 125

    def load_indic(self):
        self.stock_indic = pd.read_csv('tmp/stock_indic.csv', index_col = 0, parse_dates = True)
        self.bond_indic = pd.read_csv('tmp/bond_indic.csv', index_col = 0, parse_dates = True)
        self.bond_indic_fill = pd.read_csv('tmp/bond_indic_fill.csv', index_col = 0, parse_dates = True)
        self.commodity_indic = pd.read_csv('tmp/commodity_indic.csv', index_col = 0, parse_dates = True)

        for column in self.bond_indic.columns:
            self.bond_indic_fill[column] = self.bond_indic_fill['bondfill']
        del self.bond_indic_fill['bondfill']
        fill_len = len(self.stock_indic) - len(self.bond_indic)
        bondfill = self.bond_indic_fill[:fill_len]
        self.bond_indic = pd.concat([bondfill, self.bond_indic])
        #print self.stock_indic

    def cal_cycle_indic(self, cycle = 42, start_date = None, end_date = None):
        stock_indic = self.stock_indic[start_date:end_date].copy()
        bond_indic = self.bond_indic[start_date:end_date].copy()
        commodity_indic = self.commodity_indic[start_date:end_date].copy()
        fill_zeros = 300
        for i, indic in enumerate([stock_indic, bond_indic, commodity_indic]):
            indic_fill = pd.DataFrame(np.zeros((fill_zeros, indic.shape[1])), \
                    columns = indic.columns, index = range(len(stock_indic), \
                    len(stock_indic) + fill_zeros))
            if i == 0:
                stock_indic = pd.concat([indic, indic_fill])
            elif i == 1:
                bond_indic = pd.concat([indic, indic_fill])
            elif i == 2:
                commodity_indic = pd.concat([indic, indic_fill])

        window = self.window
        stock_filtered = pd.DataFrame()
        for column in stock_indic.columns:
            tmp_filtered = gaussian_filter(stock_indic[column], cycle/6)
            stock_filtered[column] = tmp_filtered

        bond_filtered = pd.DataFrame()
        for column in bond_indic.columns:
            tmp_filtered = gaussian_filter(bond_indic[column], cycle/6)
            bond_filtered[column] = tmp_filtered

        commodity_filtered = pd.DataFrame()
        for column in commodity_indic.columns:
            tmp_filtered = gaussian_filter(commodity_indic[column], cycle/6)
            commodity_filtered[column] = tmp_filtered

        stock_sumpled = self.sumple(stock_filtered)
        bond_sumpled = self.sumple(bond_filtered)
        commodity_sumpled = self.sumple(commodity_filtered)
        asset_sumpled = np.column_stack([stock_sumpled, bond_sumpled, commodity_sumpled])
        cycle_indic = self.sumple(asset_sumpled)[:window+1]
        return cycle_indic
        #print stock_sumpled.shape

    def cal_asset_nav(self):
        index = []
        asset_data = {}
        asset_yoy = {}
        stock_list = self.asset_id
        #读取上证指数来补充沪深300和中证500
        szzz = load_index.load_index_daily_data('2070000005', start_date = \
                self.start_date)
        szzz = szzz.groupby(szzz.index.strftime('%Y-%m')).last()
        szzz_yoy = szzz.rolling(13).apply(lambda x: (x[-1]/x[0]-1)).dropna()
        szzz_pct_chg = szzz.pct_change()
        for idx in stock_list:
            secode = self.asset_secode[idx]
            tmp_data = load_index.load_index_daily_data(secode, \
                    start_date = self.start_date)
            tmp_data = tmp_data.groupby(tmp_data.index.strftime('%Y-%m')).last()
            tmp_pct_chg = tmp_data.pct_change().dropna()
            tmp_yoy = tmp_data.rolling(13).apply(lambda x: (x[-1]/x[0]-1)).dropna()
            if idx in ['120000001', '120000002']:
                fill_num_p = len(szzz_pct_chg) - len(tmp_pct_chg)
                fill_num_y = len(szzz_yoy) - len(tmp_yoy)
                tmp_pct_chg = pd.concat([szzz_pct_chg[:fill_num_p], tmp_pct_chg])
                tmp_yoy = pd.concat([szzz_yoy[:fill_num_y], tmp_yoy])
                tmp_pct_chg = tmp_pct_chg[12:]
                tmp_asset_data = np.cumprod(1 + tmp_pct_chg.values)
            elif idx == '120000009':
                pct_chg_fill = pd.DataFrame()
                pct_chg_fill['close'] = [0]*308
                bond_indic_fill = pd.read_csv('tmp/bond_indic_fill.csv', \
                        index_col = 0, parse_dates = True)
                bond_indic_fill.columns = ['close']
                fill_num_p = len(self.bond_indic) - len(tmp_pct_chg)
                fill_num_y = len(self.bond_indic) - len(tmp_yoy)
                tmp_pct_chg = pd.concat([pct_chg_fill[:fill_num_p], tmp_pct_chg])
                tmp_yoy = pd.concat([bond_indic_fill[:fill_num_y], tmp_yoy])
                tmp_asset_data = np.cumprod(1 + tmp_pct_chg.values)
            else:
                tmp_pct_chg = tmp_pct_chg[11:]
                tmp_asset_data = np.cumprod(1 + tmp_pct_chg.values)
            if len(index) == 0:
                index = tmp_yoy.index
            asset_data[idx] = tmp_asset_data
            asset_yoy[idx] = tmp_yoy.values.flat[:]

        asset_data = pd.DataFrame(asset_data, index=index, columns = stock_list)
        asset_yoy = pd.DataFrame(asset_yoy, index=index, columns = stock_list)
        asset_data['cash'] = 1

        self.asset_nav = asset_data
        self.asset_yoy = asset_yoy
        self.asset_nav.index = self.bond_indic.index
        self.asset_yoy.index = self.bond_indic.index
        self.asset_nav.to_csv('tmp/asset_nav.csv', index_label = 'date')

    @staticmethod
    def sumple(X, maxiter = 100):
        ncor, n = X.shape
        X = hilbert(X ,axis = 0)
        X = mat(X)
        x = X[:ncor, :]
        i = 1
        weight = mat(np.ones((1,n)))
        while i <= maxiter:
            i += 1
            #weight = np.mean(np.conj(x.dot(weight.conj().T).dot(np.ones((1,n))) - \
            #        x.dot(np.diag(weight[0])))*x,0).reshape(1,-1)
            #weight = (np.sqrt(n/(weight.dot(weight.conj().T))))*weight
            weight = np.mean(np.multiply((x*weight.conj().T*np.ones((1,n)) - \
                    x*np.diag(np.array(weight.conj())[0])).conj(), x),0)
            w = weight*weight.conj().T
            w_real = np.real(w[0,0])
            weight = np.sqrt(n/w_real)*weight
        s = (X.dot(weight.conj().T))/np.sum(np.abs(weight))
        r = np.real(s)
        r = np.array(r).flat[:]
        return r

    def training(self):
        window = self.window
        asset_num = len(self.asset_id)
        fit_value = []
        pre_value = []
        #print self.asset_yoy
        #print self.stock_indic
        for i in range(len(self.asset_yoy) - window + 1):
            x1 = self.cal_cycle_indic(42, i, i+window)
            x2 = self.cal_cycle_indic(100, i, i+window)
            x3 = self.cal_cycle_indic(200, i, i+window)
            x = np.column_stack([x1[:-1], x2[:-1], x3[:-1]])
            pre_x = np.column_stack([x1, x2, x3])
            for j in range(asset_num):
                y = self.asset_yoy.iloc[i:i+window, j]
                lr = LinearRegression()
                lr.fit(x,y)
                y_fit = lr.predict(pre_x)[-1]
                y_pre = y_fit - lr.predict(x)[-1]
                fit_value.append(y_fit)
                pre_value.append(y_pre)
                #print 'y: ', y.values[-1], '', 'y_fit: ', y_fit
        self.asset_nav = self.asset_nav[window-1:]

        pre_value = np.array(pre_value).reshape(-1, asset_num)
        for idx, asset in enumerate(self.asset_name):
            self.asset_nav['%s_pre'%asset] = pre_value[:, idx]
        self.asset_nav['cash_pre'] = 0.0000000001

        fit_value = np.array(fit_value).reshape(-1, asset_num)
        for idx, asset in enumerate(self.asset_name):
            self.asset_nav['%s_fit'%asset] = fit_value[:, idx]
        self.asset_nav['cash_fit'] = 1
        self.asset_nav.to_csv('tmp/asset_nav_fit.csv', index_label = 'date')

    def investing(self):
        #cal rank
        asset_num = len(self.asset_id) + 1
        asset_value = self.asset_nav.iloc[:, :asset_num]
        asset_pre = self.asset_nav.iloc[:, asset_num: asset_num*2]

        asset_pre_pos = asset_pre.mask(asset_pre < 0, 0)
        #asset_weight = asset_pre_pos.apply(lambda x: x/sum(x), 1)
        asset_weight = asset_pre_pos.apply(lambda x: 1 + np.sign(x-max(x)), 1)
        print asset_weight

        #cal nav
        asset_pct_chg = asset_value.pct_change()
        asset_pct_chg = asset_pct_chg.dropna()
        invest_nav = asset_pct_chg.values*asset_weight[:-1].values
        #print asset_pct_chg.head()
        #print asset_weight.head()
        invest_nav = (invest_nav.sum(1) + 1).cumprod()
        invest_nav = np.append(1, invest_nav)
        self.asset_nav = self.asset_nav
        self.asset_nav['invest_nav'] = invest_nav

        #print correct
        #print total
        #print self.wr

        #concat nav and weight
        weight_column = []
        for asset in self.asset_name:
            weight_column.append('%s_weight'%asset)
        weight_column.append('cash_weight')
        asset_weight.columns = weight_column
        self.asset_nav = pd.concat([self.asset_nav, asset_weight], 1)

        rank_column = []
        for asset in self.asset_name:
            rank_column.append('%s_rank'%asset)
        rank_column.append('cash_rank')

        for i in range(5):
            self.asset_nav.iloc[:, i] = self.asset_nav.iloc[:, i]/self.asset_nav.iloc[0, i]
        print self.asset_nav['invest_nav'][-1]

    def evaluating(self):
        self.asset_nav = pd.read_csv('tmp/cycle_model_result.csv', index_col = 0,\
                parse_dates = True)
        sh300_nav = self.asset_nav['sh300'][-1]/self.asset_nav['sh300'][0]
        zz500_nav = self.asset_nav['zz500'][-1]/self.asset_nav['zz500'][0]
        sp500_nav = self.asset_nav['sp500'][-1]/self.asset_nav['sp500'][0]
        hsi_nav = self.asset_nav['hsi'][-1]/self.asset_nav['hsi'][0]
        invest_nav = self.asset_nav['invest_nav'][-1]/self.asset_nav['invest_nav'][0]
        maxdd = self.cal_maxdd(self.asset_nav['invest_nav'])
        print 'sh300_nav :', sh300_nav
        print 'zz500_nav :', zz500_nav
        print 'sp500_nav :', sp500_nav
        print 'hsi_nav :', hsi_nav
        print 'invest_nav :', invest_nav
        print 'invest_maxdd:', maxdd

    @staticmethod
    def rank(arr):
        return np.argsort(np.argsort(arr))

    @staticmethod
    def cal_maxdd(arr):
        maxdd = 0
        for i,j in enumerate(arr):
            tmp_dd = (j - max(arr[:i+1]))/(max(arr[:i+1]))
            if tmp_dd < maxdd:
                maxdd = tmp_dd
        return maxdd

    def handle(self):
        self.load_indic()

        self.cal_asset_nav()
        #self.asset_nav = pd.read_csv('tmp/asset_nav.csv', index_col = 0, \
        #        parse_dates = True)

        self.training()
        print 'train over'
        #self.asset_nav = pd.read_csv('tmp/asset_nav_fit.csv', index_col = 0, \
        #        parse_dates = True)

        self.investing()
        #print self.asset_nav
        self.asset_nav.to_csv('tmp/cycle_model_result.csv', index_label = 'date')

    def plot(self):
        self.load_indic()
        cycle_42 = self.cal_cycle_indic(42)
        cycle_100 = self.cal_cycle_indic(100)
        cycle_200 = self.cal_cycle_indic(200)
        cycle_df = pd.DataFrame()
        index = self.stock_indic.index
        cycle_df['cycle_42'] = cycle_42
        cycle_df['cycle_100'] = cycle_100
        cycle_df['cycle_200'] = cycle_200
        cycle_df.index = index
        #print cycle_df
        cycle_df.to_csv('tmp/cycle_df.csv')

    def cal_view(self):
        self.handle()
        self.asset_nav = pd.read_csv('tmp/cycle_model_result.csv', index_col = 0, \
                parse_dates = True)
        print self.asset_nav.tail()
        ori_view = pd.read_csv('data/view_frame.csv', index_col = 0, \
                parse_dates = True)
        new_view = {}
        asset = ori_view.columns
        for columns in asset:
            new_view[columns] = []
        for idx in ori_view.index:
            idx = idx.strftime('%Y-%m')
            for i,j in enumerate(np.arange(19, 24)):
                new_view[asset[i]].append(self.asset_nav.ix[idx, j].values[0])
        new_view = pd.DataFrame(new_view, columns = asset, index = ori_view.index)
        new_view = new_view.mask(new_view > 0, 1)
        print new_view.tail()

        new_view['idx'] = new_view.index
        tmp_view = new_view.groupby(new_view.index.strftime('%Y-%m')).last()
        tmp_view.index = tmp_view.idx
        del tmp_view['idx']
        del new_view['idx']
        new_view = pd.DataFrame(np.zeros(new_view.shape), columns = ori_view.columns, \
                index = new_view.index)
        new_view = new_view + tmp_view
        new_view = new_view.fillna(method = 'ffill')
        new_view = new_view.fillna(0)
        new_view = new_view.astype('int')
        new_view.to_csv('data/view.csv', index_label = 'date')

if __name__ == '__main__':
    cycle  = Cycle()
    #cycle.handle()
    #print cycle.wr

    #cycle.evaluating()
    #cycle.plot()

    cycle.cal_view()

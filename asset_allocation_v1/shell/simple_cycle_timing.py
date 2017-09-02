# -*- coding: utf-8 -*-
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
from numpy import cos, pi
import itertools
from datetime import datetime
from db import caihui_tq_qt_index as load_index

class Simple_cycle(object):

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
        self.start_date = '19930130'
        self.end_date = None
        self.train_num = 241
        self.cycle = np.array([42, 100, 200])

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
        self.asset_nav.to_csv('tmp/asset_nav.csv', index_label = 'date')

    def training(self):
        t1, t2, t3 = self.cycle
        #train_yoy = self.asset_yoy[:self.train_num]
        train_yoy = self.asset_yoy.iloc[:-61]
        #test_yoy = self.asset_yoy[self.train_num:]
        train_sh300 = train_yoy.iloc[:, 4].values
        #test_sh300 = test_yoy.iloc[:, 0].values
        phase = np.arange(-pi/2, pi/2, pi/15)
        result_df = pd.DataFrame()
        count = 0
        i_list = []
        j_list = []
        k_list = []
        model_list = []
        score_list = []
        for i, j, k in itertools.product(phase,phase,phase):
            count += 1
            print count
            x_42 = cos(2*pi/t1*np.arange(len(train_sh300)) + i)
            x_100 = cos(2*pi/t2*np.arange(len(train_sh300)) + j)
            x_200 = cos(2*pi/t3*np.arange(len(train_sh300)) + k)
            x = np.column_stack([x_42, x_100, x_200, range(len(train_sh300))])
            y = train_sh300
            lr = LinearRegression()
            lr.fit(x, y)
            score = lr.score(x, y)
            i_list.append(i)
            j_list.append(j)
            k_list.append(k)
            model_list.append(lr)
            score_list.append(score)
        result_df['phase_42'] = i_list
        result_df['phase_100'] = j_list
        result_df['phase_200'] = k_list
        result_df['model'] = model_list
        result_df['score'] = score_list
        result_df.to_csv('tmp/st_result.csv')
        best_result = result_df.loc[np.argmax(result_df['score'])]
        i = best_result['phase_42']
        j = best_result['phase_100']
        k = best_result['phase_200']
        model = best_result['model']
        pre_len = 30
        x_42 = cos(2*pi/t1*np.arange(len(self.asset_yoy)+pre_len) + i)
        x_100 = cos(2*pi/t2*np.arange(len(x_42)) + j)
        x_200 = cos(2*pi/t3*np.arange(len(x_42)) + k)
        x = np.column_stack([x_42, x_100, x_200, range(len(x_42))])

        y_fit = model.predict(x)
        yoy = np.append(self.asset_yoy.iloc[:, 4].values, np.zeros(pre_len))
        score = best_result['score']
        print score

        index = pd.date_range(datetime(1994, 2, 28), periods = 283+pre_len, freq = 'm')
        cycle_df = pd.DataFrame({'cycle_42': x_42,'cycle_100': x_100,\
                'cycle_200': x_200, 'yoy': yoy, 'y_fit':y_fit}, columns =\
                ['cycle_42','cycle_100', 'cycle_200', 'yoy', 'y_fit'], \
                 index = index)
        cycle_df.to_csv('/home/ipython/yaojiahui/cycle_df_hsi.csv')

    def handle(self):
        self.cal_asset_nav()
        self.training()

if __name__ == '__main__':
    st = Simple_cycle()
    st.handle()

# -*- coding: utf-8 -*-
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
from numpy import cos, pi
import numpy.fft as fft
import itertools
import time
import calendar
from datetime import datetime
from db import caihui_tq_qt_index as load_index

class Simple_cycle(object):

    def __init__(self):
        self.asset_secode = {
                '120000001': '2070000060',#沪深300
                '120000002': '2070000187',#中证500
                '120000009': '2070006886',#中证全债
                '120000013': '2070006545',#标普500
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
        self.retrace_pnts = dict(zip(self.asset_id, [0.4, 0.4, 0.3, 0.3, 0.45]))
        self.asset_num = len(self.asset_id)
        self.start_date = '19930130'
        self.end_date = None
        self.window = 84
        #self.cycle = np.array([42, 100, 200])
        #self.cycle = np.array([23, 34.5, 69])

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

        new_index = []
        for idx in index:
            y,m = time.strptime(idx, '%Y-%m')[:2]
            d = calendar.monthrange(y,m)[1]
            new_index.append(datetime(y, m, d))

        asset_data = pd.DataFrame(asset_data, index=new_index, columns = stock_list)
        asset_yoy = pd.DataFrame(asset_yoy, index=new_index, columns = stock_list)
        asset_data['cash'] = 1

        self.asset_nav = asset_data
        self.asset_yoy = asset_yoy
        #print self.asset_yoy
        self.asset_nav.to_csv('tmp/asset_nav.csv', index_label = 'date')

    def cal_asset_prob(self, start, end, asset_column):
        yoy = self.asset_yoy.iloc[: end, asset_column].values

        tp_x, tp_y = self.eff(yoy, self.retrace_pnts[self.asset_id[asset_column]])
        if tp_y[1] < tp_y[0]:
            tp_y[::2] = 1
            tp_y[1::2] = -1
            tp_y[-1] = 0
        else:
            tp_y[::2] = -1
            tp_y[1::2] = 1
            tp_y[-1] = 0
        x = np.arange(len(yoy))
        interp = interp1d(tp_x, tp_y)
        tp = interp(x)
        tp = tp[start-end:]
        return tp

    @staticmethod
    def eff(a, RETRACE_PNTS):
        xs = []
        ys = []
        pivots = []
        up = True
        p1, p2 = 0, 0
        p1_bn, p2_bn = 0, 0

        for x, y in enumerate(a):
            if up:
                if y > p1:
                    p2_bn = p1_bn = x
                    p2 = p1 = y
                elif y < p2:
                    p2_bn = x
                    p2 = y
            else:
                if y < p1:
                    p2_bn = p1_bn = x
                    p2 = p1 = y
                elif y > p2:
                    p2_bn = x
                    p2 = y

            # Found new pivot
            if abs(p1 - p2) >= RETRACE_PNTS * 1:
                pivots.append((p1_bn, p1))
                up = not up
                p1_bn, p1 = p2_bn, p2
        for x, y in pivots:
            xs.append(x)
            ys.append(y)
        xs.append(len(a) - 1)
        ys.append(0)
        if xs[0] != 0:
            xs = [0] + xs
            ys = [0] + ys
        xs = np.array(xs)
        ys = np.array(ys)

        return xs, ys


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

    def training_prob(self, start, end, asset_column):
        t1, t2, t3 = self.cal_cycle(start, end, asset_column)
        asset_prob = self.cal_asset_prob(start, end, asset_column)
        asset_yoy = self.asset_yoy[start:end]
        phase = np.arange(-pi/2, pi/2, pi/6)
        result_df = pd.DataFrame()
        i_list = []
        j_list = []
        k_list = []
        model_list = []
        score_list = []
        y = asset_prob
        for i, j, k in itertools.product(phase,phase,phase):
            x_42 = cos(2*pi/t1*np.arange(len(asset_prob)) + i)
            x_100 = cos(2*pi/t2*np.arange(len(asset_prob)) + j)
            x_200 = cos(2*pi/t3*np.arange(len(asset_prob)) + k)
            x = np.column_stack([x_42, x_100, x_200])
            lr = LinearRegression()
            try:
                lr.fit(x, y)
            except Exception, e:
                print e
                x = np.nan_to_num(x)
                y = np.nan_to_num(y)
                lr.fit(x,y)
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
        best_result = result_df.loc[np.argmax(result_df['score'])]
        i = best_result['phase_42']
        j = best_result['phase_100']
        k = best_result['phase_200']
        model = best_result['model']
        pre_len = 30
        x_42 = cos(2*pi/t1*np.arange(len(asset_prob)+pre_len) + i)
        x_100 = cos(2*pi/t2*np.arange(len(x_42)) + j)
        x_200 = cos(2*pi/t3*np.arange(len(x_42)) + k)
        x = np.column_stack([x_42, x_100, x_200])

        y_fit = model.predict(x)
        pre_change = np.diff(y_fit[-pre_len-1:])
        pre_prob = y_fit[-pre_len:]
        pre_date = asset_yoy.index[-1]

        return pre_date, pre_change, pre_prob

        '''
        prob = np.append(asset_prob.iloc[:, asset_column].values, np.zeros(pre_len))
        yoy = np.append(asset_yoy.iloc[:, asset_column].values, np.zeros(pre_len))
        #score = best_result['score']
        #print score

        start_date = asset_prob.index[0]
        y,m,d = time.strptime(start_date, '%Y-%m')[:3]
        index = pd.date_range(datetime(y, m, d), periods = \
                len(asset_prob)+pre_len, freq = 'm')
        cycle_df = pd.DataFrame({'cycle_42': x_42,'cycle_100': x_100,\
                'cycle_200': x_200, 'prob': prob, 'yoy': yoy, 'pre_prob':y_fit}, \
                columns = ['cycle_42','cycle_100', 'cycle_200', 'yoy', 'prob', 'pre_prob'], \
                 index = index)
        cycle_df.to_csv('/home/ipython/yaojiahui/cycle_prob_%s.csv'%\
                self.asset_name[asset_column])
                '''

    def cal_cycle(self, start, end, asset_column):
        asset_yoy = self.asset_yoy[start:end]
        wave = asset_yoy.iloc[:, asset_column].values
        spectrum = fft.fft(wave)
        freq = fft.fftfreq(len(wave))
        order = np.argsort(abs(spectrum)[:spectrum.size/2])[::-1]
        order = order[order != 0]
        t1 = 1/freq[order[0]]
        t2 = 1/freq[order[1]]
        t3 = 1/freq[order[2]]
        print t1, t2, t3
        return t1, t2, t3

    def handle(self):
        self.cal_asset_nav()
        #self.cal_asset_prob()
        window = self.window
        #0:sh300, 1:zz500, 2:sp500, 3:au, 4:hsi
        asset_column = 3
        dates = []
        pre_changes = []
        pre_probs = []
        for i in range(len(self.asset_yoy) - window + 1):
            pre_date, pre_change, pre_prob = \
                    self.training_prob(i, i+window, asset_column)
            print pre_date
            dates.append(pre_date)
            pre_changes.append(pre_change[0])
            pre_probs.append(pre_prob[0])
        nav = self.asset_nav.loc[dates].iloc[:, asset_column].values.tolist()
        yoy = self.asset_yoy.loc[dates].iloc[:, asset_column].values.tolist()

        nav.extend([0]*29)
        yoy.extend([0]*29)
        pre_probs.extend(pre_prob[1:])
        pre_changes.extend(pre_change[1:])
        fill_dates = pd.date_range(dates[-1], periods = 30, freq = 'm')
        dates.extend(fill_dates[1:])

        self.result_df = pd.DataFrame({'pre_changes':pre_changes, 'nav':nav, \
                'yoy':yoy, 'pre_probs':pre_probs}, columns = \
                ['nav', 'yoy', 'pre_changes', 'pre_probs'], index = dates)
        self.result_df.to_csv('/home/ipython/yaojiahui/cycle_%s_result.csv'%\
                self.asset_name[asset_column], index_label = 'date')

    def invest(self):
        #self.result_df = pd.read_csv('/home/ipython/yaojiahui/cycle_zz500_result.csv',\
        #        index_col = 0, parse_dates = True)
        result_df = self.result_df.iloc[:-29]
        invest_nav = []
        for i in range(len(result_df)-1):
            if result_df['pre_changes'][i] > 0:
                invest_nav.append(result_df['nav'][i+1]/result_df['nav'][i])
        nav = np.product(invest_nav)
        print 'asset nav: ', result_df['nav'][-1]/result_df['nav'][0]
        print 'invest nav: ', nav

    @staticmethod
    def cal_view():
        asset_view_1 = pd.read_csv('/home/ipython/yaojiahui/cycle_sh300_result.csv',\
                index_col = 0, parse_dates = True).pre_changes
        asset_view_2 = pd.read_csv('/home/ipython/yaojiahui/cycle_zz500_result.csv',\
                index_col = 0, parse_dates = True).pre_changes
        asset_view_3 = pd.read_csv('/home/ipython/yaojiahui/cycle_sp500_result.csv',\
                index_col = 0, parse_dates = True).pre_changes
        asset_view_4 = pd.read_csv('/home/ipython/yaojiahui/cycle_au_result.csv',\
                index_col = 0, parse_dates = True).pre_changes
        asset_view_5 = pd.read_csv('/home/ipython/yaojiahui/cycle_hsi_result.csv',\
                index_col = 0, parse_dates = True).pre_changes

        asset_view = pd.DataFrame(np.column_stack([asset_view_1,asset_view_2,\
            asset_view_3,asset_view_4,asset_view_5,]), asset_view_1.index, \
            ['sh300', 'zz500', 'sp500', 'au', 'hsi'])
        asset_view = np.sign(asset_view)
        ori_view = pd.read_csv('data/view_frame.csv', index_col = 0, \
                parse_dates = True)
        new_view = {}
        asset = ori_view.columns
        for columns in asset:
            new_view[columns] = []
        for idx in ori_view.index:
            idx = idx.strftime('%Y-%m')
            for i,j in enumerate(np.arange(5)):
                new_view[asset[i]].append(asset_view.ix[idx, j].values[0])
        new_view = pd.DataFrame(new_view, columns = asset, index = ori_view.index)

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
        #new_view['AU9999.SGE'] = -1
        new_view.to_csv('data/view.csv', index_label = 'date')



if __name__ == '__main__':
    st = Simple_cycle()
    #st.handle()
    #st.invest()
    st.cal_view()

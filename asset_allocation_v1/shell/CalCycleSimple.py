#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import pandas as pd
import numpy as np
import numpy.fft as fft
import itertools

from sklearn.linear_model import LinearRegression
from ipdb import set_trace
from db import asset_fl_nav


class Cycle(object):

    def __init__(self, end_date = '2018-01-01'):
        self.end_date = end_date
        self._pool = Cycle.get_pool()
        self.asset_cycle = {}
        # self.cycle = [180, 360, 720]
        self.cycle = [6, 12, 24]


    @staticmethod
    def get_pool():
        levels = {1:[1], 2:[1,2], 3:range(1, 8)}
        pool = []
        for level1, level1_pool in levels.iteritems():
            for level2 in level1_pool:
                pool.append("FL.00{:02d}{:02d}".format(level1, level2)) 
        
        return pool
    

    @staticmethod
    def cal_no_trend_nav(nav):
        lr = LinearRegression()
        X = np.arange(len(nav)).reshape(-1,1)
        res = lr.fit(X, nav)
        trend = res.predict(X)
        no_trend_nav = nav - trend

        return no_trend_nav


    @staticmethod
    def cal_cycle(id_, end_date):
        nav = asset_fl_nav.load_series(id_, end_date = '2018-01-01')
        wave = nav.values

        no_trend_wave = Cycle.cal_no_trend_nav(wave)
        spectrum = fft.fft(no_trend_wave)
        freq = fft.fftfreq(len(no_trend_wave))
        order = np.argsort(abs(spectrum)[:spectrum.size/2])[::-1]
        main_freq = 1/freq[order[:5]]

        nav = nav.to_frame(name = id_)
        nav['no_trend'] = no_trend_wave
        nav.to_csv('copula/cycle/no_trend_{}.csv'.format(id_), index_label = 'date')
        
        return main_freq


    def cal_pool_cycle(self, end_date):
        for asset in self._pool:
            print asset, self.cal_cycle(asset, end_date)


    def fit_cycle(self, id_, end_date):
        nav = asset_fl_nav.load_series(id_, end_date = '2018-01-01')
        wave = nav.values
        wave = Cycle.cal_no_trend_nav(wave)
        ## 使用月数据
        nav = nav[::-30][::-1]
        wave = wave[::-30][::-1]

        LENGTH = len(wave)
        c1, c2, c3 = self.cycle
        const = np.repeat(1, LENGTH)

        phase = np.arange(-np.pi/2, np.pi/2, np.pi/15)
        best_score = 0
        best_par = []
        best_model = None
        for phase1, phase2, phase3 in itertools.product(phase, phase, phase):

            cycle_series1 = np.sin(2*np.pi*np.arange(LENGTH)/c1 + phase1)
            cycle_series2 = np.sin(2*np.pi*np.arange(LENGTH)/c2 + phase2)
            cycle_series3 = np.sin(2*np.pi*np.arange(LENGTH)/c3 + phase3)

            feas = np.column_stack([cycle_series1, cycle_series2, cycle_series3, const])
            lr = LinearRegression()
            res = lr.fit(feas, wave) 
            score = res.score(feas, wave)
            if score > best_score:
                best_score = score
                best_par = [phase1, phase2, phase3]
                best_model = res

        fit_wave = best_model.predict(feas)

        nav = nav.to_frame(name = id_)
        nav[id_] = wave
        nav['fit_cycle'] = fit_wave
        phase1, phase2, phase3 = best_par
        nav['cycle1'] = np.sin(2*np.pi*np.arange(LENGTH)/c1 + phase1)
        nav['cycle2'] = np.sin(2*np.pi*np.arange(LENGTH)/c2 + phase2)
        nav['cycle3'] = np.sin(2*np.pi*np.arange(LENGTH)/c3 + phase3)

        nav.to_csv('copula/cycle/fit_cycle_{}.csv'.format(id_), index_label = 'date')

        return fit_wave
        

    def fit_pool_cycle(self, end_date):
        for asset in self._pool[3:]:
            self.fit_cycle(asset, end_date)


    def handle(self):
        # self.cal_pool_cycle(self.end_date)
        self.fit_pool_cycle(self.end_date)


if __name__ == '__main__':

    cycle = Cycle()
    cycle.handle()
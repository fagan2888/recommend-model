#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8

import os
import json
import logging
import logging.config
import pandas as pd
import numpy as np
# import numpy.fft as fft
from scipy.fftpack import fft
import itertools

from sklearn.linear_model import LinearRegression
from ipdb import set_trace
from db import asset_fl_nav
logger = logging.getLogger(__name__)

def setup_logging(
    default_path = './shell/logging.json',
    default_level = logging.INFO,
    env_key = 'LOG_CFG'):

    """Setup logging configuration
    """
    path = default_path
    value = os.getenv(env_key, None)
    if value:
        path = value
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

class Cycle(object):

    def __init__(self, end_date = '2018-01-01'):
        self.end_date = end_date
        self._pool = Cycle.get_pool()
        self.asset_cycle = {}
        self.cycle = [180, 360, 720]


    @staticmethod
    def get_pool():
        levels = {1:[1], 2:[1,2], 3:range(1, 8)}
        pool = []
        for level1, level1_pool in levels.iteritems():
            for level2 in level1_pool:
                pool.append("FL.00{:02d}{:02d}".format(level1, level2)) 
        
        return pool
    

    @staticmethod
    def cal_yoy(nav):
        # nav_yoy = nav.pct_change(360).dropna()
        nav_yoy = np.log(nav).diff(360).dropna()

        return nav_yoy.values


    @staticmethod
    def cal_cycle(id_, end_date):
        nav = asset_fl_nav.load_series(id_, begin_date = '2010-01-01', end_date = '2018-01-01')
        y = Cycle.cal_yoy(nav)
        n = len(y)
        t = 1.0/len(y)
        x = np.linspace(0.0, n*t, n)
        yf = fft(y)
        xf = np.linspace(0.0, 1.0/(2.0*t), n//2)
        xt = n/xf[::-1]
        xt = xt[xt < 360]
        xt = np.around(xt, 2)
        A = 2.0/n*np.abs(yf[0:n//2])[::-1]
        A = A[:len(xt)]
        main_cycles = dict((k, v) for k,v in zip(A, xt) if k > 0.03)
        if main_cycles == {}:
            logger.warning("Asset {} doesn't have short cycle, set it to be 360 days".format(id_))
            return 360
        else:
            main_cycle = sum(k*v for k, v in main_cycles.iteritems())/sum(main_cycles.keys())

        ## save freq spectrum figure to csv
        # df = pd.DataFrame(data = A, index = xt, columns = ['freq_spec'])
        # df = df.groupby(df.index.astype('int')).mean()
        # df.to_csv('copula/cycle/{}_freq_spec.csv'.format(id_), index_label = 'cycle')
        
        return main_cycle


    def cal_pool_cycle(self, end_date):
        for asset in self._pool[3:]:
            print asset, self.cal_cycle(asset, end_date)


    def handle(self):
        self.cal_pool_cycle(self.end_date)


if __name__ == '__main__':
    setup_logging()

    cycle = Cycle()
    cycle.handle()
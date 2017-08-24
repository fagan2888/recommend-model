from scipy.ndimage.filters import gaussian_filter
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

class Music(object):

    def __init__(self):
        self.data = pd.read_csv('music_month.csv', index_col = 0, parse_dates = True)
        self.assets = self.data.columns
        self.cycle = np.array([42, 100, 200])

    def cal_yoy_seq(self):
        self.data = self.data.rolling(12).apply(lambda x: np.log(x[-1]/x[0]))
        self.data = 100*self.data
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.data = self.data.dropna()
        self.data = self.data

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
    music.handle()

# -*- coding: UTF-8 -*-
from sklearn import svm
import pandas as pd
import numpy as np

class SvmPredict(object):
    def __init__(self, data_df, close_pct_mask=[1,1,1,1,1,0,0], \
                    volume_pct_mask=[1,1,1,1,1,0,0]):
        self.data_df = data_df
        self.feat_values, self.labels = self.process_data(np.array(close_pct_mask), \
                            np.array(volume_pct_mask))
        self.train_num = 500
        self.train_data = self.feat_values[:self.train_num]
        self.train_label = self.labels[:self.train_num]
    def process_data(self, pct_feat_mask=[1, 1, 1, 1, 1, 1, 1], \
            vol_feat_mask=[1, 1, 1, 1, 1, 1, 1]):
        pct_feat_mask = np.array(pct_feat_mask)
        vol_feat_mask = np.array(vol_feat_mask)
        pct_step = np.array([1, 5, 20, 60, 125, 250, 500])
        close_df = self.data_df['close']
        volume_df = self.data_df['volume']
        used_steps = np.delete(pct_step, np.where(pct_feat_mask==0))
        used_vol_steps = np.delete(pct_step, np.where(vol_feat_mask==0))
        union_data = {}
        for pct_step in used_steps:
            union_data['pct'+str(pct_step)] = \
                        np.array(SvmPredict.pct_chg_step(close_df, pct_step))

        for pct_step in used_vol_steps:
            union_data['vol'+str(pct_step)] = \
                        np.array(SvmPredict.pct_chg_step(volume_df, pct_step))
        labels = np.where(self.data_df['ratio'].shift(-1) >= 0, 1, -1)
        pct_df = pd.DataFrame(union_data, index=self.data_df.index)
        pct_df.dropna(inplace=True)
        labels = labels[-len(pct_df):]
        feat_values = pct_df.values
        return feat_values[:-1], labels[:-1]
    @staticmethod
    def pct_chg_step(df, step):
        return (df.shift(step) - df) / df
if __name__ == "__main__":
    data_df = pd.read_csv("000300_data.csv", index_col=['date'], \
                        parse_dates = ['date'])
    close_pct_mask = [1,1,1,1,0,0,0]
    volume_pct_mask = [1,1,1,1,0,0,0]
    obj = SvmPredict(data_df, close_pct_mask, volume_pct_mask)

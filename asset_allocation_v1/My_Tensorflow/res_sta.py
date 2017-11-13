#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8
import pandas as pd
def res_sta():
    asset = pd.read_csv('data/asset.csv', index_col = 0, parse_dates = True).fillna(0.0)
    w_train = pd.read_csv('result_data/tf_alloc_train.csv', index_col = 0, parse_dates = True)
    w_test = pd.read_csv('result_data/tf_alloc_test.csv', index_col = 0, parse_dates = True)

    asset_train = asset.loc[w_train.index]
    asset_test = asset.loc[w_test.index]

    res_train = pd.concat([asset_train, w_train], 1) 
    res_test = pd.concat([asset_test, w_test], 1) 

    res_train.to_csv('result_data/final_res/res_train.csv', index_label = 'date')
    res_test.to_csv('result_data/final_res/res_test.csv', index_label = 'date')

if __name__ == '__main__':
    res_sta()

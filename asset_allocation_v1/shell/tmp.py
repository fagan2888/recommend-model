#coding=utf8


import sys
sys.path.append('shell')
import LabelAsset


if __name__ == '__main__':
    df = LabelAsset.label_asset_nav('2015-06-30', '2016-10-31')
    print df
    df.to_csv('fund_pool_value.csv')

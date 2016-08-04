#coding=utf8

import sys
sys.path.append('shell')
import pandas as pd
import LabelAsset
import EqualRiskAssetRatio
import EqualRiskAsset
import HighLowRiskAsset
import os

if __name__ == '__main__':
    fpath = sys.argv[1]
    fname = sys.argv[2]
    fsep = sys.argv[3]
    flength = sys.argv[4]
    args = {"fpath":fpath, "fname":fname, "fsep":fsep, "flen":flength}
    #ind = [0, 1, 2]
    #result_df = pd.DataFrame(data=args, index=ind)
    # result_df.to_csv('/home/data/yitao/recommend-model/recommend_model/asset_allocation_offline/shell/tmp.csv')
    print fpath, fname, fsep, flength
    df = pd.read_csv(fpath+fname+'.csv', index_col = 'date', parse_dates = ['date'])
    EqualRiskAssetRatio.equalriskassetratio(df, tmpFileName = fname)
    EqualRiskAsset.equalriskasset(df, tmpFileName = fname)
    HighLowRiskAsset.highlowriskasset(sep = fsep, length = flength, tmpPath = fpath, tmpFileName = fname)

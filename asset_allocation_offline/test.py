#coding=utf8
import pandas as pd
args = {"fpath":"dir/dir", "fname":"name444", "fsep":4, "flen":6}
ind = [0, 1, 2]
result_df = pd.DataFrame(data=args, index=ind)
result_df.to_csv('tmp.csv')


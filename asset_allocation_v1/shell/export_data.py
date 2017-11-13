#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8
from db import base_ra_index_nav
import pandas as pd

df = base_ra_index_nav.load_series('120000016', begin_date = '2006', end_date = '2017-11-10')
print df.tail()
#df_ret = df.pct_change()
#df = pd.concat([df, df_ret], 1)
#df.columns = ['sz_nav', 'sz_ret']
#df.to_csv('data/sz.csv')

df_year = df.resample('A').last()
df_year = df_year.pct_change()
df_year.to_csv('data/sz_year.csv')
print df_year

#coding=utf8


from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix

import pandas as pd
import numpy as np


df = pd.read_csv('./data/index.csv', index_col = 'date', parse_dates= ['date'])

dfr = df.pct_change().fillna(0.0)
dfr = dfr[['000905.SH','000300.SH','399006.SZ']]
#print dfr


#X = sparse_random_matrix(100, 100, density=0.01, random_state=42)
#print X


svd = TruncatedSVD(n_components=1, random_state=31, n_iter=1000, tol=0.0)
data = np.matrix(dfr.values).T
#print data
svd.fit(data)
#print svd.components_
#print
#print svd.transform(data)




#TruncatedSVD(algorithm='randomized', n_components=5, n_iter=5,
#        random_state=42, tol=0.0)
#print svd.get_params(True)
#print(svd.explained_variance_ratio_)
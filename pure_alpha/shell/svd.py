#coding=utf8


from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix


import pandas as pd
import numpy as np
import statsmodels.api as sm


df = pd.read_csv('./data/clean_funds.csv', index_col = 'date', parse_dates= ['date'])


dfr = df.pct_change().fillna(0.0)
#dfr = dfr[['000905.SH','000300.SH','399006.SZ']]


#X = sparse_random_matrix(100, 100, density=0.01, random_state=42)
#print X


svd = TruncatedSVD(n_components=1, random_state=31, n_iter=1000, tol=0.0)
data = np.matrix(dfr.values).T

svd.fit(data)
#print svd.components_
#print
#print dfr.columns
ws = svd.transform(data)
#print ws
sum_w = ws[0][0] + ws[1][0] + ws[2][0]


market_dfr = pd.DataFrame(np.matrix(svd.components_).T, index=dfr.index, columns=['market']) / sum_w
market_dfr.to_csv('./tmp/index_svd.csv')


#print ws[0][0] / sum_w, ws[1][0] / sum_w ,ws[2][0] / sum_w


mr = market_dfr['market'].values


x = mr
x = sm.add_constant(x)


for col in dfr.columns:
	y = dfr[col].values
	result = sm.OLS(y, x).fit()
	#print result.summary()
	print col, result.params[0] , result.params[1]






#TruncatedSVD(algorithm='randomized', n_components=5, n_iter=5,
#        random_state=42, tol=0.0)
#print svd.get_params(True)
#print(svd.explained_variance_ratio_)

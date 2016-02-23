#coding=utf8


import string
import numpy as np
import pandas as pd
import sys
import statsmodels.api as sm
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm
from sklearn.cross_validation import KFold


df = pd.read_csv(sys.argv[1], index_col = 0, header = 0, parse_dates = [0])

colnames = df.columns

#print df.drop([colnames[4], colnames[5], colnames[6], colnames[7]])

#print colnames[4]


df = df.drop([colnames[4], colnames[5], colnames[6], colnames[7], colnames[12]], axis=1)
df = df.dropna(axis=0)

colnames = df.columns

dates = df.index

train = []

for i in range(35, len(dates)):
	tmp_df = df.loc[dates[i-35]:dates[i]]
	record = []
	for col in colnames:
		values = tmp_df[col].values
		_max    = values.max()
		_min    = values.min()
		v      = values[len(values) - 1]
		if _max == _min:
			record.append(0)
		else:
			record.append( (v - _min) * 2 / (_max - _min) - 1)
	train.append(record)


dates = dates[35:len(dates)]


index_df = pd.read_csv(sys.argv[2], index_col = 0, header = 0, parse_dates = [0])
index_df = index_df.loc[ '2009-12-31' : '2015-07-31']



hs300_net =  index_df['hs300'].values
zz500_net =  index_df['zz500'].values




zz500_r = []
for i in range(1, len(zz500_net)):
	zz500_r.append(zz500_net[i] / zz500_net[i-1] - 1)	

hs300_r = []
for i in range(1, len(hs300_net)):
	hs300_r.append(hs300_net[i] / hs300_net[i-1] - 1)	



X = train
y = []


for v in zz500_r:
	if v >= 0:
		y.append(1)
	else:
		y.append(-1)



n_samples = len(X)
n = 0
for i in range((int)(n_samples * 0.6), len(y)):
	print y[i-1]
	if(y[i - 1] == y[i]):
		n = n + 1

print 1.0 * n / (len(y) - (int)(n_samples * 0.6)  - 1)

'''			
for v in zz500_r:
	y.append(v)


for v in hs300_r:
	if v >= 0:
		y.append(1)
	else:
		y.append(-1)
'''


'''
n_samples = len(X)
cv = cross_validation.ShuffleSplit(n_samples, n_iter=5, test_size=0.4, random_state=0)
clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(clf, X, y, cv=cv)
print scores
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
'''


X = np.array(X)
y = np.array(y)

n_samples = len(X)
#kf = KFold(n_samples, n_folds=5)

#for train, test in kf:
	#clf = svm.SVC(kernel='linear', C=1).fit(X[train], y[train])
	#print clf.score(X[test], y[test])


correct = 0
error = 0
net_value = 1
for i in range((int)(n_samples * 0.6) , n_samples):
	X_train = X[0: i]
	y_train = y[0: i]
	X_test  = X[i: i + 1]
	y_test  = y[i: i + 1]
	
			
	clf = svm.SVC(kernel='rbf', C=1).fit(X_train, y_train)

	#svr = svm.SVR(kernel='rbf', C=1).fit(X_train, y_train)

	score = clf.score(X_test, y_test)

	#score = svr.predict(X_test)	

	if score == 1:
		correct = correct + 1
	else:
		error   = error   + 1
		
	if (score == 1 and y[i] == 1) or (score == 0 and y[i] == -1):
		net_value = net_value * (1 + zz500_r[i])	
	
	#print net_value
	print dates[i], score, y[i], zz500_r[i]


print 1.0 * correct / (correct + error)
print net_value


d   = np.c_[X,zz500_r]
cols = colnames.values.tolist()
cols.append('zz500')
train_df = pd.DataFrame(d, index = dates, columns = cols)
print train_df


#train_df.to_csv('hehe')


#mod = sm.OLS(zz500_r, train)
#res = mod.fit()
#print res.summary()
#print res.params
#print train
#print dir(res)

#for i in range(0, len(train)):
#	print train[i][2]	



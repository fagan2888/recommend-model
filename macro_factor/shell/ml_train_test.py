import pandas as pd
import numpy as np
import time,datetime

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR



# Gaussian Naive Bayes Classifier
def naive_bayes_classifier(train_x, train_y):
    model = GaussianNB()
    model.fit(train_x, train_y)
    return model

# KNN Classifier
def knn_classifier(train_x, train_y):
    model = KNeighborsClassifier()
    model.fit(train_x, train_y)
    return model

# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model

# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    model = RandomForestClassifier(n_estimators=8)
    model.fit(train_x, train_y)
    return model

# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    model = tree.DecisionTreeClassifier()
    model.fit(train_x, train_y)
    return model

# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model

# SVM Classifier
def svm_classifier(train_x, train_y):
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model

# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs = 1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print para, val
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model

#svr regression
def svr_regression(train_x, train_y):
    model = SVR(kernel = 'rbf', probability=True)
    model.fit(train_x, train_y)
    return model

#linear regression
def linear_regression(train_x, train_y):
    model = linear_model.LinearRegression()
    model.fit(train_x, train_y)
    return model





#test_classifiers = ['NB', 'KNN', 'LR', 'RF', 'DT', 'SVM', 'SVMCV', 'GBDT'] 
test_classifiers = ['GBDT']
classifiers = {'NB':naive_bayes_classifier,
              'KNN':knn_classifier,
               'LR':logistic_regression_classifier,
               'RF':random_forest_classifier,
               'DT':decision_tree_classifier,
              'SVM':svm_classifier,
            'SVMCV':svm_cross_validation,
             'GBDT':gradient_boosting_classifier}

data=data.dropna(how='any')
X = np.array(data.iloc[:,0:11].values)
y = np.array(data['redemp'])
train_x,test_x,train_y,test_y = train_test_split(X, y, test_size = 0.4, random_state = 0)
'''
X = np.array(data.iloc[:,[4,8,9,10,11]].values)
y = np.array(data['redemption'])
train_x,test_x,train_y,test_y = train_test_split(X, y, test_size = 0.4, random_state = 0)
'''
num_train, num_feat = train_x.shape
num_test, num_feat = test_x.shape
num_all=len(data)
num_1=len(data[data['redemp']==1])
print '******************** Data Info *********************'
print 'training data: %d, testing data: %d, dimension: %d, data_num:  %d, positive_data_num: %d' % (num_train,num_test,num_feat,num_all,num_1)

for classifier in test_classifiers:
  print '******************* %s ********************' % classifier
  start_time = time.time()
  model = classifiers[classifier](train_x, train_y)
  print 'training took %fs!' % (time.time() - start_time)
  predict = model.predict(test_x)
  precision = metrics.precision_score(test_y, predict)
  recall = metrics.recall_score(test_y, predict)
  print 'precision: %.2f%%, recall: %.2f%%' % (100 * precision, 100 * recall)


'''
gbdt = GradientBoostingClassifier(n_estimators=200)
dt = tree.DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=8)
score_gbdt = cross_val_score(gbdt, X, y, cv = 3,scoring='recall')
score_dt = cross_val_score(dt, X, y, cv = 3,scoring='recall')
score_rf = cross_val_score(rf, X, y, cv = 3,scoring='recall')
'''

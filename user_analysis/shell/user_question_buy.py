#coding=utf8


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier


from sklearn.cluster import KMeans


if __name__ == '__main__':


    df = pd.read_csv('./data/user_q.csv', index_col = ['uid'])
    df = df.fillna('N')
    #print df

    user_option_dict = {}
    all_cols = []
    options = ['A', 'B', 'C', 'D', 'E', 'F', 'N']
    l = len(options)
    for i in range(1, 28):
        col_name = 'q' + str(i)
        cols = []
        for option in options:
            cols.append('q' + str(i) + option)
        all_cols.extend(cols)
        ser = df[col_name]
        for uid in ser.index:
            option_items = list(ser.loc[uid])
            option_vec = np.zeros(l)
            for item in option_items:
                index = options.index(item)
                option_vec[index] = 1
            user_options = user_option_dict.setdefault(uid, [])
            user_options.extend(option_vec)

    uids = user_option_dict.keys()
    uids = list(uids)
    data = []
    for uid in uids:
        data.append(user_option_dict[uid])


    option_df = pd.DataFrame(data, index = uids, columns = all_cols)
    user_df = option_df
    user_df['buy'] = df['buy']
    #user_df['re_buy'] = df['re_buy']
    #user_df['total_buy'] = df['total_buy']
    #user_df['is_rebuy'] = df['is_rebuy']
    user_df['is_redeem'] = df['is_redeem']
    #print user_df.columns

    print user_df.columns
    X = np.array(user_df.values)
    y = np.array(df['is_rebuy'])
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
    #print y_train
    #print y_test

    #clf = tree.DecisionTreeClassifier()
    #clf = svm.SVC(kernel = 'rbf')
    #clf = linear_model.LogisticRegression()
    #clf = GaussianNB()
    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    #clf = RandomForestClassifier(n_estimators=10)
    #clf = AdaBoostClassifier(n_estimators=100)
    clf = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=2, random_state=0)
    #model = SelectFromModel(clf, prefit=True)
    #X_new = model.transform(X)
    #print X_new

    #clf = ExtraTreesClassifier()

    #score = cross_val_score(clf, X, y, cv = 5)
    #print score

    #print X
    #kmeans = KMeans(max_iter = 1000, n_clusters = 4, random_state = 0).fit(X)
    #print kmeans.labels_

    clf.fit(X, y)
    #print clf.feature_importances_
    #model = SelectFromModel(clf, prefit=True)
    #X_new = model.transform(X)
    #print X_new.shape
    #print chi2(X, y)
    #X_new = SelectKBest(mutual_info_classif, k=4).fit_transform(X, y)
    #print X_new
    feature_importrance_df = pd.DataFrame(clf.feature_importances_, index = user_df.columns, columns = ['importrance'])
    print feature_importrance_df
    feature_importrance_df.to_csv('question_importance.csv')

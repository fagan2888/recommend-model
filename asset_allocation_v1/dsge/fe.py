#!/home/yaojiahui/anaconda2/bin/python
# coding=utf-8
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score


def load_iris_data():
    # Load the iris datasets 
    iris = datasets.load_iris()
    # Create a list of feature names
    # Create X from the features
    X = iris.data
    # Create y from output
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    return X_train, X_test, y_train, y_test 

def feature_selection(X_train, y_train):
    feat_labels = ['Sepal Length','Sepal Width','Petal Length','Petal Width']
    # Create a random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Print the name and gini importance of each feature 
    for feature in zip(feat_labels, clf.feature_importances_):
        print feature

    sfm = SelectFromModel(clf, threshold=0.15)
    sfm.fit(X_train, y_train)

    for feature_list_index in sfm.get_support(indices=True):
        print(feat_labels[feature_list_index])

    X_important_train = sfm.transform(X_train)
    #print X_important_train
    return X_important_train


if __name__ == '__main__':
    X_train, _, y_train, _ = load_iris_data()
    X_important_train = feature_selection(X_train, y_train)
    print X_important_train


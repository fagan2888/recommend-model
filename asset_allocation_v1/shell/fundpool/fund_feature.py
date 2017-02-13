#coding=utf8


import pandas as pd
import numpy  as np
from datetime import datetime
import statsmodels.api as sm
import sklearn
from sklearn import preprocessing
from sklearn import linear_model


if __name__ == '__main__':

    manager_fund_performance_df = pd.read_csv('manager_fund_performance.csv', index_col = 'SECODE')
    fund_nav_df = pd.read_csv('./data/fund_value.csv', index_col = 'date')
    fund_nav_df.fillna(method = 'pad', inplace = True)
    #fund_nav_dfr = fund_nav_df.pct_change().fillna(0.0)
    #fund_nav_df.dropna(axis = 1, inplace = True)
    cols = fund_nav_df.columns
    codes = []
    for col in cols:
        codes.append((int)(col))
    mask = manager_fund_performance_df['FSYMBOL'].map( lambda x : x in codes)
    manager_fund_performance_df = manager_fund_performance_df.loc[mask]


    feature_cols = ['GENDER', 'DEGREE', 'JOBTITLE','BIRTH', 'FSYMBOL', 'ALLRANK', 'CLASSRANK', 'TOTYEARS', 'CURFCOUNT']
    df = manager_fund_performance_df[feature_cols]


    X  = []
    y  = []
    for i in range(0, len(df)):
        tmp_df = df.iloc[i]
        vec = []
        code   = '%06d' % tmp_df['FSYMBOL']
        gender = tmp_df['GENDER']
        degree = tmp_df['DEGREE']
        jobtitle = tmp_df['JOBTITLE']
        allrank  = tmp_df['ALLRANK']
        classrank = tmp_df['CLASSRANK']
        totyears = tmp_df['TOTYEARS']
        curfcount = tmp_df['CURFCOUNT']

        if gender == '男':
            vec.append(1)
        else:
            vec.append(0)

        if type(degree) == float:
            vec.append(0)
        elif degree.find('博士') > -1:
            vec.append(2)
        elif degree.find('硕士') > -1:
            vec.append(1)
        else:
            vec.append(0)

        if totyears.find('年') > -1:
            items = totyears.split('年')
            vec.append((int)(items[0]))
        else:
            vec.append(0)

        if type(jobtitle) == float:
            vec.append(0)
        elif jobtitle.find('注册金融分析师') > -1:
            vec.append(1)
        else:
            vec.append(0)

        items = allrank.split('/')
        vec.append((float)(items[1]) / (float)(items[0]))

        items = classrank.split('/')
        vec.append((float)(items[1]) / (float)(items[0]))

        #print fund_nav_df[code]
        #print vec
        X.append(vec)

        vs = fund_nav_df.iloc[-1025:,][code]
        r = vs[-1] / vs[0] - 1
        y.append(r)

    #print X
    #print y
    reg = linear_model.LinearRegression()
    reg.fit(X, y)
    print reg.coef_
    print reg.intercept_
    print reg.score(X, y)

    X = sm.add_constant(X)
    model  = sm.OLS(y,X)
    result = model.fit()
    print result.summary()
    #df['r'] = y
    #df.to_csv('df.csv')
    #print X, y
    #X_scaled = preprocessing.scale(X)
    #print X_scaled

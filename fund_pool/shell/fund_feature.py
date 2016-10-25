#coding=utf8


import pandas as pd
import numpy  as np
from datetime import datetime
import statsmodels.api as sm

if __name__ == '__main__':

    manager_fund_performance_df = pd.read_csv('manager_fund_performance.csv', index_col = 'SECODE')
    fund_nav_df = pd.read_csv('./data/fund_value.csv', index_col = 'date')
    #print fund_nav_df.columns
    #print manager_fund_performance_df.columns

    feature_cols = ['GENDER', 'DEGREE', 'JOBTITLE','BIRTH', 'FSYMBOL', 'ALLRANK', 'CLASSRANK', 'TOTYEARS', 'CURFCOUNT']
    #print manager_fund_performance_df.columns

    #print feature_cols
    df = manager_fund_performance_df[feature_cols]
    #print df['JOBTITLE']

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

        print fund_nav_df[code]
        #print vec

        #print vec
        #print code, gender, degree, jobtitle, allrank, classrank, totyears, curfcount


    #print fund_nav_df

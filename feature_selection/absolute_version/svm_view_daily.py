#coding: utf-8
from __future__ import division
import pandas as pd
import numpy as np
import warnings
import json

#from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
#from sklearn.feature_selection import RFE
#from sklearn.feature_selection import f_regression
#from sklearn.feature_selection import f_classif
#from sklearn.feature_selection import mutual_info_classif
#from sklearn.feature_selection import chi2

#from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
#from functools import partial

from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

best_params = {}

def neg_f1_score(x, y):
    return f1_score(x, y, pos_label = -1)

def weighted_score(x, y, weight = 1000):

    R = recall_score(x, y, pos_label = -1)
    P = accuracy_score(x, y)
    if (R == 0) or (P == 0):
        return 0

    E = (1 + weight)/(weight/P + 1/R)
    return E

class LR(LogisticRegression):
#logisticRegression binding L1 & L2 used as estimator in SelectFromModel

    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
        fit_intercept=True, intercept_scaling=1, class_weight=None,
        random_state=None, solver='liblinear', max_iter=100,
        multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

        self.threshold = threshold

        LogisticRegression.__init__(self, penalty='l2', dual=dual, tol=tol, C=C,
            fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, \
                    class_weight=class_weight,
            random_state=random_state, solver=solver, max_iter=max_iter,
            multi_class=multi_class, verbose=verbose, warm_start=warm_start, \
                    n_jobs=n_jobs)

        self.l2 = LogisticRegression(penalty='l1', dual=dual, tol=tol, C=C, \
                fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, \
                class_weight = class_weight, random_state=random_state, \
                solver=solver, max_iter=max_iter, multi_class=multi_class, \
                verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

    def fit(self, X, y, sample_weight=None):
         super(LR, self).fit(X, y, sample_weight=sample_weight)
         self.coef_old_ = self.coef_.copy()
         self.l2.fit(X, y, sample_weight=sample_weight)

         cntOfRow, cntOfCol = self.coef_.shape

         for i in range(cntOfRow):
             for j in range(cntOfCol):
                 coef = self.coef_[i][j]
                 if coef != 0:
                     idx = [j]
                     coef1 = self.l2.coef_[i][j]
                     for k in range(cntOfCol):
                         coef2 = self.l2.coef_[i][k]
                         if abs(coef1-coef2) < self.threshold and j != k and \
                                 self.coef_[i][k] == 0:
                             idx.append(k)

                     mean = coef / len(idx)
                     self.coef_[i][idx] = mean

         return self


class Svm(object):
    def __init__(self, asset, test_num):
        #数据文件路径
        self.path = './assets/' + asset + '.csv'

        #资产名字
        self.asset = asset

        #原始数据
        self.data = pd.read_csv(self.path, index_col = 0, parse_dates = True)

        #用于测试的样本数目
        self.test_num = test_num

        #排除在训练样本之外的数目
        self.predict_num = 100

        # 用上周、上上周数据作为特征
        self.lag = 1

        # cross validate的数目
        self.cv = 6

        # cpu数目
        self.threads_num = 35

        #最佳参数字典
        self.best_params = {}
        #self.best_params = \
        #        json.load(file('./output_params/best_params1.json', 'r'))

        # self.window = len(self.data) - test_num - 10
        # 测试时的训练窗口长度
        self.window = 1500
        if self.window + test_num > len(self.data):
            self.window = 500


    def handle(self):
        print 'asset: ', self.asset
        self.feature = self.preprocessing()
        self.threshold = self.get_threshold(self.test_num)
        print 'threshold: ', self.threshold

        if self.best_params:
            print 'existing params'
            self.params, self.feature_selected = self.best_params[self.asset]
        else:
            print 'non-existing params'
            self.params, self.feature_selected = \
                    self.training(self.predict_num, \
                    threads_num = self.threads_num, feature = self.feature,\
                    cv = self.cv, lag = self.lag)

        self.result_df, self.win_ratio = self.predicting(self.test_num, \
                self.window, self.params, self.feature, self.feature_selected)
        print 'win ratio: ', self.win_ratio

        self.validate_scores = self.cross_validate(self.predict_num, self.params, \
                self.feature, self.feature_selected)
        print 'validate_scores:', np.mean(self.validate_scores)
        up_ratio = (1 + svm.result_df['pre_states'].sum()/test_num)/2
        print 'up ratio: ', up_ratio

        self.return_list = self.simulating()
        if len(self.return_list) == 0:
            self.invest_value = 1
        else:
            self.invest_value = self.return_list[-1]

        print 'invest value: ', self.invest_value
        print


    # 数据预处理
    def preprocessing(self):
        feature = self.data
        feature['dummy'] = np.sign(feature['pct_chg'])
        pct_chg = feature['pct_chg']
        feature.rename(columns = {'pct_chg': 'pct_chg_f'}, inplace = True)

#        feature = feature.apply(MinMaxScaler().fit_transform)
#        feature = feature.apply(scale)
        feature_lag1 = feature.shift(1)
        feature_lag2 = feature.shift(2)
        feature_lag3 = feature.shift(3)

        feature_lag1.rename(columns = lambda x: (x + '_lag1'), inplace = True)
        feature_lag2.rename(columns = lambda x: (x + '_lag2'), inplace = True)
        feature_lag3.rename(columns = lambda x: (x + '_lag3'), inplace = True)
        feature = pd.concat([feature, feature_lag1, feature_lag2, \
                feature_lag3], axis = 1)

        feature['pct_chg'] = pct_chg
        feature['label'] = feature['dummy'].shift(-1)
        feature.dropna(inplace = True)

        return feature


    #last model
    def get_threshold(self, test_num):
        correct_num = (np.sign(self.feature['pct_chg']) == \
                self.feature['label'])[-test_num:].sum()
        win_ratio = correct_num/test_num

        return win_ratio


    @staticmethod
    def training(test_num, threads_num, feature, cv, lag = 1):
        '''
        :usage: get proper params using gridSearch
        :param test_num: 用于predict，从训练集中去除的样本数
        :param threads_num: 训练用的线程数

        :feature: 用于训练的数据
            types:DataFrame
            format:index name = date
                   column name = 'feature_1' feature_2' ... 'feature_n'
                   'feature_1_lag1' 'feature_2_lag1' ... 'feature_n_lag1'
            用Svm.preprocessing可以得到这种形式的输出

        :cv: GridSearchCV的参数
        :lag: lag=0表示只用上周的数据作为特征，
              lag=1表示中上周和上上周的数据作为特征，以此类推

        '''

        grid_scorer = make_scorer(weighted_score, greater_is_better = True)

        if lag == 0:
            x = feature.loc[:, 'close':'dummy']
        elif lag == 1:
            x = feature.loc[:, 'close':'dummy_lag1']
        elif lag == 2:
            x = feature.loc[:, 'close':'dummy_lag2']
        elif lag == 3:
            x = feature.loc[:, 'close':'dummy_lag3']

        '''
        #selectkbest using f_classif, f_regression, mutual_info_classif, chi2
        select_method = mutual_info_classif

        x = x[:-test_num]
        y = y[:-test_num]

        scaler = StandardScaler()
        selectkbest = SelectKBest(select_method, k = 10)
        pca = PCA()
        svc = SVC()
        pipe = Pipeline(steps = [('scaler', scaler), \
                ('selectkbest', selectkbest), ('pca', pca), ('svc', svc)])

        params = dict(
                pca__n_components = range(1, 6),
                svc__C = 10.0**np.arange(-5, 4),
                svc__gamma = 10.0**np.arange(-3, 5)
                )

        estimator = GridSearchCV(pipe, param_grid = params, cv = cv, \
                verbose = 0, n_jobs = threads_num, scoring = grid_scorer)
        #estimator = GridSearchCV(pipe, param_grid = params, cv = cv, \
                verbose = 0, n_jobs = threads_num, scoring = 'f1')
        estimator.fit(x, y)
        score = estimator.best_score_
        params = estimator.best_params_
        feature_selected = x.columns[estimator.best_estimator_.named_steps\
                ['selectkbest'].get_support()]
        '''

        '''
        #rfe using GBDT

        y = feature['label']

        x = x[:-test_num]
        y = y[:-test_num]

        rfe = RFE(estimator = GradientBoostingClassifier())
        svc = SVC()
        pipe = Pipeline(steps = [('rfe', rfe), ('svc', svc)])

        params = dict(
                svc__C = 10.0**np.arange(-5, 4),
                svc__gamma = 10.0**np.arange(-3, 5),
                rfe__n_features_to_select = range(1, 4)
                )

        estimator = GridSearchCV(pipe, param_grid = params, cv = cv, \
                verbose = 0, n_jobs = threads_num, scoring = grid_scorer)
        estimator.fit(x, y)
        score = estimator.best_score_
        params = estimator.best_params_
        feature_selected = x.columns[estimator.best_estimator_.named_steps\
                ['rfe'].get_support()]
        '''

        #SelectFromModel using logisticRegression(binding L1 & L2)
        y = feature['label']

        x = x[:-test_num]
        y = y[:-test_num]

        scaler = StandardScaler()
        selectfmodel = SelectFromModel(estimator = LR(threshold=0.5, C=0.1))
        #selectfmodel = SelectFromModel(estimator = LogisticRegression())
        svc = SVC()
        #pca = PCA()
        lda = LDA()
        pipe = Pipeline(steps = [('scaler', scaler), \
                ('selectfmodel', selectfmodel), ('lda', lda), ('svc', svc)])

        params = dict(
                lda__n_components = [5],
                svc__C = 10.0**np.arange(-5, 4),
                svc__gamma = 10.0**np.arange(-3, 5)
                )

        estimator = GridSearchCV(pipe, param_grid = params, cv = cv, \
                verbose = 0, n_jobs = threads_num, scoring = grid_scorer)
        #estimator = GridSearchCV(pipe, param_grid = params, cv = cv, \
        #        verbose = 0, n_jobs = threads_num, scoring = 'accuracy')
        estimator.fit(x, y)
        #score = estimator.best_score_
        params = estimator.best_params_
        feature_selected = x.columns[estimator.best_estimator_.named_steps\
                ['selectfmodel'].get_support()]

        return params, feature_selected


    # predict next week's status
    # c: svm__C, g: svm__gamma, n: lda_n_components
    @staticmethod
    def predicting(test_num, window, params, feature, feature_selected):
        '''
        :usage: 用Svm模型进行预测
        :window: 训练窗口的长度
        :params: 类型为字典，形式为
        {'svc__C': Svm模型的C参数,
         'svm__gamma': Svm模型的gamma参数,
         'lda__n_components': 用lda降维后的维度
        }
        :feature: 输入数据，格式通training的feature参数
        :feature_selected: 特征字符串组成的列表
        '''
        window = window
        c = params['svc__C']
        g = params['svc__gamma']
        n = params['lda__n_components']
        #n = self.params['pca__n_components']
        x = feature.loc[:, feature_selected]
        y = feature['label']

        scaler = StandardScaler()
        lda = LDA(n_components = n)
        #pca = PCA(n_components = n)
        svc = SVC(C = c, gamma = g, probability = False)

        pre_states = []
        distances = []
        for i in np.arange(test_num):
            x_train, x_test = x[-test_num - window + i: -test_num + i], \
                    x[-test_num + i: ]
            y_train = y[-test_num - window + i: -test_num + i]

            pipe = Pipeline(steps = [('scaler', scaler), ('lda', lda), \
                    ('svc', svc)])
            pipe.fit(x_train, y_train)
            pre_state = pipe.predict(x_test)[0]
            pre_states.append(pre_state)
            distance = np.abs(pipe.decision_function(x_test)[0])
            distances.append(distance)

        result_df = feature[-test_num: ]
        result_df.loc[:, 'pre_states'] = pre_states
        result_df.loc[:, 'distance'] = distances

        correct_num = (result_df['pre_states'] == \
                result_df['label']).values.sum()
        win_ratio = correct_num/test_num

        return result_df, win_ratio


    #做交叉验证
    @staticmethod
    def cross_validate(test_num, params, feature, feature_selected):
        c = params['svc__C']
        g = params['svc__gamma']
        n = params['lda__n_components']
        #n = self.params['pca__n_components']
        x = feature.loc[:, feature_selected][:-test_num]
        y = feature['label'][:-test_num]

        scaler = StandardScaler()
        lda = LDA(n_components = n)
        #pca = PCA(n_components = n)
        svc = SVC(C = c, gamma = g)
        pipe = Pipeline(steps = [('scaler', scaler), ('lda', lda), ('svc', svc)])
        validate_scores = cross_val_score(pipe, x, y, cv=10)

        return validate_scores



    #invest using the predicted status
    def simulating(self):
        test_num = self.test_num
        dates = self.result_df.index
        return_list = []
        threshold_value = 1.0
        invest_value = 1.0

        for i in range(test_num - 1):
            if self.result_df.loc[dates[i], 'pct_chg'] > 0:
                threshold_value *= (1 + self.result_df.loc[dates[i+1], \
                        'pct_chg']/100)
#            else:
#                threshold_value *= (1 - self.result_df.loc[dates[i+1], \
#        'pct_chg']/100)

            if self.result_df.loc[dates[i], 'pre_states'] > 0:
                invest_value *= (1 + self.result_df.loc[dates[i+1], \
                        'pct_chg']/100)
                return_list.append(invest_value)
#            else:
#                invest_value *= (1 - self.result_df.loc[dates[i+1], \
#        'pct_chg']/100)

        return return_list


    # 信号数目
    def get_signal_num(self):
        sig_trans_list = []
        for i in range(len(self.result_df['pre_states'])-1):
            sig_trans_list.append(self.result_df['pre_states'][i]*\
                    self.result_df['pre_states'][i+1])
        signal_num = sig_trans_list.count(-1.0)

        return signal_num

    # 最大回撤
    def get_max_drawdown(self):
        return_list = self.return_list
        max_drawdown = 0
        for i in range(1, len(return_list)):
            if return_list[i] - max(return_list[:i]) < max_drawdown:
                max_drawdown = return_list[i] - max(return_list[:i])

        return -max_drawdown

    # 投资周期
    def get_trans_pos_cycle(self):
        return self.test_num/self.get_signal_num()

    # 年化收益
    def get_annual_ret(self):
        return np.exp(np.log(self.invest_value)/(self.test_num//250)) - 1

    # 收益/最大回撤
    def get_ret_to_md(self):
        return self.get_annual_ret()/self.get_max_drawdown()



if __name__ == '__main__':

    test_num = 500
    #assets = ['sh300', 'zz500', 'hsi', 'nhsp', 'sp500', 'au']
    assets = ['120000001', '120000002']
    best_params_dict = {}

    for asset in assets:
        svm = Svm(asset, test_num)
        svm.handle()
        print 'signal_num: ', svm.get_signal_num()
        print 'max_drawdown: ', svm.get_max_drawdown()
        print 'trans_pos_cycle: ', svm.get_trans_pos_cycle()
        print 'annual_ret: ', svm.get_annual_ret()
        print 'ret_to_md: ', svm.get_ret_to_md()
        print

        '''
        print 'asset: ', asset
        print 'window: ',svm.window
        print 'cv: ', svm.cv
        print 'threshold: ', svm.threshold
        print 'win_ratio: ', svm.win_ratio
        print 'cv_win_ratio: ', svm.score
        print 'params: ', svm.params
        print 'features: ', svm.feature_selected
        print 'threshold_value: ', svm.threshold_value
        print 'total_ret: ', svm.invest_value - 1
        print
        '''

        best_params_dict[asset] = [svm.params, list(svm.feature_selected)]
        #result_df = svm.result_df.loc[:, ['pct_chg', 'pre_states', 'distances']]
        #result_df.to_csv('./output_data/' + asset + '.csv', index_label = 'date')

    json.dump(best_params_dict, file('output_params/best_params3.json', 'w'))

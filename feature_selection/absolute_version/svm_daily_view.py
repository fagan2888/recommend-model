#coding = utf8
from __future__ import division
import pandas as pd
import numpy as np
import warnings

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from functools import partial

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')

'''
best_params = {'hsi': [{'svc__gamma': 15.848931924611271, 'svc__C': 12.589254117941044, 'lda__n_components': 5}, [u'macd', u'atr', u'cci', u'rsi', u'mtm', u'slowkd', u'pct_chg_f',u'priceosc', u'dummy', u'high_lag1', u'low_lag1', u'open_lag1',u'atr_lag1', u'rsi_lag1', u'roc_lag1', u'pct_chg_f_lag1', u'bias_lag1',u'dummy_lag1']],
               'sh300': [{'svc__gamma': 25.118864315096026, 'svc__C': 0.19952623149688095, 'lda__n_components': 5}, [u'low', u'cci', u'mtm', u'slowkd', u'pct_chg_f', u'priceosc', u'bias',u'dpo', u'cci_lag1', u'mtm_lag1', u'roc_lag1', u'slowkd_lag1',u'pct_chg_f_lag1', u'priceosc_lag1', u'bias_lag1', u'dpo_lag1']],
               'nhsp': [{'svc__gamma': 50.118723362727764, 'svc__C': 0.050118723362725714, 'lda__n_components': 5}, [u'high', u'volume', u'cci', u'rsi', u'sobv', u'mtm', u'roc', u'bias',u'vstd', u'dummy', u'high_lag1', u'volume_lag1', u'atr_lag1',u'sobv_lag1', u'mtm_lag1', u'slowkd_lag1', u'bias_lag1', u'vstd_lag1']],
               'au': [{'svc__gamma': 5011.8723362727969, 'svc__C': 25.118864315094488, 'lda__n_components': 5}, [u'macd', u'cci', u'rsi', u'sobv', u'mtm', u'roc', u'pct_chg_f', u'pvt',u'wvad', u'vstd', u'macd_lag1', u'rsi_lag1', u'sobv_lag1', u'mtm_lag1',u'roc_lag1', u'pct_chg_f_lag1','pvt_lag1', u'wvad_lag1',u'dummy_lag1']],
               'zz500': [{'svc__gamma': 2511.886431509613, 'svc__C': 0.39810717055348227, 'lda__n_components': 5}, [u'low', u'volume', u'roc', u'wvad', u'priceosc', u'vstd', u'dummy',u'close_lag1', u'volume_lag1', u'macd_lag1', u'atr_lag1', u'rsi_lag1',u'mtm_lag1', u'roc_lag1',u'slowkd_lag1', u'pct_chg_f_lag1',u'wvad_lag1', u'priceosc_lag1', u'vstd_lag1', u'dpo_lag1']],
               'sp500': [{'svc__gamma': 39810.717055350346, 'svc__C': 0.50118723362725304, 'lda__n_components': 5}, [u'cci', u'rsi', u'mtm', u'roc', u'pvt', u'wvad', u'bias', u'vstd',u'dummy', u'atr_lag1', u'rsi_lag1', u'roc_lag1', u'pct_chg_f_lag1',u'wvad_lag1', u'vstd_lag1']]}
               '''

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

#logisticRegression binding L1 & L2 used as estimator in SelectFromModel
class LR(LogisticRegression):

    def __init__(self, threshold=0.01, dual=False, tol=1e-4, C=1.0,
        fit_intercept=True, intercept_scaling=1, class_weight=None,
        random_state=None, solver='liblinear', max_iter=100,
        multi_class='ovr', verbose=0, warm_start=False, n_jobs=1):

        self.threshold = threshold

        LogisticRegression.__init__(self, penalty='l2', dual=dual, tol=tol, C=C,
            fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
            random_state=random_state, solver=solver, max_iter=max_iter,
            multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

        self.l2 = LogisticRegression(penalty='l1', dual=dual, tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight = class_weight, random_state=random_state, solver=solver, max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs)

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
                         if abs(coef1-coef2) < self.threshold and j != k and self.coef_[i][k] == 0:
                             idx.append(k)

                     mean = coef / len(idx)
                     self.coef_[i][idx] = mean

         return self


class Svm(object):
    def __init__(self, asset, test_num):
        self.path = '../assets/' + asset + '.csv'
        self.data = pd.read_csv(self.path, index_col = 0, parse_dates = True)
        self.test_num = test_num
        self.lag = 1
        self.cv = 6
        self.threads_num = 35
#        self.window = len(self.data) - test_num - 10

        self.window = 1500
        if self.window + test_num > len(self.data):
            self.window = 500

        self.feature = self.preprocessing()
        self.threshold = self.get_threshold(test_num)

        if best_params:
            print 'existing params'
            self.params, self.feature_selected = best_params[asset]
        else:
            print 'non-existing params'
            self.score, self.params, self.feature_selected = self.pipelining(test_num, threads_num = self.threads_num, lag = self.lag)

        self.result_df, self.win_ratio = self.training(test_num)
        self.validate_scores = self.cross_validate(test_num)
        print 'validate_scores:', np.mean(self.validate_scores)
        self.return_list = self.simulating(test_num)
        self.invest_value = self.return_list[-1]

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
        feature = pd.concat([feature, feature_lag1, feature_lag2, feature_lag3], axis = 1)

        feature['pct_chg'] = pct_chg
        feature['label'] = feature['dummy'].shift(-1)
        feature.dropna(inplace = True)

        return feature


    def get_threshold(self, test_num):
        correct_num = (np.sign(self.feature['pct_chg']) == self.feature['label'])[-test_num:].sum()
        win_ratio = correct_num/test_num

        return win_ratio


    #get proper params using gridSearch
    def pipelining(self, test_num, threads_num, lag = 1):
        grid_scorer = make_scorer(weighted_score, greater_is_better = True)

        if lag == 0:
            x = self.feature.loc[:, 'close':'dummy']
        elif lag == 1:
            x = self.feature.loc[:, 'close':'dummy_lag1']
        elif lag == 2:
            x = self.feature.loc[:, 'close':'dummy_lag2']
        elif lag == 3:
            x = self.feature.loc[:, 'close':'dummy_lag3']

        '''
        #selectkbest using f_classif, f_regression, mutual_info_classif, chi2
        select_method = mutual_info_classif
        y = self.feature['label']
        cv = self.cv

        scaler = StandardScaler()
        selectkbest = SelectKBest(select_method, k = 10)
        pca = PCA()
        svc = SVC()
        pipe = Pipeline(steps = [('scaler', scaler), ('selectkbest', selectkbest), ('pca', pca), ('svc', svc)])

        params = dict(
                pca__n_components = range(1, 6),
                svc__C = 10.0**np.arange(-5, 4),
                svc__gamma = 10.0**np.arange(-3, 5)
                )

        estimator = GridSearchCV(pipe, param_grid = params, cv = cv, verbose = 0, n_jobs = threads_num, scoring = grid_scorer)
        #estimator = GridSearchCV(pipe, param_grid = params, cv = cv, verbose = 0, n_jobs = threads_num, scoring = 'f1')
        estimator.fit(x, y)
        score = estimator.best_score_
        params = estimator.best_params_
        feature_selected = x.columns[estimator.best_estimator_.named_steps['selectkbest'].get_support()]
        '''

        '''
        #rfe using GBDT
        y = self.feature['label']
        cv = self.cv

        rfe = RFE(estimator = GradientBoostingClassifier())
        svc = SVC()
        pipe = Pipeline(steps = [('rfe', rfe), ('svc', svc)])

        params = dict(
                svc__C = 10.0**np.arange(-5, 4),
                svc__gamma = 10.0**np.arange(-3, 5),
                rfe__n_features_to_select = range(1, 4)
                )

        estimator = GridSearchCV(pipe, param_grid = params, cv = cv, verbose = 0, n_jobs = threads_num, scoring = grid_scorer)
        estimator.fit(x, y)
        score = estimator.best_score_
        params = estimator.best_params_
        feature_selected = x.columns[estimator.best_estimator_.named_steps['rfe'].get_support()]
        '''

        #SelectFromModel using logisticRegression(binding L1 & L2)
        y = self.feature['label']
        cv = self.cv

        scaler = StandardScaler()
        selectfmodel = SelectFromModel(estimator = LR(threshold=0.5, C=0.1))
        #selectfmodel = SelectFromModel(estimator = LogisticRegression())
        svc = SVC()
        #pca = PCA()
        lda = LDA()
        pipe = Pipeline(steps = [('scaler', scaler), ('selectfmodel', selectfmodel), ('lda', lda), ('svc', svc)])

        params = dict(
                lda__n_components = [5],
                svc__C = 10.0**np.arange(-5, 4),
                svc__gamma = 10.0**np.arange(-3, 5)
                )

        #estimator = GridSearchCV(pipe, param_grid = params, cv = cv, verbose = 0, n_jobs = threads_num, scoring = grid_scorer)
        estimator = GridSearchCV(pipe, param_grid = params, cv = cv, verbose = 0, n_jobs = threads_num, scoring = 'accuracy')
        estimator.fit(x, y)
        score = estimator.best_score_
        params = estimator.best_params_
        feature_selected = x.columns[estimator.best_estimator_.named_steps['selectfmodel'].get_support()]

        return score, params, feature_selected


    # predict next week's status
    # c: svm__C, g: svm__gamma, n: lda_n_components
    def training(self, test_num):
        window = self.window
        c = self.params['svc__C']
        g = self.params['svc__gamma']
        n = self.params['lda__n_components']
        #n = self.params['pca__n_components']
        x = self.feature.loc[:, self.feature_selected]
        y = self.feature['label']

        scaler = StandardScaler()
        lda = LDA(n_components = n)
        #pca = PCA(n_components = n)
        svc = SVC(C = c, gamma = g, probability = False)

        pre_states = []
        distances = []
        for i in np.arange(test_num):
            x_train, x_test = x[-test_num - window + i: -test_num + i], x[-test_num + i: ]
            y_train = y[-test_num - window + i: -test_num + i]

            pipe = Pipeline(steps = [('scaler', scaler), ('lda', lda), ('svc', svc)])
            pipe.fit(x_train, y_train)
            pre_state = pipe.predict(x_test)[0]
            pre_states.append(pre_state)
            distance = np.abs(pipe.decision_function(x_test)[0])
            distances.append(distance)

        result_df = self.feature[-test_num: ]
        result_df.loc[:, 'pre_states'] = pre_states
        result_df.loc[:, 'distance'] = distances

        correct_num = (result_df['pre_states'] == result_df['label']).values.sum()
        win_ratio = correct_num/test_num

        return result_df, win_ratio


    def cross_validate(self, test_num):
        c = self.params['svc__C']
        g = self.params['svc__gamma']
        n = self.params['lda__n_components']
        #n = self.params['pca__n_components']
        x = self.feature.loc[:, self.feature_selected]
        y = self.feature['label']

        scaler = StandardScaler()
        lda = LDA(n_components = n)
        #pca = PCA(n_components = n)
        svc = SVC(C = c, gamma = g)
        pipe = Pipeline(steps = [('scaler', scaler), ('lda', lda), ('svc', svc)])
        validate_scores = cross_val_score(pipe, x, y, cv=10)

        return validate_scores



    #invest using the predicted status
    def simulating(self, test_num):
        dates = self.result_df.index
        return_list = []
        threshold_value = 1.0
        invest_value = 1.0

        for i in range(test_num - 1):
            if self.result_df.loc[dates[i], 'pct_chg'] > 0:
                threshold_value *= (1 + self.result_df.loc[dates[i+1], 'pct_chg']/100)
#            else:
#                threshold_value *= (1 - self.result_df.loc[dates[i+1], 'pct_chg']/100)

            if self.result_df.loc[dates[i], 'pre_states'] > 0:
                invest_value *= (1 + self.result_df.loc[dates[i+1], 'pct_chg']/100)
                return_list.append(invest_value)
#            else:
#                invest_value *= (1 - self.result_df.loc[dates[i+1], 'pct_chg']/100)

        return return_list


    def get_signal_num(self):
        sig_trans_list = []
        for i in range(len(self.result_df['pre_states'])-1):
            sig_trans_list.append(self.result_df['pre_states'][i]*self.result_df['pre_states'][i+1])
        signal_num = sig_trans_list.count(-1.0)

        return signal_num

    def get_max_drawdown(self):
        return_list = self.return_list
        max_drawdown = 0
        for i in range(1, len(return_list)):
            if return_list[i] - max(return_list[:i]) < max_drawdown:
                max_drawdown = return_list[i] - max(return_list[:i])

        return -max_drawdown

    def get_trans_pos_cycle(self):
        return self.test_num/self.get_signal_num()

    def get_annual_ret(self):
        return np.exp(np.log(self.invest_value)/5) - 1

    def get_ret_to_md(self):
        return self.get_annual_ret()/self.get_max_drawdown()



if __name__ == '__main__':

    test_num = 250
    assets = ['120000001', '120000002']
    best_params_dict = {}

    for asset in assets:
        svm = Svm(asset, test_num)
        up_ratio = (1 + svm.result_df['pre_states'].sum()/test_num)/2
        print 'asset: ', asset
        #print 'window: ',svm.window
        #print 'cv: ', svm.cv
        print 'threshold: ', svm.threshold
        print 'win_ratio: ', svm.win_ratio
        #print 'cv_win_ratio: ', svm.score
        print 'params: ', svm.params
        print 'features: ', svm.feature_selected
        print 'up_ratio: ', up_ratio
        #print 'threshold_value: ', svm.threshold_value
        print 'total_ret: ', svm.invest_value - 1
        print 'signal_num: ', svm.get_signal_num()
        print 'max_drawdown: ', svm.get_max_drawdown()
        print 'trans_pos_cycle: ', svm.get_trans_pos_cycle()
        print 'annual_ret: ', svm.get_annual_ret()
        print 'ret_to_md: ', svm.get_ret_to_md()
        print
        best_params_dict[asset] = [svm.params, list(svm.feature_selected)]
        result_df = svm.result_df.loc[:, ['pct_chg', 'pre_states', 'distances']]
        result_df.to_csv('./svm_absolute_view/' + asset + '.csv', index_label = 'date')

    print best_params_dict
